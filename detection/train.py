"""Transfer/fine-tune the fixed 7,681-parameter detector, validation selection only."""
import argparse
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,WeightedRandomSampler

from classifier.common import choose_device,seed_everything,file_hash,environment
from classifier.data import ManifestDataset,read_manifest
from classifier.evaluate import load_checkpoint as load_classifier
from . import ROOT,PROJECT,CLASSES,SIZE
from .model import TinyDetector,corners,profile,with_head_batch_norm,with_backbone_batch_norm,export_inference
from .data import DetectionDataset,validate_rows
from .fetch_data import read_jsonl,save_json,save_jsonl
from .metrics import aligned_iou,compute,select_thresholds
from .evaluate import predict_records


def objective(output,batch,device,objectness=True):
    positive=batch['present'].to(device)>0
    classes=batch['label'].to(device)
    truth=batch['box'].to(device)
    zero=output['class_logits'].sum()*0
    cls=nn.functional.cross_entropy(output['class_logits'][positive],classes[positive]) if positive.any() else zero
    presence=nn.functional.binary_cross_entropy_with_logits(output['presence_logits'],positive.float()) if objectness else zero
    if positive.any():
        b=truth[positive];cxcywh=torch.cat(((b[:,:2]+b[:,2:])/2,b[:,2:]-b[:,:2]),1)
        box=nn.functional.smooth_l1_loss(output['box_cxcywh'][positive],cxcywh,beta=.1)
        iou=(1-aligned_iou(corners(output['box_cxcywh'][positive]),b)).mean()
    else:box=iou=zero
    return cls+presence+5*box+2*iou,{'class':float(cls.detach()),'presence':float(presence.detach()),'box':float(box.detach()),'iou_loss':float(iou.detach())}


def draw_history(rows,path):
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');epochs=[r['epoch'] for r in rows]
    axes[0].plot(epochs,[r['loss'] for r in rows]);axes[0].set_title('Training loss')
    axes[1].plot(epochs,[r['val_map50'] for r in rows],label='AP50 mean present classes')
    axes[1].plot(epochs,[r['val_class_accuracy'] for r in rows],label='Class accuracy');axes[1].legend(fontsize=8)
    axes[2].plot(epochs,[r['val_mean_iou'] for r in rows]);axes[2].set_title('Mean IoU, all positive frames')
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.savefig(path,dpi=150);plt.close(fig)


def train(args):
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Run exists; use a new output folder')
    if args.epochs<1 or args.warmup<0 or args.patience<1:raise ValueError('Invalid training limits')
    rows=read_jsonl(args.manifest)
    counts=validate_rows(rows,require_negatives=not args.pilot,require_all_test_classes=not args.pilot)
    synthetic=any(r.get('synthetic') for r in rows)
    if synthetic and not args.smoke_test:raise ValueError('Synthetic fixtures require --smoke-test')
    if args.smoke_test and 'smoke' not in str(output).lower():raise ValueError('Use a smoke output folder')
    output.mkdir(parents=True);seed_everything(args.seed);device=choose_device(args.device)
    model=TinyDetector();transfer=model.transfer_classifier(args.classifier)
    if args.head_batch_norm:model=with_head_batch_norm(model)
    if args.backbone_batch_norm:model=with_backbone_batch_norm(model)
    model.to(device)
    datasets={s:DetectionDataset(rows,s,training=s=='train') for s in ('train','val')}
    frequencies=Counter((r['label'],r['source']) for r in datasets['train'].rows)
    sourcecounts=Counter()
    for label,source in frequencies:sourcecounts[label]+=1
    weights=[1/(sourcecounts[r['label']]*frequencies[(r['label'],r['source'])]) for r in datasets['train'].rows]
    gen=torch.Generator().manual_seed(args.seed)
    loader=DataLoader(datasets['train'],batch_size=args.batch_size,num_workers=0,
        sampler=WeightedRandomSampler(weights,len(weights),replacement=True,generator=gen))
    objectness=counts['train']['negative']>0
    if not objectness:
        for p in model.presence.parameters():p.requires_grad_(False)
    auxiliary=None;teacher=None
    if not args.no_classifier_rehearsal and not synthetic:
        old=ManifestDataset(read_manifest(PROJECT/'data/manifest.csv'),'train',training=True,cache=True)
        auxfreq=Counter(r['label'] for r in old.rows)
        auxweights=[1/auxfreq[r['label']] for r in old.rows]
        auxiliary=DataLoader(old,batch_size=args.batch_size,num_workers=0,
            sampler=WeightedRandomSampler(auxweights,len(auxweights),replacement=True,generator=gen))
        teacher,_=load_classifier(args.classifier,device)
    groups=[{'params':list(model.features.parameters())+list(model.fc.parameters()),'lr':args.backbone_lr},
            {'params':list(model.box_reduce.parameters())+list(model.box_fc.parameters())+list(model.presence.parameters()),'lr':args.head_lr}]
    opt=torch.optim.AdamW(groups,weight_decay=1e-4)
    scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,args.epochs-args.warmup),eta_min=1e-6)
               if args.schedule=='cosine' else torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=5,min_lr=1e-7))
    config={**vars(args),'counts':counts,'transfer':transfer,'model':profile(),'environment':environment(device),
            'training_parameters':sum(p.numel() for p in model.parameters()),
            'objectness_trained':objectness,'synthetic':synthetic,
            'manifest_sha256':file_hash(args.manifest),
            'selection':'Validation mean AP50 across present classes, tie-break all-positive mean IoU. No test evaluation during training.',
            'classification_rehearsal':'Original classifier TRAIN only: 0.5 CE + 0.1 KL(temperature=2), on matched augmentations' if auxiliary else None}
    save_json(output/'config.json',config);save_jsonl(output/'manifest_snapshot.jsonl',rows)
    print(json.dumps({'device':str(device),'transfer':transfer,'counts':counts,'objectness_trained':objectness,'pilot':args.pilot},indent=2),flush=True)
    best=(-1.,-1.);stale=0;history=[];start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        warm=epoch<=args.warmup
        for p in itertools.chain(model.features.parameters(),model.fc.parameters()):p.requires_grad_(not warm)
        model.train();losses=[];aux_iterator=iter(auxiliary) if auxiliary and not warm else None
        for batch in loader:
            x=batch['image'].to(device);opt.zero_grad(set_to_none=True)
            out=model(x);loss,parts=objective(out,batch,device,objectness)
            if aux_iterator is not None:
                try:ax,ay,_=next(aux_iterator)
                except StopIteration:aux_iterator=iter(auxiliary);ax,ay,_=next(aux_iterator)
                ax=ax.to(device);ay=ay.to(device)
                student=model.classify(ax)
                with torch.no_grad():oldlogits=teacher(ax)
                retention=.5*nn.functional.cross_entropy(student,ay)+.1*4*nn.functional.kl_div(
                    nn.functional.log_softmax(student/2,dim=1),nn.functional.softmax(oldlogits/2,dim=1),reduction='batchmean')
                loss=loss+retention
            if not torch.isfinite(loss):raise RuntimeError('Non-finite detector loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
        records,_=predict_records(model,datasets['val'],device,args.batch_size)
        metrics=compute(records);score=(metrics['map50_present_classes'] or 0.,metrics['mean_iou_all_positives'] or 0.)
        if not warm:
            if args.schedule=='cosine':scheduler.step()
            else:scheduler.step(score[0])
        improved=score>best
        history.append({'epoch':epoch,'warmup':warm,'loss':float(np.mean(losses)),'val_map50':score[0],
                        'val_mean_iou':score[1],'val_class_accuracy':metrics['classification_accuracy_all_positives'],
                        'backbone_lr':opt.param_groups[0]['lr'],'head_lr':opt.param_groups[1]['lr']})
        print(json.dumps(history[-1]),flush=True)
        if improved:
            best=score;stale=0
            inference_model=export_inference(model)
            state={k:v.detach().cpu().clone() for k,v in inference_model.state_dict().items()}
            checkpoint={'format_version':1,'architecture':'reuse_7681','classes':list(CLASSES),'input_size':SIZE,
                'model_state_dict':state,'epoch':epoch,'manifest_sha256':config['manifest_sha256'],'transfer':transfer,
                'incomplete_pilot':args.pilot,'objectness_trained':objectness,'smoke_test':synthetic,
                'joint_score_threshold':.3,'presence_threshold':.5,'validation_map50_present_classes':score[0],
                'validation_mean_iou':score[1],'preprocessing':'EXIF-correct RGB, aspect-preserving bilinear letterbox 96, pad128, (pixel-128)/128',
                'quantized':False,'fpga_tested':False,'training_head_batch_norm_folded':args.head_batch_norm,
                'training_backbone_batch_norm_folded':args.backbone_batch_norm}
            torch.save(checkpoint,output/'best_detector.pth')
        elif not warm:stale+=1
        if stale>=args.patience and epoch>args.warmup+args.patience:break
    if device.type=='cuda':torch.cuda.synchronize()
    training_seconds=time.perf_counter()-start
    checkpoint=torch.load(output/'best_detector.pth',map_location='cpu',weights_only=True)
    model=TinyDetector().to(device);model.load_state_dict(checkpoint['model_state_dict']);model.eval()
    records,_=predict_records(model,datasets['val'],device,args.batch_size)
    threshold,pt=select_thresholds(records)
    checkpoint.update({'joint_score_threshold':threshold,'presence_threshold':pt,'training_seconds':training_seconds})
    torch.save(checkpoint,output/'best_detector.pth')
    save_json(output/'history.json',history);draw_history(history,output/'learning_curves.png')
    save_json(output/'validation_selection.json',{'best_epoch':checkpoint['epoch'],'score_threshold_selected_on_validation':threshold,
       'presence_threshold_selected_on_validation':pt,'test_used_for_selection':False,'manifest_sha256':config['manifest_sha256']})
    save_json(output/'summary.json',{'checkpoint':str(output/'best_detector.pth'),'parameters':7681,
       'training_seconds':training_seconds,'epochs':len(history),'best_epoch':checkpoint['epoch'],
       'model_checkpoint_bytes':(output/'best_detector.pth').stat().st_size,'objectness_trained':objectness,
       'incomplete_pilot':args.pilot,'synthetic':synthetic,'int8_accuracy_measured':False,'fpga_feasibility_proven':False})
    print('Training finished',training_seconds,'seconds; best epoch',checkpoint['epoch'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--classifier',default=str(PROJECT/'best_model.pth'));p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--warmup',type=int,default=8);p.add_argument('--patience',type=int,default=15);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--backbone-lr',type=float,default=5e-5);p.add_argument('--head-lr',type=float,default=.001)
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--pilot',action='store_true',help='Explicitly limited experiment without complete ground truth; not validated four-class detection')
    p.add_argument('--smoke-test',action='store_true');p.add_argument('--no-classifier-rehearsal',action='store_true')
    p.add_argument('--head-batch-norm',action='store_true',help='Training-only normalization on new head, folded out of the 7681-parameter checkpoint')
    p.add_argument('--backbone-batch-norm',action='store_true',help='Identity-initialized training BN after transferred convs; fold completely on export')
    p.add_argument('--schedule',choices=('plateau','cosine'),default='plateau')
    train(p.parse_args())


if __name__=='__main__':main()
