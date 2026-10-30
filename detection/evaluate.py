"""Evaluate detector outputs and preserve a side-by-side classifier comparison."""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image,ImageOps,ImageDraw
from torch.utils.data import DataLoader

from classifier.common import choose_device,file_hash
from classifier.evaluate import load_checkpoint as load_classifier
from classifier.data import image_transform
from . import ROOT,PROJECT,CLASSES,NAMES
from .model import load_detector,decode
from .data import DetectionDataset,letterbox,tensorize,inverse_box
from .fetch_data import read_jsonl,save_json,save_jsonl
from .metrics import aligned_iou,compute


def predict_records(model,dataset,device,batch_size=64):
    records=[];model.eval();elapsed=0.;batches=0
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0):
            images=batch['image'].to(device)
            if device.type=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();out=decode(model(images))
            if device.type=='cuda':torch.cuda.synchronize()
            elapsed+=time.perf_counter()-start;batches+=1
            out={k:v.cpu() for k,v in out.items()};ious=aligned_iou(out['boxes'],batch['box'])
            for k,index in enumerate(batch['index'].tolist()):
                row=dataset.rows[index]
                records.append({'id':row['id'],'path':row['path'],'label':row['label'],'present':row['present'],
                    'group':row['group'],'source':row['source'],'annotation_status':row['annotation_status'],
                    'conditions':row.get('conditions',{}),'predicted_class':int(out['classes'][k]),
                    'presence_score':float(out['presence'][k]),'class_score':float(out['class_scores'][k]),
                    'score':float(out['scores'][k]),'predicted_box_letterbox':out['boxes'][k].tolist(),
                    'truth_box_letterbox':batch['box'][k].tolist() if row['present'] else None,
                    'iou':float(ious[k]) if row['present'] else None})
    return records,{'batched_model_seconds':elapsed,'batches':batches,'batch_size':batch_size,
                    'note':'LAPTOP batched model timing, not webcam latency or FPGA inference.'}


def compare_original(model,device,split='test'):
    teacher,_=load_classifier(PROJECT/'best_model.pth',device);transform=image_transform(False)
    with (PROJECT/'data/manifest.csv').open(newline='',encoding='utf-8') as f:rows=[r for r in csv.DictReader(f) if r['split']==split]
    before=[];after=[];truth=[]
    with torch.inference_mode():
        for start in range(0,len(rows),64):
            subset=rows[start:start+64];tensors=[]
            for r in subset:
                with Image.open(r['path']) as im:tensors.append(transform(im))
                truth.append(CLASSES.index(r['label']))
            x=torch.stack(tensors).to(device)
            before+=teacher(x).argmax(1).cpu().tolist();after+=model(x)['class_logits'].argmax(1).cpu().tolist()
    a=np.array(before)==np.array(truth);b=np.array(after)==np.array(truth)
    return {'split':split,'images':len(rows),'original_classifier_accuracy':float(a.mean()),
            'detector_class_head_accuracy':float(b.mean()),'change_percentage_points':float((b.mean()-a.mean())*100),
            'improved_images':int((~a&b).sum()),'regressed_images':int((a&~b).sum()),
            'per_class':{n:{'support':truth.count(i),'classifier_accuracy':float(a[np.array(truth)==i].mean()),
                         'detector_accuracy':float(b[np.array(truth)==i].mean())} for i,n in enumerate(CLASSES)},
            'scope':'Same original frozen classifier images and preprocessing, argmax class only. No box/presence ground truth inferred; this is a legacy retention benchmark.'}


def compare_on_detection(records,device):
    model,_=load_classifier(PROJECT/'best_model.pth',device);positive=[r for r in records if r['present']]
    if not positive:return None
    correct=0
    with torch.inference_mode():
        for start in range(0,len(positive),64):
            subset=positive[start:start+64];images=[]
            for row in subset:
                with Image.open(row['path']) as im:images.append(tensorize(letterbox(im)[0]))
            guesses=model(torch.stack(images).to(device)).argmax(1).cpu().tolist()
            correct+=sum(g==r['label'] for g,r in zip(guesses,subset))
    detector=sum(r['predicted_class']==r['label'] for r in positive)
    return {'positive_images':len(positive),'original_classifier_accuracy':correct/len(positive),
            'detector_classification_accuracy':detector/len(positive),'change_percentage_points':100*(detector-correct)/len(positive),
            'scope':'Same detector-positive frames and exact detector letterbox transform for both class heads.'}


def save_examples(records,path,threshold,max_images=24):
    # Fixed hash-independent order; include errors rather than cherry-pick successes.
    # Predefined stratified ID order, never selected using correctness/confidence.
    per_class=max(1,max_images//5)
    rows=[]
    for label in (0,1,2,3,-1):
        rows+=sorted((r for r in records if r['label']==label),key=lambda r:r['id'])[:per_class]
    rows=rows[:max_images]
    canvas=Image.new('RGB',(4*300,((len(rows)+3)//4)*270),'white');draw=ImageDraw.Draw(canvas)
    index=[]
    for i,r in enumerate(rows):
        with Image.open(r['path']) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');_,_,t=letterbox(im)
            box=inverse_box(r['predicted_box_letterbox'],t);w,h=im.size
            gt=None if not r['present'] else inverse_box(r['truth_box_letterbox'],t)
            im.thumbnail((290,210));ww,hh=im.size
        x=i%4*300;y=i//4*270;canvas.paste(im,(x,y))
        if r['score']>=threshold:
            b=box*np.array([ww/w,hh/h,ww/w,hh/h]);draw.rectangle((x+b[0],y+b[1],x+b[2],y+b[3]),outline='red',width=2)
        if gt is not None:
            b=gt*np.array([ww/w,hh/h,ww/w,hh/h]);draw.rectangle((x+b[0],y+b[1],x+b[2],y+b[3]),outline='lime',width=2)
        actual='background' if not r['present'] else CLASSES[r['label']]
        draw.text((x+2,y+214),f"GT {actual} | IoU {r['iou']:.2f}" if r['present'] else 'GT reviewed background',fill='black')
        draw.text((x+2,y+231),f"{CLASSES[r['predicted_class']]} {r['score']:.1%} {'accepted' if r['score']>=threshold else 'rejected'}",fill='black')
        draw.text((x+2,y+248),'RED prediction; GREEN ground truth',fill='black');index.append(r['id'])
    if rows:canvas.save(path,quality=94)
    return index


def plot_metrics(metrics,path):
    matrix=np.array(metrics['classification_confusion_matrix'])
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained');ax.imshow(matrix,cmap='Blues')
    ax.set(xticks=range(4),yticks=range(4),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='Actual',title='Positive-frame classification')
    for i in range(4):
        for j in range(4):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right');fig.savefig(path,dpi=150);plt.close(fig)


def condition_report(records,threshold,presence_threshold):
    groups=defaultdict(list)
    for r in records:
        for key in ('lighting','background','orientation'):
            value=r.get('conditions',{}).get(key,'unknown')
            if value!='unknown':groups[key+':'+value].append(r)
    return {'verified_condition_subsets':{k:compute(v,threshold,presence_threshold) for k,v in groups.items()},
            'unmeasured':['Real personal-webcam performance','Unseen physical instances unless documented in provenance'],
            'note':'No lighting/orientation/background labels are invented from source filenames or augmentations. Empty subsets mean unmeasured.'}


def evaluate(checkpoint,manifest,split,output,device,original_comparison=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    prefix=split
    if (out/f'{prefix}_metrics.json').exists():raise FileExistsError('Evaluation already exists; choose a new output')
    model,c=load_detector(checkpoint,device)
    if c['manifest_sha256']!=file_hash(manifest):raise ValueError('Checkpoint/manifest mismatch; use a separately audited external benchmark workflow')
    dataset=DetectionDataset(read_jsonl(manifest),split);records,timing=predict_records(model,dataset,device)
    threshold=c['joint_score_threshold'];pt=c['presence_threshold']
    metrics=compute(records,threshold,pt)
    metrics.update({'checkpoint_sha256':file_hash(checkpoint),'manifest_sha256':file_hash(manifest),'split':split,
                   'incomplete_pilot':c.get('incomplete_pilot',False),'objectness_trained':c.get('objectness_trained',False),
                   'hardware_tested':False,'laptop_timing':timing})
    if not c.get('objectness_trained'):
        for key in ('presence_precision','presence_recall','presence_false_positive_rate'):metrics[key]=None
        metrics['presence_warning']='No verified negative training examples; presence head not trained. Its sigmoid score must not be interpreted as validated objectness.'
    save_json(out/f'{prefix}_metrics.json',metrics);save_jsonl(out/f'{prefix}_predictions.jsonl',records)
    save_json(out/f'{prefix}_classifier_comparison.json',compare_on_detection(records,device))
    save_json(out/f'{prefix}_conditions.json',condition_report(records,threshold,pt))
    ids=save_examples(records,out/f'{prefix}_examples.jpg',threshold)
    save_json(out/f'{prefix}_examples_index.json',ids);plot_metrics(metrics,out/f'{prefix}_confusion_matrix.png')
    if original_comparison:save_json(out/'original_classifier_retention.json',compare_original(model,device))
    print(json.dumps(metrics,indent=2),flush=True)
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--split',choices=('val','test'),default='test');p.add_argument('--output',required=True);p.add_argument('--device',default='auto')
    p.add_argument('--compare-original',action='store_true');a=p.parse_args()
    evaluate(a.checkpoint,a.manifest,a.split,a.output,choose_device(a.device),a.compare_original)


if __name__=='__main__':main()
