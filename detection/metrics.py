"""Single-object, class-aware metrics; missing evidence is reported as null."""
import numpy as np
import torch

from . import CLASSES


def aligned_iou(pred,truth):
    lo=torch.maximum(pred[:,:2],truth[:,:2]);hi=torch.minimum(pred[:,2:],truth[:,2:])
    inter=(hi-lo).clamp(min=0).prod(1)
    a=(pred[:,2:]-pred[:,:2]).clamp(min=0).prod(1)
    b=(truth[:,2:]-truth[:,:2]).clamp(min=0).prod(1)
    return inter/(a+b-inter).clamp(min=1e-9)


def divide(n,d):return float(n/d) if d else None


def ap101(scores,tp,support):
    if support==0:return None
    order=np.argsort(-np.asarray(scores),kind='stable');tp=np.asarray(tp,dtype=float)[order]
    if len(tp)==0:return 0.
    cum=np.cumsum(tp);rec=cum/support;prec=cum/np.arange(1,len(tp)+1)
    return float(np.mean([np.max(prec[rec>=r]) if np.any(rec>=r) else 0. for r in np.linspace(0,1,101)]))


def compute(records,threshold=.3,presence_threshold=.5,iou_threshold=.5):
    positives=[r for r in records if r['present']];negatives=[r for r in records if not r['present']]
    per={};mat=np.zeros((4,4),dtype=int);ious=[]
    for r in positives:mat[r['label'],r['predicted_class']]+=1;ious.append(r['iou'])
    tp=fp=fn=0
    for i,name in enumerate(CLASSES):
        gt=[r for r in positives if r['label']==i]
        guesses=[r for r in records if r['predicted_class']==i]
        accepted=[r for r in guesses if r['score']>=threshold]
        true=[r for r in accepted if r['present'] and r['label']==i and r['iou']>=iou_threshold]
        ct=len(true);cf=len(accepted)-ct;cn=len(gt)-ct;tp+=ct;fp+=cf;fn+=cn
        aps={}
        for at in (.5,.55,.6,.65,.7,.75,.8,.85,.9,.95):
            aps[f'{at:.2f}']=ap101([r['score'] for r in guesses],[r['present'] and r['label']==i and r['iou']>=at for r in guesses],len(gt))
        per[name]={'support':len(gt),'classification_accuracy':divide(sum(r['predicted_class']==i for r in gt),len(gt)),
            'classification_precision':divide(sum(r['present'] and r['label']==i for r in guesses),sum(r['present'] for r in guesses)),
            'mean_iou_all_positives':float(np.mean([r['iou'] for r in gt])) if gt else None,
            'detection_precision':divide(ct,ct+cf),'detection_recall':divide(ct,len(gt)),
            'true_positives':ct,'false_positives':cf,'false_negatives':cn,'ap50':aps['0.50'],
            'ap50_95':float(np.mean(list(aps.values()))) if gt else None}
    presence_tp=sum(r['presence_score']>=presence_threshold for r in positives)
    presence_fp=sum(r['presence_score']>=presence_threshold for r in negatives)
    accepted_negative=sum(r['score']>=threshold for r in negatives)
    aps=[r['ap50'] for r in per.values() if r['ap50'] is not None]
    apall=[r['ap50_95'] for r in per.values() if r['ap50_95'] is not None]
    matched=[r['iou'] for r in positives if r['score']>=threshold and r['predicted_class']==r['label'] and r['iou']>=iou_threshold]
    return {'images':len(records),'positive_images':len(positives),'negative_images':len(negatives),
        'classification_accuracy_all_positives':divide(sum(r['label']==r['predicted_class'] for r in positives),len(positives)),
        'classification_accuracy_definition':'Argmax over four classes on all positive frames; ignores abstention. Detection recall counts abstentions as misses.',
        'presence_precision':divide(presence_tp,presence_tp+presence_fp) if negatives else None,
        'presence_recall':divide(presence_tp,len(positives)),
        'presence_false_positive_rate':divide(presence_fp,len(negatives)),
        'accepted_detection_false_positive_rate':divide(accepted_negative,len(negatives)),
        'detection_precision':divide(tp,tp+fp),'detection_recall':divide(tp,tp+fn),
        'detection_tp':tp,'detection_fp':fp,'detection_fn':fn,
        'mean_iou_all_positives':float(np.mean(ious)) if ious else None,
        'median_iou_all_positives':float(np.median(ious)) if ious else None,
        'mean_iou_matched_only':float(np.mean(matched)) if matched else None,
        'map50_present_classes':float(np.mean(aps)) if aps else None,
        'map50_95_present_classes':float(np.mean(apall)) if apall else None,
        'missing_positive_classes':[name for name,row in per.items() if not row['support']],
        'iou_threshold':iou_threshold,'joint_score_threshold':threshold,'presence_threshold':presence_threshold,
        'per_class':per,'classification_confusion_matrix':mat.tolist(),
        'metric_scope':'Custom single-object AP with 101 recall points and at most one prediction per image; missing classes excluded, not silently scored as zero.'}


def select_thresholds(records):
    # Threshold choices use validation only; AP checkpoint selection is threshold-free.
    best=(float('-inf'),.3)
    for threshold in np.linspace(.01,.95,95):
        m=compute(records,float(threshold))
        tp,fp,fn=m['detection_tp'],m['detection_fp'],m['detection_fn']
        f1=2*tp/max(1,2*tp+fp+fn)
        if f1>best[0]:best=(f1,float(threshold))
    positives=[r for r in records if r['present']];negatives=[r for r in records if not r['present']]
    pb=(float('-inf'),.5)
    if positives and negatives:
        for t in np.linspace(.01,.99,99):
            tp=sum(r['presence_score']>=t for r in positives);fp=sum(r['presence_score']>=t for r in negatives)
            f1=2*tp/max(1,len(positives)+tp+fp)
            if f1>pb[0]:pb=(f1,float(t))
    return best[1],pb[1]
