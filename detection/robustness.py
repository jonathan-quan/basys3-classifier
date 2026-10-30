"""Controlled image-transform sensitivity, explicitly not a physical webcam test."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image,ImageEnhance

from classifier.common import choose_device,file_hash
from .model import load_detector,decode
from .data import letterbox,tensorize
from .fetch_data import read_jsonl,save_json
from .metrics import aligned_iou,compute


def transform_case(image,box,case):
    if case.startswith('brightness_'):
        return ImageEnhance.Brightness(image).enhance(float(case.split('_')[1])),box
    if case=='horizontal_flip':
        from PIL import ImageOps
        return ImageOps.mirror(image),None if box is None else np.array([1-box[2],box[1],1-box[0],box[3]])
    if case=='rotate_15':
        # PIL positive rotation is counter-clockwise in image display coordinates.
        if box is not None:
            x1,y1,x2,y2=box;pts=np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])-.5
            a=np.deg2rad(-15);mat=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            pts=pts@mat.T+.5;box=np.clip(np.r_[pts.min(0),pts.max(0)],0,1)
        return image.rotate(15,resample=Image.Resampling.BILINEAR,fillcolor=(128,128,128)),box
    return image,box


def evaluate(model,rows,device,case):
    records=[]
    for start in range(0,len(rows),64):
        subset=rows[start:start+64];images=[];truth=[]
        for row in subset:
            with Image.open(row['path']) as im:image,box,_=letterbox(im,row.get('bbox'))
            image,box=transform_case(image,box,case);images.append(tensorize(image));truth.append([0,0,0,0] if box is None else box)
        with torch.inference_mode():out=decode(model(torch.stack(images).to(device)))
        out={k:v.cpu() for k,v in out.items()};ious=aligned_iou(out['boxes'],torch.tensor(np.array(truth),dtype=torch.float32))
        for i,r in enumerate(subset):
            records.append({'label':r['label'],'present':r['present'],'predicted_class':int(out['classes'][i]),
                'score':float(out['scores'][i]),'presence_score':float(out['presence'][i]),'iou':float(ious[i]) if r['present'] else None})
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--split',default='test',choices=('val','test'));p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError('Robustness output exists; choose a new name')
    device=choose_device();model,c=load_detector(a.checkpoint,device)
    if file_hash(a.manifest)!=c['manifest_sha256']:raise ValueError('Manifest mismatch')
    rows=[r for r in read_jsonl(a.manifest) if r['split']==a.split]
    report={'scope':'Controlled transformations of the same held-out photos; not new independent examples, physical lighting changes, object 3D orientation, or unseen backgrounds.',
            'split':a.split,'real_webcam_measured':False,'cases':{}}
    for case in ('original','brightness_0.6','brightness_1.4','horizontal_flip','rotate_15'):
        records=evaluate(model,rows,device,case)
        metrics=compute(records,c['joint_score_threshold'],c['presence_threshold'])
        if not c.get('objectness_trained'):
            for k in ('presence_precision','presence_recall','presence_false_positive_rate'):metrics[k]=None
        report['cases'][case]=metrics
        print(case,metrics['classification_accuracy_all_positives'],metrics['mean_iou_all_positives'],flush=True)
    save_json(a.output,report)


if __name__=='__main__':main()
