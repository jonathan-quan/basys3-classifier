"""Prepare untrusted box/background candidates for explicit human review."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from classifier.model import TinyClassifier
from . import ROOT,PROJECT,SIZE
from .data import letterbox,inverse_box,tensorize
from .fetch_data import download,fingerprints,existing_candidates,save_jsonl,save_json,read_jsonl,ALLOWED_LICENSES,MID


def proposals(rows):
    checkpoint=torch.load(PROJECT/'best_model.pth',map_location='cpu',weights_only=True)
    model=TinyClassifier('context');model.load_state_dict(checkpoint['model_state_dict']);model.eval()
    with torch.inference_mode():
        for row in rows:
            with Image.open(row['path']) as im:image,_,t=letterbox(im)
            features=model.features(tensorize(image).unsqueeze(0))
            cam=(features*model.fc.weight[row['label']][None,:,None,None]).sum(1,keepdim=True)
            cam=torch.nn.functional.interpolate(cam,(SIZE,SIZE),mode='bilinear',align_corners=False)[0,0].numpy()
            cam=np.maximum(cam,0)
            if cam.max()>0:
                yy,xx=np.where(cam>=.55*cam.max())
                if len(xx):
                    b=np.array([max(0,xx.min()-5),max(0,yy.min()-5),min(SIZE,xx.max()+6),min(SIZE,yy.max()+6)])/SIZE
                    original=inverse_box(b,t)/np.array([t['width'],t['height'],t['width'],t['height']])
                    if (original[2]-original[0])*(original[3]-original[1])<.95 and np.all(original[2:]>original[:2]):row['bbox']=original.tolist()
            row['proposal_method']='Transferred classifier class activation map; NOT ground truth'
    return rows


def negatives(count):
    cache=ROOT/'data/sources/open_images'
    ids={};candidates=[]
    for split in ('validation','test'):
        supported=set();allids=set()
        with (cache/f'{split}-boxes.csv').open(newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                allids.add(r['ImageID'])
                if r['LabelName'] in MID:supported.add(r['ImageID'])
        with (cache/f'{split}-metadata.csv').open(newline='',encoding='utf-8') as f:
            for m in csv.DictReader(f):
                if m['ImageID'] not in allids or m['ImageID'] in supported:continue
                license_url=m.get('License','').replace('http://','https://').rstrip('/')
                if license_url not in ALLOWED_LICENSES or m.get('Rotation') not in ('0','0.0'):continue
                author=m.get('AuthorProfileURL') or m.get('Author')
                if not author or author.lower() in ('nan','n/a','unknown'):continue
                candidates.append((split,m,author))
    random.Random(739).shuffle(candidates);seen=set();chosen=[]
    for split,m,author in candidates:
        key=hashlib.sha256(author.strip().casefold().encode()).hexdigest()[:16]
        if key in seen:continue
        seen.add(key);chosen.append((split,m,author,key))
        if len(chosen)>=count:break
    def one(x):
        split,m,author,key=x;uid=m['ImageID'];path=ROOT/'data/images/background_candidates'/f'{uid}.jpg'
        url=f'https://open-images-dataset.s3.amazonaws.com/{split}/{uid}.jpg';download(url,path)
        return {'id':'background_'+uid,'path':str(path),'label':-1,'present':0,'bbox':None,
                'annotation_status':'candidate','annotation_source':'No target box in source; absence is UNVERIFIED; human review required',
                'source':'https://storage.googleapis.com/openimages/web/download_v7.html','license':m['License'],
                'author':m.get('Author',''),'author_profile':author,'landing_url':m.get('OriginalLandingURL',''),
                'download_url':url,'group':'flickr-author-'+key,'fixed_split':None,'quality_flags':{},**fingerprints(path)}
    with ThreadPoolExecutor(8) as ex:rows=list(ex.map(one,chosen))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--negative-count',type=int,default=90);p.add_argument('--cube-per-split',type=int,default=20)
    p.add_argument('--personal-only',action='store_true');p.add_argument('--output',default=str(ROOT/'data/review_queue.jsonl'))
    a=p.parse_args();existing=existing_candidates();chosen=[]
    if a.personal_only:
        for path in (ROOT/'data/personal').rglob('captures.jsonl'):
            chosen += [r for r in read_jsonl(path) if r['annotation_status']=='candidate']
    else:
        for split in ('val','test'):
            subset=[r for r in existing if r['label']==0 and r['fixed_split']==split]
            random.Random(19).shuffle(subset);chosen+=subset[:a.cube_per_split]
        chosen=proposals(chosen)+negatives(a.negative_count)
    # Interleave rather than forcing the reviewer through one whole category first.
    random.Random(37).shuffle(chosen)
    from pathlib import Path
    out=Path(a.output)
    if out.exists():raise FileExistsError('Review queue already exists; do not overwrite work')
    save_jsonl(out,chosen)
    save_json(out.with_suffix('.summary.json'),{'items':len(chosen),'box_candidates':sum(r['present'] for r in chosen),
       'background_candidates':sum(not r['present'] for r in chosen),'all_unverified':True,'source_counts':dict(Counter(r['source'] for r in chosen)),
       'note':'Nothing in this queue is usable ground truth until explicitly reviewed. Original cube val/test are a secondary legacy benchmark, not fresh webcam data.'})
    print('Manual review candidates',len(chosen),'in',out,flush=True)


if __name__=='__main__':main()
