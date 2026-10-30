"""Box-aware preprocessing; candidate annotations are never training targets."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
from torch.utils.data import Dataset
from classifier.common import file_hash

from . import SIZE
from .fetch_data import read_jsonl

TRUSTED={"source_ground_truth","human_verified"}


def letterbox(image,box=None):
    image=ImageOps.exif_transpose(image).convert("RGB");w,h=image.size
    scale=min(SIZE/w,SIZE/h)
    nw=max(1,round(w*scale));nh=max(1,round(h*scale))
    left=(SIZE-nw)//2;top=(SIZE-nh)//2
    out=Image.new("RGB",(SIZE,SIZE),(128,128,128))
    out.paste(image.resize((nw,nh),Image.Resampling.BILINEAR),(left,top))
    transform={"width":w,"height":h,"resized_width":nw,"resized_height":nh,"left":left,"top":top}
    b=None if box is None else np.array([(box[0]*nw+left)/SIZE,(box[1]*nh+top)/SIZE,
                                      (box[2]*nw+left)/SIZE,(box[3]*nh+top)/SIZE],dtype=np.float32)
    return out,b,transform


def inverse_box(box,t):
    b=np.asarray(box,dtype=float).copy()
    b[[0,2]]=(b[[0,2]]*SIZE-t['left'])/t['resized_width']*t['width']
    b[[1,3]]=(b[[1,3]]*SIZE-t['top'])/t['resized_height']*t['height']
    b[[0,2]]=np.clip(b[[0,2]],0,t['width']);b[[1,3]]=np.clip(b[[1,3]],0,t['height'])
    return b


def tensorize(image):
    a=np.asarray(image,dtype=np.float32).copy().transpose(2,0,1)
    return torch.from_numpy(a).sub_(128).div_(128)


def augment(image,box):
    # Affine includes translation, scale, and moderate rotation; transform all corners.
    angle=random.uniform(-20,20);scale=random.uniform(.82,1.10)
    tx=random.uniform(-.08,.08)*SIZE;ty=random.uniform(-.08,.08)*SIZE
    theta=np.deg2rad(angle);c=np.cos(theta)*scale;s=np.sin(theta)*scale
    forward=np.array([[c,-s,SIZE/2+tx-c*SIZE/2+s*SIZE/2],
                      [s,c,SIZE/2+ty-s*SIZE/2-c*SIZE/2],[0,0,1]])
    inv=np.linalg.inv(forward)
    new_box=None
    if box is not None:
        x1,y1,x2,y2=np.asarray(box)*SIZE
        pts=np.array([[x1,y1,1],[x2,y1,1],[x1,y2,1],[x2,y2,1]])@forward.T
        low=pts[:,:2].min(0);high=pts[:,:2].max(0)
        clipped=np.clip(np.r_[low,high],0,SIZE)/SIZE
        original_area=max(1e-6,float(np.prod(high-low)))
        visible=float(np.prod((clipped[2:]-clipped[:2])*SIZE))/original_area
        # Reject transforms which almost remove the target; never turn it into a false negative.
        if visible<.65 or np.any((clipped[2:]-clipped[:2])*SIZE<1):
            forward=np.eye(3);inv=forward;new_box=np.array(box,copy=True)
        else:new_box=clipped.astype(np.float32)
    image=image.transform((SIZE,SIZE),Image.Transform.AFFINE,tuple(inv[:2].reshape(-1)),
                          resample=Image.Resampling.BILINEAR,fillcolor=(128,128,128))
    if random.random()<.5:
        image=ImageOps.mirror(image)
        if new_box is not None:new_box=np.array([1-new_box[2],new_box[1],1-new_box[0],new_box[3]],dtype=np.float32)
    image=ImageEnhance.Brightness(image).enhance(random.uniform(.65,1.35))
    image=ImageEnhance.Contrast(image).enhance(random.uniform(.75,1.25))
    image=ImageEnhance.Color(image).enhance(random.uniform(.8,1.2))
    if random.random()<.12:image=image.filter(ImageFilter.GaussianBlur(random.uniform(.2,.8)))
    return image,new_box


class DetectionDataset(Dataset):
    def __init__(self,rows,split,training=False):
        self.rows=[r for r in rows if r.get('split')==split]
        self.training=training;self.cache=[]
        for r in self.rows:
            if r.get('annotation_status') not in TRUSTED:
                raise ValueError(f"Unverified candidate cannot be used: {r['id']}")
            if file_hash(r['path'])!=r['sha256']:raise ValueError('Image changed after dataset preparation')
            if r['present'] and r.get('bbox') is None:raise ValueError("Positive needs a box")
            with Image.open(r['path']) as im:
                image,box,_=letterbox(im,r.get('bbox'))
                self.cache.append((image,box))

    def __len__(self):return len(self.rows)

    def __getitem__(self,i):
        row=self.rows[i];image,box=self.cache[i];image=image.copy()
        if self.training:image,box=augment(image,box)
        return {"image":tensorize(image),"label":row['label'],"present":float(row['present']),
                "box":torch.tensor(box if box is not None else [0,0,0,0],dtype=torch.float32),"index":i}


def validate_rows(rows,require_negatives=True,require_all_test_classes=True):
    if not rows:raise ValueError("Empty detection manifest")
    ids=set();groups={};pixels={};counts={s:{'positive':[0]*4,'negative':0} for s in ('train','val','test')}
    for r in rows:
        if r['id'] in ids:raise ValueError("Repeated record ID")
        ids.add(r['id']);split=r['split']
        if split not in counts:raise ValueError("Invalid split")
        if r.get('present') not in (0,1):raise ValueError('Invalid object-presence target')
        if not all(r.get(k) for k in ('source','license','group','pixel_sha256','sha256')):raise ValueError('Missing provenance/group/hash')
        if r.get('annotation_status') not in TRUSTED:raise ValueError("Unreviewed candidates in manifest")
        for table,k in ((groups,r['group']),(pixels,r['pixel_sha256'])):
            if k in table and table[k]!=split:raise ValueError("Group/pixel leakage across splits")
            table[k]=split
        if r['present']:
            if not 0<=r['label']<4:raise ValueError("Invalid class")
            b=r['bbox']
            if not (len(b)==4 and 0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1):raise ValueError("Invalid positive box")
            counts[split]['positive'][r['label']]+=1
        else:
            if r['label']!=-1 or r.get('bbox') is not None:raise ValueError("Negative must not have a target")
            counts[split]['negative']+=1
    for s,c in counts.items():
        if not any(c['positive']):raise ValueError(f"No positives in {s}")
        if (s=='train' or require_all_test_classes) and not all(c['positive']):raise ValueError(f"Missing class in {s}: {c['positive']}")
        if require_negatives and not c['negative']:raise ValueError(f"No verified real negative frames in {s}; review candidates first")
    return counts
