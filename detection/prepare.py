"""Freeze source/author/session-disjoint detection splits; exclude candidates."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageOps

from . import ROOT,PROJECT,CLASSES
from .data import validate_rows,TRUSTED
from .fetch_data import read_jsonl,save_jsonl,save_json,fingerprints


def original_constraints():
    with (PROJECT/'data/manifest.csv').open(newline='',encoding='utf-8') as f:original=list(csv.DictReader(f))
    split_by_path={str(Path(r['path']).resolve()):r['split'] for r in original}
    authors={}
    with (PROJECT/'data/attributions.csv').open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            split=split_by_path.get(str(Path(row['path']).resolve()))
            if split and 'openimages' in row['source']:
                profile=row.get('author_profile','')
                author=profile if profile not in ('','n/a') else row.get('author')
                if author and author.lower() not in ('n/a','unknown'):
                    key='flickr-author-'+hashlib.sha256(author.strip().casefold().encode()).hexdigest()[:16]
                    if key in authors and authors[key]!=split:raise ValueError('Original author split conflict')
                    authors[key]=split
    return original,authors


def freeze(rows,instance_disjoint=False):
    original,authors=original_constraints()
    rows=sorted(rows,key=lambda r:r['id'])
    parents=list(range(len(rows)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):parents[find(j)]=find(i)
    groups={};pixels={};instances={};byclass=defaultdict(list);quarantine=[]
    for i,r in enumerate(rows):
        for table,key in ((groups,r['group']),(pixels,r['pixel_sha256'])):
            if key in table:union(i,table[key])
            else:table[key]=i
        byclass[r['label']].append(i)
        if instance_disjoint and r['present'] and r.get('instance'):
            key=(r['label'],r['instance'])
            if key in instances:union(i,instances[key])
            else:instances[key]=i
        if r['group'] in authors:r.setdefault('constraints',[]).append(authors[r['group']])
    for inds in byclass.values():
        for pos,i in enumerate(inds):
            for j in inds[pos+1:]:
                if (int(rows[i]['dhash'],16)^int(rows[j]['dhash'],16)).bit_count()<=3:union(i,j)
    oldbyclass=defaultdict(list)
    for r in original:oldbyclass[CLASSES.index(r['label'])].append(r)
    oldpixels={r['pixel_sha256']:r['split'] for r in original}
    # New detector holdouts must also be disjoint from classifier pretraining.
    for r in rows:
        if r['pixel_sha256'] in oldpixels:r.setdefault('constraints',[]).append(oldpixels[r['pixel_sha256']])
        old_candidates=original if not r['present'] else oldbyclass.get(r['label'],[])
        for old in old_candidates:
            if r['pixel_sha256']==old['pixel_sha256'] or (int(r['dhash'],16)^int(old['dhash'],16)).bit_count()<=3:
                r.setdefault('constraints',[]).append(old['split'])
    components=defaultdict(list)
    for i,r in enumerate(rows):components[find(i)].append(r)
    forced={};free=[];usable=[]
    for comp in components.values():
        roles=set()
        for r in comp:
            if r.get('fixed_split'):roles.add(r['fixed_split'])
            roles.update(r.get('constraints',[]))
        if len(roles)>1:
            quarantine.extend({'id':r['id'],'reason':'Connected author/session/duplicate component conflicts with inherited split','roles':sorted(roles)} for r in comp)
            continue
        key=hashlib.sha256('|'.join(sorted(r['id'] for r in comp)).encode()).hexdigest()[:20]
        for r in comp:r['group']=key
        if roles:forced[key]=(next(iter(roles)),comp)
        else:free.append((key,comp))
        usable+=comp
    totals=np.zeros(5)
    def vector(comp):
        v=np.zeros(5)
        for r in comp:v[r['label'] if r['present'] else 4]+=1
        return v
    counts=np.zeros((3,5));names=('train','val','test')
    for split,comp in forced.values():counts[names.index(split)]+=vector(comp)
    for r in usable:totals[r['label'] if r['present'] else 4]+=1
    vectors=np.stack([vector(c) for _,c in free]) if free else np.zeros((0,5))
    rng=np.random.default_rng(42);best=None;bestscore=float('inf');target=np.array([.7,.15,.15])[:,None]
    for _ in range(5000):
        assignment=rng.choice(3,len(free),p=[.7,.15,.15]);dist=counts.copy()
        for k in range(3):dist[k]+=vectors[assignment==k].sum(0)
        missing=(dist==0)&(totals[None,:]>0)
        score=float(np.square(dist/np.maximum(totals,1)-target).sum()+missing.sum()*10)
        if score<bestscore:bestscore=score;best=assignment
    for key,(split,comp) in forced.items():
        for r in comp:r['split']=split
    for k,(_,comp) in enumerate(free):
        for r in comp:r['split']=names[int(best[k])]
    return usable,quarantine


def contact_sheets(rows,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for label in (-1,0,1,2,3):
        selected=[r for r in rows if r['label']==label];random.Random(83).shuffle(selected)
        for page in range(min(2,(len(selected)+35)//36)):
            canvas=Image.new('RGB',(6*200,6*180),'white');draw=ImageDraw.Draw(canvas)
            for k,r in enumerate(selected[page*36:(page+1)*36]):
                x=k%6*200;y=k//6*180
                with Image.open(r['path']) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB');w,h=im.size
                    im.thumbnail((192,145));ww,hh=im.size;canvas.paste(im,(x,y))
                if r.get('bbox'):
                    b=r['bbox'];draw.rectangle((x+b[0]*ww,y+b[1]*hh,x+b[2]*ww,y+b[3]*hh),outline='red',width=2)
                draw.text((x+2,y+147),f"{r.get('split','?')} {r['id'][-23:]}",fill='black')
            name='background' if label==-1 else CLASSES[label]
            canvas.save(out/f'{name}_{page}.jpg',quality=92)


def export_annotations(rows,base):
    base=Path(base);labeldir=base/'yolo_labels';labeldir.mkdir(parents=True,exist_ok=True)
    for split in ('train','val','test'):
        subset=[r for r in rows if r['split']==split];images=[];anns=[]
        for i,r in enumerate(subset,1):
            if 'width' not in r:r.update(fingerprints(r['path']))
            w,h=r['width'],r['height'];images.append({'id':i,'file_name':r['path'],'width':w,'height':h,'record_id':r['id']})
            yolo=''
            if r['present']:
                x1,y1,x2,y2=r['bbox'];b=[x1*w,y1*h,(x2-x1)*w,(y2-y1)*h]
                anns.append({'id':len(anns)+1,'image_id':i,'category_id':r['label'],'bbox':b,'area':b[2]*b[3],'iscrowd':0})
                yolo=f"{r['label']} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}\n"
            (labeldir/(r['id']+'.txt')).write_text(yolo,encoding='utf-8')
        save_json(base/f'{split}.coco.json',{'images':images,'annotations':anns,'categories':[{'id':i,'name':n} for i,n in enumerate(CLASSES)],
                 'info':{'description':'Single-object detector subset; source provenance and review status are in manifest.jsonl'}})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=str(ROOT/'data/manifest.jsonl'))
    p.add_argument('--allow-incomplete',action='store_true',help='Prepare a clearly limited pilot with missing evaluation classes/negatives')
    p.add_argument('--instance-disjoint',action='store_true',help='Also keep each documented physical instance in one split; requires independent capture cohorts')
    p.add_argument('--inherit-manifest',help='Preserve every existing image role from an earlier frozen detector manifest')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Manifest already frozen; use a new output name')
    source=read_jsonl(ROOT/'data/source_annotations.jsonl')
    for captures in sorted((ROOT/'data/personal').rglob('captures.jsonl')):
        source+=read_jsonl(captures)
    byid={r['id']:r for r in source}
    reviews=ROOT/'data/reviews.jsonl'
    if reviews.exists():
        for r in read_jsonl(reviews):
            if r.get('annotation_status')!='human_verified' or not r.get('reviewer'):raise ValueError('Invalid manual review record')
            byid[r['id']]=r
    inherited={}
    if a.inherit_manifest:
        inherited={r['id']:r['split'] for r in read_jsonl(a.inherit_manifest)}
        for uid,split in inherited.items():
            if uid in byid:
                if byid[uid].get('fixed_split') and byid[uid]['fixed_split']!=split:raise ValueError('Inherited split conflicts with fixed original role')
                byid[uid]['fixed_split']=split
    trusted=[r for r in byid.values() if r.get('annotation_status') in TRUSTED]
    exclusions_path=ROOT/'data/exclusions.json'
    exclusions=json.loads(exclusions_path.read_text(encoding='utf-8')) if exclusions_path.exists() else []
    excluded_ids={r['id'] for r in exclusions}
    trusted=[r for r in trusted if r['id'] not in excluded_ids]
    for r in trusted:
        if r['annotation_status']=='source_ground_truth' and 'openimages/web/download' in r.get('source',''):
            split=r['original_source_split'];method=r['annotation_source'].rsplit(': ',1)[-1]
            r['annotation_source']=f"Open Images {'v6 train' if split=='train' else 'v5 '+split} boxes: {method}"
    # Verify both image identity and annotation range before grouping/export.
    for r in trusted:
        fp=fingerprints(r['path'])
        if r['sha256']!=fp['sha256'] or r['pixel_sha256']!=fp['pixel_sha256']:raise ValueError('Source image changed')
        r.update(fp)
    seen={};unique=[];duplicate_records=[]
    for r in sorted(trusted,key=lambda x:(x['annotation_status']!='human_verified',x['id'])):
        old=seen.get(r['pixel_sha256'])
        if old:
            if (old['label'],old['present'])!=(r['label'],r['present']):
                raise ValueError('Conflicting annotations on identical decoded pixels; manual reconciliation required')
            duplicate_records.append({'id':r['id'],'kept':old['id'],'reason':'Identical decoded image'})
            continue
        seen[r['pixel_sha256']]=r;unique.append(r)
    rows,quarantine=freeze(unique,a.instance_disjoint)
    try:counts=validate_rows(rows,require_negatives=not a.allow_incomplete,require_all_test_classes=not a.allow_incomplete)
    except ValueError as e:
        save_json(out.with_suffix('.blocked.json'),{'reason':str(e),'trusted':len(rows),'quarantined':quarantine,
                 'counts':dict(Counter((r['split']+':'+str(r['label'])) for r in rows)),
                 'action':'Complete local manual review; rerun. No candidate was promoted to ground truth.'})
        raise
    save_jsonl(out,rows)
    export_annotations(rows,out.parent/(out.stem+'_exports'))
    contact_sheets(rows,out.parent/(out.stem+'_review'))
    summary={'counts':counts,'total':len(rows),'groups':len({r['group'] for r in rows}),
             'annotation_status':dict(Counter(r['annotation_status'] for r in rows)),
             'sources':dict(Counter(r['source'] for r in rows)),'quarantined':quarantine,'exact_duplicates_removed':duplicate_records,
             'incomplete_pilot':a.allow_incomplete,'instance_disjoint':a.instance_disjoint,'manifest_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
             'inherited_manifest':a.inherit_manifest,'inherited_image_roles':len(inherited),
             'annotation_qa_exclusions':exclusions,
             'data_signature_sha256':hashlib.sha256(json.dumps([{k:r[k] for k in ('id','label','present','bbox','pixel_sha256','split','group')} for r in rows],sort_keys=True).encode()).hexdigest(),
             'limitations':['Cube webcam source kept entirely train-only.','No personal webcam benchmark exists.',
                 'Public source metadata cannot prove all unknown capture/physical instance independence.',
                 'Reviewed original classifier val/test images are legacy benchmarks, not fresh detector holdouts.']}
    save_json(out.with_suffix('.audit.json'),summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
