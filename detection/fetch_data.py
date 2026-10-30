"""Download documented existing annotations, never invent ground truth.

Open Images core validation/test boxes and per-image license/author metadata;
phonght32's MIT project with explicitly self-captured webcam cube images.
Original classifier photos are referenced, not modified. Unboxed originals
are candidates only. Cube source is conservatively a single train-only group.
"""
import argparse
import csv
import hashlib
import json
import random
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps

from . import ROOT, PROJECT, CLASSES

MID = {"/m/0h0b5": 0, "/m/0k1tl": 1, "/m/0bt_c3": 2, "/m/020lf": 3}
CUBE_REPO = "phonght32/yoloV2-detectRubik"
CUBE_COMMIT = "7bcf8dcf0c1d84ae5657b18ecd05af0d15e1edcf"
OI_SOURCE = "https://storage.googleapis.com/openimages/web/download_v7.html"
ALLOWED_LICENSES = {
    "https://creativecommons.org/licenses/by/2.0",
    "https://creativecommons.org/licenses/by/3.0",
    "https://creativecommons.org/licenses/by/4.0",
    "https://creativecommons.org/publicdomain/zero/1.0",
}


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows), encoding="utf-8")


def download(url, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size:
        return path
    partial = path.with_suffix(path.suffix+".part")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Basys3Detector-EducationalResearch/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response, partial.open("wb") as out:
                while chunk := response.read(1024*1024):
                    out.write(chunk)
            partial.replace(path)
            return path
        except Exception:
            if attempt == 2: raise
            time.sleep(attempt+1)


def fingerprints(path):
    path = Path(path)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        w,h = image.size
        if min(w,h)<32: raise ValueError("Image too small")
        pixel = hashlib.sha256(f"{w},{h}:".encode()+image.tobytes()).hexdigest()
        tiny = list(image.convert("L").resize((9,8),Image.Resampling.LANCZOS).get_flattened_data())
        bits = sum(int(tiny[y*9+x+1]>tiny[y*9+x]) << (y*8+x) for y in range(8) for x in range(8))
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pixel_sha256": pixel,
            "dhash": f"{bits:016x}", "width":w,"height":h}


def fetch_open_images(limit=900, workers=8, include_train=False):
    cache = ROOT/"data/sources/open_images"
    splits=("train","validation","test") if include_train else ("validation","test")
    jobs=[]
    for split in splits:
        boxes_url=("https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv" if split=="train" else f"https://storage.googleapis.com/openimages/v5/{split}-annotations-bbox.csv")
        meta_url=("https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv" if split=="train" else f"https://storage.googleapis.com/openimages/2018_04/{split}/{split}-images-with-rotation.csv")
        jobs += [(boxes_url, cache/f"{split}-boxes.csv"),(meta_url,cache/f"{split}-metadata.csv")]
    with ThreadPoolExecutor(4) as ex:
        for p in ex.map(lambda j: download(*j), jobs): print("Metadata cached",p.name,flush=True)
    boxes=defaultdict(list)
    for split in splits:
        with (cache/f"{split}-boxes.csv").open(newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["LabelName"] not in MID: continue
                # Retain all supported boxes while filtering single-target images.
                boxes[(split,row["ImageID"])].append(row)
    counts=Counter(); metadata=[]; excluded=Counter()
    for split in splits:
        with (cache/f"{split}-metadata.csv").open(newline="",encoding="utf-8") as f:
            for m in csv.DictReader(f):
                key=(split,m["ImageID"]); candidates=boxes.get(key,[])
                if len(candidates)!=1:
                    if candidates: excluded["multiple_supported_instances"]+=1
                    continue
                box=candidates[0]
                if any(box.get(flag,"0")!="0" for flag in ("IsGroupOf","IsDepiction","IsInside")):
                    excluded["annotation_flags"]+=1;continue
                license_url=m.get("License","").replace("http://","https://").rstrip("/")
                if license_url not in ALLOWED_LICENSES:
                    excluded["license"]+=1;continue
                # A rotation mismatch would invalidate coordinates. Only zero is accepted.
                if m.get("Rotation","0") not in ("0","0.0"):
                    excluded["rotation_not_zero"]+=1;continue
                author=m.get("AuthorProfileURL") or m.get("Author")
                if not author or author.lower() in ("nan","unknown","n/a"):
                    excluded["unknown_author"]+=1;continue
                xyxy=[float(box[k]) for k in ("XMin","YMin","XMax","YMax")]
                if not (0<=xyxy[0]<xyxy[2]<=1 and 0<=xyxy[1]<xyxy[3]<=1):
                    excluded["invalid_box"]+=1;continue
                if min(xyxy[2]-xyxy[0],xyxy[3]-xyxy[1])*96<2:
                    excluded["less_than_two_resized_pixels"]+=1;continue
                label=MID[box["LabelName"]];counts[CLASSES[label]]+=1
                metadata.append((split,m,box,xyxy,label,author))
    rng=random.Random(42);rng.shuffle(metadata)
    authorcounts=Counter();selected=[];cc=Counter()
    for item in metadata:
        split,m,b,xy,label,author=item
        group="flickr-author-"+hashlib.sha256(author.strip().casefold().encode()).hexdigest()[:16]
        if cc[label]>=limit or authorcounts[(label,group)]>=15:continue
        selected.append((item,group));cc[label]+=1;authorcounts[(label,group)]+=1
    print("OI eligible before caps",dict(counts),"selected",dict(cc),"excluded",dict(excluded),flush=True)
    def one(pair):
        (split,m,box,xy,label,author),group=pair
        image_id=m["ImageID"];url=f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
        path=ROOT/"data/images/open_images"/f"{image_id}.jpg"
        download(url,path)
        return {"id":"oi_"+image_id,"path":str(path),"label":label,"bbox":xy,"present":1,
                "annotation_status":"source_ground_truth","annotation_source":f"Open Images {'v6 train' if split=='train' else 'v5 '+split} boxes: {box['Source']}",
                "source":OI_SOURCE,"license":m["License"],"author":m.get("Author",""),
                "author_profile":author,"original_url":m.get("OriginalURL",""),
                "landing_url":m.get("OriginalLandingURL",""),"download_url":url,
                "group":group,"original_source_split":split,"fixed_split":None,
                "quality_flags":{k:box.get(k) for k in ("IsOccluded","IsTruncated")},
                **fingerprints(path)}
    accepted=[];failures=[]
    with ThreadPoolExecutor(workers) as ex:
        futures={ex.submit(one,pair):pair for pair in selected}
        for i,f in enumerate(as_completed(futures),1):
            try:accepted.append(f.result())
            except Exception as e:failures.append({"id":futures[f][0][1]["ImageID"],"error":str(e)})
            if i%100==0 or i==len(futures):print("OI image downloads",i,"/",len(futures),flush=True)
    save_json(cache/"selection.json",{"eligible_counts":dict(counts),"excluded":dict(excluded),"failures":failures,
              "note":"Official source boxes, not all locally human-reviewed; source core validation/test are re-grouped into a new benchmark, not original OI evaluation."})
    return accepted


def fetch_cubes(workers=8, limit=800):
    source=ROOT/"data/sources/cube_webcam"
    raw=f"https://raw.githubusercontent.com/{CUBE_REPO}/{CUBE_COMMIT}/"
    for name in ("README.md","LICENSE"):
        download(raw+name,source/name)
    treepath=download(f"https://api.github.com/repos/{CUBE_REPO}/git/trees/{CUBE_COMMIT}?recursive=1",source/"tree.json")
    tree=json.loads(treepath.read_text(encoding="utf-8"))["tree"]
    images=sorted([x["path"] for x in tree if x["path"].startswith("datasets/Images/001/") and x["path"].lower().endswith((".jpg",".jpeg",".png"))])[:limit]
    paths={x["path"] for x in tree}
    def one(name):
        stem=Path(name).stem
        labname="datasets/Labels/001/"+stem+".txt"
        if labname not in paths:return None
        labelpath=download(raw+urllib.parse.quote(labname),source/"labels"/(stem+".txt"))
        tokens=labelpath.read_text().split()
        # Original BBoxLabelTool format: count, then xmin ymin xmax ymax in pixels.
        if not tokens or tokens[0]!="1" or len(tokens)!=5:return None
        path=ROOT/"data/images/cube_webcam"/(stem+".jpg")
        download(raw+urllib.parse.quote(name),path);fp=fingerprints(path)
        x1,y1,x2,y2=map(float,tokens[1:]);w,h=fp["width"],fp["height"]
        if not (0<=x1<x2<=w and 0<=y1<y2<=h):return None
        return {"id":"cube_webcam_"+stem,"path":str(path),"label":0,"present":1,
                "bbox":[x1/w,y1/h,x2/w,y2/h],"annotation_status":"source_ground_truth",
                "annotation_source":"Original BBoxLabelTool labels, source provided; not generated by this project",
                "source":f"https://github.com/{CUBE_REPO}","source_commit":CUBE_COMMIT,"license":"MIT",
                "author":"phonght32","group":"cube-webcam-entire-source","fixed_split":"train",
                "download_url":raw+urllib.parse.quote(name),"quality_flags":{},**fp}
    rows=[];failures=[]
    with ThreadPoolExecutor(workers) as ex:
        fs={ex.submit(one,n):n for n in images}
        for i,f in enumerate(as_completed(fs),1):
            try:
                row=f.result()
                if row:rows.append(row)
            except Exception as e:failures.append({"path":fs[f],"error":str(e)})
            if i%100==0 or i==len(fs):print("Cube downloads",i,"/",len(fs),flush=True)
    save_json(source/"provenance.json",{"repository":CUBE_REPO,"commit":CUBE_COMMIT,"license":"MIT",
          "note":"Author describes images as self-captured webcam photos. Entire collection forced train-only because capture sessions/instances are undocumented.",
          "images_in_tree":len(images),"accepted":len(rows),"failures":failures})
    return rows


def existing_candidates():
    rows=[]
    with (PROJECT/"data/manifest.csv").open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"id":"original_"+Path(row["path"]).stem,"path":row["path"],
                "label":CLASSES.index(row["label"]),"present":1,"bbox":None,
                "annotation_status":"candidate","annotation_source":"No box; manual review required",
                "source":row["source"],"license":row["license"],"group":"original_"+row["group"],
                "fixed_split":row["split"],"original_classifier_split":row["split"],
                "sha256":row["sha256"],"pixel_sha256":row["pixel_sha256"],"dhash":row["dhash"],
                "quality_flags":{}})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--limit",type=int,default=900)
    p.add_argument("--workers",type=int,default=8);p.add_argument("--cube-limit",type=int,default=800)
    p.add_argument("--skip-cubes",action="store_true");a=p.parse_args()
    (ROOT/"data").mkdir(exist_ok=True)
    rows=fetch_open_images(a.limit,a.workers,include_train=True)
    if not a.skip_cubes:rows+=fetch_cubes(a.workers,a.cube_limit)
    save_jsonl(ROOT/"data/source_annotations.jsonl",sorted(rows,key=lambda r:r['id']))
    save_jsonl(ROOT/"data/original_candidates.jsonl",existing_candidates())
    print("Accepted provided annotations",dict(Counter(CLASSES[r['label']] for r in rows)),flush=True)


if __name__=="__main__":main()
