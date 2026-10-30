"""SOFTWARE reference only: actual learned boxes, no contours or color thresholds."""
import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from classifier.common import choose_device,seed_everything
from classifier.webcam import open_camera
from . import ROOT,NAMES
from .model import load_detector,decode
from .data import letterbox,tensorize,inverse_box
from .fetch_data import save_json


def infer(model,image,device):
    image,_,transform=letterbox(image);x=tensorize(image).unsqueeze(0).to(device)
    if device.type=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.inference_mode():out=decode(model(x))
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    return {'box':inverse_box(out['boxes'][0].cpu().numpy(),transform).tolist(),
            'class_id':int(out['classes'][0]),'joint_score':float(out['scores'][0]),
            'class_score':float(out['class_scores'][0]),'presence_score':float(out['presence'][0]),
            'model_ms':elapsed*1000}


def overlay(frame,result,checkpoint,age=0.,preview_fps=0.,detection_fps=0.):
    pilot=checkpoint.get('incomplete_pilot',False)
    header='SOFTWARE reference | Laptop inference | FPGA not used'
    lines=[header,f'Preview {preview_fps:.1f} FPS | Detection {detection_fps:.1f} FPS']
    if result:
        lines.append(f"Model {result['model_ms']:.1f} ms | Result age {age*1000:.0f} ms")
        if age<.5 and result['joint_score']>=checkpoint['joint_score_threshold']:
            x1,y1,x2,y2=map(round,result['box'])
            if x2>x1 and y2>y1:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(40,240,80),2)
                label=f"{NAMES[result['class_id']]}: {result['class_score'] if pilot else result['joint_score']:.0%} {'class score' if pilot else 'score'}"
                cv2.putText(frame,label,(max(0,x1),max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),4,cv2.LINE_AA)
                cv2.putText(frame,label,(max(0,x1),max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,(40,240,80),1,cv2.LINE_AA)
        elif age>=.5:lines.append('Stale result; box hidden')
        else:lines.append('No prediction above validation-selected score threshold')
    if pilot:lines.append('INCOMPLETE PILOT: no verified background training / cube holdout')
    lines.append('Scores are uncalibrated | Q/Escape quits')
    for i,line in enumerate(lines):
        color=(30,130,255) if 'PILOT' in line else (255,255,255)
        cv2.putText(frame,line,(10,24+24*i),cv2.FONT_HERSHEY_SIMPLEX,.49,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(frame,line,(10,24+24*i),cv2.FONT_HERSHEY_SIMPLEX,.49,color,1,cv2.LINE_AA)
    return frame


class Worker:
    def __init__(self,model,device):
        self.model=model;self.device=device;self.condition=threading.Condition();self.pending=None
        self.result=None;self.error=None;self.stopped=False;self.count=0;self.started=time.perf_counter()
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def submit(self,frame,frame_id,timestamp):
        with self.condition:self.pending=(frame.copy(),frame_id,timestamp);self.condition.notify()
    def run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda:self.pending is not None or self.stopped)
                    if self.stopped:return
                    frame,uid,timestamp=self.pending;self.pending=None
                out=infer(self.model,Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),self.device)
                self.result={**out,'frame_id':uid,'captured_at':timestamp,'finished_at':time.perf_counter()};self.count+=1
        except Exception as e:self.error=e
    def close(self):
        with self.condition:self.stopped=True;self.condition.notify()
        self.thread.join(timeout=5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default=str(ROOT/'best_detector.pth'));p.add_argument('--camera',type=int,default=0)
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto');p.add_argument('--image');p.add_argument('--output')
    p.add_argument('--allow-pilot',action='store_true',help='Explicitly allow an incomplete, clearly labeled foreground-only model')
    a=p.parse_args();seed_everything(42);device=choose_device(a.device)
    if not Path(a.checkpoint).is_file():raise FileNotFoundError('No trained detector at this path. Original best_model.pth is a classifier, not a detector.')
    model,c=load_detector(a.checkpoint,device)
    if c.get('smoke_test'):raise ValueError('Synthetic test checkpoint is not an object detector for use')
    if c.get('incomplete_pilot') and not a.allow_pilot:raise ValueError('This model is incomplete. Use --allow-pilot only to test its clearly labeled limitations.')
    if a.image:
        with Image.open(a.image) as im:
            result=infer(model,im,device)
            from PIL import ImageOps
            rgb=np.array(ImageOps.exif_transpose(im).convert('RGB'));frame=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        print(result)
        if a.output:
            output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(output),overlay(frame,result,c)):raise OSError('Could not write prediction image')
            save_json(output.with_suffix('.json'),{**result,'software_reference':True,'incomplete_pilot':c.get('incomplete_pilot',False)})
        return
    camera=open_camera(a.camera);worker=Worker(model,device);start=time.perf_counter();frames=0
    try:
        while True:
            ok,frame=camera.read()
            if not ok:raise RuntimeError('Camera stopped delivering frames')
            now=time.perf_counter();frames+=1;worker.submit(frame,frames,now)
            if worker.error:raise worker.error
            result=worker.result;age=now-result['captured_at'] if result else 0
            overlay(frame,result,c,age,frames/max(.001,now-start),worker.count/max(.001,now-worker.started))
            cv2.imshow('Basys3 detector - SOFTWARE reference',frame)
            if cv2.waitKey(1)&0xFF in (27,ord('q')):break
            if cv2.getWindowProperty('Basys3 detector - SOFTWARE reference',cv2.WND_PROP_VISIBLE)<1:break
    finally:worker.close();camera.release();cv2.destroyAllWindows()


if __name__=='__main__':main()
