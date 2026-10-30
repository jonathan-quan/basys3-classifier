"""Laptop preview with classification executed ONLY by the Basys 3 FPGA."""
import argparse
import threading
import time

import cv2
import numpy as np
from PIL import Image

from .client import FPGAClient


def text(frame,value,y,color=(255,255,255),scale=.5):
    (w,h),baseline=cv2.getTextSize(value,cv2.FONT_HERSHEY_SIMPLEX,scale,1)
    mask=np.zeros((h+baseline+6,w+6),np.uint8)
    cv2.putText(mask,value,(3,h+3),cv2.FONT_HERSHEY_SIMPLEX,scale,255,1,cv2.LINE_AA)
    outline=cv2.dilate(mask,np.ones((3,3),np.uint8));left,top=9,y-h-3
    x1,y1=max(0,left),max(0,top);x2,y2=min(frame.shape[1],left+mask.shape[1]),min(frame.shape[0],top+mask.shape[0])
    if x2<=x1 or y2<=y1:return
    a=mask[y1-top:y2-top,x1-left:x2-left,None].astype(np.float32)/255
    b=outline[y1-top:y2-top,x1-left:x2-left,None].astype(np.float32)/255
    patch=frame[y1:y2,x1:x2].astype(np.float32)*(1-b)
    frame[y1:y2,x1:x2]=(patch*(1-a)+np.array(color)*a).astype(np.uint8)


class Worker:
    def __init__(self,port):
        self.port=port;self.client=None;self.condition=threading.Condition();self.pending=None;self.result=None
        self.error='Connecting to FPGA...';self.closed=False;self.count=0;self.started=time.perf_counter()
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def submit(self,frame,captured_at,full_frame):
        with self.condition:self.pending=(frame.copy(),captured_at,full_frame);self.condition.notify()
    def run(self):
        while not self.closed:
            try:
                if self.client is None:self.client=FPGAClient(self.port);self.error=None
                with self.condition:
                    self.condition.wait_for(lambda:self.pending is not None or self.closed)
                    if self.closed:break
                    frame,timestamp,full=self.pending;self.pending=None
                if not full:
                    h,w=frame.shape[:2];side=int(min(h,w)*.75);left=(w-side)//2;top=(h-side)//2;frame=frame[top:top+side,left:left+side]
                image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                result=self.client.classify(image);result['captured_at']=timestamp;self.result=result;self.count+=1
            except Exception as error:
                self.error=str(error);self.result=None
                if self.client:
                    try:self.client.close()
                    except Exception:pass
                    self.client=None
                with self.condition:
                    if not self.closed:self.condition.wait(timeout=2)
        if self.client:self.client.close()
    def close(self):
        with self.condition:self.closed=True;self.condition.notify()
        self.thread.join(timeout=5)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port');p.add_argument('--camera',type=int,default=0);a=p.parse_args()
    camera=cv2.VideoCapture(a.camera,cv2.CAP_DSHOW)
    if not camera.isOpened():camera.release();camera=cv2.VideoCapture(a.camera)
    if not camera.isOpened():raise RuntimeError('Cannot open webcam; close other camera apps and check Windows camera permissions')
    worker=Worker(a.port);frames=0;started=time.perf_counter();full=False
    try:
        while True:
            ok,frame=camera.read()
            if not ok:raise RuntimeError('Webcam stopped delivering frames')
            now=time.perf_counter();frames+=1;worker.submit(frame,now,full);result=worker.result
            text(frame,'BASYS 3 FPGA CLASSIFICATION - no laptop neural inference',25,(50,240,100),.53)
            if result and now-result['captured_at']<1.5:
                text(frame,f"{result['name']}: {result['class_score']:.1%} score",62,(50,240,100),.8)
                text(frame,f"FPGA {result['inference_ms']:.2f} ms | UART round trip {result['roundtrip_ms']:.0f} ms",91)
            else:text(frame,'Hardware inference unavailable' if worker.error else 'Waiting for FPGA prediction...',62,(20,160,255),.6)
            text(frame,f"Preview {frames/max(.01,now-started):.1f} FPS | FPGA updates {worker.count/max(.01,now-worker.started):.1f} FPS",117)
            text(frame,'Input: full frame' if full else 'Input: central 75% square; hold object near center',143)
            text(frame,'F changes framing | Q/Escape quits | Uncalibrated four-class score',169,scale=.45)
            if worker.error:text(frame,worker.error[:105],195,(20,160,255),.43)
            cv2.imshow('Basys 3 - REAL FPGA recognition',frame);key=cv2.waitKey(1)&0xff
            if key in (27,ord('q'),ord('Q')):break
            if key in (ord('f'),ord('F')):full=not full;worker.result=None
            if cv2.getWindowProperty('Basys 3 - REAL FPGA recognition',cv2.WND_PROP_VISIBLE)<1:break
    finally:worker.close();camera.release();cv2.destroyAllWindows()


if __name__=='__main__':main()
