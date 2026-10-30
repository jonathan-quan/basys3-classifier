"""Capture full webcam frames with explicit session/instance metadata.

Space saves a positive candidate requiring a manually drawn box. N explicitly
confirms that NONE of the four supported objects is visible and saves a reviewed
negative. No boxes are guessed from the frame boundary.
"""
import argparse
import re
from datetime import datetime,timezone
from pathlib import Path

import cv2

from classifier.webcam import open_camera
from . import ROOT,CLASSES
from .fetch_data import fingerprints,save_jsonl,read_jsonl,save_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--label',choices=CLASSES,required=True)
    p.add_argument('--session',required=True);p.add_argument('--instance',required=True);p.add_argument('--camera',type=int,default=0)
    p.add_argument('--reviewer',default='owner');p.add_argument('--lighting',choices=('bright','dim','backlit','unknown'),default='unknown')
    p.add_argument('--background',choices=('plain','cluttered','unknown'),default='unknown');p.add_argument('--orientation',choices=('front','side','tilted','unknown'),default='unknown')
    a=p.parse_args()
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+',x) for x in (a.session,a.instance)):raise ValueError('Session/instance use letters, digits, underscore or hyphen')
    folder=ROOT/'data/personal'/a.session/a.instance;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'captures.jsonl';rows=read_jsonl(path) if path.exists() else []
    camera=open_camera(a.camera)
    try:
        while True:
            ok,frame=camera.read()
            if not ok:raise RuntimeError('Camera read failed')
            show=frame.copy()
            for i,line in enumerate(('Space: save POSITIVE candidate (box review later)','N: confirm NO cube / pen / book / computer mouse','Q: quit | Vary position, distance, lighting, background')):
                cv2.putText(show,line,(10,25+i*25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1,cv2.LINE_AA)
            cv2.imshow('Detector data capture',show);key=cv2.waitKey(1)&0xFF
            if key in (27,ord('q')):break
            if key not in (32,ord('n')):continue
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f');uid='personal_'+a.session+'_'+a.instance+'_'+stamp
            target=folder/(stamp+'.jpg')
            if not cv2.imwrite(str(target),frame):raise OSError('Cannot save capture')
            negative=key==ord('n')
            rows.append({'id':uid,'path':str(target),'label':-1 if negative else CLASSES.index(a.label),
                'present':0 if negative else 1,'bbox':None,'annotation_status':'human_verified' if negative else 'candidate',
                'annotation_source':'User explicitly confirmed absence with N' if negative else 'User capture, box not annotated',
                'reviewer':a.reviewer if negative else None,'source':'Own webcam capture','license':'User-owned',
                'group':'personal_session_'+a.session,'instance':a.instance,'session':a.session,'fixed_split':None,
                'conditions':{'lighting':a.lighting,'background':a.background,'orientation':a.orientation},
                **fingerprints(target)})
            save_jsonl(path,rows);print('Saved',uid,'reviewed negative' if negative else 'unboxed candidate',flush=True)
    finally:camera.release();cv2.destroyAllWindows()


if __name__=='__main__':main()
