"""Hardware verification benchmark, not normal webcam inference.

Every prediction comes from the board and is checked against the independent
integer software reference. No result is substituted with a laptop prediction.
"""
import argparse
import csv
import json
import time

import numpy as np
from PIL import Image

from . import ROOT,PROJECT
from .client import FPGAClient,preprocess
from .quantize import IntegerClassifier,classify_logits,save_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port');p.add_argument('--limit',type=int)
    p.add_argument('--output',default=str(ROOT/'hardware_benchmark.json'));a=p.parse_args()
    with (PROJECT/'data/manifest.csv').open(newline='',encoding='utf-8') as f:rows=[r for r in csv.DictReader(f) if r['split']=='test']
    if a.limit:rows=rows[:a.limit]
    labels=['rubiks_cube','pen','book','mouse'];ref=IntegerClassifier.load(ROOT/'export');client=FPGAClient(a.port)
    matrix=np.zeros((4,4),int);results=[];start=time.perf_counter()
    try:
        for i,row in enumerate(rows):
            with Image.open(row['path']) as image:
                actual=client.classify(image)
                pixels=np.array(preprocess(image));q=(pixels.astype(np.int16)-128).astype(np.int8).transpose(2,0,1)
                expected=ref.run(q)[0];label,score=classify_logits(expected)
            if actual['logits_q8_8']!=expected.tolist() or actual['class_id']!=label or actual['confidence_u16']!=score:
                raise AssertionError({'image':row['path'],'expected_logits':expected.tolist(),'actual':actual})
            truth=labels.index(row['label']);matrix[truth,actual['class_id']]+=1
            results.append({'image_id':row['path'].replace('\\','/').split('/')[-1],
                            'truth':row['label'],**actual,'integer_reference_exact':True})
            if (i+1)%25==0 or i+1==len(rows):print(f'Hardware frames checked {i+1}/{len(rows)} | correct {matrix.trace()}/{i+1}',flush=True)
    finally:client.close()
    elapsed=time.perf_counter()-start
    report={'examples':len(rows),'hardware_accuracy':float(matrix.trace()/matrix.sum()),'correct':int(matrix.trace()),
        'all_logits_and_scores_match_integer_reference':True,'model_identity':client.identity,
        'confusion_matrix':matrix.tolist(),'per_class':{n:{'examples':int(matrix[i].sum()),'accuracy':float(matrix[i,i]/matrix[i].sum()) if matrix[i].sum() else None} for i,n in enumerate(labels)},
        'median_fpga_inference_ms':float(np.median([r['inference_ms'] for r in results])),
        'median_uart_roundtrip_ms':float(np.median([r['roundtrip_ms'] for r in results])),
        'total_verification_seconds':elapsed,'results':results,
        'scope':'Actual FPGA inference on the original frozen public-photo classifier test set; not a personal-object webcam accuracy measurement.'}
    save_json(a.output,report);print('HARDWARE BENCHMARK COMPLETE',json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
