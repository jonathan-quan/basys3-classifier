"""PTQ/export of the preserved classifier. No retraining or checkpoint changes."""
import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from classifier.common import seed_everything,choose_device,file_hash
from classifier.data import image_transform
from classifier.evaluate import load_checkpoint
from . import ROOT,PROJECT


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def round_away(x):return np.sign(x)*np.floor(np.abs(x)+.5)


def multiplier(ratio):
    # Positive <=23-bit multiplier; exact signed 64-bit intermediate in reference/RTL.
    shift=max(0,min(55,int(math.floor(math.log2((2**23-1)/ratio)))))
    m=int(math.floor(ratio*2**shift+.5))
    while m>=2**23:m=(m+1)//2;shift-=1
    if m<=0 or shift<0:raise ValueError('Unrepresentable scale')
    return m,shift


def requant(acc,m,shift,bits=8,relu=True):
    acc=np.asarray(acc,dtype=np.int64);m=np.asarray(m,dtype=np.int64);shift=np.asarray(shift,dtype=np.int64)
    product=acc*m
    half=np.where(shift>0,np.left_shift(np.int64(1),np.maximum(shift-1,0)),0)
    out=np.sign(product)*np.right_shift(np.abs(product)+half,shift)
    return np.clip(out,0 if relu else -(1<<(bits-1)),(1<<(bits-1))-1).astype(np.int8 if bits==8 else np.int16)


class IntegerClassifier:
    def __init__(self,description,arrays):
        self.description=description;self.arrays=arrays

    @classmethod
    def load(cls,folder):
        folder=Path(folder);return cls(json.loads((folder/'model.json').read_text()),dict(np.load(folder/'weights.npz')))

    def run(self,q,trace=False):
        x=np.asarray(q,dtype=np.int8)
        if x.ndim==3:x=x[None]
        states=[]
        for i,layer in enumerate(self.description['layers']):
            w=self.arrays[f'w{i}'].astype(np.int64);b=self.arrays[f'b{i}'].astype(np.int64)
            k=layer['kernel'];s=layer['stride'];p=layer['pad']
            padded=np.pad(x.astype(np.int64),((0,0),(0,0),(p,p),(p,p)))
            windows=np.lib.stride_tricks.sliding_window_view(padded,(k,k),axis=(2,3))[:,:,::s,::s]
            n,ci,h,ww,_,_=windows.shape
            if layer['depthwise']:
                acc=(windows*w[None,:,None,None,0,:,:]).sum((-2,-1))+b[None,:,None,None]
            else:
                patches=windows.transpose(0,2,3,1,4,5).reshape(-1,ci*k*k)
                acc=(patches@w.reshape(w.shape[0],-1).T).reshape(n,h,ww,w.shape[0]).transpose(0,3,1,2)+b[None,:,None,None]
            if np.max(np.abs(acc))>=2**31:raise OverflowError('INT32 convolution overflow')
            x=requant(acc,self.arrays[f'm{i}'][None,:,None,None],self.arrays[f's{i}'][None,:,None,None])
            if trace:states.append(x.copy())
        pooled=((x.astype(np.int64).sum((2,3))+18)//36).astype(np.int8)
        acc=pooled.astype(np.int64)@self.arrays['wf'].astype(np.int64).T+self.arrays['bf']
        logits=requant(acc,self.arrays['mf'],self.arrays['sf'],bits=16,relu=False)
        return (logits,states,pooled) if trace else logits


def export(model,ranges,percentile,calibration):
    layers=[];arrays={};scale=1/128;woff=0;poff=0;hw=96
    for name,module in model.features.named_children():
        if not isinstance(module,torch.nn.Conv2d):continue
        i=len(layers);output_scale=max(float(np.quantile(ranges[name],percentile)),1e-8)/127
        weights=module.weight.detach().cpu().numpy();bias=module.bias.detach().cpu().numpy()
        ws=np.maximum(np.abs(weights).reshape(weights.shape[0],-1).max(1)/127,1e-12)
        qw=round_away(weights/ws[:,None,None,None]).clip(-127,127).astype(np.int8)
        qb=round_away(bias/(scale*ws)).astype(np.int64)
        if np.max(np.abs(qb))>=2**31:raise OverflowError('Bias does not fit INT32')
        ms=[multiplier(float(v)) for v in scale*ws/output_scale]
        arrays.update({f'w{i}':qw,f'b{i}':qb.astype(np.int32),f'm{i}':np.array([v[0] for v in ms],np.int32),f's{i}':np.array([v[1] for v in ms],np.int32)})
        k=module.kernel_size[0];s=module.stride[0];p=module.padding[0];out=math.ceil(hw/s)
        layers.append({'name':'features.'+name,'cin':module.in_channels,'cout':module.out_channels,'kernel':k,'stride':s,'pad':p,
            'depthwise':module.groups==module.in_channels and module.groups>1,'input_hw':hw,'output_hw':out,
            'weight_offset':woff,'parameter_offset':poff,'input_scale':scale,'output_scale':output_scale})
        woff+=qw.size;poff+=qw.shape[0];scale=output_scale;hw=out
    weights=model.fc.weight.detach().cpu().numpy();bias=model.fc.bias.detach().cpu().numpy()
    ws=np.maximum(np.abs(weights).max(1)/127,1e-12);qw=round_away(weights/ws[:,None]).clip(-127,127).astype(np.int8)
    qb=round_away(bias/(scale*ws)).astype(np.int32);ms=[multiplier(float(v)) for v in scale*ws/(1/256)]
    arrays.update(wf=qw,bf=qb,mf=np.array([v[0] for v in ms],np.int32),sf=np.array([v[1] for v in ms],np.int32))
    desc={'architecture':'context_classifier_int8','classes':['rubiks_cube','pen','book','mouse'],'input':[3,96,96],
        'checkpoint_sha256':file_hash(PROJECT/'best_model.pth'),'layers':layers,'fc_weight_offset':woff,'fc_parameter_offset':poff,
        'activation_percentile':percentile,'input_scale':1/128,'logit_scale':1/256,'zero_points':0,
        'weight_quantization':'Signed symmetric INT8 per output channel, -127..127',
        'activation_quantization':'Signed INT8 per layer; ReLU 0..127; input pixel-128',
        'rounding':'Nearest, halfway away from zero','gap':'(positive integer sum + 18)//36',
        'calibration_image_ids':calibration,'retrained':False}
    return IntegerClassifier(desc,arrays)


def load_images(rows):
    transform=image_transform(False);tensors=[]
    for r in rows:
        with Image.open(r['path']) as image:tensors.append(transform(image))
    return torch.stack(tensors)


def write_mem(path,values,width):
    mask=(1<<width)-1
    # Never coerce mixed high/low Python uint64 words through np.asarray: it can
    # choose float64 and silently lose low bits of packed bias/multiplier/shift.
    flat=values.reshape(-1) if isinstance(values,np.ndarray) else values
    Path(path).write_text(''.join(f'{int(v)&mask:0{(width+3)//4}x}\n' for v in flat),encoding='ascii')


def classify_logits(logits):
    logits=np.asarray(logits,dtype=np.int64)
    label=int(logits.argmax())
    index=np.clip((logits.max()-logits+4)>>3,0,512)
    exp=round_away(65535*np.exp(-index/32)).astype(np.int64)
    total=int(exp.sum())
    return label,(65535*65535+total//2)//total


def write_export(reference,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/'model.json',reference.description);np.savez(folder/'weights.npz',**reference.arrays)
    weights=[];meta=[]
    for i in range(10):
        suffix=str(i) if i<9 else 'f';a=reference.arrays
        weights.extend(a['w'+suffix].reshape(-1))
        for b,m,s in zip(a['b'+suffix],a['m'+suffix],a['s'+suffix]):
            meta.append(((int(b)&0xffffffff)<<32)|(int(m)<<8)|int(s))
    weights.extend([0]*(8192-len(weights)));meta.extend([0]*(512-len(meta)))
    write_mem(folder/'weights.mem',weights,8);write_mem(folder/'parameters.mem',meta,64)
    exp=round_away(65535*np.exp(-np.arange(513)/32)).astype(np.uint16)
    write_mem(folder/'exp.mem',exp,16)
    model_id=hashlib.sha256(b''.join((folder/n).read_bytes() for n in ('weights.mem','parameters.mem','exp.mem'))).hexdigest()[:16]
    (folder/'hardware_id.svh').write_text(f"localparam [63:0] MODEL_ID=64'h{model_id};\n",encoding='ascii')
    save_json(folder/'hardware_identity.json',{'model_id_hex':model_id,'source_checkpoint_sha256':reference.description['checkpoint_sha256']})
    lines=['// Generated from the preserved classifier. Do not hand-edit.']
    for key in ('cin','cout','kernel','stride','pad','input_hw','output_hw','weight_offset','parameter_offset'):
        vals=[str(int(r[key])) for r in reference.description['layers']]
        lines.append(f"localparam integer L_{key.upper()} [0:8] = '{{{','.join(vals)}}};")
    for key,values in {
        'INPUT_PLANE':[r['input_hw']**2 for r in reference.description['layers']],
        'OUTPUT_PLANE':[r['output_hw']**2 for r in reference.description['layers']],
        'KERNEL_AREA':[r['kernel']**2 for r in reference.description['layers']],
        'WEIGHT_STEP':[(1 if r['depthwise'] else r['cin'])*r['kernel']**2 for r in reference.description['layers']],
    }.items():lines.append(f"localparam integer L_{key} [0:8] = '{{{','.join(map(str,values))}}};")
    lines.append("localparam bit L_DEPTHWISE [0:8] = '{"+','.join("1'b1" if r['depthwise'] else "1'b0" for r in reference.description['layers'])+'};')
    (folder/'layers.svh').write_text('\n'.join(lines)+'\n',encoding='ascii')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=str(ROOT/'export'));a=p.parse_args()
    output=Path(a.output)
    if (output/'model.json').exists():raise FileExistsError('Existing integer export; choose another output directory')
    seed_everything(42);device=choose_device();model,_=load_checkpoint(PROJECT/'best_model.pth',device)
    with (PROJECT/'data/manifest.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    rng=np.random.default_rng(42);cal=[]
    for label in ('rubiks_cube','pen','book','mouse'):
        bysource=defaultdict(list)
        for r in rows:
            if r['split']=='train' and r['label']==label:bysource[r['source']].append(r)
        for subset in bysource.values():
            inds=rng.permutation(len(subset))[:80//len(bysource)];cal.extend(subset[int(i)] for i in inds)
    ranges=defaultdict(list);hooks=[]
    for name,module in model.features.named_children():
        if isinstance(module,torch.nn.Conv2d):
            def record(module,inp,out,name=name):ranges[name].append(out.detach().relu().cpu().numpy().reshape(-1))
            hooks.append(module.register_forward_hook(record))
    for start in range(0,len(cal),32):
        with torch.inference_mode():model(load_images(cal[start:start+32]).to(device))
    for h in hooks:h.remove()
    ranges={k:np.concatenate(v) for k,v in ranges.items()}
    val=[r for r in rows if r['split']=='val'];images=load_images(val);q=round_away(images.numpy()*128).astype(np.int8)
    labels=np.array([('rubiks_cube','pen','book','mouse').index(r['label']) for r in val])
    candidates=[];best=None
    for percentile in (.999,.9999,1.):
        ref=export(model,ranges,percentile,[Path(r['path']).stem for r in cal]);preds=[]
        started=time.perf_counter()
        for start in range(0,len(q),16):preds.extend(ref.run(q[start:start+16]).argmax(1).tolist())
        accuracy=float((np.array(preds)==labels).mean());candidates.append({'percentile':percentile,'validation_accuracy':accuracy,'seconds':time.perf_counter()-started})
        print(json.dumps(candidates[-1]),flush=True)
        if best is None or accuracy>best[0]:best=(accuracy,ref)
    ref=best[1];write_export(ref,output)
    save_json(output/'quantization_validation.json',{'calibration_images':len(cal),'calibration_split':'train only',
        'candidates':candidates,'selected_validation_accuracy':best[0],'fp32_validation_accuracy':.7347826086956522,
        'test_used_for_selection':False,'original_checkpoint_sha256':file_hash(PROJECT/'best_model.pth')})
    # Predetermined real-image simulation vectors, no test set tuning.
    vectors=output/'vectors';vectors.mkdir(exist_ok=True)
    for index in range(3):
        logits,trace,pooled=ref.run(q[index],trace=True)
        write_mem(vectors/f'input{index}.mem',q[index],8)
        for layer,state in enumerate(trace):write_mem(vectors/f'case{index}_layer{layer}.mem',state,8)
        write_mem(vectors/f'pooled{index}.mem',pooled,8);write_mem(vectors/f'logits{index}.mem',logits,16)
        _,confidence=classify_logits(logits[0]);write_mem(vectors/f'confidence{index}.mem',[confidence],16)
        save_json(vectors/f'case{index}.json',{'image':val[index]['path'],'label':val[index]['label'],'logits':logits[0].tolist(),'predicted':int(logits[0].argmax())})
    print('Exported',output,'validation accuracy',best[0],flush=True)


if __name__=='__main__':main()
