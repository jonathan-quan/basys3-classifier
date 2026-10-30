"""CRC-framed USB UART client. In hardware mode no laptop NN is executed."""
import argparse
import json
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports
from PIL import Image,ImageOps

from . import ROOT

MAGIC=b'\xb3\x5d\xa7\xc1'
BAUD=1000000
NAMES=("Rubik's Cube","Pen","Book","Computer Mouse")


def encode(kind,frame_id,payload=b''):
    header=struct.pack('<BBII',1,kind,frame_id,len(payload))
    return MAGIC+header+payload+struct.pack('<I',zlib.crc32(header+payload))


def preprocess(image):
    image=ImageOps.exif_transpose(image).convert('RGB')
    return ImageOps.pad(image,(96,96),method=Image.Resampling.BILINEAR,color=(128,128,128),centering=(.5,.5))


def find_port():
    ports=[p for p in list_ports.comports() if p.vid==0x0403 and p.pid==0x6010]
    if len(ports)!=1:raise RuntimeError(f'Expected one Basys 3 UART; found {[p.device for p in ports]}. Use --port explicitly.')
    return ports[0].device


class FPGAClient:
    def __init__(self,port=None,timeout=3.):
        self.port=port or find_port();self.timeout=timeout;self.frame_id=0
        self.serial=serial.Serial(self.port,BAUD,timeout=.1,write_timeout=5)
        self.serial.reset_input_buffer()
        try:self.identity=self.hello()
        except Exception:self.close();raise

    def close(self):self.serial.close()

    def _read(self,n,deadline):
        result=bytearray()
        while len(result)<n:
            if time.perf_counter()>deadline:raise TimeoutError('FPGA response timeout; check bitstream, power and COM port')
            result.extend(self.serial.read(n-len(result)))
        return bytes(result)

    def receive(self):
        deadline=time.perf_counter()+self.timeout;window=bytearray()
        while bytes(window)!=MAGIC:
            window.extend(self._read(1,deadline))
            if len(window)>4:del window[0]
        header=self._read(10,deadline);version,kind,uid,length=struct.unpack('<BBII',header)
        if version!=1 or length>256:raise ValueError('Invalid FPGA packet header')
        payload=self._read(length,deadline);crc=struct.unpack('<I',self._read(4,deadline))[0]
        if zlib.crc32(header+payload)!=crc:raise ValueError('FPGA response CRC mismatch')
        return kind,uid,payload

    def request(self,kind,payload=b''):
        self.frame_id=(self.frame_id+1)&0xffffffff;uid=self.frame_id
        self.serial.write(encode(kind,uid,payload));self.serial.flush()
        rkind,rid,result=self.receive()
        if rid!=uid:raise ValueError('Stale/mismatched FPGA frame ID')
        if rkind==0xff:raise RuntimeError(f'FPGA rejected packet (error code {result[0]})')
        return rkind,result

    def hello(self):
        kind,payload=self.request(1)
        if kind!=0x81 or len(payload)!=16:raise ValueError('Unexpected HELLO response')
        signature,side,classes,mode,model_id=struct.unpack('<4sHBBQ',payload)
        if (signature,side,classes,mode)!=(b'B3C1',96,4,1):raise ValueError('Wrong FPGA design/protocol')
        expected=json.loads((ROOT/'export/hardware_identity.json').read_text())['model_id_hex']
        if f'{model_id:016x}'!=expected:raise ValueError('FPGA model does not match this integer export')
        return {'port':self.port,'baud':BAUD,'model_id_hex':expected,'input':[96,96,3]}

    def classify(self,image):
        payload=struct.pack('<HBB',96,1,0)+preprocess(image).tobytes()
        start=time.perf_counter();kind,data=self.request(0x10,payload);elapsed=time.perf_counter()-start
        if kind!=0x91 or len(data)!=16:raise ValueError('Invalid classification response')
        label,status,confidence,*remaining=struct.unpack('<BBH4hI',data)
        if status or label>=4:raise ValueError('Invalid class/status returned by FPGA')
        return {'class_id':label,'name':NAMES[label],'class_score':confidence/65535,
                'confidence_u16':confidence,'logits_q8_8':remaining[:4],
                'inference_cycles':remaining[4],'inference_ms':remaining[4]/100000,
                'roundtrip_ms':elapsed*1000,'frame_id':self.frame_id,'inference_location':'Basys 3 FPGA'}


def self_test(client,output):
    from .quantize import IntegerClassifier,classify_logits
    ref=IntegerClassifier.load(ROOT/'export');rows=[]
    for i in range(3):
        metadata=json.loads((ROOT/f'export/vectors/case{i}.json').read_text())
        with Image.open(metadata['image']) as image:
            pixels=np.array(preprocess(image));q=(pixels.astype(np.int16)-128).astype(np.int8).transpose(2,0,1)
            expected=ref.run(q)[0];label,confidence=classify_logits(expected);actual=client.classify(image)
        assert actual['logits_q8_8']==expected.tolist(),(expected,actual)
        assert actual['class_id']==label and actual['confidence_u16']==confidence,(label,confidence,actual)
        rows.append(actual);print('PASS hardware integer match',i,actual,flush=True)
    # Deliberate corruption must be rejected, then a clean request must recover.
    uid=0x12345678;packet=bytearray(encode(1,uid));packet[-1]^=1
    client.serial.write(packet);kind,rid,payload=client.receive()
    assert kind==0xff and rid==uid and payload[0]==1,'Bad CRC was not rejected'
    identity=client.hello()
    client.serial.write(encode(1,123)[:7]);client.serial.flush();time.sleep(.65)
    assert client.hello()['model_id_hex']==identity['model_id_hex'],'Partial-packet recovery failed'
    report={'hardware_identity':identity,'vectors':rows,'all_logits_and_scores_exact':True,
            'bad_crc_rejected':True,'partial_packet_recovery':True,'neural_inference_on_fpga':True,
            'note':'Functional integer agreement on three real validation photos, not a new hardware accuracy benchmark.'}
    Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('HARDWARE SELF TEST PASSED',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port');p.add_argument('--image');p.add_argument('--self-test',action='store_true')
    p.add_argument('--output',default=str(ROOT/'hardware_test.json'));a=p.parse_args()
    client=FPGAClient(a.port)
    try:
        print(json.dumps(client.identity),flush=True)
        if a.self_test:self_test(client,a.output)
        elif a.image:
            with Image.open(a.image) as image:print(json.dumps(client.classify(image),indent=2),flush=True)
    finally:client.close()


if __name__=='__main__':main()
