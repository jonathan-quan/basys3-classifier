"""Local manual annotation UI. Candidate proposals are never auto-approved."""
import argparse
import html
import json
import secrets
import threading
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image,ImageOps

from . import ROOT
from .fetch_data import read_jsonl,save_jsonl

PAGE=r'''<!doctype html><html><head><meta charset="utf-8"><title>Detector annotation review</title>
<style>body{font:16px system-ui;background:#15202b;color:#ecf3f9;max-width:1100px;margin:24px auto;padding:12px}button,select,input{font:inherit;padding:9px;margin:4px}canvas{max-width:100%;background:#444;cursor:crosshair} .note{color:#ffd285}#message{min-height:25px}a{color:#87d5ff}</style></head>
<body><h1>Manual detector annotation review</h1><p class="note">Suggested boxes are candidates, not ground truth. Draw a tight box around the actual visible object. Mark background only when none of the four supported objects is visible. Skip ambiguous/multiple-object images. Nothing is approved without clicking a review button.</p>
<div id="progress"></div><div id="info"></div><canvas id="canvas"></canvas><br>
<select id="label"><option value="0">Rubik's Cube</option><option value="1">Pen</option><option value="2">Book</option><option value="3">Computer mouse</option></select>
<input id="reviewer" placeholder="Your reviewer name" value="owner" aria-label="Reviewer name">
<button onclick="save(true)">Verify positive + box</button><button onclick="save(false)">Verify no supported object</button><button onclick="skip()">Skip / uncertain</button><button onclick="back()">Previous</button>
<p>Optional real capture conditions: <select id="lighting"><option value="unknown">Lighting unknown</option><option>bright</option><option>dim</option><option>backlit</option></select><select id="background"><option value="unknown">Background unknown</option><option>plain</option><option>cluttered</option></select><select id="orientation"><option value="unknown">Orientation unknown</option><option>front</option><option>side</option><option>tilted</option></select></p>
<div id="message"></div><p>Reviews are saved locally. After reviewing enough whole groups, rerun dataset preparation. Test/validation photos retain their existing roles.</p>
<script>
let items=[],index=0,current=null,box=null,start=null,img=new Image(),token='__TOKEN__';const c=document.getElementById('canvas'),ctx=c.getContext('2d');
function draw(){ctx.drawImage(img,0,0,c.width,c.height);if(box){ctx.strokeStyle='#59ff9d';ctx.lineWidth=3;ctx.strokeRect(box[0]*c.width,box[1]*c.height,(box[2]-box[0])*c.width,(box[3]-box[1])*c.height)}}
function pos(e){const r=c.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]}
c.onpointerdown=e=>{start=pos(e);c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(!start)return;let p=pos(e);box=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])];draw()};c.onpointerup=()=>start=null;
function show(){if(!items.length){document.getElementById('message').textContent='No candidates. Run review_queue first.';return}index=(index+items.length)%items.length;current=items[index];box=current.bbox;document.getElementById('label').value=Math.max(0,current.label);document.getElementById('progress').textContent=`${index+1}/${items.length}; ${current.annotation_status}`;document.getElementById('info').textContent=`${current.id} | ${current.fixed_split||'split will be grouped'} | ${current.source}`;document.getElementById('message').textContent='';img.onload=()=>{let s=Math.min(1,900/img.width,570/img.height);c.width=Math.round(img.width*s);c.height=Math.round(img.height*s);draw()};img.src='/image/'+encodeURIComponent(current.id)}
async function save(present){let reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){alert('Enter reviewer name');return}if(present&&(!box||box[2]<=box[0]||box[3]<=box[1])){alert('Draw a valid box first');return}let data={id:current.id,present:+present,label:present?+document.getElementById('label').value:-1,bbox:present?box:null,reviewer,conditions:{lighting:document.getElementById('lighting').value,background:document.getElementById('background').value,orientation:document.getElementById('orientation').value}};let r=await fetch('/review',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify(data)});if(!r.ok){document.getElementById('message').textContent=await r.text();return}Object.assign(current,data,{annotation_status:'human_verified'});index++;show()}
function skip(){index++;show()}function back(){index--;show()}
fetch('/items').then(r=>r.json()).then(x=>{items=x;show()});
</script></body></html>'''


def serve(queue,port):
    rows=read_jsonl(queue);byid={r['id']:r for r in rows}
    if len(byid)!=len(rows):raise ValueError("Queue IDs must be unique")
    out=ROOT/'data/reviews.jsonl';lock=threading.Lock();token=secrets.token_urlsafe(24)
    if out.exists():
        for r in read_jsonl(out):
            if r['id'] in byid:byid[r['id']].update(r)
    class Handler(BaseHTTPRequestHandler):
        def send(self,data,kind='application/json',status=200):
            if isinstance(data,str):data=data.encode('utf-8')
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/':return self.send(PAGE.replace('__TOKEN__',token),'text/html; charset=utf-8')
            if path=='/items':return self.send(json.dumps(list(byid.values())))
            if path.startswith('/image/'):
                from urllib.parse import unquote
                row=byid.get(unquote(path[7:]))
                if row:
                    with Image.open(row['path']) as im:
                        im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1200,900));b=BytesIO();im.save(b,format='JPEG',quality=90)
                    return self.send(b.getvalue(),'image/jpeg')
            self.send('Not found','text/plain',404)
        def do_POST(self):
            if self.path!='/review' or self.headers.get('X-Review-Token')!=token:return self.send('Invalid request','text/plain',403)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length<1 or length>16384:raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(length));row=byid[data['id']]
                if not isinstance(data.get('reviewer'),str) or not data['reviewer'].strip():raise ValueError('Reviewer required')
                if data['present'] not in (0,1):raise ValueError('Invalid presence')
                if data['present']:
                    b=data['bbox']
                    if data['label'] not in range(4) or len(b)!=4 or not (0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1):raise ValueError('Invalid box/class')
                elif data['label']!=-1 or data['bbox'] is not None:raise ValueError('Negative must have no box')
                accepted={**row,**{k:data[k] for k in ('label','present','bbox','reviewer','conditions')},
                          'annotation_status':'human_verified','reviewed_utc':datetime.now(timezone.utc).isoformat(),
                          'annotation_source':'Manual review in local annotation tool'}
                with lock:
                    existing={r['id']:r for r in read_jsonl(out)} if out.exists() else {}
                    existing[accepted['id']]=accepted
                    tmp=out.with_suffix('.tmp');save_jsonl(tmp,list(existing.values()));tmp.replace(out)
                    byid[accepted['id']]=accepted
                self.send('{"saved":true}')
            except (ValueError,KeyError,TypeError) as e:self.send(str(e),'text/plain',400)
    print(f'Local manual review: http://127.0.0.1:{port} | {len(rows)} items | saving {out}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue',default=str(ROOT/'data/review_queue.jsonl'));p.add_argument('--port',type=int,default=8765)
    a=p.parse_args();serve(a.queue,a.port)


if __name__=='__main__':main()
