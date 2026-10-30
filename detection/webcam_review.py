"""Grid review of DRAFT boxes for the owner's webcam frames (local only).

Opens http://127.0.0.1:8766 . Each page shows 24 frames with their draft box.
Everything on a page is ACCEPTED when you press "Save page" unless you click a
frame to REJECT it (red) or press its pencil to redraw the box. Nothing becomes
training data until it has been saved from this page. Decisions are written to
detection/data/webcam/<session>/reviews.jsonl (drafts are never modified).
"""
import argparse
import json
import secrets
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
NAMES = {0: "Rubik's Cube", 1: "Pen", 2: "Book", 3: "Computer Mouse", -1: "No object"}

PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><title>Webcam box review</title>
<style>
body{font:15px system-ui;background:#111820;color:#e8eef3;margin:0;padding:14px}
h1{font-size:20px;margin:0 0 6px}.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0}
button{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid #567;background:#24323f;color:#e8eef3;cursor:pointer}
button.primary{background:#1f7a4d;border-color:#2c9b65}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}
.tile{position:relative;border:3px solid #2c9b65;border-radius:6px;overflow:hidden;cursor:pointer;background:#000}
.tile.rejected{border-color:#d9443b}.tile.rejected canvas{opacity:.35}.tile.edited{border-color:#3b8bd9}
.tile canvas{width:100%;display:block}.cap{position:absolute;left:0;right:0;bottom:0;background:#000a;padding:3px 6px;font-size:12px;display:flex;justify-content:space-between}
.pen{position:absolute;top:6px;right:6px;padding:3px 8px;font-size:13px}.tag{position:absolute;top:6px;left:6px;background:#d9443b;padding:2px 6px;font-size:12px;border-radius:4px;display:none}
.tile.rejected .tag{display:block}#modal{position:fixed;inset:0;background:#000c;display:none;align-items:center;justify-content:center;flex-direction:column}
#big{max-width:94vw;max-height:80vh;cursor:crosshair;background:#333}.help{color:#9fb3c4;font-size:13px}
</style></head><body>
<h1>Webcam box review</h1>
<div class="help">Green = keep this box. <b>Click a frame</b> to reject it (red) if the box is wrong and you don't want to fix it. Press <b>&#9998;</b> to redraw a box. Empty-scene frames are kept as "No object". Then press <b>Save page</b>. Rejected frames are simply not used.</div>
<div class="bar"><button onclick="go(-1)">&larr; Previous</button><button class="primary" onclick="savePage()">Save page &amp; next &rarr; (Enter)</button>
<span id="progress"></span><span id="msg"></span></div>
<div class="grid" id="grid"></div>
<div id="modal"><canvas id="big"></canvas><div class="bar"><button class="primary" onclick="closeEdit(true)">Use this box</button><button onclick="closeEdit(false)">Cancel</button><span class="help">Drag a tight box around the whole visible object.</span></div></div>
<script>
const token='__TOKEN__',PER=24;let items=[],page=0,state={},editing=null,start=null,ebox=null,eimg=new Image();
const NAMES={0:"Rubik's Cube",1:"Pen",2:"Book",3:"Computer Mouse","-1":"No object"};
function drawBox(ctx,w,h,b,color){if(!b)return;ctx.strokeStyle=color;ctx.lineWidth=Math.max(2,w/160);ctx.strokeRect(b[0]*w,b[1]*h,(b[2]-b[0])*w,(b[3]-b[1])*h)}
function tile(it){const d=document.createElement('div');d.className='tile';const c=document.createElement('canvas');c.width=320;c.height=180;d.appendChild(c);
 const tag=document.createElement('div');tag.className='tag';tag.textContent='REJECTED';d.appendChild(tag);
 const cap=document.createElement('div');cap.className='cap';cap.innerHTML=`<span>${NAMES[it.label]}</span><span>${it.split} #${it.frame_index}</span>`;d.appendChild(cap);
 if(it.present){const p=document.createElement('button');p.className='pen';p.innerHTML='&#9998;';p.onclick=e=>{e.stopPropagation();openEdit(it)};d.appendChild(p)}
 const im=new Image();im.onload=()=>{const x=c.getContext('2d');x.drawImage(im,0,0,320,180);drawBox(x,320,180,state[it.id].bbox,state[it.id].edited?'#3b8bd9':'#39ff8a')};im.src='/frame/'+encodeURIComponent(it.id);
 d.onclick=()=>{const s=state[it.id];s.status=s.status==='rejected'?'accepted':'rejected';if(s.status==='accepted'&&it.present&&!s.bbox)s.status='rejected';render()};
 const s=state[it.id];d.classList.toggle('rejected',s.status==='rejected');d.classList.toggle('edited',!!s.edited&&s.status!=='rejected');return d}
function render(){const g=document.getElementById('grid');g.innerHTML='';const pages=Math.ceil(items.length/PER);page=Math.max(0,Math.min(page,pages-1));
 items.slice(page*PER,page*PER+PER).forEach(it=>g.appendChild(tile(it)));const done=items.filter(i=>i.reviewed).length;
 document.getElementById('progress').textContent=`Page ${page+1}/${pages} | reviewed ${done}/${items.length}`}
function go(d){page+=d;render();window.scrollTo(0,0)}
async function savePage(){const rows=items.slice(page*PER,page*PER+PER).map(it=>({id:it.id,status:state[it.id].status,bbox:state[it.id].status==='accepted'&&it.present?state[it.id].bbox:null,edited:!!state[it.id].edited}));
 const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify(rows)});
 const m=document.getElementById('msg');if(!r.ok){m.textContent='Save failed: '+await r.text();return}
 rows.forEach(x=>{const it=items.find(i=>i.id===x.id);it.reviewed=true});m.textContent='Saved.';
 if((page+1)*PER<items.length)go(1);else{render();m.textContent='All pages saved. You can close this tab.'}}
function openEdit(it){editing=it;ebox=state[it.id].bbox;const c=document.getElementById('big');eimg.onload=()=>{c.width=eimg.width*1.5;c.height=eimg.height*1.5;drawEdit()};eimg.src='/frame/'+encodeURIComponent(it.id);document.getElementById('modal').style.display='flex'}
function drawEdit(){const c=document.getElementById('big'),x=c.getContext('2d');x.drawImage(eimg,0,0,c.width,c.height);drawBox(x,c.width,c.height,ebox,'#3b8bd9')}
function pos(e){const c=document.getElementById('big'),r=c.getBoundingClientRect();return[Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]}
const big=document.getElementById('big');big.onpointerdown=e=>{start=pos(e);big.setPointerCapture(e.pointerId)};
big.onpointermove=e=>{if(!start)return;const p=pos(e);ebox=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])];drawEdit()};big.onpointerup=()=>start=null;
function closeEdit(use){if(use&&ebox&&ebox[2]-ebox[0]>.01&&ebox[3]-ebox[1]>.01){Object.assign(state[editing.id],{bbox:ebox.map(v=>Math.round(v*1e4)/1e4),status:'accepted',edited:true})}document.getElementById('modal').style.display='none';editing=null;render()}
document.addEventListener('keydown',e=>{if(e.key==='Enter'&&!editing)savePage();if(e.key==='Escape'&&editing)closeEdit(false)});
fetch('/items').then(r=>r.json()).then(x=>{items=x;x.forEach(it=>state[it.id]={status:it.status||(it.present&&!it.bbox?'rejected':'accepted'),bbox:it.bbox,edited:it.edited});
 const first=items.findIndex(i=>!i.reviewed);page=first<0?0:Math.floor(first/PER);render()});
</script></body></html>'''


FIX_BANNER = ('<h1>Fix pass: boxes that are probably wrong</h1><div style="background:#5a3b10;padding:10px;border-radius:6px;margin:6px 0">'
              '<b>Look carefully at each frame.</b> These boxes were flagged as likely wrong (box jumps between frames, '
              'or the book). Keep a box green <b>only if it tightly covers the WHOLE object</b>. Otherwise press &#9998; '
              'and drag a new box, or click the frame to reject it. Then press Enter.</div>')


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if Path(path).exists() else []


def unpack_frames(folder):
    """Frames are delivered as frames.zip; unpack once (never overwrites existing frames)."""
    archive = folder / "frames.zip"
    if archive.exists():
        import zipfile
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                target = (folder / name).resolve()
                if not str(target).startswith(str((folder / "frames").resolve())) or not name.endswith(".jpg"):
                    raise ValueError(f"Unexpected archive entry {name}")
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(z.read(name))


def serve(folder, port, open_browser=True, fix=False):
    unpack_frames(folder)
    drafts = read_jsonl(folder / "drafts.jsonl"); byid = {r["id"]: r for r in drafts}
    if len(byid) != len(drafts):
        raise ValueError("Duplicate draft IDs")
    if fix:  # second pass: only frames flagged as likely-wrong boxes
        wanted = set(json.loads((folder / "fixlist.json").read_text(encoding="utf-8")))
        drafts = [r for r in drafts if r["id"] in wanted]
    out = folder / "reviews.jsonl"; lock = threading.Lock(); token = secrets.token_urlsafe(24)

    def items():
        reviews = {r["id"]: r for r in read_jsonl(out)}
        rows = []
        for r in drafts:
            row = {k: r[k] for k in ("id", "label", "present", "bbox", "split", "frame_index")}
            if r["id"] in reviews:
                rv = reviews[r["id"]]
                row.update({"bbox": rv["bbox"] if rv["status"] == "accepted" else r["bbox"], "status": rv["status"],
                            "edited": rv.get("edited", False), "reviewed": rv.get("fix_pass", False) if fix else True})
            rows.append(row)
        return rows

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, data, kind="application/json", status=200):
            if isinstance(data, str):
                data = data.encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", kind); self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                page = PAGE.replace("__TOKEN__", token)
                if fix:
                    page = page.replace("<h1>Webcam box review</h1>", FIX_BANNER).replace("minmax(300px,1fr)", "minmax(440px,1fr)")
                return self.send(page, "text/html; charset=utf-8")
            if path == "/items":
                return self.send(json.dumps(items()))
            if path.startswith("/frame/"):
                row = byid.get(unquote(path[7:]))
                if row:
                    return self.send((folder / row["file"]).read_bytes(), "image/jpeg")
            self.send("Not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/save" or self.headers.get("X-Review-Token") != token:
                return self.send("Invalid request", "text/plain", 403)
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n < 2 or n > 200000:
                    raise ValueError("Invalid size")
                rows = json.loads(self.rfile.read(n)); now = datetime.now(timezone.utc).isoformat(); clean = []
                for r in rows:
                    d = byid[r["id"]]
                    if r["status"] not in ("accepted", "rejected"):
                        raise ValueError("Invalid status")
                    b = r.get("bbox")
                    if r["status"] == "accepted" and d["present"]:
                        if not (isinstance(b, list) and len(b) == 4 and 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
                            raise ValueError(f"Accepted positive {r['id']} needs a valid box")
                    else:
                        b = None
                    clean.append({"id": r["id"], "status": r["status"], "bbox": b, "edited": bool(r.get("edited")),
                                  "label": d["label"], "present": d["present"], "reviewer": "owner", "reviewed_utc": now,
                                  **({"fix_pass": True} if fix else {})})
                with lock:
                    existing = {x["id"]: x for x in read_jsonl(out)}
                    existing.update({x["id"]: x for x in clean})
                    tmp = out.with_suffix(".tmp")
                    tmp.write_text("".join(json.dumps(x) + "\n" for x in existing.values()), encoding="utf-8"); tmp.replace(out)
                self.send(json.dumps({"saved": len(clean)}))
            except Exception as error:
                self.send(str(error), "text/plain", 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Webcam box review running at {url}  (close this window to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", default="s1"); p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--fix", action="store_true", help="Second pass over frames listed in fixlist.json")
    a = p.parse_args()
    serve(ROOT / "data" / "webcam" / a.session, a.port, not a.no_browser, a.fix)
