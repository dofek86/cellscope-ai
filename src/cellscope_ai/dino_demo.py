"""Local, research-only DINOv2 candidate interface with single/batch analysis."""

from __future__ import annotations

import base64
import binascii
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from .dino_predict import MODEL_PATH, DinoCandidate
from .relative_texture_review import CELL_TYPES
from .review_queue import ReviewQueue

PAGE = """<!doctype html><html lang="sl"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CellScope AI — DINOv2 kandidat</title><style>
body{font:16px system-ui;background:#f2f7f5;color:#18332e;margin:0}main{max-width:930px;margin:48px auto;padding:0 20px}
h1{font-size:42px;margin:5px 0}.tag{color:#16815e;font-weight:800;letter-spacing:.1em}
.card{background:white;padding:25px;border:1px solid #d9e8e2;border-radius:18px;margin:20px 0}
.muted{color:#536b64;line-height:1.5}input,button,select{font:inherit}input{display:block;margin:15px 0;width:100%}
input[type=checkbox]{display:inline;width:auto}input[type=text]{box-sizing:border-box;padding:10px;border:1px solid #c7d9d2;border-radius:8px}
select{display:block;width:100%;padding:10px;margin:12px 0;border:1px solid #c7d9d2;border-radius:8px}
button{background:#16815e;color:white;border:0;border-radius:9px;padding:12px 20px;cursor:pointer}
button:disabled{opacity:.6}.bad{color:#a14424}.good{color:#16815e}table{border-collapse:collapse;width:100%}
td,th{text-align:left;border-bottom:1px solid #e1ebe7;padding:10px}th{color:#536b64}
</style><main><div class="tag">RAZISKOVALNI PROTOTIP · DINOv2</div><h1>CellScope AI</h1>
<p class="muted">Ocena kakovosti svetlopoljske slike kulture. Ne ugotavlja bolezni ali vzroka zavrnitve.</p>
<p class="muted"><strong>Model je že naučen</strong> na 3.072 slikah s strokovnima oznakama dobra/slaba. Ob nalaganju uporabi naučene vzorce za napoved; ena nova slika modela sama ne nauči ničesar. Nadaljnje učenje je ločen postopek po strokovnem pregledu novih zajemnih sej.</p>
<section class="card"><h2>Analiza ene slike ali ene zajemne seje</h2>
<p class="muted">Izberi eno ali več slik <strong>iste zajemne seje</strong>. Pri več slikah model uporabi tudi povprečje ocen znotraj te seje. Pri eni sliki ta korak nima učinka.</p>
<form id="form"><input id="images" type="file" accept="image/png,image/jpeg,image/tiff,image/bmp" multiple required>
<label>Oznaka zajemne seje (obvezna pri shranjevanju ali primerjavi teksture)<input id="session" type="text" maxlength="100" placeholder="npr. laboratorij-2026-09-25-zajem-01"></label>
<label>Vrsta celic za poskusno opozorilo o teksturi<select id="cellType"><option value="">Ni navedena</option><option>A549</option><option>CACO</option><option>HPMEC</option><option>HSAEC</option><option>HUVEC</option><option>NHBE</option></select></label>
<p class="muted">Opozorilo je na voljo pri vsaj 5 slikah iste seje in vrste celic. Primerja teksturo znotraj naložene skupine. Pri 6 ali več slikah preveri, ali izpust ene druge slike spremeni opozorilo. To meri občutljivost na izbor, <strong>ne pravilnosti ocene</strong>. Signal ni potrjen na novih sejah in ne spremeni glavne ocene.</p>
<label><input id="save" type="checkbox"> Shrani te slike lokalno za poznejši pregled. Gre za nov zajem, ne za kopijo razvojnega nabora.</label>
<p class="muted">Slike se shranijo samo, če označiš to možnost. Učne in nedotaknjene ocenjevalne seje se ločijo pred vnosom strokovnih oznak. <a href="/review">Odpri čakalno vrsto za pregled</a>.</p>
<button id="submit">Analiziraj</button> <span id="status" class="muted"></span></form></section>
<section id="result" class="card" hidden><h2>Rezultati</h2><p id="mode" class="muted"></p>
<table><thead><tr><th>Slika</th><th>Ocena</th><th>P(dobra)</th><th>Dodatni pregled</th></tr></thead><tbody id="rows"></tbody></table></section>
<p class="muted">Raziskovalna ocena na izločenih sejah istih 3.072 razvojnih slik: 81,03 % uravnotežene točnosti z več slikami iste seje in 80,23 % brez povprečenja po seji. Arhitekturo sva izbirala na teh podatkih, zato sta oceni raziskovalni. Model v tem prikazu je ponovno naučen na celotnem naboru; njegov rezultat na novih sejah še ni izmerjen. Posamezne verjetnosti niso potrjena biološka presoja.</p>
<script>
const form=document.getElementById('form'), files=document.getElementById('images'),status=document.getElementById('status'),button=document.getElementById('submit');
function textureStatus(item){if(item.texture_review_flag===null||item.texture_review_flag===undefined)return 'Ni ocenjeno';
const stability=item.texture_review_stability;
if(item.texture_review_flag===true){if(typeof stability!=='number')return 'Poskusno opozorilo · stabilnost ni ocenjena';
return stability<.8?'Poskusno opozorilo · občutljivo na izbor slik':'Poskusno opozorilo · obstalo ob večini izpustov'}
return typeof stability==='number'&&stability<.8?'Ni opozorila · občutljivo na izbor slik':'Ni dodatnega signala'}
form.addEventListener('submit',async event=>{event.preventDefault();const chosen=[...files.files];
if(!chosen.length)return;if(chosen.length>20){status.textContent='Največ 20 slik naenkrat.';return}
if(chosen.some(file=>file.size>14*1024*1024)){status.textContent='Posamezna slika presega 14 MB.';return}
const save=document.getElementById('save').checked,session=document.getElementById('session').value.trim(),cellType=document.getElementById('cellType').value;
if(save&&!session){status.textContent='Za shranjevanje vnesi oznako zajemne seje.';return}
if(cellType&&chosen.length>=5&&!session){status.textContent='Za primerjavo teksture vnesi oznako iste zajemne seje.';return}
button.disabled=true;status.textContent='Obdelujem slike …';try{
const images=await Promise.all(chosen.map(file=>new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve({name:file.name,data:reader.result});reader.onerror=reject;reader.readAsDataURL(file)})));
const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({images,session_id:session,cell_type:cellType,save_for_review:save})});
const data=await response.json();if(!response.ok)throw Error(data.error||'Analiza ni uspela');
document.getElementById('mode').textContent=images.length===1?'Ena slika: povprečenje po seji nima učinka.':'Skupina '+images.length+' slik: povprečenje znotraj ene zajemne seje.';
const rows=document.getElementById('rows');rows.replaceChildren();for(const item of data.results){const tr=document.createElement('tr');
for(const value of [item.image,item.decision==='good'?'Sprejemljiva':'Za pregled / slabša',(100*item.probability_good).toFixed(1)+' %',textureStatus(item)]){const td=document.createElement('td');td.textContent=value;tr.append(td)}
tr.children[1].className=item.decision;rows.append(tr)}document.getElementById('result').hidden=false;status.textContent=save?'Končano; slike so shranjene za pregled.':'Končano.';
}catch(error){status.textContent=error.message}finally{button.disabled=false}});
</script></main></html>"""

REVIEW_PAGE = """<!doctype html><html lang="sl"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CellScope AI — pregled slik</title><style>
body{font:16px system-ui;background:#f2f7f5;color:#18332e;margin:0}main{max-width:1100px;margin:35px auto;padding:0 20px}
.card{background:white;padding:18px;border:1px solid #d9e8e2;border-radius:15px;margin:15px 0}
.row{display:flex;gap:18px;flex-wrap:wrap}.row img{max-width:360px;width:100%;object-fit:contain;background:#ddd}
.details{flex:1;min-width:260px}.muted{color:#536b64}button,input,select{font:inherit;padding:8px;margin:5px}
button{background:#16815e;color:white;border:0;border-radius:7px;cursor:pointer}
input[type=checkbox]{padding:0}label{display:block;margin:8px 0}
</style><main><a href="/">← Analiza</a><h1>Slike za strokovni pregled</h1>
<p class="muted">Ocena modela in prednost pregleda nista potrjen razlog za oznako. Za učenje štejejo le oznake, ki jih strokovnjak izrecno potrdi. Razdelitev po zajemnih sejah je določena ob shranjevanju slike.</p>
<div id="counts" class="card"></div><div id="items"></div>
<script>
const root=document.getElementById('items');
async function refresh(){const res=await fetch('/api/review');const data=await res.json();
document.getElementById('counts').textContent='Potrjene oznake po razdelitvi: '+JSON.stringify(data.training_status);
root.replaceChildren();if(!data.items.length){root.textContent='Za zdaj ni shranjenih slik.';return}
for(const item of data.items){const card=document.createElement('section');card.className='card row';
const img=document.createElement('img');img.src='/api/review/image/'+item.image_id;img.alt=item.original_name;card.append(img);
const detail=document.createElement('div');detail.className='details';
const title=document.createElement('h3');title.textContent=item.original_name;detail.append(title);
const info=document.createElement('p');info.textContent='Seja: '+item.session_id+' · '+(item.split==='locked_evaluation'?'zaklenjeno ocenjevanje':'učenje kandidata');detail.append(info);
const state=document.createElement('p');state.className='muted';state.textContent=item.label?'Oznaka: '+item.label+(item.expert_confirmed?' (potrjena)':' (začasna)'):'Brez potrjene oznake';detail.append(state);
if(item.texture_review_flag===1){const signal=document.createElement('p');signal.className='bad';signal.textContent='Poskusno opozorilo teksture: dodatni pregled'+(typeof item.texture_review_stability==='number'?(item.texture_review_stability<.8?' · občutljivo na izbor slik':' · obstalo ob večini izpustov'):' · stabilnost ni ocenjena');detail.append(signal)}
const reviewer=document.createElement('input');reviewer.placeholder='Ime pregledovalca';reviewer.value=item.reviewer||'';detail.append(reviewer);
const select=document.createElement('select');for(const value of ['unsure','good','bad']){const opt=document.createElement('option');opt.value=value;opt.textContent=value==='unsure'?'Neodločeno':value==='good'?'Dobra':'Slaba';select.append(opt)}select.value=item.label||'unsure';detail.append(select);
const expert=document.createElement('input');expert.type='checkbox';expert.checked=Boolean(item.expert_confirmed);
const label=document.createElement('label');label.append(expert,document.createTextNode(' Strokovno potrjena oznaka za učenje/ocenjevanje'));detail.append(label);
const note=document.createElement('input');note.placeholder='Opomba (neobvezno)';note.value=item.note||'';detail.append(note);
const button=document.createElement('button');button.textContent='Shrani oznako';button.onclick=async()=>{const reply=await fetch('/api/review/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_id:item.image_id,label:select.value,reviewer:reviewer.value,expert_confirmed:expert.checked,note:note.value})});const result=await reply.json();if(reply.ok)refresh();else alert(result.error||'Napaka')};detail.append(button);card.append(detail);root.append(card)} }
refresh();
</script></main></html>"""

MODEL: DinoCandidate | None = None
QUEUE: ReviewQueue | None = None
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status: int, payload: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/":
            self.reply(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/review":
            self.reply(200, REVIEW_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/api/review":
            payload = {"items": QUEUE.list_items(),
                       "training_status": QUEUE.training_status()}
            self.reply(200, json.dumps(payload).encode(), "application/json")
            return
        if self.path.startswith("/api/review/image/"):
            image_id = self.path.removeprefix("/api/review/image/")
            if len(image_id) == 64 and all(char in "0123456789abcdef" for char in image_id):
                path = QUEUE.directory / "images" / f"{image_id}.jpg"
                if path.exists():
                    self.reply(200, path.read_bytes(), "image/jpeg")
                    return
        self.reply(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/api/predict", "/api/review/label"):
            self.reply(404, b"Not found", "text/plain")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= (110 * 1024 * 1024 if self.path == "/api/predict" else 8192):
                raise ValueError("Zahteva je prevelika.")
            request = json.loads(self.rfile.read(size))
            if self.path == "/api/review/label":
                QUEUE.set_label(request["image_id"], request["label"],
                                request["reviewer"], request["expert_confirmed"],
                                request.get("note", ""))
                self.reply(200, b'{"ok":true}', "application/json")
                return
            images = request["images"]
            if not isinstance(images, list) or not 1 <= len(images) <= 20:
                raise ValueError("Izberi od 1 do 20 slik.")
            save_for_review = request.get("save_for_review", False)
            session_id = request.get("session_id", "")
            cell_type = request.get("cell_type", "")
            if (not isinstance(save_for_review, bool) or not isinstance(session_id, str)
                    or not isinstance(cell_type, str)):
                raise TypeError("Neveljavna seja ali izbira shranjevanja.")
            session_id = session_id.strip()
            cell_type = cell_type.strip()
            if cell_type and cell_type not in CELL_TYPES:
                raise ValueError("Neznana vrsta celic.")
            if save_for_review and not session_id:
                raise ValueError("Za shranjevanje vnesi oznako zajemne seje.")
            if cell_type and len(images) >= 5 and not session_id:
                raise ValueError("Za primerjavo teksture vnesi oznako zajemne seje.")
            with tempfile.TemporaryDirectory(prefix="cellscope-dino-") as directory:
                paths = []
                names = []
                raw = []
                for index, entry in enumerate(images):
                    data = base64.b64decode(entry["data"].split(",", 1)[1], validate=True)
                    if not data or len(data) > 14 * 1024 * 1024:
                        raise ValueError("Posamezna slika presega 14 MB.")
                    name = Path(entry["name"]).name
                    path = Path(directory) / f"{index:02d}.png"
                    path.write_bytes(data)
                    with Image.open(path) as image:
                        image.verify()
                    paths.append(path)
                    names.append(name)
                    raw.append(data)
                with LOCK:
                    groups = [session_id or "uploaded_session"] * len(paths)
                    results = (MODEL.predict_paths(paths, groups, cell_type=cell_type)
                               if cell_type else MODEL.predict_paths(paths, groups))
                for item, name in zip(results, names, strict=True):
                    item["image"] = name
                if save_for_review:
                    for item, name, data in zip(results, names, raw, strict=True):
                        item["review"] = QUEUE.save(data, name, session_id, item)
            self.reply(200, json.dumps({"results": results}).encode(), "application/json")
        except (ValueError, KeyError, IndexError, TypeError, OSError, binascii.Error) as exc:
            self.reply(400, json.dumps({"error": str(exc)}).encode(), "application/json")


def main():
    global MODEL, QUEUE
    if not MODEL_PATH.exists():
        raise RuntimeError("Run scripts/package_dino_candidate.py first")
    MODEL = DinoCandidate()
    QUEUE = ReviewQueue()
    server = ThreadingHTTPServer(("127.0.0.1", 7862), Handler)
    print("CellScope AI DINOv2: http://127.0.0.1:7862/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
