"""Small local browser demo for the validated V6 research checkpoints."""

from __future__ import annotations

import base64
import binascii
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .v6_predict import CELL_TYPES, predict
from .visual_signals import measure_image
from .reason_hypotheses import describe_visible_outliers, suggest_rejection_factors


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "artifacts/cellscope-v6-full-dual-balanced-auc"
LOCK = threading.Lock()
MAX_REQUEST = 20 * 1024 * 1024
SIGNAL_REFERENCE = ROOT / "artifacts/v6-visual-signal-review/visual_signals.csv"
SIGNAL_LABELS = {
    "brightness_median": "Svetlost",
    "intensity_range": "Razpon sivin",
    "focus_proxy": "Mera ostrih robov",
    "illumination_spread": "Neenakomernost svetlosti",
    "texture_proxy": "Drobna tekstura",
}


def image_signal_profile(path: Path, metadata: dict, rejected: bool = False) -> dict:
    values = measure_image(path)
    reference_kind = "vse slike iz raziskovalnega nabora"
    reference_count = 0
    percentiles = {}
    rejection_factors = []
    visible_outliers = []
    if SIGNAL_REFERENCE.exists():
        reference = pd.read_csv(SIGNAL_REFERENCE, dtype={"image_id": str})
        manifest = pd.read_csv(ROOT / "data/manifest.csv", dtype={"image_id": str})
        reference = reference.merge(manifest[["image_id", "cell_type", "day", "label"]],
                                    on="image_id", validate="one_to_one")
        reason_reference = reference
        cell = metadata.get("cell_type")
        if cell in CELL_TYPES:
            same_cell = reference.loc[reference.cell_type == cell]
            reason_reference = same_cell
            day = metadata.get("day")
            if day is not None:
                same_day = same_cell.loc[same_cell.day.map(day_group) == day_group(day)]
                if len(same_day) >= 25:
                    reference = same_day
                    reference_kind = f"{cell}, podobno obdobje kulture"
                else:
                    reference = same_cell
                    reference_kind = cell
            else:
                reference = same_cell
                reference_kind = cell
        reference_count = len(reference)
        for name, value in values.items():
            if name not in reference or reference.empty:
                continue
            distribution = reference[name].to_numpy(float)
            percentiles[name] = float((np.mean(distribution < value) +
                                       0.5 * np.mean(distribution == value)) * 100)
        if rejected:
            rejection_factors = suggest_rejection_factors(values, reason_reference)
            if not rejection_factors:
                visible_outliers = describe_visible_outliers(values, reason_reference)
    return {
        "reference": reference_kind,
        "reference_count": reference_count,
        "rejection_factors": rejection_factors,
        "visible_outliers": visible_outliers,
        "signals": [{"name": name, "label": label, "value": values[name],
                     "percentile": percentiles.get(name)}
                    for name, label in SIGNAL_LABELS.items()],
    }


def day_group(day: float) -> str:
    day = float(day)
    return "0-1" if day <= 1 else "2-3" if day <= 3 else "4" if day <= 4 else "4+"

PAGE = """<!doctype html>
<html lang="sl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CellScope AI V6</title>
<style>
body{margin:0;background:#f2f7f5;color:#17302b;font-family:system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:920px;margin:50px auto;padding:0 22px}
h1{font-size:42px;margin:5px 0 8px}h2{font-size:24px;margin-top:0}
.eyebrow{color:#147b5c;font-weight:800;letter-spacing:.13em;font-size:14px}
.card{background:white;border:1px solid #d6e5df;border-radius:20px;padding:26px;box-shadow:0 8px 35px #133e2510;margin:22px 0}
.hint{color:#536b64;line-height:1.5}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}
label{display:block;font-weight:650;font-size:14px;margin-bottom:7px}input,select{box-sizing:border-box;width:100%;padding:12px;border-radius:10px;border:1px solid #bed2ca;background:white;font:inherit}
button{background:#16845f;color:white;border:0;border-radius:11px;padding:14px 22px;font:inherit;font-weight:750;cursor:pointer;margin-top:17px}
button:disabled{opacity:.55;cursor:wait}.result{display:none}.big{font-size:35px;font-weight:800;margin:7px 0}.review{background:#fff5de;border-radius:10px;padding:13px;color:#795318}
.twocol{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px;align-items:start}
img{width:100%;max-height:390px;object-fit:contain;border-radius:12px;background:#edf3f0}
.metric{font-size:25px;font-weight:750}.small{font-size:13px;color:#5a7169}
.signal-row{display:flex;justify-content:space-between;gap:16px;border-bottom:1px solid #e3eee9;padding:10px 0}
.signal-row:last-child{border-bottom:0}.signal-row strong{white-space:nowrap}
.factor{border-left:4px solid #bd7336;padding:10px 14px;margin:12px 0;background:#fff8ed;border-radius:0 10px 10px 0}
.factor strong{display:block;margin-bottom:3px}.factor small{color:#685b4c}
</style></head><body><main>
<div class="eyebrow">RAZISKOVALNI PROTOTIP · VERZIJA 6</div><h1>CellScope AI</h1>
<p class="hint">Ocena kakovosti svetlopoljske slike kulture v organu na čipu. Model ne prepoznava bolezni in ne nadomešča strokovnjaka.</p>
<section class="card"><h2>Preveri sliko</h2>
<form id="form"><div class="grid"><div><label for="image">Mikroskopska slika</label><input id="image" type="file" accept="image/*" required></div>
<div><label for="cell">Vrsta celic (če je znana)</label><select id="cell"><option value="">Ni zabeležena</option><option>A549</option><option>CACO</option><option>HPMEC</option><option>HSAEC</option><option>HUVEC</option><option>NHBE</option></select></div>
<div><label for="day">Dan kulture</label><input id="day" type="number" step="any" placeholder="npr. 3"></div>
<div><label for="density">Gostota zasejanja (celic/ml)</label><input id="density" type="number" step="any" placeholder="npr. 2600000"></div>
<div><label for="hours">Ure po zasejanju</label><input id="hours" type="number" step="any" placeholder="npr. 24"></div>
<div><label for="flow">Pretok (µl/min)</label><input id="flow" type="number" step="any" placeholder="npr. 1.66"></div></div>
<button id="submit" type="submit">Analiziraj sliko</button><p id="status" class="hint"></p></form></section>
<section id="result" class="card result"><div class="twocol"><img id="preview" alt="Naložena mikroskopska slika"><div>
<div class="eyebrow">REZULTAT V6</div><div id="decision" class="big"></div><div id="probability" class="metric"></div>
<p id="basis" class="hint"></p><p class="review">To je raziskovalna ocena kakovosti slike/kulture. Tudi samozavesten rezultat lahko zgreši; pri pomembnih odločitvah naj ga pregleda strokovnjak.</p>
<p id="range" class="small"></p></div></div></section>
<section id="reason-card" class="card result"><h2>Možni vidni razlogi za zavrnitev</h2><div id="reasons"></div>
<p class="small">Namigi so naučene povezave slikovnih mer z oznako »bad« pri primerljivih slikah. Izvirne oznake ne vsebujejo razlogov, zato to niso potrjeni vzroki ali natančna razlaga notranje odločitve modela.</p></section>
<section id="signal-card" class="card result"><h2>Kaj je mogoče izmeriti na sliki</h2>
<p id="signal-reference" class="hint"></p><div id="signals"></div>
<p class="small">Percentil pove, kje je ta mera glede na slike iz primerljivega dela raziskovalnega nabora. To ni meja med dobrim in slabim in ne pove, zakaj je strokovnjak sliko ocenil tako.</p></section>
<p class="small">Lokalni preizkus: 3.072 slik, 59 zajemnih sej; uravnotežena natančnost 72,3 %. Povprečje petih modelov na povsem novih laboratorijskih sejah še ni bilo potrjeno.</p>
</main><script>
const form=document.getElementById('form'), file=document.getElementById('image'), button=document.getElementById('submit'), status=document.getElementById('status');
form.addEventListener('submit',async e=>{e.preventDefault();const f=file.files[0];if(!f)return;if(f.size>14000000){status.textContent='Slika je prevelika (največ 14 MB).';return}
button.disabled=true;status.textContent='Analiziram sliko …';try{const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsDataURL(f)});
const number=id=>{const value=document.getElementById(id).value;return value===''?null:Number(value)};
const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:data,cell_type:document.getElementById('cell').value||null,day:number('day'),seeding_density_cells_ml:number('density'),time_after_seeding_h:number('hours'),flow_rate:number('flow')})});
const result=await response.json();if(!response.ok)throw Error(result.error||'Analiza ni uspela');
document.getElementById('preview').src=data;document.getElementById('decision').textContent=result.decision==='good'?'Sprejemljiva kakovost':'Slabša kakovost';
document.getElementById('probability').textContent=(100*result.probability_good).toFixed(1)+' % verjetnost sprejemljive kakovosti';
document.getElementById('basis').textContent=result.used_context?'Ocena iz slike in podatkov o poskusu.':'Ocena samo iz slike; vrsta celic ni podana.';
const probabilities=result.fold_predictions.map(row=>row[result.used_context?'context':'image']);
document.getElementById('range').textContent='Razpon petih modelov: '+(100*Math.min(...probabilities)).toFixed(1)+'–'+(100*Math.max(...probabilities)).toFixed(1)+' %';
document.getElementById('result').style.display='block';status.textContent='Analiza zaključena.';
const profile=result.image_signals;
const reasonCard=document.getElementById('reason-card'), reasonBox=document.getElementById('reasons');
reasonBox.replaceChildren();if(result.decision==='bad'){
const factors=profile.rejection_factors||[];
if(factors.length){for(const factor of factors){const item=document.createElement('div');item.className='factor';
const title=document.createElement('strong');title.textContent=factor.title;
const detail=document.createElement('div');detail.textContent=factor.detail;
const support=document.createElement('small');support.textContent='Pri '+factor.reference_images+' primerljivih označenih slikah je bila ocena »bad« v tem vzorcu pogostejša za '+Math.round(100*factor.risk_difference)+' odstotnih točk.';
item.append(title,detail,support);reasonBox.append(item)}}else{
const intro=document.createElement('p');intro.textContent='Za to vrsto slike nimamo dovolj močnega naučenega vzorca, ki bi pojasnil zavrnitev. Spodnje vidne posebnosti niso potrjeni razlogi za odločitev modela.';reasonBox.append(intro);
for(const observation of profile.visible_outliers||[]){const item=document.createElement('div');item.className='factor';
const title=document.createElement('strong');title.textContent=observation.title;
const detail=document.createElement('div');detail.textContent=observation.detail;
const support=document.createElement('small');support.textContent='Približno '+observation.percentile+'. percentil med '+observation.reference_images+' primerljivimi slikami.';
item.append(title,detail,support);reasonBox.append(item)}
if(!(profile.visible_outliers||[]).length){const note=document.createElement('p');note.textContent='Tudi preproste slikovne mere ne izstopajo. Model verjetno uporablja kombinacijo tekstur, ki jih še ne znamo pretvoriti v kratko razlago.';reasonBox.append(note)}}
reasonCard.style.display='block'}else{reasonCard.style.display='none'}
document.getElementById('signal-reference').textContent='Primerjalni nabor: '+profile.reference+' ('+profile.reference_count+' slik).';
document.getElementById('signals').replaceChildren(...profile.signals.map(item=>{const row=document.createElement('div');row.className='signal-row';
const label=document.createElement('span');label.textContent=item.label;const value=document.createElement('strong');
value.textContent=item.percentile===null?'izmerjeno':'približno '+Math.round(item.percentile)+'. percentil';
row.append(label,value);return row}));document.getElementById('signal-card').style.display='block';
}catch(error){status.textContent=error.message}finally{button.disabled=false}});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status: int, payload: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path != "/":
            self._reply(404, b"Not found", "text/plain")
            return
        self._reply(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/predict":
            self._reply(404, b"Not found", "text/plain")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_REQUEST:
                raise ValueError("Slika ali zahteva je prevelika.")
            request = json.loads(self.rfile.read(size))
            encoded = request["image"].split(",", 1)[1]
            raw = base64.b64decode(encoded, validate=True)
            if not raw or len(raw) > 14 * 1024 * 1024:
                raise ValueError("Slika je prevelika.")
            metadata = {name: request.get(name) for name in (
                "cell_type", "day", "seeding_density_cells_ml",
                "time_after_seeding_h", "flow_rate",
            )}
            if metadata["cell_type"] is not None and metadata["cell_type"] not in CELL_TYPES:
                raise ValueError("Neznana vrsta celic.")
            with tempfile.TemporaryDirectory(prefix="cellscope-v6-") as directory:
                image = Path(directory) / "image.png"
                image.write_bytes(raw)
                with Image.open(image) as source:
                    source.verify()
                with LOCK:
                    result = predict(image, CHECKPOINTS, metadata)
                result["image_signals"] = image_signal_profile(
                    image, metadata, rejected=result["decision"] == "bad"
                )
            self._reply(200, json.dumps(result).encode(), "application/json")
        except (ValueError, KeyError, IndexError, TypeError, OSError, binascii.Error) as exc:
            self._reply(400, json.dumps({"error": str(exc)}).encode(), "application/json")


def main():
    if len(list(CHECKPOINTS.glob("fold_*.pt"))) != 5:
        raise RuntimeError("The five V6 checkpoints are missing")
    server = ThreadingHTTPServer(("127.0.0.1", 7861), Handler)
    print("CellScope AI V6: http://127.0.0.1:7861/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
