"""Prepare immutable-source contact sheets and a separate editable review run."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.vision.evidence import PageEvidence
from diagex.vision.legend_rows import native_legend_rows, union_boxes


def main():
    root = Path(__file__).resolve().parents[1]
    original = next((root / "runs/2401").glob("*r-4344"))
    run = root / "runs/2401/2026-09-09_audited_r-4344-derived"
    out = root / "output/2401-delivery/audit"
    out.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    for name in ("detection.json", "workbench.json"):
        target = run / name
        if not target.exists():
            shutil.copy2(original / name, target)
    if not (run / "evidence").exists():
        shutil.copytree(original / "evidence", run / "evidence")
    meta = json.loads((original / "workbench.json").read_text())
    bundle = json.loads((original / "detection.json").read_text())
    source = Path(meta["source_path"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == bundle["source_sha256"]
    atomic_write_json(out / "provenance.json", {
        "original_run": str(original), "review_run": str(run), "source": str(source),
        "source_sha256": bundle["source_sha256"],
        "detection_sha256": hashlib.sha256((original / "detection.json").read_bytes()).hexdigest(),
        "original_manifest": json.loads((original / "checkpoints/manifest.json").read_text()),
        "original_settings": meta["settings"],
    })
    font = ImageFont.truetype('/System/Library/Fonts/STHeiti Light.ttc', 15)
    small = ImageFont.truetype('/System/Library/Fonts/STHeiti Light.ttc', 12)
    def contact(records, name, images):
        for start in range(0, len(records), 20):
            canvas = Image.new('RGB', (1800, 1250), 'white')
            draw = ImageDraw.Draw(canvas)
            for j, record in enumerate(records[start:start+20]):
                x, y = (j % 4)*450, (j//4)*250
                b = record['bbox']; page = images[record['page_index']]
                pad = record.get('padding', 70)
                crop = page.crop((max(0,b['x']-pad), max(0,b['y']-pad), min(page.width,b['x']+b['w']+pad), min(page.height,b['y']+b['h']+pad)))
                crop.thumbnail((438, 195))
                canvas.paste(crop, (x+6, y+40))
                draw.rectangle((x, y, x+449, y+249), outline='#aaaaaa')
                draw.text((x+6,y+3), record['title'][:50],font=font,fill='black')
                draw.text((x+6,y+23), record.get('subtitle','')[:64],font=small,fill='#555555')
            canvas.save(out/f'{name}-{start//20+1:02d}.png')
    images = {}
    with fitz.open(source) as doc:
        for info in bundle['pages']:
            pi=info['page_index']; page=doc[pi]
            pix=page.get_pixmap(matrix=fitz.Matrix(info['width']/page.rect.width, info['height']/page.rect.height),alpha=False)
            image=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
            images[pi]=image
            image.save(out/f'page-{pi+1:02d}.png')
            overview=image.copy(); overview.thumbnail((2000,2000)); overview.save(out/f'overview-{pi+1:02d}.png')
    legend_rows=[]
    for path in sorted((original/'evidence').glob('page-*.json')):
        evidence=PageEvidence.model_validate_json(path.read_text())
        if evidence.role=='legend':
            legend_rows.extend(native_legend_rows(evidence))
    rows_by_id={r.id:r for r in legend_rows}
    legends=[]
    for i,e in enumerate(bundle['legend_pack']['entries']):
        if e.get('source_bbox') is None or e.get('source_page_index') is None:
            continue
        row=rows_by_id.get(e.get('source_row_id'))
        box=union_boxes([row.bbox,row.label_bbox]).model_dump() if row else e['source_bbox']
        legends.append({'id':f'legend-{i}','entry':e,'bbox':box,'page_index':e['source_page_index'],'padding':12,
                        'title':f'L{i}: {e["label"]}', 'subtitle':f'{e["kind"]} / {e["symbol_class"]}'})
    contact(legends,'legends',images)
    symbols=[]
    for i,d in enumerate(bundle['detections']):
        attrs=d['attributes']
        symbols.append({'id':d['id'],'detection':d,'bbox':d['bbox'],'page_index':d['page_index'],
                        'title':f'D{i} p{d["page_index"]+1}: {d["kind"]} {d["label"]}',
                        'subtitle':str({k:attrs[k] for k in ('equipment_class','valve_type','instrument_function','printed_tag') if k in attrs})})
    contact(symbols,'symbols',images)
    atomic_write_json(out/'inventory.json',{'legends':legends,'symbols':symbols,'native_legend_rows':[r.model_dump(mode='json') for r in legend_rows]})
    print(json.dumps({'review_run':str(run),'source_legends':len(legends),'detections':len(symbols),'source_pages':len(images)}))


if __name__=='__main__':
    main()
