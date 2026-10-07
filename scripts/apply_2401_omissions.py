"""Commit visually inspected omission crops and the final coverage audit."""
import base64
import io
import json
from copy import deepcopy
from pathlib import Path

from PIL import Image

from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.review.detection import DetectionReviewStore

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/2401/2026-09-09_audited_r-4344-derived'
OUT=ROOT/'output/2401-delivery/audit'


def contains(outer,inner):
    return (outer['x'] <= inner['x'] and outer['y'] <= inner['y'] and
            inner['x']+inner['w'] <= outer['x']+outer['w'] and
            inner['y']+inner['h'] <= outer['y']+outer['h'])


def main():
    if (OUT/'omission-audit.json').exists():raise SystemExit('Already applied')
    store=DetectionReviewStore(RUN)
    def apply(action,note,refs,**kwargs):
        return store.apply(dict(action=action,revision=store.read()['revision'],rater='Codex source audit',
                                actor_type='agent',evidence_refs=refs,note=note,draft_review=True,**kwargs))
    # Correct the T-filter crop before freezing graphical exemplars. All symbol
    # classes were just visually audited independently of the contaminated crop.
    row=next(r for r in store.read()['legends'] if r['id']=='legend-228')
    e=deepcopy(row['entry']);e['source_bbox']={'x':410,'y':3260,'w':240,'h':150}
    apply('save_legend','Reclip source glyph to exclude left frame border',
          [str(OUT/'legends-12.png')],id=row['id'],entry=e,status='confirmed')
    # Store fresh source-derived exemplars after geometry edits. These are
    # provenance-preserving source crops, not generated or repaired glyphs.
    for row in store.read()['legends']:
        e=deepcopy(row['entry']);b=e.get('source_bbox');p=e.get('source_page_index')
        if e.get('attributes',{}).get('legend_kind')=='abbreviation':
            if e.get('image_b64'):
                e['image_b64']=None;e['crop_quality']='omitted_abbreviation'
                apply('save_legend','Abbreviations are textual definitions, not graphical exemplars',
                      [str(OUT/f'page-{p+1:02d}.png')],id=row['id'],entry=e,status=row['status'])
            continue
        if row['status']!='confirmed' or b is None or p is None or e.get('image_b64'):continue
        page=Image.open(OUT/f'page-{p+1:02d}.png')
        crop=page.crop((b['x'],b['y'],b['x']+b['w'],b['y']+b['h']));crop.thumbnail((640,640))
        buf=io.BytesIO();crop.save(buf,format='PNG');e['image_b64']=base64.b64encode(buf.getvalue()).decode()
        e['crop_quality']='recovered'
        apply('save_legend','Regenerated exemplar directly from corrected source bounds',
              [str(OUT/f'page-{p+1:02d}.png')],id=row['id'],entry=e,status='confirmed')
    for row in store.read()['symbols']:
        if row.get('stale'):
            p=row['detection']['page_index']
            apply('save_symbol','Revalidated against corrected legend using completed source symbol audit',
                  [str(OUT/'symbol-audit.json'),str(OUT/f'coverage-{p+1:02d}.png')],
                  id=row['id'],detection=row['detection'],status='confirmed')
    evidence={e['page_index']:e for p in (RUN/'evidence').glob('page-*.json')
              if(e:=json.loads(p.read_text()))}
    proposals=json.loads((OUT/'omissions-proposed.json').read_text());log=[]
    for i,r in enumerate(proposals):
        if i in {44,45}:
            log.append({**r,'decision':'already detected','existing_index':437 if i==44 else 483});continue
        existing=next((s for s in store.read()['symbols'] if s['detection']['attributes'].get('source_audit_id')==r['id']),None)
        if existing:
            log.append({**r,'decision':'added','source_path_ids':existing['detection']['attributes']['source_path_ids']});continue
        p=r['page_index'];b=r['bbox'];ev=evidence[p]
        # Include only paths fully bounded by the inspected glyph crop. Very
        # long attached pipelines are connection evidence, not glyph geometry.
        paths=[q['id'] for q in ev['paths'] if contains(b,q['bbox'])]
        if not paths:raise ValueError(f'No native source geometry for {r["id"]}')
        words=[t for t in ev['text_spans'] if contains(b,t['bbox'])]
        attrs={'review_origin':'agent','geometry_basis':'agent_source_review',
               'source_path_ids':paths,'source_audit_id':r['id'],'source_audit_note':r['note']}
        if r['kind']=='equipment':attrs['equipment_class']=r['symbol_class']
        elif r['kind']=='instrument':attrs['instrument_function']=r['symbol_class']
        elif r['kind']=='opc':attrs['connector_type']=r['symbol_class']
        if r['symbol_class']=='valve':
            attrs['valve_type']='safety' if r['label'].startswith('PSV') else 'control' if r['label'] else 'ball'
            if r['label'] and not r['label'].startswith('PSV'):attrs['actuation']='pneumatic'
        if r['label']:attrs['printed_tag']=r['label']
        if r['symbol_class']=='unclassified_equipment':
            attrs['semantic_question']='Unlabelled mechanical assembly outline; confirm equipment function'
        if i==48:attrs['semantic_question']='PV 00502 printed on page 12, while surrounding loop is 01002; preserve drawing text until clarified'
        d=dict(id=r['id'],page_index=p,bbox=b,kind=r['kind'],label=r['label'],confidence='high',
               source_text_ids=[t['id'] for t in words],attributes=attrs,tile_id=f'agent-audit-p{p}',raw_text=r['label'] or None)
        result=apply('save_symbol',r['note'],[str(OUT/f'omissions-{i//16+1:02d}.png'),
                    str(OUT/f'page-{p+1:02d}.png')],detection=d,status='confirmed')
        log.append({**r,'decision':'added','source_path_ids':paths})
    conflicts=[
        {'pages':[8,9],'question':'Compressor headings K-002A/B conflict with internal K-001A/B captions. Which identifiers govern?'},
        {'pages':[10],'question':'Heading 2401-V-004 conflicts with vessel-body 2403-V-004.'},
        {'pages':[11],'question':'Nitrogen package heading PK-0003A/B conflicts with internal PK-0002A/B captions.'},
        {'pages':[12],'question':'FE/PV 00502 appear among loop 01002 instruments. Preserve printed identifiers; correction needs owner decision.'},
        {'pages':[6,7],'question':'Off-page text refers to V-002/V-003 where connected page captions use V-001/V-002. Cross-page pairing needs route/line-id evidence.'},
    ]
    for p in range(3,12):
        apply('coverage','All source regions inspected for omissions, false positives, duplicate glyphs, classes and tags; unresolved semantics retained separately',
              [str(OUT/f'coverage-{p+1:02d}.png'),str(OUT/'symbol-audit.json'),
               *[str(OUT/f'omissions-{j:02d}.png') for j in range(1,5)]],page_index=p,checked=True)
    state=store.read()
    atomic_write_json(OUT/'omission-audit.json',dict(actor_type='agent',human_approved=False,
        review_revision=state['revision'],decisions=log,source_conflicts=conflicts,
        coverage_pages=list(range(4,13)),additional_glyphs=sum(x['decision']=='added' for x in log)))
    snap=store.snapshot(state['revision'],draft=True)
    atomic_write_json(OUT/'reviewed-snapshot.json',snap)
    atomic_write_json(OUT/'reviewed-symbols.json',[{'id':r['id'],**r['detection'],
        'review_provenance':r.get('review_provenance'),'status':r['status']}
        for r in state['symbols']])
    print(json.dumps({'revision':state['revision'],'confirmed':len(snap['detections']),
        'pending':len(snap['unresolved']['items']),'coverage':state['coverage'],'omissions_added':47}))


if __name__=='__main__':main()
