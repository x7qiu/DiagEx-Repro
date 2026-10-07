"""Apply the documented 2401 source audit, without changing extraction output.

This is a document-specific review migration, not an extractor or benchmark rule.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.review.detection import DetectionReviewStore

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/2401/2026-09-09_audited_r-4344-derived'
OUT = ROOT / 'output/2401-delivery/audit'

# All indices refer to immutable inventory.json, not mutable review ordering.
COOLERS = {24, 38, 80, 81, 203, 217, 293, 294}
RECTANGLES = {5, 55, 189, 263}
DUPLICATES = {228:227, 236:234, 237:235, 231:230, 240:238, 241:239,
              324:321, 325:322, 326:323, 330:327, 331:328, 332:329}
FITTINGS = {230, 238, 239, 327, 328, 329}
BOUNDARIES = {25, 204, 244, 496}
PRESSURE = {353, 354}
BALL = {101,125,126,131,147,148,153,165,173,174,227,234,235,311,321,322,323,
        402,403,423,437,451,452,469,483,495,522}
SOLENOID = {26,41,82,94,205,232,295,304}
TEXT_OVERRIDE = {223:'PG 00201',224:'TG 00201',229:'TG 00202',282:'PG 00301',
                 283:'TG 00301',318:'TG 00302'}
TEXT_OVERRIDE.update({i:'P' for i in [116,119,120,139,140,141,350,353,354,365+14,
    382,383,401,413,415,416,419,435,450,459,461,462,465,481]})
FUNCTIONS = {'PI':'indicator','TI':'indicator','YL':'status_light','PG':'gauge','TG':'gauge',
             'PT':'transmitter','TT':'transmitter','TE':'element','FE':'element','FT':'transmitter',
             'LG':'gauge','LT':'transmitter','LICA':'indicate_control_alarm','LIA':'indicate_alarm',
             'PIA':'indicate_alarm','PICA':'indicate_control_alarm','PIC':'indicate_control',
             'FIQ':'indicate_integrate','I':'interlock','P':'unclassified_instrument'}
VARIABLES = {'P':'pressure','T':'temperature','F':'flow','L':'level'}


def inside(a, b, pad=0):
    return (a['x']-pad <= b['x']+b['w']/2 <= a['x']+a['w']+pad and
            a['y']-pad <= b['y']+b['h']/2 <= a['y']+a['h']+pad)


def main():
    if (OUT/'symbol-audit.json').exists():
        raise SystemExit('Already applied; further corrections require a new review revision')
    store=DetectionReviewStore(RUN)
    inventory=json.loads((OUT/'inventory.json').read_text())['symbols']
    evidence={e['page_index']:e for p in (RUN/'evidence').glob('page-*.json')
              if (e:=json.loads(p.read_text()))}
    log=[]; exceptions=[]
    def apply(action, note, refs, **kwargs):
        store.apply(dict(action=action,revision=store.read()['revision'],rater='Codex source audit',
                         actor_type='agent',evidence_refs=refs,note=note,draft_review=True,**kwargs))
    for i,item in enumerate(inventory):
        d=deepcopy(item['detection']); a=d['attributes']; p=d['page_index']; b=d['bbox']
        refs=[str(OUT/f'symbols-{i//20+1:02d}.png'),str(OUT/f'coverage-{p+1:02d}.png')]
        status='confirmed'; reasons=[]
        if i in DUPLICATES:
            status='rejected'; a['duplicate_of']=inventory[DUPLICATES[i]]['id']
            reasons.append('Internal circle is part of the same source glyph, not another device')
        elif i in COOLERS:
            d['kind']='equipment';a['equipment_class']='heat_exchanger'
            a.pop('instrument_function',None);a.pop('valve_type',None)
            reasons.append('Circular zigzag exchanger with two utility connections; not a motor, pump or instrument')
        elif i in RECTANGLES:
            d['kind']='equipment';a['equipment_class']='unclassified_equipment';a.pop('valve_type',None)
            a['semantic_question']='Unlabelled rectangle in compressor mechanical/auxiliary assembly; exact equipment function is not printed'
            exceptions.append({'id':item['id'],'question':a['semantic_question'],'page_index':p,'bbox':b})
            reasons.append('Retain visible equipment geometry; do not assign a vessel or valve subtype')
        elif i in FITTINGS:
            d['kind']='equipment';a['equipment_class']='piping_component';a.pop('valve_type',None)
            a['semantic_question']='Double-circle inline component lacks a matching unambiguous definition; confirm component subtype'
            exceptions.append({'id':item['id'],'question':a['semantic_question'],'page_index':p,'bbox':b})
            reasons.append('Keep one source-bounded inline component, with unresolved subtype')
        elif i in BOUNDARIES:
            d['kind']='opc';a.pop('equipment_class',None);a.pop('valve_type',None)
            a['connector_type']='boundary';a['legend_definition']='legend-222'
            reasons.append('Filled arrow in circle matches the printed in/out boundary convention')
        elif i in PRESSURE:
            d['kind']='instrument';a.pop('equipment_class',None);a.pop('valve_type',None)
            reasons.append('Source circle contains P, not M; fix known benchmark classification error')
        elif i in {397,448}:
            d['kind']='instrument';a.pop('equipment_class',None);a.pop('valve_type',None)
            a['instrument_function']='analyzer';a['measured_variable']='nitrogen'
            reasons.append('Printed label is nitrogen analyzer (氮分析仪)')
        elif i in {404,453}:
            a['equipment_class']='vessel';a['equipment_service']='nitrogen_package_internal'
            reasons.append('Crossed internals in a vessel outline do not prove heat-exchanger service')
        elif i==102:
            a['equipment_class']='vessel';a['equipment_subtype']='vertical_buffer_vessel'
            reasons.append('Printed air buffer vessel and rounded heads, not atmospheric storage-tank glyph')
        elif i in {345,374}:
            status='pending';reasons.append('Detached circular arrow above pipeline: boundary mark or directional annotation remains unresolved')
        elif i==504:
            d['kind']='equipment';a['equipment_class']='valve'
            for k in ('instrument_function','loop_number','printed_tag','measured_variable'):a.pop(k,None)
            d['label']='';reasons.append('Filled sample cock has no instrument letters; nearby TI text belongs to a separate bubble')
        if d['kind']=='equipment' and a.get('equipment_class')=='valve':
            # Original outputs inconsistently asserted check/angle/three-way for
            # identical two-port bowties. The plain generic glyph cannot support
            # all those mechanisms; preserve a subtype only when visible.
            a['valve_type']='ball' if i in BALL else 'general'
            if i in SOLENOID:a['actuation']='solenoid'
            if i not in BALL:a['subtype_status']='not_resolved_from_glyph'
        if d['kind']=='instrument':
            spans=[t for t in evidence[p]['text_spans'] if inside(b,t['bbox'])]
            spans.sort(key=lambda t:(t['bbox']['y'],t['bbox']['x']))
            words=list(dict.fromkeys(t['text'].strip() for t in spans))
            # Retain only text inside this glyph. Prior candidate source ids can
            # point to a neighboring bubble, particularly tiny sample cocks.
            codes=[w for w in words if re.fullmatch(r'[A-Z]{1,5}',w)]
            numbers=[w for w in words if re.fullmatch(r'\d{3,6}',w)]
            tag=TEXT_OVERRIDE.get(i)
            if tag is None and len(codes)==1:
                tag=' '.join([codes[0],*numbers[:1]])
            if tag:
                d['label']=tag;d['raw_text']=tag;a['printed_tag']=tag
                d['source_text_ids']=[t['id'] for t in spans];a['source_text_ids']=d['source_text_ids']
                code=tag.split()[0];a['instrument_code']=code
                if code in FUNCTIONS:a['instrument_function']=FUNCTIONS[code]
                if code[:1] in VARIABLES:a['measured_variable']=VARIABLES[code[0]]
                if len(tag.split())>1:a['loop_number']=tag.split()[1]
                if i in TEXT_OVERRIDE:a['tag_evidence']='agent_transcription_of_vector_outlined_text'
            elif i not in {397,448}:
                exceptions.append({'id':item['id'],'page_index':p,'bbox':b,'question':'Instrument text association remains incomplete'})
        if d['kind']=='opc':
            texts=[t for t in evidence[p]['text_spans'] if inside(b,t['bbox'])]
            ref=next((t for t in texts if re.search(r'DW02[-－]\d+',t['text'])),None)
            if ref:
                d['label']=ref['text'];d['source_text_ids']=[ref['id']];a['target_drawing']=ref['text']
        a['review_origin']='agent';a['source_audit_index']=i
        if d['kind']=='equipment' and a.get('equipment_class')=='valve':
            # The QV glyphs inside the controller wiring sketch are representations
            # of devices drawn elsewhere, not extra physical process valves.
            if (p==5 and b['y']<1050) or (p==5 and 2100<b['y']<2340) or (
                p==9 and (1150<b['y']<1330 or 2900<b['y']<3100)) or (
                p==10 and b['x']<2800 and (650<b['y']<1140 or 2300<b['y']<2820)):
                a['representation_role']='controller_wiring_symbol'
        note='; '.join(reasons) or 'Source glyph inspected; native text association and conservative class checked'
        apply('save_symbol',note,refs,id=item['id'],detection=d,status=status)
        log.append({'index':i,'id':item['id'],'status':status,'note':note,'label':d['label'],
                    'class':a.get('equipment_class',d['kind']),'subtype':a.get('valve_type')})
        if i%50==0:print(f'Audited {i+1}/{len(inventory)}',flush=True)
    # Unselected native candidates and free visual proposals were each inspected
    # on targets-01/02. Blank regions, text fragments, existing valve parts and
    # overlapping proposals are rejected, with recoverable originals preserved.
    unanchored=json.loads((OUT/'unanchored-inventory.json').read_text())
    recover={0:('equipment',{'equipment_class':'valve','valve_type':'general'}),
             1:('equipment',{'equipment_class':'valve','valve_type':'general'}),
             2:('opc',{'connector_type':'boundary'}),5:('opc',{'connector_type':'continuation'}),
             6:('opc',{'connector_type':'boundary'}),
             18:('equipment',{'equipment_class':'valve','valve_type':'ball'}),
             19:('equipment',{'equipment_class':'valve','valve_type':'ball'}),
             20:('equipment',{'equipment_class':'valve','valve_type':'ball'})}
    for i,row in enumerate(unanchored):
        d=deepcopy(row['detection']);status='rejected'
        note='Source inspection: no separate new device; blank/text fragment, existing component part, or overlapping proposal'
        if i in recover:
            status='confirmed';kind,attrs=recover[i];d['kind']=kind;d['attributes'].update(attrs,review_origin='agent')
            note='Source glyph recovered from rejected native candidate; geometry and page inspected'
            if i==1:d['bbox']={'x':4080,'y':1432,'w':22,'h':26}
            if kind=='opc':d['label']='DW02-0006' if i==5 else ''
        apply('save_symbol',note,[str(OUT/f'targets-{i//20+1:02d}.png')],id=row['id'],detection=d,status=status)
    atomic_write_json(OUT/'symbol-audit.json',{'actor_type':'agent','human_approved':False,
        'source_detections_inspected':526,'unselected_candidates_and_proposals_inspected':26,
        'review_revision':store.read()['revision'],'changes':log,'semantic_exceptions':exceptions,
        'coverage_status':'Nine overview overlays inspected; omission recovery in a separate revision',
        'subtype_policy':'General valve retained where mechanism is not resolved; not counted as subtype success'})
    print('Saved source symbol review',store.read()['revision'],flush=True)


if __name__=='__main__':main()
