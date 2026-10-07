"""Apply the agent's recorded source-image legend audit to the derived run only.

The audit is specific to this immutable 2401 document. It is neither training
truth nor an automatic extractor rule. Original r-4344 remains untouched.
"""
import json
from copy import deepcopy
from pathlib import Path

from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.review.detection import DetectionReviewStore
from diagex.vision.legend_tables import _semantics

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'output/2401-delivery/audit'
RUN=ROOT/'runs/2401/2026-09-09_audited_r-4344-derived'


def main():
    store=DetectionReviewStore(RUN)
    if (AUDIT/'legend-audit.json').exists():
        raise SystemExit('Legend audit already recorded; use revisioned review edits for further changes')
    inventory=json.loads((AUDIT/'inventory.json').read_text())
    checked_refs=[str(AUDIT/f'legends-{i:02d}.png') for i in range(1,17)]
    log=[]
    def apply(action, **kwargs):
        note=kwargs.pop('note','Source image and complete printed definition inspected by agent')
        result=store.apply({'action':action,'revision':store.read()['revision'],'rater':'Codex source audit',
                           'actor_type':'agent','evidence_refs':kwargs.pop('evidence_refs',checked_refs), 'note':note,**kwargs})
        log.append({'action':action,'item_id':kwargs.get('id'),'note':note})
        return result
    def entry(i):
        return deepcopy(next(r['entry'] for r in store.read()['legends'] if r['id']==f'legend-{i}'))
    def save(i, e, note, status='confirmed'):
        e['attributes'].update(review_origin='agent',row_status='accept' if status=='confirmed' else 'uncertain' if status=='pending' else 'reject')
        apply('save_legend',id=f'legend-{i}',entry=e,status=status,note=note)
    # The original table reading is retained; repair interpretation using the
    # printed description instead of letters inside qualifiers or valve mentions.
    for i in range(156):
        e=entry(i); attrs=e['attributes']
        kind,cls,semantics=_semantics(e['label'],attrs.get('raw_description',e.get('description','')),attrs.get('abbreviation_section','general'))
        before=(e['kind'],e['symbol_class'],{k:attrs.get(k) for k in ('instrument_function','measured_variable','valve_type')})
        e.update(kind=kind,symbol_class=cls)
        for key in ('instrument_function','measured_variable','valve_type'):
            attrs.pop(key,None)
        attrs.update(semantics)
        after=(kind,cls,{k:attrs.get(k) for k in ('instrument_function','measured_variable','valve_type')})
        if before!=after:
            save(i,e,'Printed abbreviation meaning overrides substring/code-letter classification')
    # Line and installation conventions must not teach the OPC classifier that
    # every piping mark is an off-page connector.
    line_types={169:('boundary_line',None),175:('boundary_line',None),178:('slope_arrow',None),
                190:('line_crossing',None),195:('piping_boundary',None),199:('insulation',None),203:('insulation',None),
                209:('personnel_protection',None),212:('heat_tracing',None),231:('process_line','process'),
                235:('capillary','instrument_capillary'),237:('pneumatic_signal','signal_pneumatic'),240:('pneumatic_signal','signal_pneumatic'),
                242:('hydraulic_signal_line',None),245:('internal_system_link',None)}
    for i,(cls,line_type) in line_types.items():
        e=entry(i);e.update(kind='line',symbol_class=cls)
        e['attributes'].update(symbol_role='line_convention',candidate_shapes='[]')
        if line_type:e['attributes']['line_type']=line_type
        save(i,e,'Source defines a line/installation convention, not a physical continuation endpoint')
    for i in [172,177,183,185,188,192,196,197,201,204,207,208,213,305]:
        e=entry(i);e['kind']='other';e['attributes'].update(symbol_role='piping_component',candidate_shapes='[]')
        save(i,e,'Source defines a piping fitting/component; keep separate from valves and OPC endpoints')
    # Numbering examples contain instrument-looking circles but are not distinct
    # symbol definitions. Their page remains available as tag-format context.
    for i in [270,271]:
        save(i,entry(i),'Instrument numbering example, not a graphical definition',status='rejected')
    e=entry(214)
    save(214,e,'Printed caption says throttle valve, but the central glyph is absent in the source PDF. Definition quarantined pending clarification.',status='pending')
    # Repair the mixed primary/secondary line row into source-bounded children.
    template=entry(156)
    children=[]
    for label,y,cls in [('主工艺管线',367,'process_line'),('次工艺管线',540,'secondary_process_line')]:
        e=deepcopy(template)
        e.update(label=label,kind='line',symbol_class=cls,source_bbox={'x':410,'y':y,'w':240,'h':42},
                 source_label_bbox={'x':790,'y':y-2,'w':180,'h':38},source_row_id=None,image_b64=None,
                 description=label+'; individually inspected source sample')
        e['attributes']={'line_type':'process','symbol_role':'line_convention','row_status':'accept','review_origin':'agent'}
        children.append(e)
    apply('split_legend',id='legend-156',entries=children,note='Split title-border-contaminated grouping into primary and secondary process-line samples',evidence_refs=[str(AUDIT/'legend-top-left.png')])
    save(163,entry(163),'Duplicate secondary-line definition superseded by the source-bounded split child',status='rejected')
    # Restore two definitions previously lost to rejection/grouping.
    for label,kind,cls,box,lb,attrs in [
        ('电信号线','line','electric_signal',{'x':410,'y':351,'w':220,'h':48},{'x':681,'y':357,'w':140,'h':35},{'line_type':'signal_electric','symbol_role':'line_convention'}),
        ('薄膜执行机构','equipment','actuator',{'x':495,'y':1637,'w':44,'h':66},{'x':681,'y':1659,'w':209,'h':35},{'equipment_class':'actuator','symbol_role':'actuator','actuation':'pneumatic','candidate_shapes':'["connected_frame"]'}),
    ]:
        e={'label':label,'kind':kind,'symbol_class':cls,'source':'legend_extracted','source_page_index':2,
           'source_bbox':box,'source_label_bbox':lb,'description':label,'attributes':{**attrs,'row_status':'accept','review_origin':'agent'}}
        apply('save_legend',entry=e,status='confirmed',note='Restored definition after direct source inspection',evidence_refs=[str(AUDIT/'legend-line-types.png'),str(AUDIT/'legend-actuator.png')])
    # Specific printed valve mechanisms and vessel orientation remain explicit.
    for i,cls,attrs in [(174,'globe_valve',{'valve_type':'globe'}),(206,'horizontal_vessel',{'equipment_class':'vessel'}),
                         (250,'control_valve',{'valve_type':'globe','control_service':'true'}),
                         (253,'control_valve',{'valve_type':'ball','control_service':'true'}),
                         (256,'control_valve',{'valve_type':'butterfly','control_service':'true'}),
                         (274,'three_way_valve',{'valve_type':'three_way','actuation':'solenoid'})]:
        e=entry(i);e['symbol_class']=cls;e['attributes'].update(attrs);save(i,e,'Subtype/orientation is explicit in the printed definition and source glyph')
    clean=[r['id'] for r in store.read()['legends'] if r['status']=='pending' and r['id']!='legend-214']
    apply('confirm_legends',ids=clean,note='Agent inspected all 308 source entries on contact sheets; remaining built-ins retained as separately attributed defaults')
    atomic_write_json(AUDIT/'legend-audit.json',{'reviewer':'Codex source audit','actor_type':'agent','review_revision':store.read()['revision'],
        'source_images':checked_refs,'source_entries_inspected':308,'builtin_entries':25,'actions':log,
        'pending':['legend-214'],'excluded_source_rows':'Title/frame fragments, duplicate metadata and tag-numbering examples; original native-row inventory retained',
        'human_approved':False})
    print(json.dumps({'revision':store.read()['revision'],'legend_counts':{status:sum(r['status']==status for r in store.read()['legends']) for status in ['confirmed','pending','rejected']}}))


if __name__=='__main__':main()
