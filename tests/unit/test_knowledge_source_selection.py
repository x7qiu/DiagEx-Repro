"""Source combinations and source-bound attribution; no paid model calls."""
import copy
import itertools

import pytest

from diagex.knowledge import resolver
from diagex.knowledge.library import load_library, validate_match
from diagex.knowledge.sources import SOURCES
from diagex.vision.legend_context import select_legend_context
from diagex.vision.symbol_interpretation import (
    PerceivedObject,
    PerceptionBatch,
    _bind_knowledge,
    _bind_legend,
)
from diagex.web.evidence_origin import evidence_origin


@pytest.fixture
def catalog(monkeypatch):
    catalog = load_library(resolver.library_root())
    monkeypatch.setattr(resolver, 'load_library', lambda root: catalog)
    return catalog


def test_every_source_combination_filters_retrieval_and_cache(catalog):
    identities = set()
    for flags in itertools.product((False, True), repeat=3):
        selected = [identity for identity, flag in zip(SOURCES, flags, strict=True) if flag]
        snap = resolver.knowledge_snapshot('general', source_ids=selected)
        identities.add(snap['identity'])
        ids = resolver.applicable_references(snap)['reference_ids']
        for source in SOURCES:
            assert any(id_.startswith(source + '.') for id_ in ids) == (source in selected)
        context = resolver.resolve(snap, 'symbol_interpretation', query='instrument valve signal connector heat exchanger')
        assert all(e['source'].get('document_id') in [None, *selected] for e in context['references'])
        assert context['selected_sources'] == sorted(selected)
    assert len(identities) == 8
    assert resolver.knowledge_snapshot('off', source_ids=list(SOURCES)) == {}
    for invalid in ('isa-5.1-2009', ['unknown'], [True]):
        with pytest.raises(ValueError, match='source'):
            resolver.knowledge_snapshot('general', source_ids=invalid)


def test_explicit_editions_sheet_overrides_and_legacy_behavior(catalog):
    profile = dict(id='test', name='Test', version=1, confirmed=True,
                   context={'standards':[{'name':'ISA 5.1','edition':'2024'}]})
    snapshot = resolver.knowledge_snapshot('profile', profile, source_ids=list(SOURCES))
    assert not any(i.startswith('isa-') for i in resolver.applicable_references(snapshot)['reference_ids'])
    profile['context']['standards'] = []
    snapshot = resolver.knowledge_snapshot('profile', profile,
        [{'pages':[2], 'context':{'standards':[]}}], source_ids=list(SOURCES))
    assert any(i.startswith('isa-') for i in resolver.applicable_references(snapshot, 0)['reference_ids'])
    ids = resolver.applicable_references(snapshot, 1)['reference_ids']
    assert not any(i.startswith(('isa-', 'sht-')) for i in ids)
    assert any(i.startswith('projectmaterials.') for i in ids)
    legacy = resolver.knowledge_snapshot('general')
    assert legacy['selected_sources'] is None
    assert not any(i.startswith('isa-') for i in resolver.applicable_references(legacy)['reference_ids'])


def legend():
    return [{'source_row_id':'row-17', 'source':'legend_extracted', 'source_page_index':0,
             'source_bbox':{'x':1, 'y':2, 'w':30, 'h':20}, 'label':'Signal connector',
             'kind':'connector', 'symbol_class':'opc'}]


def test_model_citations_are_validated_without_losing_uncertainty(catalog):
    ctx = resolver.resolve(resolver.knowledge_snapshot('general', source_ids=['isa-5.1-2009']),
                           'symbol_interpretation', query='drawing-to-drawing signal connector')
    reference = ctx['reference_ids'][0]
    obj = PerceivedObject(kind='opc', bbox={'x':.1,'y':.1,'w':.2,'h':.1}, confidence='low',
        knowledge_reference_id=reference, knowledge_evidence='Two compartments in source',
        legend_entry_ids=['row-17'], legend_evidence='Rounded end matches legend',
        attributes={'legend_matches':[{'legend_entry_id':'forged'}]})
    batch = PerceptionBatch(objects=[obj])
    compact, _, _ = select_legend_context(legend(), [], [])
    _bind_knowledge(batch, ctx)
    _bind_legend(batch, compact)
    origin = evidence_origin({'attributes':obj.attributes})
    assert [s['label'] for s in origin['sources']] == ['ISA 5.1:2009','图纸图例表']
    assert origin['sources'][1]['page_index'] == 0
    assert obj.confidence == 'low' and obj.bbox.x == .1
    obj.legend_entry_ids = ['invented']
    obj.knowledge_reference_id = 'invented'
    _bind_knowledge(batch, ctx)
    _bind_legend(batch, compact)
    origin = evidence_origin({'attributes':obj.attributes})
    assert all(s['kind'] == 'other' for s in origin['sources'])
    assert 'reference_not_supplied' in origin['citation_errors']
    assert 'legend_not_supplied' in origin['citation_errors']
    assert obj.confidence == 'low'


def test_supplied_does_not_mean_matched_and_old_runs_remain_honest():
    row = {'attributes': {'recognition_method':'vlm', 'recognition_evidence':'Visible valve body',
                         'supplied_knowledge':{'references':[{'id':'isa-5.1-2009.example','version':1}]}}}
    original = copy.deepcopy(row)
    origin = evidence_origin(row)
    assert origin['sources'][0]['kind'] == 'other'
    assert '未引用参考' in origin['summary']
    assert origin['supplied_sources'] == ['ISA 5.1:2009'] and row == original
    assert '未记录' in evidence_origin({})['summary']
    assert '内置符号' in evidence_origin({'source':'built_in'}, is_legend=True)['summary']
    assert '图纸图例' in evidence_origin(legend()[0], is_legend=True)['summary']


def test_selected_source_does_not_validate_an_unsupplied_match(catalog):
    ctx = resolver.resolve(resolver.knowledge_snapshot('general', source_ids=['projectmaterials.pid-symbols']),
                           'symbol_interpretation', query='heat exchanger')
    assert validate_match(ctx, 'isa-5.1-2009.t5.3.2-17', None, 'Visible source glyph')[1] == 'reference_not_supplied'


def test_multiple_knowledge_sources_on_one_symbol(catalog):
    from diagex.vision.raster_semantics import normalize_semantics
    from diagex.vision.symbol_interpretation import _tool_input_schema

    refs = [next(e for e in catalog['entries'] if e['source'].get('document_id') == source
                 and e['reference_type'] == 'symbol' and e['catalog']['role'] == 'catalog')
            for source in ('isa-5.1-2009', 'sht-3101-2017')]
    context = {'references':refs}
    citations = [{'reference_id':e['id'], 'drawing_evidence':'Visible geometry in the drawing'} for e in refs]
    obj = PerceivedObject(kind='instrument', candidate_id='candidate-1', knowledge_citations=citations)
    _bind_knowledge(PerceptionBatch(objects=[obj]), context)
    origin = evidence_origin({'attributes':obj.attributes})
    assert {s['label'] for s in origin['sources']} == {'ISA 5.1:2009', 'SH/T 3101-2017'}
    schema = _tool_input_schema()['properties']['proposals']['items']['properties']['knowledge_citations']['items']
    assert schema['properties']['reference_id']['type'] == 'string' and '$ref' not in schema
    rows, _ = normalize_semantics({'results':[{'symbol_id':'s1', 'decision':'interpreted',
        'reason':'Visible geometry', 'legend_entry_ids':[], 'knowledge_citations':citations,
        'symbol':{'kind':'instrument','instrument_function':'indicator'}}]}, ['s1'], [], context)
    assert rows[0]['decision'] == 'interpreted' and len(rows[0]['knowledge_matches']) == 2


def test_explicit_selection_cannot_leak_unversioned_builtin_pack():
    from diagex.config import Config
    from diagex.extractors.pid_legend import load_run_builtin_pack

    config = Config()
    assert load_run_builtin_pack('isa-5.1', config).entries
    for sources in ([], ['sht-3101-2017'], ['isa-5.1-2009'], list(SOURCES)):
        config.knowledge = {'selected_sources': sources}
        assert not load_run_builtin_pack('isa-5.1', config).entries
