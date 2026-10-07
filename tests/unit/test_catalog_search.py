"""Exercise the browser's actual search implementation against the curated library."""
import copy
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from diagex.knowledge.library import load_library
from diagex.knowledge.resolver import digest, knowledge_snapshot, library_root, resolve

ROOT = Path(__file__).resolve().parents[2]
SEARCH = ROOT / 'src/diagex/web/static/catalog-search.js'


def search(entries, query, **options):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required to test browser search')
    script = """
    const search = require(process.argv[1]);
    const input = JSON.parse(require('fs').readFileSync(0,'utf8'));
    const result=input.entries.map(e=>({id:e.id,...search.match(e,input.query,input.options)}))
      .filter(e=>e.kind).sort((a,b)=>b.score-a.score);
    process.stdout.write(JSON.stringify(result));
    """
    result = subprocess.run([node, '-e', script, str(SEARCH)],
                            input=json.dumps(dict(entries=entries, query=query, options=options)),
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


@pytest.fixture(scope='module')
def entries():
    return load_library(library_root())['entries']


def test_heat_exchanger_names_and_curated_families(entries):
    recipe = yaml.safe_load((library_root() / 'recipes/catalog-search.yaml').read_text())
    results = search(entries, 'heat exchanger')
    ids = {r['id'] for r in results}
    assert set(recipe['families'][0]['reference_ids']) <= ids
    assert not set(recipe['excluded_from_family']) & ids
    assert results[0]['kind'] == 'name'
    assert any(r['kind'] == 'family' for r in results)
    assert ids == {r['id'] for r in search(entries, 'HEAT-EXCHANGERS')}
    assert set(recipe['families'][0]['reference_ids']) <= {r['id'] for r in search(entries, '换热器')}
    assert not search(entries, 'heat exchanger impossiblexyz')


def test_source_notes_are_opt_in(entries):
    tower = next(e for e in entries if e['id'].endswith('heat-exchangers.cooling-tower'))
    assert not search([tower], 'heat exchanger')
    assert search([tower], 'heat exchanger', expanded=True)[0]['kind'] == 'description'
    assert 'Heat Exchangers P&ID Symbols' not in tower['tags']
    assert 'Heat Exchangers P&ID Symbols' not in tower['catalog']['interpretation']
    assert tower['source']['section'] == 'Heat Exchangers P&ID Symbols'


def test_field_matching_ranking_and_word_boundaries():
    entries = [
        dict(id='family', concept='Shell and tube', catalog={'search_terms':['heat exchanger']}),
        dict(id='alias', concept='HX', aliases=['Heat exchanger']),
        dict(id='exact', concept='Heat exchanger'),
        dict(id='description', concept='Boiler', explanation='Near heat exchanger'),
        dict(id='pump.gas', concept='Gasoline engine'),
    ]
    assert [r['id'] for r in search(entries, 'heat exchanger')] == ['exact','alias','family']
    assert [r['id'] for r in search(entries, 'heat exch')] == ['exact','alias','family']
    assert not search(entries, 'gas')
    assert not search(entries, 'pump')
    assert search(entries, 'pump.gas')[0]['kind'] == 'id'
    assert search(entries[:1], 'retired label', replacementLabels=['Retired label'])[0]['kind'] == 'alias'
    assert len(search(entries, '')) == len(entries)


def test_pipeline_vocabulary_and_cache(entries):
    result = resolve(knowledge_snapshot('general'), 'symbol_interpretation', query='heat exchanger')
    assert result['reference_ids']
    assert not any('cooling-tower' in i or i.endswith(('.boiler','.boiler-2','.heater'))
                   for i in result['reference_ids'])
    original = next(e for e in entries if e['catalog'] and e['catalog']['search_terms'])
    modified = copy.deepcopy(original)
    modified['catalog']['search_terms'].append('another name')
    assert digest(original) != digest(modified)


def test_curation_is_idempotent_and_preserves_source(tmp_path, entries):
    spec = importlib.util.spec_from_file_location('curate_search_test', ROOT / 'scripts/curate_catalog_search.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = copy.deepcopy(next(e for e in entries if e['id'].endswith('heat-exchangers.cooling-tower')))
    section = entry['source']['section']
    entry['tags'].append(section)
    entry['catalog']['interpretation'] = f"Projectmaterials labels this illustration ‘{entry['concept']}’ ({section})."
    (tmp_path / 'recipes').mkdir()
    (tmp_path / 'recipes/catalog-search.yaml').write_text(yaml.safe_dump({
        'families': [], 'excluded_from_family': [entry['id']]}))
    path = tmp_path / f"{entry['id']}.yaml"
    path.write_text(yaml.safe_dump(entry))
    assert module.curate(tmp_path) == 1
    saved = path.read_text()
    assert module.curate(tmp_path) == 0 and path.read_text() == saved
    restored = yaml.safe_load(saved)
    assert restored['version'] == entry['version'] + 1
    assert restored['source'] == {k: v for k, v in entry['source'].items() if v is not None}
