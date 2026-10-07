"""Source-only public legend runner contracts; no live client or paid calls."""
import copy
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from diagex.config import LLMConfig
from diagex.llm.budget import BudgetExceeded
from eval import public_legend_canary as canary


@pytest.fixture
def fixture(tmp_path):
    original = canary.DEFAULT_STUDY / canary.SOURCE_DIR
    root = tmp_path / canary.SOURCE_DIR
    root.mkdir()
    raw = canary.load(original / 'inference-input.json')
    for field in ('source_pdf', 'image', 'page_evidence'):
        src = original / Path(raw[field]).name
        shutil.copyfile(src, root / src.name)
        raw[field] = str(root / src.name)
    shutil.copytree(original / 'assets', root / 'assets')
    provenance = canary.load(original / 'provenance.json')
    for asset in provenance['assets']:
        asset['asset_file'] = str(root / 'assets' / Path(asset['asset_file']).name)
    # This text is forbidden annotation data, never part of model input.
    raw['annotations'] = [{'caption': 'FORBIDDEN_GOLD', 'bbox': {'x': 1}}]
    raw['instructions'] = 'FORBIDDEN_GOLD'
    canary.save(root / 'inference-input.json', raw)
    canary.save(root / 'provenance.json', provenance)
    canary.save(root / 'freeze.json', {
        'constructed': True, 'freeze_before_model_predictions': True,
        'artifacts': {p.name: canary.sha(p) for p in root.iterdir() if p.is_file()},
    })
    return tmp_path, root, raw


def test_allowlist_drops_annotations_and_requires_public_artifact_checksums(fixture):
    study, root, raw = fixture
    checked = canary.checked_input(root / 'inference-input.json', study)
    assert 'FORBIDDEN_GOLD' not in json.dumps(checked)
    assert checked['source_pdf'] == raw['source_pdf']
    with open(raw['image'], 'ab') as image:
        image.write(b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        canary.checked_input(root / 'inference-input.json', study)


def test_external_or_nonpublic_asset_rejected_even_when_manifest_refrozen(fixture):
    study, root, raw = fixture
    provenance = canary.load(root / 'provenance.json')
    provenance['assets'][0]['asset_url'] = 'https://private.example/image.png'
    canary.save(root / 'provenance.json', provenance)
    freeze = canary.load(root / 'freeze.json')
    freeze['artifacts']['provenance.json'] = canary.sha(root / 'provenance.json')
    canary.save(root / 'freeze.json', freeze)
    with pytest.raises(ValueError, match='Nonpublic'):
        canary.checked_input(root / 'inference-input.json', study)
    with pytest.raises(ValueError, match='prepared public legend'):
        canary.checked_input(root / 'inference-input.json', study / 'different')


def tool(name, payload, identifier='mock'):
    return {'type': 'tool_use', 'id': identifier, 'name': name, 'input': payload}


def reply(blocks):
    return SimpleNamespace(content=blocks, usage=None, stop_reason='tool_use')


@pytest.mark.parametrize("tile_image_budget,region_count", [(None, 4), (3000, 1)])
def test_production_raster_legend_route_with_mocked_model_has_source_only_requests(
    fixture, tmp_path, tile_image_budget, region_count
):
    study, root, raw = fixture
    source = canary.checked_input(root / 'inference-input.json', study)
    requests = []

    class Client:
        config = LLMConfig(model='offline', reasoning_mode='disabled')
        next_region = 0

        def messages_create(self, **kwargs):
            requests.append(copy.deepcopy({k: v for k, v in kwargs.items() if not callable(v)}))
            # One complete source-driven tool sequence for each scheduled region.
            messages = kwargs['messages']
            if len(messages) == 1:
                tile = f'legend-{self.next_region}'
                self.next_region += 1
                return reply([tool('get_overview', {}),
                              tool('get_tile', {'tile_id': tile}, 'tile')])
            last = json.dumps(messages[-1])
            tag = re.search(r'view_tag=(tile:legend-\d+)', last).group(1)
            return reply([
                tool('annotate', {'view_tag': tag, 'kind': 'equipment', 'label': 'Offline mock caption',
                                 'bbox': {'x': 100, 'y': 100, 'w': 20, 'h': 20},
                                 'attributes': {'legend_kind': 'symbol', 'legend_symbol_class': 'filter'}}),
                tool('finish', {'answer': 'done'}, 'finish'),
            ])

    client = Client()
    calls = canary.recorded_client(client, tmp_path / 'run', 8)
    result = canary.invoke(source, client, tmp_path / 'run', tile_image_budget)
    assert len(calls) == region_count * 2
    assert result['source'] == 'explicit_pages' and not result['used_builtin_only']
    assert result['entries'] and result['coverage']
    assert len(result['coverage']) == region_count
    assert all(c['status'] == 'complete' for c in result['coverage'])
    assert all(e['source'] == 'legend_extracted' for e in result['entries'])
    assert all(r['max_attempts'] == 1 and r['time_budget_s'] == 120 for r in requests)
    assert 'FORBIDDEN_GOLD' not in json.dumps(requests)
    assert not (root / 'annotations.json').exists()  # Inference works without this file.
    assert len(list((tmp_path / 'run').glob('response-*.json'))) == len(calls)


def test_dispatch_limit_stops_before_ninth_underlying_call(tmp_path):
    dispatched = []
    client = SimpleNamespace(messages_create=lambda **kw: dispatched.append(kw) or reply([]))
    calls = canary.recorded_client(client, tmp_path, 8)
    for _ in range(8):
        client.messages_create(messages=[])
    with pytest.raises(BudgetExceeded, match='before dispatch'):
        client.messages_create(messages=[])
    assert len(dispatched) == len(calls) == 8


def test_caption_swap_giant_crop_duplicate_and_actuator_role_are_visible():
    def row(identity, x, caption, kind='line', role=None):
        return {'id': identity, 'caption': caption, 'expected_kind': kind, 'symbol_role': role,
                'glyph_bbox': {'x': x, 'y': 10, 'w': 10, 'h': 2},
                'caption_bbox': {'x': x+20, 'y': 10, 'w': 20, 'h': 10}}

    def entry(label, x, **updates):
        return {'label': label, 'kind': 'line', 'symbol_class': 'line',
                'source_bbox': {'x': x, 'y': 9, 'w': 10, 'h': 4}, **updates}

    annotations = {'rows': [row('a', 10, 'Process Lines'), row('b', 70, 'Equipment Outline'),
                            row('c', 130, 'Piston Operator', 'equipment', 'actuator')]}
    entries = [entry('Process Lines', 10), entry('Process Lines', 10),
               entry('Equipment Outline', 10),  # Caption paired to wrong glyph.
               entry('Piston Operator', 130, kind='equipment', symbol_class='valve'),
               entry('extra', 300)]
    score = canary.metrics(entries, annotations)
    assert score['captions_found'] == 3 and score['duplicate_caption_count'] == 1
    assert score['rows'][0]['glyph_coverage'] == 1 and score['rows'][0]['crop_iou'] == .5
    assert not score['rows'][1]['caption_glyph_pairing_supported']
    assert not score['rows'][2]['role_correct']
    assert score['unknown_caption_indices'] == [4]
    giant = entry('Process Lines', 0, source_bbox={'x': 0, 'y': 0, 'w': 200, 'h': 30})
    scored = canary.metrics([giant], annotations)['rows'][0]
    assert scored['glyph_coverage'] == 1 and not scored['caption_glyph_pairing_supported']
    assert scored['caption_overlap_area'] > 0 and scored['overlapping_other_rows'] == ['b', 'c']
