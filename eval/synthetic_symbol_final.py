"""Frozen PID2Graph center-crop validation; GraphML is read only by score()."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from eval.public_symbol_study import DEFAULT_STUDY, ROOT, _prepare_checked_source, load, save, sha


def checked_protocol(study):
    final = load(study / 'final-selection.json')
    protocol = load(study / 'synthetic-final-protocol.json')
    if sha(study / 'final-selection.json') != protocol['selection_sha256']:
        raise ValueError('Frozen method selection changed')
    if sha(study / 'heldout-selection.json') != protocol['source_selection_sha256']:
        raise ValueError('Frozen drawing selection changed')
    for path, expected in final['source_hashes'].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'Frozen implementation changed: {path}')
    return final, protocol, load(study / 'heldout-selection.json')


def center_crop(size):
    width, height = size
    if width < 1800 or height < 1400:
        raise ValueError('Source too small for the frozen crop protocol')
    return {'x': (width - 1800) // 2, 'y': (height - 1400) // 2, 'w': 1800, 'h': 1400}


def owned(box, core):
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return core['x'] <= x < core['x'] + core['w'] and core['y'] <= y < core['y'] + core['h']


def source_rows(study):
    final, protocol, selected = checked_protocol(study)
    root = Path(selected['dataset_root']).resolve()
    if root != (ROOT.parent / 'testdata/PID2Graph').resolve():
        raise ValueError('Only the existing public PID2Graph dataset is eligible')
    for row in selected['drawings']:
        if row['collection'] not in protocol['source_collections']:
            raise ValueError('Non-synthetic collection is outside this protocol')
        source = (root / row['image']).resolve()
        if not source.is_relative_to(root) or sha(source) != row['image_sha256']:
            raise ValueError('Frozen source image changed')
        yield final, protocol, row, source


def prepare(study):
    for final, protocol, row, source in source_rows(study):
        name = row['id'].replace('/', '-').replace(' ', '-').lower()
        out = study / 'synthetic-final-inputs' / name
        if out.exists():
            raise ValueError('Do not overwrite frozen source preparations')
        with Image.open(source) as original:
            if list(original.size) != row['size']:
                raise ValueError('Source dimensions changed')
            box = center_crop(original.size)
            crop = original.convert('RGB').crop((box['x'], box['y'], box['x']+box['w'], box['y']+box['h']))
        out.mkdir(parents=True)
        image = out / 'source.png'
        crop.save(image)
        manifest = {'panel_id': name, 'source_pdf': str(source),
                    'source_format': 'image (legacy source_pdf field is the original source path)',
                    'source_sha256': row['image_sha256'], 'page_index': 0, 'render_dpi': 300,
                    'image': str(image.resolve()), 'image_sha256': sha(image), 'image_pixel_size': list(crop.size),
                    'context_bbox_global': box, 'evaluation_core_bbox_local': {'x': 140, 'y': 140, 'w': 1520, 'h': 1120}}
        manifest_path = out / 'inference-input.json'
        save(manifest_path, manifest)
        guidance = out / 'cv.json'
        subprocess.run([str(ROOT/'.venv-cv/bin/python'), '-m', 'diagex.vision.raster_detector',
                        '--image', str(image), '--checkpoint', str(ROOT/final['checkpoint']),
                        '--checkpoint-sha256', final['checkpoint_sha256'], '--out', str(guidance)], cwd=ROOT,
                       env={**os.environ, 'PYTHONPATH': str(ROOT/'src')}, check=True)
        for condition in protocol['conditions']:
            config = next(c for c in final['conditions'] if c['id'] == condition)
            destination = study / 'public-prepared' / ('synthetic-final-'+name+'-'+condition)
            args = SimpleNamespace(study=study, input=manifest_path, out=destination,
                                   variant=config['variant'], backend=config['backend'], knowledge='off',
                                   reference_images='all', workflow='fixed', grid=1,
                                   cv_guidance=guidance if config['cv'] else None)
            with contextlib.redirect_stdout(io.StringIO()):
                _prepare_checked_source(args, manifest)
            prepared = load(destination/'prepared.json')
            prepared['source_attribution'] = {'dataset': 'PID2Graph', 'url': protocol['public_dataset'],
                                               'collection': row['collection'], 'id': row['id'],
                                               'frozen_protocol_sha256': sha(study/'synthetic-final-protocol.json')}
            save(destination/'prepared.json', prepared)
        print('Prepared source-only', name, flush=True)


def run(study):
    from eval.public_symbol_study import run as run_public
    final, protocol, selected = checked_protocol(study)
    for row in selected['drawings']:
        name = row['id'].replace('/', '-').replace(' ', '-').lower()
        for condition in protocol['conditions']:
            identity = 'synthetic-final-'+name+'-'+condition
            run_public(SimpleNamespace(study=study, prepared=study/'public-prepared'/identity,
                                       out=study/'public-live'/identity, model=protocol['model'],
                                       provider_tag=protocol['provider'], tool_choice=protocol['tool_choice'],
                                       max_usd=protocol['max_usd_per_run'], max_calls=protocol['max_calls_per_run'],
                                       final_validation=True))


def score(study):
    from eval.pid2graph.data import read_graph
    from eval.pid2graph.runner import record
    from eval.pid2graph.scoring import aggregate, score_drawing
    _, protocol, selected = checked_protocol(study)
    results = {c: [] for c in protocol['conditions']}
    for row in selected['drawings']:
        name = row['id'].replace('/', '-').replace(' ', '-').lower()
        source = load(study/'synthetic-final-inputs'/name/'inference-input.json')
        box = source['context_bbox_global']
        local = source['evaluation_core_bbox_local']
        core = {**local, 'x': box['x']+local['x'], 'y': box['y']+local['y']}
        # Every frozen condition must finish prediction before this drawing's truth is opened.
        outputs = {c: study/'public-live'/('synthetic-final-'+name+'-'+c)/'result.json' for c in protocol['conditions']}
        if not all(path.exists() for path in outputs.values()):
            raise ValueError('Complete all predictions before scoring')
        graph_path = Path(selected['dataset_root']) / row['graph']
        if sha(graph_path) != row['graph_sha256']:
            raise ValueError('Frozen GraphML changed')
        graph = read_graph(graph_path)
        truth = [n for n in graph['nodes'] if n['label'] in protocol['primary_classes'] and owned(n['bbox'], core)]
        for condition, path in outputs.items():
            run_result = load(path)
            predictions=[]
            for detection in run_result.get('detections', []):
                bbox = {**detection['bbox'], 'x': detection['bbox']['x']+box['x'], 'y': detection['bbox']['y']+box['y']}
                converted = record(detection['id'], detection['kind'], detection.get('attributes', {}), bbox,
                                   detection.get('confidence', 'low'), 1, 1, 'accepted')
                if owned(converted['bbox'], core):
                    predictions.append(converted)
            result = score_drawing({'nodes': truth, 'edges': []}, predictions)
            result.update(drawing=row['id'], result_sha256=sha(path), status=run_result['status'],
                          billing=run_result['billing'], elapsed_s=run_result['elapsed_s'],
                          excluded_truth_counts={label: sum(n['label']==label and owned(n['bbox'],core) for n in graph['nodes']) for label in protocol['excluded_classes']})
            results[condition].append(result)
    report={'protocol': protocol, 'drawings': results, 'aggregate': {c: aggregate(r) for c,r in results.items()}}
    save(study/'synthetic-final-score.json',report)
    print(json.dumps(report['aggregate'],indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','score'])
    parser.add_argument('--study',type=Path,default=DEFAULT_STUDY)
    args=parser.parse_args()
    globals()[args.command](args.study.resolve())


if __name__ == '__main__':
    main()
