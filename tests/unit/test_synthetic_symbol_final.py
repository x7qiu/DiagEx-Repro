from pathlib import Path

import pytest
from PIL import Image

from eval import synthetic_symbol_final as final
from eval.public_symbol_study import load, save, sha


def test_fixed_source_crop_and_half_open_ownership():
    assert final.center_crop((7168,4562)) == {'x':2684,'y':1581,'w':1800,'h':1400}
    with pytest.raises(ValueError):
        final.center_crop((100,100))
    core={'x':140,'y':140,'w':1520,'h':1120}
    assert final.owned([130,130,150,150],core)
    assert not final.owned([1650,130,1670,150],core)


@pytest.fixture
def study(tmp_path,monkeypatch):
    root=tmp_path/'repo'
    root.mkdir()
    monkeypatch.setattr(final,'ROOT',root)
    directory=root/'study'
    directory.mkdir()
    data=tmp_path/'testdata/PID2Graph'
    image=data/'Complete/Dataset PID/71.png'
    image.parent.mkdir(parents=True)
    Image.new('RGB',(3000,2000),'white').save(image)
    chosen={'dataset_root':str(data),'drawings':[{'id':'Dataset PID/71','collection':'Dataset PID',
            'image':'Complete/Dataset PID/71.png','image_sha256':sha(image),'size':[3000,2000],
            'graph':'forbidden.graphml','graph_sha256':'unopened'}]}
    save(directory/'heldout-selection.json',chosen)
    save(directory/'final-selection.json',{'source_hashes':{},'checkpoint':'unused','checkpoint_sha256':'unused',
         'conditions':[{'id':'baseline-vlm','variant':'baseline','backend':'baseline','cv':False}]})
    save(directory/'synthetic-final-protocol.json',{'selection_sha256':sha(directory/'final-selection.json'),
         'source_selection_sha256':sha(directory/'heldout-selection.json'),
         'source_collections':['Dataset PID','PID2Graph Synthetic'],'conditions':['baseline-vlm'],
         'public_dataset':'https://zenodo.org/records/14803338'})
    return directory


def test_preparation_uses_source_pixels_and_never_requires_graphml(study,monkeypatch):
    captures=[]
    monkeypatch.setattr(final.subprocess,'run',lambda *a,**k: None)
    def prepare(args,source):
        captures.append(source)
        save(args.out/'prepared.json',{'input':source})
    monkeypatch.setattr(final,'_prepare_checked_source',prepare)
    final.prepare(study)
    assert len(captures)==1
    source=captures[0]
    assert source['context_bbox_global']=={'x':600,'y':300,'w':1800,'h':1400}
    assert Image.open(source['image']).size==(1800,1400)
    assert not (Path(load(study/'heldout-selection.json')['dataset_root'])/'forbidden.graphml').exists()
    assert 'graph' not in source and 'targets' not in source
    with pytest.raises(ValueError,match='overwrite'):
        final.prepare(study)


def test_frozen_selection_changes_are_rejected(study):
    save(study/'heldout-selection.json',{'drawings':[]})
    with pytest.raises(ValueError,match='drawing selection changed'):
        list(final.source_rows(study))


def test_scoring_requires_prediction_completion_before_truth(study):
    directory=study/'synthetic-final-inputs/dataset-pid-71'
    save(directory/'inference-input.json',{'context_bbox_global':{'x':600,'y':300,'w':1800,'h':1400},
         'evaluation_core_bbox_local':{'x':140,'y':140,'w':1520,'h':1120}})
    with pytest.raises(ValueError,match='Complete all predictions'):
        final.score(study)
