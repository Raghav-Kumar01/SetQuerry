"""Architecture-mode tests reuse the live HTTP harness, never an optimizer."""
import base64
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from integration.test_demo_api import images

# Parametrize the existing server factory via a mode environment hook.
@pytest.fixture(scope='module')
def api(tmp_path_factory):
    from integration.test_demo_api import start_api
    yield from start_api(tmp_path_factory, 'architecture')

@pytest.mark.parametrize('route,fields,shape', [('single',['image'],[1,6359]),('change',['image_before','image_after'],[1,1,256,256]),('optical-sar',['image_optical','image_sar'],[1,2])])
def test_network_http(api, images, route, fields, shape):
    def request(image):
        response=api.post('/api/analyze/'+route,files={field:(field+'.png', images[image] if i==len(fields)-1 else images['empty'],'image/png') for i,field in enumerate(fields)},data={'query':'Is there a road?'})
        assert response.status_code==200,response.text
        return response.json()
    first=request('empty'); second=request('changed'); repeated=request('changed')
    assert second['label']=='UNTRAINED MODEL' and second['weights_source']=='random'
    assert second['diagnostics']['eval_mode'] and not second['diagnostics']['requires_grad']
    assert second['diagnostics']['output_shape']==shape
    assert first['diagnostics']['logits_sha256'] != second['diagnostics']['logits_sha256']
    assert second['diagnostics']['logits_sha256']==repeated['diagnostics']['logits_sha256']
    if route=='change':
        mask=np.asarray(Image.open(io.BytesIO(base64.b64decode(second['mask_base64']))))
        assert np.count_nonzero(mask)==second['changed_pixels']
    if route=='optical-sar':
        assert 'alignment_probability' not in second
        assert len(second['class_scores'])==2
    for bad in [None,b'invalid']:
        files={} if bad is None else {f:(f+'.png',bad,'image/png') for f in fields}
        assert api.post('/api/analyze/'+route,files=files).status_code==422
    summary={k:v for k,v in second.items() if k not in ['mask_base64','logits']}
    print(json.dumps(summary))
    target=Path('output/architecture');target.mkdir(parents=True,exist_ok=True)
    (target/(route+'.json')).write_text(json.dumps(second,indent=2))

@pytest.mark.parametrize('number',[1,2,3])
def test_checkpoint_loading_is_strict_and_never_claims_training(number, monkeypatch, tmp_path):
    import torch
    from ml.inference.architecture import _load, predict
    from ml.inference.assets import ModelUnavailableError
    # Serialize existing random parameters only: no training or optimizer.
    model,_ = _load(number,None,None,'cpu')
    checkpoint=tmp_path/'compatible.pt'
    torch.save({'model_state_dict': model.state_dict()},checkpoint)
    monkeypatch.setenv(f'SETQUERY_MODEL{number}_CHECKPOINT',str(checkpoint))
    image=tmp_path/'image.png';Image.new('RGB',(16,16),(40,80,120)).save(image)
    result=predict(number,[str(image)]*(1 if number==1 else 2),'What is here?')
    assert result['weights_source']=='checkpoint' and result['is_trained'] is None
    assert result['diagnostics']['initialization']=='success'
    checkpoint=tmp_path/'incompatible.pt'
    torch.save({'model_state_dict': {}},checkpoint)
    monkeypatch.setenv(f'SETQUERY_MODEL{number}_CHECKPOINT',str(checkpoint))
    with pytest.raises(ModelUnavailableError):
        predict(number,[str(image)]*2,'What is here?')


def test_production_isolated_from_all_architecture_caches(monkeypatch,tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from integration.test_demo_api import png
    for number in [1,2,3]:
        monkeypatch.setenv(f'SETQUERY_MODEL{number}_CHECKPOINT',str(tmp_path/f'missing-{number}.pt'))
    with TestClient(app) as client:
        for mode in ['architecture','production']:
            monkeypatch.setenv('MODEL_MODE',mode)
            for route,fields in [('single',['image']),('change',['image_before','image_after']),('optical-sar',['image_optical','image_sar'])]:
                data=png(np.zeros((16,16,3),dtype=np.uint8))
                response=client.post('/api/analyze/'+route,files={field:(field+'.png',data,'image/png') for field in fields})
                assert response.status_code==(200 if mode=='architecture' else 503),response.text
