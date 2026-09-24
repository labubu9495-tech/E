"""Frozen, audited attachment3 inference for the completed comparison."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from data import adapt,load_pickle,normalize,TextEncoder,observation_audit,sha256,dump_json
from infer import natural_file_key
from models import Predictor
from paper_models import make_model


def load_frozen(spec,device):
    models=[]
    for relative in spec['checkpoints']:
        p=ROOT/relative
        if sha256(p)!=spec['checkpoint_sha256'][relative]:
            raise ValueError('Frozen checkpoint hash mismatch')
        checkpoint=torch.load(p,map_location=device,weights_only=False)
        if spec['implementation']=='legacy':
            model=Predictor(**checkpoint['model_args'])
        else:
            model=make_model(spec['model'],**checkpoint['model_args'])
        model.load_state_dict(checkpoint['state_dict'])
        models.append(model.to(device).eval())
    return models


@torch.inference_mode()
def predict(models,inputs):
    probabilities,regressions=[],[]
    for model in models:
        logits,reg=model(*inputs)
        probabilities.append(logits.softmax(-1).cpu().numpy())
        regressions.append(reg.cpu().numpy())
    return np.mean(probabilities,0),np.mean(regressions,0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path)
    parser.add_argument('--which',choices=['overall','paper'],default='overall')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(2)
    selection=json.loads((HERE/'selection.json').read_text())
    for relative,expected in selection['frozen_inputs'].items():
        if sha256(ROOT/relative)!=expected:raise ValueError('Input pipeline changed after selection')
    if sha256(HERE/'paper_models.py')!=selection['paper_model_sha256']:raise ValueError('Paper model source changed')
    if sha256(HERE/'protocol.json')!=selection['protocol_sha256']:raise ValueError('Frozen protocol changed')
    spec=selection[args.which]
    models=load_frozen(spec,args.device)
    encoder=TextEncoder(args.device)
    stats=np.load(ROOT/'assets/normalization.npz')
    if args.input is None:
        files=[p for p in (ROOT.parent/'E题').rglob('*.pkl') if '附件3-' in str(p) and p.parent.name=='对齐版本']
    else:
        files=list(args.input.glob('*.pkl'))
    files=sorted(files,key=natural_file_key)
    assert len(files)==30,'Expected exactly 30 official aligned samples'
    rows,provenance=[],[]
    for p in files:
        source=load_pickle(p);source=source.get('test',source)
        a=adapt(source)
        assert len(a['tokens'])==1
        text=encoder.encode(a['tokens'],a['observed'][...,0]).astype(np.float32)
        audio=normalize(a['audio'],stats['audio_mean'],stats['audio_std'],a['observed'][...,1])
        vision=normalize(a['vision'],stats['vision_mean'],stats['vision_std'],a['observed'][...,2])
        inputs=[torch.from_numpy(x).to(args.device) for x in [text,audio,vision,a['sequence'],a['observed']]]
        prob,reg=predict(models,inputs)
        assert np.isfinite(prob).all() and np.isfinite(reg).all()
        assert np.allclose(prob.sum(-1),1.,atol=1e-6) and (np.abs(reg)<=3).all()
        rows.append(dict(sample_id=p.stem,polarity=['Negative','Neutral','Positive'][int(prob[0].argmax())],
            intensity=float(reg[0]),probability_negative=float(prob[0,0]),probability_neutral=float(prob[0,1]),probability_positive=float(prob[0,2])))
        provenance.append(dict(sample_id=p.stem,input_sha256=sha256(p),**observation_audit(a,0)))
    assert len({r['sample_id'] for r in rows})==30
    output=HERE/f'attachment3_{args.which}_predictions.csv'
    with output.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump_json(output.with_suffix('.provenance.json'),dict(model=spec['model'],selection_sha256=sha256(HERE/'selection.json'),
        checkpoints=spec['checkpoint_sha256'],prediction_sha256=sha256(output),inputs=selection['frozen_inputs'],rows=provenance,
        label_policy='No attachment3 labels available; no accuracy claim; frozen selection before inference.'))
    print(json.dumps(dict(model=spec['model'],rows=len(rows),output=str(output)),ensure_ascii=False))


if __name__=='__main__':main()
