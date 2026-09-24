"""Separate frozen inference for the new linear-family experiment."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import csv
import json
import re
import joblib
import numpy as np
import torch
from data import adapt,load_pickle,normalize,TextEncoder,sha256,dump_json,observation_audit
from features import pooled_features,visible_documents


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=HERE/'attachment3_predictions.csv')
    args=parser.parse_args();torch.set_num_threads(2)
    selection=json.loads((HERE/'selection.json').read_text());family=selection['selected']
    model_path=HERE/f'{family}.joblib'
    if sha256(model_path)!=selection['model_sha256'][family]:raise ValueError('Frozen model hash mismatch')
    payload=joblib.load(model_path);encoder=TextEncoder('cpu');stats=np.load(ROOT/'assets/normalization.npz')
    files=sorted(args.input.glob('*.pkl'),key=lambda p:[int(t) if t.isdigit() else t for t in re.split(r'(\d+)',p.name)])
    if not files:raise ValueError('Supply directory containing aligned attachment3 PKLs')
    rows=[];provenance=[]
    for path in files:
        s=load_pickle(path);s=s.get('test',s);a=adapt(s)
        text=encoder.encode(a['tokens'],a['observed'][:,:,0])
        audio=normalize(a['audio'],stats['audio_mean'],stats['audio_std'],a['observed'][:,:,1])
        vision=normalize(a['vision'],stats['vision_mean'],stats['vision_std'],a['observed'][:,:,2])
        dense=pooled_features(text,audio,vision,a['sequence'],a['observed'])
        docs=visible_documents(a['tokens'][:,0],a['observed'][:,:,0],a['sequence'])
        x=payload['transform'].transform(dense,docs,family)
        prob=np.mean([m.predict_proba(x) for m in payload['classifiers']],0)
        reg=np.mean([np.clip(m.predict(x),-3,3) for m in payload['regressors']],0)
        empty=~a['observed'].any((1,2));prob[empty]=payload['class_prior'];reg[empty]=payload['target_median']
        assert np.isfinite(prob).all() and np.isfinite(reg).all() and np.allclose(prob.sum(1),1)
        for i in range(len(reg)):
            sid=path.stem if len(reg)==1 else f'{path.stem}:{i}'
            label=int(prob[i].argmax());rows.append(dict(sample_id=sid,polarity=['Negative','Neutral','Positive'][label],intensity=float(reg[i]),
                probability_negative=float(prob[i,0]),probability_neutral=float(prob[i,1]),probability_positive=float(prob[i,2])))
            provenance.append(dict(sample_id=sid,source_file=path.name,source_sha256=sha256(path),**observation_audit(a,i)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump_json(args.output.with_suffix('.provenance.json'),dict(family=family,selection_sha256=sha256(HERE/'selection.json'),
        model_sha256=sha256(model_path),prediction_sha256=sha256(args.output),rows=provenance,n_predictions=len(rows),
        note='New method experiment; no attachment3 labels available. Original MLP outputs are retained separately.'))
    print(f'Wrote {len(rows)} predictions: {args.output}',flush=True)


if __name__=='__main__':main()
