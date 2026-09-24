"""Frozen question-2 inference, also used from the self-contained submission."""
import argparse
import json
import csv
from pathlib import Path
import numpy as np
import torch
from data import ROOT,adapt,load_pickle,normalize,TextEncoder,dump_json,sha256
from models import Predictor


def load_models(selection,device):
    models=[]
    for relative in selection['checkpoints']:
        ckpt=torch.load(ROOT/relative,map_location=device,weights_only=False)
        model=Predictor(**ckpt['model_args']).to(device).eval()
        model.load_state_dict(ckpt['state_dict'])
        models.append(model)
    return models


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True,help='Directory containing official aligned attachment3 PKLs')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'attachment3_predictions.csv')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    torch.set_num_threads(2)
    selection=json.loads((ROOT/'selection.json').read_text())
    models=load_models(selection,args.device)
    stats=np.load(ROOT/'assets'/'normalization.npz')
    encoder=TextEncoder(args.device)
    files=sorted(args.input.glob('*.pkl'))
    if not files:
        raise ValueError('No .pkl files in input directory; supply aligned directory directly')
    rows=[];provenance=[]
    for path in files:
        s=load_pickle(path)
        s=s.get('test',s)
        a=adapt(s)
        text=encoder.encode(a['tokens'],a['observed'][:,:,0]).astype(np.float32)
        audio=normalize(a['audio'],stats['audio_mean'],stats['audio_std'],a['observed'][:,:,1])
        vision=normalize(a['vision'],stats['vision_mean'],stats['vision_std'],a['observed'][:,:,2])
        inputs=[torch.from_numpy(x).to(args.device) for x in [text,audio,vision,a['sequence'],a['observed']]]
        probs=[];values=[]
        for model in models:
            logits,r=model(*inputs)
            probs.append(logits.softmax(-1).cpu().numpy());values.append(r.cpu().numpy())
        probability=np.mean(probs,0);intensity=np.mean(values,0)
        if not np.isfinite(probability).all() or not np.isfinite(intensity).all():
            raise ValueError('Nonfinite predictions')
        for i in range(len(text)):
            sample_id=str(s['id'][i]) if 'id' in s else (path.stem if len(text)==1 else f'{path.stem}:{i}')
            label=int(probability[i].argmax())
            rows.append(dict(sample_id=sample_id,polarity=['Negative','Neutral','Positive'][label],intensity=float(intensity[i]),
                probability_negative=float(probability[i,0]),probability_neutral=float(probability[i,1]),probability_positive=float(probability[i,2])))
            provenance.append(dict(sample_id=sample_id,source_file=path.name,row=i,endpoint_source=a['endpoint_source'][i],
                no_observations=bool(~a['observed'][i].any())))
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate output sample IDs')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump_json(args.output.with_suffix('.provenance.json'),dict(model=selection['config'],selection_sha256=sha256(ROOT/'selection.json'),n_files=len(files),n_predictions=len(rows),rows=provenance))
    print(f'Wrote {len(rows)} predictions to {args.output}',flush=True)


if __name__=='__main__':
    main()
