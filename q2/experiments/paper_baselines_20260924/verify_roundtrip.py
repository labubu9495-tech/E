"""Check re-encoded visible text and restored checkpoints against saved validation."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import BertTokenizerFast

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from data import TextEncoder,dump_json
from train import text_bank_index
from paper_models import make_model


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    cache=np.load(ROOT/'cache/valid.npz')
    ix=np.array([0,9,51,101,305,727])
    tokenizer=BertTokenizerFast.from_pretrained(ROOT/'assets/bert-mini')
    tok=tokenizer(cache['raw_text'][ix].tolist(),padding='max_length',truncation=True,max_length=50)
    tokens=np.stack([tok['input_ids'],tok['attention_mask'],tok['token_type_ids']],axis=1)
    encoder=TextEncoder('cpu')
    results=[]
    masks=np.load(ROOT/'cache/valid_conditions.npz')
    for c in ['clean','text_30_middle']:
        obs=masks[c][ix]
        fresh=encoder.encode(tokens,obs[...,0]).astype(np.float32)
        bank=0 if c=='clean' else 5
        expected=cache['text_bank'][bank,ix].astype(np.float32)
        max_text_error=float(np.max(abs(fresh-expected)))
        # CPU/GPU BERT FP16 cache rounding can differ in rare values.
        assert np.allclose(fresh,expected,atol=.004,rtol=.003),max_text_error
        inputs=[torch.from_numpy(x) for x in [fresh,cache['audio'][ix],cache['vision'][ix],cache['sequence'][ix],obs]]
        for name in ['mult','emt_dlfr','emt_no_restore']:
            p=HERE/'runs'/f'{name}_seed17'
            ckpt=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
            model=make_model(name,**ckpt['model_args']).eval()
            model.load_state_dict(ckpt['state_dict'])
            logits,reg=model(*inputs);prob=logits.softmax(-1).numpy();reg=reg.numpy()
            with np.load(p/'validation_predictions.npz') as saved:
                p0=saved[c+'_probabilities'][ix];r0=saved[c+'_intensity'][ix]
            pe=float(np.max(abs(prob-p0)));re=float(np.max(abs(reg-r0)))
            assert np.array_equal(prob.argmax(-1),p0.argmax(-1)),(name,c)
            assert pe<.002 and re<.003,(name,c,pe,re)
            results.append(dict(model=name,condition=c,n=len(ix),max_text_error=max_text_error,
                max_probability_error=pe,max_intensity_error=re,classes_match=True))
    dump_json(HERE/'roundtrip_verification.json',dict(passed=True,checks=results,
        scope='Six validation rows; frozen tokenizer re-encoding, audited A/V cache and checkpoint reload; CPU vs GPU tolerances explicit.'))
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
