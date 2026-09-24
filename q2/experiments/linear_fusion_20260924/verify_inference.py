"""Check inference features against the training cache on validation inputs."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import json
import joblib
import numpy as np
import torch
from transformers import BertTokenizerFast
from data import TextEncoder,dump_json
from features import pooled_features,visible_documents


def main():
    torch.set_num_threads(2)
    selection=json.loads((HERE/'selection.json').read_text());family=selection['selected'];payload=joblib.load(HERE/f'{family}.joblib')
    ix=np.array([0,1,2,100,400,727])
    with np.load(ROOT/'cache/valid.npz') as d:
        text=d['text_bank'][0,ix];audio=d['audio'][ix];vision=d['vision'][ix];seq=d['sequence'][ix];obs=d['observed'][ix];raw=d['raw_text'][ix]
    tokenizer=BertTokenizerFast.from_pretrained(ROOT/'assets/bert-mini');encoded=tokenizer(raw.tolist(),padding='max_length',truncation=True,max_length=50)
    tokens=np.stack([encoded['input_ids'],encoded['attention_mask'],encoded['token_type_ids']],axis=1)
    fresh=TextEncoder('cpu').encode(tokens,obs[:,:,0])
    # Different batch shapes can cause small FP16 rounding changes in BERT.
    delta=float(np.max(abs(fresh.astype(np.float32)-text.astype(np.float32))))
    assert np.allclose(fresh,text,atol=.004,rtol=.003),delta
    features=pooled_features(fresh,audio,vision,seq,obs)
    docs=visible_documents(tokens[:,0],obs[:,:,0],seq)
    x=payload['transform'].transform(features,docs,family)
    p=np.mean([m.predict_proba(x) for m in payload['classifiers']],0)
    v=np.mean([np.clip(m.predict(x),-3,3) for m in payload['regressors']],0)
    with np.load(HERE/f'{family}_validation_predictions.npz') as d:
        pdiff=float(np.max(abs(p-d['clean_probabilities'][ix])));rdiff=float(np.max(abs(v-d['clean_intensity'][ix])))
        assert np.allclose(p,d['clean_probabilities'][ix],atol=.003)
        assert np.allclose(v,d['clean_intensity'][ix],atol=.003)
        assert np.array_equal(p.argmax(1),d['clean_probabilities'][ix].argmax(1))
    result={'validation_rows_checked':ix.tolist(),'fresh_BERT_vs_cached_max_abs_delta':delta,'probability_max_abs_delta':pdiff,'intensity_max_abs_delta':rdiff,'classifications_identical':True}
    dump_json(HERE/'inference_verification.json',result);print(json.dumps(result))


if __name__=='__main__':main()
