"""Bounded, predeclared alternative experiment; never fits on special samples."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import csv
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor,as_completed
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.exceptions import ConvergenceWarning
from transformers import BertTokenizerFast
from scipy import sparse
from data import conditions,dump_json,sha256
from train import metrics,SELECT,text_bank_index
from features import FeatureTransform,cache_features,augmented_views


def table(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def load(split,tokenizer):
    with np.load(ROOT/'cache'/f'{split}.npz') as source:
        d={k:source[k] for k in ['text_bank','text_observed_bank','audio','vision','sequence','observed','labels','targets','ids','raw_text']}
    d['token_ids']=np.asarray(tokenizer(d['raw_text'].tolist(),padding='max_length',truncation=True,max_length=50)['input_ids'],dtype=np.int64)
    # prepare.py verified this tokenizer against every official train/valid row.
    verification=json.loads((ROOT/'results/audit.json').read_text())['tokenizer_verification']
    check=next(x for x in verification if x['split']==split)
    assert check['matching_tokenizations']==len(d['labels'])==check['n']
    assert not ((d['token_ids']==100)&d['sequence']).any()
    return d


def main():
    import torch
    torch.set_num_threads(2)
    plan=json.loads((HERE/'protocol.json').read_text());(HERE/'runs').mkdir(exist_ok=True)
    started=time.time()
    protected={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'selection.json',ROOT/'results/attachment3_predictions.csv',ROOT/'results/selected_validation_predictions.npz']}
    dump_json(HERE/'source_fingerprints.json',dict(protocol_sha256=sha256(HERE/'protocol.json'),protected=protected,
        code_sha256={p.name:sha256(p) for p in [HERE/'features.py',HERE/'run.py']}))
    tokenizer=BertTokenizerFast.from_pretrained(ROOT/'assets/bert-mini')
    train=load('train',tokenizer);valid=load('valid',tokenizer)
    with np.load(ROOT/'cache/valid_conditions.npz') as d:masks=dict(d)
    clean,docs=cache_features(train,np.zeros(len(train['labels']),np.int64),train['observed'])
    transform=FeatureTransform().fit(clean,docs)
    joblib.dump(transform,HERE/'feature_transform.joblib',compress=3)
    print('Prepared',len(train['labels']),'train,',len(valid['labels']),'valid; lexical vocabulary',len(transform.tfidf.vocabulary_),flush=True)
    val_features={}
    for c in conditions():
        banks=np.full(len(valid['labels']),text_bank_index(c),dtype=np.int64)
        val_features[c['name']]=cache_features(valid,banks,masks[c['name']])
    select_x={family:{c:transform.transform(*val_features[c],family) for c in SELECT} for family in plan['families']}
    def fit_seed(seed):
        candidate_rows=[];reg_rows=[]
        dense=[];documents=[];weight=[]
        for banks,obs,w in augmented_views(train,seed):
            z,t=cache_features(train,banks,obs);dense.append(z);documents.extend(t);weight.extend([w]*len(z))
        dense=np.concatenate(dense);weight=np.asarray(weight)
        y=np.tile(train['labels'],5);target=np.tile(train['targets'],5)
        for family in plan['families']:
            x=transform.transform(dense,documents,family)
            for alpha in plan['regression']['alpha']:
                path=HERE/'runs'/f'{family}_seed{seed}_reg_{alpha:g}.joblib'
                if path.exists():model=joblib.load(path)
                else:
                    model=Ridge(alpha=alpha,solver='lsqr',tol=1e-5,max_iter=2000)
                    model.fit(x,target,sample_weight=weight);joblib.dump(model,path,compress=3)
                values={c:np.clip(model.predict(select_x[family][c]),-3,3) for c in SELECT}
                maes=[float(np.abs(values[c]-valid['targets']).mean()) for c in SELECT]
                reg_rows.append(dict(family=family,seed=seed,alpha=alpha,selection_mae=float(np.mean(maes))))
                print(f'{family} seed={seed} ridge alpha={alpha:g} MAE={np.mean(maes):.4f}',flush=True)
            for C in plan['classification']['C']:
                for class_weight in plan['classification']['class_weight']:
                    key='balanced' if class_weight else 'none';path=HERE/'runs'/f'{family}_seed{seed}_clf_{C:g}_{key}.joblib'
                    if path.exists():model=joblib.load(path);converged=bool(np.max(model.n_iter_)<plan['classification']['max_iter'])
                    else:
                        model=LogisticRegression(C=C,class_weight=class_weight,solver='lbfgs',max_iter=plan['classification']['max_iter'],tol=1e-4)
                        model.fit(x,y,sample_weight=weight)
                        converged=bool(np.max(model.n_iter_)<plan['classification']['max_iter'])
                        joblib.dump(model,path,compress=3)
                    scores=[]
                    for c in SELECT:
                        prob=model.predict_proba(select_x[family][c]);s=metrics(valid['labels'],prob.argmax(1),valid['targets'],np.zeros(len(prob)))
                        scores.append(s)
                    row=dict(family=family,seed=seed,C=C,class_weight=key,selection_macro_f1=float(np.mean([s['macro_f1'] for s in scores])),clean_macro_f1=scores[0]['macro_f1'],converged=converged,max_iterations=int(np.max(model.n_iter_)))
                    candidate_rows.append(row);print(row,flush=True)
            del x
        return candidate_rows,reg_rows
    candidate_rows=[];reg_rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(fit_seed,seed) for seed in plan['seeds']]
        for future in as_completed(futures):
            cr,rr=future.result();candidate_rows.extend(cr);reg_rows.extend(rr)
            table(HERE/'classifier_candidates.csv',candidate_rows);table(HERE/'regressor_candidates.csv',reg_rows)
    if not all(r['converged'] for r in candidate_rows):raise RuntimeError('Some classifiers did not converge; inspect before selecting')
    decisions=[];outputs={}
    for family in plan['families']:
        alpha=min(plan['regression']['alpha'],key=lambda a:np.mean([r['selection_mae'] for r in reg_rows if r['family']==family and r['alpha']==a]))
        candidates=[]
        for C in plan['classification']['C']:
            for cw in ['none','balanced']:
                vals=[r['selection_macro_f1'] for r in candidate_rows if r['family']==family and r['C']==C and r['class_weight']==cw]
                candidates.append((float(np.mean(vals)),float(np.std(vals,ddof=1)),C,cw))
        score,sd,C,cw=max(candidates,key=lambda x:x[0])
        clfs=[joblib.load(HERE/'runs'/f'{family}_seed{s}_clf_{C:g}_{cw}.joblib') for s in plan['seeds']]
        regs=[joblib.load(HERE/'runs'/f'{family}_seed{s}_reg_{alpha:g}.joblib') for s in plan['seeds']]
        decision=dict(family=family,C=C,class_weight=cw,alpha=alpha,selection_macro_f1_seed_mean=score,selection_macro_f1_seed_std=sd,
                      selection_mae_seed_mean=float(np.mean([r['selection_mae'] for r in reg_rows if r['family']==family and r['alpha']==alpha])))
        decisions.append(decision)
        payload=dict(family=family,transform=transform,classifiers=clfs,regressors=regs,selection=decision,
                     class_prior=np.bincount(train['labels'],minlength=3)/len(train['labels']),target_median=float(np.median(train['targets'])))
        joblib.dump(payload,HERE/f'{family}.joblib',compress=3)
        predictions={'ids':valid['ids'],'labels':valid['labels'],'targets':valid['targets']};scores={}
        for c in conditions():
            name=c['name'];x=transform.transform(*val_features[name],family)
            prob=np.mean([m.predict_proba(x) for m in clfs],0);reg=np.mean([np.clip(m.predict(x),-3,3) for m in regs],0)
            empty=~masks[name].any((1,2));prob[empty]=payload['class_prior'];reg[empty]=payload['target_median']
            predictions[name+'_probabilities']=prob;predictions[name+'_intensity']=reg
            scores[name]=metrics(valid['labels'],prob.argmax(1),valid['targets'],reg)
        np.savez(HERE/f'{family}_validation_predictions.npz',**predictions)
        dump_json(HERE/f'{family}_metrics.json',scores);outputs[family]=scores
    decisions.sort(key=lambda d:(-d['selection_macro_f1_seed_mean'],d['selection_mae_seed_mean']))
    dump_json(HERE/'selection.json',dict(selected=decisions[0]['family'],ranking=decisions,protocol_sha256=sha256(HERE/'protocol.json'),
        model_sha256={f:sha256(HERE/f'{f}.joblib') for f in plan['families']},note='New family/hyperparameter selection on development validation; not an independent test.'))
    baseline=json.loads((ROOT/'results/selected_validation_metrics.json').read_text())
    rows=[]
    for family,result in [('original_mlp_ensemble',baseline)]+list(outputs.items()):
        for scope,names in [('clean',['clean']),('selection',SELECT),('grid27',[c['name'] for c in conditions()[1:28]])]:
            rows.append(dict(method=family,scope=scope,**{m:float(np.mean([result[c][m] for c in names])) for m in ['accuracy','macro_f1','mae','pearson']}))
    table(HERE/'ensemble_comparison.csv',rows)
    assert all(sha256(ROOT/p)==digest for p,digest in protected.items())
    dump_json(HERE/'verification.json',dict(original_outputs_unchanged=True,all_classifiers_converged=True,selected_family=decisions[0]['family'],
        elapsed_seconds=time.time()-started,train_samples=len(train['labels']),validation_samples=len(valid['labels']),test_labels_used=False,
        attachment3_used_for_selection=False))
    print('FINAL',json.dumps(dict(selection=decisions,comparison=rows),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
