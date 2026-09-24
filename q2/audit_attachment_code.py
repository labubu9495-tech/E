"""Read-only raw-input/cache checks for the attachment-specific adapter repair.

No test labels, cross-attachment label lookup, fitting, or model selection.
Outputs are descriptive audit evidence, not a replacement experiment protocol.
"""
import argparse
import gc
from pathlib import Path
import numpy as np
from data import ROOT, adapt, load_pickle, normalize, sha256, dump_json, observation_audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=ROOT.parent/'E题')
    args=parser.parse_args()
    output=ROOT/'results'/'attachment_code_audit'
    output.mkdir(parents=True,exist_ok=True)
    candidates=list(args.data_root.rglob('aligned_50.pkl'))
    if len(candidates)!=1:raise ValueError('Ambiguous official aligned file')
    path=candidates[0]
    raw=load_pickle(path)
    stats=np.load(ROOT/'assets'/'normalization.npz')
    report={'scope':'Question2 code and input semantics; no test labels or test metrics inspected',
            'aligned_sha256':sha256(path),'splits':{}}
    for name,s in raw.items():
        tokens=np.asarray(s['text_bert'])
        a=adapt(s)
        info={'n':len(tokens),'unk_positions':int((tokens[:,0]==100).sum()),
              'all_zero_vision_samples':int((~np.any(s['vision']!=0,axis=(1,2))).sum()),
              'content_positions':int(a['sequence'].sum()),
              'endpoint_sources':sorted(set(a['endpoint_source']))}
        if name in ('train','valid'):
            old=adapt(s,text_missing_policy='ordinary_unk')
            info['legacy_adapter_unchanged']=all(np.array_equal(a[k],old[k]) for k in ['tokens','sequence','observed','audio','vision'])
            with np.load(ROOT/'cache'/f'{name}.npz') as cache:
                checks={k:np.array_equal(a[k],cache[k]) for k in ['sequence','observed']}
                for j,m in enumerate(['audio','vision'],1):
                    normalized=normalize(a[m],stats[m+'_mean'],stats[m+'_std'],a['observed'][:,:,j])
                    checks[m]=np.array_equal(normalized,cache[m])
                checks['ids']=np.array_equal(s['id'],cache['ids'])
                checks['clean_text_observed']=np.array_equal(a['observed'][:,:,0],cache['text_observed_bank'][0])
                info['cache_checks']=checks
                assert all(checks.values()),(name,checks)
            assert info['legacy_adapter_unchanged'],name
            info['label_sign_mapping']=bool(np.array_equal(np.where(s['regression_labels']<0,0,np.where(s['regression_labels']>0,2,1)),s['classification_labels']))
        if name=='train':
            errors={}
            for j,m in enumerate(['audio','vision'],1):
                x=a[m][a['observed'][:,:,j]].astype(np.float64)
                mean=x.mean(0).astype(np.float32);std=np.maximum(x.std(0),1e-5).astype(np.float32)
                errors[m]={'mean_max_abs_delta':float(np.max(abs(mean-stats[m+'_mean']))),'std_max_abs_delta':float(np.max(abs(std-stats[m+'_std'])))}
                assert np.array_equal(mean,stats[m+'_mean']) and np.array_equal(std,stats[m+'_std'])
            info['train_only_normalization_recomputed']=errors
        report['splits'][name]=info
        print(name,info,flush=True)
    del raw,s,a,old
    gc.collect()

    report['attachment3']=[]
    paths=sorted(p for p in args.data_root.rglob('*.pkl') if '附件3-' in str(p) and p.parent.name=='对齐版本')
    for path in paths:
        s=load_pickle(path)['test'];a=adapt(s);old=adapt(s,text_missing_policy='ordinary_unk')
        assert len(a['tokens'])==1
        info=dict(file=path.name,sha256=sha256(path),**observation_audit(a,0))
        info['old_false_text_observations']=int((old['observed'][:,:,0]&~a['observed'][:,:,0]).sum())
        info['unk_matches_audio_zero_in_content']=bool(np.array_equal(a['text_missing_marker'][0],a['sequence'][0]&~a['observed'][0,:,1]))
        info['audio_vision_unavailable_masks_equal']=bool(np.array_equal(a['observed'][0,:,1],a['observed'][0,:,2]))
        info['raw_token_dtype']=str(s['text_bert'].dtype)
        report['attachment3'].append(info)
    assert len(paths)==30
    report['attachment3_summary']={
        'n':len(paths),'samples_with_unk':sum(r['text_unavailable_positions']>0 for r in report['attachment3']),
        'unk_positions_total':sum(r['old_false_text_observations'] for r in report['attachment3']),
        'unk_audio_match_samples':sum(r['unk_matches_audio_zero_in_content'] for r in report['attachment3']),
        'audio_vision_match_samples':sum(r['audio_vision_unavailable_masks_equal'] for r in report['attachment3'])}
    print('attachment3',report['attachment3_summary'],flush=True)
    dump_json(output/'raw_input_checks.partial.json',report)

    report['attachment4_aligned']=[]
    paths=sorted(p for p in args.data_root.rglob('*.pkl') if '附件4-' in str(p) and p.parent.name=='对齐版本')
    for path in paths:
        s=load_pickle(path);s=s.get('test',s)
        # Attachment4 stores individual arrays without attachment3's batch
        # axis. This is a read-only audit adaptation, not a Q3 inference API.
        shapes={k:list(np.asarray(s[k]).shape) for k in ['text_bert','audio','vision']}
        batched={k:np.asarray(s[k])[None] if np.asarray(s[k]).ndim==2 else s[k] for k in shapes}
        a=adapt(batched)
        report['attachment4_aligned'].append(dict(file=path.name,original_shapes=shapes,**observation_audit(a,0)))

    path=next(args.data_root.rglob('unaligned_50.pkl'))
    raw=load_pickle(path)
    report['unaligned_visual_lengths']={}
    for name,s in raw.items():
        nz=np.any(np.asarray(s['vision'])!=0,axis=-1);lengths=np.asarray(s['vision_lengths']).reshape(-1)
        after=(nz&(np.arange(nz.shape[1])[None,:]>=lengths[:,None])).any(1)
        first_zero=np.where((~nz).any(1),(~nz).argmax(1),nz.shape[1]).clip(min=1)
        report['unaligned_visual_lengths'][name]={'n':len(nz),'samples_with_nonzero_after_declared_length':int(after.sum()),'first_zero_rule_matches':int((lengths==first_zero).sum())}
    del raw,s
    gc.collect()
    report['attachment4_unaligned_examples']=[]
    for path in sorted(p for p in args.data_root.rglob('*.pkl') if '附件4-' in str(p) and p.parent.name=='未对齐版本' and p.stem in ('13','16')):
        s=load_pickle(path);s=s.get('test',s)
        vision=np.asarray(s['vision'])
        if vision.ndim==3:vision=vision[0]
        nz=np.any(vision!=0,axis=-1)
        report['attachment4_unaligned_examples'].append(dict(file=path.name,declared_length=int(np.asarray(s['vision_lengths']).reshape(-1)[0]),nonzero_positions=np.flatnonzero(nz).tolist(),nonzero_rows=int(nz.sum())))
    dump_json(output/'raw_input_checks.json',report)
    print('Raw/cache audit passed:',output/'raw_input_checks.json',flush=True)


if __name__=='__main__':main()
