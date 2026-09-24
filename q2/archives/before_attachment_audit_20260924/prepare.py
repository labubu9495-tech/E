"""Audit train/valid, freeze the experimental protocol and cache frozen BERT."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from data import ROOT, ENCODER_ID, adapt, load_pickle, sha256, dump_json, TextEncoder, interval_mask, conditions, condition_observed, normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT.parent/'E题')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cache_dir = ROOT/'cache'
    cache_dir.mkdir(exist_ok=True)
    official = next(args.data_root.rglob('aligned_50.pkl'))
    data = load_pickle(official)
    audit = {'official_sha256': sha256(official), 'splits': {}, 'label_mapping': {'0': 'Negative', '1': 'Neutral', '2': 'Positive'}}
    for split in ['train', 'valid']:
        s = data[split]
        a = adapt(s)
        y = s['regression_labels']
        expected = np.where(y < 0, 0, np.where(y > 0, 2, 1))
        assert np.array_equal(expected, s['classification_labels'])
        assert len(set(s['id'])) == len(s['id'])
        audit['splits'][split] = dict(n=len(y), class_counts=np.bincount(expected, minlength=3).tolist(),
            intensity_range=[float(y.min()), float(y.max())], token_shape=list(s['text_bert'].shape),
            sequence_length_quantiles=np.quantile(a['sequence'].sum(1), [0,.25,.5,.75,1]).tolist(),
            zero_rows_within_content={m:int((a['sequence'] & ~a['observed'][:,:,j]).sum()) for j,m in enumerate(['text','audio','vision'])},
            nonfinite={m:int((~np.isfinite(s[m])).sum()) for m in ['audio','vision']},
            endpoint_sources=sorted(set(a['endpoint_source'])))
    groups = {s: {x.split('$_$')[0] for x in data[s]['id']} for s in data}
    overlaps = {f'{a}_{b}': len(groups[a]&groups[b]) for a,b in [('train','valid'),('train','test'),('valid','test')]}
    audit['video_group_overlaps'] = overlaps
    assert not any(overlaps.values())
    # Test labels and test performance are deliberately not examined.
    audit['test_n_only'] = len(data['test']['id'])
    del data['test']
    special = sorted(p for p in args.data_root.rglob('*.pkl') if '附件3-' in str(p) and p.parent.name=='对齐版本')
    schemas = []
    for p in special:
        s = load_pickle(p)['test']
        # Input interface checks only: no missing-frequency summaries or fitting.
        adapt(s)
        schemas.append(dict(file=p.name, fields={k:list(v.shape) for k,v in s.items() if hasattr(v,'shape')}))
    audit['special_interface_only'] = schemas
    audit['zero_policy'] = 'Within SEP-defined content, finite nonzero A/V rows are operational observations. Natural zeros and artificial loss cannot be distinguished. Padding is excluded separately.'
    dump_json(ROOT/'results'/'audit.json', audit)
    print('Audit passed; train/valid and special interfaces verified.', flush=True)

    from transformers import BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained(ROOT/'assets'/'bert-mini')
    checks = []
    for split in ['train','valid']:
        ids = tok(data[split]['raw_text'].tolist(), padding='max_length', truncation=True, max_length=50)['input_ids']
        match = np.all(np.asarray(ids) == data[split]['text_bert'][:,0], axis=1)
        checks.append({'split':split,'matching_tokenizations':int(match.sum()),'n':len(match)})
    audit['tokenizer_verification'] = checks
    dump_json(ROOT/'results'/'audit.json', audit)
    if any(x['matching_tokenizations'] < .95*x['n'] for x in checks):
        raise RuntimeError('Tokenizer mismatch: resolve before encoding.')

    protocol = dict(encoder=ENCODER_ID, feature_version='aligned_50', seed_list=[17,29,43],
        hidden=32, batch_size=64, max_epochs=40, patience=6, learning_rate=.001,
        weight_decay=.0001, dropout=.2, regression_weight=1.,
        augmentation_probabilities=[.4,.4,.2], missing_fraction=[.1,.5],
        training_text_bank_variants=8,
        selection='Maximum mean macro-F1 on clean plus T/A/V 30%-middle validation; tie-break lower average MAE. Selection validation is not independent test evidence.',
        conditions=conditions(),
        limitations=['Missing duration is sequence positions, not seconds.',
                      'Artificial raw-token loss is applied before BERT; official A/V zero rows have unknown provenance.',
                      'No attachment3 distribution fitting; attachment2 test remains untouched.',
                      'No timestamp reconstruction needed for question2.'])
    dump_json(ROOT/'protocol.json', protocol)
    encoder = TextEncoder(args.device)
    train = adapt(data['train'])
    stats = {}
    for j,m in enumerate(['audio','vision'],1):
        vals = train[m][train['observed'][:,:,j]].astype(np.float64)
        stats[m+'_mean'] = vals.mean(0).astype(np.float32)
        stats[m+'_std'] = np.maximum(vals.std(0),1e-5).astype(np.float32)
    stats['class_prior'] = np.bincount(data['train']['classification_labels'].astype(int), minlength=3).astype(np.float32)/len(train['tokens'])
    stats['target_median'] = np.array(np.median(data['train']['regression_labels']), dtype=np.float32)
    np.savez(ROOT/'assets'/'normalization.npz', **stats)
    rng = np.random.default_rng(20260923)
    for split in ['train','valid']:
        s = data[split]
        a = adapt(s)
        bank_obs = [a['observed'][:,:,0]]
        if split=='train':
            for v in range(protocol['training_text_bank_variants']):
                mask = np.zeros_like(a['sequence'])
                for i in range(len(mask)):
                    mask[i:i+1] = interval_mask(a['sequence'][i:i+1], rng.uniform(.1,.5), rng=rng)
                bank_obs.append(a['observed'][:,:,0] & ~mask)
        else:
            for ratio in [.1,.3,.5]:
                for loc in ['front','middle','back']:
                    bank_obs.append(a['observed'][:,:,0] & ~interval_mask(a['sequence'], ratio, loc))
        bank=[]
        for i,seen in enumerate(bank_obs):
            print(f'Encoding {split} view {i+1}/{len(bank_obs)}',flush=True)
            bank.append(encoder.encode(a['tokens'],seen))
        out=dict(text_bank=np.stack(bank), text_observed_bank=np.stack(bank_obs),
            audio=normalize(a['audio'],stats['audio_mean'],stats['audio_std'],a['observed'][:,:,1]),
            vision=normalize(a['vision'],stats['vision_mean'],stats['vision_std'],a['observed'][:,:,2]),
            sequence=a['sequence'], observed=a['observed'],
            labels=s['classification_labels'].astype(np.int64),targets=s['regression_labels'].astype(np.float32),
            ids=np.array(s['id']), raw_text=s['raw_text'])
        np.savez(cache_dir/f'{split}.npz', **out)
        if split=='valid':
            np.savez(cache_dir/'valid_conditions.npz', **{c['name']:condition_observed(a,c) for c in conditions()})
    dump_json(cache_dir/'manifest.json', dict(official_sha256=audit['official_sha256'], protocol_sha256=sha256(ROOT/'protocol.json'), encoder_sha256=sha256(ROOT/'assets'/'bert-mini'/'model.safetensors')))
    print('Prepared all train/valid caches. No special predictions generated.',flush=True)


if __name__=='__main__':
    main()
