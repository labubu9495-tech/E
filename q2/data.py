"""Official aligned input adapter. Fit all statistics on train only."""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ENCODER_ID = 'google/bert_uncased_L-4_H-256_A-4'
MODALITIES = ('text', 'audio', 'vision')
TEXT_MISSING_POLICY = 'official_aligned_unk_v1'


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def load_pickle(path):
    # Only use with the competition files supplied by the user.
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def integer_tokens(value):
    a = np.asarray(value)
    if a.ndim != 3 or a.shape[1:] != (3, 50):
        raise ValueError(f'Unexpected text_bert shape: {a.shape}')
    if not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError('Non-finite or noninteger BERT inputs')
    a = a.astype(np.int64)
    if not np.isin(a[:, 1], [0, 1]).all() or not np.isin(a[:, 2], [0, 1]).all():
        raise ValueError('Invalid attention mask or segment IDs')
    if (a[:, 0] < 0).any() or (a[:, 0] >= 30522).any():
        raise ValueError('Token IDs incompatible with uncased BERT vocabulary')
    return a


def adapt(split, *, text_missing_policy=TEXT_MISSING_POLICY):
    # Dataset-specific convention: token 100 marks unavailable content in
    # the official aligned attachment3 files. In general BERT data it is
    # merely an unknown word; callers can explicitly use ordinary_unk.
    if text_missing_policy not in (TEXT_MISSING_POLICY, 'ordinary_unk'):
        raise ValueError(f'Unknown text missing policy: {text_missing_policy}')
    tokens = integer_tokens(split['text_bert'])
    audio = np.asarray(split['audio'], dtype=np.float32)
    vision = np.asarray(split['vision'], dtype=np.float32)
    n, _, length = tokens.shape
    if audio.shape != (n, length, 74) or vision.shape != (n, length, 35):
        raise ValueError('Expected aligned audio/vision interface')
    ids, attention = tokens[:, 0], tokens[:, 1].astype(bool)
    # Preserve positions even when the three token channels are zero internally.
    # A retained SEP anchors the endpoint. Without it use observed support's
    # last index, explicitly flagging that joint trailing loss is unidentifiable.
    sequence = np.zeros((n, length), dtype=bool)
    endpoint_source = []
    av_support = np.any(np.isfinite(audio) & (audio != 0), -1) | np.any(np.isfinite(vision) & (vision != 0), -1)
    for i in range(n):
        sep = np.flatnonzero(ids[i] == 102)
        if len(sep):
            end = int(sep[-1])
            endpoint_source.append('SEP')
        else:
            evidence = np.flatnonzero(attention[i] | av_support[i])
            end = int(evidence[-1] + 1) if len(evidence) else 1
            endpoint_source.append('observed_support_approximate')
        start = 1 if ids[i, 0] in (0, 101) else 0
        sequence[i, start:end] = True
    text_missing_marker = sequence & (ids == 100) if text_missing_policy == TEXT_MISSING_POLICY else np.zeros_like(sequence)
    text_seen = sequence & attention & ~np.isin(ids, [0, 101, 102]) & ~text_missing_marker
    audio_seen = sequence & np.isfinite(audio).all(-1) & np.any(audio != 0, -1)
    vision_seen = sequence & np.isfinite(vision).all(-1) & np.any(vision != 0, -1)
    obs = np.stack([text_seen, audio_seen, vision_seen], -1)
    # Zero-valued A/V rows are operationally treated as unavailable; their
    # provenance (normal zero / extraction failure / imposed loss) is unknown.
    audio = np.where(audio_seen[..., None], np.nan_to_num(audio), 0)
    vision = np.where(vision_seen[..., None], np.nan_to_num(vision), 0)
    return dict(tokens=tokens, audio=audio, vision=vision, sequence=sequence,
                observed=obs, endpoint_source=endpoint_source,
                padding=~attention & ~sequence, special=np.isin(ids, [101, 102]),
                text_missing_marker=text_missing_marker, text_missing_policy=text_missing_policy,
                nonfinite_rows=np.stack([~np.isfinite(np.asarray(split[m])).all(-1) for m in ['audio','vision']], -1))


def interval_mask(sequence, ratio, location='random', rng=None):
    if not np.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('Missing ratio must be finite and within [0, 1]')
    if location not in ('random', 'front', 'middle', 'back'):
        raise ValueError(f'Unknown interval location: {location}')
    rng = np.random.default_rng(0) if rng is None else rng
    result = np.zeros_like(sequence)
    if ratio == 0:
        return result
    for i, p in enumerate(sequence):
        positions = np.flatnonzero(p)
        if not len(positions):
            continue
        width = max(1, min(len(positions), int(round(len(positions) * ratio))))
        last = len(positions) - width
        start = {'front': 0, 'middle': last // 2, 'back': last}.get(location)
        if start is None:
            start = int(rng.integers(last + 1))
        result[i, positions[start:start + width]] = True
    return result


def position_intervals(mask):
    """Zero-based stored positions, half-open [start, end), never seconds."""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    edges = np.diff(padded)
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def observation_audit(adapted, row):
    sequence = adapted['sequence'][row]
    n = int(sequence.sum())
    result = dict(content_positions=n, text_missing_policy=adapted['text_missing_policy'],
        text_unk_content_positions=np.flatnonzero(adapted['text_missing_marker'][row]).tolist(),
        padding_positions=np.flatnonzero(adapted['padding'][row]).tolist(),
        special_positions=np.flatnonzero(adapted['special'][row]).tolist(),
        interval_coordinate='zero-based stored sequence positions; [start,end); not seconds',
        missing_provenance='Unavailable observations; natural extraction failure and imposed A/V loss are not distinguishable')
    flags=[]
    for j, m in enumerate(MODALITIES):
        unseen=sequence & ~adapted['observed'][row,:,j]
        result[m+'_unavailable_positions']=int(unseen.sum())
        result[m+'_unavailable_fraction']=float(unseen.sum()/n) if n else None
        result[m+'_unavailable_intervals']=position_intervals(unseen)
        if n and not adapted['observed'][row,:,j].any():
            flags.append(m+'_entirely_unavailable')
    if adapted['endpoint_source'][row]!='SEP':flags.append('approximate_endpoint')
    if not n:flags.append('empty_content')
    if adapted['nonfinite_rows'][row].any():flags.append('nonfinite_av_rows')
    result['quality_flags']=flags
    return result


def conditions():
    result = [dict(name='clean', modalities=[], ratio=0., location='none')]
    for m in range(3):
        for ratio in [.1, .3, .5]:
            for loc in ['front', 'middle', 'back']:
                result.append(dict(name=f'{MODALITIES[m]}_{int(ratio*100)}_{loc}',
                                   modalities=[m], ratio=ratio, location=loc))
    result += [dict(name='audio_vision_30_sync', modalities=[1, 2], ratio=.3, location='middle'),
               dict(name='text_audio_30_sync', modalities=[0, 1], ratio=.3, location='middle'),
               dict(name='audio_vision_30_offset', modalities=[1, 2], ratio=.3, location='offset'),
               dict(name='audio_30_two_blocks', modalities=[1], ratio=.3, location='two_blocks')]
    return result


def condition_observed(adapted, condition):
    result = adapted['observed'].copy()
    sequence = adapted['sequence']
    for j, m in enumerate(condition['modalities']):
        loc = condition['location']
        if loc == 'offset':
            loc = 'front' if j == 0 else 'back'
        if loc == 'two_blocks':
            mask = np.zeros_like(sequence)
            for i, p in enumerate(sequence):
                pos = np.flatnonzero(p)
                total = min(len(pos), max(1, round(len(pos)*condition['ratio'])))
                first = total // 2
                if first:
                    mask[i, pos[:first]] = True
                if total-first:
                    mask[i, pos[-(total-first):]] = True
        else:
            mask = interval_mask(sequence, condition['ratio'], loc)
        result[:, :, m] &= ~mask
    return result


class TextEncoder:
    def __init__(self, device='cuda'):
        from transformers import BertModel
        self.device = device
        self.path = ROOT / 'assets' / 'bert-mini'
        self.model = BertModel.from_pretrained(self.path).eval().to(device)
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, tokens, observed, batch_size=128):
        parts = []
        for start in range(0, len(tokens), batch_size):
            t = tokens[start:start+batch_size].copy()
            seen = observed[start:start+batch_size]
            # Keep special tokens as context anchors; erase missing content
            # before BERT so no unmasked representation sees removed words.
            content = ~np.isin(t[:, 0], [0, 101, 102])
            lost = content & ~seen
            t[:, 0][lost] = 0
            t[:, 1][lost] = 0
            t[:, 2][lost] = 0
            x = torch.from_numpy(t).to(self.device)
            out = self.model(input_ids=x[:, 0], attention_mask=x[:, 1],
                             token_type_ids=x[:, 2]).last_hidden_state
            out = out * torch.as_tensor(seen, device=self.device).unsqueeze(-1)
            parts.append(out.cpu().numpy().astype(np.float16))
        return np.concatenate(parts)


def normalize(values, mean, std, observed):
    return np.where(observed[..., None], (values.astype(np.float32)-mean)/std, 0).astype(np.float32)


def to_device(cache, device):
    result = {}
    for k in ['text_bank', 'audio', 'vision', 'sequence', 'observed', 'text_observed_bank', 'labels', 'targets']:
        if k in cache:
            value = torch.from_numpy(cache[k])
            if value.dtype == torch.float16:
                value = value.float()
            result[k] = value.to(device)
    return result


def missing_runs(observed, sequence):
    d = torch.zeros_like(observed[:, 0], dtype=torch.float32)
    values = []
    for t in range(observed.shape[1]):
        p = sequence[:, t, None]
        d = torch.where(p, torch.where(observed[:, t], 0., d+1), d)
        values.append(torch.where(p, d, 0.))
    return torch.log1p(torch.stack(values, 1))
