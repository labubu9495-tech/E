"""Competition adaptations of MulT (ACL 2019) and EMT-DLFR (TAC 2023).

Architectures follow pinned official sources in vendor/. See SOURCES.md and
LICENSES/ for attribution and a complete description of intentional changes.
Inputs are the common frozen BERT-Mini / official aligned A/V representations.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(x, mask):
    return (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class Position(nn.Module):
    def __init__(self, dim, dropout=0., scale=False):
        super().__init__()
        pos = torch.arange(1024).float()[:, None]
        freq = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.) / dim))
        pe = torch.zeros(1024, dim)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq[:dim // 2])
        self.register_buffer('pe', pe[None])
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(dim) if scale else 1.

    def forward(self, x, sequence):
        # Stored positions are retained across internal holes. Never infer
        # position/padding from one projected feature channel, as old MulT did.
        return self.dropout(x * self.scale + self.pe[:, :x.shape[1]]) * sequence[..., None]


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0., bias=False):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.d = heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)
        self.v = nn.Linear(dim, dim, bias=bias)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x, context, mask):
        b, t, _ = x.shape
        shape = lambda z: z.reshape(b, -1, self.heads, self.d).transpose(1, 2)
        q, k, v = shape(self.q(x)), shape(self.k(context)), shape(self.v(context))
        # PyTorch SDPA returns zero for an entirely masked context. Explicit
        # gating also removes the output bias for those rows.
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.)
        h = h.transpose(1, 2).reshape(b, t, -1)
        return self.out(h) * mask.any(1)[:, None, None]


class CrossLayer(nn.Module):
    def __init__(self, dim, heads, attn_dropout=.1, dropout=.1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, attn_dropout, bias=True)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, sequence, context_mask):
        y = self.norm1(x)
        ctx = y if context is None else self.norm1(context)
        x = x + self.dropout(self.attention(y, ctx, context_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x * sequence[..., None]


class MulTStack(nn.Module):
    def __init__(self, dim, heads, depth, attn_dropout):
        super().__init__()
        self.position = Position(dim, dropout=.25, scale=True)
        self.layers = nn.ModuleList([CrossLayer(dim, heads, attn_dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, sequence, context=None, context_mask=None):
        x = self.position(x, sequence)
        if context is not None:
            context = self.position(context, context_mask)
        else:
            context_mask = sequence
        for layer in self.layers:
            x = layer(x, context, sequence, context_mask)
        return self.norm(x) * sequence[..., None]


class OutputBase(nn.Module):
    def __init__(self, class_prior, target_median):
        super().__init__()
        self.register_buffer('class_prior', torch.tensor(class_prior, dtype=torch.float32))
        self.register_buffer('target_median', torch.tensor(target_median, dtype=torch.float32))

    def finish(self, logits, reg, observed):
        reg = 3 * torch.tanh(reg.squeeze(-1) / 3)
        empty = ~observed.flatten(1).any(1)
        logits = torch.where(empty[:, None], self.class_prior.clamp_min(1e-8).log()[None], logits)
        return logits, torch.where(empty, self.target_median, reg)

    @staticmethod
    def sanitize(text, audio, vision, sequence, observed):
        observed = observed & sequence[..., None]
        xs = [torch.where(observed[..., i, None], x, 0.) for i, x in enumerate((text, audio, vision))]
        return xs, observed


class MulT(OutputBase):
    def __init__(self, class_prior=(.3,.2,.5), target_median=0., dim=30, heads=5, depth=5):
        super().__init__(class_prior, target_median)
        self.projections = nn.ModuleList([nn.Linear(d, dim, bias=False) for d in (256, 74, 35)])
        self.cross = nn.ModuleDict({f'{t}_{s}': MulTStack(dim, heads, depth, .1 if s == 0 else 0.)
                                   for t in range(3) for s in range(3) if s != t})
        self.memory = nn.ModuleList([MulTStack(2 * dim, heads, max(3, depth), .1) for _ in range(3)])
        self.residual = nn.Sequential(nn.Linear(6 * dim, 6 * dim), nn.ReLU(), nn.Linear(6 * dim, 6 * dim))
        self.classifier, self.regressor = nn.Linear(6 * dim, 3), nn.Linear(6 * dim, 1)

    def forward(self, text, audio, vision, sequence, observed):
        xs, observed = self.sanitize(text, audio, vision, sequence, observed)
        xs = [p(x) for p, x in zip(self.projections, xs)]
        pooled = []
        # Every genuine content position remains a query, even if its target
        # modality is unavailable. Only observed source rows act as cross keys.
        last = (torch.arange(sequence.shape[1], device=text.device)[None] * sequence).max(1).values
        batch = torch.arange(len(text), device=text.device)
        for t in range(3):
            h = torch.cat([self.cross[f'{t}_{s}'](xs[t], sequence, xs[s], observed[..., s])
                           for s in range(3) if s != t], -1)
            h = self.memory[t](h, sequence)
            pooled.append(h[batch, last])
        feature = torch.cat(pooled, -1)
        feature = feature + self.residual(feature)
        return self.finish(self.classifier(feature), self.regressor(feature), observed)


class GEGLU(nn.Module):
    def forward(self, x):
        value, gate = x.chunk(2, -1)
        return value * F.gelu(gate)


class MPUHalf(nn.Module):
    """Official EMT: pre-norm cross attention, self attention, GEGLU FFN."""
    def __init__(self, dim, heads):
        super().__init__()
        self.cross, self.self_attn = Attention(dim, heads), Attention(dim, heads)
        self.qnorm, self.knorm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.snorm, self.fnorm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 8), GEGLU(), nn.Linear(dim * 4, dim))

    def forward(self, x, context, mask, context_mask):
        x = x + self.cross(self.qnorm(x), self.knorm(context), context_mask)
        normalized = self.snorm(x)
        x = x + self.self_attn(normalized, normalized, mask)
        return (x + self.ff(self.fnorm(x))) * mask[..., None]


class EMTFusion(nn.Module):
    def __init__(self, dim=128, heads=4, depth=2):
        super().__init__()
        self.depth = depth
        self.position = Position(dim)
        # MOSEI official settings share MPU across directions, modalities,
        # and layers. Attention pooling is also shared across layers.
        self.mpu = MPUHalf(dim, heads)
        self.pool_score = nn.Sequential(nn.Linear(3 * dim, 3 * dim), nn.Tanh(), nn.Linear(3 * dim, 1))

    def forward(self, global_tokens, local, sequence):
        local = [self.position(x, sequence) for x in local]
        gm = torch.ones(global_tokens.shape[:2], dtype=torch.bool, device=global_tokens.device)
        for _ in range(self.depth):
            contexts, updated = [], []
            for x in local:
                updated.append(self.mpu(x, global_tokens, sequence, gm))
                contexts.append(self.mpu(global_tokens, x, gm, sequence))
            choices = torch.stack(contexts, 1).flatten(2)
            weight = self.pool_score(choices).softmax(1)
            global_tokens = (choices * weight).sum(1).reshape_as(global_tokens)
            local = updated
        return global_tokens, local


class AVEncoder(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, batch_first=True)

    def forward(self, x, sequence, observed):
        # Move only CLS/SEP/padding to suffix; internal missing locations stay
        # in order and consume steps. Restore original positions afterward.
        order = (~sequence).long().argsort(dim=1, stable=True)
        compact = x.gather(1, order[..., None].expand_as(x))
        h, _ = self.rnn(compact)
        lengths = sequence.sum(1).clamp_min(1)
        global_h = h[torch.arange(len(x), device=x.device), lengths - 1]
        global_h = global_h * observed.any(1)[:, None]
        restored = torch.zeros_like(h).scatter(1, order[..., None].expand_as(h), h)
        return restored * sequence[..., None], global_h


class SiameseHead(nn.Module):
    def __init__(self, dim, predictor_dim):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim, affine=False))
        self.predictor = nn.Sequential(nn.Linear(dim, predictor_dim, bias=False), nn.BatchNorm1d(predictor_dim),
                                       nn.ReLU(), nn.Linear(predictor_dim, dim))

    def forward(self, x):
        z = self.projector(x)
        return self.predictor(z), z.detach()


class EMTDLFR(OutputBase):
    def __init__(self, class_prior=(.3,.2,.5), target_median=0., dim=128, heads=4, depth=2):
        super().__init__(class_prior, target_median)
        self.audio_encoder, self.vision_encoder = AVEncoder(74, 16), AVEncoder(35, 32)
        self.projections = nn.ModuleList([nn.Linear(d, dim, bias=False) for d in (256, 16, 32)])
        self.fusion = EMTFusion(dim, heads, depth)
        self.siamese = nn.ModuleList([SiameseHead(d, p) for d, p in [(3 * dim, 128), (256, 256), (16, 8), (32, 16)]])
        self.reconstruction = nn.ModuleList([nn.Linear(dim, d) for d in (256, 74, 35)])
        self.post = nn.Sequential(nn.Linear(256 + 16 + 32 + 3 * dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.classifier, self.regressor = nn.Linear(128, 3), nn.Linear(128, 1)

    def forward(self, text, audio, vision, sequence, observed, return_aux=False):
        xs, observed = self.sanitize(text, audio, vision, sequence, observed)
        text_utt = masked_mean(xs[0], observed[..., 0])
        audio_seq, audio_utt = self.audio_encoder(xs[1], sequence, observed[..., 1])
        vision_seq, vision_utt = self.vision_encoder(xs[2], sequence, observed[..., 2])
        utterances = [text_utt, audio_utt, vision_utt]
        global_tokens = torch.stack([p(x) for p, x in zip(self.projections, utterances)], 1)
        local = [p(x) for p, x in zip(self.projections, (xs[0], audio_seq, vision_seq))]
        global_tokens, local = self.fusion(global_tokens, local, sequence)
        global_flat = global_tokens.flatten(1)
        feature = self.post(torch.cat(utterances + [global_flat], -1))
        logits, intensity = self.finish(self.classifier(feature), self.regressor(feature), observed)
        if not return_aux:
            return logits, intensity
        aux = dict(siamese=[head(x) for head, x in zip(self.siamese, [global_flat] + utterances)],
                   recon=[head(x) for head, x in zip(self.reconstruction, local)])
        return logits, intensity, aux


def restoration_losses(missing_aux, complete_aux, complete_inputs, missing_observed):
    xs = complete_inputs[:3]
    original_observed = complete_inputs[4] & complete_inputs[3][..., None]
    hidden = original_observed & ~missing_observed
    reconstruction = xs[0].new_zeros(())
    for i, (prediction, target) in enumerate(zip(missing_aux['recon'], xs)):
        mask = hidden[..., i, None]
        error = F.smooth_l1_loss(prediction, target.detach(), reduction='none')
        reconstruction = reconstruction + (error * mask).sum() / (mask.sum() * target.shape[-1]).clamp_min(1)
    attraction = xs[0].new_zeros(())
    available = [original_observed.flatten(1).any(1)] + [original_observed[..., i].any(1) for i in range(3)]
    for (pm, zm), (pc, zc), use in zip(missing_aux['siamese'], complete_aux['siamese'], available):
        # Symmetric stop-gradient SimSiam target, as in official EMT-DLFR.
        error = -.5 * (F.cosine_similarity(pm, zc.detach(), dim=-1) + F.cosine_similarity(pc, zm.detach(), dim=-1))
        attraction = attraction + (error * use).sum() / use.sum().clamp_min(1)
    return reconstruction, attraction


def make_model(name, **kwargs):
    if name == 'mult':
        return MulT(**kwargs)
    if name in ('emt_dlfr', 'emt_no_restore'):
        return EMTDLFR(**kwargs)
    raise ValueError(name)
