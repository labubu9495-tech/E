import math
import torch
from torch import nn


def gap_features(observed, sequence):
    clock = sequence.long().cumsum(1)
    last = torch.where(observed, clock.unsqueeze(-1), 0).cummax(1).values
    gap = (clock.unsqueeze(-1)-last).float()
    return torch.log1p(gap) * sequence.unsqueeze(-1)


class Predictor(nn.Module):
    def __init__(self, kind='gru', hidden=32, dropout=.2, use_gap=True,
                 class_prior=(.3,.2,.5), target_median=0.):
        super().__init__()
        self.kind = kind
        self.use_gap = use_gap
        self.projections = nn.ModuleList([nn.Sequential(nn.Linear(d, hidden),nn.LayerNorm(hidden),nn.Tanh()) for d in [256,74,35]])
        self.score = nn.Sequential(nn.Linear(hidden+6,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.fusion = nn.Sequential(nn.Linear(hidden*3+3,hidden),nn.Tanh())
        self.gru = nn.GRU(hidden+6,hidden,batch_first=True) if kind=='gru' else None
        self.gain = nn.Linear(2*hidden+6,hidden) if kind=='state' else None
        # Initial retention .98 corresponds to a roughly 50-position scale.
        self.decay_logit = nn.Parameter(torch.full((hidden,),math.log(.98/.02))) if kind=='state' else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden,3)
        self.regressor = nn.Linear(hidden,1)
        self.register_buffer('class_prior',torch.tensor(class_prior,dtype=torch.float32))
        self.register_buffer('target_median',torch.tensor(target_median,dtype=torch.float32))

    def forward(self, text, audio, vision, sequence, observed):
        observed = observed & sequence.unsqueeze(-1)
        gap = gap_features(observed,sequence) if self.use_gap else torch.zeros_like(observed,dtype=text.dtype)
        info = torch.cat([observed.to(text.dtype),gap],-1)
        hs=[]
        for m,(x,projection) in enumerate(zip([text,audio,vision],self.projections)):
            # Remove unavailable content before any learned transformation.
            x = torch.where(observed[:,:,m,None],x,0.)
            hs.append(projection(x)*observed[:,:,m,None])
        h = torch.stack(hs,2)
        score = self.score(torch.cat([h,info.unsqueeze(2).expand(-1,-1,3,-1)],-1)).squeeze(-1)
        score = score.masked_fill(~observed,-1e4)
        weights = score.softmax(-1)*observed
        weights = weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        u = self.fusion(torch.cat([(h*weights.unsqueeze(-1)).flatten(2), observed.to(text.dtype)],-1))
        any_seen = observed.any(-1)
        u = u*any_seen.unsqueeze(-1)
        if self.kind=='gru':
            # Compact genuine positions only. Artificial internal holes remain
            # genuine positions and still advance the recurrent computation.
            lengths=sequence.sum(1).clamp_min(1)
            order=torch.argsort((~sequence).long(),dim=1,stable=True)
            x=torch.cat([u,info],-1)
            compact=x.gather(1,order.unsqueeze(-1).expand_as(x))
            # Unidirectional GRU: padded suffix cannot affect valid prefixes.
            # Dense cuDNN avoids device-to-host length synchronization.
            states,_=self.gru(compact)
            valid=torch.arange(sequence.shape[1],device=text.device)[None,:]<lengths[:,None]
            pooled=(states*valid.unsqueeze(-1)).sum(1)/lengths[:,None]
        elif self.kind in ('state','smooth'):
            z=torch.zeros_like(u[:,0])
            total=torch.zeros_like(z)
            decay=torch.sigmoid(self.decay_logit) if self.kind=='state' else .98
            for t in range(u.shape[1]):
                prior=z*decay
                if self.kind=='state':
                    k=torch.sigmoid(self.gain(torch.cat([prior,u[:,t],info[:,t]],-1)))
                else:
                    k=torch.full_like(z,.5)
                k=k*any_seen[:,t,None]
                proposal=(1-k)*prior+k*u[:,t]
                z=torch.where(sequence[:,t,None],proposal,z)
                total=total+z*sequence[:,t,None]
            pooled=total/sequence.sum(1,keepdim=True).clamp_min(1)
        elif self.kind=='no_history':
            pooled=(u*sequence.unsqueeze(-1)).sum(1)/sequence.sum(1,keepdim=True).clamp_min(1)
        elif self.kind in ('mlp','text'):
            means=(h*observed.unsqueeze(-1)).sum(1)/observed.sum(1).clamp_min(1).unsqueeze(-1)
            if self.kind=='text':
                pooled=means[:,0]
            else:
                presence=observed.any(1)
                pooled=self.fusion(torch.cat([means.flatten(1),presence.to(text.dtype)],-1))
        else:
            raise ValueError(self.kind)
        out=self.dropout(pooled)
        logits=self.classifier(out)
        intensity=3*torch.tanh(self.regressor(out).squeeze(-1)/3)
        empty=~observed.flatten(1).any(1)
        logits=torch.where(empty[:,None],torch.log(self.class_prior.clamp_min(1e-8))[None,:],logits)
        intensity=torch.where(empty,self.target_median,intensity)
        return logits,intensity
