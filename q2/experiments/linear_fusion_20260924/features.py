"""Features shared by the alternative training and raw-input inference paths."""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

BLOCKS=(512,148,70,3)


def visible_documents(tokens, observed, sequence):
    # Keep a sentinel at every unavailable position; never create a bigram
    # across an erased word. CLS/SEP/padding do not enter the document.
    return [[str(int(t)) if seen else '~' for t,seen in zip(ids[pos],obs[pos])]
            for ids,obs,pos in zip(tokens,observed,sequence)]


def visible_ngrams(document):
    for i,t in enumerate(document):
        if t=='~':continue
        yield 'u'+t
        if i+1<len(document) and document[i+1]!='~':
            yield 'b'+t+'_'+document[i+1]


def masked_moments(values, observed):
    x=np.where(observed[...,None],np.asarray(values,dtype=np.float32),0)
    count=observed.sum(1,keepdims=True).clip(min=1)
    mean=x.sum(1)/count
    variance=(x*x).sum(1)/count-mean*mean
    return np.concatenate([mean,np.sqrt(np.maximum(variance,0))],axis=1).astype(np.float32)


def pooled_features(text,audio,vision,sequence,observed):
    observed=observed&sequence[:,:,None]
    parts=[masked_moments(x,observed[:,:,m]) for m,x in enumerate([text,audio,vision])]
    fractions=observed.sum(1)/sequence.sum(1,keepdims=True).clip(min=1)
    return np.concatenate(parts+[fractions.astype(np.float32)],axis=1)


def cache_features(cache,banks,observed,batch=256):
    n=len(cache['labels']);out=[]
    for start in range(0,n,batch):
        ix=np.arange(start,min(n,start+batch))
        text=cache['text_bank'][banks[ix],ix]
        out.append(pooled_features(text,cache['audio'][ix],cache['vision'][ix],cache['sequence'][ix],observed[ix]))
    dense=np.concatenate(out)
    docs=visible_documents(cache['token_ids'],observed[:,:,0],cache['sequence'])
    return dense,docs


class FeatureTransform:
    def fit(self,dense,documents):
        self.scaler=StandardScaler().fit(dense)
        self.block_scale=np.concatenate([np.full(d,1/np.sqrt(d)) for d in BLOCKS]).astype(np.float32)
        self.tfidf=TfidfVectorizer(analyzer=visible_ngrams,min_df=2,max_features=12000,
                                 sublinear_tf=True,dtype=np.float32)
        self.tfidf.fit(documents)
        return self

    def transform(self,dense,documents,family):
        z=self.scaler.transform(dense).astype(np.float32)*self.block_scale
        if family=='pooled_linear':return np.asarray(z,dtype=np.float64,order='C')
        if family!='lexical_multimodal_linear':raise ValueError(family)
        return sparse.hstack([sparse.csr_matrix(z),3*self.tfidf.transform(documents)],format='csr')


def augmented_views(cache,seed):
    """40% clean weight + 60% missing weight, with 2:1 single/double mix."""
    rng=np.random.default_rng(seed);n=len(cache['labels'])
    yield np.zeros(n,dtype=np.int64),cache['observed'].copy(),.4
    for _ in range(4):
        obs=cache['observed'].copy();banks=np.zeros(n,dtype=np.int64)
        for i in range(n):
            positions=np.flatnonzero(cache['sequence'][i])
            if not len(positions):continue
            mods=rng.choice(3,size=1 if rng.random()<2/3 else 2,replace=False)
            width=max(1,round(len(positions)*rng.uniform(.1,.5)))
            start=int(rng.integers(len(positions)-width+1));chosen=positions[start:start+width]
            sync=rng.random()<.5
            if 0 in mods:
                banks[i]=rng.integers(1,len(cache['text_bank']))
                obs[i,:,0]=cache['text_observed_bank'][banks[i],i]
                if sync:chosen=np.flatnonzero(cache['sequence'][i]&~obs[i,:,0])
            for m in mods:
                if m==0:continue
                if not sync:
                    start=int(rng.integers(len(positions)-width+1));chosen=positions[start:start+width]
                obs[i,chosen,m]=False
        yield banks,obs,.15
