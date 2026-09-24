import argparse
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error,confusion_matrix
from data import ROOT,dump_json,conditions
from models import Predictor

CONFIGS={
    'text':dict(kind='text',augment=True,use_gap=True),
    'mlp':dict(kind='mlp',augment=True,use_gap=True),
    'gru_clean':dict(kind='gru',augment=False,use_gap=True),
    'gru_aug':dict(kind='gru',augment=True,use_gap=True),
    'state_aug':dict(kind='state',augment=True,use_gap=True),
    'state_clean':dict(kind='state',augment=False,use_gap=True),
    'state_no_gap':dict(kind='state',augment=True,use_gap=False),
    'state_no_history':dict(kind='no_history',augment=True,use_gap=True),
    'smooth_aug':dict(kind='smooth',augment=True,use_gap=True),
}
SELECT=['clean','text_30_middle','audio_30_middle','vision_30_middle']


def metrics(y,pred,targets,reg):
    correlation=None
    if np.std(reg)>1e-10 and np.std(targets)>1e-10:
        correlation=float(np.corrcoef(targets,reg)[0,1])
    return dict(accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
        weighted_f1=float(f1_score(y,pred,labels=[0,1,2],average='weighted',zero_division=0)),
        mae=float(mean_absolute_error(targets,reg)),pearson=correlation,
        confusion_matrix=confusion_matrix(y,pred,labels=[0,1,2]).tolist())


def load_cache(split,device):
    raw=dict(np.load(ROOT/'cache'/f'{split}.npz',allow_pickle=False))
    out={}
    for k in ['text_bank','audio','vision','sequence','observed','text_observed_bank','labels','targets']:
        out[k]=torch.from_numpy(raw[k]).to(device)
    out['ids']=raw['ids']
    for k in ['observed','sequence','text_observed_bank']:
        out[k+'_np']=raw[k]
    return out


def training_batch(data,indices,rng,augment):
    device=data['audio'].device
    ix=torch.as_tensor(indices,device=device)
    obs=data['observed_np'][indices].copy()
    sequence=data['sequence'][ix]
    banks=np.zeros(len(indices),dtype=np.int64)
    if augment:
        base_seq=data['sequence_np'][indices]
        for i in range(len(indices)):
            draw=rng.random()
            if draw<.4:
                continue
            mods=rng.choice(3,size=1 if draw<.8 else 2,replace=False)
            pos=np.flatnonzero(base_seq[i])
            if not len(pos):
                continue
            width=max(1,round(len(pos)*rng.uniform(.1,.5)))
            start=int(rng.integers(len(pos)-width+1))
            chosen=pos[start:start+width]
            sync=rng.random()<.5
            if 0 in mods:
                banks[i]=int(rng.integers(1,data['text_bank'].shape[0]))
                text_obs=data['text_observed_bank_np'][banks[i],indices[i]]
                obs[i,:,0]=text_obs
                if sync:
                    chosen=np.flatnonzero(base_seq[i] & ~text_obs)
            for m in mods:
                if m==0:
                    continue
                if not sync:
                    start=int(rng.integers(len(pos)-width+1))
                    chosen=pos[start:start+width]
                obs[i,chosen,int(m)]=False
    b=torch.as_tensor(banks,device=device)
    text=data['text_bank'][b,ix].float()
    return (text,data['audio'][ix],data['vision'][ix],sequence,torch.from_numpy(obs).to(device)),data['labels'][ix],data['targets'][ix]


def text_bank_index(condition):
    if 0 not in condition['modalities']:
        return 0
    return 1+[.1,.3,.5].index(condition['ratio'])*3+['front','middle','back'].index(condition['location'])


@torch.inference_mode()
def evaluate(model,data,condition_masks,names=None,batch=128,collect=False):
    model.eval()
    result={}
    predictions={}
    for c in conditions():
        name=c['name']
        if names is not None and name not in names:
            continue
        obs=condition_masks[name]
        bank=text_bank_index(c)
        probs=[]; regs=[]
        for start in range(0,len(data['labels']),batch):
            sl=slice(start,start+batch)
            logits,r=model(data['text_bank'][bank,sl].float(),data['audio'][sl],data['vision'][sl],data['sequence'][sl],obs[sl])
            probs.append(logits.softmax(-1).cpu().numpy());regs.append(r.cpu().numpy())
        probs=np.concatenate(probs);reg=np.concatenate(regs)
        pred=probs.argmax(-1)
        result[name]=metrics(data['labels'].cpu().numpy(),pred,data['targets'].cpu().numpy(),reg)
        if collect:
            predictions[name+'_probabilities']=probs
            predictions[name+'_intensity']=reg
    return result,predictions


def run(config_name,seed,device='cuda',max_epochs=None):
    protocol=json.loads((ROOT/'protocol.json').read_text())
    config=CONFIGS[config_name]
    out=ROOT/'runs'/f'{config_name}_seed{seed}'
    if (out/'metrics.json').exists():
        print(f'Skip completed {out.name}',flush=True)
        return
    out.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    train=load_cache('train',device);valid=load_cache('valid',device)
    masks={k:torch.from_numpy(v).to(device) for k,v in dict(np.load(ROOT/'cache'/'valid_conditions.npz')).items()}
    stats=np.load(ROOT/'assets'/'normalization.npz')
    model_args=dict(kind=config['kind'],hidden=protocol['hidden'],dropout=protocol['dropout'],use_gap=config['use_gap'],
        class_prior=stats['class_prior'].tolist(),target_median=float(stats['target_median']))
    model=Predictor(**model_args).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    n=len(train['labels']);batch=protocol['batch_size']
    best=(-float('inf'),-float('inf'));bad=0;history=[];start_time=time.time()
    shuffle_rng=np.random.default_rng(seed)
    mask_rng=np.random.default_rng(seed+10000)
    dump_json(out/'config.json',dict(name=config_name,seed=seed,**config,model_args=model_args,protocol=protocol,
        parameters=sum(p.numel() for p in model.parameters()),device=str(device),torch_version=torch.__version__))
    for epoch in range(1,(max_epochs or protocol['max_epochs'])+1):
        model.train(); losses=[]
        order=shuffle_rng.permutation(n)
        for start in range(0,n,batch):
            inputs,y,target=training_batch(train,order[start:start+batch],mask_rng,config['augment'])
            optimizer.zero_grad(set_to_none=True)
            logits,r=model(*inputs)
            loss=F.cross_entropy(logits,y)+protocol['regression_weight']*F.huber_loss(r,target)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss {config_name} epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        scores,_=evaluate(model,valid,masks,names=SELECT)
        score=float(np.mean([scores[c]['macro_f1'] for c in SELECT]))
        mae=float(np.mean([scores[c]['mae'] for c in SELECT]))
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),selection_macro_f1=score,selection_mae=mae,
                 clean=scores['clean'],elapsed_seconds=time.time()-start_time)
        history.append(row)
        key=(score,-mae)
        if key>best:
            best=key;bad=0
            torch.save(dict(state_dict=model.state_dict(),model_args=model_args,epoch=epoch,selection=row),out/'best.pt')
        else:
            bad+=1
        dump_json(out/'history.json',history)
        print(f'{out.name} epoch={epoch} loss={row["train_loss"]:.4f} clean_F1={scores["clean"]["macro_f1"]:.4f} robust_select_F1={score:.4f} MAE={mae:.4f} seconds={row["elapsed_seconds"]:.1f}',flush=True)
        if bad>=protocol['patience']:
            break
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    scores,preds=evaluate(model,valid,masks,collect=True)
    dump_json(out/'metrics.json',dict(config=config_name,seed=seed,best_epoch=checkpoint['epoch'],
        train_seconds=time.time()-start_time,selection_macro_f1=best[0],selection_mae=-best[1],conditions=scores))
    np.savez(out/'validation_predictions.npz',ids=valid['ids'],labels=valid['labels'].cpu().numpy(),targets=valid['targets'].cpu().numpy(),**preds)
    print(f'Completed {out.name}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',choices=CONFIGS,required=True)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--max-epochs',type=int)
    args=parser.parse_args()
    run(args.config,args.seed,args.device,args.max_epochs)
