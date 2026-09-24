"""Descriptive calculations from frozen validation predictions; no fitting."""
from pathlib import Path
import json, csv, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
EXP=ROOT/'experiments/paper_baselines_20260924'
def readj(p):return json.loads(p.read_text())
models=['mlp','state_aug','mult','emt_dlfr','emt_no_restore']
scores={m:readj(EXP/f'{m}_ensemble_metrics.json') for m in models}
pred={m:dict(np.load(EXP/f'{m}_ensemble_predictions.npz',allow_pickle=False)) for m in models}
y=pred['emt_dlfr']['labels'];targets=pred['emt_dlfr']['targets']
for a in pred.values():
 assert np.array_equal(a['labels'],y) and np.array_equal(a['targets'],targets)
 assert np.array_equal(a['ids'],pred['emt_dlfr']['ids'])
report={'scope':'仅从冻结开发验证预测计算描述统计；无新增训练、调参或专项标签；不用于改选模型','n_validation':len(y),'per_class':{},'paired_classification':{},'opposite_sign':{},'strong_and_weak':{},'modality_50':{},'grid27':{}}
for m,a in pred.items():
 c=a['clean_probabilities'].argmax(1);v=a['clean_intensity']
 assert abs(np.mean(c==y)-scores[m]['clean']['accuracy'])<1e-7
 assert abs(np.mean(abs(v-targets))-scores[m]['clean']['mae'])<1e-6
 report['per_class'][m]=[{'class':int(k),'n':int((y==k).sum()),'correct':int(((c==y)&(y==k)).sum()),'recall':float((c[y==k]==k).mean()),'mae':float(np.mean(abs(v[y==k]-targets[y==k])))} for k in range(3)]
 report['opposite_sign'][m]=int((((c==0)&(v>0))|((c==2)&(v<0))).sum())
 report['strong_and_weak'][m]=[]
 for name,mask in [('neutral',targets==0),('weak_abs_le_0.5',(abs(targets)>0)&(abs(targets)<=.5)),('remaining_abs_gt_0.5',abs(targets)>.5)]:
  report['strong_and_weak'][m].append({'group':name,'n':int(mask.sum()),'accuracy':float((c[mask]==y[mask]).mean()),'mae':float(np.mean(abs(v[mask]-targets[mask])))})
 report['modality_50'][m]={}
 for mod in ['text','audio','vision']:
  values=[scores[m][f'{mod}_50_{loc}'] for loc in ['front','middle','back']]
  f1=float(np.mean([r['macro_f1'] for r in values]));mae=float(np.mean([r['mae'] for r in values]))
  report['modality_50'][m][mod]={'macro_f1':f1,'mae':mae,'delta_f1':f1-scores[m]['clean']['macro_f1'],'delta_mae':mae-scores[m]['clean']['mae']}
 grid=[v for k,v in scores[m].items() if any(k.startswith(mod+'_') for mod in ['text','audio','vision']) and k.split('_')[1] in ['10','30','50'] and k.split('_')[-1] in ['front','middle','back']]
 assert len(grid)==27
 report['grid27'][m]={k:float(np.mean([x[k] for x in grid])) for k in ['macro_f1','mae']}
a=pred['mlp']['clean_probabilities'].argmax(1)==y;b=pred['emt_dlfr']['clean_probabilities'].argmax(1)==y
report['paired_classification']={'both_correct':int((a&b).sum()),'only_mlp_correct':int((a&~b).sum()),'only_emt_correct':int((~a&b).sum()),'both_wrong':int((~a&~b).sum())}
report['selected_seed_difference']=readj(EXP/'selection.json')['ranking'][0]['selection_macro_f1_mean']-readj(EXP/'selection.json')['ranking'][1]['selection_macro_f1_mean']
(OUT/'derived_evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
