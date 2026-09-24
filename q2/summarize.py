"""Summarize validation evidence and freeze selection before special inference."""
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import ROOT,dump_json,conditions,sha256
from train import metrics,SELECT

MAIN=['mlp','gru_clean','gru_aug','state_aug','smooth_aug']
SEEDS=[17,29,43]


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    results=ROOT/'results';results.mkdir(exist_ok=True)
    runs=[];grouped=defaultdict(list)
    for path in sorted((ROOT/'runs').glob('*/metrics.json')):
        run=json.loads(path.read_text());run['directory']=path.parent
        runs.append(run);grouped[run['config']].append(run)
    for name in MAIN:
        if sorted(r['seed'] for r in grouped[name])!=SEEDS:
            raise RuntimeError(f'{name}: require three completed seeds before selection')
    rows=[];long=[]
    for name,rr in grouped.items():
        row=dict(model=name,n_seeds=len(rr))
        for scope,conds in [('clean',['clean']),('selection',SELECT),('grid27',[c['name'] for c in conditions()[1:28]])]:
            for metric in ['accuracy','macro_f1','mae','pearson']:
                v=np.array([np.mean([r['conditions'][c][metric] for c in conds]) for r in rr])
                row[f'{scope}_{metric}_mean']=float(v.mean())
                row[f'{scope}_{metric}_std']=float(v.std(ddof=1)) if len(v)>1 else None
        rows.append(row)
        for r in rr:
            for name_c,m in r['conditions'].items():
                long.append(dict(model=name,seed=r['seed'],condition=name_c,**{k:v for k,v in m.items() if k!='confusion_matrix'}))
    write_csv(results/'model_comparison.csv',rows)
    write_csv(results/'all_condition_metrics.csv',long)
    ranking=sorted([r for r in rows if r['model'] in MAIN],key=lambda r:(-r['selection_macro_f1_mean'],r['selection_mae_mean']))
    winner=ranking[0]['model']
    checkpoints=[str((ROOT/'runs'/f'{winner}_seed{s}'/'best.pt').relative_to(ROOT)) for s in SEEDS]
    selection=dict(config=winner,seeds=SEEDS,checkpoints=checkpoints,aggregation='mean class probabilities; mean intensity',
        rule='Architecture selected by mean across seeds of predeclared validation clean+T/A/V 30%-middle Macro-F1; MAE breaks ties. Three-seed ensemble fixed in advance of special inference.',
        validation_ranking=ranking,checkpoint_sha256={p:sha256(ROOT/p) for p in checkpoints})
    dump_json(ROOT/'selection.json',selection)
    predictions=[dict(np.load(ROOT/'runs'/f'{winner}_seed{s}'/'validation_predictions.npz')) for s in SEEDS]
    y=predictions[0]['labels'];target=predictions[0]['targets'];ids=predictions[0]['ids']
    ensemble={};out={}
    for c in conditions():
        name=c['name']
        prob=np.mean([d[name+'_probabilities'] for d in predictions],0)
        reg=np.mean([d[name+'_intensity'] for d in predictions],0)
        ensemble[name]=metrics(y,prob.argmax(-1),target,reg)
        out[name+'_probabilities']=prob;out[name+'_intensity']=reg
    np.savez(results/'selected_validation_predictions.npz',ids=ids,labels=y,targets=target,**out)
    dump_json(results/'selected_validation_metrics.json',ensemble)

    train=np.load(ROOT/'cache'/'train.npz')
    majority=int(np.bincount(train['labels']).argmax());median=float(np.median(train['targets']))
    constant=metrics(y,np.full_like(y,majority),target,np.full_like(target,median))
    dump_json(results/'constant_baseline.json',constant)
    raw=np.load(ROOT/'cache'/'valid.npz')['raw_text']
    pred=out['clean_probabilities'].argmax(-1);reg=out['clean_intensity']
    errorrows=[dict(sample_id=str(ids[i]),true_polarity=int(y[i]),predicted_polarity=int(pred[i]),true_intensity=float(target[i]),
        predicted_intensity=float(reg[i]),absolute_error=float(abs(target[i]-reg[i])),classification_error=bool(pred[i]!=y[i]),
        classification_regression_sign_conflict=bool((pred[i]==0 and reg[i]>0) or (pred[i]==2 and reg[i]<0)),text=str(raw[i])) for i in range(len(y))]
    write_csv(results/'validation_predictions_and_errors.csv',errorrows)
    write_csv(results/'largest_regression_errors.csv',sorted(errorrows,key=lambda x:x['absolute_error'],reverse=True)[:30])
    error_summary={}
    for name,ix in [('negative',y==0),('neutral',y==1),('positive',y==2),('weak_nonzero',(np.abs(target)>0)&(np.abs(target)<=.5))]:
        error_summary[name]=dict(n=int(ix.sum()),accuracy=float(np.mean(pred[ix]==y[ix])),mae=float(np.mean(np.abs(target[ix]-reg[ix]))))
    error_summary['sign_conflict_count']=sum(r['classification_regression_sign_conflict'] for r in errorrows)
    dump_json(results/'validation_error_summary.json',error_summary)
    valid_cache=np.load(ROOT/'cache'/'valid.npz')
    base_sequence=valid_cache['sequence']
    base_observed=valid_cache['observed']
    mask_cache=np.load(ROOT/'cache'/'valid_conditions.npz')
    budget_rows=[]
    for c in conditions()[1:]:
        obs=mask_cache[c['name']]
        for i,sample_id in enumerate(ids):
            length=int(base_sequence[i].sum())
            for m in c['modalities']:
                base=base_observed[i,:,m]
                removed=int((base & ~obs[i,:,m]).sum())
                available=int(base.sum())
                budget_rows.append(dict(sample_id=str(sample_id),condition=c['name'],modality=['text','audio','vision'][m],
                    content_positions=length,requested_fraction=c['ratio'],
                    interval_position_budget=max(1,round(length*c['ratio'])) if length else 0,
                    originally_available=available,newly_removed_observations=removed,
                    actual_removed_fraction_of_available=removed/available if available else None))
    write_csv(results/'validation_missing_budgets.csv',budget_rows)

    # Paired grouped bootstrap: compare models on the same video groups.
    group_ids=np.array([str(x).split('$_$')[0] for x in ids]);groups=np.unique(group_ids)
    members=[np.flatnonzero(group_ids==g) for g in groups]
    pair={}
    for c in ['clean','text_30_middle']:
        a=np.mean([np.load(ROOT/'runs'/f'state_aug_seed{s}'/'validation_predictions.npz')[c+'_intensity'] for s in SEEDS],0)
        b=np.mean([np.load(ROOT/'runs'/f'gru_aug_seed{s}'/'validation_predictions.npz')[c+'_intensity'] for s in SEEDS],0)
        delta=np.abs(target-a)-np.abs(target-b)
        rng=np.random.default_rng(123);samples=[]
        for _ in range(1000):
            ix=np.concatenate([members[j] for j in rng.integers(0,len(groups),len(groups))])
            samples.append(float(delta[ix].mean()))
        pair[c]=dict(state_minus_gru_mae=float(delta.mean()),percentile_95=np.quantile(samples,[.025,.975]).tolist(),
                     n_video_groups=len(groups),note='Development validation used for selection; interval describes paired validation variability, not independent generalization proof.')
    dump_json(results/'paired_group_bootstrap.json',pair)

    figures=results/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'savefig.dpi':180})
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for m,ax in enumerate(axes):
        modality=['text','audio','vision'][m]
        for name in ['gru_clean','gru_aug','state_aug']:
            rr=grouped[name]
            means=[];stds=[]
            for ratio in [10,30,50]:
                vals=[np.mean([r['conditions'][f'{modality}_{ratio}_{loc}']['macro_f1'] for loc in ['front','middle','back']]) for r in rr]
                means.append(np.mean(vals));stds.append(np.std(vals,ddof=1) if len(vals)>1 else 0.)
            ax.errorbar([10,30,50],means,yerr=stds,marker='o',label=f'{name} (n={len(rr)})',capsize=3)
        ax.set(title=f'{modality.capitalize()} missing',xlabel='Masked content positions (%)',ylabel='Macro-F1');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'missing_fraction_macro_f1.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for m,ax in enumerate(axes):
        modality=['text','audio','vision'][m]
        for name in ['gru_clean','gru_aug','state_aug',winner]:
            rr=grouped[name]
            values=[[np.mean([r['conditions'][f'{modality}_{ratio}_{loc}']['mae'] for loc in ['front','middle','back']]) for r in rr] for ratio in [10,30,50]]
            ax.errorbar([10,30,50],[np.mean(v) for v in values],yerr=[np.std(v,ddof=1) if len(v)>1 else 0 for v in values],marker='o',label=name,capsize=3)
        ax.set(title=f'{modality.capitalize()} missing',xlabel='Masked content positions (%)',ylabel='MAE');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'missing_fraction_mae.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    mats=[np.array([[ensemble[f'{m}_{r}_{loc}']['macro_f1'] for loc in ['front','middle','back']] for r in [10,30,50]]) for m in ['text','audio','vision']]
    vmin=min(x.min() for x in mats);vmax=max(x.max() for x in mats)
    for ax,m,matrix in zip(axes,['Text','Audio','Vision'],mats):
        im=ax.imshow(matrix,vmin=vmin,vmax=vmax,cmap='viridis')
        ax.set_xticks(range(3),['Front','Middle','Back']);ax.set_yticks(range(3),['10%','30%','50%']);ax.set_title(m)
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Macro-F1',shrink=.8);fig.savefig(figures/'position_fraction_heatmap.png',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    cm=np.array(ensemble['clean']['confusion_matrix']);axes[0].imshow(cm,cmap='Blues')
    for i in range(3):
        for j in range(3):axes[0].text(j,i,str(cm[i,j]),ha='center',va='center')
    axes[0].set_xticks(range(3),['Neg','Neutral','Pos']);axes[0].set_yticks(range(3),['Neg','Neutral','Pos']);axes[0].set(xlabel='Predicted',ylabel='True',title='Validation confusion matrix')
    axes[1].scatter(target,reg,s=9,alpha=.4);axes[1].plot([-3,3],[-3,3],'k--');axes[1].set(xlabel='True intensity',ylabel='Predicted intensity',title='Validation regression');fig.tight_layout();fig.savefig(figures/'validation_prediction.png');plt.close(fig)

    clean=ensemble['clean'];audit=json.loads((results/'audit.json').read_text())
    lines=['# 第二问实测报告','',
       '本报告只覆盖第二问。全部性能指标来自附件2验证集，且该验证集参与早停及结构选择；不能视为独立测试精度。附件2 test 未用于训练、选模或性能统计，附件3只在冻结后推理。','',
       '## 数据与输入','',
       '训练3395条，验证728条，三分类负/中/正；强度范围[-3,3]。训练与验证词元均与所选uncased BERT词表完全匹配。官方划分之间原视频ID无交叉。',
       '统一使用text_bert，经冻结BERT-Mini得到256维文本表示，与官方74维音频和35维视觉按aligned位置输入。未混入第一问特征或其他情感训练数据。',
       '使用SEP位置界定内容范围，排除CLS、SEP与padding。内容内音视频全零行按操作性不可用标记处理，不能判定其真实成因。缺失时长以位置数/比例报告，不虚构秒数。',
       '标准化仅拟合训练集有效音视频观测。文本在编码前遮挡；缓存8种训练缺失文本视图，音视频区间在线采样。该有限文本增强库是当前实现的限制。','',
       '## 实验设置','',
       '共同投影与门控、32维时序表示、CE+Huber、AdamW、batch 64、学习率0.001、最多40轮、早停6轮。完整/单模态/双模态增强概率为40%/40%/20%，长度10%—50%。',
       '每次运行在完整输入与T/A/V各30%中部缺失四个验证条件上平均Macro-F1选检查点。五种主要候选各3个种子；其余消融/基线单种子，只能视为初步证据。模型选择不使用27条件的汇总排名。无增强GRU在首轮验证后补齐重复并纳入候选，未查看附件3分布或性能。',
       '状态模型为学习式历史传播与门控更新，初始传播系数0.98；无协方差、不声称卡尔曼最优性或真实逐时刻情绪。GRU获得相同缺失状态提示。','',
       '## 模型比较','',
       '|模型|种子数|完整Macro-F1|完整MAE|选模条件平均Macro-F1|27条件平均Macro-F1|',
       '|---|---:|---:|---:|---:|---:|']
    for row in sorted(rows,key=lambda r:-r['selection_macro_f1_mean']):
        lines.append(f'|{row["model"]}|{row["n_seeds"]}|{row["clean_macro_f1_mean"]:.4f}|{row["clean_mae_mean"]:.4f}|{row["selection_macro_f1_mean"]:.4f}|{row["grid27_macro_f1_mean"]:.4f}|')
    lines += ['',f'多数类/训练标签中位数基线：Accuracy={constant["accuracy"]:.4f}，Macro-F1={constant["macro_f1"]:.4f}，MAE={constant["mae"]:.4f}；常数回归的Pearson未定义。','',
       '## 最终配置','',f'选定 **{winner}**，对3个种子平均类别概率和回归输出。完整验证集：Accuracy={clean["accuracy"]:.4f}，Macro-F1={clean["macro_f1"]:.4f}，MAE={clean["mae"]:.4f}，Pearson={clean["pearson"]:.4f}。',
       '集成值与上表逐种子均值不是同一统计量，应分别标注。选择记录与权重哈希见selection.json。','',
       '## 缺失规律与错误检查','',
       '27条件完整数值见all_condition_metrics.csv；同步、错位和双块缺失为补充压力测试。各模型使用完全相同的评价遮挡，缺失率按内容位置预算定义；短样本因取整可能与目标比例有差异。',
       '模型对比图显示各种子标准差；单种子没有重复稳定性证据。state与GRU的配对MAE差及原视频分组bootstrap见paired_group_bootstrap.json，不应将验证集区间写成独立泛化证明。',
       'validation_predictions_and_errors.csv记录全部验证样本的预测、分类错误、回归残差与分类/回归符号冲突；largest_regression_errors.csv列出30条最大回归误差。具体语言/模态错因需查看样本证据，不由错误标签自动推断。','',
       '## 可复现与边界','',
       '运行顺序、输入规范见README.md。缓存不进入提交包，通用冻结编码器权重随第二问包提供。第二问包不包含第一问100条特征或第三问解释，因此不等于整题完整提交。',
       '未确认的官方尾部全模态缺失可能与padding不可辨识。若专项样本缺少SEP，使用最末可见支持位置估计端点，并在预测provenance文件中标注。',
       '文献强基线LMF/MulT尚未实现；当前证据支持已运行模型之间的比较，不支持超过所有现有方法或获奖能力的判断。','']
    lines += ['## 本轮结果的具体分析','',
        '以下结论仅针对本轮编码器、32维结构和训练预算，不推广为所有状态空间或时序方法的优劣。','',
        '|缺失模态|10%位置缺失 Macro-F1 / MAE|30%位置缺失 Macro-F1 / MAE|50%位置缺失 Macro-F1 / MAE|',
        '|---|---:|---:|---:|']
    for m in ['text','audio','vision']:
        parts=[]
        for ratio in [10,30,50]:
            f=np.mean([ensemble[f'{m}_{ratio}_{loc}']['macro_f1'] for loc in ['front','middle','back']])
            e=np.mean([ensemble[f'{m}_{ratio}_{loc}']['mae'] for loc in ['front','middle','back']])
            parts.append(f'{f:.4f} / {e:.4f}')
        lines.append('|'+m+'|'+'|'.join(parts)+'|')
    lines += ['',
        '表中对前/中/后位置取平均。当前模型对文本缺失更敏感；音视频局部遮挡影响较小，不能仅据此宣布融合有效或音视频没有情感信息。应把“模型实际利用有限”与“模态本身无用”分开。个别缺失条件F1略高于完整输入，不强行解释为普遍规律。',
        f'完整输入中，中性类召回率为{error_summary["neutral"]["accuracy"]:.2%}，低于负向{error_summary["negative"]["accuracy"]:.2%}与正向{error_summary["positive"]["accuracy"]:.2%}；{cm[1,2]}/{cm[1].sum()}条中性被预测为正向。中性区分是当前主要分类瓶颈。',
        f'弱非零强度样本(|y|≤0.5且y≠0)共{error_summary["weak_nonzero"]["n"]}条，准确率{error_summary["weak_nonzero"]["accuracy"]:.2%}。分类与回归正负符号冲突共{error_summary["sign_conflict_count"]}/{len(y)}条；双头输出并非自动一致。',
        'state相对GRU的配对回归误差差值见bootstrap文件。完整输入区间若跨零，不能宣称改进稳定；文本缺失条件下差值为正表示state误差更大。保留负结果，当前不把递推或缺失长度提示写成已验证优势。','']
    (results/'第二问实测报告.md').write_text('\n'.join(lines))
    print(json.dumps(dict(selected=winner,clean=clean),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
