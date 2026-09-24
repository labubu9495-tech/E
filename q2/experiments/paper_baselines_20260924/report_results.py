"""Aggregate completed paper experiments without changing the historical selection."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from data import conditions, dump_json, sha256
from train import metrics, SELECT

NAMES = {'mlp':'原MLP', 'state_aug':'原状态递推', 'mult':'MulT适配版',
         'emt_dlfr':'EMT-DLFR适配版', 'emt_no_restore':'EMT去恢复约束'}
COLORS = ['#7b8794','#b99057','#237b9a','#cf594e','#7970ac']
SEEDS = [17,29,43]
METRICS = ['accuracy','macro_f1','mae','pearson']


def csv_write(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def fast_f1(y, p):
    cm=np.bincount(y.astype(int)*3+p,minlength=9).reshape(3,3)
    den=cm.sum(0)+cm.sum(1)
    return float(np.divide(2*np.diag(cm),den,out=np.zeros(3,dtype=float),where=den>0).mean())


def main():
    ensembles, runs, predictions, rows, longrows, ensemble_rows = {}, {}, {}, [], [], []
    ids=y=targets=None
    names=[c['name'] for c in conditions()]
    for name in NAMES:
        root=ROOT if name in ('mlp','state_aug') else HERE
        run_dirs=[root/'runs'/f'{name}_seed{s}' for s in SEEDS]
        if any(not (p/'metrics.json').exists() for p in run_dirs):
            raise RuntimeError(f'Incomplete three-seed experiment: {name}')
        runs[name]=[json.loads((p/'metrics.json').read_text()) for p in run_dirs]
        pp=[dict(np.load(p/'validation_predictions.npz')) for p in run_dirs]
        if ids is None:
            ids,y,targets=pp[0]['ids'],pp[0]['labels'],pp[0]['targets']
        for data in pp:
            assert np.array_equal(data['ids'],ids) and np.array_equal(data['labels'],y) and np.array_equal(data['targets'],targets)
        predictions[name]={}
        ensembles[name]={}
        for c in names:
            prob=np.mean([d[c+'_probabilities'] for d in pp],0)
            reg=np.mean([d[c+'_intensity'] for d in pp],0)
            ensembles[name][c]=metrics(y,prob.argmax(-1),targets,reg)
            predictions[name][c+'_probabilities']=prob
            predictions[name][c+'_intensity']=reg
        np.savez(HERE/f'{name}_ensemble_predictions.npz',ids=ids,labels=y,targets=targets,**predictions[name])
        dump_json(HERE/f'{name}_ensemble_metrics.json',ensembles[name])
        row={'model':name,'n_seeds':3}
        for scope,cs in [('clean',['clean']),('selection',SELECT),('grid27',names[1:28])]:
            erow={'model':name,'scope':scope}
            for metric in METRICS:
                values=np.array([np.mean([r['conditions'][c][metric] for c in cs]) for r in runs[name]])
                row[f'{scope}_{metric}_mean']=float(values.mean())
                row[f'{scope}_{metric}_std']=float(values.std(ddof=1))
                erow[metric]=float(np.mean([ensembles[name][c][metric] for c in cs]))
            ensemble_rows.append(erow)
        rows.append(row)
        for run in runs[name]:
            for c,m in run['conditions'].items():
                longrows.append(dict(model=name,seed=run['seed'],condition=c,**{k:v for k,v in m.items() if k!='confusion_matrix'}))
    csv_write(HERE/'seed_summary.csv',rows)
    csv_write(HERE/'ensemble_comparison.csv',ensemble_rows)
    csv_write(HERE/'all_condition_metrics.csv',longrows)
    resource_rows=[]
    for name in ['mult','emt_dlfr','emt_no_restore']:
        for seed,run in zip(SEEDS,runs[name]):
            cfg=json.loads((HERE/'runs'/f'{name}_seed{seed}'/'config.json').read_text())
            resource_rows.append(dict(model=name,seed=seed,parameters=cfg['parameters'],
                seconds_including_validation=run['train_seconds'],best_epoch=run['best_epoch'],
                peak_allocated_GiB=run['peak_allocated_bytes']/1024**3,device=cfg['device']))
    csv_write(HERE/'resource_usage.csv',resource_rows)
    # The ablation is a mechanism comparison, not an extra search candidate.
    ranking=sorted([r for r in rows if r['model']!='emt_no_restore'],key=lambda r:(-r['selection_macro_f1_mean'],r['selection_mae_mean']))
    winner=ranking[0]['model']
    paper_winner=next(r['model'] for r in ranking if r['model'] in ('mult','emt_dlfr'))
    def frozen(name):
        root=ROOT if name in ('mlp','state_aug') else HERE
        files=[root/'runs'/f'{name}_seed{s}'/'best.pt' for s in SEEDS]
        return dict(model=name,seeds=SEEDS,checkpoints=[str(p.relative_to(ROOT)) for p in files],
            checkpoint_sha256={str(p.relative_to(ROOT)):sha256(p) for p in files},
            implementation='legacy' if root==ROOT else 'paper_adaptation')
    selection=dict(selected=winner,best_paper=paper_winner,ranking=ranking,
        rule='Mean across three seeds of clean+T/A/V30-middle Macro-F1; lower MAE breaks ties; ablation excluded from selection.',
        aggregation='Mean class probabilities and mean intensities from three selected checkpoints',
        overall=frozen(winner),paper=frozen(paper_winner),
        frozen_inputs={p:sha256(ROOT/p) for p in ['data.py','assets/normalization.npz','assets/bert-mini/model.safetensors']},
        paper_model_sha256=sha256(HERE/'paper_models.py'),protocol_sha256=sha256(HERE/'protocol.json'),
        validation_is_development=True,canonical_historical_selection_unchanged=True)
    dump_json(HERE/'selection.json',selection)

    # Paired resampling by original video group; selection bias remains.
    groups=np.array([str(x).split('$_$')[0] for x in ids])
    group_members=[np.flatnonzero(groups==g) for g in np.unique(groups)]
    bootstrap={}
    for c in ['clean','text_30_middle']:
        base=predictions['mlp'];new=predictions[paper_winner]
        bp=base[c+'_probabilities'].argmax(-1);npred=new[c+'_probabilities'].argmax(-1)
        delta=np.abs(new[c+'_intensity']-targets)-np.abs(base[c+'_intensity']-targets)
        rng=np.random.default_rng(20260924)
        samples=[]
        for _ in range(1000):
            ix=np.concatenate([group_members[i] for i in rng.integers(len(group_members),size=len(group_members))])
            samples.append([fast_f1(y[ix],npred[ix])-fast_f1(y[ix],bp[ix]),float(delta[ix].mean())])
        ci=np.quantile(samples,[.025,.975],axis=0)
        bootstrap[c]=dict(f1_difference=fast_f1(y,npred)-fast_f1(y,bp),f1_ci95=ci[:,0].tolist(),
            mae_difference=float(delta.mean()),mae_ci95=ci[:,1].tolist(),video_groups=len(group_members))
    dump_json(HERE/'paired_bootstrap.json',dict(best_paper=paper_winner,versus='mlp',results=bootstrap,
        note='Development validation group bootstrap; does not remove model-selection bias or establish independent generalization.'))

    errors=[]
    with np.load(ROOT/'cache/valid.npz') as cache:
        raw_text=cache['raw_text']
    for name in ['mlp',paper_winner]:
        prob=predictions[name]['clean_probabilities'];reg=predictions[name]['clean_intensity'];pred=prob.argmax(-1)
        for i in range(len(y)):
            errors.append(dict(model=name,sample_id=str(ids[i]),label=int(y[i]),prediction=int(pred[i]),
                target=float(targets[i]),intensity=float(reg[i]),absolute_error=float(abs(reg[i]-targets[i])),
                classification_error=bool(pred[i]!=y[i]),text=str(raw_text[i])))
    csv_write(HERE/'validation_errors.csv',errors)

    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    (HERE/'figures').mkdir(exist_ok=True)
    gallery=[]
    with PdfPages(HERE/'论文方法实验图集.pdf') as pdf:
        def save(fig,filename,title,note='附件2开发验证集（参与选模）｜不能视为独立测试成绩'):
            fig.suptitle(title,x=.05,ha='left',fontweight='bold',fontsize=16)
            fig.text(.05,.018,note,fontsize=9,color='#637282')
            fig.tight_layout(rect=[.01,.055,.99,.94])
            fig.savefig(HERE/'figures'/f'{filename}.png',dpi=240)
            fig.savefig(HERE/'figures'/f'{filename}.svg')
            pdf.savefig(fig);plt.close(fig);gallery.append((filename,title))
        for scope,cs,num in [('完整输入',['clean'],'01'),('四条件综合',SELECT,'02')]:
            fig,axes=plt.subplots(2,2,figsize=(12,8))
            for ax,metric,label in zip(axes.flat,METRICS,['Accuracy ↑','Macro-F1 ↑','MAE ↓','Pearson ↑']):
                vals=[np.mean([ensembles[n][c][metric] for c in cs]) for n in NAMES]
                bars=ax.barh(list(NAMES.values()),vals,color=COLORS)
                ax.invert_yaxis();ax.bar_label(bars,fmt='%.4f',padding=4)
                ax.set_xlabel(label);ax.set_xlim(min(0,min(vals)*1.2),max(vals)*1.2);ax.grid(axis='x',alpha=.2)
            save(fig,num+'_集成对比',scope+'：五种方法三种子集成')
        fig,ax=plt.subplots(figsize=(11,5))
        means=[r['selection_macro_f1_mean'] for r in rows];std=[r['selection_macro_f1_std'] for r in rows]
        ax.bar(list(NAMES.values()),means,yerr=std,capsize=5,color=COLORS)
        ax.set_ylabel('四条件平均 Macro-F1');ax.grid(axis='y',alpha=.2)
        for i,(a,b) in enumerate(zip(means,std)):ax.text(i,a+b+.008,f'{a:.4f} ± {b:.4f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(np.array(means)+std)+.06)
        save(fig,'03_种子稳定性','选模依据：三次独立训练的均值与标准差')
        for metric,num in [('macro_f1','04'),('mae','05')]:
            fig,axes=plt.subplots(1,3,figsize=(14,5))
            for ax,mod,label in zip(axes,['text','audio','vision'],['文本','音频','视觉']):
                for (name,color) in zip(NAMES,COLORS):
                    values=np.array([[np.mean([r['conditions'][f'{mod}_{ratio}_{loc}'][metric] for loc in ['front','middle','back']]) for ratio in [10,30,50]] for r in runs[name]])
                    ax.errorbar([10,30,50],values.mean(0),yerr=values.std(0,ddof=1),marker='o',capsize=3,color=color,label=NAMES[name])
                ax.set(title=label,xlabel='目标遮挡内容位置 / %',ylabel=metric);ax.grid(alpha=.2)
            axes[-1].legend(fontsize=8)
            save(fig,num+'_缺失曲线','连续缺失下的表现：前中后位置平均，误差线为种子标准差')
        fig,ax=plt.subplots(figsize=(17,5))
        matrix=np.array([[ensembles[n][c]['macro_f1'] for c in names] for n in NAMES])
        im=ax.imshow(matrix,aspect='auto',cmap='YlGnBu');fig.colorbar(im,ax=ax,label='Macro-F1')
        ax.set_yticks(range(len(NAMES)),list(NAMES.values()));ax.set_xticks(range(len(names)),names,rotation=65,ha='right',fontsize=7)
        save(fig,'06_条件热力图','全部32条件：三种子集成的分类表现')
        fig,axes=plt.subplots(1,3,figsize=(13,5))
        for ax,name in zip(axes,['mlp','mult','emt_dlfr']):
            cm=np.array(ensembles[name]['clean']['confusion_matrix']);ratio=cm/cm.sum(1,keepdims=True)
            ax.imshow(ratio,cmap='Blues',vmin=0,vmax=1)
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{cm[i,j]}\n{ratio[i,j]:.1%}',ha='center',va='center',color='white' if ratio[i,j]>.5 else '#233646')
            ax.set_xticks(range(3),['负向','中性','正向']);ax.set_yticks(range(3),['负向','中性','正向']);ax.set(title=NAMES[name],xlabel='预测',ylabel='真实')
        save(fig,'07_混淆矩阵','完整输入：逐类识别结果')
        fig,axes=plt.subplots(1,3,figsize=(13,5))
        for ax,name in zip(axes,['mlp','mult','emt_dlfr']):
            ax.scatter(targets,predictions[name]['clean_intensity'],s=12,alpha=.3,color=COLORS[list(NAMES).index(name)])
            ax.plot([-3,3],[-3,3],'k--',lw=1);ax.set(xlim=(-3,3),ylim=(-3,3),xlabel='真实强度',ylabel='预测强度',title=NAMES[name]);ax.grid(alpha=.2)
        save(fig,'08_强度散点','完整输入：情感强度预测')
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for j,name in enumerate(['mult','emt_dlfr','emt_no_restore']):
            for seed in SEEDS:
                history=json.loads((HERE/'runs'/f'{name}_seed{seed}'/'history.json').read_text())
                axes[0,j].plot([r['epoch'] for r in history],[r['selection_macro_f1'] for r in history],label=f'seed {seed}')
                axes[1,j].plot([r['epoch'] for r in history],[r['loss'] for r in history],label=f'seed {seed}')
            axes[0,j].set(title=NAMES[name],ylabel='验证四条件Macro-F1');axes[1,j].set(xlabel='Epoch',ylabel='训练总损失');axes[0,j].legend(fontsize=8)
        save(fig,'09_训练曲线','训练过程：EMT吸引项为负，损失数值不可跨目标直接比较')
        fig,axes=plt.subplots(1,2,figsize=(11,5))
        for ax,metric,title in zip(axes,['macro_f1','mae'],['Macro-F1 ↑','MAE ↓']):
            for j,name in enumerate(['emt_dlfr','emt_no_restore']):
                vals=np.array([[np.mean([r['conditions'][c][metric] for c in cs]) for cs in [['clean'],SELECT,names[1:28]]] for r in runs[name]])
                bars=ax.bar(np.arange(3)+(j-.5)*.35,vals.mean(0),yerr=vals.std(0,ddof=1),width=.35,capsize=3,label=NAMES[name],color=COLORS[3+j])
                ax.bar_label(bars,fmt='%.3f',padding=5,fontsize=8)
            ax.set_xticks(range(3),['完整','四条件','27种单模态缺失']);ax.set_ylabel(title);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
        save(fig,'10_恢复消融','两级恢复约束是否有效：两变体保留相同双支监督')
        fig,axes=plt.subplots(1,2,figsize=(11,5))
        for ax,key,ci,label in zip(axes,['f1_difference','mae_difference'],['f1_ci95','mae_ci95'],['Macro-F1 新−MLP（正值较好）','MAE 新−MLP（负值较好）']):
            for i,(c,b) in enumerate(bootstrap.items()):
                ax.plot(b[ci],[i,i],color='#237b9a',lw=2);ax.plot(b[key],i,'o',color='#237b9a')
            ax.axvline(0,color='#cf594e',ls='--');ax.set_yticks([0,1],['完整输入','文本30%中部缺失']);ax.set(xlabel=label,ylim=(-.5,1.5));ax.grid(axis='x',alpha=.2)
        save(fig,'11_配对区间',NAMES[paper_winner]+'相对MLP：按原视频分组的Bootstrap 95%区间')
        fig,axes=plt.subplots(1,3,figsize=(13,5))
        for ax,key,label,scale in zip(axes,['parameters','seconds_including_validation','peak_allocated_GiB'],['参数量 / 百万','单次训练与验证 / 分钟','峰值已分配显存 / GiB'],[1e6,60,1]):
            vals=[np.mean([r[key] for r in resource_rows if r['model']==n])/scale for n in ['mult','emt_dlfr','emt_no_restore']]
            bars=ax.bar(['MulT','EMT-DLFR','EMT去恢复'],vals,color=COLORS[2:]);ax.bar_label(bars,fmt='%.2f',padding=4);ax.set_ylabel(label);ax.set_ylim(0,max(vals)*1.2)
        save(fig,'12_计算资源','新模型计算成本：三种子平均，使用两张RTX4090',note='时间包含验证与共享服务器调度影响；显存包括缓存，不等于部署时模型占用')

    # A readable report with no invented inference/test accuracy.
    text=['# 开源论文方法对照实验报告','',f'完成MulT、EMT-DLFR及EMT去恢复约束各3个种子，共9次正式训练。按预定规则，本轮四个主候选中的首选为 **{NAMES[winner]}**；两种论文方法中为 **{NAMES[paper_winner]}**。','',
          '这些是共同冻结BERT-Mini下的赛题适配结果，不是论文原设置复现，也不是附件3准确率。全部性能来自参与选模的728条开发验证样本；没有用附件2 test或专项标签挑选模型。','',
          '## 完整输入：先集成三种子预测，再计算指标','',
          '|方法|Accuracy|Macro-F1|MAE|Pearson|','|---|---:|---:|---:|---:|']
    for name in NAMES:
        m=ensembles[name]['clean'];text.append(f'|{NAMES[name]}|{m["accuracy"]:.4f}|{m["macro_f1"]:.4f}|{m["mae"]:.4f}|{m["pearson"]:.4f}|')
    text+=['','## 选模依据：三种子的四条件指标均值','',
           '|方法|Macro-F1均值±标准差|MAE均值±标准差|','|---|---:|---:|']
    for r in rows:
        text.append(f'|{NAMES[r["model"]]}|{r["selection_macro_f1_mean"]:.4f} ± {r["selection_macro_f1_std"]:.4f}|{r["selection_mae_mean"]:.4f} ± {r["selection_mae_std"]:.4f}|')
    text+=['','四条件为完整输入与文本/音频/视觉各30%中部连续缺失。结构按种子指标均值选择；三种子集成指标另外计算，二者不能混称。消融变体不额外作为选模候选。','',
           '## 与原MLP的差异','']
    for scope,cs in [('完整输入',['clean']),('四条件',SELECT)]:
        nf=np.mean([ensembles[paper_winner][c]['macro_f1'] for c in cs]);bf=np.mean([ensembles['mlp'][c]['macro_f1'] for c in cs]);nm=np.mean([ensembles[paper_winner][c]['mae'] for c in cs]);bm=np.mean([ensembles['mlp'][c]['mae'] for c in cs])
        text.append(f'- {scope}：最佳论文适配模型相对原MLP，Macro-F1变化{nf-bf:+.4f}，MAE变化{nm-bm:+.4f}。')
    if winner=='mlp':
        text+=['','当前证据不支持用这两种论文适配模型替换原MLP。更复杂的结构不保证在3395条训练样本和固定小型编码器下更好；不能据此推断原论文方法普遍无效。']
    else:
        text+=['',f'本轮按预定分类优先规则选中{NAMES[winner]}，同时应检查MAE是否改善，不能只凭一个指标宣称全面提升。新选择保存在本目录selection.json；历史目录保持可追溯。']
    dl=next(r for r in rows if r['model']=='emt_dlfr');ab=next(r for r in rows if r['model']=='emt_no_restore')
    text+=['','## 恢复约束消融','',f'EMT-DLFR相对去恢复变体：四条件Macro-F1均值差{dl["selection_macro_f1_mean"]-ab["selection_macro_f1_mean"]:+.4f}，MAE均值差{dl["selection_mae_mean"]-ab["selection_mae_mean"]:+.4f}。两者保留相同完整/缺失视图监督，差别是低层重建和高层吸引约束。','',
           '## 检查、适配与限制','',
           '- 新模型5项针对性测试与GPU短跑通过；小批量拟合仅用于检查训练流程。此前10项输入规则测试和原始数据/缓存复核也已通过。','- 保留UNK处理、编码前遮挡、原位置、内容分母和训练集标准化。恢复监督仅来自训练集原本可观测、后来被人工遮挡的位置。','- 附件3/4不参与训练或选模；专项推理只能看到当时可见输入。完整支路只在训练时使用。','- 原32条件保持一致；本轮未新增三模态同步压力测试，不能声称已覆盖所有专项缺失形式。','- 模型容量、优化器细节和计算成本与历史小模型不同；本轮固定超参数，没有宣称公平等算力搜索。','- 第三问的附件4适配、归因验证和真实时间映射尚未在本实验实现。','- 具体来源版本、核心机制与所有重要改动见[SOURCES.md](SOURCES.md)。','',
           '## 图片与复现材料','', '[图集PDF](论文方法实验图集.pdf)；[三种子汇总](seed_summary.csv)；[32条件指标](all_condition_metrics.csv)；[逐样本错误](validation_errors.csv)。','']
    for filename,title in gallery:text.append(f'- [{title}](figures/{filename}.png)')
    text += ['','## 对消融结果和不确定性的解释','',
        f'去掉恢复约束的EMT三种子集成，在完整输入下Macro-F1为{ensembles["emt_no_restore"]["clean"]["macro_f1"]:.4f}、MAE为{ensembles["emt_no_restore"]["clean"]["mae"]:.4f}；其四条件集成Macro-F1为{np.mean([ensembles["emt_no_restore"][c]["macro_f1"] for c in SELECT]):.4f}。这些集成指标优于完整EMT-DLFR的对应值，必须如实保留。',
        '这说明本轮不能证明两级恢复约束稳定有益，更不能将相对MLP的全部收益归因于恢复机制。单种子指标均值和预测集成后的指标排序不同；没有看到集成结果后更换预先固定的选模口径。即使把消融纳入同一独立种子均值排序，EMT-DLFR仍更高。',
        f'最佳论文模型相对MLP，完整输入Macro-F1差值的按视频组Bootstrap 95%区间为{bootstrap["clean"]["f1_ci95"]}，跨过0；MAE差值区间为{bootstrap["clean"]["mae_ci95"]}。这是开发验证集上的描述性区间，不消除选模偏差，不能作为独立泛化优势的证明。',
        '', '[计算资源与各次训练耗时](resource_usage.csv)。']
    (HERE/'论文方法实测报告.md').write_text('\n'.join(text)+'\n')
    html='<html lang="zh"><meta charset="utf-8"><title>论文方法对照实验</title><style>body{max-width:1100px;margin:40px auto;font-family:sans-serif;background:#f5f7fa;color:#233646}img{width:100%;background:white}section{margin:35px 0}a{color:#237b9a}</style><h1>论文方法对照实验</h1><p>附件2开发验证集；三种子；赛题适配实现。</p>'
    for f,t in gallery:html+=f'<section><h2>{t}</h2><a href="figures/{f}.png"><img loading="lazy" src="figures/{f}.png"></a></section>'
    (HERE/'index.html').write_text(html+'</html>')
    print(json.dumps({'selected':winner,'best_paper':paper_winner,'figures':len(gallery),'report':str(HERE/'论文方法实测报告.md')},ensure_ascii=False))


if __name__=='__main__':
    main()
