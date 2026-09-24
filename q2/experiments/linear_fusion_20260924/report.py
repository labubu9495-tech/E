"""Report actual new-method results next to the frozen MLP baseline."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import json
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from data import dump_json
from train import SELECT

NAMES={'original_mlp_ensemble':'原MLP集成','pooled_linear':'统计特征线性模型','lexical_multimodal_linear':'TF-IDF＋多模态线性模型'}
COLORS=['#798B98','#DB9A45','#176B87']


def macro_f1(y,p):
    cm=np.bincount(y*3+p,minlength=9).reshape(3,3)
    den=cm.sum(0)+cm.sum(1)
    return float(np.divide(2*np.diag(cm),den,out=np.zeros(3,dtype=float),where=den>0).mean())


def main():
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'text.parse_math':False,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'font.size':10})
    selection=json.loads((HERE/'selection.json').read_text());winner=selection['selected']
    scores={'original_mlp_ensemble':json.loads((ROOT/'results/selected_validation_metrics.json').read_text())}
    for family in ['pooled_linear','lexical_multimodal_linear']:scores[family]=json.loads((HERE/f'{family}_metrics.json').read_text())
    (HERE/'figures').mkdir(exist_ok=True);pdf=PdfPages(HERE/'新方法对比图集.pdf');figure_names=[]
    def save(fig,name,title):
        fig.suptitle(title,x=.055,y=.98,ha='left',fontsize=17,fontweight='bold')
        fig.text(.055,.025,'附件2开发验证集（参与选模）｜三种子预测集成｜不是独立测试精度',fontsize=9,color='#687C89')
        fig.tight_layout(rect=[.03,.065,.99,.91]);fig.savefig(HERE/'figures'/f'{name}.png',dpi=300);fig.savefig(HERE/'figures'/f'{name}.svg');pdf.savefig(fig);plt.close(fig);figure_names.append((name,title))
    for scope,conditions in [('完整输入',['clean']),('四条件选模',SELECT)]:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,key,label in zip(axes.flat,['accuracy','macro_f1','mae','pearson'],['Accuracy ↑','Macro-F1 ↑','MAE ↓','Pearson ↑']):
            vals=[np.mean([s[c][key] for c in conditions]) for s in scores.values()]
            bars=ax.barh(list(NAMES.values()),vals,color=COLORS);ax.invert_yaxis();ax.bar_label(bars,fmt='%.4f',padding=4);ax.set_xlim(0,max(vals)*1.22);ax.set_xlabel(label);ax.grid(axis='x',alpha=.2)
        save(fig,'01_完整输入对比' if scope=='完整输入' else '02_四条件对比',f'更换方法后的实际结果：{scope}')
    fig,axes=plt.subplots(1,3,figsize=(14,6))
    for ax,(method,s) in zip(axes,scores.items()):
        cm=np.array(s['clean']['confusion_matrix']);ratio=cm/cm.sum(1,keepdims=True)
        ax.imshow(ratio,cmap='Blues',vmin=0,vmax=1)
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{cm[i,j]}\n{ratio[i,j]:.1%}',ha='center',va='center',color='white' if ratio[i,j]>.5 else '#21394A')
        ax.set_xticks(range(3),['负向','中性','正向']);ax.set_yticks(range(3),['负向','中性','正向']);ax.set(xlabel='预测类别',ylabel='真实类别',title=NAMES[method])
    save(fig,'03_混淆矩阵对比','三个方法的逐类识别表现')
    for metric in ['macro_f1','mae']:
        fig,axes=plt.subplots(1,3,figsize=(14,5.5))
        for ax,mod,name in zip(axes,['text','audio','vision'],['文本','语音','视觉']):
            for (method,s),color in zip(scores.items(),COLORS):
                val=[np.mean([s[f'{mod}_{r}_{loc}'][metric] for loc in ['front','middle','back']]) for r in [10,30,50]]
                ax.plot([10,30,50],val,'o-',color=color,label=NAMES[method])
            ax.set(title=name,xlabel='目标遮挡内容位置比例 / %',ylabel='Macro-F1 ↑' if metric=='macro_f1' else 'MAE ↓');ax.set_xticks([10,30,50]);ax.grid(alpha=.2)
        axes[-1].legend(fontsize=8)
        save(fig,'04_缺失分类对比' if metric=='macro_f1' else '05_缺失回归对比','不同连续缺失比例下的'+('分类表现' if metric=='macro_f1' else '回归误差'))
    old=dict(np.load(ROOT/'results/selected_validation_predictions.npz'));new=dict(np.load(HERE/f'{winner}_validation_predictions.npz'))
    assert np.array_equal(old['ids'],new['ids'])
    y=old['labels'];targets=old['targets'];groups=np.array([s.split('$_$')[0] for s in old['ids']]);unique=np.unique(groups);indices=[np.flatnonzero(groups==g) for g in unique]
    rng=np.random.default_rng(20260924);boot={}
    for condition in ['clean','text_30_middle']:
        op=old[condition+'_probabilities'].argmax(1);npred=new[condition+'_probabilities'].argmax(1)
        oe=np.abs(old[condition+'_intensity']-targets);ne=np.abs(new[condition+'_intensity']-targets)
        samples=[]
        for _ in range(1000):
            ix=np.concatenate([indices[i] for i in rng.integers(len(indices),size=len(indices))])
            samples.append([macro_f1(y[ix],npred[ix])-macro_f1(y[ix],op[ix]),float(np.mean(ne[ix]-oe[ix]))])
        ci=np.quantile(samples,[.025,.975],axis=0)
        boot[condition]=dict(macro_f1_new_minus_old=macro_f1(y,npred)-macro_f1(y,op),macro_f1_ci95=ci[:,0].tolist(),
            mae_new_minus_old=float(np.mean(ne-oe)),mae_ci95=ci[:,1].tolist(),video_groups=len(indices))
    dump_json(HERE/'paired_bootstrap.json',{'results':boot,'note':'Paired video-group bootstrap on model-selection validation; cannot remove selection bias or establish independent generalization.'})
    fig,axes=plt.subplots(1,2,figsize=(12,5.7))
    for ax,key,ci,label in zip(axes,['macro_f1_new_minus_old','mae_new_minus_old'],['macro_f1_ci95','mae_ci95'],['Macro-F1 新 − 旧（正值较好）','MAE 新 − 旧（负值较好）']):
        for i,(c,d) in enumerate(boot.items()):
            v=d[key];lo,hi=d[ci];ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=5,color='#176B87')
        ax.axvline(0,ls='--',color='#A85678');ax.set_yticks([0,1],['完整输入','文本30%中部缺失']);ax.set(xlabel=label,ylim=(-.5,1.5));ax.grid(axis='x',alpha=.2)
    save(fig,'06_配对差值区间','新旧模型配对差值：视频组Bootstrap 95%区间')
    pdf.close()
    best=scores[winner];base=scores['original_mlp_ensemble'];cmp=pd.read_csv(HERE/'ensemble_comparison.csv');current=next(d for d in selection['ranking'] if d['family']==winner)
    new_select_f1=np.mean([best[c]['macro_f1'] for c in SELECT]);old_select_f1=np.mean([base[c]['macro_f1'] for c in SELECT])
    new_select_mae=np.mean([best[c]['mae'] for c in SELECT]);old_select_mae=np.mean([base[c]['mae'] for c in SELECT])
    if new_select_f1<=old_select_f1:
        advice='本轮不建议替换原MLP：新候选没有提高相同四条件的平均Macro-F1。已经换方法不代表结果一定变好，保留全部负结果。'
    elif new_select_mae>old_select_mae:
        advice='新方法提高了四条件平均Macro-F1，但回归MAE变大，存在分类与回归的取舍；不能宣称四项指标全面改善。'
    else:
        advice='新方法的四条件平均Macro-F1更高、MAE更低，可作为当前更优的开发验证候选；仍需独立评估确认泛化。'
    texts=['# 更换方法实验：真实结果与建议','',f'两种新路线内部选中 **{NAMES[winner]}**。原MLP、原权重及其预测保留在原目录，新实验单独保存。','',advice,'',
        '## 为什么上次验证指标没变','',
        '上次修的是附件3的[UNK]缺失标记。train/valid不含100，因此训练输入、模型权重和验证预测均未改变；专项预测已变化。这一次才真正更换学习方法。','',
        '## 方法','',
        '第一条路线对文本/语音/视觉分别计算可用位置上的均值、标准差和观测比例，用L2正则逻辑回归做三分类、Ridge回归预测强度。第二条路线再融合可见WordPiece一元/二元TF-IDF，补充直接的词汇线索。没有跨缺失位置拼接二元词组，也没有使用专项缺失词的原始文本。','',
        '沿用原冻结BERT-Mini、aligned输入、官方划分和10%—50%连续缺失。特征标准化、词表和IDF只由训练集拟合。每个随机种子使用40%完整权重、40%单模态缺失权重、20%双模态缺失权重的预定采样方案；没有引入其他情感训练数据。三种子平均分类概率和强度。','',
        f'新方法选定：C={current["C"]}，class_weight={current["class_weight"]}，Ridge alpha={current["alpha"]}。参数范围在实验开始前记录于protocol.json。','',
        '## 完整输入：728条开发验证样本','',
        '|方法|Accuracy|Macro-F1|MAE|Pearson|','|---|---:|---:|---:|---:|']
    for method,s in scores.items():
        d=s['clean'];texts.append(f'|{NAMES[method]}|{d["accuracy"]:.4f}|{d["macro_f1"]:.4f}|{d["mae"]:.4f}|{d["pearson"]:.4f}|')
    texts += ['',f'选中新方法相对旧MLP：Accuracy变化 {(best["clean"]["accuracy"]-base["clean"]["accuracy"])*100:+.2f} 个百分点；Macro-F1变化 {best["clean"]["macro_f1"]-base["clean"]["macro_f1"]:+.4f}；MAE变化 {best["clean"]["mae"]-base["clean"]["mae"]:+.4f}。','',
        '## 相同四条件的综合结果','',
        '|方法|四条件平均Macro-F1|四条件平均MAE|','|---|---:|---:|']
    for r in cmp[cmp.scope=='selection'].itertuples():texts.append(f'|{NAMES[r.method]}|{r.macro_f1:.4f}|{r.mae:.4f}|')
    texts += ['', '四条件为完整输入及文本/语音/视觉30%中部缺失。上表是三种子预测先集成后的指标；selection.json另存按三种子指标均值选择超参数的记录，二者不能混称。','',
        '## 可用证据与限制','',
        '这是在已经看过原结果后新增的方法比较。虽然参数范围先冻结且没有使用附件3选择模型，验证集仍参与选模，增加候选也增加选择偏差。不能把提升称为独立测试精度或比赛名次保证。Bootstrap保留视频组内相关性，但无法消除验证集选择偏差。','',
        '线性统计特征方法丢失部分时序顺序；TF-IDF二元词元仅补充局部顺序，不等于完整句法理解。当前缺失增强没有专门覆盖三模态同步缺失。若只有某个指标改善，必须同时报告其他指标，不能把所有指标都写成提升。','',
        '附件3无真实标签，新方法预测只能作为预测文件交付，不能计算其准确率。原方法结果不被覆盖。','',
        '## 下一轮优先检验的方向（尚未运行）','',
        '建议首先比较“冻结文本编码器”与“仅解冻BERT-Mini最后一层”的控制变量实验，沿用相同融合头、缺失规则和选模条件，仅用附件2训练集微调，并限制学习率与轮数。此前比较的方法共享冻结表示，因此这些负结果不能证明表示本身已经足够。此方向是待验证假设，不保证提高。','',
        '其次可比较分类/回归分开的任务表示，减少联合损失的相互牵制；针对中性类弱项，可比较三分类与“中性/非中性→正/负”的层次分类。每次只改变一个核心因素，避免同时换编码器、损失、融合和缺失分布而无法解释收益。三模态同步连续缺失应作为独立标注的补充压力测试。','',
        '## 图表与文件','']
    texts += [f'- [{title}](figures/{name}.png)' for name,title in figure_names]
    texts += ['','- `classifier_candidates.csv` / `regressor_candidates.csv`：全部预定候选及种子结果。',
        '- `ensemble_comparison.csv`：相同clean/四条件/27条件比较。','- `selection.json`：新方法选择及模型哈希。','- `*_validation_predictions.npz`：32条件全量验证预测。','- `新方法对比图集.pdf`：6页矢量图。','',
        '运行：在q2目录执行 `.venv/bin/python experiments/linear_fusion_20260924/run.py`；选模后运行本目录 `infer_new.py --input <附件3对齐目录>`，输出留在新实验目录。']
    (HERE/'新方法实测报告.md').write_text('\n'.join(texts),encoding='utf-8')
    print(json.dumps({'selected':winner,'clean':best['clean'],'bootstrap':boot},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
