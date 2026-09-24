"""Read-only visualization of existing Q2 evidence; never retrains/selects models."""
from pathlib import Path
import csv
import hashlib
import html
import json
import math
import textwrap
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, precision_recall_curve, average_precision_score
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
RES=ROOT/'results'
OUT=RES/'visualization_atlas'
NAMES={'mlp':'简单融合MLP','gru_clean':'GRU·无增强','gru_aug':'GRU·缺失增强',
       'state_aug':'状态递推·缺失增强','smooth_aug':'固定平滑·缺失增强',
       'state_clean':'状态递推·无增强','state_no_gap':'状态递推·无间隔提示',
       'state_no_history':'去历史递推','text':'仅文本基线'}
MAIN=['mlp','gru_clean','gru_aug','state_aug','smooth_aug']
COLORS=dict(zip(MAIN,['#176B87','#6B7B8C','#E49B39','#A85678','#629C88']))
CLASS_NAMES=['负向','中性','正向']
CLASS_COLORS=['#467AA5','#C3A34F','#B46373']
MOD_NAMES={'text':'文本','audio':'语音','vision':'视觉'}
LOCS=['front','middle','back'];LOC_NAMES=['前部','中部','后部']
METRIC_NAMES={'accuracy':'Accuracy ↑','macro_f1':'Macro-F1 ↑','mae':'MAE ↓','pearson':'Pearson ↑'}
NOTE='附件2验证集（参与选模）｜结果为开发评价，不代表独立测试精度'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame(rows):
    return rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)


class Atlas:
    def __init__(self):
        for d in ['png','svg','data','thumbnails','cards','contact_sheets']:
            (OUT/d).mkdir(parents=True,exist_ok=True)
        font_manager.fontManager.addfont(FONT)
        plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),
            'axes.unicode_minus':False,'font.size':10,'axes.titlesize':12,'axes.labelsize':10,
            'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#BDCAD3',
            'axes.labelcolor':'#33485A','xtick.color':'#33485A','ytick.color':'#33485A',
            'text.color':'#21394A','figure.facecolor':'white','axes.facecolor':'white',
            'text.parse_math':False,
            'grid.color':'#DDE5EB','grid.alpha':.6,'svg.fonttype':'none','pdf.fonttype':42,
            'legend.frameon':False,'savefig.facecolor':'white'})
        self.entries=[];self.card_entries=[]
        self.pdf=PdfPages(OUT/'第二问可视化图集.pdf',metadata={'Title':'第二问实测结果可视化','Author':''})
        self.card_pdf=PdfPages(OUT/'样本卡片合集.pdf',metadata={'Title':'第二问预测样本卡片','Author':''})
        self.comp=pd.read_csv(RES/'model_comparison.csv').set_index('model')
        self.long=pd.read_csv(RES/'all_condition_metrics.csv')
        self.selected=json.loads((RES/'selected_validation_metrics.json').read_text())
        self.pred=dict(np.load(RES/'selected_validation_predictions.npz'))
        self.y=self.pred['labels'];self.target=self.pred['targets'];self.ids=self.pred['ids']
        self.prob=self.pred['clean_probabilities'];self.reg=self.pred['clean_intensity'];self.guess=self.prob.argmax(1)
        self.error=self.reg-self.target;self.ae=np.abs(self.error);self.correct=self.guess==self.y
        self.errors=pd.read_csv(RES/'validation_predictions_and_errors.csv')
        self.special=pd.read_csv(RES/'attachment3_predictions.csv')
        self.special['polarity_label']=self.special['polarity']
        self.special['polarity']=self.special['polarity'].map({'Negative':0,'Neutral':1,'Positive':2}).astype(int)
        self.cache={}
        for s in ['train','valid']:
            with np.load(ROOT/'cache'/f'{s}.npz') as d:
                self.cache[s]={k:d[k] for k in ['labels','targets','sequence','observed','ids','raw_text']}
        assert np.array_equal(self.cache['valid']['ids'],self.ids), 'Validation sample order differs'
        assert np.array_equal(self.cache['valid']['labels'],self.y), 'Validation labels differ'
        assert np.allclose(self.prob.sum(1),1), 'Invalid classification probabilities'
        self.hist={};self.run_metrics={};self.configs={}
        for p in sorted((ROOT/'runs').glob('*/metrics.json')):
            name=p.parent.name
            self.run_metrics[name]=json.loads(p.read_text())
            self.hist[name]=pd.DataFrame(json.loads((p.parent/'history.json').read_text()))
            self.configs[name]=json.loads((p.parent/'config.json').read_text())

    def fig(self,title,subtitle='',nrows=1,ncols=1,size=None):
        if size is None:size=(10.8,6.3) if ncols==1 else (13,5.8 if nrows==1 else 8.4)
        fig,axes=plt.subplots(nrows,ncols,figsize=size,squeeze=False)
        fig.suptitle(title,x=.055,y=.975,ha='left',fontsize=18,fontweight='bold')
        if subtitle:fig.text(.055,.918,subtitle,fontsize=9.4,color='#627786')
        return fig,axes.item() if axes.size==1 else axes.squeeze()

    def save(self,fig,title,category,caption,data,source,slug,card=False,note=NOTE):
        entries=self.card_entries if card else self.entries
        prefix=('C' if card else '')+f'{len(entries)+1:02d}'
        stem=f'{prefix}_{slug}'
        fig.text(.055,.025,note,fontsize=8,color='#6B7B88')
        fig.tight_layout(rect=[.035,.065,.98,getattr(fig,'atlas_top',.88)])
        png=OUT/('cards' if card else 'png')/f'{stem}.png'
        svg=OUT/'svg'/f'{stem}.svg'
        fig.savefig(png,dpi=300)
        fig.savefig(svg)
        (self.card_pdf if card else self.pdf).savefig(fig)
        table=frame(data)
        table.to_csv(OUT/'data'/f'{stem}.csv',index=False,encoding='utf-8-sig')
        with Image.open(png) as im:
            thumb=im.convert('RGB');thumb.thumbnail((640,400))
            thumb.save(OUT/'thumbnails'/f'{stem}.jpg',quality=88)
        entries.append(dict(number=prefix,stem=stem,title=title,category=category,caption=caption,
            source=source,png=str(png.relative_to(OUT)),svg=str(svg.relative_to(OUT)),data=f'data/{stem}.csv',
            note=note,rows=len(table)))
        plt.close(fig)
        print(f'{prefix} {title}',flush=True)

    def heat(self,ax,values,xlabels,ylabels,cmap='Blues',fmt='.3f',vmin=None,vmax=None,center=False):
        values=np.asarray(values)
        kw={}
        if center:
            bound=max(float(np.nanmax(np.abs(values))),1e-6);kw['norm']=TwoSlopeNorm(vcenter=0,vmin=-bound,vmax=bound)
        else:kw.update(vmin=vmin,vmax=vmax)
        im=ax.imshow(values,aspect='auto',cmap=cmap,**kw)
        ax.set_xticks(range(len(xlabels)),xlabels);ax.set_yticks(range(len(ylabels)),ylabels)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val=values[i,j]
                rgba=im.cmap(im.norm(val));lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(j,i,format(val,fmt),ha='center',va='center',fontsize=9,color='white' if lum<.5 else '#173546')
        return im

    def overview(self):
        fig,axes=self.fig('第二问实测结果总览','19次训练 · 5种主要候选各3个随机种子 · 728条验证样本',2,2,(13,8.5))
        ax=axes[0,0];names=['accuracy','macro_f1','mae','pearson'];vals=[self.selected['clean'][k] for k in names]
        bars=ax.bar([METRIC_NAMES[k] for k in names],vals,color=['#176B87','#398DA3','#C3A34F','#A85678'],width=.65)
        ax.set_ylim(0,1);ax.set_title('最终MLP三种子集成：完整输入');ax.grid(axis='y')
        ax.bar_label(bars,fmt='%.4f',padding=3)
        ax=axes[0,1];df=self.comp.loc[MAIN].sort_values('selection_macro_f1_mean')
        ax.errorbar(df['selection_macro_f1_mean'],range(len(df)),xerr=df['selection_macro_f1_std'],fmt='o',color='#176B87',capsize=4)
        ax.set_yticks(range(len(df)),[NAMES[x] for x in df.index]);ax.set_xlabel('预设4条件平均Macro-F1（均值±种子标准差）');ax.grid(axis='x')
        ax=axes[1,0];rec=np.diag(self.selected['clean']['confusion_matrix'])/np.sum(self.selected['clean']['confusion_matrix'],axis=1)
        bars=ax.bar(CLASS_NAMES,rec,color=CLASS_COLORS);ax.bar_label(bars,labels=[f'{x:.1%}' for x in rec],padding=3);ax.set_ylim(0,1);ax.set_title('类别召回率：中性类是当前短板');ax.grid(axis='y')
        ax=axes[1,1]
        for m,color in zip(MOD_NAMES,['#176B87','#E49B39','#A85678']):
            v=[np.mean([self.selected[f'{m}_{r}_{l}']['macro_f1'] for l in LOCS]) for r in [10,30,50]]
            ax.plot([10,30,50],v,'o-',label=MOD_NAMES[m],color=color)
        ax.set(xlabel='目标遮挡位置比例 / %',ylabel='Macro-F1',title='最终集成：各位置平均缺失表现');ax.legend();ax.grid()
        rows=[dict(type='selected_clean',metric=k,value=v) for k,v in zip(names,vals)]
        rows += [dict(type=m,metric='selection_macro_f1_seed_mean',value=self.comp.loc[m,'selection_macro_f1_mean']) for m in MAIN]
        rows += [dict(type=m,metric='selection_macro_f1_seed_std',value=self.comp.loc[m,'selection_macro_f1_std']) for m in MAIN]
        rows += [dict(type=f'class_{c}',metric='recall',value=float(v)) for c,v in enumerate(rec)]
        rows += [dict(type=f'{m}_{r}',metric='location_mean_macro_f1',value=float(np.mean([self.selected[f'{m}_{r}_{loc}']['macro_f1'] for loc in LOCS]))) for m in MOD_NAMES for r in [10,30,50]]
        self.save(fig,'第二问实测结果总览','总览','状态递推当前未显示优于简单融合的证据；最终集成指标与单次训练的种子均值是两种统计量。',rows,'model_comparison.csv; selected_validation_metrics.json','结果总览')

    def data_figures(self):
        fig,ax=self.fig('训练与验证的情感类别分布','使用官方划分；不重新随机切分')
        rows=[];x=np.arange(3)
        for j,(split,label,color) in enumerate([('train','训练集','#176B87'),('valid','验证集','#D69A46')]):
            c=np.bincount(self.cache[split]['labels'],minlength=3);v=c/c.sum()
            bars=ax.bar(x+(j-.5)*.32,v,.32,label=f'{label} n={c.sum()}',color=color)
            ax.bar_label(bars,labels=[f'{n}\n{p:.1%}' for n,p in zip(c,v)],padding=3,fontsize=10)
            rows.extend(dict(split=split,polarity=i,n=int(c[i]),proportion=float(v[i])) for i in range(3))
        ax.set_xticks(x,CLASS_NAMES);ax.set_ylim(0,.62);ax.set_ylabel('样本占比');ax.legend();ax.grid(axis='y')
        self.save(fig,'训练与验证的情感类别分布','数据审查','正向样本占比较高，不能只看总体准确率；应同时观察Macro-F1和各类召回率。',rows,'cache/train.npz; cache/valid.npz','类别分布',note='附件2官方训练/验证划分｜本图未使用test标签')

        fig,axes=self.fig('情感强度分布与中性点质量','连续回归标签包含大量精确零值；不将任务改成离散强度分类',ncols=2)
        rows=[]
        for ax,split,title in zip(axes,['train','valid'],['训练集','验证集']):
            a=self.cache[split]['targets'];ax.hist(a,bins=np.linspace(-3.05,3.05,38),color='#176B87',alpha=.85)
            ax.axvline(0,color='#C3A34F',ls='--');ax.set(title=title,xlabel='真实情感强度',ylabel='样本数');ax.grid(axis='y')
            ax.text(.03,.93,f'精确零值：{np.sum(a==0)}/{len(a)}',transform=ax.transAxes,va='top')
            rows.extend(dict(split=split,index=i,target=float(v)) for i,v in enumerate(a))
        self.save(fig,'情感强度分布与中性点质量','数据审查','零标签只属于中性。标签分布用于理解难度，不据此修改真实标签。',rows,'cache/train.npz; cache/valid.npz','强度分布',note='附件2训练/验证标签｜标准化与训练参数仅由训练集拟合')

        fig,axes=self.fig('有效序列长度分布','有效内容位置由SEP端点确定；位置数不等于词数或秒数',ncols=2)
        rows=[]
        for split,color in [('train','#176B87'),('valid','#C4864F')]:
            a=self.cache[split]['sequence'].sum(1)
            axes[0].hist(a,bins=np.arange(.5,50,3),density=True,alpha=.5,label=split,color=color)
            axes[1].plot(np.sort(a),np.arange(1,len(a)+1)/len(a),label=split,color=color,lw=2)
            rows.extend(dict(split=split,sample_id=str(s),content_positions=int(v)) for s,v in zip(self.cache[split]['ids'],a))
        axes[0].set(xlabel='有效内容位置数',ylabel='概率密度');axes[1].set(xlabel='有效内容位置数',ylabel='累计样本比例')
        for ax in axes:ax.legend();ax.grid()
        self.save(fig,'有效序列长度分布','数据审查','存储长度为50，但真实内容长度随样本变化；遮挡预算不能把填充算进分母。',rows,'cache/*: sequence','有效长度分布',note='仅绘制train/valid｜CLS、SEP及padding已排除')

        fig,axes=self.fig('内容范围内的操作性观测可用率','有限非零音视频行记为可用；全零的真实成因未知',ncols=2)
        rows=[]
        for ax,split,title in zip(axes,['train','valid'],['训练集','验证集']):
            d=self.cache[split];rat=d['observed'].sum(1)/d['sequence'].sum(1)[:,None]
            ax.boxplot([rat[:,m] for m in range(3)],tick_labels=list(MOD_NAMES.values()),showfliers=False,patch_artist=True,
                       boxprops={'facecolor':'#D4E7ED'},medianprops={'color':'#176B87'})
            ax.set(title=title,ylabel='可用位置数 / 内容位置数',ylim=(-.05,1.05));ax.grid(axis='y')
            rows.extend(dict(split=split,sample_id=str(d['ids'][i]),modality=m,availability=float(rat[i,j])) for i in range(len(rat)) for j,m in enumerate(MOD_NAMES))
        self.save(fig,'内容范围内的操作性观测可用率','数据审查','这是适配器定义下的观测可用率，不是真实随机缺失率。箱线图不展示离群点，明细表保留全部样本。',rows,'cache/*: sequence, observed','观测可用率',note='操作性标记｜不把全部全零行解释为人工缺失')

        fig,axes=self.fig('验证样本的观测状态图','按内容长度排序，等间隔抽取60条；横轴保留官方存储位置',nrows=3,size=(12,9))
        d=self.cache['valid'];order=np.argsort(d['sequence'].sum(1),kind='stable');ix=order[np.linspace(0,len(order)-1,60).astype(int)]
        cmap=ListedColormap(['#E5EAEE','#E5B360','#2C849C']);rows=[]
        for j,ax in enumerate(axes):
            z=np.where(d['sequence'][ix],np.where(d['observed'][ix,:,j],2,1),0)
            ax.imshow(z,aspect='auto',cmap=cmap,vmin=0,vmax=2,interpolation='nearest');ax.set(title=list(MOD_NAMES.values())[j],ylabel='抽样行（按长度排序）');ax.set_yticks([0,19,39,59],[1,20,40,60])
            rows.extend(dict(sample_id=str(d['ids'][i]),display_row=r,position=t,modality=list(MOD_NAMES)[j],state=int(z[r,t])) for r,i in enumerate(ix) for t in range(50))
        axes[-1].set_xlabel('官方存储位置索引（0—49）');fig.legend(handles=[Patch(color=c,label=s) for c,s in zip(cmap.colors,['特殊标记/填充','内容内不可用','内容内可用'])],ncol=3,loc='upper left',bbox_to_anchor=(.055,.895),fontsize=9);fig.atlas_top=.84
        self.save(fig,'验证样本的观测状态图','数据审查','灰色位置不进入池化；黄色位置仍属于真实序列。二者不能合并成一个“缺失率”。',rows,'cache/valid.npz: sequence, observed','观测状态图')

        fig,axes=self.fig('人工连续缺失的输入状态示例','同一样本：完整、文本中部30%、音视频同步30%、音视频错位30%',nrows=4,size=(12,8.8))
        cond=['clean','text_30_middle','audio_vision_30_sync','audio_vision_30_offset']
        i=int(np.argmin(abs(d['sequence'].sum(1)-30)));masks=np.load(ROOT/'cache'/'valid_conditions.npz');rows=[]
        cmap=ListedColormap(['#E5EAEE','#E5B360','#2C849C','#B66379'])
        for ax,c in zip(axes,cond):
            obs=masks[c][i];base=d['observed'][i];p=d['sequence'][i]
            z=np.where(p[:,None],np.where(base,np.where(obs,2,3),1),0).T
            ax.imshow(z,aspect='auto',cmap=cmap,vmin=0,vmax=3,interpolation='nearest');ax.set_yticks(range(3),list(MOD_NAMES.values()));ax.set_title(c,fontsize=10,loc='left')
            rows.extend(dict(condition=c,sample_id=str(d['ids'][i]),modality=list(MOD_NAMES)[m],position=t,state=int(z[m,t])) for m in range(3) for t in range(50))
        axes[-1].set_xlabel('官方序列位置索引');fig.legend(handles=[Patch(color=c,label=s) for c,s in zip(cmap.colors,['特殊/填充','原不可用','可用','新增遮挡'])],loc='upper left',bbox_to_anchor=(.055,.895),ncol=4,fontsize=9);fig.atlas_top=.84
        self.save(fig,'人工连续缺失的输入状态示例','数据审查','示例取内容长度最接近30的位置样本。遮挡改变观测状态，不删除序列位置。',rows,'cache/valid_conditions.npz; cache/valid.npz','连续缺失示例')

    def comparison_figures(self):
        for scope,title,slug in [('clean','主要模型：完整输入四项指标','完整输入模型对比'),('selection','主要模型：预设选模条件四项指标','选模条件模型对比'),('grid27','主要模型：27种缺失条件四项指标','缺失条件模型对比')]:
            fig,axes=self.fig(title,'每点为3种子均值；误差线为样本标准差，非置信区间',2,2,(12,8.7))
            for ax,metric in zip(axes.flat,METRIC_NAMES):
                vals=self.comp.loc[MAIN,f'{scope}_{metric}_mean'];sd=self.comp.loc[MAIN,f'{scope}_{metric}_std']
                for j,m in enumerate(MAIN):
                    ax.errorbar(vals.loc[m],j,xerr=sd.loc[m],fmt='o',color=COLORS[m],capsize=4,ms=7)
                ax.set_yticks(range(5),[NAMES[m] for m in MAIN],fontsize=9);ax.invert_yaxis();ax.set_xlabel(METRIC_NAMES[metric]);ax.grid(axis='x')
            self.save(fig,title,'模型比较','各模型使用共同输入和评价遮挡；三种子波动不可直接解释为显著性。',self.comp.loc[MAIN].reset_index(),'model_comparison.csv',slug)

        fig,ax=self.fig('全部模型与消融：选模指标分布','单种子配置明确标注n=1，不与三种子证据混称稳定提升',size=(12,7.6))
        order=self.comp.sort_values('selection_macro_f1_mean').index.tolist();rows=[]
        for j,m in enumerate(order):
            vals=[]
            for run,d in self.run_metrics.items():
                if d['config']==m:vals.append(d['selection_macro_f1']);rows.append(dict(model=m,seed=d['seed'],selection_macro_f1=d['selection_macro_f1']))
            ax.scatter(vals,np.full(len(vals),j),s=48,alpha=.75,color=COLORS.get(m,'#93A3AD'))
            ax.scatter(np.mean(vals),j,marker='D',s=45,color='#21394A')
        ax.set_yticks(range(len(order)),[f'{NAMES[m]}  n={int(self.comp.loc[m,"n_seeds"])}' for m in order]);ax.set_xlabel('完整 + 文本/语音/视觉30%中部缺失：平均Macro-F1');ax.grid(axis='x')
        self.save(fig,'全部模型与消融：选模指标分布','模型比较','圆点是各次训练，深色菱形是均值；单种子消融仅提供初步证据。',rows,'runs/*/metrics.json','全部模型种子分布')

        fig,axes=self.fig('状态递推模型的控制变量消融','全部使用seed=17；收益以实际差值呈现，不预设新增模块有效',ncols=2,size=(13,6.4))
        mods=['state_aug','state_clean','state_no_gap','state_no_history'];rows=[]
        for ax,metric in zip(axes,['macro_f1','mae']):
            vals=[]
            for m in mods:
                d=self.run_metrics[f'{m}_seed17'];v=np.mean([d['conditions'][c][metric] for c in ['clean','text_30_middle','audio_30_middle','vision_30_middle']]);vals.append(v);rows.append(dict(model=m,seed=17,metric=metric,value=v))
            ax.barh(range(4),vals,color=['#A85678','#C2B0B9','#C2B0B9','#C2B0B9']);ax.set_yticks(range(4),[NAMES[m] for m in mods]);ax.invert_yaxis();ax.set_xlabel(METRIC_NAMES[metric]);ax.set_xlim(0,max(vals)*1.14)
            for i,v in enumerate(vals):ax.text(v+.005,i,f'{v:.4f}',va='center')
            ax.grid(axis='x')
        self.save(fig,'状态递推模型的控制变量消融','模型比较','共享同一随机种子的消融；差距小于重复实验波动时不宣称可靠改进。',rows,'runs/state*/metrics.json','递推消融')

        fig,ax=self.fig('预测精度与回归误差的权衡','每点为一个随机种子；向右、向下分别代表分类更好、回归误差更小')
        rows=[]
        for m in MAIN:
            df=self.long[(self.long.model==m)&(self.long.condition=='clean')]
            ax.scatter(df.macro_f1,df.mae,label=NAMES[m],s=65,color=COLORS[m],alpha=.85)
            rows.extend(df.to_dict('records'))
        ax.set(xlabel='完整验证集Macro-F1',ylabel='完整验证集MAE');ax.legend(ncol=2);ax.grid()
        self.save(fig,'预测精度与回归误差的权衡','模型比较','分类与回归不一定同步改善。按既定选模规则选模型，不能事后只挑有利指标。',rows,'all_condition_metrics.csv','分类回归权衡')

        fig,axes=self.fig('模型规模与本轮运行成本','小型预测头及共享冻结文本编码器分开计数；耗时含该运行的训练与评价',ncols=2,size=(13,6))
        rows=[]
        for m in MAIN:
            configs=[c for k,c in self.configs.items() if c['name']==m];times=[d['train_seconds'] for d in self.run_metrics.values() if d['config']==m]
            rows.append(dict(model=m,registered_predictor_parameters=configs[0]['parameters'],mean_run_seconds=np.mean(times),std_run_seconds=np.std(times,ddof=1),n_runs=len(times),device=configs[0]['device']))
        df=pd.DataFrame(rows)
        axes[0].barh([NAMES[m] for m in df.model],df.registered_predictor_parameters/1000,color=[COLORS[m] for m in df.model]);axes[0].set_xlabel('注册预测器参数量 / 千（不含冻结BERT）')
        axes[1].barh([NAMES[m] for m in df.model],df.mean_run_seconds,xerr=df.std_run_seconds,color=[COLORS[m] for m in df.model],capsize=4);axes[1].set_xlabel('单次训练与评价实测耗时 / 秒')
        for ax in axes:ax.invert_yaxis();ax.grid(axis='x')
        self.save(fig,'模型规模与本轮运行成本','模型比较','共享服务器CPU运行，早停轮数不同；耗时不是受控硬件基准。注册参数包括部分分支未使用的公共模块，不等于有效计算参数量。',df,'runs/*/config.json; metrics.json','模型规模与耗时')

    def training_figures(self):
        rows=[]
        for key,h in self.hist.items():
            for _,r in h.iterrows():
                rows.append(dict(run=key,epoch=r.epoch,train_loss=r.train_loss,selection_macro_f1=r.selection_macro_f1,selection_mae=r.selection_mae,clean_macro_f1=r['clean']['macro_f1'],elapsed_seconds=r.elapsed_seconds))
        data=pd.DataFrame(rows)
        for col,title,ylabel,slug in [('train_loss','训练损失曲线','训练联合损失（CE + Huber）','训练损失'),('selection_macro_f1','验证选模指标随训练变化','预设4条件平均Macro-F1','选模训练曲线'),('selection_mae','验证回归误差随训练变化','预设4条件平均MAE','回归训练曲线')]:
            fig,axes=self.fig(title,'每条细线为一个随机种子；没有在早停之后外推或补齐曲线',2,3,(14,8.5))
            for ax,m in zip(axes.flat,MAIN):
                for seed,ls in zip([17,29,43],['-','--',':']):
                    key=f'{m}_seed{seed}';d=data[data.run==key];ax.plot(d.epoch,d[col],ls=ls,color=COLORS[m],label=f'seed {seed}',lw=1.8)
                    best=self.run_metrics[key]['best_epoch'];r=d[d.epoch==best].iloc[0];ax.scatter(best,r[col],s=25,color=COLORS[m])
                ax.set(title=NAMES[m],xlabel='Epoch',ylabel=ylabel);ax.grid();ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.legend(fontsize=8)
            axes.flat[-1].axis('off');axes.flat[-1].text(.05,.8,'点标记：最终选定轮次\n选模依据：4条件平均Macro-F1\n不是按训练损失最低点选模',transform=axes.flat[-1].transAxes,fontsize=12,linespacing=2)
            self.save(fig,title,'训练过程','共同早停规则下的实际训练轨迹；验证集用于选模，不能以曲线平滑证明泛化。',data,'runs/*/history.json',slug)

        fig,ax=self.fig('19次运行的选定轮次与停止轮次','横线表示选定检查点至训练停止之间的间隔',size=(12,9.5))
        rows=[]
        for j,(key,h) in enumerate(self.hist.items()):
            best=self.run_metrics[key]['best_epoch'];last=int(h.epoch.max());m=self.run_metrics[key]['config']
            ax.plot([best,last],[j,j],color=COLORS.get(m,'#93A3AD'),lw=3);ax.scatter(best,j,marker='D',s=35,color='#176B87');ax.scatter(last,j,marker='|',s=100,color='#A85678')
            rows.append(dict(run=key,best_epoch=best,stop_epoch=last))
        ax.set_yticks(range(len(rows)),[r['run'] for r in rows],fontsize=9);ax.invert_yaxis();ax.set_xlabel('Epoch');ax.grid(axis='x');ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.save(fig,'19次运行的选定轮次与停止轮次','训练过程','不同模型的停止轮次不同，不能把单次总耗时直接解释为网络固有速度。',rows,'runs/*/history.json; metrics.json','早停轮次')

    def classification_figures(self):
        cm=np.array(self.selected['clean']['confusion_matrix'])
        for normalized in [False,True]:
            title='分类混淆矩阵：按真实类别归一化' if normalized else '分类混淆矩阵：样本计数'
            fig,ax=self.fig(title,'最终MLP三种子集成 · 完整验证集 n=728')
            values=cm/cm.sum(1,keepdims=True) if normalized else cm
            im=self.heat(ax,values,CLASS_NAMES,CLASS_NAMES,fmt='.1%' if normalized else 'd',vmin=0,vmax=1 if normalized else None)
            fig.colorbar(im,ax=ax,label='该真实类别中的比例' if normalized else '样本数',shrink=.85)
            ax.set(xlabel='预测类别',ylabel='真实类别')
            rows=[dict(true=i,predicted=j,count=int(cm[i,j]),recall_fraction=float(cm[i,j]/cm[i].sum())) for i in range(3) for j in range(3)]
            self.save(fig,title,'分类诊断','中性样本常被预测为正向；行归一化可避免多数类样本数掩盖类别短板。',rows,'selected_validation_metrics.json','混淆矩阵_比例' if normalized else '混淆矩阵_计数')

        prec,rec,f1,support=precision_recall_fscore_support(self.y,self.guess,labels=[0,1,2],zero_division=0)
        fig,ax=self.fig('各类别的精确率、召回率与F1','最终集成；图中给出每类真实样本量')
        for j,(v,label,color) in enumerate([(prec,'精确率','#176B87'),(rec,'召回率','#E49B39'),(f1,'F1','#A85678')]):
            bars=ax.bar(np.arange(3)+(j-1)*.24,v,width=.24,label=label,color=color);ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
        ax.set_xticks(range(3),[f'{n}\nn={s}' for n,s in zip(CLASS_NAMES,support)]);ax.set_ylim(0,1);ax.legend(ncol=3);ax.grid(axis='y')
        self.save(fig,'各类别的精确率、召回率与F1','分类诊断','中性类别的召回率偏低。后续改进应结合类别代价与实际验证，不能只提高总体准确率。',[dict(polarity=i,precision=float(prec[i]),recall=float(rec[i]),f1=float(f1[i]),support=int(support[i])) for i in range(3)],'selected_validation_predictions.npz','分类逐类指标')

        fig,axes=self.fig('分类概率分布与预测置信度','概率为模型输出，未经校准；高置信度不保证正确',ncols=2)
        conf=self.prob.max(1);rows=[]
        for ok,label,color in [(True,'预测正确','#176B87'),(False,'预测错误','#A85678')]:
            a=conf[self.correct==ok];axes[0].hist(a,bins=np.linspace(1/3,1,15),alpha=.6,label=f'{label} n={len(a)}',color=color)
        axes[0].set(xlabel='最大类别概率',ylabel='样本数');axes[0].legend();axes[0].grid(axis='y')
        actual=self.prob[np.arange(len(self.y)),self.y]
        axes[1].boxplot([actual[self.y==i] for i in range(3)],tick_labels=CLASS_NAMES,showfliers=False,patch_artist=True,boxprops={'facecolor':'#D4E7ED'})
        axes[1].set(xlabel='真实类别',ylabel='分配给真实类别的概率',ylim=(0,1));axes[1].grid(axis='y')
        rows=[dict(sample_id=str(s),true_class=int(y),prediction=int(p),confidence=float(c),true_class_probability=float(t),correct=bool(k)) for s,y,p,c,t,k in zip(self.ids,self.y,self.guess,conf,actual,self.correct)]
        self.save(fig,'分类概率分布与预测置信度','分类诊断','正确与错误预测的置信度分布有重叠；不能把最大概率当作已验证的正确概率。',rows,'selected_validation_predictions.npz','置信度分布')

        fig,axes=self.fig('预测概率可靠性诊断','固定10个等宽概率分箱；不进行温度缩放或阈值再优化',ncols=2)
        rows=[];conf=self.prob.max(1);binid=np.minimum((conf*10).astype(int),9);ece=0.
        for b in range(10):
            ix=binid==b
            if not ix.any():continue
            c=float(conf[ix].mean());acc=float(self.correct[ix].mean());n=int(ix.sum());ece+=n/len(conf)*abs(c-acc)
            rows.append(dict(bin=b,lower=b/10,upper=(b+1)/10,n=n,mean_confidence=c,accuracy=acc))
        df=pd.DataFrame(rows)
        axes[0].plot([0,1],[0,1],'--',color='#8B9BA6',label='理想参考线')
        axes[0].plot(df.mean_confidence,df.accuracy,'o-',color='#176B87',label=f'ECE(10箱)={ece:.3f}')
        axes[0].set(xlabel='分箱平均最大概率',ylabel='分箱实际准确率',xlim=(0,1),ylim=(0,1));axes[0].legend();axes[0].grid()
        axes[1].bar((df.lower+df.upper)/2,df.n,width=.08,color='#176B87');axes[1].set(xlabel='最大类别概率分箱',ylabel='样本数',xlim=(0,1));axes[1].grid(axis='y')
        self.save(fig,'预测概率可靠性诊断','分类诊断','ECE依赖分箱，且当前验证集参与了选模；本图仅诊断，不声称概率已经校准。',df,'selected_validation_predictions.npz','概率可靠性')

        for kind in ['roc','pr']:
            fig,ax=self.fig('三分类一对其余ROC曲线' if kind=='roc' else '三分类一对其余PR曲线','完整验证集；曲线用于诊断，未据此调整决策阈值')
            rows=[]
            for c in range(3):
                binary=self.y==c
                if kind=='roc':
                    x,y,thr=roc_curve(binary,self.prob[:,c]);score=auc(x,y);label=f'{CLASS_NAMES[c]} AUC={score:.3f}'
                    rows.extend(dict(polarity=c,false_positive_rate=float(a),true_positive_rate=float(b)) for a,b in zip(x,y))
                else:
                    pr,re,thr=precision_recall_curve(binary,self.prob[:,c]);x,y=re,pr;score=average_precision_score(binary,self.prob[:,c]);label=f'{CLASS_NAMES[c]} AP={score:.3f}'
                    rows.extend(dict(polarity=c,recall=float(a),precision=float(b)) for a,b in zip(x,y))
                    ax.axhline(binary.mean(),ls=':',color=CLASS_COLORS[c],alpha=.5)
                ax.plot(x,y,color=CLASS_COLORS[c],label=label,lw=2)
            if kind=='roc':ax.plot([0,1],[0,1],'--',color='#9CAAB3')
            ax.set(xlabel='假阳性率' if kind=='roc' else '召回率',ylabel='真阳性率' if kind=='roc' else '精确率',xlim=(0,1),ylim=(0,1.03));ax.legend();ax.grid()
            title='三分类一对其余ROC曲线' if kind=='roc' else '三分类一对其余PR曲线'
            self.save(fig,title,'分类诊断','逐类one-vs-rest指标，非题面四项核心指标的替代；PR虚线是各类样本占比参考线。' if kind=='pr' else 'AUC评价概率排序，不等于当前三分类准确率；主结论仍报告Accuracy、F1、MAE、Pearson。',rows,'selected_validation_predictions.npz',kind.upper()+'曲线')

        fig,ax=self.fig('中性类别：真实样本去向与预测来源','左侧按真实中性统计；右侧按预测中性统计')
        source=cm[:,1];dest=cm[1];x=np.arange(3)
        bars=ax.bar(x-.18,dest,.36,label=f'真实中性 → 预测类别（n={dest.sum()}）',color='#176B87')
        ax.bar_label(bars,padding=3)
        bars=ax.bar(x+.18,source,.36,label=f'预测中性 ← 真实类别（n={source.sum()}）',color='#C3A34F')
        ax.bar_label(bars,padding=3);ax.set_xticks(x,CLASS_NAMES);ax.set_ylabel('样本数');ax.set_ylim(0,max(source.max(),dest.max())*1.3);ax.legend();ax.grid(axis='y')
        self.save(fig,'中性类别：真实样本去向与预测来源','分类诊断','真实中性被错分为正向较多；需区分召回不足与预测中性的精确率问题。',[dict(polarity=c,true_neutral_destination=int(dest[c]),predicted_neutral_source=int(source[c])) for c in range(3)],'selected_validation_metrics.json','中性错误流向')

    def regression_figures(self):
        data=pd.DataFrame({'sample_id':self.ids,'true_intensity':self.target,'predicted_intensity':self.reg,
                           'residual':self.error,'absolute_error':self.ae,'true_class':self.y})
        fig,ax=self.fig('情感强度：预测值与真实值','残差定义为预测值减真实值；每个点是一条验证样本')
        for c in range(3):
            ix=self.y==c;ax.scatter(self.target[ix],self.reg[ix],s=19,alpha=.42,color=CLASS_COLORS[c],label=f'{CLASS_NAMES[c]} n={ix.sum()}',edgecolors='none')
        ax.plot([-3,3],[-3,3],'--',color='#7D8A95',label='理想预测');ax.axhline(0,color='#DFE5EB');ax.axvline(0,color='#DFE5EB')
        ax.set(xlabel='真实情感强度',ylabel='预测情感强度',xlim=(-3.15,3.15),ylim=(-3.15,3.15));ax.legend();ax.grid()
        self.save(fig,'情感强度：预测值与真实值','回归诊断','极端强度的预测向中心收缩；没有为分离重叠点而更改标签坐标。',data,'selected_validation_predictions.npz','回归散点')

        fig,axes=self.fig('回归残差与误差累积分布','展示误差方向、误差大小和长尾',ncols=2)
        axes[0].hist(self.error,bins=32,color='#176B87',alpha=.85);axes[0].axvline(0,color='#A85678',ls='--');axes[0].set(xlabel='残差：预测 − 真实',ylabel='样本数');axes[0].grid(axis='y')
        s=np.sort(self.ae);axes[1].plot(s,np.arange(1,len(s)+1)/len(s),color='#176B87',lw=2)
        for q in [.5,.9]:
            v=np.quantile(s,q);axes[1].axhline(q,color='#9AACB7',ls=':',lw=1);axes[1].text(v,q-.04,f'P{int(q*100)}={v:.3f}',fontsize=9)
        axes[1].set(xlabel='绝对误差',ylabel='累计样本比例',ylim=(0,1.02));axes[1].grid()
        self.save(fig,'回归残差与误差累积分布','回归诊断','分位数补充MAE，描述典型误差和尾部；这些是同一验证集的经验分布。',data,'selected_validation_predictions.npz','回归误差分布')

        fig,axes=self.fig('残差随真实强度与预测强度变化','图中的系统偏差用于定位问题，不据此重新修正当前提交结果',ncols=2)
        for ax,x,name in zip(axes,[self.target,self.reg],['真实情感强度','预测情感强度']):
            ax.scatter(x,self.error,s=15,alpha=.35,color='#176B87',edgecolors='none');ax.axhline(0,color='#A85678',ls='--');ax.set(xlabel=name,ylabel='预测 − 真实');ax.grid()
        self.save(fig,'残差随真实强度与预测强度变化','回归诊断','真实强度较大时常低估、较小时常高估，体现向中心收缩；横轴与残差共享变量，应避免因果解释。',data,'selected_validation_predictions.npz','残差结构')

        fig,axes=self.fig('不同真实强度区间的分类与回归表现','区间按绝对强度划分；精确零值独立统计',ncols=2)
        mag=np.abs(self.target);masks=[mag==0,(mag>0)&(mag<=.5),(mag>.5)&(mag<=1),(mag>1)&(mag<=2),mag>2]
        labels=['精确0','(0, 0.5]','(0.5, 1]','(1, 2]','(2, 3]'];rows=[]
        for label,ix in zip(labels,masks):rows.append(dict(bin=label,n=int(ix.sum()),mae=float(self.ae[ix].mean()),accuracy=float(self.correct[ix].mean())))
        df=pd.DataFrame(rows)
        for ax,col,name,color in zip(axes,['mae','accuracy'],['MAE ↓','Accuracy ↑'],['#176B87','#A85678']):
            bars=ax.bar(range(len(df)),df[col],color=color);ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
            ax.set_xticks(range(len(df)),[f'{r.bin}\nn={r.n}' for r in df.itertuples()],fontsize=9);ax.set(xlabel='真实强度绝对值',ylabel=name,ylim=(0,max(df[col])*1.25));ax.grid(axis='y')
        self.save(fig,'不同真实强度区间的分类与回归表现','回归诊断','弱非零情感与精确中性不是同一组。大强度区间样本较少，同时报告样本量。',df,'selected_validation_predictions.npz','强度分层误差')

        fig,axes=self.fig('按真实类别分组的回归误差','箱体为四分位区间；须为1.5倍IQR范围内的观测值',ncols=2)
        for ax,values,name in zip(axes,[self.error,self.ae],['残差：预测 − 真实','绝对误差']):
            b=ax.boxplot([values[self.y==i] for i in range(3)],tick_labels=[f'{CLASS_NAMES[i]}\nn={(self.y==i).sum()}' for i in range(3)],patch_artist=True,flierprops={'markersize':3,'alpha':.4})
            for box,color in zip(b['boxes'],CLASS_COLORS):box.set_facecolor(color);box.set_alpha(.7)
            ax.axhline(0,color='#9AAAB5',ls=':');ax.set_ylabel(name);ax.grid(axis='y')
        self.save(fig,'按真实类别分组的回归误差','回归诊断','类别分组展示误差方向差异，不把箱线图跨度解释为均值置信区间。',data,'selected_validation_predictions.npz','类别回归误差')

        regclass=np.where(self.reg<0,0,np.where(self.reg>0,2,1));mat=np.zeros((3,3),int)
        np.add.at(mat,(self.guess,regclass),1)
        fig,ax=self.fig('分类输出与回归符号的一致性','分类头与回归头独立输出；回归恰好为0时才归入零值列')
        im=self.heat(ax,mat,['强度<0','强度=0','强度>0'],CLASS_NAMES,fmt='d');fig.colorbar(im,ax=ax,label='样本数');ax.set(xlabel='回归输出符号',ylabel='分类预测')
        rows=[dict(predicted_class=i,regression_sign=j-1,n=int(mat[i,j])) for i in range(3) for j in range(3)]
        self.save(fig,'分类输出与回归符号的一致性','回归诊断','负向分类与正强度、正向分类与负强度为方向冲突；中性分类伴随非零强度需另行理解，不在此强行统一阈值。',rows,'selected_validation_predictions.npz','双任务一致性')

    def missing_figures(self):
        grid=[f'{m}_{r}_{loc}' for m in MOD_NAMES for r in [10,30,50] for loc in LOCS]
        df=self.long[self.long.model.isin(MAIN)&self.long.condition.isin(grid)].copy()
        df[['modality','ratio','location']]=df.condition.str.extract(r'^(text|audio|vision)_(10|30|50)_(front|middle|back)$')
        df['ratio']=df.ratio.astype(int)
        for metric in METRIC_NAMES:
            fig,axes=self.fig(f'主要模型的缺失比例响应：{METRIC_NAMES[metric]}','先在每个种子内平均前/中/后位置，再计算3种子均值和样本标准差',ncols=3,size=(15,6))
            data=df.groupby(['model','seed','modality','ratio'],as_index=False)[metric].mean()
            agg=data.groupby(['model','modality','ratio'])[metric].agg(['mean','std']).reset_index()
            for ax,mod in zip(axes,MOD_NAMES):
                for m in MAIN:
                    d=agg[(agg.model==m)&(agg.modality==mod)].sort_values('ratio')
                    ax.errorbar(d.ratio,d['mean'],yerr=d['std'],fmt='o-',capsize=3,label=NAMES[m],color=COLORS[m],lw=1.5,ms=4)
                ax.set(title=MOD_NAMES[mod],xlabel='目标遮挡位置比例 / %',ylabel=METRIC_NAMES[metric]);ax.set_xticks([10,30,50]);ax.grid()
            axes[-1].legend(fontsize=8,loc='best')
            self.save(fig,f'主要模型的缺失比例响应：{METRIC_NAMES[metric]}','连续缺失','误差线反映训练随机性；位置条件共享样本，未当作独立重复。横轴不代表秒数。',agg,'all_condition_metrics.csv',f'模型缺失曲线_{metric}')

        for metric in METRIC_NAMES:
            rows=[];values=[]
            for mod in MOD_NAMES:
                z=np.array([[self.selected[f'{mod}_{r}_{loc}'][metric] for loc in LOCS] for r in [10,30,50]]);values.append(z)
                rows.extend(dict(modality=mod,ratio=r,location=loc,metric=metric,value=float(z[i,j])) for i,r in enumerate([10,30,50]) for j,loc in enumerate(LOCS))
            fig,axes=self.fig(f'最终集成：缺失比例×位置的{METRIC_NAMES[metric]}','同一指标的三个子图共享颜色范围；每格均评价完整728条验证样本',ncols=3,size=(14,6))
            lo=min(z.min() for z in values);hi=max(z.max() for z in values)
            for ax,mod,z in zip(axes,MOD_NAMES,values):
                im=self.heat(ax,z,LOC_NAMES,['10%','30%','50%'],vmin=lo,vmax=hi,cmap='YlOrRd' if metric=='mae' else 'Blues');ax.set(title=MOD_NAMES[mod],xlabel='遮挡位置',ylabel='目标遮挡比例')
                fig.colorbar(im,ax=ax,shrink=.8)
            self.save(fig,f'最终集成：缺失比例×位置的{METRIC_NAMES[metric]}','连续缺失','位置索引定义前/中/后；音视频的小幅波动不等于缺失带来稳定收益。',rows,'selected_validation_metrics.json',f'集成缺失热图_{metric}')

        for metric in ['macro_f1','mae']:
            fig,axes=self.fig(f'相对完整输入的变化：{METRIC_NAMES[metric]}','每格数值 = 该缺失条件指标 − 完整输入指标',ncols=3,size=(14,6));rows=[]
            values=[np.array([[self.selected[f'{mod}_{r}_{loc}'][metric]-self.selected['clean'][metric] for loc in LOCS] for r in [10,30,50]]) for mod in MOD_NAMES]
            bound=max(np.max(np.abs(z)) for z in values)
            for ax,mod,z in zip(axes,MOD_NAMES,values):
                im=self.heat(ax,z,LOC_NAMES,['10%','30%','50%'],cmap='RdBu_r' if metric=='mae' else 'RdBu',vmin=-bound,vmax=bound,fmt='+.3f');fig.colorbar(im,ax=ax,shrink=.8);ax.set(title=MOD_NAMES[mod],xlabel='遮挡位置',ylabel='目标遮挡比例')
                rows.extend(dict(modality=mod,ratio=r,location=loc,metric=metric,delta=float(z[i,j])) for i,r in enumerate([10,30,50]) for j,loc in enumerate(LOCS))
            self.save(fig,f'相对完整输入的变化：{METRIC_NAMES[metric]}','连续缺失','文本缺失的影响通常较明显。差值是配对开发验证结果，不单独证明模态因果重要性。',rows,'selected_validation_metrics.json',f'缺失相对变化_{metric}')

        for metric in ['macro_f1','mae']:
            fig,axes=self.fig(f'五种主要模型的27条件矩阵：{METRIC_NAMES[metric]}','每格为3次独立训练的种子均值；不是三种子预测集成的指标',nrows=3,size=(15,10));rows=[]
            vals=df.groupby(['model','condition'])[metric].mean();lo=vals.min();hi=vals.max()
            for ax,mod in zip(axes,MOD_NAMES):
                conditions=[f'{mod}_{r}_{loc}' for r in [10,30,50] for loc in LOCS]
                z=np.array([[vals.loc[m,c] for c in conditions] for m in MAIN])
                im=self.heat(ax,z,[f'{r}%{l}' for r in [10,30,50] for l in LOC_NAMES],[NAMES[m] for m in MAIN],cmap='YlOrRd' if metric=='mae' else 'Blues',vmin=lo,vmax=hi);ax.set_title(MOD_NAMES[mod],loc='left');fig.colorbar(im,ax=ax,shrink=.85)
                rows.extend(dict(model=m,condition=c,metric=metric,seed_mean=float(vals.loc[m,c])) for m in MAIN for c in conditions)
            self.save(fig,f'五种主要模型的27条件矩阵：{METRIC_NAMES[metric]}','连续缺失','完整呈现条件矩阵，避免只挑选状态递推较好的条件；不要在看过矩阵后改变原定选模规则。',rows,'all_condition_metrics.csv',f'全模型条件矩阵_{metric}')

        conditions=['clean','audio_30_middle','audio_vision_30_sync','audio_vision_30_offset','text_audio_30_sync','audio_30_two_blocks']
        labels=['完整','语音中部30%','音视同步30%','音视错位30%','文本语音同步30%','语音双块30%']
        fig,axes=self.fig('同步、错位与双块缺失的扩展评价','最终MLP集成；不同模态组合改变的信息量不同',ncols=2,size=(14,7));rows=[]
        for ax,metric in zip(axes,['macro_f1','mae']):
            v=[self.selected[c][metric] for c in conditions];bars=ax.barh(range(len(v)),v,color=['#849AA8']+['#176B87']*5)
            ax.set_yticks(range(len(v)),labels);ax.invert_yaxis();ax.bar_label(bars,fmt='%.4f',padding=4,fontsize=9);ax.set_xlim(0,max(v)*1.17);ax.set_xlabel(METRIC_NAMES[metric]);ax.grid(axis='x')
            rows.extend(dict(condition=c,metric=metric,value=float(x)) for c,x in zip(conditions,v))
        self.save(fig,'同步、错位与双块缺失的扩展评价','连续缺失','音视同步/错位用于比较位置关系；语音单块/双块使用相同目标比例，实际移除的可用观测仍可能不同。',rows,'selected_validation_metrics.json','复杂缺失条件')

        fig,axes=self.fig('单样本预测受30%中部缺失的影响','与该样本完整输入预测比较；每种条件均使用同一728条样本',ncols=2,size=(13,6));rows=[]
        distributions=[];change=[]
        for mod in MOD_NAMES:
            c=f'{mod}_30_middle';r=self.pred[c+'_intensity'];p=self.pred[c+'_probabilities'];delta=np.abs(r-self.target)-self.ae;distributions.append(delta);change.append(np.mean(p.argmax(1)!=self.guess))
            rows.extend(dict(sample_id=str(s),modality=mod,absolute_error_change=float(d),intensity_change=float(rv-b),class_changed=bool(a!=g)) for s,d,rv,b,a,g in zip(self.ids,delta,r,self.reg,p.argmax(1),self.guess))
        axes[0].boxplot(distributions,tick_labels=list(MOD_NAMES.values()),patch_artist=True,boxprops={'facecolor':'#D5E7ED'},flierprops={'markersize':3,'alpha':.3});axes[0].axhline(0,color='#A85678',ls='--');axes[0].set_ylabel('绝对误差变化：缺失 − 完整');axes[0].grid(axis='y')
        bars=axes[1].bar(list(MOD_NAMES.values()),change,color=['#176B87','#E49B39','#A85678']);axes[1].bar_label(bars,labels=[f'{v:.1%}' for v in change],padding=4);axes[1].set(ylabel='类别预测改变比例',ylim=(0,max(change)*1.3));axes[1].grid(axis='y')
        self.save(fig,'单样本预测受30%中部缺失的影响','连续缺失','预测改变不必然意味着错误，误差差值也可为负；这里展示扰动敏感性，不作特征归因。',rows,'selected_validation_predictions.npz','样本缺失敏感性')

        budget=pd.read_csv(RES/'validation_missing_budgets.csv');budget=budget[budget.condition.isin(grid)].copy()
        # Keep only the modality targeted by the single-modality condition.
        budget=budget[budget.apply(lambda r: str(r.condition).startswith(str(r.modality)+'_'),axis=1)]
        fig,axes=self.fig('目标遮挡比例与实际移除观测比例','分母为原本可用的内容观测；本来不可用的位置不会重复计为新缺失',ncols=3,size=(15,6))
        for ax,mod in zip(axes,MOD_NAMES):
            d=budget[budget.modality==mod];groups=[d.loc[np.isclose(d.requested_fraction,r),'actual_removed_fraction_of_available'].dropna().values for r in [.1,.3,.5]]
            ax.boxplot(groups,tick_labels=['10%','30%','50%'],patch_artist=True,boxprops={'facecolor':'#D5E7ED'},showfliers=False)
            ax.plot([1,2,3],[.1,.3,.5],'D--',color='#A85678',label='目标比例');ax.set(title=MOD_NAMES[mod],xlabel='目标遮挡位置比例',ylabel='移除比例 / 原可用观测',ylim=(-.03,1.03));ax.legend(fontsize=9);ax.grid(axis='y')
        self.save(fig,'目标遮挡比例与实际移除观测比例','连续缺失','箱体包含样本与三个位置条件的描述性分布，不是独立重复误差条；0可用观测时分母规则见原始预算表。',budget,'validation_missing_budgets.csv','目标与实际遮挡')

    def diagnostic_figures(self):
        evidence=json.loads((RES/'paired_group_bootstrap.json').read_text());rows=[]
        fig,ax=self.fig('状态递推与GRU：配对分组Bootstrap差值','均为3种子预测集成；按239个视频组重采样，保留组内相关性')
        for i,(c,d) in enumerate(evidence.items()):
            mean=d['state_minus_gru_mae'];lo,hi=d['percentile_95'];ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=6,color='#A85678',ms=8)
            rows.append(dict(condition=c,mae_difference=mean,lower95=lo,upper95=hi,n_video_groups=d['n_video_groups']))
            ax.text(hi+.0005,i,f'{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]',va='center',fontsize=10)
        ax.axvline(0,color='#7B8D99',ls='--');ax.set_yticks(range(2),['完整输入','文本30%中部缺失']);ax.set(xlabel='MAE差值：状态递推 − GRU（正值代表状态递推误差更大）',ylim=(-.65,1.65),xlim=(-.006,.049));ax.grid(axis='x')
        self.save(fig,'状态递推与GRU：配对分组Bootstrap差值','稳健性诊断','当前结果不支持状态递推优于GRU；区间只刻画开发验证集的配对波动，不是独立泛化证明。',rows,'paired_group_bootstrap.json','配对Bootstrap')

        fig,axes=self.fig('最终MLP：单次训练与三种子集成','集成指标由先平均预测再评分得到，与各次训练指标平均值不同',ncols=2,size=(13,6));rows=[]
        for ax,metric in zip(axes,['macro_f1','mae']):
            vals=[self.run_metrics[f'mlp_seed{s}']['conditions']['clean'][metric] for s in [17,29,43]]+[self.selected['clean'][metric]]
            bars=ax.bar(['seed17','seed29','seed43','三种子集成'],vals,color=['#AEC8D2']*3+['#176B87']);ax.bar_label(bars,fmt='%.4f',padding=4);ax.set(ylim=(0,max(vals)*1.25),ylabel=METRIC_NAMES[metric]);ax.grid(axis='y')
            rows.extend(dict(model=m,metric=metric,value=v) for m,v in zip(['mlp_seed17','mlp_seed29','mlp_seed43','ensemble'],vals))
        self.save(fig,'最终MLP：单次训练与三种子集成','稳健性诊断','集成无需再次训练；本图不把一次集成结果当作额外独立随机种子。',rows,'runs/mlp*/metrics.json; selected_validation_metrics.json','单模型与集成')

        lengths=self.cache['valid']['sequence'].sum(1);bins=[(0,10),(10,20),(20,30),(30,40),(40,50)];rows=[]
        for lo,hi in bins:
            ix=(lengths>lo)&(lengths<=hi)
            if ix.any():rows.append(dict(length_bin=f'{lo+1}—{hi}',n=int(ix.sum()),accuracy=float(self.correct[ix].mean()),mae=float(self.ae[ix].mean())))
        df=pd.DataFrame(rows);fig,axes=self.fig('按内容序列长度分层的误差','长度来自有效内容位置计数，不含CLS、SEP和填充',ncols=2)
        for ax,col,name in zip(axes,['accuracy','mae'],['Accuracy ↑','MAE ↓']):
            bars=ax.bar(range(len(df)),df[col],color='#176B87');ax.bar_label(bars,fmt='%.3f',padding=3);ax.set_xticks(range(len(df)),[f'{r.length_bin}\nn={r.n}' for r in df.itertuples()]);ax.set(ylabel=name,xlabel='有效内容位置数',ylim=(0,max(df[col])*1.25));ax.grid(axis='y')
        self.save(fig,'按内容序列长度分层的误差','稳健性诊断','长度分层只描述观察到的差异，长度与标签、内容等因素可能相关；不能据此断言长度导致误差。',df,'cache/valid.npz; selected_validation_predictions.npz','长度分层表现')

    def special_figures(self):
        df=self.special.copy();pcols=['probability_negative','probability_neutral','probability_positive'];p=df[pcols].to_numpy()
        note='附件3：30条无标签样本｜2026-09-24已修正UNK掩码｜仅展示预测'
        fig,axes=self.fig('附件3：预测类别与强度分布','最终MLP三种子集成 · 30条专项样本 · 无真实标签',ncols=2)
        counts=np.bincount(df.polarity,minlength=3);bars=axes[0].bar(CLASS_NAMES,counts,color=CLASS_COLORS);axes[0].bar_label(bars,padding=3);axes[0].set(ylabel='预测样本数',ylim=(0,max(counts)*1.25));axes[0].grid(axis='y')
        axes[1].hist(df.intensity,bins=np.linspace(-3,3,19),color='#176B87');axes[1].axvline(0,color='#A85678',ls='--');axes[1].set(xlabel='预测情感强度',ylabel='样本数');axes[1].grid(axis='y')
        self.save(fig,'附件3：预测类别与强度分布','附件3预测','类别比例与强度仅是模型输出的分布，不代表真实类别组成或性能。',df,'attachment3_predictions.csv','附件3预测分布',note=note)

        ordered=df.sort_values('intensity');fig,ax=self.fig('附件3：30条样本的情感强度预测','按预测强度从负到正排序；颜色表示分类头输出',size=(12,11.5))
        for i,r in enumerate(ordered.itertuples()):ax.plot([0,r.intensity],[i,i],color=CLASS_COLORS[r.polarity],lw=2);ax.scatter(r.intensity,i,color=CLASS_COLORS[r.polarity],s=30);ax.text(r.intensity+(.04 if r.intensity>=0 else -.04),i,f'{r.intensity:+.3f}',ha='left' if r.intensity>=0 else 'right',va='center',fontsize=8)
        ax.set_yticks(range(len(df)),ordered.sample_id.astype(str),fontsize=8);ax.invert_yaxis();ax.axvline(0,color='#8C9EA9',ls='--');ax.set(xlim=(-3.1,3.1),xlabel='预测情感强度');ax.grid(axis='x')
        ax.legend(handles=[Patch(color=c,label=n) for c,n in zip(CLASS_COLORS,CLASS_NAMES)],ncol=3,loc='lower right')
        self.save(fig,'附件3：30条样本的情感强度预测','附件3预测','全部样本均展示；分类头与回归头可能存在边界不一致，不擅自修改提交预测。',ordered,'attachment3_predictions.csv','附件3逐样本强度',note=note)

        fig,ax=self.fig('附件3：每条样本的三类预测概率','按文件中的原始顺序展示；每行三个概率之和约为1',size=(10,12))
        im=self.heat(ax,p,CLASS_NAMES,df.sample_id.astype(str).tolist(),vmin=0,vmax=1,fmt='.3f');ax.tick_params(axis='y',labelsize=8);fig.colorbar(im,ax=ax,label='预测概率',shrink=.75)
        self.save(fig,'附件3：每条样本的三类预测概率','附件3预测','概率未经校准。不同类别概率接近表示模型区分不明确，不可直接推导正确率。',df,'attachment3_predictions.csv','附件3概率热图',note=note)

        entropy=-(p*np.log(np.clip(p,1e-12,1))).sum(1)/np.log(3);df['normalized_entropy']=entropy;df['max_probability']=p.max(1)
        fig,axes=self.fig('附件3：预测分歧与置信度诊断','归一化熵接近1表示三类概率接近；接近0表示概率集中',ncols=2)
        for c in range(3):
            ix=df.polarity==c;axes[0].scatter(df.loc[ix,'intensity'],entropy[ix],s=50,color=CLASS_COLORS[c],label=CLASS_NAMES[c]);axes[1].scatter(p.max(1)[ix],entropy[ix],s=50,color=CLASS_COLORS[c])
        axes[0].set(xlabel='预测情感强度',ylabel='归一化预测熵',ylim=(0,1.05));axes[0].legend();axes[1].set(xlabel='最大类别概率',ylabel='归一化预测熵',ylim=(0,1.05),xlim=(.3,1));axes[0].grid();axes[1].grid()
        self.save(fig,'附件3：预测分歧与置信度诊断','附件3预测','熵和最大概率均来自同一概率向量，仅是输出集中程度，不是经过验证的置信区间。',df,'attachment3_predictions.csv','附件3置信度',note=note)

    def sample_cards(self):
        # Deterministic descriptive selection, never cherry-pick these as aggregate evidence.
        choices=[]
        for c in range(3):
            ix=np.flatnonzero((self.y==c)&self.correct)
            if len(ix):
                median=np.median(self.ae[ix]);i=ix[np.argmin(abs(self.ae[ix]-median))];choices.append((int(i),f'{CLASS_NAMES[c]}正确分类组：回归误差最接近组内中位数'))
        for i in np.argsort(-self.ae)[:3]:choices.append((int(i),'全体验证样本绝对误差排名靠前'))
        wrong=np.flatnonzero(~self.correct)
        for i in wrong[np.argsort(-self.prob[wrong].max(1))[:3]]:choices.append((int(i),'错误分类组中最大类别概率排名靠前'))
        conflicts=np.flatnonzero(((self.guess==0)&(self.reg>0))|((self.guess==2)&(self.reg<0)))
        if len(conflicts):choices.append((int(conflicts[np.argmax(self.ae[conflicts])]),'分类与回归方向冲突组中回归误差最大'))
        seen=set();text_by_id={str(s):str(t) for s,t in zip(self.cache['valid']['ids'],self.cache['valid']['raw_text'])}
        for i,criterion in choices:
            if i in seen:continue
            seen.add(i);sid=str(self.ids[i]);fig,axes=self.fig(f'验证样本诊断 · {sid}',criterion,nrows=2,ncols=2,size=(13,8.7))
            axes[0,0].axis('off');txt=text_by_id.get(sid,'')
            axes[0,0].text(0,1,'原始文本\n\n'+textwrap.fill(txt,66),transform=axes[0,0].transAxes,va='top',fontsize=9,linespacing=1.5)
            bars=axes[0,1].bar(CLASS_NAMES,self.prob[i],color=CLASS_COLORS);axes[0,1].bar_label(bars,fmt='%.3f',padding=3);axes[0,1].set(ylim=(0,1.1),ylabel='预测概率',title=f'真实：{CLASS_NAMES[self.y[i]]}  /  预测：{CLASS_NAMES[self.guess[i]]}');axes[0,1].grid(axis='y')
            axes[1,0].barh(['真实强度','预测强度'],[self.target[i],self.reg[i]],color=['#93AFBB','#176B87']);axes[1,0].axvline(0,color='#9DAAB4');axes[1,0].set(xlim=(-3.3,3.3),xlabel='情感强度',title=f'绝对误差 = {self.ae[i]:.3f}');axes[1,0].grid(axis='x')
            conditions=['clean']+[f'{m}_30_middle' for m in MOD_NAMES];values=[self.pred[c+'_intensity'][i] for c in conditions]
            axes[1,1].plot(range(4),values,'o-',color='#176B87');axes[1,1].axhline(self.target[i],ls='--',color='#A85678',label='真实强度');axes[1,1].set_xticks(range(4),['完整','文本缺失','语音缺失','视觉缺失']);axes[1,1].set(ylabel='预测强度',title='30%中部缺失的预测敏感性');axes[1,1].grid();axes[1,1].legend()
            rows=[dict(sample_id=sid,selection_criterion=criterion,text=txt,true_class=int(self.y[i]),true_intensity=float(self.target[i]),condition=c,predicted_intensity=float(self.pred[c+'_intensity'][i]),**{f'probability_{k}':float(self.pred[c+'_probabilities'][i,k]) for k in range(3)}) for c in conditions]
            self.save(fig,f'验证样本诊断 · {sid}','验证样本卡片','案例按明确规则筛选，仅用于解释误差形态；敏感性曲线不是词级或模态因果归因。',rows,'cache/valid.npz; selected_validation_predictions.npz',f'验证样本_{len(seen):02d}',card=True)

        for i,r in enumerate(self.special.itertuples()):
            p=np.array([r.probability_negative,r.probability_neutral,r.probability_positive]);ent=-(p*np.log(np.clip(p,1e-12,1))).sum()/np.log(3)
            fig,axes=self.fig(f'附件3预测卡片 · {r.sample_id}','无真实标签 · 保留模型原始输出 · 最终MLP三种子集成',ncols=2,size=(10.8,5.5))
            bars=axes[0].bar(CLASS_NAMES,p,color=CLASS_COLORS);axes[0].bar_label(bars,fmt='%.3f',padding=4);axes[0].set(ylim=(0,1.1),ylabel='预测概率');axes[0].grid(axis='y')
            axes[1].axis('off');axes[1].text(.08,.9,f'预测类别：{CLASS_NAMES[r.polarity]}\n\n预测强度：{r.intensity:+.4f}\n\n最大类别概率：{p.max():.3f}\n\n归一化预测熵：{ent:.3f}',transform=axes[1].transAxes,fontsize=15,va='top',linespacing=1.4)
            self.save(fig,f'附件3预测卡片 · {r.sample_id}','附件3样本卡片','只展示预测结果及输出集中程度，不能据此判断该样本是否预测正确。',[r._asdict()], 'attachment3_predictions.csv',f'附件3样本_{i+1:02d}',card=True,note='附件3无标签｜2026-09-24已修正UNK掩码｜预测概率未经校准')

    def finish(self,before):
        self.pdf.close();self.card_pdf.close();entries=self.entries+self.card_entries
        (OUT/'manifest.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf-8')
        recommended=[e for e in self.entries if any(s in e['stem'] for s in ['结果总览','类别分布','选模条件模型对比','递推消融','混淆矩阵_比例','回归散点','强度分层误差','集成缺失热图_macro_f1','缺失相对变化_mae','配对Bootstrap','附件3概率热图'])]
        notes=['# 第二问可视化图集', '',f'包含 {len(self.entries)} 张分析图、{len(self.card_entries)} 张样本卡片。每图导出300 dpi PNG、可编辑文字SVG和绘图数据CSV；分析图与卡片分别汇入PDF。','',
            '## 使用范围','', '所有图基于已有真实运行结果生成，没有重新训练、重选模型或修改预测。附件2验证集参与模型选择，图中指标仅代表开发验证结果；未评价附件2 test。附件3没有标签，只能展示预测。', '',
            '2026-09-24更新：附件3图表和卡片已使用修正UNK缺失掩码后的预测；旧版本归档。验证集图表和冻结模型权重未变。','',
            '误差条如无另外说明均为3个随机种子的样本标准差，不是置信区间。单种子消融明确标注。Bootstrap图区间复用原有按视频组重采样的结果。模型种子指标均值与平均预测后的集成指标分开报告。','',
            '缺失比例描述序列位置，不代表秒数；填充、特殊标记与内容内不可用位置分开处理。音视频内容内零行按操作规则标为不可用，不据此断言真实缺失原因。预测概率未经校准。','',
            '## 建议优先用于论文的图','']
        notes += [f'- [{e["number"]} {e["title"]}]({e["png"]})：{e["caption"]}' for e in recommended]
        notes += ['', '## 全部图表索引','']
        for e in entries:
            notes += [f'### {e["number"]} {e["title"]}', '',f'类别：{e["category"]}。{e["caption"]}', '',f'[PNG]({e["png"]}) · [SVG]({e["svg"]}) · [数据CSV]({e["data"]})', '',f'来源：`{e["source"]}`。{e["note"]}', '']
        notes += ['## 重现方法','', '`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python visualize_results.py`', '', '请在包含原始 runs、results、cache 的 q2 项目中运行。图集压缩包附绘图脚本，不重复打包训练数据与模型。SVG文字需安装Noto Sans CJK字体，跨机器保真展示优先使用PDF或PNG。', '', '图集单独打包，没有塞入比赛提交包。']
        (OUT/'图表索引与使用说明.md').write_text('\n'.join(notes),encoding='utf-8')
        cats=list(dict.fromkeys(e['category'] for e in entries));cards=[]
        for e in entries:
            esc=html.escape
            cards.append(f'<article data-category="{esc(e["category"])}"><a href="{esc(e["png"])}" target="_blank"><img loading="lazy" src="thumbnails/{esc(e["stem"])}.jpg" alt="{esc(e["title"])}"></a><div><small>{esc(e["category"])}</small><h2>{e["number"]} {esc(e["title"])}</h2><p>{esc(e["caption"])}</p><nav><a href="{esc(e["png"])}" target="_blank">PNG</a><a href="{esc(e["svg"])}" target="_blank">SVG</a><a href="{esc(e["data"])}" download>数据 CSV</a></nav></div></article>')
        page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>第二问可视化图集</title><style>
        *{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#21394a;font:15px/1.6 system-ui,"Noto Sans CJK SC",sans-serif}header{background:#173e50;color:white;padding:36px max(5vw,20px)}h1{margin:0 0 12px}header p{max-width:1000px;color:#deebf0}header a{color:white;margin-right:24px}.filters{padding:18px 5vw;position:sticky;top:0;background:#f2f5f7ed;backdrop-filter:blur(8px);z-index:2;display:flex;gap:8px;flex-wrap:wrap}button{border:1px solid #b8cbd5;border-radius:20px;background:white;padding:7px 13px;color:#23485c;cursor:pointer}button.active{background:#176b87;color:white}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:22px;padding:8px 5vw 60px}article{background:white;border-radius:10px;overflow:hidden;border:1px solid #dce6eb}article img{width:100%;display:block;aspect-ratio:1.65;object-fit:contain;background:white}article div{padding:16px}h2{font-size:17px;margin:4px 0}p{margin:8px 0}small{color:#617e8d}nav a{color:#176b87;margin-right:18px}article[hidden]{display:none}#count{padding:0 5vw;color:#617e8d}</style><header><h1>第二问 · 实测结果图集</h1><p>__COUNT__ 张分析图与 __CARDS__ 张样本卡片。点击缩略图查看300 dpi原图，支持下载SVG和数据表。验证集参与选模；附件3只展示预测。</p><a href="第二问可视化图集.pdf">分析图PDF</a><a href="样本卡片合集.pdf">样本卡片PDF</a><a href="图表索引与使用说明.md">图注与使用说明</a></header><section class="filters">__FILTERS__</section><p id="count"></p><main>__CONTENT__</main><script>
        const buttons=[...document.querySelectorAll('button')], cards=[...document.querySelectorAll('article')];function choose(category){buttons.forEach(b=>b.classList.toggle('active',b.dataset.category===category));cards.forEach(c=>c.hidden=category!=='全部'&&c.dataset.category!==category);document.getElementById('count').textContent='显示 '+cards.filter(c=>!c.hidden).length+' 张图'}buttons.forEach(b=>b.addEventListener('click',()=>choose(b.dataset.category)));choose('全部');</script></html>'''
        filters=''.join(f'<button data-category="{html.escape(c)}">{html.escape(c)}</button>' for c in ['全部']+cats)
        page=page.replace('__COUNT__',str(len(self.entries))).replace('__CARDS__',str(len(self.card_entries))).replace('__FILTERS__',filters).replace('__CONTENT__','\n'.join(cards))
        (OUT/'index.html').write_text(page,encoding='utf-8')
        font=ImageFont.truetype(FONT,22);smallfont=ImageFont.truetype(FONT,17)
        for group,items in [('分析图',self.entries),('样本卡片',self.card_entries),('论文优先',recommended)]:
            for start in range(0,len(items),12):
                subset=items[start:start+12];cols=3;rows=math.ceil(len(subset)/cols);canvas=Image.new('RGB',(1800,rows*420+90),'#edf3f6');draw=ImageDraw.Draw(canvas);draw.text((28,20),f'第二问 · {group} · {start+1}—{start+len(subset)}',fill='#21394a',font=font)
                for j,e in enumerate(subset):
                    x=(j%cols)*600;y=90+(j//cols)*420
                    with Image.open(OUT/e['png']) as im:
                        thumb=ImageOps.contain(im.convert('RGB'),(584,350));canvas.paste(thumb,(x+(600-thumb.width)//2,y))
                    title=f'{e["number"]} {e["title"]}';lines=textwrap.wrap(title,29)
                    draw.text((x+14,y+355),'\n'.join(lines[:2]),font=smallfont,fill='#21394a',spacing=4)
                canvas.save(OUT/'contact_sheets'/f'{group}_{start//12+1:02d}.jpg',quality=94)
        (OUT/'visualize_results.py').write_text(Path(__file__).read_text(),encoding='utf-8')
        protected_unchanged=all(sha(p)==digest for p,digest in before.items())
        if not protected_unchanged:raise RuntimeError('Protected input changed during plotting')
        for e in entries:
            for key in ['png','svg','data']:
                if not (OUT/e[key]).is_file():raise RuntimeError(f'Missing artifact {e[key]}')
            with Image.open(OUT/e['png']) as im:
                im.verify()
        check={'analysis_figures':len(self.entries),'sample_cards':len(self.card_entries),'png_files':len(entries),'svg_files':len(entries),'csv_files':len(entries),'all_pngs_verified':True,'protected_inputs_unchanged':protected_unchanged,'protected_sha256':{str(p.relative_to(ROOT)):d for p,d in before.items()},'versions':{'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__}}
        (OUT/'verification.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
        dest=RES/'第二问可视化图集_全套.zip'
        with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(OUT.rglob('*')):
                if p.is_file():z.write(p,arcname=str(Path('第二问可视化图集')/p.relative_to(OUT)))
        print(json.dumps({'figures':len(self.entries),'cards':len(self.card_entries),'zip':str(dest),'zip_MB':round(dest.stat().st_size/1024**2,2)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    protected=[ROOT/'selection.json',RES/'attachment3_predictions.csv',RES/'selected_validation_predictions.npz',RES/'model_comparison.csv']+sorted((ROOT/'runs').glob('*/best.pt'))
    before={p:sha(p) for p in protected}
    atlas=Atlas()
    try:
        for method in ['overview','data_figures','comparison_figures','training_figures','classification_figures','regression_figures','missing_figures','diagnostic_figures','special_figures','sample_cards']:
            getattr(atlas,method)()
    except Exception:
        atlas.pdf.close();atlas.card_pdf.close()
        raise
    atlas.finish(before)
