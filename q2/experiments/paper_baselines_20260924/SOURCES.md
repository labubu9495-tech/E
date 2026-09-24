# 论文来源与赛题适配记录

本实验为基于官方开源结构的赛题适配实现，不宣称复现原论文数据集上的成绩。原始仓库保存在vendor，仅作来源核对；模型由paper_models.py实现。两个项目均为MIT许可，版权与许可全文保存在LICENSES，发布派生代码时必须保留。

|方法|论文和官方代码|固定版本|
|---|---|---|
|MulT|Tsai et al., Multimodal Transformer for Unaligned Multimodal Language Sequences, ACL 2019; https://github.com/yaohungt/Multimodal-Transformer |a670936824ee722c8494fd98d204977a1d663c7a|
|EMT-DLFR|Sun et al., Efficient Multimodal Transformer with Dual-Level Feature Restoration for Robust Multimodal Sentiment Analysis, IEEE TAC 2023; https://github.com/sunlicai/EMT-DLFR |e4216215565292360e9c3fc43c8cd6e127df19c2|

## 共用输入与评价

- 附件2官方train3395/valid728，aligned版本；不读取test进行模型选择，不下载仓库推荐的外部情感数据或情感模型权重。
- 沿用冻结BERT-Mini的256维缓存、音频74维、视觉35维，以及先遮token再编码的8组训练文本视图。训练掩码直接复用当前train.training_batch；验证使用同一32条件和同一已缓存的文本遮挡。
- 内容位置由独立sequence mask确定。输入先按observed清除不可用数值；位置不因中间缺失压缩。全空样本返回同一训练先验。
- 三分类CE与强度Huber损失，强度限制在[-3,3]；同一检查点报告全部指标。选模沿用完整输入加T/A/V30%中部缺失的平均Macro-F1、MAE打破平局。
- 3个种子17/29/43，上限40轮、早停6轮、批量64。历史模型与新模型的容量和计算预算不同，不声称同参数量或同训练FLOPs。
- 第一轮不搜索超参数。MulT学习率1e-3/weight_decay1e-4，EMT两变体学习率1e-4/weight_decay1e-3。此设置在正式验证结果出现前固定。

## MulT保留与改动

保留官方六个方向的两两跨模态Transformer、每个目标模态两路结果拼接、各自的自注意力记忆网络、三路最终表示拼接与残差预测头。维度30、5头、跨模态和记忆均5层，与官方默认深度一致。

改动：用现代PyTorch SDPA实现注意力；显式位置正弦编码，不根据投影特征某一维是否为0猜padding；不可用source位置不作为cross-attention key，真实但缺失的target位置保留为query。末位置读取改为最后真实内容位置；不是固定索引49。增加三分类头及回归范围限制，使用同一训练增强；采用AdamW与公共梯度裁剪。初始化采用当前PyTorch默认，各dropout位置与现代实现有工程差异。

标准正弦编码使用交替sin/cos布局，与旧MulT的sin/cos分块布局不同。这里只比较适配架构在赛题数据上的效果，不能将差异归为论文原模型的严格优劣。

## EMT-DLFR保留与改动

保留音频/视觉单层LSTM（hidden16/32）、全局三模态上下文、双向global-local MPU、自注意力+GEGLU、对三路global更新的注意力池化。沿用官方MOSEI设置dim128、4头、2层，方向/模态/层共享MPU参数，层间共享pool。

保留低层SmoothL1重建，以及全局/文本/音频/视觉四路SimSiam的projector、predictor与对称stop-gradient余弦吸引；完整和缺失训练视图分别受预测损失监督。只有训练会构造完整参考分支。

改动：文本使用共用256维冻结特征，句级文本表示由可用内容池化得到（不重新引入未缓存CLS表示）；AV LSTM仅把非内容特殊位置/padding移至尾部，内部空位保留，并将局部状态还原至原始位置。fusion中真实内容位置允许作为推断表示参与交互，原始不可用输入始终已被清除；重建局部表示不当作真实观测证据。注意力显式处理空context。重建只监督原本可观测且本次被人工遮挡的位置；自然视觉全空等没有目标的情况不伪造监督。缺失原因不从数值臆断。

每次训练总损失：缺失视图CE+Huber + 完整视图CE+Huber + 三模态平均特征SmoothL1之和 + 四路对称负余弦之和。吸引项可以为负，因此总loss为负不等于数值错误，报告同时记录各损失分量。自然整路无观测时，跳过该路高层吸引监督。

消融emt_no_restore保留完全相同的预测结构、初始化及完整/缺失两支监督，只去掉低层恢复与高层吸引损失。避免把额外完整视图监督的收益误归于恢复机制。

## 安全边界与复现

恢复器和完整视图仅用于训练。验证/专项调用的forward不接受完整参考或标签，跳过恢复与SimSiam头。第三问可以复用预测接口，但本实验未实现附件4适配、模态归因或真实时间映射。

在第二问项目目录运行：

```bash
.venv/bin/python experiments/paper_baselines_20260924/test_paper_models.py
.venv/bin/python experiments/paper_baselines_20260924/run_experiments.py --model mult --smoke --device cuda:0
.venv/bin/python experiments/paper_baselines_20260924/run_experiments.py --model emt_dlfr --smoke --device cuda:2
.venv/bin/python experiments/paper_baselines_20260924/run_experiments.py --model mult --device cuda:0
.venv/bin/python experiments/paper_baselines_20260924/run_experiments.py --model emt_dlfr --device cuda:2
.venv/bin/python experiments/paper_baselines_20260924/run_experiments.py --model emt_no_restore --device cuda:2
```

使用实际可用GPU编号。日志、配置、哈希、损失历史、最佳检查点、32条件预测各自保存在本目录；不覆盖既有MLP选择和专项输出。是否更新最终选择取决于正式验证结果。
