# 第二问：局部缺失下的多模态情感预测

本项目独立完成第二问的数据审查、训练、验证、缺失实验及附件3预测；不依赖第一问自生成特征。训练与选模仅使用附件2 train/valid。附件2 test保留，附件3不用于拟合、选模或缺失分布设计。

## 输入和标签

- 统一使用aligned_50版本：text_bert `(N,3,50)`、audio `(N,50,74)`、vision `(N,50,35)`。
- 分类标签0/1/2分别为Negative/Neutral/Positive；强度在[-3,3]，零仅属中性。
- 文本词表是uncased BERT，冻结编码器为`google/bert_uncased_L-4_H-256_A-4`；精确revision与权重格式见`assets/bert-mini/source.json`。
- 不使用附件2预计算text训练后再切换文本表示。text_bert统一经本地冻结编码器，生成256维文本表示。
- SEP确定真实内容范围，CLS、SEP与padding不参加时序池化。文本内部零区间保留位置；音视频在内容范围内的有限非零行作为操作性可用观测。全零行的真实成因未知。
- 若专项样本缺少SEP，以最末可见支持估计终点，标记`observed_support_approximate`；不能恢复完全无观测的真实尾长。
- 缺失长度按内容序列位置数及比例定义，不解释为秒数；取整后的实际位置数可能不同于目标比例。

## 环境

Python 3.13、PyTorch 2.10已实测。其他依赖包括numpy、scikit-learn、transformers 4.x、safetensors、matplotlib、scipy。运行版本以`results/runtime_versions.json`为准。

开发目录使用独立`.venv`并继承服务器已有PyTorch；提交包不包含虚拟环境。普通环境可安装`requirements.txt`，GPU用户按设备安装对应PyTorch轮子。所有命令在本项目目录执行。

## 从原始数据复现实验

本地`assets/bert-mini/`包含编码器配置、词表和FP16存储权重。实际加载与缓存计算使用FP32；初次实验开始前即固定这些权重。

```bash
python prepare.py --data-root /path/to/E题 --device cuda
python -m unittest test_invariants -v
python run_queue.py --queue baseline --device cpu
python run_queue.py --queue state --device cpu
python run_queue.py --queue clean_repeats --device cpu
python summarize.py
```

CPU环境可将prepare的device改为cpu。两条实验队列可独立运行；重复调用会跳过已有`metrics.json`的完整运行。若修改配置重新实验，应使用新实验目录或先归档旧runs，不能将旧缓存/结果与新配置混合。

单次训练示例：

```bash
python train.py --config gru_aug --seed 17 --device cpu
python train.py --config state_aug --seed 17 --device cpu
```

`--max-epochs`仅供单独排错实验，正式队列遵循protocol.json。首版实验固定40轮上限、早停6轮、batch64、32维表示、CE+Huber、AdamW学习率1e-3。增强为40%原输入、40%单模态、20%双模态，长度10%—50%。文本缺失在BERT之前施加，训练缓存8种缺失视图；音视频在线采样。缺失块的原位置保留。

主要候选mlp、gru_clean、gru_aug、state_aug、smooth_aug各运行17/29/43三个种子；其余对照单种子。无增强GRU在首轮验证后补齐重复并纳入候选，未使用专项测试信息。各次模型选择使用clean与text/audio/vision的30%-middle四个条件平均Macro-F1，MAE打破平局。结构也按同一规则的种子均值选择，最后对三种子平均类别概率和强度。验证集用于模型选择，因此全部验证表现属于开发评价，不是独立测试结果。

本实现未运行LMF/MulT，不能据此声称超过文献方法。

## 冻结后的附件3推理

先确认`selection.json`已经生成。该文件固定结构、检查点与权重哈希，再进行专项推理：

```bash
python infer.py --input /path/to/附件3-模态缺失特征样本/对齐版本 --output results/attachment3_predictions.csv --device cpu
```

也可使用`--device cuda`。代码只加载`selection.json`中的模型和训练集标准化参数，不重新训练。

CSV字段：

|字段|含义|
|---|---|
|sample_id|原文件id；缺少id时使用文件名主干，多样本文件附行号|
|polarity|Negative、Neutral或Positive|
|intensity|连续强度[-3,3]|
|probability_negative / probability_neutral / probability_positive|三分类概率，用于复核|

题面未提供固定列名模板，以上是本项目明确约定的格式；若后续发布模板，做字段映射而不改预测。`*.provenance.json`记录源文件、样本行和端点规则。不能给无标签附件3报告Accuracy/F1/MAE/Pearson。

## 结果与交付

- `results/audit.json`：字段、划分、标签、词表核验与操作性掩码统计。
- `protocol.json`：编码缓存和正式训练前固定的实验设置。
- `runs/*/config.json, history.json, best.pt, metrics.json`：配置、逐轮日志、选定权重与全部评价。
- `results/model_comparison.csv`：逐模型种子均值和标准差。
- `results/all_condition_metrics.csv`：全部条件、各次运行四项指标。
- `results/selected_validation_metrics.json`：最终集成的验证表现。
- `results/paired_group_bootstrap.json`：按原视频分组的配对回归误差差值区间。
- `results/validation_predictions_and_errors.csv`：全验证集预测与错误待分析清单。
- `results/figures/`：论文用静态图。
- `results/第二问实测报告.md`：数据、模型、实测结果与限制。
- `submission_q2.zip`：第二问独立复现包，包含冻结文本编码器、最终预测器及CSV。

该包只覆盖第二问。整题提交还需第一问100条特征与第三问解释材料，总体≤50MB；不要把第二问包当作整题已交齐。

## 实现注意事项

状态递推不是严格卡尔曼滤波；隐状态不是逐帧真实情绪。部分缺失使用剩余模态；真实位置全缺失只传播历史；padding不推进状态、不池化；整样本全缺失使用训练先验。

所有方法采用共同文本表示、评价遮挡和数据划分。去掉历史的消融完全取消历史路径；去掉gap只取消显式连续缺失长度提示。权重、掩码、空输入、padding以及BERT遮挡前后的信息隔离有专门测试。

预测错误原因需要实际阅读样本或检查特征；不要把模型指标、归因名称或未训练的候选方案写成实测结论。
