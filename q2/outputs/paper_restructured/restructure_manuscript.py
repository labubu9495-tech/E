"""Reorganize the verified manuscript around completed modeling work."""
from pathlib import Path
import importlib.util, inspect, json, re, shutil
OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'paper_chapters_1_5'
spec=importlib.util.spec_from_file_location('base_manuscript',SOURCE/'build_manuscript.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
old=json.loads((SOURCE/'manuscript_blocks.json').read_text())
blocks=[]

def append(start,end):blocks.extend([list(b) for b in old[start:end]])
def h(level,text):blocks.append(['h',level,text])
def p(text):blocks.append(['p',text])
append(0,69)
for b in blocks:
 if b[0]=='h' and b[2].startswith('1.3.4 '):b[1]=2;b[2]='1.4 研究现状与方法选择依据'
 elif b[0]=='h' and b[2].startswith('1.4 '):b[2]='1.5 总体建模思路与技术路线'
# Chapter 4 groups implemented models and objectives, leaving numerical
# comparisons in Chapter 5. Unexecuted Q1 plans are a separate writing aid.
h(1,'4 局部模态缺失下的情感预测模型')
p('局部缺失同时改变可用观测和跨模态交互条件。本章在统一输入与掩码规则下，构建观测状态递推候选，并引入MulT和EMT-DLFR适配结构进行对照。前者检验缺失间隔与历史记忆的作用，后两者检验跨模态注意力及恢复监督的作用。模型是否保留用于专项推理由第5章的预定验证规则决定，而非由结构复杂度或先验直觉决定。')
append(124,156)
h(2,'4.3 跨模态交互与特征恢复对照模型')
h(3,'4.3.1 普通基线与比较边界');append(174,175)
h(3,'4.3.2 MulT的成对跨模态注意力');append(175,176)
h(3,'4.3.3 EMT-DLFR的全局—局部交互');append(176,179)
h(2,'4.4 联合预测目标与两级恢复约束');append(158,165)
h(2,'4.5 连续区间缺失增强');append(166,168)
h(1,'5 实验验证、模型选择与专项预测')
p('实验依次回答四个问题：复杂融合结构是否优于简单基线，缺失模态与区间位置如何影响预测，递推与恢复机制是否带来稳定收益，以及冻结模型在专项样本上给出何种输出。各比较均保留原始数据划分、统一的输入预处理和明确的统计口径。')
h(2,'5.1 训练配置与模型选择规则');append(169,172)
h(2,'5.2 评价指标与结果统计')
h(3,'5.2.1 分类与回归指标');append(180,185)
h(3,'5.2.2 随机种子与分组重采样');append(186,188)
append(188,234)
h(2,'5.8 适用范围与后续任务衔接')
p('现有结果主要支持两点：输入掩码的正确处理是所有比较成立的前提；在本轮开发验证中，EMT-DLFR适配版的回归误差优于MLP，而分类改善与恢复机制收益仍存在不确定性。因此，对模型优势应限定到对应任务、缺失条件与统计口径，不能将单项指标提升扩展为全面最优。')
p('第一问后续需从附件1原视频自主提取特征并建立真实时间映射。已完成的100条原媒体核查只能作为处理清单，不能代替特征成果。第三问可复用第二问的预测接口和输入扰动工具，但模态贡献、局部证据及其忠实性需要单独验证，且必须回查附件4原素材。二者所需的证据不能由现有验证分数或重建表示替代。')
append(234,len(old))
# Renumber headings retained from the original experimental chapter.
section_map={'5.1':'4.1','5.2':'4.2','5.5':'5.3','5.6':'5.4','5.7':'5.5','5.8':'5.6','5.9':'5.7'}
# Only original headings are remapped; new headings above must retain numbers.
original_headings={b[2] for b in old if b[0]=='h'}
for b in blocks:
 if b[0]=='h' and b[2] in original_headings:
  match=re.match(r'(5\.\d+)(\.\d+)?( .*)',b[2])
  if match and match[1] in section_map:b[2]=section_map[match[1]]+(match[2] or '')+match[3]
# The new ending was not in original_headings, so remains 5.8.
# Number formulas in reading order and repair every formula/table reference.
eqmap={};tablemap={};figmap={};chapter=0;eqcounts={};tn=0;fn=0
for b in blocks:
 if b[0]=='h' and b[1]==1 and b[2][0].isdigit():chapter=int(b[2][0])
 if b[0]=='eq':
  eqcounts[chapter]=eqcounts.get(chapter,0)+1
  eqmap[b[2]]=f'{chapter}-{eqcounts[chapter]}';b[2]=eqmap[b[2]]
 if b[0]=='table':
  tn+=1;key=re.match(r'表(\d+)',b[1])[1];tablemap[key]=str(tn)
 if b[0]=='fig':
  fn+=1;key=re.match(r'图(\d+)',b[2])[1];figmap[key]=str(fn)

def rewrite(text):
 text=re.sub(r'（(\d+-\d+)）',lambda a:'（'+eqmap.get(a[1],a[1])+'）',text)
 text=re.sub(r'表(\d+)',lambda a:'表'+tablemap.get(a[1],a[1]),text)
 text=re.sub(r'图(\d+)',lambda a:'图'+figmap.get(a[1],a[1]),text)
 text=text.replace('本稿第4章给出可实施的数学方案并保留待填结果，避免把计划写成已完成实验。','因此，正文重点展开已实现的缺失预测模型与实验，第一问的待实施方案另存写作素材；第三问的证据映射与解释评估须待实际完成后纳入。')
 text=text.replace('其结构和训练目标在5.3、5.4节明确给出','其结构和训练目标在第4章给出')
 return text
for b in blocks:
 if b[0] in ('p','note','table'):b[1]=rewrite(b[1])
 if b[0]=='fig':
  b[2]=rewrite(b[2]);src=Path(b[1]);dest=OUT/'figures'/src.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(src,dest);b[1]=dest
# BERT remains used; omit references to tools not used in the new main text.
blocks=[b for b in blocks if not (b[0]=='p' and (b[1].startswith('[7] ') or b[1].startswith('[8] ')))]
for b in blocks:
 if b[0]=='p' and '经固定BERT-Mini编码' in b[1]:b[1]=b[1].replace('经固定BERT-Mini编码','经固定BERT-Mini编码[6]')
# Reuse the checked Word renderer, with a new cover and file naming.
renderer=inspect.getsource(m.render)
renderer=renderer.replace('复杂场景下多模态情感预测：第1—5章论文初稿','局部模态缺失下的多模态情感预测：论文重组稿')
renderer=renderer.replace('复杂场景下多模态情感预测\\n数学建模与算法设计','局部模态缺失下的\\n多模态情感预测')
renderer=renderer.replace('第1—5章论文初稿｜根据现有代码与实测结果整理','论文重组稿｜以已实现方法与实测结果为主线')
renderer=renderer.replace('本稿可编辑，章节按给定目录编排；增加1.3.4相关工作和5.7.4恢复约束消融，以对应实际方法与实验。','本稿根据已完成的研究重组章节，原目录仅作为选材参考。第4章集中说明模型，第5章集中呈现实验与专项预测。')
renderer=renderer.replace('第二问已有结果直接取自实验文件；第一问仅完成原视频清单核查，自主特征与对齐尚未完成，第4章所有待补内容均明确标记。','正文以第二问实测工作为主体。第一问与第三问保留任务背景及衔接说明；未执行的第一问方案和100条原始清单另存写作素材。')
renderer=renderer.replace('本文是写作初稿，尚未套用最终竞赛模板。提交前需补齐第一问实测内容及第三问正文，并统一参考文献、图表和附件体积。','本文为阶段性论文工作稿，并非整题完稿。后续应按第一问、第三问的实际成果调整整体章序，最终统一竞赛模板与参考文献格式。')
renderer=renderer.replace('第一问含待实施方案与真实原始清单，不包含虚构特征成果；第二问结果来自实测。','本稿以第二问已完成的建模与实测为主体；第一问和第三问完整成果尚待补充。')
renderer=renderer.replace('E题论文初稿_第1至5章','E题论文_按实际工作重组版')
m.OUT=OUT;m.blocks=blocks
exec(renderer,m.__dict__);m.render()
# Retain Q1 proposed methodology as an editable supporting file, not results.
source_md=(SOURCE/'E题论文初稿_第1至5章.md').read_text()
q1=source_md[source_md.index('# 4 问题一：'):source_md.index('# 5 问题二：')]
(OUT/'第一问待实施方案_写作素材.md').write_text('# 第一问后续写作素材（未实施方案）\n\n以下沿用旧稿编号，独立于重组正文。不得作为已完成的特征提取结果引用。\n\n'+q1)
shutil.copyfile(SOURCE/'附件1_100条原始样本核查清单_非特征结果.csv',OUT/'附件1_100条原始样本核查清单_非特征结果.csv')
(OUT/'重组说明.md').write_text('''# 本次重组说明

目录用作思路参考，按实际研究证据安排篇幅。

1. 问题背景与任务联系仍介绍三问；相关工作独立成1.4，总体路线为1.5。
2. 保留模型假设、符号、数据审查和预处理，将容易影响结论的输入问题说清楚。
3. 第4章围绕第二问组织数学描述、自建递推模型、MulT与EMT适配、训练目标及缺失增强。
4. 第5章集中说明训练/选模协议、实验结果、消融、错误分析及附件3预测。模型结构与验证结论分开叙述。
5. 第一问尚未实施的工具方案、100条清单不再占用正文一个长章，另存素材；第三问完整正文待获得解释证据后续写。

这是根据现有成果整理的阶段性工作稿，并非整题完整论文。EMT-DLFR适配版按预定规则用于专项预测，自建递推和两级恢复约束均未表现为所有指标稳定最优。现有验证用于模型选择，不是独立测试。

所有实验数值继续使用上一版已核验的文件，没有重新训练或改写结果。正文中的图、表、公式已按新顺序重新编号。
''')
print('New headings:',*[b[2] for b in blocks if b[0]=='h' and b[1]<=2],sep='\n')
