"""Package comparison evidence and a standalone Q2 inference runtime."""
import json
import sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from data import sha256,dump_json


def write_zip(target,items,base,extra=None):
    items=sorted(set(p for p in items if p.is_file()))
    with ZipFile(target,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for p in items:z.write(p,str(p.relative_to(base)))
        if extra:
            for name,content in extra.items():z.writestr(name,content)
    with ZipFile(target) as z:
        assert z.testzip() is None
    return dict(bytes=target.stat().st_size,sha256=sha256(target),files=len(items),
        contents={str(p.relative_to(base)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in items})


def main():
    selection=json.loads((HERE/'selection.json').read_text())
    evidence=list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+list(HERE.glob('*.csv'))+list(HERE.glob('*.md'))+list(HERE.glob('*.pdf'))+[HERE/'index.html']
    evidence+=list((HERE/'figures').glob('*'))+list((HERE/'LICENSES').glob('*'))
    evidence+=list((HERE/'runs').glob('*/metrics.json'))+list((HERE/'runs').glob('*/history.json'))+list((HERE/'runs').glob('*/config.json'))
    evidence=[p for p in evidence if p.name!='package_manifest.json']
    evidence_zip=ROOT/'results/第二问论文方法对照实验.zip'
    em=write_zip(evidence_zip,evidence,HERE)
    # Keep the directory layout used by infer_selected and ROOT-relative hashes.
    runtime=[ROOT/x for x in ['data.py','models.py','train.py','infer.py','prepare.py','protocol.json','requirements.txt','test_invariants.py']]
    # BertModel uses the fixed safetensors file. Do not also bundle the
    # duplicate original FP32 pytorch_model.bin checkpoint.
    runtime += [p for p in (ROOT/'assets/bert-mini').glob('*') if p.suffix in {'.json','.txt','.safetensors'}]+[ROOT/'assets/normalization.npz']
    runtime += [HERE/x for x in ['paper_models.py','run_experiments.py','infer_selected.py','protocol.json','selection.json','SOURCES.md','preflight.json','roundtrip_verification.json','test_paper_models.py','论文方法实测报告.md','seed_summary.csv','ensemble_comparison.csv','attachment3_overall_predictions.csv','attachment3_overall_predictions.provenance.json']]
    runtime += list((HERE/'LICENSES').glob('*'))
    for relative in selection['overall']['checkpoints']:
        p=ROOT/relative
        runtime.append(p)
        runtime += [p.parent/n for n in ['config.json','metrics.json','history.json','validation_predictions.npz']]
    experiment_rel=str(HERE.relative_to(ROOT))
    readme=f'''# 第二问论文方法实验选定模型

当前选定模型：{selection['selected']}。本包仅覆盖第二问，不是整题提交完成。

安装requirements.txt中的依赖；GPU环境需要对应PyTorch轮子。包内提供固定BERT-Mini、训练统计及三种子预测器。

在解压根目录运行：

```bash
python {experiment_rel}/infer_selected.py --input /path/to/附件3对齐版本 --device cpu
```

输出位于{experiment_rel}/attachment3_overall_predictions.csv及其provenance.json。附件3共30条无标签输入，不能报告专项准确率。

重建训练缓存可运行python prepare.py --data-root /path/to/E题 --device cuda。然后用{experiment_rel}/run_experiments.py运行对应方法。训练原始文件由赛题另外提供，不包含在本包中。原始对照实验的历史MLP/状态模型材料位于开发项目，未全部重复打包。

论文来源、MIT许可及适配说明见{experiment_rel}/SOURCES.md和LICENSES。模型是赛题适配实现，非原论文设置完全复现。验证结果参与选模，不能称独立测试成绩。

本包不含第一问100条自提特征、第三问解释模型/时间映射或原视频。整题所有附件总和仍须≤50MB。
'''
    runtime_zip=ROOT/'results/第二问论文实验选定模型_复现包.zip'
    rm=write_zip(runtime_zip,runtime,ROOT,{'README.md':readme})
    assert rm['bytes']<=50_000_000,'Q2 runtime alone exceeds the full competition limit'
    # Scan text members for obvious local identity paths; this is not a full
    # audit of all possible identifying text or figure metadata.
    hits=[]
    with ZipFile(runtime_zip) as z:
        for n in z.namelist():
            if Path(n).suffix in {'.py','.md','.json','.csv','.txt'}:
                text=z.read(n).decode('utf-8-sig',errors='replace')
                if any(s in text for s in ['dinglina','Lina_Ding','/home/daji/users/','C:/Users/','C:\\Users\\']):hits.append(n)
    assert not hits,hits
    dump_json(HERE/'package_manifest.json',dict(evidence=em,runtime=rm,
        remaining_whole_submission_budget_bytes=50_000_000-rm['bytes'],obvious_personal_path_hits=hits))
    print(json.dumps({'evidence_MB':em['bytes']/1e6,'runtime_MB':rm['bytes']/1e6,'selected':selection['selected']},ensure_ascii=False))


if __name__=='__main__':main()
