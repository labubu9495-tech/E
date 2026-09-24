"""Create the question-2 portion of the <=50 MB full competition attachment."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from importlib.metadata import version
import json
from data import ROOT,dump_json,sha256


def main():
    versions={name:version(name) for name in ['torch','numpy','transformers','safetensors','scikit-learn','scipy','matplotlib']}
    dump_json(ROOT/'results'/'runtime_versions.json',versions)
    requirements='\n'.join(f'{name}=={v.split("+")[0]}' for name,v in versions.items())+'\n'
    (ROOT/'requirements.txt').write_text(requirements)
    selection=json.loads((ROOT/'selection.json').read_text())
    paths=list(ROOT.glob('*.py'))+[ROOT/'README.md',ROOT/'method.md',ROOT/'requirements.txt',ROOT/'protocol.json',ROOT/'selection.json',ROOT/'assets'/'normalization.npz']
    paths += [p for p in (ROOT/'assets'/'bert-mini').iterdir() if p.suffix in ['.json','.txt','.safetensors']]
    paths += [ROOT/p for p in selection['checkpoints']]
    paths += [p for p in (ROOT/'results').iterdir() if p.suffix in ['.csv','.json','.md']]
    paths += list((ROOT/'results'/'figures').glob('*.png'))
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in paths}
    dump_json(ROOT/'submission_manifest.json',manifest)
    paths += [ROOT/'submission_manifest.json']
    target=ROOT/'submission_q2.zip'
    with ZipFile(target,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(paths):
            z.write(path,str(path.relative_to(ROOT)))
    with ZipFile(target) as z:
        assert z.testzip() is None
    size=target.stat().st_size
    if size>50_000_000:
        raise ValueError(f'Question2 package too large: {size} bytes')
    print(json.dumps(dict(package=str(target),bytes=size,decimal_MB=size/1e6,
        remaining_whole_submission_budget_bytes=50_000_000-size,files=len(paths)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
