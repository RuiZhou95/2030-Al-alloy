"""Stage released numerical reference outputs without changing the input data.

This is a replay prerequisite, not evidence that the training was rerun.
"""
from pathlib import Path
import hashlib,json,shutil
from release_paths import REPO_ROOT, OUTPUT_ROOT


def main():
    manifest=json.loads((REPO_ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        source=REPO_ROOT/record['path']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError(f'Input checksum mismatch: {record["path"]}')
    count=0
    for section,subdir in [('scientific','scientific/data'),('paper','paper/data'),('descriptors','paper/data_materials_v2')]:
        for source in sorted((REPO_ROOT/'data/reference'/section).glob('*')):
            if not source.is_file():continue
            dest=OUTPUT_ROOT/subdir/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and dest.read_bytes()!=source.read_bytes():
                raise FileExistsError(f'Refusing to overwrite a different result: {dest.name}. Use a new ALLOY_OUTPUT_ROOT.')
            shutil.copyfile(source,dest);count+=1
    print(f'Verified {len(manifest["files"])} data files; staged {count} reference tables. No model training was performed.')


if __name__=='__main__':main()
