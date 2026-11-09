"""Fetch pinned dataset/processor files; large weights use download_parts.py."""
from bootstrap import ROOT,CODE_ROOT
import json
from huggingface_hub import hf_hub_download

def main():
    spec=json.loads((CODE_ROOT/'sources.json').read_text())
    (ROOT/'kernel-cache/general').mkdir(parents=True,exist_ok=True)
    for key,kind,folder in [('dataset','dataset','dataset'),('official','model','official')]:
        item=spec[key]
        for name in item['files']:
            print(f"DOWNLOAD {item['repo']}@{item['revision']} {name}",flush=True)
            hf_hub_download(repo_id=item['repo'],revision=item['revision'],filename=name,
                            repo_type=kind,local_dir=ROOT/folder)
    print('Pinned dataset and processor files ready; no packages installed.',flush=True)
if __name__=='__main__':main()
