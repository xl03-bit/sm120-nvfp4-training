"""Read-only H3 readiness check; no training, download or package installation."""
from bootstrap import ROOT,CODE_ROOT,MUSUBI_ROOT,torch
import argparse,hashlib,importlib.metadata,json,subprocess

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--verify-model-hashes',action='store_true');a=p.parse_args()
    spec=json.loads((CODE_ROOT/'sources.json').read_text());errors=[]
    def require(condition,message):
        if not condition:errors.append(message)
    revision=subprocess.run(['git','-C',str(MUSUBI_ROOT),'rev-parse','HEAD'],capture_output=True,text=True)
    require(revision.returncode==0 and revision.stdout.strip()==spec['musubi']['revision'],'Check the pinned musubi source revision')
    require((ROOT/'kernel-cache').is_dir(),'Run download_inputs.py to prepare the example work directory')
    for item in spec['model']['files']:
        path=ROOT/'models'/item['file']
        require(path.is_file() and path.stat().st_size==item['bytes'],'Missing or incorrect model file: '+item['file'])
        if a.verify_model_hashes and path.is_file():require(digest(path)==item['sha256'],'Model SHA256 mismatch: '+item['file'])
    for name in spec['official']['files']:
        require((ROOT/'official'/name).is_file(),'Missing official processor/config file: '+name)
    manifest=ROOT/'data/manifest.json';prompts=ROOT/'cache/prompts.json'
    for path,key in [(manifest,'manifest_sha256'),(prompts,'prompts_sha256')]:
        require(path.is_file() and digest(path)==spec['cached_recipe'][key],'Recorded recipe cache mismatch: '+str(path.relative_to(ROOT)))
    if manifest.is_file():
        rows=json.loads(manifest.read_text(encoding='utf8'))
        require(sum(r['split']=='train' for r in rows)==32 and sum(r['split']=='heldout' for r in rows)==8,'Expected the original 32/8 split')
        for r in rows:
            for path in [ROOT/r['path'],ROOT/'cache'/f"item-{r['id']:03d}.pt",ROOT/'cache'/f"image-{r['id']:03d}.pt"]:
                require(path.is_file(),'Missing cached input: '+str(path.relative_to(ROOT)))
    for name in ['uncond','silence','style-fox','style-dragon','style-robot','control-photo','control-landscape','control-portrait']:
        require((ROOT/'cache'/(name+'.pt')).is_file(),'Missing probe cache: '+name)
    versions={}
    for name in ['torch','transformers','diffusers','accelerate','safetensors','einops','numpy','pillow','av','pyarrow','toml','voluptuous','huggingface-hub','requests']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:errors.append('Missing dependency: '+name)
    require(torch.cuda.is_available(),'CUDA is unavailable')
    if torch.cuda.is_available():require(torch.cuda.get_device_capability()==(12,0),'The packed backend requires SM120')
    require(hasattr(torch.nn.functional,'scaled_mm'),'This PyTorch lacks torch.nn.functional.scaled_mm')
    if not errors:
        import h3_runtime
        from musubi_tuner.minimax_h3 import packing,sampling,video_vae,audio_vae,text_encoder
    report=dict(status='failed' if errors else 'passed',scope='readiness and imports only; no training run',
        model_hashes_verified=a.verify_model_hashes,versions=versions,errors=errors)
    print(json.dumps(report,indent=2),flush=True)
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
