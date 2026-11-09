"""Pinned HTTP Range downloads. Every part is resumable; final SHA256 must match HF LFS."""
from pathlib import Path
import argparse,concurrent.futures as cf,hashlib,json,os,threading,time
import requests

from bootstrap import ROOT
(ROOT/'metadata').mkdir(parents=True,exist_ok=True)
REPO='FenomAI/MiniMax-H3'
REV='15646562c19be9d62aee6d28a7070d00d026027c'
FILES=['minimax_h3_ref2va_pruned_nvfp4.safetensors',
       'vae/minimax_h3_video_vae_fp16.safetensors',
       'vae/minimax_h3_audio_vae_fp32.safetensors',
       'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors']
P=argparse.ArgumentParser();P.add_argument('--workers',type=int,default=12);P.add_argument('--chunk-mib',type=int,default=8)
P.add_argument('--probe',type=int,default=0);P.add_argument('--file',choices=FILES);args=P.parse_args()
CHUNK=args.chunk_mib*1024**2
log=ROOT/'metadata/http-download.jsonl'
def say(**data):
    data['time']=time.time();line=json.dumps(data);print(line,flush=True)
    with log.open('a',encoding='utf8') as f:f.write(line+'\n')
meta_path=ROOT/'metadata/hf-lfs.json'
if not meta_path.exists():
    response=requests.get(f'https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true',timeout=(15,45));response.raise_for_status()
    meta_path.write_text(json.dumps(response.json(),indent=2),encoding='utf8')
meta=json.loads(meta_path.read_text());assert meta['sha']==REV
by_name={x['rfilename']:x for x in meta['siblings']}
stop=threading.Event()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
for name in ([args.file] if args.file else FILES):
    info=by_name[name];size=info['size'];expected=info['lfs']['sha256']
    target=ROOT/'models'/name;target.parent.mkdir(exist_ok=True,parents=True)
    if target.exists():
        if target.stat().st_size!=size or sha(target)!=expected:raise RuntimeError('Existing file failed SHA256: '+name)
        say(event='already_verified',file=name);continue
    parts=ROOT/'parts'/Path(name).name;parts.mkdir(parents=True,exist_ok=True)
    count=(size+CHUNK-1)//CHUNK
    spec=dict(repo=REPO,revision=REV,file=name,bytes=size,sha256=expected,chunk_bytes=CHUNK)
    manifest=parts/'source.json'
    if manifest.exists():assert json.loads(manifest.read_text())==spec,'Part layout changed; use the original chunk size'
    else:manifest.write_text(json.dumps(spec,indent=2))
    ids=list(range(min(count,args.probe) if args.probe else count))
    def one(i):
        lo=i*CHUNK;hi=min(size,lo+CHUNK)-1;n=hi-lo+1
        done=parts/f'{i:05d}.bin';proof=parts/f'{i:05d}.sha256'
        if done.exists() and proof.exists() and done.stat().st_size==n and sha(done)==proof.read_text():return i,n,True
        for attempt in range(3):
            if stop.is_set():raise RuntimeError('Stopped after another part failed')
            temp=parts/f'{i:05d}.partial';start=time.monotonic()
            try:
                # Resolver signs each exact range. Never reuse a URL signed for a different range.
                with requests.get(f'https://huggingface.co/{REPO}/resolve/{REV}/{name}',
                    headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(15,90)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{size}':
                        raise RuntimeError(f'Incorrect range response: {r.status_code} {r.headers.get("Content-Range")}')
                    h=hashlib.sha256();got=0
                    with temp.open('wb') as f:
                        for b in r.iter_content(256*1024):
                            if stop.is_set():raise RuntimeError('Stopped after another part failed')
                            got+=len(b)
                            if got>n:raise RuntimeError('Range exceeded expected size')
                            f.write(b);h.update(b)
                if got!=n:raise RuntimeError(f'Short range: {got}/{n}')
                temp.replace(done);proof.write_text(h.hexdigest())
                return i,n,False
            except Exception as e:
                say(event='part_retry',file=name,part=i,attempt=attempt+1,error=type(e).__name__,seconds=round(time.monotonic()-start,1))
                if attempt==2:stop.set();raise RuntimeError(f'Download failed: {name} part {i}; valid parts retained') from None
                time.sleep(1+attempt)
    start=time.monotonic();downloaded=0;complete=0;cached=0;last=start
    say(event='start',file=name,parts=len(ids),workers=args.workers,bytes=size)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(one,i) for i in ids]
        try:
            for future in cf.as_completed(jobs):
                i,n,old=future.result();complete+=1;cached+=n if old else 0;downloaded+=0 if old else n
                now=time.monotonic()
                if now-last>=15 or complete==len(ids):
                    speed=downloaded/max(now-start,1)
                    say(event='progress',file=name,parts_done=complete,parts_total=len(ids),downloaded_gb=round(downloaded/1e9,3),
                        cached_gb=round(cached/1e9,3),MB_per_second=round(speed/1e6,2),seconds=round(now-start,1))
                    last=now
        except BaseException:
            stop.set()
            for f in jobs:f.cancel()
            raise
    if args.probe:
        say(event='probe_complete',file=name,seconds=round(time.monotonic()-start,1));break
    temp=target.with_suffix(target.suffix+'.assembling');h=hashlib.sha256()
    with temp.open('wb') as out:
        for i in range(count):
            with (parts/f'{i:05d}.bin').open('rb') as f:
                for b in iter(lambda:f.read(8*1024**2),b''):out.write(b);h.update(b)
    if temp.stat().st_size!=size or h.hexdigest()!=expected:raise RuntimeError('Final SHA256 mismatch: '+name)
    temp.replace(target)
    say(event='verified',file=name,sha256=expected,bytes=size,seconds=round(time.monotonic()-start,1))
    # Delete only this downloader's exact numbered files after the complete file passes the source hash.
    for i in range(count):
        for suffix in ('.bin','.sha256'):(parts/f'{i:05d}{suffix}').unlink()
