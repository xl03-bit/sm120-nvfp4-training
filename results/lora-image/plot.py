"""Rebuild charts directly from the published, measured traces."""
from pathlib import Path
import json,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text())
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
records=read('flux-training.json');steps=[r for r in records if r['event']=='step']
valid={r['step']:r['loss'] for r in records if r['event']=='validation'}
x=[r['step'] for r in steps];loss=[r['loss'] for r in steps]
fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
ax[0].plot(x,loss,color='#a8bbd1',alpha=.55,lw=.5,label='Per-step flow MSE')
ax[0].plot(x,[statistics.mean(loss[max(0,i-19):i+1]) for i in range(len(loss))],color='#2463a0',label='20-step mean')
ax[0].set(title='FLUX.2 Klein NVFP4 LoRA: training',xlabel='Optimizer updates',ylabel='Flow MSE');ax[0].legend()
ax[1].plot(sorted(valid),[valid[k] for k in sorted(valid)],'o-',color='#b54e3f')
ax[1].set(title='Heldout photo: eight fixed noise/time pairs',xlabel='Optimizer updates',ylabel='Mean flow MSE')
fig.savefig(ROOT/'flux-loss.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
ax[0].plot(x,[r['peak_allocated_GiB'] for r in steps],label='Peak allocated')
ax[0].plot(x,[r['peak_reserved_GiB'] for r in steps],label='Peak reserved');ax[0].legend()
ax[0].set(title='Training memory',xlabel='Optimizer updates',ylabel='GiB',ylim=(0,9))
ax[1].plot(x,[r['seconds'] for r in steps],lw=.7)
ax[1].axvspan(568,601,color='#edba52',alpha=.35,label='Concurrent preview (excluded from clean timing)');ax[1].legend(fontsize=8)
ax[1].set(title='Full update time: all observations retained',xlabel='Optimizer updates',ylabel='Seconds')
fig.savefig(ROOT/'flux-memory-time.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
table=['| Mode | Mean s/update | Images/s | Speed vs BF16 | Peak allocated GiB | Peak reserved GiB | Validation at 0 / 40 |',
       '|---|---:|---:|---:|---:|---:|---:|']
baseline=read('benchmark/bf16/summary.json')['mean_seconds']
for key,label in [('bf16','BF16 preserve'),('cache','NVFP4 cache'),('recompute','NVFP4 recompute')]:
    rs=read('benchmark/'+key+'/trace.json');ts=[r for r in rs if r.get('event')=='step'];vs=[r for r in rs if r.get('event')=='validation']
    ax[0].plot([r['step'] for r in ts],[r['loss'] for r in ts],label=label,lw=1)
    ax[1].plot([r['step'] for r in ts],[r['peak_allocated_GiB'] for r in ts],label=label)
    s=read('benchmark/'+key+'/summary.json')
    table.append(f"| {label} | {s['mean_seconds']:.3f} | {s['images_per_second']:.3f} | {baseline/s['mean_seconds']:.2f}× | {s['peak_allocated_GiB']:.3f} | {s['peak_reserved_GiB']:.3f} | {vs[0]['loss']:.6f} / {vs[-1]['loss']:.6f} |")
for a in ax:a.set_xlabel('Optimizer updates');a.legend()
ax[0].set(title='Same training images, noise and timesteps',ylabel='Flow MSE')
ax[1].set(title='Training memory by mode',ylabel='Peak allocated GiB',ylim=(0,11))
fig.savefig(ROOT/'mode-comparison.png',dpi=160);plt.close(fig)
(ROOT/'benchmark-table.md').write_text('\n'.join(table)+'\n',encoding='utf8')
h=read('h3-training.json');hs=[r for r in h if r['event']=='step'];hv={r['step']:r for r in h if r['event']=='validation'}
fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
hl=[r['loss'] for r in hs]
ax[0].plot([r['step'] for r in hs],hl,color='#a8bbd1',lw=.6,label='Guidance-preserving train loss')
ax[0].plot([r['step'] for r in hs],[statistics.mean(hl[max(0,i-19):i+1]) for i in range(len(hl))],label='20-step mean')
ax[0].legend();ax[0].set(title='H3: 200 single-frame engineering updates',xlabel='Optimizer updates',ylabel='Training objective')
ax[1].plot([r['step'] for r in hs],[r['peak_allocated_gib'] for r in hs],label='Peak allocated')
ax[1].plot([r['step'] for r in hs],[r['peak_reserved_gib'] for r in hs],label='Peak reserved');ax[1].legend()
ax[1].set(title='H3 training memory',xlabel='Optimizer updates',ylabel='GiB',ylim=(0,16))
fig.savefig(ROOT/'h3-training.png',dpi=160);plt.close(fig)
print('\n'.join(table))
