"""Regenerate the two figures from the adjacent recorded FineWeb trace."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
rows=json.loads((root/'fineweb.json').read_text())['rows']
names=['PyTorch LoRA reference','Preserve','Preserve + checkpoint','NVFP4','NVFP4 + checkpoint']
colors=['#697785','#1d72aa','#48a89b','#df6d27','#b94269']
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for row,name,color in zip(rows,names,colors):
    style='--' if row['checkpoint'] else '-'
    v=row['validation'];axes[0].plot([p['step'] for p in v],[p['loss'] for p in v],label=name,color=color,linestyle=style,marker='o',ms=3)
    losses=[p['loss'] for p in row['training']]
    smooth=[sum(losses[max(0,i-31):i+1])/len(losses[max(0,i-31):i+1]) for i in range(len(losses))]
    axes[1].plot(range(1,len(smooth)+1),smooth,color=color,linestyle=style,lw=1.5)
axes[0].set_title('Fixed-window validation cross-entropy');axes[1].set_title('Training loss: trailing 32 updates')
for ax in axes:ax.set_xlabel('Optimizer update');ax.set_ylabel('Cross-entropy');ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Pretrained GPT-2 · FineWeb · rank 16 · same seed and data\nCheckpoint curves overlap their corresponding non-checkpoint curves',fontsize=12)
fig.savefig(root/'loss.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
for row,name,color in zip(rows,names,colors):
    ax.plot([p['step'] for p in row['training']],[p['reserved_gib'] for p in row['training']],label=name,color=color,linestyle='--' if row['checkpoint'] else '-')
ax.set(title='GPU allocator reservation during 512 LoRA updates',xlabel='Optimizer update',ylabel='PyTorch reserved (GiB)')
ax.legend(fontsize=9);ax.grid(alpha=.2);fig.savefig(root/'memory.png',dpi=160)
