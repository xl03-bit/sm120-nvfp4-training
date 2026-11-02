"""Render the recorded loss and GPU memory curves. No GPU is needed."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np


def render(path, out):
    data = json.loads(path.read_bytes())
    if data["status"] != "complete":
        raise ValueError(f"Run is {data['status']}")
    out.mkdir(exist_ok=True, parents=True)
    train = data["steps"]
    valid = data["validation"]
    caps = data["captures"]
    x = np.array([r["step"] for r in train])
    if not np.array_equal(x, np.arange(1, 2049)) or valid[-1]["step"] != 2048:
        raise ValueError("Incomplete trajectory")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "axes.spines.left":False, "axes.spines.bottom":False,
                         "xtick.color":"#35434c", "ytick.color":"#35434c",
                         "axes.labelcolor":"#26383f", "text.color":"#152f37",
                         "savefig.facecolor":"white"})
    def canvas(title):
        fig, ax = plt.subplots(figsize=(10, 5.35))
        fig.subplots_adjust(left=.085, right=.95, top=.80, bottom=.21)
        fig.text(.085, .925, title, fontsize=18, weight="bold")
        fig.text(.085, .86, "4.064B parameters   /   RTX 5060 Ti 16GB   /   FineWeb   /   2 × 512 tokens per update",
                 color="#435a63", fontsize=10.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dee6e8", linewidth=.85)
        ax.set_xlim(-15,2065)
        ax.set_xticks([0,256,512,768,1024,1280,1536,1792,2048])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda n,_:format(int(n),",")))
        ax.tick_params(length=0, pad=5)
        ax.set_xlabel("AdamW update",labelpad=7)
        return fig,ax
    loss = np.array([r["ce"] for r in train])
    vx = np.array([r["step"] for r in valid])
    vl = np.array([r["ce"] for r in valid])
    fig,ax = canvas("Training and held-out cross-entropy")
    ax.plot(x,loss,color="#bccdd2",lw=.65,label="Training batch")
    ax.plot(x[31:],np.convolve(loss,np.ones(32)/32,mode="valid"),
            color="#126d71",lw=1.8,label="Training, trailing 32 updates")
    ax.plot(vx,vl,"o-",color="#b45725",lw=2,ms=4.7,label="Fixed validation set")
    ax.set_ylabel("Cross-entropy (nats/token)")
    ax.legend(frameon=False, loc="upper right")
    fig.text(.085,.035,f"Seed 1234  ·  2,097,152 training tokens  ·  64 held-out batches per evaluation  ·  Final validation CE {vl[-1]:.4f}",fontsize=9)
    fig.savefig(out/"loss.png",dpi=165)
    plt.close(fig)
    fig,ax = canvas("GPU memory during training and validation")
    def gib(rows,key):
        return np.array([r[key] for r in rows])/2**30
    ax.plot(x,gib(train,"device_used"),color="#384550",ls="--",lw=1.4,label="Device used after update")
    ax.plot(x,gib(train,"peak_reserved"),color="#206482",lw=2.2,label="Training reserved")
    ax.plot(x,gib(train,"peak_allocated"),color="#18806a",lw=2.2,label="Training allocated")
    ax.scatter(vx,gib(valid,"peak_reserved"),s=30,edgecolors="#b45725",facecolors="white",
               zorder=4,label="Validation reserved")
    ax.scatter(vx,gib(valid,"peak_allocated"),s=20,color="#b45725",zorder=4,label="Validation allocated")
    ax.scatter([r["step"] for r in caps],gib(caps,"peak_reserved"),s=55,marker="x",
               color="#77618d",zorder=5,label="Graph capture reserved")
    total=data["device_total_bytes"]/2**30
    ax.axhline(total,color="#60696e",ls=":",lw=1.2,label=f"Device capacity ({total:.2f} GiB)")
    ax.set_ylim(0,total+1)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylabel("GPU memory (GiB)")
    ax.legend(frameon=False,loc="lower center",ncol=2)
    high=max(r["peak_reserved"] for r in [data["initialization"],*train,*valid,*caps])/2**30
    fig.text(.085,.035,f"PyTorch allocator peaks per update or evaluation  ·  Highest recorded reservation {high:.3f} GiB",fontsize=9)
    fig.savefig(out/"memory.png",dpi=165)
    plt.close(fig)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("metrics",type=Path)
    ap.add_argument("out",type=Path)
    args=ap.parse_args()
    render(args.metrics,args.out)
