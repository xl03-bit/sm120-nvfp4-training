"""Mean next-token cross-entropy, with FP32 reductions and BF16 logits/gradients."""
import torch
import triton
import triton.language as tl

@triton.jit
def _parts(X, MX, SUM, V:tl.constexpr, TILES:tl.constexpr, B:tl.constexpr):
    row,tile=tl.program_id(0),tl.program_id(1)
    idx=tile*B+tl.arange(0,B)
    value=tl.load(X+row.to(tl.int64)*V+idx,idx<V,other=-float('inf')).to(tl.float32)
    m=tl.max(value,0)
    tl.store(MX+row*TILES+tile,m)
    tl.store(SUM+row*TILES+tile,tl.sum(tl.exp(value-m),0))

@triton.jit
def _rows(X,Y,MX,SUM,MAXROW,LOGSUM,LOSSES,V:tl.constexpr,TILES:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    ids=tl.arange(0,B)
    maxima=tl.load(MX+row*TILES+ids,ids<TILES,other=-float('inf'))
    sums=tl.load(SUM+row*TILES+ids,ids<TILES,other=0.)
    m=tl.max(maxima,0)
    logsum=tl.log(tl.sum(sums*tl.exp(maxima-m),0))
    label=tl.load(Y+row)
    target=tl.load(X+row.to(tl.int64)*V+label,(label>=0)&(label<V),other=float('nan')).to(tl.float32)
    tl.store(MAXROW+row,m)
    tl.store(LOGSUM+row,logsum)
    tl.store(LOSSES+row,(m-target)+logsum)

@triton.jit
def _back(X,Y,M,L,DL,DX,V:tl.constexpr,N:tl.constexpr,B:tl.constexpr):
    row,tile=tl.program_id(0),tl.program_id(1)
    ids=tile*B+tl.arange(0,B)
    v=tl.load(X+row.to(tl.int64)*V+ids,ids<V,other=0.).to(tl.float32)
    prob=tl.exp((v-tl.load(M+row))-tl.load(L+row))
    coeff=tl.load(DL)/N
    grad=(prob-(ids==tl.load(Y+row)).to(tl.float32))*coeff
    tl.store(DX+row.to(tl.int64)*V+ids,grad,ids<V)

class _TokenLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y):
        v=x.shape[-1]
        n=y.numel()
        b=1024
        tiles=triton.cdiv(v,b)
        mx=torch.empty((n,tiles),device=x.device,dtype=torch.float32)
        sums=torch.empty_like(mx)
        m=torch.empty(n,device=x.device,dtype=torch.float32)
        logsum=torch.empty_like(m)
        loss=torch.empty_like(m)
        _parts[(n,tiles)](x,mx,sums,v,tiles,b)
        _rows[(n,)](x,y,mx,sums,m,logsum,loss,v,tiles,triton.next_power_of_2(tiles))
        ctx.save_for_backward(x,y,m,logsum)
        return loss.mean()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,grad):
        x,y,m,logsum=ctx.saved_tensors
        v=x.shape[-1]
        dx=torch.empty_like(x)
        _back[(y.numel(),triton.cdiv(v,1024))](x,y,m,logsum,grad,dx,v,y.numel(),1024)
        return dx,None

def token_cross_entropy(logits,labels):
    """Unweighted, unmasked mean CE. Targets must be valid vocabulary indices."""
    if (logits.ndim<2 or not logits.is_cuda or logits.dtype!=torch.bfloat16 or not logits.is_contiguous()
            or labels.device!=logits.device or labels.dtype!=torch.int64 or not labels.is_contiguous()
            or labels.shape!=logits.shape[:-1] or labels.numel()==0 or logits.shape[-1]==0):
        raise ValueError("Expected contiguous CUDA BF16 logits and matching nonempty int64 targets")
    torch._assert_async(((labels>=0)&(labels<logits.shape[-1])).all(),"Invalid vocabulary index")
    return _TokenLoss.apply(logits,labels)
