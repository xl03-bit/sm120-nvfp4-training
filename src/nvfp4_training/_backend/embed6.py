"""Represent the tied input embedding's gradient in a minibatch-sized vocabulary."""
import torch
from torch import nn
from torch.nn import functional as F

def compact_gradient(ids, dout, vocab):
    ids=ids.contiguous().flatten()
    n=ids.numel()
    slots=torch.full((vocab,),n,device=ids.device,dtype=torch.int64)
    at=torch.arange(n,device=ids.device)
    slots.scatter_reduce_(0,ids,at,reduce='amin',include_self=True)
    local=slots.index_select(0,ids)
    values=torch.ops.aten.embedding_dense_backward(dout.reshape(n,-1).contiguous(),local,n,-1,False)
    return slots,values

class _TiedLookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, p):
        ctx.save_for_backward(ids,p)
        return F.embedding(ids,p)

    @staticmethod
    def backward(ctx, grad):
        ids,p=ctx.saved_tensors
        sink=getattr(p,'_v4_pending',None)
        if sink is None or (p,'embedding') in sink:
            return None,torch.ops.aten.embedding_dense_backward(grad,ids,p.shape[0],-1,False)
        sink[p,'embedding']=compact_gradient(ids,grad,p.shape[0])
        return None,None

class TiedInputEmbedding(nn.Embedding):
    def forward(self,ids):
        return _TiedLookup.apply(ids,self.weight)

def configure_tied(model,mode):
    from .dense6 import DenseStoredLinear
    for lookup in list(model.modules()):
        if (type(lookup) is not nn.Embedding or lookup.padding_idx is not None or lookup.max_norm is not None
                or lookup.sparse or lookup.scale_grad_by_freq or lookup.weight.numel()<4096):
            continue
        heads=[m for m in model.modules() if isinstance(m,nn.Linear) and m.weight is lookup.weight]
        if len(heads)!=1:continue
        if mode=='bf16':
            heads[0].__class__=DenseStoredLinear
            lookup.weight._v6_defer_dense=True
        lookup.__class__=TiedInputEmbedding
