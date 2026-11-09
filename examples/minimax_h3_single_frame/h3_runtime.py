"""H3 experiment integration. Frozen original weights or native FP4 + RHT/SR.

The original public release package and the downloaded checkpoints are never edited.
"""
from bootstrap import ROOT,torch
from torch import nn
import torch.nn.functional as F
import hashlib,json,time
import nv4_cuda as cuda
from musubi_tuner.minimax_h3.model import MiniMaxH3Config,MiniMaxH3Model
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

def signs(name):
    seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'little')%(2**63)
    return (torch.randint(0,2,(128,),generator=torch.Generator().manual_seed(seed))*2-1).float().cuda()

class _FrozenFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,layer):
        ctx.layer=layer
        if layer.mode=='preserve':return F.linear(x,cuda.dequant(*layer.quant),layer.bias)
        x2=x.reshape(-1,x.shape[-1]).contiguous()
        # Published forward weights are already quantized in their original basis.
        # Quantize activations in that same basis; do not re-quantize or rotate W.
        qa=cuda.quantize(x2)
        y=cuda.matmul(qa,layer.quant)[:len(x2)].reshape(*x.shape[:-1],layer.out_features)
        if layer.bias is not None:y=y+layer.bias
        return y
    @staticmethod
    def backward(ctx,g):
        layer=ctx.layer
        if layer.mode=='nvfp4':
            if layer.backward_strategy=='repack':
                q=cuda.transpose_rht_quant(layer.quant,layer.sign_out,layer.backward_scale)
            else:
                if layer.backward_quant is None:raise RuntimeError('Backward through a block declared frozen before all adapters')
                q=tuple(t.to(g.device,non_blocking=True) for t in layer.backward_quant)
            flat=g.reshape(-1,g.shape[-1]).contiguous()
            qa=cuda.quantize(flat,layer.sign_out,seed=layer.seed+layer.sr_counter,sr=True)
            layer.sr_counter+=1
            dx=cuda.matmul(qa,q)[:len(flat)].reshape(*g.shape[:-1],layer.in_features)
        else:dx=F.linear(g,cuda.dequant(*layer.quant).t())
        return dx,None

class FrozenQuantLinear(nn.Module):
    def __init__(self,q,s,g,bias=None,*,mode='preserve',name='',backward=False,backward_strategy='stream'):
        super().__init__();self.in_features=q.shape[1]*2;self.out_features=q.shape[0]
        self.mode=mode;self.seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
        self.sr_counter=0;self.backward_quant=None;self.backward_strategy=backward_strategy
        if mode=='nvfp4' and backward:
            self.register_buffer('sign_out',signs(name+':output'))
            self.register_buffer('backward_scale',cuda.backward_scale((q,s,g),self.sign_out))
            # Shared packed-only kernel: no full dense weight during loading or backward.
            if backward_strategy in ('stream','cached'):
                bq=cuda.transpose_rht_quant((q,s,g),self.sign_out,self.backward_scale)
                self.backward_quant=tuple(t.cpu() for t in bq) if backward_strategy=='stream' else bq
            elif backward_strategy!='repack':raise ValueError(backward_strategy)
        self.register_buffer('packed',q);self.register_buffer('block_scale',s);self.register_buffer('tensor_scale',g)
        self.register_buffer('bias',bias)
    @property
    def quant(self):return self.packed,self.block_scale,self.tensor_scale
    def forward(self,x):return _FrozenFn.apply(x,self)

class Adapter(nn.Module):
    def __init__(self,base,rank=8,alpha=8):
        super().__init__();self.base=base;self.rank=rank;self.alpha=alpha;self.enabled=True
        self.a=nn.Parameter(torch.empty(rank,base.in_features,device='cuda',dtype=torch.float32))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,device='cuda',dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a,a=5**.5)
    def forward(self,x):
        y=self.base(x)
        if self.enabled:
            h=F.linear(x,self.a.to(x.dtype));y=y+F.linear(h,self.b.to(x.dtype))*(self.alpha/self.rank)
        return y

class StoredPrecisionLinear(nn.Linear):
    """Keep source FP16 AdaLN parameters unchanged; return the residual stream's dtype."""
    def forward(self,x):return F.linear(x.to(self.weight.dtype),self.weight,self.bias).to(x.dtype)

def load_model(mode='preserve',first_train_block=20,rank=8,checkpoint=True,backward_cache_gib=0.0,fast_rope=True):
    assert mode in ('preserve','native','nvfp4')
    path=ROOT/'models/minimax_h3_ref2va_pruned_nvfp4.safetensors'
    config=MiniMaxH3Config(time_embed_dim=8,adaln_curve_grid=1025)
    model=MiniMaxH3Model(config,attn_mode='torch',split_attn=False,dtype=torch.bfloat16,device='meta')
    if fast_rope:
        from musubi_tuner.minimax_h3 import model as h3_impl
        h3_impl._apply_rope_split_half=cuda.rope
    model.requires_grad_(False)
    t=time.monotonic();consumed=set();count=0;cached=0;limit=int(backward_cache_gib*2**30)
    with MemoryEfficientSafeOpen(str(path),disable_numpy_memmap=True) as f:
        keys=set(f.keys())
        quant_stems=sorted(k[:-12] for k in keys if k.endswith('.comfy_quant'))
        assert len(quant_stems)==200,len(quant_stems)
        for stem in quant_stems:
            spec=json.loads(bytes(f.get_tensor(stem+'.comfy_quant').tolist()))
            assert spec['format']=='nvfp4',spec
            old=model.get_submodule(stem);index=int(stem.split('.')[1])
            q,s,g=[f.get_tensor(stem+suf,device=torch.device('cuda')) for suf in ('.weight','.weight_scale','.weight_scale_2')]
            size=sum(t.numel()*t.element_size() for t in (q,s,g))
            strategy='repack'
            if mode=='nvfp4' and index>=first_train_block and cached+size<=limit:
                strategy='cached';cached+=size
            layer=FrozenQuantLinear(q,s,g,mode=mode,name=stem,backward=index>=first_train_block,backward_strategy=strategy)
            assert layer.in_features==old.in_features and layer.out_features==old.out_features
            parent,_,leaf=stem.rpartition('.');setattr(model.get_submodule(parent),leaf,layer)
            consumed.update(stem+s for s in ('.weight','.weight_scale','.weight_scale_2','.comfy_quant'))
            count+=1
            if count%20==0:print(f'LOAD {mode} {count}/200 GPU={torch.cuda.memory_allocated()/2**30:.2f} GiB {time.monotonic()-t:.1f}s',flush=True)
        state=model.state_dict()
        for key in sorted(keys-consumed):
            if key not in state:raise ValueError('Unexpected checkpoint tensor '+key)
            v=f.get_tensor(key,device=torch.device('cuda'));parent,_,leaf=key.rpartition('.')
            owner=model.get_submodule(parent) if parent else model
            if tuple(v.shape)!=tuple(state[key].shape):raise ValueError('Checkpoint shape mismatch '+key)
            if leaf in owner._parameters:setattr(owner,leaf,nn.Parameter(v,requires_grad=False))
            else:setattr(owner,leaf,v)
            if isinstance(owner,nn.Linear) and v.dtype==torch.float16:owner.__class__=StoredPrecisionLinear
        del state
    assert not any(t.is_meta for t in list(model.parameters())+list(model.buffers()))
    torch.cuda.empty_cache()
    print(f'BACKWARD_CACHE GPU={cached/2**30:.3f} GiB CPU=0 GiB; other backward layouts rebuilt directly from packed FP4; fast_rope={fast_rope}',flush=True)
    model.eval()
    model._recipe=dict(mode=mode,rank=rank,alpha=rank,first_train_block=first_train_block,checkpoint=checkpoint,
                       source='FenomAI/MiniMax-H3@15646562c19be9d62aee6d28a7070d00d026027c')
    # Attach only after the entire frozen model has loaded; parameter initialization is reproducible.
    torch.manual_seed(20261109)
    if first_train_block<50:
        for name,m in list(model.named_modules()):
            if isinstance(m,FrozenQuantLinear) and int(name.split('.')[1])>=first_train_block:
                parent,_,leaf=name.rpartition('.');setattr(model.get_submodule(parent),leaf,Adapter(m,rank,rank))
    if checkpoint:model.enable_gradient_checkpointing()
    print('MODEL_READY',json.dumps(model._recipe),'trainable',sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)
    return model

def adapters(model,enabled):
    for m in model.modules():
        if isinstance(m,Adapter):m.enabled=enabled

def save_adapter(path,model,optimizer=None,**extra):
    tensors={k:v.detach().cpu() for k,v in model.named_parameters() if v.requires_grad}
    sr={k:m.sr_counter for k,m in model.named_modules() if isinstance(m,FrozenQuantLinear)}
    obj=dict(recipe=model._recipe,adapter=tensors,sr=sr,extra=extra,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
    if optimizer is not None:obj['optimizer']=optimizer.state_dict()
    temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)

def restore_adapter(path,model,optimizer=None):
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['recipe']==model._recipe
    p=dict(model.named_parameters())
    with torch.no_grad():
        for k,v in state['adapter'].items():p[k].copy_(v)
    modules=dict(model.named_modules())
    for k,v in state['sr'].items():modules[k].sr_counter=v
    if optimizer is not None:optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
    return state['extra']
