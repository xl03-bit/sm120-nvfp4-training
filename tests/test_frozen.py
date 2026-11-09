"""Synthetic regression tests for frozen storage, caching and graph replay."""
import copy,tempfile,unittest
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from nvfp4_training import pack_frozen_layers,save_frozen_cache,load_frozen_cache,capture_frozen
from nvfp4_training import frozen

class PolicyTests(unittest.TestCase):
    def test_reject_trainable_before_mutation(self):
        m=nn.Sequential(nn.Linear(8,8).requires_grad_(False),nn.Linear(8,8))
        with self.assertRaises(ValueError):pack_frozen_layers(m,include=('*',))
        self.assertIs(type(m[0]),nn.Linear)
    def test_explicit_patterns(self):
        m=nn.Sequential(nn.Linear(8,8).requires_grad_(False))
        with self.assertRaises(ValueError):pack_frozen_layers(m,include='0')
    def test_reject_unsupported_convolution(self):
        m=nn.Sequential(nn.Conv2d(8,8,3).requires_grad_(False))
        with self.assertRaises(ValueError):pack_frozen_layers(m,include=('*',))
    def test_reject_aliases(self):
        layer=nn.Linear(8,8).requires_grad_(False)
        m=nn.ModuleDict({'first':layer,'second':layer})
        with self.assertRaises(ValueError):pack_frozen_layers(m,include=('*',))
    def test_reject_shared_boundary_weight(self):
        m=nn.ModuleDict({'core':nn.Linear(8,8),'boundary':nn.Linear(8,8)}).requires_grad_(False)
        m['boundary'].weight=m['core'].weight
        with self.assertRaises(ValueError):pack_frozen_layers(m,include=('core',))
    def test_protected_attention_descendant(self):
        m=nn.MultiheadAttention(8,2).requires_grad_(False)
        with self.assertRaisesRegex(ValueError,'No eligible'):
            pack_frozen_layers(m,include=('*',),keep=('out_proj',))

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(128,128)
        self.frozen=nn.Linear(128,128).requires_grad_(False)
        self.output=nn.Linear(128,4)
    def forward(self,x):
        x=self.input(x)
        for _ in range(2):x=x+0.1*self.frozen(x)
        return self.output(x)

@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability()==(12,0),'SM120 CUDA GPU required')
class GPUTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.backend=frozen.runtime()[0]
    def test_native_parity_and_cache(self):
        for k,n in ((128,256),(65,67)):
            original=nn.Linear(k,n,device='cuda',dtype=torch.bfloat16).requires_grad_(False)
            model=pack_frozen_layers(copy.deepcopy(original),include=('*',))
            _,_,_,native=__import__('nvfp4_training._runtime',fromlist=['load_runtime']).load_runtime()
            reference=copy.deepcopy(original);reference.__class__=native
            reference.register_buffer('_nvfp4_signs',self.backend.get_signs(original.weight.device),persistent=False)
            reference._nvfp4_save_packed=True;reference.weight._v6_remat=True;reference.weight._v6_fast_scale=True
            x=torch.randn(17,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            g=torch.randn(17,n,device='cuda',dtype=torch.bfloat16)
            self.backend._SEED=123;a=reference(x);ga=torch.autograd.grad(a,x,g)[0]
            self.backend._SEED=123;b=model(x);gb=torch.autograd.grad(b,x,g)[0]
            self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(ga,gb))
            self.assertEqual(frozen.inventory(model)['master_parameter_elements'],0)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'cache.pt'
                save_frozen_cache(path,model,model_fingerprint='synthetic-v1')
                count=frozen.PACKS
                restored=load_frozen_cache(path,copy.deepcopy(original),expected_fingerprint='synthetic-v1')
                self.assertEqual(frozen.PACKS,count)
                self.assertTrue(torch.equal(model(x),restored(x)))
                with self.assertRaises(ValueError):load_frozen_cache(path,copy.deepcopy(original),expected_fingerprint='wrong-revision')
                with path.open('ab') as f:f.write(b'changed')
                with self.assertRaisesRegex(ValueError,'checksum'):
                    load_frozen_cache(path,copy.deepcopy(original),expected_fingerprint='synthetic-v1')
    def test_pointwise_and_attention_cache(self):
        modules=[nn.Conv2d(16,32,1).cuda().bfloat16(),nn.MultiheadAttention(128,4,batch_first=True).cuda().bfloat16()]
        for original in modules:
            original.requires_grad_(False)
            model=pack_frozen_layers(copy.deepcopy(original),include=('*',))
            x=torch.randn(2,16,3,3,device='cuda',dtype=torch.bfloat16) if isinstance(original,nn.Conv2d) else torch.randn(2,9,128,device='cuda',dtype=torch.bfloat16)
            call=lambda m:m(x) if isinstance(original,nn.Conv2d) else m(x,x,x,need_weights=False)[0]
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'cache.pt';save_frozen_cache(path,model,model_fingerprint='synthetic-v1')
                restored=load_frozen_cache(path,copy.deepcopy(original),expected_fingerprint='synthetic-v1')
                self.assertTrue(torch.equal(call(model),call(restored)))
    def test_graph_backward_dynamic_sr_and_changed_input(self):
        model=pack_frozen_layers(Toy().cuda(),include=('frozen',),keep=('input','output'))
        x=torch.randn(8,128,device='cuda');target=torch.randn(8,4,device='cuda')
        objective=lambda m,x,y:F.mse_loss(m(x).float(),y)
        self.backend._SEED=0
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=objective(model,x,target)
        loss.backward();expected=[p.grad.detach().clone() for p in model.parameters() if p.requires_grad]
        del loss
        with capture_frozen(model,objective,x,target) as replay:
            self.assertEqual(replay.sr_calls,2)
            value=replay.replay(x,target).clone();torch.cuda.synchronize()
            grads=[p.grad.detach().clone() for p in model.parameters() if p.requires_grad]
            for a,b in zip(expected,grads):self.assertTrue(torch.allclose(a,b,rtol=1e-5,atol=1e-6))
            replay.replay(x,target);torch.cuda.synchronize()
            self.assertTrue(any(not torch.equal(a,p.grad) for a,p in zip(grads,[p for p in model.parameters() if p.requires_grad])))
            changed=replay.replay(x+0.25,target).clone();torch.cuda.synchronize()
            self.assertFalse(torch.equal(value,changed))
            with self.assertRaises(ValueError):replay.replay(x[:1],target[:1])
            model.frozen.data=model.frozen.data.clone()
            with self.assertRaisesRegex(RuntimeError,'packed buffers'):
                replay.replay(x,target)
    def test_inference_graph(self):
        model=pack_frozen_layers(nn.Linear(128,64).cuda().requires_grad_(False),include=('*',))
        x=torch.randn(8,128,device='cuda')
        with torch.no_grad():expected=model(x)
        with capture_frozen(model,lambda m,x:m(x),x,backward=False) as replay:
            actual=replay.replay(x).clone();torch.cuda.synchronize()
            self.assertTrue(torch.equal(actual,expected))
            self.assertEqual(replay.sr_calls,0)
    def test_input_gradient_replay(self):
        model=pack_frozen_layers(nn.Linear(128,64).cuda().requires_grad_(False),include=('*',))
        x=torch.randn(8,128,device='cuda',requires_grad=True)
        objective=lambda m,x:m(x).float().square().mean()
        self.backend._SEED=0
        loss=objective(model,x)
        expected=torch.autograd.grad(loss,x)[0]
        del loss
        with capture_frozen(model,objective,x) as replay:
            replay.replay(x);torch.cuda.synchronize()
            self.assertTrue(torch.equal(expected,replay.inputs[0].grad))
            self.assertIsNone(x.grad)  # External input graphs are not captured.

if __name__=='__main__':unittest.main()
