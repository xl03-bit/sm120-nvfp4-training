"""Regression for master-weight overflow, padded FP4 packing, and scalar checkpoint migration."""
import tempfile
from pathlib import Path
import torch
from torch import nn
from nvfp4_training import prepare_training, save_checkpoint, load_checkpoint, check_finite
from nvfp4_training._backend.master8 import decoded, encode
from nvfp4_training._backend.pair import pair


def model(precision):
    layer=nn.Linear(65,67,bias=False)
    with torch.no_grad():
        layer.weight.fill_(0.0001)
    return prepare_training(layer,profile="experimental_fp8_master",precision=precision,lr=.25,
                            betas=(0.,0.),eps=1e-8,weight_decay=0.)


def run(precision):
    layer,opt=model(precision)
    p=layer.weight
    initial=p.detach().view(torch.uint8).clone()
    units=p._v6_unit.clone()
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"state.pt"
        save_checkpoint(path,layer)
        legacy=torch.load(path,map_location="cpu",weights_only=True)
        legacy["model"]["_v6_unit_buffer"]=legacy["model"]["_v6_unit_buffer"][0].clone()
        torch.save(legacy,path)
        other,_=model(precision)
        assert load_checkpoint(path,other)==0
        assert torch.equal(other.weight._v6_unit,units)
        assert torch.equal(decoded(other.weight),decoded(p))
        grad=torch.zeros(p.shape,device="cuda")
        grad.flatten()[1]=1.
        grad.flatten()[-1]=-1.
        p.grad=grad.to(p.dtype)
        opt.step()
        check_finite(layer)
        vals=decoded(p).reshape(-1).float()
        assert .22 < -vals[1].item() < .29 and .22 < vals[-1].item() < .29
        assert (p._v6_unit>units).nonzero().flatten().tolist()==[0,len(units)-1]
        assert torch.equal(initial.flatten()[512:-512],p.view(torch.uint8).flatten()[512:-512])
        assert p._v6_unit[0]>units[0] and .22>448*units[0].item()
        plain=(p.float().flatten()*p._v6_unit.repeat_interleave(512)[:p.numel()]).reshape(p.shape).bfloat16()
        assert torch.equal(decoded(p),plain)
        scale=plain.float().abs().max()/2688
        encoded=encode(p,scale)
        reference=pair(plain,scale,(64,128,4))
        for left,right in zip(encoded,reference):
            assert torch.equal(left,right)
        save_checkpoint(path,layer)
        assert load_checkpoint(path,other)==1
        assert torch.equal(decoded(other.weight),plain)
        assert torch.equal(p._v6_unit,other.weight._v6_unit)
        p.grad.view(torch.uint8).flatten()[1]=127
        opt.step()
        try: check_finite(layer)
        except FloatingPointError: pass
        else: raise AssertionError("Nonfinite input was hidden")
    print(precision, "PASS: bounded stochastic FP8 output beyond the old finite range, local scale growth, "
          "unchanged unaffected blocks, both FP4 orientations, partial block, checkpoint migration, "
          "exact new checkpoint decoding and nonfinite detection.")


def test_fp8_master_range():
    for precision in ("fp4","bf16"):
        run(precision)


if __name__=="__main__":
    test_fp8_master_range()
