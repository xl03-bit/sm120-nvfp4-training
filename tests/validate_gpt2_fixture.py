"""Optional CPU audit against Transformers, in an environment already providing it."""
import json
from pathlib import Path
import sys
import torch
from transformers import GPT2LMHeadModel
from benchmark_lora import GPT2

torch.set_num_threads(4)
path=Path(sys.argv[1])
ref=GPT2LMHeadModel.from_pretrained(path,local_files_only=True,use_safetensors=True,
                                 dtype=torch.float32,attn_implementation='sdpa').eval()
ours=GPT2(path).eval()
torch.manual_seed(415)
tokens=torch.randint(0,50257,(2,32))
with torch.no_grad():
    a=ref(tokens,use_cache=False).logits
    b=ours(tokens)
err=((a-b).norm()/a.norm()).item()
assert err<2e-6,(err,(a-b).abs().max().item())
print(json.dumps(dict(status='passed',relative_logits_error=err,maximum_logits_error=(a-b).abs().max().item(),
                     parameters=sum(p.numel() for p in ours.parameters()),torch=torch.__version__)),flush=True)
