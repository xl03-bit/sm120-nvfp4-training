"""GPU acceptance + conservative backwards-path autotuning for RTX 5060 Ti.

This is a correctness gate BEFORE benchmarking, not a silent runtime fallback.
All five choices are FP4 paths; SR/RHT are never disabled. A hardware-bound
selection is stored so subsequent Python processes use exactly verified code.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import statistics
import time
import torch
from torch import nn
from torch.nn import functional as F
from .._backend import nvfp4_fast_v2 as fast


def relative(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def wrapped(q):
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    return NVFP4Tensor(q.data, q.scale.view(torch.float8_e4m3fn), 16, q.dtype,
                       q.pts, None, True, False, None)


def padded(x):
    return F.pad(x, (0, fast.round128(x.shape[1]) - x.shape[1],
                     0, fast.round128(x.shape[0]) - x.shape[0])).contiguous()


def quantization_parity():
    from . import nvfp4_cuda as old
    from .hadamard import hadamard16
    torch.manual_seed(9107)
    signs = fast.get_signs(torch.device("cuda", 0))
    h = hadamard16("cuda")
    rows = []
    for dtype in (torch.bfloat16, torch.float32):
        a = torch.randn(128, 256, device="cuda", dtype=dtype)
        b = torch.randn(193, 129, device="cuda", dtype=dtype).t()
        c = torch.randn(130, 300, device="cuda", dtype=dtype)[::2, ::2]
        for label, x in (("aligned", a), ("transpose+pad", b), ("strided+pad", c)):
            for sr, rht in ((False, False), (True, False), (False, True), (True, True)):
                seed = 173
                ref = old.quant_nvfp4_cuda(padded(x), stochastic=sr, rht=rht,
                                           H=h, signs=signs, seed=seed)
                q = fast.quantize(x, sr=sr, rht=rht, signs=signs, seed=seed)
                torch.testing.assert_close(q.pts, ref.per_tensor_scale.reshape(()),
                                           rtol=5e-6, atol=1e-18)
                agreement = (q.data == ref.qdata).float().mean().item()
                err = relative(wrapped(q).dequantize(torch.float32),
                               ref.dequantize(torch.float32))
                if agreement < 0.99 or err > 0.005:
                    block_scale_agreement = (
                        q.scale.view(torch.uint8) == ref.scale.view(torch.uint8)
                    ).float().mean().item()
                    max_abs = (wrapped(q).dequantize(torch.float32) -
                               ref.dequantize(torch.float32)).abs().max().item()
                    raise RuntimeError(
                        f"Quantizer mismatch: {dtype} {label} SR={sr} RHT={rht}; "
                        f"byte agreement={agreement:.6f}, relative error={err:.6g}, "
                        f"block_scale_agreement={block_scale_agreement:.6f}, "
                        f"max_diff={max_abs:.9g}")
                rows.append({"dtype": str(dtype), "input": label, "sr": sr, "rht": rht,
                             "byte_agreement": agreement, "relative_error": err})
    qz = fast.quantize(torch.zeros(17, 35, device="cuda"), sr=True, rht=True,
                       signs=signs)
    if wrapped(qz).dequantize().abs().max().item() != 0:
        raise RuntimeError("Zero input did not remain zero")
    print(f"QUANTIZER PARITY: PASS ({len(rows)} explicit-seed cases) "
          f"legacy_odd_transpose={fast.stats()['legacy_fp4_quant_odd_transpose']}", flush=True)
    return rows


def sr_test():
    torch.manual_seed(833)
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    one = fast.quantize(x, sr=True, seed=10)
    two = fast.quantize(x, sr=True, seed=11)
    if torch.equal(one.data, two.data):
        raise RuntimeError("SR output is independent of seed")
    acc = torch.zeros_like(x, dtype=torch.float32)
    single = relative(wrapped(one).dequantize(torch.float32), x)
    for seed in range(32):
        q = fast.quantize(x, sr=True, seed=1000 + seed)
        acc += wrapped(q).dequantize(torch.float32)
    avg = relative(acc / 32, x)
    if avg >= single * 0.55:
        raise RuntimeError(f"SR averaging unexpectedly worse: {single=}, {avg=}")
    print(f"SR: PASS (single error {single:.5f}; mean32 {avg:.5f})", flush=True)
    return {"single_relative_error": single, "mean32_relative_error": avg}


def single_model_comparison(*, mode, batch, input_dim, hidden, output_dim,
                            input_dtype, save_packed):
    """Compare exact initialized weights, batch, target and explicit SR seeds."""
    from . import nvfp4_cuda as old_cuda
    from .adapter import enable_nvfp4 as enable_old
    torch.manual_seed(2027)
    ref_base = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(),
                             nn.Linear(hidden, output_dim)).cuda()
    ref = enable_old(copy.deepcopy(ref_base))
    new = copy.deepcopy(ref_base)
    fast.enable_nvfp4(new, save_quantized_activations=save_packed)
    for a, b in zip(new.parameters(), ref.parameters()):
        if not torch.equal(a, b):
            raise RuntimeError("Model parameters changed during FP4 conversion")
    inp = torch.randn(batch, input_dim, device="cuda", dtype=input_dtype,
                      requires_grad=True)
    cloned = inp.detach().clone().requires_grad_(True)
    target = torch.randn(batch, output_dim, device="cuda", dtype=torch.float32)
    old_cuda._SEED[0] = 0
    ref_out = ref(inp)
    F.mse_loss(ref_out.float(), target).backward()
    fast.reset_stats(0)
    new_out = new(cloned)
    F.mse_loss(new_out.float(), target).backward()
    errors = {"output": relative(new_out, ref_out),
              "input_grad": relative(cloned.grad, inp.grad)}
    for (name, p), (_, rp) in zip(new.named_parameters(), ref.named_parameters()):
        errors[name] = relative(p.grad, rp.grad)
        if not torch.isfinite(p.grad).all().item():
            raise RuntimeError(f"Non-finite gradient: {mode} {name}")
    # Compare MAX relative error among model tensors, not per-element relative
    # error, which is ill-defined around FP4 zeros.
    return errors


def _small_latency(mode, *, aligned):
    """Short complete optimizer-step timing, only among accepted options."""
    if aligned:
        m, k, n = 128, 1536, 1536
    else:
        m, k, n = 17, 150, 210
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(k, n), nn.GELU(), nn.Linear(n, n)).cuda()
    fast.enable_nvfp4(model)
    inp = torch.randn(m, k, device="cuda")
    target = torch.randn(m, n, device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(inp)
            loss = F.mse_loss(pred.float(), target.float())
        loss.backward()
        opt.step()
        return loss
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    trials = []
    for _ in range(2):
        start = time.perf_counter()
        for _ in range(8):
            step()
        torch.cuda.synchronize()
        trials.append((time.perf_counter() - start) / 8)
    return statistics.median(trials)


def select_modes():
    """Validate all three native implementations with the same paired-seed gate."""
    cases = {
        "irregular": [
            dict(batch=17, input_dim=150, hidden=210, output_dim=65,
                 input_dtype=torch.float32, save_packed=True),
            dict(batch=17, input_dim=150, hidden=210, output_dim=65,
                 input_dtype=torch.float32, save_packed=False),
            dict(batch=17, input_dim=150, hidden=210, output_dim=65,
                 input_dtype=torch.bfloat16, save_packed=True),
        ],
        "aligned": [
            dict(batch=128, input_dim=128, hidden=128, output_dim=128,
                 input_dtype=torch.float32, save_packed=True),
            dict(batch=128, input_dim=1536, hidden=1536, output_dim=1536,
                 input_dtype=torch.bfloat16, save_packed=True),
        ],
    }
    modes = ("fast_pair", "fast_reuse", "fast_fresh")
    chosen = {}
    evidence = {}
    for category, suite in cases.items():
        attempts = []
        for mode in modes:
            # Keep other category in reference mode for predictable test setup.
            fast.set_backward_modes(aligned=mode if category == "aligned" else "fast_reuse",
                                    irregular=mode if category == "irregular" else "fast_reuse")
            errs = []
            try:
                for case in suite:
                    outcome = single_model_comparison(mode=mode, **case)
                    errs.append(outcome)
                    if not all(math.isfinite(x) for x in outcome.values()):
                        raise RuntimeError(f"Non-finite parity metric: {outcome}")
                    if max(outcome.values()) > 0.03:
                        raise RuntimeError(f"gradient threshold exceeded (0.03): {outcome}")
                latency = _small_latency(mode, aligned=category == "aligned")
                status = "PASS"
            except Exception as ex:
                latency = None
                raise
            entry = {"mode": mode, "status": status,
                     "seconds_per_mini_step": latency, "per_case": errs}
            attempts.append(entry)
            print(f"[BACKWARD GATE] {category:9s} {mode:16s} {status}"
                  + (f" | {latency*1e3:.3f} ms" if latency else ""), flush=True)
        # The public profiles and long runs use paired row/column weight packing.
        # Timing on this small FP32 optimizer workload does not select their recipe.
        released = next(row for row in attempts if row["mode"] == "fast_pair")
        chosen[category] = released["mode"]
        evidence[category] = {"chosen": released["mode"], "mini_step_s": released["seconds_per_mini_step"],
                              "candidates": attempts}
        print(f"[BACKWARD SELECT] {category}: {chosen[category]} (verified release profile)", flush=True)
    fast.set_backward_modes(aligned=chosen["aligned"], irregular=chosen["irregular"])
    return chosen, evidence


def public_usage_test():
    torch.manual_seed(201)
    model = nn.Sequential(nn.Linear(150, 210), nn.SiLU(), nn.Linear(210, 65)).cuda()
    params_before = [id(p) for p in model.parameters()]
    names_before = list(model.state_dict())
    model = fast.enable_nvfp4(model)
    if params_before != [id(p) for p in model.parameters()] or names_before != list(model.state_dict()):
        raise RuntimeError("Conversion changed Parameter identity or state_dict keys")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    inp = torch.randn(17, 150, device="cuda", requires_grad=True)
    target = torch.randn(17, 65, device="cuda")
    original = torch._scaled_mm
    actual_fp4 = 0
    def trace(*args, **kwargs):
        nonlocal actual_fp4
        if (args[0].dtype != torch.float4_e2m1fn_x2 or
                args[1].dtype != torch.float4_e2m1fn_x2):
            raise RuntimeError("Non-FP4 GEMM detected in converted layer")
        actual_fp4 += 1
        return original(*args, **kwargs)
    fast.reset_stats(19)
    torch._scaled_mm = trace
    losses = []
    try:
        for _ in range(24):
            opt.zero_grad(set_to_none=True)
            inp.grad = None
            out = model(inp)
            loss = F.mse_loss(out.float(), target)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is None or not torch.isfinite(p.grad).all().item():
                    raise RuntimeError(f"Non-finite or missing gradient: {name}")
            opt.step()
            losses.append(loss.item())
        with torch.no_grad():
            model.eval()
            out = model(inp)
        if out.shape != target.shape or not torch.isfinite(out).all().item():
            raise RuntimeError("FP4 inference failed")
    finally:
        torch._scaled_mm = original
    counts = fast.stats()
    if losses[-1] >= losses[0]:
        raise RuntimeError(f"Training did not improve: {losses}")
    if actual_fp4 != 24 * 6 + 2:
        raise RuntimeError(f"Missing native FP4 GEMMs: {actual_fp4}")
    if min(counts["sr_quant"], counts["rht_quant"]) <= 0:
        raise RuntimeError(f"SR/RHT were not exercised: {counts}")
    print(f"PUBLIC USAGE: PASS (24 optimizer steps + FP4 inference, "
          f"{actual_fp4} native GEMMs, SR/RHT on)", flush=True)
    return {"losses": losses, "actual_fp4_gemms": actual_fp4, "counters": counts}


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 GPU required")
    print("Checking reference FP4 parity and the released fast_pair recipe.", flush=True)
    fast.extension()
    evidence = {"quantization": quantization_parity(), "sr": sr_test()}
    selected, candidates = select_modes()
    evidence["backward_candidates"] = candidates
    evidence["selected"] = selected
    evidence["usage"] = public_usage_test()
    from .gradient_stats import run as gradient_suite
    evidence["multi_seed"] = gradient_suite()
    from .convergence import run as convergence_suite
    evidence["convergence"] = convergence_suite()
    torch.cuda.synchronize()
    logs = fast.verification_file().parent
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    config_path = fast.write_verified_modes(aligned=selected["aligned"],
                                            irregular=selected["irregular"])
    print("V3 ACCEPTANCE PASS: FP4 GEMM + SR + RHT + gradients + padding + training/inference", flush=True)
    print(f"VERIFIED MODES: aligned={selected['aligned']}, irregular={selected['irregular']}", flush=True)
    print("Verification JSON:", logs / "verification.json", flush=True)
    print("Reusable backend config:", config_path, flush=True)

if __name__ == "__main__":
    main()
