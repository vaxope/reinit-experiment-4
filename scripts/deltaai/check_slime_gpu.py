"""GPU check of a freshly built slime image (runs at the end of build_slime.sbatch, ~1-2 min).

Purpose : catch a broken GPU stack before any paid training job: the parts of the image that
          training actually runs on the GPU, each checked against a plain reference.
            0. Every module the image must provide imports WITH the GPU driver (the build's own
               import check runs without one and defers driver-only failures to here).
               IMPORT_MODULES must equal the recipe's step-12 list (tests/test_slime_container.py).
            1. CUDA visible, GPU is sm90 (GH200).
            2. TransformerEngine: te.pytorch.Linear forward + backward = torch matmul.
            2b. TransformerEngine attention as Megatron runs Qwen3.5's full-attention layers on
               slime's packed batches (added 2026-11-19): DotProductAttention with the model's
               heads from configs/train_reinit.toml, once for the whole model and once per
               tensor-parallel GPU (Qwen3.5-4B, TP 2: 16/4 and 8/2 heads/groups, head size 256), causal, packed
               variable-length sequences (qkv_format "thd" + cu_seqlens, as Megatron's
               TEDotProductAttention passes them), bf16, forward + backward = PyTorch
               scaled_dot_product_attention per sequence (fp32). The image has neither
               flash-attn nor FA3, so this is TE's own (cuDNN) kernel; if no kernel supports
               these shapes, training would fail only in its first step.
            3. flash-linear-attention, exactly as Qwen3.5's linear-attention layers call it in
               training (slime_plugins/models/qwen3_5.py Qwen3_5GatedDeltaNet):
               a. chunk_gated_delta_rule (the fast kernel training uses) = a plain-PyTorch
                  step-by-step version of the same recurrence (gated_delta_rule_reference below,
                  autograd for the gradients), on two packed sequences of different lengths
                  (cu_seqlens), with use_qk_l2norm_in_kernel=True; outputs and all gradients.
                  (fla's own fused_recurrent_gated_delta_rule has no backward pass in fla 0.5.2,
                  build job 3620006, so it cannot be the gradient reference.)
               b. ShortConvolution (causal depthwise conv + SiLU, varlen) = torch conv1d per sequence;
               c. FusedRMSNormGated (RMSNorm * weight * SiLU(gate)) = the plain formula.
            4. torch_memory_saver imports (it is exercised for real in the first training step).
            5. cuDNN (added 2026-11-27, diagnosis job 3686686): after TE has run, every libcudnn file
               the process loaded comes from ONE folder, and torch's own cuDNN check passes. The
               image has PyTorch's cuDNN 9.20 and a system 9.14; TE loads the system one first,
               and the mix gave wrong attention gradients (check 2b, job 3680134). Jobs run with
               scripts/deltaai/slime_env.sh, which points TE at PyTorch's copy.
Inputs  : none (random tensors, fixed seed). No model weights, no data.
Output  : one line per check; exit 1 if any check fails. Run INSIDE the slime image:
            apptainer exec --nv <slime.sif> python3 scripts/deltaai/check_slime_gpu.py
"""
import importlib
import sys
from pathlib import Path

import torch

# Modules the image must provide (same list as containers/slime_gh200.def step 12).
IMPORT_MODULES = ["torch", "sglang", "sglang_router", "megatron.core", "transformer_engine", "fla",
                  "torch_memory_saver", "ray", "transformers", "slime", "slime_plugins.models.qwen3_5"]

# Tolerances for bf16 kernels vs references computed from the same bf16 inputs. bf16 keeps
# ~3 significant digits, so element-wise differences of a few 1e-2 relative are normal; these
# bounds catch wrong results (orders of magnitude off), not rounding.
TE_ATOL, TE_RTOL = 5e-2, 5e-2
FLA_MAX_REL_ERR = 5e-2
SEED = 0
# Small test shapes. SEQ_LENS: two packed sequences of different lengths, so the varlen path
# (cu_seqlens) that slime uses for every training batch is exercised, including a length that is
# not a multiple of fla's 64-token chunk. CONV_KERNEL = Qwen3.5's linear_conv_kernel_dim (4,
# config.json). NORM_EPS = Qwen3.5's rms_norm_eps (1e-6, config.json).
SEQ_LENS = (100, 156)
HEADS, HEAD_DIM = 2, 64
CONV_CHANNELS, CONV_KERNEL = 128, 4
NORM_EPS = 1e-6
# Full-attention shapes come from configs/train_reinit.toml [megatron_model] (2026-11-24; this check
# had Qwen3.5-2B's 8 / 2 / 256 written in): the whole model's heads, and what each GPU runs under
# [hardware] tensor_model_parallel_size (heads and key/value groups divided by it). For Qwen3.5-4B
# with tensor parallel 2: 16/4/256 and 8/2/256.
ATTN_MAX_REL_ERR = 5e-2


def attention_shapes(path=None):
    """[(query heads, key/value heads, head dim), ...] to test: full model, then per tensor-parallel rank."""
    import tomllib
    path = path or Path(__file__).resolve().parents[2] / "configs" / "train_reinit.toml"
    cfg = tomllib.loads(Path(path).read_text())
    m, tp = cfg["megatron_model"], cfg["hardware"]["tensor_model_parallel_size"]
    shapes = [(m["num_attention_heads"], m["num_query_groups"], m["kv_channels"])]
    if tp > 1:
        shapes.append((m["num_attention_heads"] // tp, m["num_query_groups"] // tp, m["kv_channels"]))
    return shapes


def report(name, ok, detail=""):
    print(f"[check_slime_gpu] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    return ok


def check_imports():
    failed = {}
    for name in IMPORT_MODULES:
        try:
            importlib.import_module(name)
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {e}"
    return report("imports", not failed, f"{len(IMPORT_MODULES) - len(failed)}/{len(IMPORT_MODULES)} "
                  + (f"failed: {failed}" if failed else ""))


def check_cuda():
    ok = torch.cuda.is_available()
    cap = torch.cuda.get_device_capability() if ok else None
    return report("cuda", ok and cap == (9, 0), f"available={ok} capability={cap} "
                  f"device={torch.cuda.get_device_name() if ok else None}")


def check_te():
    import transformer_engine.pytorch as te
    torch.manual_seed(SEED)
    lin = te.Linear(256, 512, bias=False, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = lin(x)
    ref = x @ lin.weight.t()
    y.float().sum().backward()
    ok = torch.allclose(y.float(), ref.float(), atol=TE_ATOL, rtol=TE_RTOL) and x.grad is not None
    return report("transformer_engine Linear", ok, f"max|diff|={(y - ref).abs().max().item():.3e}")


def check_te_attention():
    return all([check_te_attention_shape(*shape) for shape in attention_shapes()])


def check_te_attention_shape(ATTN_HEADS, ATTN_KV_HEADS, ATTN_HEAD_DIM):   # noqa: N803 (names kept from before)
    import transformer_engine.pytorch as te
    torch.manual_seed(SEED)
    total, groups = sum(SEQ_LENS), ATTN_HEADS // ATTN_KV_HEADS
    cu = cu_seqlens().to(torch.int32)
    q = torch.randn(total, ATTN_HEADS, ATTN_HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total, ATTN_KV_HEADS, ATTN_HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(total, ATTN_KV_HEADS, ATTN_HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    attn = te.DotProductAttention(num_attention_heads=ATTN_HEADS, kv_channels=ATTN_HEAD_DIM,
                                  num_gqa_groups=ATTN_KV_HEADS, attention_dropout=0.0,
                                  qkv_format="thd", attn_mask_type="padding_causal")
    out = attn(q, k, v, cu_seqlens_q=cu, cu_seqlens_kv=cu,
               max_seqlen_q=max(SEQ_LENS), max_seqlen_kv=max(SEQ_LENS))
    out = out.reshape(total, ATTN_HEADS, ATTN_HEAD_DIM)
    grad_out = torch.randn_like(out)
    gq, gk, gv = torch.autograd.grad(out, (q, k, v), grad_out)
    # reference: each sequence separately, key/value heads repeated for their query groups, fp32
    qr, kr, vr = (t.detach().float().requires_grad_() for t in (q, k, v))
    outs = []
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
        qs = qr[a:b].transpose(0, 1)                                     # [H, T, D]
        ks = kr[a:b].repeat_interleave(groups, dim=1).transpose(0, 1)
        vs = vr[a:b].repeat_interleave(groups, dim=1).transpose(0, 1)
        outs.append(torch.nn.functional.scaled_dot_product_attention(qs, ks, vs, is_causal=True).transpose(0, 1))
    ref = torch.cat(outs)
    rq, rk, rv = torch.autograd.grad(ref, (qr, kr, vr), grad_out.float())
    errs = [rel_err(x, y) for x, y in ((out, ref), (gq, rq), (gk, rk), (gv, rv))]
    return report(f"transformer_engine DotProductAttention (thd, causal, GQA {ATTN_HEADS}/{ATTN_KV_HEADS}, "
                  f"head {ATTN_HEAD_DIM}) vs SDPA",
                  max(errs) <= ATTN_MAX_REL_ERR,
                  f"rel. errors out/dq/dk/dv = {[f'{e:.1e}' for e in errs]}")


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def cu_seqlens():
    bounds = [0]
    for n in SEQ_LENS:
        bounds.append(bounds[-1] + n)
    return torch.tensor(bounds, device="cuda", dtype=torch.long)


def gated_delta_rule_reference(q, k, v, g, beta):
    """The gated delta rule, one token at a time, in fp32 plain PyTorch (one sequence).

    Shapes: q, k [T, H, K]; v [T, H, V]; g (log decay) and beta [T, H]. Same definition as fla's
    test reference (fla tests/ops/test_gated_delta.py, recurrent_gated_delta_rule_ref):
      q, k <- l2-normalised over K (use_qk_l2norm_in_kernel=True); q <- q * K**-0.5 (fla's default scale)
      S <- S * exp(g_t)                       state S [H, K, V], decayed
      u <- beta_t * (v_t - S^T k_t)           delta-rule correction
      S <- S + k_t u^T;   o_t = S^T q_t
    """
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = torch.nn.functional.normalize(q, dim=-1) * q.shape[-1] ** -0.5
    k = torch.nn.functional.normalize(k, dim=-1)
    T, H, K = k.shape
    S = torch.zeros(H, K, v.shape[-1], device=q.device, dtype=torch.float32)
    out = []
    for t in range(T):
        S = S * g[t].exp()[:, None, None]
        u = beta[t][:, None] * (v[t] - torch.einsum("hk,hkv->hv", k[t], S))
        S = S + k[t][:, :, None] * u[:, None, :]
        out.append(torch.einsum("hk,hkv->hv", q[t], S))
    return torch.stack(out)


def check_fla_gated_delta_rule():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    T, H, K, dev, dt = sum(SEQ_LENS), HEADS, HEAD_DIM, "cuda", torch.bfloat16

    def inputs():
        # the dtypes slime's layer passes: q/k/v/beta in bf16, the log-decay g in fp32
        torch.manual_seed(SEED)
        q = torch.randn(1, T, H, K, device=dev, dtype=dt)
        k = torch.randn(1, T, H, K, device=dev, dtype=dt)
        v = torch.randn(1, T, H, K, device=dev, dtype=dt)
        g = torch.nn.functional.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
        beta = torch.randn(1, T, H, device=dev, dtype=dt).sigmoid()
        return [t.requires_grad_(True) for t in (q, k, v, g, beta)]

    a, b = inputs(), inputs()   # identical values; separate leaves so each side has its own .grad
    o_chunk, _ = chunk_gated_delta_rule(*a[:3], g=a[3], beta=a[4], initial_state=None, output_final_state=False,
                                        use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens())
    refs, start = [], 0
    for n in SEQ_LENS:   # packed sequences are independent: the state starts at zero in each
        refs.append(gated_delta_rule_reference(*(x[0, start:start + n] for x in b)))
        start += n
    o_ref = torch.cat(refs).unsqueeze(0)
    o_chunk.float().sum().backward()
    o_ref.sum().backward()
    errs = [rel_err(o_chunk, o_ref)] + [rel_err(x.grad, y.grad) for x, y in zip(a, b)]
    return report("fla chunk_gated_delta_rule vs PyTorch reference (varlen, qk l2norm)", max(errs) < FLA_MAX_REL_ERR,
                  f"rel. errors out/dq/dk/dv/dg/dbeta = {[f'{e:.1e}' for e in errs]}")


def check_fla_short_conv():
    from fla.modules import ShortConvolution
    torch.manual_seed(SEED)
    conv = ShortConvolution(hidden_size=CONV_CHANNELS, kernel_size=CONV_KERNEL, bias=False).cuda().to(torch.bfloat16)
    x = torch.randn(1, sum(SEQ_LENS), CONV_CHANNELS, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y, _ = conv(x=x, cu_seqlens=cu_seqlens())
    refs, start = [], 0
    for n in SEQ_LENS:   # each sequence on its own: causal = pad K-1 on the left, keep the first n
        seg = x[0, start:start + n].t().unsqueeze(0).float()
        out = torch.nn.functional.conv1d(seg, conv.weight.float(), padding=CONV_KERNEL - 1, groups=CONV_CHANNELS)
        refs.append(torch.nn.functional.silu(out[..., :n])[0].t())
        start += n
    ref = torch.cat(refs).unsqueeze(0)
    y.float().sum().backward()
    err = rel_err(y, ref)
    return report("fla ShortConvolution vs torch conv1d (varlen)", err < FLA_MAX_REL_ERR and x.grad is not None,
                  f"rel. error {err:.1e}")


def check_fla_gated_norm():
    from fla.modules import FusedRMSNormGated
    torch.manual_seed(SEED)
    norm = FusedRMSNormGated(HEAD_DIM, eps=NORM_EPS, activation="silu", device="cuda", dtype=torch.bfloat16)
    x = torch.randn(sum(SEQ_LENS) * HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    z = torch.randn_like(x, requires_grad=True)
    y = norm(x, z)
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + NORM_EPS) * norm.weight.float() \
        * torch.nn.functional.silu(z.float())
    y.float().sum().backward()
    err = rel_err(y, ref)
    return report("fla FusedRMSNormGated vs formula", err < FLA_MAX_REL_ERR and z.grad is not None,
                  f"rel. error {err:.1e}")


def check_cudnn():
    """One cuDNN in the process, and the one torch was built for (run after the TE checks)."""
    import os
    try:
        with open("/proc/self/maps") as f:
            libs = sorted({ln.split()[-1] for ln in f if "libcudnn" in ln and "/" in ln})
    except OSError as e:
        return report("cudnn", False, f"/proc/self/maps unreadable: {e}")
    folders = sorted({os.path.dirname(p) for p in libs})
    try:
        torch_ok, torch_msg = True, f"torch sees {torch.backends.cudnn.version()}"
    except RuntimeError as e:   # torch raises on a version mismatch
        torch_ok, torch_msg = False, str(e).split(". ")[0]
    return report("cudnn", len(folders) == 1 and torch_ok,
                  f"{torch_msg}; {len(libs)} libcudnn files from {folders} "
                  f"(CUDNN_HOME={os.environ.get('CUDNN_HOME')})")


def check_tms():
    import torch_memory_saver  # noqa: F401
    return report("torch_memory_saver import", True)


def main():
    results = []
    for check in (check_imports, check_cuda, check_te, check_te_attention, check_fla_gated_delta_rule, check_fla_short_conv,
                  check_fla_gated_norm, check_tms, check_cudnn):
        try:
            results.append(check())
        except Exception as e:   # report every failure, keep checking the rest
            results.append(report(check.__name__, False, f"{type(e).__name__}: {e}"))
    print(f"[check_slime_gpu] {'ALL PASS' if all(results) else 'FAILED'} ({sum(results)}/{len(results)})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
