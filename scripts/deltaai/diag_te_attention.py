"""Diagnose the TransformerEngine attention failure of setup job 3680134, and re-test the fla kernels at real shapes (one GPU, minutes).

Why     : check 2b of check_slime_gpu.py failed in job 3680134: TE DotProductAttention's FORWARD
          matches PyTorch (rel. error 1.9e-3) but its GRADIENTS do not (dq/dk/dv rel. error 0.49 at
          16/4 heads, NaN at 8/2, head size 256, packed "thd" sequences). Megatron runs Qwen3.5's
          full-attention layers (every 4th layer) through exactly this call, so training would learn
          from wrong gradients. This script narrows down which part is broken and which setting
          avoids it, without changing anything:
            - TE backend: default selection / cuDNN deterministic algorithm / TE's unfused (plain
              PyTorch) path. Each backend runs in its own subprocess, because TE reads the NVTE_*
              variables once per process.
            - shape: head size 256 vs 128 (a pass at 128 shows the check itself is right), GQA vs
              not, ragged vs 128-aligned lengths, packed "thd" vs padded-free "bshd".
          TE's own log line ("Running with ... backend") is printed, so the backend is known.
          cuDNN library (added after diagnosis job 3686595): that job stopped with "cuDNN version
          incompatibility: PyTorch was compiled against (9, 20, 0) but found runtime version
          (9, 14, 0)". TE 2.18 loads cuDNN from the system FIRST (transformer_engine/common/
          __init__.py _load_cuda_library: $CUDNN_HOME, $CUDNN_PATH, $CUDA_HOME, $CUDA_PATH,
          /usr/local/cuda, then LD_LIBRARY_PATH; PyTorch's pip copy only if none is found), with
          RTLD_GLOBAL, so the whole process then runs on that older cuDNN, under kernels built for
          another version. The "pip_cudnn" settings point TE at PyTorch's own copy (CUDNN_HOME and
          LD_LIBRARY_PATH = site-packages/nvidia/cudnn). Every child reports which libcudnn files it
          really loaded (/proc/self/maps), so a mix of versions is visible. Nothing crashes on the
          version error any more (job 3686595 stopped in versions()).
          Also (default backend only): check 3's flash-linear-attention kernels at the shapes
          Qwen3.5-4B really runs (32 value heads after slime repeats the 16 key heads, head size
          128, 8,192 conv channels; config.json @ 851bf6e8) and on a longer packed batch, because
          check_slime_gpu.py tests them only at a small shape (2 heads, size 64) and the
          attention failure above depends on the shape. Uses check_slime_gpu.py's own functions.
Inputs  : none (random tensors, fixed seed; the same reference as check_slime_gpu.py).
Output  : one line per (setting, case) with PASS/FAIL and the rel. errors of out/dq/dk/dv; per
          setting the cuDNN files loaded and PyTorch's cuDNN check; package versions; cuDNN files found. Changes nothing; exit code always 0 (it is a report).
Run     : inside the slime image, on one GPU (scripts/deltaai/submit.sh diag-attn):
            apptainer exec --nv <slime.sif> python3 scripts/deltaai/diag_te_attention.py
"""
import json
import os
import subprocess
import sys

# Same tolerance and seed as check_slime_gpu.py, so a PASS here means check 2b would pass.
ATTN_MAX_REL_ERR = 5e-2
SEED = 0
# (format, query heads, key/value heads, head size, sequence lengths)
#   the job's two shapes; then one change at a time from the 8/2/256 case.
CASES = [
    ("thd", 16, 4, 256, (100, 156)),   # whole 4B model, as in job 3680134
    ("thd", 8, 2, 256, (100, 156)),    # one of 2 tensor-parallel GPUs, as in job 3680134
    ("thd", 8, 2, 256, (128, 128)),    # lengths a multiple of 128
    ("thd", 8, 8, 256, (100, 156)),    # no GQA
    ("thd", 8, 2, 128, (100, 156)),    # head size 128: if this passes, the check itself is right
    ("bshd", 8, 2, 256, (128, 128)),   # not packed: batch of 2 equal-length sequences
    ("thd", 8, 2, 256, (1000, 3096)),  # longer packed batch (more kernel tiles per sequence)
]
# check 3 (fla) at the real model's shapes, 4B config.json @ 851bf6e8: linear_num_value_heads 32
# (q and k are repeated from 16 key heads to 32 before the kernel, slime qwen3_5.py), key/value head
# size 128, conv channels 2 x 16 x 128 + 32 x 128 = 8,192. Small and longer packed lengths.
FLA_HEADS, FLA_HEAD_DIM, FLA_CONV_CHANNELS = 32, 128, 8192
FLA_SEQ_LENS = [(100, 156), (700, 1348)]
# TE search paths for cuDNN, in its order (transformer_engine/common/__init__.py, TE 2.18).
TE_CUDNN_VARS = ("CUDNN_HOME", "CUDNN_PATH", "CUDA_HOME", "CUDA_PATH")
TE_CUDNN_DEFAULT_DIR = "/usr/local/cuda"
DETERMINISTIC = {"NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0"}
UNFUSED = {"NVTE_FUSED_ATTN": "0", "NVTE_FLASH_ATTN": "0"}


def pip_cudnn_dir():
    """site-packages/nvidia/cudnn (PyTorch's own cuDNN), found without importing torch."""
    import importlib.util
    try:
        spec = importlib.util.find_spec("nvidia.cudnn")
    except (ImportError, ValueError):
        return None
    return list(spec.submodule_search_locations)[0] if spec and spec.submodule_search_locations else None


def settings():
    """Environment changes to test, by name. TE reads NVTE_* and its cuDNN once per process."""
    out = {"as_job": {}}
    pip = pip_cudnn_dir()
    if pip:
        lib = os.path.join(pip, "lib")
        pip_env = {"CUDNN_HOME": pip, "LD_LIBRARY_PATH": lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")}
        out["pip_cudnn"] = pip_env
        out["pip_cudnn_deterministic"] = {**pip_env, **DETERMINISTIC}
    out["unfused"] = UNFUSED
    return out


def cudnn_files():
    """Every libcudnn.so* TE could pick, per place it looks (recursive where TE is recursive)."""
    import glob
    places = [(v, os.environ.get(v)) for v in TE_CUDNN_VARS] + [("default", TE_CUDNN_DEFAULT_DIR)]
    found = {}
    for name, d in places:
        if d:
            found[f"{name}={d}"] = sorted(p for p in glob.glob(f"{d}/**/libcudnn.so*", recursive=True) if "stub" not in p)
    for d in filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)):
        found[f"LD_LIBRARY_PATH {d}"] = sorted(glob.glob(f"{d}/libcudnn.so*"))
    pip = pip_cudnn_dir()
    if pip:
        found[f"pip {pip}"] = sorted(glob.glob(f"{pip}/**/libcudnn.so*", recursive=True))
    return {k: v for k, v in found.items() if v}


def loaded_cudnn():
    """libcudnn* files mapped into THIS process (what really runs)."""
    try:
        with open("/proc/self/maps") as f:
            return sorted({ln.split()[-1] for ln in f if "libcudnn" in ln and "/" in ln})
    except OSError as e:
        return [f"unreadable: {e}"]


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def run_case(fmt, heads, kv_heads, dim, lens):
    import torch
    import transformer_engine.pytorch as te
    torch.manual_seed(SEED)
    total, groups = sum(lens), heads // kv_heads
    mk = lambda h: torch.randn(total, h, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)  # noqa: E731
    q, k, v = mk(heads), mk(kv_heads), mk(kv_heads)
    attn = te.DotProductAttention(num_attention_heads=heads, kv_channels=dim, num_gqa_groups=kv_heads,
                                  attention_dropout=0.0, qkv_format=fmt,
                                  attn_mask_type="padding_causal" if fmt == "thd" else "causal")
    if fmt == "thd":
        cu = torch.tensor([0, lens[0], total], device="cuda", dtype=torch.int32)
        out = attn(q, k, v, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(lens), max_seqlen_kv=max(lens))
    else:   # bshd: [batch, seq, heads, dim]; needs equal lengths
        assert lens[0] == lens[1]
        b = lambda t: t.view(2, lens[0], t.shape[1], dim)  # noqa: E731
        out = attn(b(q), b(k), b(v))
    out = out.reshape(total, heads, dim)
    grad_out = torch.randn_like(out)
    gq, gk, gv = torch.autograd.grad(out, (q, k, v), grad_out)
    qr, kr, vr = (t.detach().float().requires_grad_() for t in (q, k, v))
    refs, a = [], 0
    for n in lens:
        sl = slice(a, a + n)
        qs = qr[sl].transpose(0, 1)
        ks = kr[sl].repeat_interleave(groups, dim=1).transpose(0, 1)
        vs = vr[sl].repeat_interleave(groups, dim=1).transpose(0, 1)
        refs.append(torch.nn.functional.scaled_dot_product_attention(qs, ks, vs, is_causal=True).transpose(0, 1))
        a += n
    ref = torch.cat(refs)
    rq, rk, rv = torch.autograd.grad(ref, (qr, kr, vr), grad_out.float())
    errs = [rel_err(x, y) for x, y in ((out, ref), (gq, rq), (gk, rk), (gv, rv))]
    nan = any(e != e for e in errs)
    return {"ok": (not nan) and max(errs) <= ATTN_MAX_REL_ERR, "errs": [f"{e:.1e}" for e in errs]}


def child(backend):
    """Runs every case under one setting's environment; prints one JSON line per case, then INFO."""
    for case in CASES:
        try:
            r = run_case(*case)
        except Exception as e:   # an unsupported combination is a result, not a crash
            r = {"ok": False, "errs": None, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print("RESULT " + json.dumps({"backend": backend, "case": case, **r}), flush=True)
    info = {"cudnn_loaded": loaded_cudnn()}
    try:
        import torch
        info["cudnn_torch"] = torch.backends.cudnn.version()   # raises on a version mismatch
    except Exception as e:
        info["cudnn_torch"] = f"{type(e).__name__}: {str(e)[:200]}"
    print("INFO " + json.dumps(info), flush=True)


def child_fla():
    """check_slime_gpu.py's fla checks with its shape constants set to the real model's."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import check_slime_gpu as g
    for lens in FLA_SEQ_LENS:
        g.SEQ_LENS, g.HEADS, g.HEAD_DIM, g.CONV_CHANNELS = lens, FLA_HEADS, FLA_HEAD_DIM, FLA_CONV_CHANNELS
        print(f"[diag_te_attention] fla at heads {FLA_HEADS}, head {FLA_HEAD_DIM}, conv {FLA_CONV_CHANNELS}, "
              f"lens {lens}:", flush=True)
        for check in (g.check_fla_gated_delta_rule, g.check_fla_short_conv, g.check_fla_gated_norm):
            try:
                check()   # prints its own PASS/FAIL line
            except Exception as e:
                g.report(check.__name__, False, f"{type(e).__name__}: {str(e)[:300]}")


def versions():
    """Package versions WITHOUT importing torch or TE (the parent must not load any cuDNN)."""
    from importlib import metadata
    v = {}
    for dist in ("torch", "transformer_engine", "transformer_engine_cu12", "nvidia-cudnn-cu12",
                 "nvidia-cudnn-cu13", "flash_attn", "flash-linear-attention"):
        try:
            v[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    v["env"] = {k: os.environ.get(k) for k in TE_CUDNN_VARS + ("LD_LIBRARY_PATH",)}
    return v


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
        return
    if len(sys.argv) == 2 and sys.argv[1] == "--child-fla":
        child_fla()
        return
    print("[diag_te_attention] versions " + json.dumps(versions()), flush=True)
    for place, files in cudnn_files().items():
        print(f"[diag_te_attention] cudnn files {place}: {files}", flush=True)
    for name, extra in settings().items():
        # NVTE_DEBUG=1, level 2 (debug): TE logs which backend it picked for each call.
        env = {**os.environ, **extra, "NVTE_DEBUG": "1", "NVTE_DEBUG_LEVEL": "2"}
        p = subprocess.run([sys.executable, __file__, "--child", name], env=env, capture_output=True, text=True)
        picked = sorted({ln.split("Running with", 1)[1].strip() for ln in (p.stdout + p.stderr).splitlines()
                         if "Running with" in ln})
        print(f"[diag_te_attention] setting {name} (sets {sorted(extra) or 'nothing'}): TE picked {picked or 'unknown'}"
              f"{'' if p.returncode == 0 else f' (child exit {p.returncode})'}", flush=True)
        for ln in p.stdout.splitlines():
            if ln.startswith("INFO "):
                print(f"[diag_te_attention]   {name}: {ln[len('INFO '):]}", flush=True)
            if ln.startswith("RESULT "):
                r = json.loads(ln[len("RESULT "):])
                fmt, h, kv, d, lens = r["case"]
                detail = f"out/dq/dk/dv = {r['errs']}" if r["errs"] else r.get("error", "")
                print(f"[diag_te_attention]   {'PASS' if r['ok'] else 'FAIL'} {fmt} {h}/{kv} head {d} lens {lens}: "
                      f"{detail}", flush=True)
        if p.returncode != 0:
            print("[diag_te_attention]   child stderr (last lines): "
                  + " | ".join(p.stderr.strip().splitlines()[-8:]), flush=True)
    p = subprocess.run([sys.executable, __file__, "--child-fla"], capture_output=True, text=True)
    for ln in p.stdout.splitlines():
        print(ln.replace("[check_slime_gpu]", "[diag_te_attention]  "), flush=True)
    if p.returncode != 0:
        print("[diag_te_attention]   fla child stderr (last lines): " + " | ".join(p.stderr.strip().splitlines()[-8:]))
    print("[diag_te_attention] done (a report; nothing was changed)")


if __name__ == "__main__":
    main()
