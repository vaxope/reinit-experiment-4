#!/bin/bash
# scripts/deltaai/slime_env.sh: runs INSIDE the slime image; sets the cuDNN library, then runs the command.
#
# Why (diagnosis job 3686686, 2026-11-27): the image holds two cuDNN builds, PyTorch's own
# (site-packages/nvidia/cudnn, 9.20, what torch 2.13 is compiled against) and an older system copy
# (/usr/lib/aarch64-linux-gnu/libcudnn*.so.9.14.0). TransformerEngine 2.18 loads cuDNN from the
# system FIRST ($CUDNN_HOME, $CUDNN_PATH, $CUDA_HOME, $CUDA_PATH, /usr/local/cuda, then
# LD_LIBRARY_PATH; transformer_engine/common/__init__.py), so a process ran with a MIX of 9.20 and
# 9.14 files. Result: TE's attention gave wrong gradients at Qwen3.5's head size 256 on packed
# batches (rel. error up to 0.71; setup job 3680134), and torch refused its own cuDNN check.
# With CUDNN_HOME and LD_LIBRARY_PATH pointing at PyTorch's copy, every process loads 9.20 only
# and all 7 attention cases match PyTorch (rel. errors ~3e-3), in the same job.
# What  : CUDNN_HOME = PyTorch's nvidia/cudnn folder (found with the image's python, not written
#         in), its lib/ first on LD_LIBRARY_PATH, then exec the command. Child processes (Ray
#         workers, Megatron, SGLang engines) inherit both.
# Used by: S() in slime_common.sh, i.e. every command run in the slime image. Not for the control
#         (SGLang) image, which has no TransformerEngine.
# Check : scripts/deltaai/check_slime_gpu.py "cudnn" fails if a process loads cuDNN from more than one
#         folder or torch's cuDNN check fails.
set -euo pipefail
CUDNN_DIR="$(python3 -c 'import importlib.util as u; s = u.find_spec("nvidia.cudnn"); print(list(s.submodule_search_locations)[0] if s else "")' 2>/dev/null || true)"
if [ -z "$CUDNN_DIR" ] || [ ! -d "$CUDNN_DIR/lib" ]; then
  echo "[slime_env] PyTorch's cuDNN (python package nvidia.cudnn) not found in this image" >&2
  exit 1
fi
export CUDNN_HOME="$CUDNN_DIR"
export LD_LIBRARY_PATH="$CUDNN_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$@"
