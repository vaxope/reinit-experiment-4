#!/bin/bash
# scripts/deltaai/slime_setup.sbatch: one-time setup before training + the smoke test (one GH200 node, 4 GPUs).
#
# Submit through submit.sh:   scripts/deltaai/submit.sh slime-setup
# Steps:
#   0. (seconds, before any GPU work) every config check: src/train_args.py --check: model, template,
#      sampling, answer rule and kernels = the control config ([run] control_config), DAPO settings
#      consistent, GPU layout disaggregated, architecture = config.json. Stops here if, e.g., the
#      4B control config does not exist yet.
#   1. check the slime image is the recorded one (SHA-256), then run its GPU check
#      (scripts/deltaai/check_slime_gpu.py, ~1-2 min; the build job runs it too, but build job
#      3620006's run had a broken reference in the check itself, so it is repeated here);
#   2. convert the pinned Hugging Face weights ([model] in configs/train_reinit.toml) into Megatron format with slime's
#      tools/convert_hf_to_torch_dist.py, using EXACTLY the architecture flags training uses
#      (src/train_args.py --model-args, checked against the model's config.json). This copy is
#      the starting point of training (and the frozen KL reference if KL is on) (--ref-load);
#   3. convert it back to Hugging Face format (scripts/deltaai/convert_to_hf.py = slime's
#      tools/convert_torch_dist_to_hf.py, reading the checkpoint format Megatron ae76c60e writes) and require every
#      language-model weight to match the original (src/check_roundtrip.py); then delete that
#      temporary copy (it was created by this job and is not needed again);
#   4. build the training flags once (src/train_args.py) and run them through slime's REAL parser
#      (scripts/deltaai/check_slime_args.py, seconds): slime drops unknown Megatron flags
#      silently, so a flag it doesn't know, or a value it changes, would otherwise only show up
#      as a wrong setting in a paid training run. Needs the converted copy from step 2 (--ref-load).
#   5. the smoke run (scripts/deltaai/smoke_train.sh; time unmeasured for 4B + DAPO): 2 real DAPO steps
#      with training on 2 GPUs and SGLang on the other 2, checkpoint
#      saved, converted, checked and served with the control image; marker <folder>.smoke_pass.
#      Training refuses to start without that marker (user, 2026-11-19, after job 3622008).
# Rerun after a smoke-run failure (user, 2026-11-19): when <folder>.setup_pass exists, the passed
# Megatron copy is kept and steps 2-3 are skipped; steps 1 and 4 (GPU check, flag check, ~2 min)
# run again because they depend on the node and on the current config, then step 5.
# Output: $SLIME_CKPT_DIR/<model>_<rev>_torch_dist (about 8 GB). Never overwritten: if it exists
# WITHOUT the marker, the job stops (delete it yourself to redo the conversion). As soon as the
# round trip (step 3) passes, it writes the marker <that folder>.setup_pass (job, git, date) =
# "this Megatron copy is verified": training refuses a copy without it, so a half-finished
# conversion is never trained on. Written after step 3, not after step 4 (job 3686748: the flag
# check failed after a good conversion, and the unmarked 7.9 GB copy then blocked the rerun):
# step 4 reruns every time anyway. The flag check's own proof is the smoke run's marker
# (.smoke_pass), which only step 5 writes, after step 4 passed in the same job.
#SBATCH --job-name=gar-slime-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
# GPUs = configs/train_reinit.toml [hardware] gpus_per_node (2 training + 2 generation, separate;
# tests/test_train_args.py checks this line against the config). Slurm reads #SBATCH lines
# before any script runs, so the number is written here, not read from the config.
#SBATCH --gpus-per-node=4
# CPUs and memory: 2 training processes + 2 SGLang engines + Ray. The optimizer lives in CPU
# memory ([hardware] optimizer_cpu_offload): ~2.6B params x 16 bytes (fp32 master + Adam + grad
# copy) ~ 42 GB per training GPU, ~84 GB for both, plus pinned buffers. 240G = that with margin,
# kept well under a GH200 node's total (4 x 120 GB LPDDR) so the request cannot exceed the node.
# Accepted by Slurm on ghx4 (setup jobs 3680134 and 3686748 were scheduled with it; the node
# reported 476 GB host memory). Peak use during training is still unmeasured.
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --output=runs/_slurm/%x_%j.out
#SBATCH --constraint=projects
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/deltaai/env.sh
source scripts/deltaai/slime_common.sh
echo "[setup] $(date -u +%FT%TZ) host $(hostname) | git $(git rev-parse --short HEAD) | job ${SLURM_JOB_ID:-local}"

# ---- 0 (config checks: seconds, no GPU work)
check_slime_sif
slime_paths
S python3 -m src.train_args --hf-checkpoint "$HF_SNAPSHOT" --check

# ---- 1
S python3 scripts/deltaai/check_slime_gpu.py
echo "[setup] model $HF_SNAPSHOT"
echo "[setup] Megatron copy -> $REF_DIR"
REUSE=0
if [ -f "$REF_DIR.setup_pass" ]; then
  REUSE=1
  echo "[setup] $REF_DIR passed the round trip before ($(cat "$REF_DIR.setup_pass")): skipping steps 2-3"
elif [ -e "$REF_DIR" ]; then
  echo "[setup] $REF_DIR already exists without a setup_pass marker; not overwriting (delete it to redo)."; exit 1
fi

if [ "$REUSE" -eq 0 ]; then
# ---- 2
mapfile -t MODEL_ARGS < <(S python3 -m src.train_args --hf-checkpoint "$HF_SNAPSHOT" --model-args)
echo "[setup] architecture flags: ${MODEL_ARGS[*]}"
T0=$SECONDS
S env MASTER_ADDR=127.0.0.1 MASTER_PORT="$(free_port)" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
  PYTHONPATH=/opt/Megatron-LM:/opt/slime \
  python3 /opt/slime/tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_SNAPSHOT" --save "$REF_DIR"
echo "[setup] conversion done in $((SECONDS - T0)) s: $(du -sh "$REF_DIR" | cut -f1)"

# ---- 3
# The checkpoint folder is the one with torch_dist's .metadata file (the converter saves it as
# <REF_DIR>/release). Megatron ae76c60e writes no common.pt any more (setup job 3631056).
mapfile -t META < <(find "$REF_DIR" -name .metadata)
if [ "${#META[@]}" -ne 1 ]; then echo "[setup] expected one checkpoint (.metadata) in $REF_DIR, found ${#META[@]}"; exit 1; fi
CKPT_SUBDIR="$(dirname "${META[0]}")"
echo "[setup] Megatron checkpoint folder: $CKPT_SUBDIR"
RT_DIR="$SLIME_CKPT_DIR/_roundtrip_check_${SLURM_JOB_ID:-local}"
S env PYTHONPATH=/opt/Megatron-LM:/opt/slime python3 scripts/deltaai/convert_to_hf.py --input-dir "$CKPT_SUBDIR" \
  --output-dir "$RT_DIR" --origin-hf-dir "$HF_SNAPSHOT" --vocab-size "$VOCAB_SIZE"
set +e
S python3 -m src.check_roundtrip "$HF_SNAPSHOT" "$RT_DIR"
RT_STATUS=$?
set -e
rm -rf "$RT_DIR"   # temporary copy made by this job
if [ "$RT_STATUS" -ne 0 ]; then
  echo "[setup] FAIL: the round trip changed weights; do not train on $REF_DIR"; exit 1
fi
echo "job ${SLURM_JOB_ID:-local} | git $(git rev-parse --short HEAD) | $(date -u +%FT%TZ)" > "$REF_DIR.setup_pass"
echo "[setup] Megatron copy verified (marker $REF_DIR.setup_pass)"
fi   # steps 2-3

# ---- 4
S python3 -m src.train_args --hf-checkpoint "$HF_SNAPSHOT" --ref-load "$REF_DIR" \
  --save "$SLIME_CKPT_DIR/$RUN_NAME" --update-weight-disk-dir "$SLIME_CKPT_DIR/_weight_sync_$RUN_NAME" \
  --out "runs/$RUN_NAME/slime_args_setup.txt" > /dev/null
echo "[setup] training flags built ($(wc -l < "runs/$RUN_NAME/slime_args_setup.txt") lines, runs/$RUN_NAME/slime_args_setup.txt)"
# CUDA_DEVICE_MAX_CONNECTIONS=1: the value train_inside.sh exports for training. Megatron's
# validate_args (ae76c60e) refuses tensor parallel > 1 without it (setup job 3686748), so the check
# must parse with the same environment training runs with.
S env GAR_TRAIN_CONFIG="$REPO_DIR/configs/train_reinit.toml" PYTHONPATH="/opt/Megatron-LM" CUDA_DEVICE_MAX_CONNECTIONS=1 \
  python3 scripts/deltaai/check_slime_args.py "runs/$RUN_NAME/slime_args_setup.txt"
echo "[setup] $(date -u +%FT%TZ) PASS: steps 0-4 (Megatron copy, marker $REF_DIR.setup_pass)"

# ---- 5
bash scripts/deltaai/smoke_train.sh
