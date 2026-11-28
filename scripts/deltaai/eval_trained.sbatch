#!/bin/bash
# scripts/deltaai/eval_trained.sbatch: evaluate a trained checkpoint exactly like the control run, 1 GH200 node.
#
# Submit through submit.sh:   scripts/deltaai/submit.sh eval-trained [STEP|best]   (default best)
# Also run INSIDE the training job (train_reinit.sbatch step 3, as `bash ... best`), on the GPUs
# training just freed; the #SBATCH lines below are then plain comments.
# Steps:
#   0. refuse uncommitted changes in src/, configs/, scripts/ (the run must be traceable);
#   1. pick the checkpoint (src/pick_checkpoint.py): the saved step with the highest validation
#      v3 accuracy on the held-out SuperGPQA slice (ties -> later), or the STEP given. GPQA is
#      never used for this choice;
#   2. convert it to Hugging Face format with scripts/deltaai/convert_to_hf.py (the converter the
#      setup's round trip and the smoke run use), copying the untouched vision/MTP weights and the
#      tokenizer/config files from the pinned base snapshot; then src/check_trained.py: same tensors
#      and files as the base, only language-model weights changed (at least one), and a
#      fingerprint (gar_checkpoint.json). Output: $SLIME_CKPT_DIR/<run>_hf/step<N> (~9 GB).
#      Skipped if that folder already passed (a resubmission resumes); stops if it exists unchecked;
#   3. the control evaluation, unchanged except for the weights: serve_and_generate.sh with the
#      SGLang image and every setting of the control config ([run] control_config of the training
#      config, exported as CONTROL_CONFIG): GPQA Diamond, 198 x 4 samples x 3 modes, the control
#      sampling, 81,920 cap, [server] dp_size model copies. Run id eval_<run>_step<N> (generate.py
#      refuses any pin mismatch for eval_* ids); resubmitting continues where a time limit stopped it;
#   4. score (src/score.py: rule v3 + lenient) and the mechanical checks (src/check_run.py).
#      Comparison with the control run: python3 -m src.compare_runs (locally or here).
# Port of Experiment-3 @ 97e4706. Changes: the control config comes from the training config
# (the 4B control, not the default file of the time), 4 GPUs (one model copy per GPU).
#SBATCH --job-name=gar-eval-trained
#SBATCH --nodes=1
#SBATCH --ntasks=1
# GPUs = the control config's [server] dp_size (tests/test_slime_container.py checks it). RAM: the
# converter loads the checkpoint's weights (~9 GB, optimizer state skipped), and 4 SGLang copies
# load the model; CPUs/RAM come with the 4 GPUs (billing is per GPU).
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --output=runs/_slurm/%x_%j.out
#SBATCH --constraint=projects
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
WHICH="${1:-best}"
JOB="${SLURM_JOB_ID:-local}"
CFG=configs/train_reinit.toml
echo "[eval] $(date -u +%FT%TZ) host $(hostname) | git $(git rev-parse --short HEAD) | job $JOB | checkpoint: $WHICH"

# ---- 0
if [ -n "$(git status --porcelain -- src configs scripts)" ]; then
  echo "[eval] uncommitted changes in src/, configs/ or scripts/: commit first so the run is traceable"; exit 1
fi
source scripts/deltaai/env.sh
source scripts/deltaai/slime_common.sh
check_slime_sif
slime_paths
SAVE_DIR="$SLIME_CKPT_DIR/$RUN_NAME"
CONTROL_CONFIG="$(cfg_value "$CFG" run control_config)"
export CONTROL_CONFIG   # common.sh (load_cfg) and serve_and_generate.sh read it

# ---- 1
PICK=(python3 -m src.pick_checkpoint --rollouts "runs/$RUN_NAME/rollouts" --ckpt-dir "$SAVE_DIR")
if [ "$WHICH" != "best" ]; then PICK+=(--step "$WHICH"); fi
S "${PICK[@]}"
eval "$(S "${PICK[@]}" --shell)"   # sets STEP and ITER_DIR
EVAL_DIR="$SLIME_CKPT_DIR/${RUN_NAME}_hf/step$STEP"
RUN_ID="eval_${RUN_NAME}_step$STEP"
echo "[eval] step $STEP: $ITER_DIR -> $EVAL_DIR, run id $RUN_ID, control config $CONTROL_CONFIG"

# ---- 2
if [ -f "$EVAL_DIR/gar_checkpoint.json" ]; then
  echo "[eval] $EVAL_DIR already converted and checked: skipping conversion"
elif [ -e "$EVAL_DIR" ]; then
  echo "[eval] $EVAL_DIR exists without gar_checkpoint.json; delete it yourself and resubmit"; exit 1
else
  PART="$EVAL_DIR.partial_$JOB"   # made by this job only; renamed once the check passes
  mkdir -p "$(dirname "$EVAL_DIR")"
  T0=$SECONDS
  S env PYTHONPATH=/opt/Megatron-LM:/opt/slime \
    python3 scripts/deltaai/convert_to_hf.py --input-dir "$ITER_DIR" --output-dir "$PART" \
    --origin-hf-dir "$HF_SNAPSHOT" --add-missing-from-origin-hf --vocab-size "$VOCAB_SIZE"
  SOURCE="{\"run\": \"$RUN_NAME\", \"step\": $STEP, \"iter_dir\": \"$ITER_DIR\", \"job\": \"$JOB\", \"git\": \"$(git rev-parse HEAD)\"}"
  set +e
  S python3 -m src.check_trained "$HF_SNAPSHOT" "$PART" --source "$SOURCE"
  CHECK=$?
  set -e
  if [ "$CHECK" -ne 0 ]; then
    rm -rf "$PART"   # this job's own temporary copy
    echo "[eval] FAIL: the converted checkpoint did not pass src/check_trained.py"; exit 1
  fi
  mv "$PART" "$EVAL_DIR"
  echo "[eval] converted and checked in $((SECONDS - T0)) s"
fi

# ---- 3
export EVAL_MODEL_DIR="$EVAL_DIR"   # serve_and_generate.sh serves this folder; common.sh binds it
set +e
bash scripts/deltaai/serve_and_generate.sh --run-id "$RUN_ID" --checkpoint "$EVAL_DIR"
GEN=$?
set -e
echo "[eval] generation exited with status $GEN"

# ---- 4 (also after errors: errors are records and are counted; a time limit never reaches here)
source scripts/deltaai/common.sh
HF_HUB_OFFLINE=1 run python3 -m src.score "runs/$RUN_ID/generations.jsonl" --config "$CONTROL_CONFIG" --overwrite
HF_HUB_OFFLINE=1 run python3 -m src.check_run "runs/$RUN_ID" --config "$CONTROL_CONFIG" \
  || echo "[eval] check_run reported problems (see above)"
echo "[eval] $(date -u +%FT%TZ) done: runs/$RUN_ID/summary.md"
exit "$GEN"
