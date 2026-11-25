#!/bin/bash
# scripts/deltaai/smoke_train.sh: a 2-step REAL DAPO training run, inside an allocation, before the long job.
#
# Why: the long job (setup-train, 10.5 h) waits about a day in the queue, and job 3622008 lost
# that slot to a crash in minute 2. Everything training does that no earlier check covers
# (Ray + SGLang on its own GPUs + Megatron on its own GPUs, dynamic sampling filling the batch,
# the first DAPO update at full batch and length, memory, the weight push over NCCL from the
# training GPUs into SGLang on the other GPUs, saving a checkpoint, converting it, and serving it
# for the evaluation) runs here once, in the short setup job.
# Called by: slime_setup.sbatch (step 5) and setup_and_train.sbatch (when the marker is missing),
# as `bash scripts/deltaai/smoke_train.sh` inside the job. Not submitted on its own.
# Steps:
#   1. 16-question slice of the validation file ([smoke] eval_questions), flags from
#      src/train_args.py --smoke: the real flags with only the step count (2) and the
#      validation/save interval (2) changed ([smoke] in configs/train_reinit.toml);
#   2. slime train.py with those flags through train_inside.sh, exactly as the long job runs it
#      (private Ray, our reward and hooks); no time budget;
#   3. src/check_smoke.py: both steps logged with the full batch (with dynamic sampling: at least
#      16 complete groups of 8 with a learning signal), step 2 generated with the updated weights
#      (weight push), validation before and after, rewards = correctness + overlong penalty; it
#      prints each step's time and answer lengths;
#   4. src/pick_checkpoint.py --smoke picks the saved step-2 checkpoint, which is converted to
#      Hugging Face format with the same command as eval_trained.sbatch and checked with
#      src/check_trained.py (only language-model weights changed, fingerprint written);
#   5. GPUs freed, then the converted checkpoint is served with the control image and
#      src/generate.py --checkpoint against the control config of [run] control_config (exported
#      as CONTROL_CONFIG for common.sh / serve_and_generate.sh), as the real evaluation does: [smoke] serve_questions
#      question x 1 sample x 3 modes, capped at [smoke] serve_max_new_tokens; then check_run.
#      Any version/backend WARNING from generate.py fails the smoke run, because the real
#      evaluation would refuse to start on it;
#   6. delete this run's own checkpoint folder (~30 GB with optimizer state) and write the marker
#      <Megatron copy>.smoke_pass (job, git, date).
# Outputs kept: runs/smoke_train_<job>/ (flags, rollout log, generations of the serving test).
# Any failure stops with exit 1 and no marker.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/deltaai/env.sh
source scripts/deltaai/slime_common.sh
JOB="${SLURM_JOB_ID:-local}"
CFG=configs/train_reinit.toml
echo "[smoke-train] $(date -u +%FT%TZ) host $(hostname) | git $(git rev-parse --short HEAD) | job $JOB"
slime_paths
if [ ! -f "$REF_DIR.setup_pass" ]; then echo "[smoke-train] $REF_DIR has no setup_pass marker: run the setup first"; exit 1; fi

SMOKE_ID="smoke_train_$JOB"
SMOKE_DIR="runs/$SMOKE_ID"
SAVE_DIR="$SLIME_CKPT_DIR/_$SMOKE_ID"
if [ -e "$SAVE_DIR" ]; then echo "[smoke-train] $SAVE_DIR already exists; not overwriting"; exit 1; fi
mkdir -p "$SMOKE_DIR/rollouts"
T0=$SECONDS

# ---- 1
N_VAL="$(cfg_value "$CFG" smoke eval_questions)"
DATA_DIR="$(cfg_value "$CFG" data output_dir)"
head -n "$N_VAL" "$DATA_DIR/val.jsonl" > "$SMOKE_DIR/val_smoke.jsonl"
[ "$(wc -l < "$SMOKE_DIR/val_smoke.jsonl")" -eq "$N_VAL" ] || { echo "[smoke-train] could not take $N_VAL validation questions"; exit 1; }
FLAGS_FILE="$SMOKE_DIR/slime_args.txt"
WEIGHT_SYNC_DIR="$SLIME_CKPT_DIR/_weight_sync_$JOB"   # used only if [hardware] update_weight_transport = "disk"
S python3 -m src.train_args --hf-checkpoint "$HF_SNAPSHOT" --ref-load "$REF_DIR" --save "$SAVE_DIR" \
  --smoke --eval-data "$REPO_DIR/$SMOKE_DIR/val_smoke.jsonl" --update-weight-disk-dir "$WEIGHT_SYNC_DIR" \
  --out "$FLAGS_FILE" > /dev/null
echo "[smoke-train] flags $FLAGS_FILE (real flags except steps/intervals), checkpoints $SAVE_DIR"

# ---- 2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
RAY_TMP="/tmp/ray_smk_$JOB"   # short path: Ray's socket paths must stay under ~100 characters
# Ray gets every GPU of the node: slime places Megatron on [hardware] actor_num_gpus of them and the
# SGLang engines on the other rollout_num_gpus (train_args checked they add up to gpus_per_node).
NUM_GPUS="$(cfg_value "$CFG" hardware gpus_per_node)"
# Worst-case memory: the first answer of each of the first N groups runs to the full response
# length ([smoke] force_max_len_groups; src/slime_hooks.py). Set ONLY here, never for real training.
FORCE_N="$(cfg_value "$CFG" smoke force_max_len_groups)"
S env GAR_TRAIN_CONFIG="$REPO_DIR/$CFG" GAR_ROLLOUT_LOG_DIR="$REPO_DIR/$SMOKE_DIR/rollouts" REPO_DIR="$REPO_DIR" \
  GAR_SMOKE_FORCE_MAX_LEN_GROUPS="$FORCE_N" \
  bash scripts/deltaai/train_inside.sh "$FLAGS_FILE" "$RAY_TMP" "$(free_port)" "$NUM_GPUS" "${SLURM_CPUS_PER_TASK:-64}" 0
echo "[smoke-train] $(date -u +%FT%TZ) train.py finished after $((SECONDS - T0)) s"

# ---- 3
S python3 -m src.check_smoke "$SMOKE_DIR/rollouts"

# ---- 4
eval "$(S python3 -m src.pick_checkpoint --smoke --rollouts "$SMOKE_DIR/rollouts" --ckpt-dir "$SAVE_DIR" --shell)"
# Measured size of one save, for the disk budget ([checkpoint] est_save_gb is an estimate)
echo "[smoke-train] one save (step $STEP, weights + optimizer state): $(du -sh "$ITER_DIR" | cut -f1)" \
     "| [checkpoint] est_save_gb = $(cfg_value "$CFG" checkpoint est_save_gb) GB"
HF_OUT="$SAVE_DIR/hf_step$STEP"
S env PYTHONPATH=/opt/Megatron-LM:/opt/slime \
  python3 scripts/deltaai/convert_to_hf.py --input-dir "$ITER_DIR" --output-dir "$HF_OUT" \
  --origin-hf-dir "$HF_SNAPSHOT" --add-missing-from-origin-hf --vocab-size "$VOCAB_SIZE"
S python3 -m src.check_trained "$HF_SNAPSHOT" "$HF_OUT" \
  --source "{\"run\": \"$SMOKE_ID\", \"step\": $STEP, \"iter_dir\": \"$ITER_DIR\", \"job\": \"$JOB\"}"
echo "[smoke-train] step-$STEP checkpoint converted and checked"

# ---- 5 (wait until training's processes have left every GPU, then serve on the freed node)
LEFT=""
for _ in $(seq 1 36); do
  LEFT="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || true)"
  [ -z "$LEFT" ] && break
  sleep 5
done
if [ -n "$LEFT" ]; then
  echo "[smoke-train] stopping processes still on the GPU after 3 min: $(echo $LEFT)"
  for pid in $LEFT; do kill -9 "$pid" 2>/dev/null || true; done
  sleep 10
fi
SERVE_ID="smoke_eval_$JOB"
GEN_LOG="$SMOKE_DIR/serve_test.log"
export EVAL_MODEL_DIR="$HF_OUT"
export CONTROL_CONFIG="$(cfg_value "$CFG" run control_config)"   # the 4B control's config, not the 2B default
set +e
bash scripts/deltaai/serve_and_generate.sh --run-id "$SERVE_ID" --checkpoint "$HF_OUT" \
  --limit-questions "$(cfg_value "$CFG" smoke serve_questions)" --samples 1 \
  --max-new-tokens "$(cfg_value "$CFG" smoke serve_max_new_tokens)" 2>&1 | tee "$GEN_LOG"
GEN=${PIPESTATUS[0]}
set -e
if [ "$GEN" -ne 0 ]; then echo "[smoke-train] FAIL: serving the trained checkpoint exited with $GEN"; exit 1; fi
if grep -q "WARNING: .*!=" "$GEN_LOG"; then
  echo "[smoke-train] FAIL: the real evaluation would refuse this server (see the WARNING above)"; exit 1
fi
source scripts/deltaai/common.sh
HF_HUB_OFFLINE=1 run python3 -m src.check_run "runs/$SERVE_ID"
echo "[smoke-train] serving test passed (runs/$SERVE_ID)"

# ---- 6
rm -rf "$SAVE_DIR" "$WEIGHT_SYNC_DIR"   # this job's own smoke checkpoints (and disk weight-push folder, if any)
echo "job $JOB | git $(git rev-parse --short HEAD) | $(date -u +%FT%TZ) | $((SECONDS - T0)) s" > "$REF_DIR.smoke_pass"
echo "[smoke-train] $(date -u +%FT%TZ) PASS in $((SECONDS - T0)) s (marker $REF_DIR.smoke_pass)"
