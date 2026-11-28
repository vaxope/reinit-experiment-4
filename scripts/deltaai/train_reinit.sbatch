#!/bin/bash
# scripts/deltaai/train_reinit.sbatch: the long DAPO training job, 1 GH200 node (2 training + 2 SGLang GPUs).
#
# Submit through submit.sh:   scripts/deltaai/submit.sh train       (or setup-train: setup + this)
# What it runs: slime's train.py inside the slime image, through train_inside.sh (private Ray on
# loopback, our reward and hooks), with the flags src/train_args.py builds from
# configs/train_reinit.toml. The run name and step count come from there too. Outputs:
#   runs/<run name>/slime_args_<job>.txt   the exact flags of this job
#   runs/<run name>/rollouts/              every training and validation answer with its reward
#   $SLIME_CKPT_DIR/<run name>/            Megatron checkpoints every [checkpoint] save_interval steps
# Resume: the checkpoint folder is also --load, so resubmitting after a time limit continues from
# the last saved step (the steps after it are redone).
# Steps:
#   0. refuse uncommitted changes in src/, configs/, scripts/ (the run must be traceable);
#      the image hash; both markers (setup_pass = Megatron copy verified, smoke_pass = the 2-step
#      smoke run passed), with a WARNING if the training config or code changed since the smoke run;
#   1. the config check (train_args --check) and slime's real parser on THIS job's flags
#      (check_slime_args.py, ~2 min). Since 2026-11-27 setup_pass no longer implies the flag check
#      passed (it is written after the round trip), so the check runs here, before the long run;
#   2. time budget ([job], user 2026-11-17/27): train until [job] eval_reserve_min minutes are left
#      (train_inside.sh's timeout), then
#   3. free the GPUs and evaluate the best saved checkpoint in this job (eval_trained.sbatch best:
#      the GPQA control evaluation on [server] dp_size GPUs). A crash, not the budget, stops the
#      job without evaluating.
# Port of Experiment-3 @ 97e4706. Changes: 4 GPUs (Ray gets [hardware] gpus_per_node), 64 CPUs /
# 240G, step 1 (the flag check), the smoke-marker staleness warning, the disk weight-push folder.
# Requires: the image (submit.sh build-slime, hash recorded), submit.sh slime-setup (both markers),
# the slime input files (python3 -m src.train_prep, copied to DeltaAI).
#SBATCH --job-name=gar-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
# GPUs = configs/train_reinit.toml [hardware] gpus_per_node; CPUs/RAM as slime_setup.sbatch (the
# optimizer state lives in CPU memory, ~84 GB for both training GPUs). tests/test_slime_container.py
# checks these lines against the config and slime_setup.sbatch.
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --output=runs/_slurm/%x_%j.out
#SBATCH --constraint=projects
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/deltaai/env.sh
source scripts/deltaai/slime_common.sh
JOB="${SLURM_JOB_ID:-local}"
CFG=configs/train_reinit.toml
echo "[train] $(date -u +%FT%TZ) host $(hostname) | git $(git rev-parse --short HEAD) | job $JOB"

# ---- 0
if [ -n "$(git status --porcelain -- src configs scripts)" ]; then
  echo "[train] uncommitted changes in src/, configs/ or scripts/: commit first so the run is traceable"; exit 1
fi
check_slime_sif
slime_paths
RUN_DIR="runs/$RUN_NAME"
SAVE_DIR="$SLIME_CKPT_DIR/$RUN_NAME"
if [ ! -f "$REF_DIR.setup_pass" ]; then
  echo "[train] $REF_DIR has no setup_pass marker: run submit.sh slime-setup first"; exit 1
fi
if [ ! -f "$REF_DIR.smoke_pass" ]; then
  echo "[train] no smoke_pass marker: the 2-step smoke run (scripts/deltaai/smoke_train.sh) has not passed"; exit 1
fi
# The marker reads "job <id> | git <short hash> | ...". Training code or config changed since then
# is allowed (e.g. the final prompts), but said, so a re-smoke can be chosen knowingly.
SMOKE_GIT="$(sed -n 's/.*| git \([0-9a-f]*\) |.*/\1/p' "$REF_DIR.smoke_pass")"
if [ -n "$SMOKE_GIT" ] && git cat-file -e "$SMOKE_GIT" 2>/dev/null; then
  CHANGED="$(git diff --name-only "$SMOKE_GIT" HEAD -- src configs scripts/deltaai | tr '\n' ' ')"
  if [ -n "$CHANGED" ]; then echo "[train] WARNING: changed since the smoke run ($SMOKE_GIT): $CHANGED"; fi
else
  echo "[train] WARNING: could not read the smoke run's git commit from $REF_DIR.smoke_pass"
fi
mkdir -p "$RUN_DIR/rollouts" "$SAVE_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ---- 1
S python3 -m src.train_args --hf-checkpoint "$HF_SNAPSHOT" --check
FLAGS_FILE="$RUN_DIR/slime_args_$JOB.txt"
WEIGHT_SYNC_DIR="$SLIME_CKPT_DIR/_weight_sync_$RUN_NAME"   # used only if [hardware] update_weight_transport = "disk"
S python3 -m src.train_args --hf-checkpoint "$HF_SNAPSHOT" --ref-load "$REF_DIR" --save "$SAVE_DIR" \
  --update-weight-disk-dir "$WEIGHT_SYNC_DIR" --out "$FLAGS_FILE" > /dev/null
# CUDA_DEVICE_MAX_CONNECTIONS=1 as train_inside.sh exports it (Megatron refuses TP 2 otherwise; job 3686748)
S env GAR_TRAIN_CONFIG="$REPO_DIR/$CFG" PYTHONPATH="/opt/Megatron-LM" CUDA_DEVICE_MAX_CONNECTIONS=1 \
  python3 scripts/deltaai/check_slime_args.py "$FLAGS_FILE"
echo "[train] run $RUN_NAME | flags $FLAGS_FILE | checkpoints $SAVE_DIR"
# One run = one setup. A resubmitted job continues the same checkpoints, so its flags and its
# training data (the manifest records the prompt hash and the files' hashes) must be the ones
# the run started with; otherwise it is a new run (CLAUDE.md Rule 3: new run name).
DATA_DIR="$(cfg_value "$CFG" data output_dir)"
IDENTITY="flags $(sha256sum "$FLAGS_FILE" | cut -d' ' -f1) | data manifest $(sha256sum "$DATA_DIR/manifest.json" | cut -d' ' -f1)"
ID_FILE="$RUN_DIR/run_identity.txt"
if [ -f "$ID_FILE" ]; then
  if [ "$(cat "$ID_FILE")" != "$IDENTITY" ]; then
    echo "[train] $RUN_NAME was started with: $(cat "$ID_FILE")"
    echo "[train] this job has:            $IDENTITY"
    echo "[train] flags or training data changed: give the run a new [run] name instead of resuming it"; exit 1
  fi
  echo "[train] resuming $RUN_NAME with the same flags and data ($ID_FILE)"
else
  echo "$IDENTITY" > "$ID_FILE"
fi

# ---- 2 (time budget: seconds of training allowed before the evaluation reserve)
EVAL_AFTER="$(cfg_value "$CFG" job eval_after_training)"   # True / False (Python's spelling)
RESERVE_MIN="$(cfg_value "$CFG" job eval_reserve_min)"
MIN_TRAIN_MIN="$(cfg_value "$CFG" job min_train_min)"
BUDGET_S=0
if [ "$EVAL_AFTER" = "True" ]; then
  # The job's end time, from the first source that answers (it includes any `scontrol update` of
  # the limit while the job was pending): Slurm's SLURM_JOB_END_TIME (epoch seconds), else squeue's
  # %e, else scontrol's EndTime (both like 2026-11-28T09:44:10). An empty answer is skipped (GNU
  # `date -d ""` would mean "today 00:00").
  END_S=""
  if [[ "${SLURM_JOB_END_TIME:-}" =~ ^[0-9]+$ ]]; then END_S="$SLURM_JOB_END_TIME"; fi
  if [ -z "$END_S" ]; then
    END="$(squeue -h -j "${SLURM_JOB_ID:-none}" -o %e 2>/dev/null || true)"
    if [ -n "$END" ]; then END_S="$(date -d "$END" +%s 2>/dev/null || true)"; fi
  fi
  if [ -z "$END_S" ]; then
    END="$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | grep -o 'EndTime=[^ ]*' | cut -d= -f2 || true)"
    if [ -n "$END" ]; then END_S="$(date -d "$END" +%s 2>/dev/null || true)"; fi
  fi
  END="unknown"; if [ -n "$END_S" ]; then END="$(date -u -d "@$END_S" +%FT%TZ 2>/dev/null || echo unknown)"; fi
  if [ -n "$END_S" ] && [ "$END_S" -gt "$(date +%s)" ]; then
    BUDGET_S=$(( END_S - $(date +%s) - RESERVE_MIN * 60 ))
    echo "[train] job ends $END; training budget $((BUDGET_S / 60)) min, then $RESERVE_MIN min for the evaluation"
    if [ "$BUDGET_S" -lt $(( MIN_TRAIN_MIN * 60 )) ]; then
      echo "[train] less than $MIN_TRAIN_MIN min of budget: not training; submit a longer job"; exit 1
    fi
  else
    echo "[train] WARNING: could not read the job's end time ($END): training without a budget, no evaluation in this job"
    EVAL_AFTER="False"
  fi
fi

NUM_GPUS="$(cfg_value "$CFG" hardware gpus_per_node)"
RAY_TMP="/tmp/ray_gar_$JOB"   # short path: Ray's socket paths must stay under ~100 characters
set +e
S env GAR_TRAIN_CONFIG="$REPO_DIR/$CFG" GAR_ROLLOUT_LOG_DIR="$REPO_DIR/$RUN_DIR/rollouts" REPO_DIR="$REPO_DIR" \
  bash scripts/deltaai/train_inside.sh "$FLAGS_FILE" "$RAY_TMP" "$(free_port)" "$NUM_GPUS" "${SLURM_CPUS_PER_TASK:-64}" "$BUDGET_S"
STATUS=$?
set -e
case "$STATUS" in
  0)       echo "[train] $(date -u +%FT%TZ) train.py exited normally (all steps done)" ;;
  124|137) echo "[train] $(date -u +%FT%TZ) time budget reached: training stopped (saved checkpoints kept)" ;;
  *)       echo "[train] $(date -u +%FT%TZ) train.py FAILED with status $STATUS: not evaluating"; exit "$STATUS" ;;
esac
S python3 -m src.train_report "$RUN_DIR/rollouts" || echo "[train] train_report failed (the rollout log is kept)"
if [ "$EVAL_AFTER" != "True" ]; then exit 0; fi

# ---- 3 (free the GPUs: SGLang/Ray processes that outlived train.py would block the evaluation server;
#        this job holds the whole node, so every GPU process on it is this job's)
LEFT=""
for _ in $(seq 1 36); do
  # only numeric PIDs count (some driver versions print a "No running processes" line instead)
  LEFT="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || true)"
  [ -z "$LEFT" ] && break
  sleep 5
done
if [ -n "$LEFT" ]; then
  echo "[train] stopping processes still on the GPUs after 3 min: $(echo $LEFT)"
  for pid in $LEFT; do kill -9 "$pid" 2>/dev/null || true; done
  sleep 10
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "[train] $(date -u +%FT%TZ) starting the evaluation (eval_trained.sbatch best)"
bash scripts/deltaai/eval_trained.sbatch best
