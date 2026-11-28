#!/bin/bash
# scripts/deltaai/setup_and_train.sbatch: setup, the 4B control run and training in ONE Slurm job,
# so there is one queue wait instead of three (user, 2026-11-17 and 2026-11-27). 1 GH200 node, 4 GPUs.
#
# Submit through submit.sh:   scripts/deltaai/submit.sh setup-train
# (or inside a held node: bash scripts/deltaai/in_alloc.sh scripts/deltaai/setup_and_train.sbatch)
# Steps:
#   0. refuse uncommitted changes in src/, configs/, scripts/ (checked here first, so a dirty tree
#      fails in seconds instead of after the setup);
#   1. setup + smoke run = slime_setup.sbatch unchanged, run as a plain script inside this
#      allocation. Skipped only if the smoke run passed (<Megatron copy>.smoke_pass) AND nothing in
#      src/, configs/ or scripts/deltaai/ changed since the commit it records: new prompts, data or
#      code are smoke-tested again automatically (no marker has to be deleted by hand).
#      slime_setup.sbatch reuses a verified Megatron copy (setup_pass) by itself;
#   2. the 4B control run ([job] control_run_in_job, user 2026-11-27): run_control's command with run
#      id [job] control_run_id on this node's GPUs, capped at [job] control_max_min; then score and
#      check_run. Resumes if part of it exists; does NOT stop the job on failure (training does not
#      depend on it; it is resumed later with submit.sh full <id>);
#   3. training = train_reinit.sbatch unchanged (re-checks the tree, image, markers and flags, trains
#      within the time budget, then evaluates the best checkpoint).
# Steps 1 and 3 stop the job on failure (set -e), so training never starts after a failed setup.
# Port of Experiment-3 @ 97e4706. Changes: 4 GPUs; step 1's change detection; step 2 (new).
#SBATCH --job-name=gar-setup-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Resources = train_reinit.sbatch's (tests/test_train_args.py checks they are identical);
# #SBATCH lines inside the called scripts are plain comments when they run this way.
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --output=runs/_slurm/%x_%j.out
#SBATCH --constraint=projects
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
CFG=configs/train_reinit.toml
echo "[setup-train] $(date -u +%FT%TZ) host $(hostname) | git $(git rev-parse --short HEAD) | job ${SLURM_JOB_ID:-local}"

# ---- 0
if [ -n "$(git status --porcelain -- src configs scripts)" ]; then
  echo "[setup-train] uncommitted changes in src/, configs/ or scripts/: commit first so the run is traceable"; exit 1
fi
source scripts/deltaai/env.sh
source scripts/deltaai/slime_common.sh
slime_paths

# ---- 1
RESMOKE=1
if [ -f "$REF_DIR.smoke_pass" ]; then
  SMOKE_GIT="$(sed -n 's/.*| git \([0-9a-f]*\) |.*/\1/p' "$REF_DIR.smoke_pass")"
  if [ -n "$SMOKE_GIT" ] && git cat-file -e "$SMOKE_GIT" 2>/dev/null \
     && git diff --quiet "$SMOKE_GIT" HEAD -- src configs scripts/deltaai; then
    RESMOKE=0
    echo "[setup-train] smoke run passed at the current code ($(cat "$REF_DIR.smoke_pass")): skipping setup + smoke"
  else
    echo "[setup-train] code or config changed since the smoke run ($(cat "$REF_DIR.smoke_pass")): running it again"
  fi
fi
if [ "$RESMOKE" = 1 ]; then
  T0=$SECONDS
  bash scripts/deltaai/slime_setup.sbatch
  echo "[setup-train] setup + smoke run took $((SECONDS - T0)) s"
fi

# ---- 2 (the control run never stops this job: its status is reported, training follows)
if [ "$(cfg_value "$CFG" job control_run_in_job)" = "True" ]; then
  CTL_ID="$(cfg_value "$CFG" job control_run_id)"
  CTL_MAX_MIN="$(cfg_value "$CFG" job control_max_min)"
  export CONTROL_CONFIG="$(cfg_value "$CFG" run control_config)"
  T0=$SECONDS
  echo "[setup-train] $(date -u +%FT%TZ) control run $CTL_ID ($CONTROL_CONFIG), at most $CTL_MAX_MIN min"
  set +e
  # SIGTERM at the cap: serve_and_generate.sh stops the server and generate.py keeps every finished record
  timeout --signal=TERM --kill-after=120 "$((CTL_MAX_MIN * 60))" \
    bash scripts/deltaai/serve_and_generate.sh --run-id "$CTL_ID"
  CTL=$?
  (
    source scripts/deltaai/common.sh
    HF_HUB_OFFLINE=1 run python3 -m src.score "runs/$CTL_ID/generations.jsonl" --config "$CONTROL_CONFIG" --overwrite \
      && HF_HUB_OFFLINE=1 run python3 -m src.check_run "runs/$CTL_ID" --config "$CONTROL_CONFIG"
  )
  CHECK=$?
  set -e
  unset CONTROL_CONFIG
  echo "[setup-train] control run $CTL_ID: generation status $CTL, score/check_run status $CHECK, $((SECONDS - T0)) s"
  if [ "$CTL" -ne 0 ] || [ "$CHECK" -ne 0 ]; then
    echo "[setup-train] WARNING: control run $CTL_ID is not complete and clean: resume it later with" \
         "scripts/deltaai/submit.sh full $CTL_ID (training continues now)"
  fi
  # the evaluation server must be gone before Ray and SGLang take the GPUs
  for _ in $(seq 1 36); do
    LEFT="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || true)"
    [ -z "$LEFT" ] && break
    sleep 5
  done
  if [ -n "${LEFT:-}" ]; then
    echo "[setup-train] stopping processes still on the GPUs: $(echo $LEFT)"
    for pid in $LEFT; do kill -9 "$pid" 2>/dev/null || true; done
    sleep 10
  fi
fi

# ---- 3
bash scripts/deltaai/train_reinit.sbatch
echo "[setup-train] $(date -u +%FT%TZ) done"
