#!/bin/bash
# scripts/deltaai/setup_and_train.sbatch: the setup job and the training job in ONE Slurm job,
# so there is one queue wait instead of two (user, 2026-11-17). 1 GH200 node, 4 GPUs.
#
# Submit through submit.sh:   scripts/deltaai/submit.sh setup-train
# Steps:
#   0. refuse uncommitted changes in src/, configs/, scripts/ (checked here first, so a dirty tree
#      fails in seconds instead of after the setup);
#   1. setup + smoke run = slime_setup.sbatch unchanged, run as a plain script inside this
#      allocation, unless the smoke run already passed (<Megatron copy>.smoke_pass exists, e.g. a
#      resubmission after a time limit). slime_setup.sbatch itself reuses a verified Megatron copy
#      (setup_pass) and stops on a copy without it;
#   2. training = train_reinit.sbatch unchanged (it re-checks the tree, image, both markers and the
#      flags, trains within the time budget, then evaluates the best checkpoint).
# Each step stops the job on failure (set -e), so training never starts after a failed setup.
# Port of Experiment-3 @ 97e4706. Changes: 4 GPUs; step 1 keys on smoke_pass (this repo's
# slime_setup.sbatch includes the smoke run and reuses a verified copy by itself).
#SBATCH --job-name=gar-setup-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Resources = train_reinit.sbatch's (tests/test_slime_container.py checks they are identical);
# #SBATCH lines inside the two scripts are plain comments when they run this way.
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --output=runs/_slurm/%x_%j.out
#SBATCH --constraint=projects
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "[setup-train] $(date -u +%FT%TZ) host $(hostname) | git $(git rev-parse --short HEAD) | job ${SLURM_JOB_ID:-local}"

# ---- 0
if [ -n "$(git status --porcelain -- src configs scripts)" ]; then
  echo "[setup-train] uncommitted changes in src/, configs/ or scripts/: commit first so the run is traceable"; exit 1
fi
source scripts/deltaai/env.sh
source scripts/deltaai/slime_common.sh
slime_paths

# ---- 1
if [ -f "$REF_DIR.smoke_pass" ]; then
  echo "[setup-train] setup and smoke run already passed ($(cat "$REF_DIR.smoke_pass")): skipping them"
else
  T0=$SECONDS
  bash scripts/deltaai/slime_setup.sbatch
  echo "[setup-train] setup + smoke run took $((SECONDS - T0)) s"
fi

# ---- 2
bash scripts/deltaai/train_reinit.sbatch
echo "[setup-train] $(date -u +%FT%TZ) done"
