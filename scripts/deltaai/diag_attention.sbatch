#!/bin/bash
# scripts/deltaai/diag_attention.sbatch: diagnose the TE attention-gradient failure of setup job 3680134.
#
# Submit through submit.sh:   scripts/deltaai/submit.sh diag-attn
# Runs scripts/deltaai/diag_te_attention.py in the slime image on ONE GPU (a few minutes; 20 min
# limit = at most 0.33 GPU-hours). It only reports; it writes nothing except its Slurm log.
#SBATCH --job-name=gar-diag-attn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=runs/_slurm/%x_%j.out
#SBATCH --constraint=projects
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/deltaai/env.sh
source scripts/deltaai/slime_common.sh
echo "[diag] $(date -u +%FT%TZ) host $(hostname) | git $(git rev-parse --short HEAD) | job ${SLURM_JOB_ID:-local}"
check_slime_sif
S python3 scripts/deltaai/diag_te_attention.py
