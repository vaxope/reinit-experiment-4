#!/bin/bash
# scripts/deltaai/run_control.sbatch: Slurm batch job for the pilot and the full control run.
#
# Submit through submit.sh (it adds --account, --partition and --time and asks for
# confirmation):   scripts/deltaai/submit.sh pilot     /     scripts/deltaai/submit.sh full
# Arguments after the script name go to src/generate.py (submit.sh passes --run-id and --pilot).
#
# Resources: 1 node, [server] dp_size GPUs (4 = a whole GH200 node, one model copy per GPU,
# user 2026-11-27; tests/test_slime_container.py checks this line against the config), CPU cores
# for the 4 SGLang processes, tokenization and the HTTP client, RAM for 4 copies loading the
# weights. Billing is per GPU (a node's CPUs and RAM come with its 4 GPUs). The job's time
# limit is set by submit.sh; if it is hit, everything written so far is kept, and submitting
# the same command again resumes where it stopped.
#SBATCH --job-name=gar-control
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --output=runs/_slurm/%x_%j.out
# The job reads and writes /projects (repo, runs, HF cache, container). NCSA asks jobs to
# declare that, so they can be held or handled if /projects has problems (DeltaAI user guide,
# "File System Dependency Specification for Jobs"). Charges are unaffected.
#SBATCH --constraint=projects
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # Slurm starts the job in the directory sbatch was run from
if [ -f scripts/deltaai/env.sh ]; then source scripts/deltaai/env.sh; fi

echo "[job] $(date -u +%FT%TZ) run_control: args: $*"
bash scripts/deltaai/serve_and_generate.sh "$@"
