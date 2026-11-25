#!/usr/bin/env bash
# scripts/deltaai/submit.sh: the ONE way to submit DeltaAI jobs, control and training (run on a login node).
#
# Usage:  scripts/deltaai/submit.sh smoke            [--dry-run]
#         scripts/deltaai/submit.sh pilot [RUN_ID]   [--dry-run]   (default RUN_ID pilot_v1)
#         scripts/deltaai/submit.sh full  [RUN_ID]   [--dry-run]   (default RUN_ID control_v1)
#         scripts/deltaai/submit.sh build-slime      [--dry-run]   (slime training image, build_slime.sbatch)
#         scripts/deltaai/submit.sh slime-setup      [--dry-run]   (config checks, weights -> Megatron + round
#                                                                     trip, flag check, 2-step DAPO smoke run; 4 GPUs)
# (The long training run and the trained-model evaluation jobs are not ported yet.)
#
# It reads scripts/deltaai/env.sh (account, partition, time limits), prints exactly what will
# be requested (GPUs, read from the job script's #SBATCH --gpus-per-node line; time limit; the
# maximum GPU-hours that limit allows), asks for
# confirmation (showing the account's remaining balance from `accounts`), then calls sbatch with --account/--partition/--time added on the command
# line (those can't be hard-coded in the .sbatch files because they differ per user).
# --dry-run prints the sbatch command without submitting.
# Re-submitting the same pilot/full command after a time-limit kill RESUMES the run.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ ! -f scripts/deltaai/env.sh ]; then
  echo "scripts/deltaai/env.sh not found: cp scripts/deltaai/env.example.sh scripts/deltaai/env.sh and fill it in"; exit 1
fi
source scripts/deltaai/env.sh

DRY=0; ARGS=()
for a in "$@"; do if [ "$a" = "--dry-run" ]; then DRY=1; else ARGS+=("$a"); fi; done
set -- "${ARGS[@]+"${ARGS[@]}"}"
KIND="${1:-}"

case "$KIND" in
  smoke) TIME="${SMOKE_TIME:-00:30:00}"; SCRIPT=scripts/deltaai/smoke_test.sbatch; JOB_ARGS=() ;;
  pilot) TIME="${PILOT_TIME:-02:00:00}"; SCRIPT=scripts/deltaai/run_control.sbatch; JOB_ARGS=(--run-id "${2:-pilot_v1}" --pilot) ;;
  full)  TIME="${FULL_TIME:-24:00:00}";  SCRIPT=scripts/deltaai/run_control.sbatch; JOB_ARGS=(--run-id "${2:-control_v1}") ;;
  build-slime) TIME="${BUILD_SLIME_TIME:-03:30:00}"; SCRIPT=scripts/deltaai/build_slime.sbatch; JOB_ARGS=() ;;
  slime-setup) TIME="${SLIME_SETUP_TIME:-03:00:00}"; SCRIPT=scripts/deltaai/slime_setup.sbatch; JOB_ARGS=() ;;
  *) echo "usage: $0 smoke|pilot [RUN_ID]|full [RUN_ID]|build-slime|slime-setup [--dry-run]"; exit 2 ;;
esac
for v in DELTAAI_ACCOUNT DELTAAI_PARTITION; do
  case "${!v:-}" in ""|"<"*) echo "$v is not set in scripts/deltaai/env.sh"; exit 1 ;; esac
done

# Preflight for the training setup/smoke job (2026-11-24): everything that would otherwise stop the
# job only AFTER its queue wait is checked here, on the login node, in seconds. The login node's
# python3 is 3.6 (no tomllib), so config values are read with sed (the keys are unique lines in
# configs/train_reinit.toml; tests/test_train_args.py checks that) and the full config check runs
# in the SGLang image's python (src/train_args.py uses only the standard library).
toml_get() { sed -n "s/^$1 = \"\(.*\)\"\$/\1/p" configs/train_reinit.toml | head -n 1; }
preflight_slime() {
  local bad=0 v ctl out repo rev snap
  for v in REPO_DIR HF_HOME SGLANG_SIF SLIME_SIF_DIR SLIME_CKPT_DIR; do
    case "${!v:-}" in ""|"<"*|*"<your-"*) echo "preflight: $v is not set in scripts/deltaai/env.sh"; bad=1 ;; esac
  done
  [ "$bad" = 0 ] || return 1
  ctl="$(toml_get control_config)"; out="$(toml_get output_dir)"; repo="$(toml_get hf_repo)"; rev="$(toml_get revision)"
  snap="$HF_HOME/hub/models--${repo//\//--}/snapshots/$rev"
  for f in "$ctl" "$out/train.jsonl" "$out/val.jsonl" data/gpqa_diamond.jsonl "$snap/config.json" "$SGLANG_SIF"; do
    if [ -f "$f" ]; then echo "preflight: ok      $f"; else echo "preflight: MISSING $f"; bad=1; fi
  done
  [ "$bad" = 0 ] || { echo "preflight: control config -> write it; data -> src.train_data + src.train_prep and copy;"
                      echo "           model -> bash scripts/deltaai/setup.sh"; return 1; }
  local sif_name
  sif_name="$(apptainer exec --bind "$REPO_DIR" "$SGLANG_SIF" python3 -m src.slime_container --sif-name)"
  if [ -f "$SLIME_SIF_DIR/$sif_name" ]; then echo "preflight: ok      $SLIME_SIF_DIR/$sif_name"
  else echo "preflight: MISSING slime image $SLIME_SIF_DIR/$sif_name (submit.sh build-slime)"; return 1; fi
  apptainer exec --bind "$REPO_DIR" --bind "$HF_HOME" "$SGLANG_SIF" \
    python3 -m src.train_args --hf-checkpoint "$snap" --check || return 1
  echo "preflight: free space where checkpoints go (the project quota is what counts: \`quota\`):"
  df -h "$SLIME_CKPT_DIR" 2>/dev/null | tail -n 1 || true
}
if [ "$KIND" = slime-setup ]; then
  preflight_slime || { echo "Preflight failed: not submitted."; exit 1; }
fi

# GPUs the job asks for: its own #SBATCH --gpus-per-node line (the training jobs take a whole node).
GPUS=$(grep -m1 -oE '^#SBATCH --gpus-per-node=[0-9]+' "$SCRIPT" | cut -d= -f2 || true)   # || true: keep our message under pipefail
if [ -z "$GPUS" ]; then echo "no #SBATCH --gpus-per-node line in $SCRIPT"; exit 1; fi
# Maximum GPU-hours = GPUs x time limit (Slurm charges the time actually used, usually less).
IFS=: read -r hh mm ss <<< "$TIME"
MAX_GPUH=$(awk -v g="$GPUS" -v h="$hh" -v m="$mm" -v s="$ss" 'BEGIN { printf "%.2f", g * (h*3600 + m*60 + s) / 3600 }')
mkdir -p runs/_slurm   # Slurm does not create the --output directory itself

CMD=(sbatch --account="$DELTAAI_ACCOUNT" --partition="$DELTAAI_PARTITION" --time="$TIME" "$SCRIPT" "${JOB_ARGS[@]+"${JOB_ARGS[@]}"}")
case "$KIND" in
  smoke) DESC="smoke test: 2 questions x 1 sample x 3 modes" ;;
  build-slime) DESC="build the slime training image (compiles on 64 CPUs; holds 1 GPU, the billing unit)" ;;
  slime-setup) DESC="config checks + model -> Megatron + round trip + flag check + 2-step DAPO smoke run (2 training + 2 SGLang GPUs)" ;;
  *) DESC="${JOB_ARGS[*]}" ;;
esac
echo "Job:       $KIND ($DESC)"
echo "Request:   1 node, $GPUS GPU(s), time limit $TIME on partition $DELTAAI_PARTITION, account $DELTAAI_ACCOUNT"
echo "Cost:      at most $MAX_GPUH GPU-hours (charged for time actually used)"
# Remaining balance of the account (DeltaAI's `accounts` prints: Account  Balance(Hours)  Deposited(Hours) ...).
# Display only: the allocation is shared, so every submission shows what is left.
BAL=$(accounts 2>/dev/null | awk -v a="$DELTAAI_ACCOUNT" '$1 == a { print $2 " of " $3 " hours left" }' || true)
echo "Balance:   ${BAL:-unavailable (run \`accounts\` to check)} on $DELTAAI_ACCOUNT"
echo "Command:   ${CMD[*]}"
if [ "$DRY" = 1 ]; then echo "(dry run: not submitted)"; exit 0; fi
read -r -p "Submit this job? [y/N] " ans
if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then echo "Not submitted."; exit 0; fi
"${CMD[@]}"
echo "Follow it with:  squeue -u \$USER   and   tail -f runs/_slurm/<job-name>_<job-id>.out"
