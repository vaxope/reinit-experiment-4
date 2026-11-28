#!/bin/bash
# scripts/deltaai/in_alloc.sh: run one of the job scripts INSIDE a held allocation (hold_node.sbatch).
#
# Usage (in code-server's terminal on the held node, or after `srun --jobid <id> --overlap --pty bash`):
#   bash scripts/deltaai/in_alloc.sh scripts/deltaai/setup_and_train.sbatch
#   bash scripts/deltaai/in_alloc.sh scripts/deltaai/eval_trained.sbatch best
#   bash scripts/deltaai/in_alloc.sh scripts/deltaai/slime_setup.sbatch
# What : the job scripts name their outputs after $SLURM_JOB_ID and size the training budget from the
#        job's end time. Inside one allocation every attempt would share the allocation's id, and a
#        second attempt would find the first one's folders ("already exists; not overwriting"). So:
#          - SLURM_JOB_ID = <allocation id>r<attempt time>, unique per attempt (e.g. 3690001r1128093015);
#          - SLURM_JOB_END_TIME = the allocation's end (squeue), so train_reinit.sbatch's budget and
#            in-job evaluation fit inside the allocation;
#          - SLURM_SUBMIT_DIR = the repo, SLURM_CPUS_PER_TASK kept;
#        then runs the script with nohup in its own session (a closed browser tab or SSH tunnel does
#        not stop it) and follows its log: runs/_slurm/<script>_<attempt id>.out. Ctrl-C stops only
#        the following; the script keeps running (stop it with kill <pid>, printed below).
# Refuses outside an allocation (no $SLURM_JOB_ID): login nodes have no GPUs.
set -euo pipefail
cd "$(dirname "$0")/../.."
SCRIPT="${1:?usage: in_alloc.sh <job script> [args...]}"; shift
[ -f "$SCRIPT" ] || { echo "[in_alloc] no such script: $SCRIPT"; exit 1; }
ALLOC="${SLURM_JOB_ID:-}"
if [ -z "$ALLOC" ]; then
  echo "[in_alloc] not inside a Slurm allocation: run it in the held node's code-server terminal or after"
  echo "           srun --jobid <hold job id> --overlap --pty bash"; exit 1
fi
ALLOC="${ALLOC%%r*}"   # a nested call keeps the allocation's own id
END="$(squeue -h -j "$ALLOC" -o %e 2>/dev/null || true)"
END_S="$(date -d "$END" +%s 2>/dev/null || true)"
if [ -z "$END_S" ]; then echo "[in_alloc] could not read the end time of allocation $ALLOC"; exit 1; fi
ATTEMPT="${ALLOC}r$(date +%m%d%H%M%S)"
LOG="runs/_slurm/$(basename "$SCRIPT" | sed 's/\.[a-z]*$//')_$ATTEMPT.out"
mkdir -p runs/_slurm
echo "[in_alloc] $SCRIPT $* | attempt $ATTEMPT | allocation ends $END | log $LOG"
SLURM_JOB_ID="$ATTEMPT" SLURM_JOB_END_TIME="$END_S" SLURM_SUBMIT_DIR="$(pwd)" \
  SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-64}" \
  nohup setsid bash "$SCRIPT" "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
# hold_node.sbatch counts the node as busy while any of these pid files names a live process
echo "$PID" > "runs/_slurm/alloc_${ALLOC}_$ATTEMPT.pid"
echo "[in_alloc] running as pid $PID (stop it: kill -- -$PID). Following the log (Ctrl-C stops following only):"
sleep 2
set +e   # report the script's status instead of stopping here
tail -n +1 -f "$LOG" --pid="$PID"
wait "$PID" 2>/dev/null; STATUS=$?
set -e
echo "[in_alloc] $SCRIPT finished with status $STATUS (log $LOG)"
exit "$STATUS"
