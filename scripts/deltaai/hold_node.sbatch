#!/bin/bash
# scripts/deltaai/hold_node.sbatch: hold a whole GH200 node (4 GPUs) and serve code-server on it.
#
# Submit through submit.sh:   scripts/deltaai/submit.sh hold [SCRIPT [ARGS...]]   (time limit HOLD_TIME, default 8 h)
# With SCRIPT (e.g. scripts/deltaai/setup_and_train.sbatch), the node runs it through in_alloc.sh as
# soon as it starts, so nobody has to be awake when the queue lets it in. If it succeeds, the node
# is released; if it fails, the node stays held for a fix and a retry inside it (no new queue wait).
# Why (user, 2026-11-22 and 2026-11-27): every retry of a batch job waits in the queue again. Inside
#   this allocation, the job scripts run directly on the node (scripts/deltaai/in_alloc.sh, from
#   code-server's terminal or `srun --jobid <id> --overlap --pty bash`), so a fix is retried in
#   minutes. Cost: the node is CHARGED THE WHOLE TIME, idle or not (4 GPU-hours per hour);
#   `scancel <job id>` as soon as it is not needed.
# What : 1. writes runs/_slurm/hold_<job>.info: node, code-server port, end time, and the exact SSH
#           tunnel and run commands;
#        2. starts code-server (configs/code_server.toml, installed by install_code_server.sh) bound
#           to 127.0.0.1 on the node, password from $CODE_SERVER_HOME/password, workspace = the repo.
#           If it is not installed, the node is held anyway (use srun --overlap or ssh);
#        3. runs SCRIPT if given; then holds until the time limit, scancel, success of SCRIPT, or
#           configs/code_server.toml idle_release_min minutes with no job script running (a held node
#           is charged while idle), then stops code-server.
# Resources = the training jobs' (so setup_and_train / train_reinit / eval_trained fit inside;
# tests/test_train_args.py checks they are identical).
#SBATCH --job-name=gar-hold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --output=runs/_slurm/%x_%j.out
#SBATCH --constraint=projects
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/deltaai/env.sh
JOB="${SLURM_JOB_ID:-local}"
NODE="$(hostname -s)"
CS_HOME="${CODE_SERVER_HOME:-$HOME/.gar_code_server}"
get() { sed -n "s/^$1 = \"\(.*\)\"\$/\1/p" configs/code_server.toml | head -n 1; }
VERSION="$(get version)"; PORT="$(get port)"
BIN="$CS_HOME/code-server-$VERSION-linux-arm64/bin/code-server"
port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
for _ in $(seq 1 100); do port_free "$PORT" && break; PORT=$((PORT + 1)); done
END="$(squeue -h -j "$JOB" -o %e 2>/dev/null || echo unknown)"
INFO="runs/_slurm/hold_$JOB.info"
LOGIN="${DELTAAI_LOGIN_HOST:-dtai-login.delta.ncsa.illinois.edu}"
{
  echo "held node $NODE | job $JOB | ends $END | git $(git rev-parse --short HEAD)"
  echo
  echo "code-server (on your laptop, one command; Duo once; leave it open; then open http://127.0.0.1:$PORT):"
  echo "  ssh -t -L $PORT:127.0.0.1:$PORT $USER@$LOGIN ssh -N -L $PORT:127.0.0.1:$PORT $NODE"
  echo "  password: cat $CS_HOME/password   (on DeltaAI)"
  echo
  echo "a shell inside this allocation (from a login node), without code-server:"
  echo "  srun --jobid $JOB --overlap --pty bash"
  echo "  (then nvidia-smi must list 4 GPUs; if it lists none, exit and use: srun --jobid $JOB --overlap --gres=gpu:4 --pty bash)"
  echo
  echo "run a job script inside the allocation (in code-server's terminal or that shell), e.g.:"
  echo "  cd $REPO_DIR && bash scripts/deltaai/in_alloc.sh scripts/deltaai/setup_and_train.sbatch"
  echo
  echo "release the node when done (charged until then):  scancel $JOB"
} > "$INFO"
cat "$INFO"

CS_PID=""
stop() { trap '' TERM INT; [ -n "$CS_PID" ] && kill "$CS_PID" 2>/dev/null || true; echo "[hold] $(date -u +%FT%TZ) released"; }
trap stop EXIT
trap 'exit 0' TERM INT
if [ -x "$BIN" ] && [ -f "$CS_HOME/password" ]; then
  # PASSWORD is read by code-server itself (never on the command line, so `ps` does not show it)
  PASSWORD="$(cat "$CS_HOME/password")" "$BIN" --bind-addr "127.0.0.1:$PORT" --auth password \
    --user-data-dir "$CS_HOME/user-data" --extensions-dir "$CS_HOME/extensions" \
    --disable-telemetry --disable-update-check "$REPO_DIR" > "runs/_slurm/code_server_$JOB.log" 2>&1 &
  CS_PID=$!
  echo "[hold] code-server $VERSION started (pid $CS_PID) on 127.0.0.1:$PORT; log runs/_slurm/code_server_$JOB.log"
else
  echo "[hold] code-server not installed ($BIN or $CS_HOME/password missing: bash scripts/deltaai/install_code_server.sh"
  echo "[hold] on a login node). Holding the node anyway: use the srun line above."
fi
IDLE_MAX="$(get idle_release_min)"
if [ "$#" -gt 0 ]; then
  echo "[hold] $(date -u +%FT%TZ) starting $* inside this allocation"
  set +e
  bash scripts/deltaai/in_alloc.sh "$@" > "runs/_slurm/hold_${JOB}_autorun.out" 2>&1
  AUTO=$?
  set -e
  if [ "$AUTO" -eq 0 ]; then echo "[hold] $(date -u +%FT%TZ) $1 succeeded: releasing the node"; exit 0; fi
  echo "[hold] $(date -u +%FT%TZ) $1 FAILED (status $AUTO; runs/_slurm/hold_${JOB}_autorun.out): the node stays"
  echo "[hold] held for a fix and a retry (bash scripts/deltaai/in_alloc.sh ...), released after $IDLE_MAX idle minutes"
fi
IDLE=0
while [ "$IDLE" -lt "$IDLE_MAX" ]; do
  sleep 60
  BUSY=0
  for f in runs/_slurm/alloc_"${JOB}"_*.pid; do
    [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && BUSY=1
  done
  if [ "$BUSY" = 1 ]; then IDLE=0; else IDLE=$((IDLE + 1)); fi
done
echo "[hold] $(date -u +%FT%TZ) no job script ran for $IDLE_MAX minutes: releasing the node"
