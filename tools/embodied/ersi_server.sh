#!/usr/bin/env bash
# Serve a LoongForge pi05 checkpoint to the ERSI sim box (4090).
#
# Creates tmux session "ersi_server", starts the policy server inside it, waits until it is
# healthy, then holds a reverse SSH tunnel so the sim box reaches the server on its own
# localhost:PORT / HEALTH_PORT. Killing the session (or Ctrl-C inside it) stops both.
#
# Usage:
#   ERSI_REMOTE=<user>@<sim-host> tools/embodied/ersi_server.sh <ckpt_dir>
#                                                                 # steps_N dir with model.safetensors
#   CONFIG=<yaml> ERSI_REMOTE=... tools/embodied/ersi_server.sh <ckpt_dir>
#                                                                 # other model layout; default is
#                                                                 # ersi_server.yaml (16-dim joint pi05,
#                                                                 # delta actions). For an absolute-action
#                                                                 # checkpoint use ersi_server_abs.yaml.
# Logs and the generated per-checkpoint yaml land in the run directory (<ckpt_dir>/../..).
#
# Environment:
#   ERSI_REMOTE           required, <user>@<host> of the sim box receiving the reverse tunnel
#   ERSI_REMOTE_SSH_PORT  sshd port on the sim box (default 22)
#   ERSI_SSH_KEY          private key for that host (default ~/.ssh/id_ed25519)
#   PYTHON                interpreter running the policy server (default python3 on PATH)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
CONFIG=${CONFIG:-$HERE/ersi_server.yaml}
SESSION=ersi_server
PORT=12093          # must match server.port / server.health_port in CONFIG
HEALTH_PORT=12094
REMOTE=${ERSI_REMOTE:?set ERSI_REMOTE=<user>@<sim-host> (the box that should reach the policy server)}
REMOTE_SSH_PORT=${ERSI_REMOTE_SSH_PORT:-22}
SSH_KEY=${ERSI_SSH_KEY:-$HOME/.ssh/id_ed25519}
PYTHON=${PYTHON:-python3}

if [[ "${1:-}" != "--inner" ]]; then
  CKPT=${1:?usage: $0 <ckpt_dir>   (a steps_N directory containing model.safetensors)}
  CKPT=$(readlink -f "$CKPT")
  [[ -f "$CKPT/model.safetensors" ]] || { echo "no model.safetensors in $CKPT"; exit 1; }
  [[ -f "$CONFIG" ]] || { echo "base config not found: $CONFIG"; exit 1; }
  tmux has-session -t "$SESSION" 2>/dev/null && { echo "tmux session '$SESSION' already exists: tmux attach -t $SESSION"; exit 1; }
  RUN_DIR=$(dirname "$(dirname "$CKPT")")
  GEN="$RUN_DIR/ersi_server_$(basename "$CKPT").yaml"
  sed "s#^\([[:space:]]*ckpt_path:\).*#\1 $CKPT#" "$CONFIG" > "$GEN"
  grep -q "ckpt_path: $CKPT" "$GEN" || { echo "failed to set ckpt_path in $GEN"; exit 1; }
  # A pre-existing tmux server hands new sessions ITS environment, not ours, so pass
  # every override explicitly instead of relying on inheritance.
  tmux new-session -d -s "$SESSION" -c "$RUN_DIR" \
    "CONFIG='$GEN' ERSI_REMOTE='$REMOTE' ERSI_REMOTE_SSH_PORT='$REMOTE_SSH_PORT' \
     ERSI_SSH_KEY='$SSH_KEY' PYTHON='$PYTHON' '$HERE/$(basename "$0")' --inner"
  echo "serving $CKPT  (config $GEN)"
  echo "tmux session '$SESSION' started. attach: tmux attach -t $SESSION   stop: tmux kill-session -t $SESSION"
  exit 0
fi

# ---- inside tmux ----
LOG=$PWD/ersi_server.log
cd "$REPO"
CUDA_MODULE_LOADING=EAGER PYTHONPATH=. "$PYTHON" \
  -m loongforge.embodied.eval.servers.loongforge_server --config "$CONFIG" >"$LOG" 2>&1 &
SERVER=$!
trap 'echo "stopping server $SERVER"; kill $SERVER 2>/dev/null' EXIT

echo "server starting with $CONFIG (pid $SERVER, log $LOG)"
until curl -s --noproxy '*' -m 3 "http://127.0.0.1:$HEALTH_PORT/healthz" >/dev/null; do
  kill -0 "$SERVER" 2>/dev/null || { echo "server exited, see $LOG"; tail -5 "$LOG"; exit 1; }
  sleep 10
done
echo "server ready, holding tunnel to $REMOTE:$REMOTE_SSH_PORT (remote localhost:$PORT, :$HEALTH_PORT)"
while true; do
  env -u http_proxy -u https_proxy -u all_proxy \
    ssh -i "$SSH_KEY" -p "$REMOTE_SSH_PORT" -N \
      -R "$PORT:localhost:$PORT" -R "$HEALTH_PORT:localhost:$HEALTH_PORT" \
      -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
      "$REMOTE"
  echo "$(date '+%F %T') tunnel exited rc=$?, retrying in 5s"; sleep 5
done
