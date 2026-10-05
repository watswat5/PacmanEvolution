#!/usr/bin/env bash
# Hook back into a training run that is already going in the background.
#
#   ./attach_board_graph_run.sh                 attach to the most recently active run
#   ./attach_board_graph_run.sh runs/<dir>      attach to that run
#   ./attach_board_graph_run.sh --stop [dir]    stop it cleanly without following
#   ./attach_board_graph_run.sh --status [dir]  say whether it is running, then exit
#
# While attached:
#   Ctrl-C   stop training cleanly (it finishes writing its checkpoint and plots;
#            rerun the original run script to continue later)
#   q        stop following; training keeps running
#
# Works for every run_board_graph_*.sh script. The run is found through the
# environment its launcher gave it, so no PID file is needed.
set -euo pipefail

cd "$(dirname "$0")"

mode=attach
case "${1:-}" in
  --stop) mode=stop; shift ;;
  --status) mode=status; shift ;;
  -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

run_directory=${1:-}
if [[ -z "$run_directory" ]]; then
  # The run whose log was written to most recently.
  log_path=$(ls -t runs/*/training.log 2>/dev/null | head -n 1 || true)
  if [[ -z "$log_path" ]]; then
    echo "No runs/*/training.log found." >&2
    exit 1
  fi
  run_directory=$(dirname "$log_path")
fi
run_directory=${run_directory%/}
log_path="$run_directory/training.log"
if [[ ! -f "$log_path" ]]; then
  echo "No training.log in $run_directory" >&2
  exit 1
fi

# The detached process of a run carries BOARD_GRAPH_DETACHED=1 and its run
# directory in its environment, and leads its own session. Its children (the
# trainer and its workers) inherit the same environment, so keep only the
# session leader.
find_run_pid() {
  local environ pid directory
  for environ in /proc/[0-9]*/environ; do
    pid=${environ#/proc/}; pid=${pid%/environ}
    [[ -r "$environ" ]] || continue
    grep -qz '^BOARD_GRAPH_DETACHED=1$' "$environ" 2>/dev/null || continue
    directory=$(tr '\0' '\n' < "$environ" 2>/dev/null \
      | sed -n 's/^BOARD_GRAPH_\(EXPERIMENT\|OUTPUT\)_DIRECTORY=//p' | head -n 1)
    [[ "${directory%/}" == "$run_directory" ]] || continue
    [[ "$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')" == "$pid" ]] || continue
    echo "$pid"
    return 0
  done
  return 1
}

if ! run_pid=$(find_run_pid); then
  echo "Nothing is running for $run_directory."
  echo "Last log lines:"
  tail -n 3 "$log_path"
  exit 1
fi

# Multi-run scripts are a shell loop that stops on USR1; single-run scripts
# hand over to the trainer itself, which stops on an interrupt.
if [[ "$(ps -o comm= -p "$run_pid")" == bash ]]; then
  stop_signal=USR1
else
  stop_signal=INT
fi

request_stop() {
  kill "-$stop_signal" "$run_pid" 2>/dev/null || true
}

if [[ "$mode" == status ]]; then
  echo "Running: $run_directory (PID $run_pid)"
  tail -n 2 "$log_path"
  exit 0
fi

if [[ "$mode" == stop ]]; then
  echo "Stopping $run_directory (PID $run_pid); it will finish writing its checkpoint and plots."
  request_stop
  while kill -0 "$run_pid" 2>/dev/null; do
    sleep 1
  done
  echo "Stopped."
  tail -n 3 "$log_path"
  exit 0
fi

printf 'Attached to %s (PID %s).\nCtrl-C stops training cleanly; q stops following and leaves it running.\n\n' \
  "$run_directory" "$run_pid"
# The log follower runs in the background, where the shell makes it ignore
# Ctrl-C, so no log lines are lost; it exits when training does.
tail --pid="$run_pid" -n 15 -f "$log_path" &
tail_pid=$!
interrupted=0
forward_interrupt() {
  if (( interrupted )); then
    printf '\nStopped following. PID %s may still be writing its final artifacts.\n' "$run_pid"
    kill "$tail_pid" 2>/dev/null || true
    exit 130
  fi
  interrupted=1
  printf '\nStopping training (PID %s); waiting for it to finish writing. Ctrl-C again to stop following.\n' "$run_pid"
  request_stop
}
trap forward_interrupt INT
while kill -0 "$tail_pid" 2>/dev/null; do
  key=
  if [[ -t 0 ]]; then
    read -rsn1 -t 1 key 2>/dev/null || true
  else
    sleep 1
  fi
  if [[ "$key" == q || "$key" == Q ]]; then
    kill "$tail_pid" 2>/dev/null || true
    printf '\nStopped following. Training continues (PID %s).\n' "$run_pid"
    exit 0
  fi
done
exit 0
