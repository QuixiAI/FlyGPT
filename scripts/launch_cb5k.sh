#!/usr/bin/env bash
# Launch the §13 paired runs (real + degree_preserving × seeds 1–5) as per-GPU queues, detached from
# the calling shell so they survive session teardown. Two of the eight GPUs run two seeds back to back.
#   scripts/launch_cb5k.sh [configs/launch.yaml] [logdir]
set -u
CFG=${1:-configs/launch.yaml}
LOG=${2:-runs/launch_logs}
cd "$(dirname "$0")/.." && mkdir -p "$LOG"
source ~/.venv/bin/activate
queue() {  # queue <gpu> <condition:seed> ...
  local gpu=$1; shift
  ( for job in "$@"; do
      c=${job%%:*}; k=${job##*:}
      CUDA_VISIBLE_DEVICES=$gpu python train.py "$CFG" --condition "$c" --seed "$k" > "$LOG/${c}_seed${k}.log" 2>&1
      echo "$(date -u +%FT%TZ) gpu$gpu finished $job exit=$?" >> "$LOG/queue.log"
    done ) &
}
queue 0 real:1 real:5
queue 1 degree_preserving:1 degree_preserving:5
queue 2 real:2
queue 3 degree_preserving:2
queue 4 real:3
queue 5 degree_preserving:3
queue 6 real:4
queue 7 degree_preserving:4
wait
