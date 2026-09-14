#!/usr/bin/env bash
# plan.md §11: parameter-matched engineered baselines (tanh RNN, GRU, tiny Transformer), 3 seeds each,
# same data/split/context/steps as the FlyGPT config, as two GPU queues.
#   scripts/launch_baselines.sh [configs/launch_100k.yaml] [logdir] [gpuA] [gpuB]
set -u
CFG=${1:-configs/launch_100k.yaml}; LOG=${2:-runs/launch_logs}; A=${3:-0}; B=${4:-1}
cd "$(dirname "$0")/.." && mkdir -p "$LOG"; source ~/.venv/bin/activate
queue() { local gpu=$1; shift; ( for job in "$@"; do c=${job%%:*}; k=${job##*:}
  CUDA_VISIBLE_DEVICES=$gpu python train.py "$CFG" --condition "$c" --seed "$k" > "$LOG/${c}_seed${k}.log" 2>&1
  echo "$(date -u +%FT%TZ) gpu$gpu finished $job exit=$?" >> "$LOG/queue.log"; done ) & }
queue "$A" rnn:1 gru:1 transformer:1 rnn:3 gru:3
queue "$B" rnn:2 gru:2 transformer:2 transformer:3
wait
