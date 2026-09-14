#!/usr/bin/env bash
# Fills two GPUs once the baseline queue is done: FrozenFly reservoir at 5k (§11, 3 seeds) and the 10k rung of
# the scaling ladder (§14, real + degree_preserving seed 1). Waits for graphs/cb10k/gate.json to pass first.
#   scripts/launch_followups.sh [logdir] [gpuA] [gpuB]
set -u
LOG=${1:-runs/launch_logs}; A=${2:-0}; B=${3:-1}
cd "$(dirname "$0")/.." && mkdir -p "$LOG"; source ~/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.2 PATH=/usr/local/cuda-13.2/bin:$PATH
while pgrep -f "train.py configs/launch_100k.yaml --condition (rnn|gru|transformer)" > /dev/null; do sleep 60; done
until [ -f graphs/cb10k/gate.json ]; do sleep 30; done
if ! python -c "import json,sys; g=json.load(open('graphs/cb10k/gate.json')); sys.exit(0 if all(g.values()) else 1)"; then
  echo "$(date -u +%FT%TZ) cb10k gate FAILED; skipping the 10k runs" >> "$LOG/queue.log"; TENK=0; else TENK=1; fi
run() { local gpu=$1 cfg=$2 c=$3 k=$4; CUDA_VISIBLE_DEVICES=$gpu python train.py "$cfg" --condition "$c" --seed "$k" > "$LOG/$(basename $cfg .yaml)_${c}_seed${k}.log" 2>&1
        echo "$(date -u +%FT%TZ) gpu$gpu finished $(basename $cfg .yaml) $c:$k exit=$?" >> "$LOG/queue.log"; }
( run "$A" configs/launch_100k.yaml frozen 1; run "$A" configs/launch_100k.yaml frozen 3; [ $TENK = 1 ] && run "$A" configs/scale_10k.yaml real 1 ) &
( run "$B" configs/launch_100k.yaml frozen 2; [ $TENK = 1 ] && run "$B" configs/scale_10k.yaml degree_preserving 1 ) &
wait
