#!/usr/bin/env bash
# Whole-CNS data-parallel queue (plan.md §10 / §16 step 14). Waits for the graph gate to pass and for the
# 5k launch runs to release the GPUs, then trains the real graph and the degree-preserving control one after
# the other on NGPU GPUs with torchrun (one model, NGPU × batch_size windows per step).
#   scripts/launch_cns_ddp.sh [configs/full_cns.yaml] [logdir] [ngpu]
set -u
CFG=${1:-configs/full_cns.yaml}
LOG=${2:-runs/launch_logs}
NGPU=${3:-8}
cd "$(dirname "$0")/.." && mkdir -p "$LOG"
source ~/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.2 PATH=/usr/local/cuda-13.2/bin:$PATH
GRAPH=$(python -c "from flygpt import Config; print(Config.load('$CFG').graph_dir)")
until [ -f "$GRAPH/gate.json" ]; do sleep 60; done
if ! python -c "import json,sys; g=json.load(open('$GRAPH/gate.json')); sys.exit(0 if g.get('real') and g.get('degree_preserving_seed1') else 1)"; then
  echo "$(date -u +%FT%TZ) $GRAPH gate FAILED; not training" >> "$LOG/queue.log"; exit 1
fi
# GPUs are pinned by the caller via CUDA_VISIBLE_DEVICES; no wait for other runs
for c in real degree_preserving; do
  echo "$(date -u +%FT%TZ) starting cns $c on $NGPU GPUs" >> "$LOG/queue.log"
  torchrun --nproc_per_node "$NGPU" --master_port 29601 train.py "$CFG" --condition "$c" --seed 1 > "$LOG/cns_${c}_seed1.log" 2>&1
  echo "$(date -u +%FT%TZ) finished cns $c exit=$?" >> "$LOG/queue.log"
done
