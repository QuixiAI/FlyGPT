#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
curl -fsSL -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
wc -c input.txt
