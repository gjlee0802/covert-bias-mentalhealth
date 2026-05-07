#!/bin/bash

set -euo pipefail

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

DEVICE="${1:-0}"
MODEL_LIST=${2:-"openai:gpt-4.1-mini"}
MAX_PAIRS="${3:-500}"

for variable in top500_sae_sae top500_sae_aae
do
    for model in $MODEL_LIST
    do
        python3 -u ../probing/mgp.py \
        --model "$model" \
        --variable "$variable" \
        --attribute mental_attitudes \
        --device "$DEVICE" \
        --max_pairs "$MAX_PAIRS" \
        --calibrate
    done
done
