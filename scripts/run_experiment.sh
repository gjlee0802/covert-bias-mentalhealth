#!/bin/bash

set -euo pipefail

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

DEVICE="${1:-0}"
MODEL_LIST=${2:-"openai:gpt-4.1-mini hf:meta-llama/Llama-3.1-8B-Instruct hf:Qwen/Qwen3-8B hf:mistralai/Mistral-7B-Instruct-v0.3"}
MAX_PAIRS="${3:-1000}"

for variable in mentalchat16k_sae_sae mentalchat16k_sae_aae
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
