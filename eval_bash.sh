#!/usr/bin/env bash
set -euo pipefail

# Evaluate all generation folders under baseline_generations.
# Usage:
#   ./eval_bash.sh [BASE_DIR]
# Env options:
#   JOBS=N                 # run up to N evaluations in parallel (default: 1)
#   GPU_GROUPS="0,1 2,3"    # optional space-separated CUDA device groups, round-robin per job

# BASE_DIR=${1:-"./baseline_generations"}
# JOBS=${JOBS:-1}
# GPU_GROUPS=${GPU_GROUPS:-}


BASE_DIR="./baseline_generations"
JOBS=3
GPU_GROUPS="0,1 2,3 4,5"

if [ ! -d "$BASE_DIR" ]; then
  echo "Base directory not found: $BASE_DIR" >&2
  exit 1
fi

# Collect immediate subdirectories (model generation folders)
mapfile -t FOLDERS < <(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -not -name ".cache" -not -name "GPT4o_generation_1024" -printf "%P\n" | sort)
if [ ${#FOLDERS[@]} -eq 0 ]; then
  echo "No generation folders found in: $BASE_DIR"
  exit 0
fi

echo "Found ${#FOLDERS[@]} folders under $BASE_DIR"

# Parse GPU groups into an array if provided
read -r -a GPU_ARR <<< "$GPU_GROUPS"
GPU_COUNT=${#GPU_ARR[@]}

idx=0
active=0
pids=()

run_one() {
  local folder="$1"
  local gpu="$2"
  local full_path="$BASE_DIR/$folder"
  echo "[eval] $folder (GPU: ${gpu:-none})"
  if [ -n "$gpu" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" python eval.py --generation_folder "$full_path"
  else
    python eval.py --generation_folder "$full_path"
  fi
}

for folder in "${FOLDERS[@]}"; do
  gpu=""
  if [ $GPU_COUNT -gt 0 ]; then
    gpu="${GPU_ARR[$((idx % GPU_COUNT))]}"
  fi
  if [ "$JOBS" -gt 1 ]; then
    run_one "$folder" "$gpu" &
    pids+=("$!")
    active=$((active + 1))
    # Limit to JOBS parallel processes
    while [ "$active" -ge "$JOBS" ]; do
      if wait -n; then
        active=$((active - 1))
      fi
    done
  else
    run_one "$folder" "$gpu"
  fi
  idx=$((idx + 1))
done

# Wait for remaining background jobs
if [ "$JOBS" -gt 1 ]; then
  wait
fi

echo "All evaluations complete."

CUDA_VISIBLE_DEVICES=0,1 python eval.py --generation_folder "baseline_generations/AnyEdit_generation"
