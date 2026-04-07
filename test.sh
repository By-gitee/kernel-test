#!/bin/bash

M_LIST=(1 4 16 32 64 128)

# ---------------------------
# Group 1: Attention (K = N)
# ---------------------------
ATTN_SIZES=(
  "512 512"
  "1024 1024"
  "2048 2048"
  "4096 4096"
  "5120 5120"
)

# ---------------------------
# Group 2: FFN gate/up
# ---------------------------
FFN_UP_SIZES=(
  "512 2048"
  "1024 4096"
  "2048 8192"
  "4096 11008"
  "5120 13824"
)

# ---------------------------
# Group 3: FFN down
# ---------------------------
FFN_DOWN_SIZES=(
  "2048 512"
  "4096 1024"
  "8192 2048"
  "11008 4096"
  "13824 5120"
)

run_group () {
  local GROUP_NAME=$1
  shift
  local SIZES=("$@")

  echo "=============================="
  echo "Running group: $GROUP_NAME"
  echo "=============================="

  for size in "${SIZES[@]}"; do
    read K N <<< "$size"
    for M in "${M_LIST[@]}"; do
      echo ">>> M=$M K=$K N=$N"
      python run_all_benchmarks.py --M $M --K $K --N $N
    done
  done
}

run_group "Attention" "${ATTN_SIZES[@]}"
run_group "FFN Gate/Up" "${FFN_UP_SIZES[@]}"
run_group "FFN Down" "${FFN_DOWN_SIZES[@]}"

echo "All benchmarks completed."