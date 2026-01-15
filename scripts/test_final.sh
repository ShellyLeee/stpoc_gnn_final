#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

EXP_NAME="run_01"
CFG_DIR="logs/${EXP_NAME}/config.yaml"
LOG_DIR="logs"
RUN_DIR="${LOG_DIR}/${EXP_NAME}"
CHECKPOINT_PATH="${RUN_DIR}/checkpoints/gnn_best.pt"
OUTPUT_PATH="${RUN_DIR}/predictions_${EXP_NAME}_final.csv"

# Optional: set this if your cluster/job environment does not provide it.
# PROJECT_ROOT="/path/to/your/project"
# cd "${PROJECT_ROOT}"

python test_final.py \
  --config "${CFG_DIR}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --exp_name "${EXP_NAME}"
