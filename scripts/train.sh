export CUDA_VISIBLE_DEVICES=0

EXP_NAME="run_01"

CONFIG_DIR=cfgs/gnn.yaml

python train.py \
  --config "${CONFIG_DIR}" \
  --exp_name "${EXP_NAME}"
