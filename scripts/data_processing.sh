export CUDA_VISIBLE_DEVICES=0

CONFIG_DIR=cfgs/gnn.yaml

# Extract train H&E image features (Slice A)
python datasets/processing/extract_image_features.py --config "${CONFIG_DIR}" --mode train

# Extract train H&E image features (Slice B)
python datasets/processing/extract_image_features.py --config "${CONFIG_DIR}" --mode test
