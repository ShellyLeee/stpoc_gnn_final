# Multimodal GNN for Spatial Transcriptomics-to-Proteomics Prediction

A comprehensive Graph Neural Network framework that integrates **Spatial Transcriptomics (RNA)**, **Histology Images (H&E)**, and **Cell Abundance Features** to predict spatial protein expression at cellular resolution.

The solution ranked **1st 🏆** in the development phase and **2nd 🥈** in the testing phase of the [STP Open Challenge on Codabench](https://www.codabench.org/competitions/10696/) (participant: liyx2022). Results are available on the [competition leaderboard](https://www.codabench.org/competitions/10696/#/results-tab).

## Environment Setup

**Requirements:** Ubuntu 22.04.5 LTS, Python 3.10, CUDA 12.1, RTX 4090 (24 GB VRAM)

### 1. Clone Project

```bash
# Clone repository
git clone https://github.com/shellyleee/stpoc_gnn.git
cd stpoc_gnn
```

### 2. Create Conda Environment

```bash
# Create environment
conda create -n stp_gnn python=3.10 -y
conda activate stp_gnn
```

### 3. Install Core Dependencies
```bash
# Spatial omics packages
pip install scanpy anndata h5py numpy pandas matplotlib seaborn

# PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

# ML utilities
pip install scikit-learn tqdm jupyter ipykernel scipy pyyaml

# Register kernel for Jupyter
python -m ipykernel install --user --name stp_gnn --display-name "STP Challenge"
```

### 4. Install PyTorch Geometric

```bash
# Verify PyTorch installation
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

# Install PyG and extensions (choose version based the environment)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

### 5. Install Foundation Model Libraries (for H&E feature extraction)

``` bash
# TIMM for foundation models (timm==1.0.7 for all FMs except ctranspath)
pip install timm

# Optional: Hugging Face Hub for model downloads
pip install huggingface_hub
```

## Data Preparation

Download the challenge dataset from [Google Drive](https://drive.google.com/drive/folders/1eq6sbTUaWCCOKcnkei6B65rozx-VX70K).

Place data files in the `data/` directory:

```
data/
├── train_rna.h5ad                    # Training RNA expression (Slice A)
├── train_pro.h5ad                    # Training protein expression (Slice A)
├── valid_rna.h5ad                    # Validation RNA (Slice A internal split)
├── test_rna.h5ad                     # Test RNA (Slice B - external validation)
├── HE_image_full_resolution.tif      # H&E image (Slice A)
├── test_HE_image_full_resolution.tif # H&E image (Slice B)
└── bio_info/                         # Cell abundance and co-expression data
    ├── cell2location_predicted_cell_abundance_mean_train.csv
    ├── cell2location_predicted_cell_abundance_mean_valid.csv
    ├── cell2location_predicted_cell_abundance_mean_test.csv
    └── protein_coexpression_matrix.txt
```
## 🚀 Reproduce Best Results (Quick Inference)

We provide pre-trained model weights and preprocessed features to reproduce our best leaderboard results (testing phase) directly.

**Note on Reproducibility**:
All model hyperparameters and random seeds are fixed and stored in `logs/best_run/config.yaml`. This controls all major sources of randomness in data preprocessing, training, and inference.

Due to floating-point arithmetic and GPU-level non-determinism, repeated inference runs may still exhibit minor numerical differences (<5%), which do not affect ranking-based evaluation metrics such as Spearman correlation.

### Download Pre-trained Checkpoints

Download the pre-trained model package `best_run` from [Google Drive](https://drive.google.com/drive/folders/1ww_mRSO0mRrSCqaWFgvZ9XO7xyPBlUeQ?usp=sharing):
```
best_run/
├── checkpoints/
│   ├── gnn_best.pt              # Best GNN model weights
│   └── mlp_reducer_best.pt      # Trained MLP reducer
├── config.yaml                  # Best configuration used for training
├── he_scaler.joblib             # H&E feature scaler
└── cell_mean_scaler.joblib      # Cell abundance scaler
```

After downloading, extract the archive and place the `best_run/` directory under `logs/`.

### Quick Inference with Pre-trained Model

**Step 1: Prepare H&E features for test set**

To ensure fast and reliable reproduction of our best leaderboard results, we provide two options:

#### **Option 1: Download Pre-extracted Features (Recommended ⭐)**

Download our pre-extracted H&E features in the test set: `data/extracted_features/test_he_features_robust_all.npy` from the [Google Drive](https://drive.google.com/drive/folders/1ww_mRSO0mRrSCqaWFgvZ9XO7xyPBlUeQ?usp=sharing) and placing it under `data/extracted_features/`. This allows you to skip H&E feature extraction entirely and directly run inference using the pre-trained model and fixed configuration in `logs/best_run/config.yaml`.

**Skip to Step 2** after placing the features.

#### **Option 2: Extract Features Yourself**

If you prefer to extract H&E features from scratch:

Download the ImageNet pretrained ResNet50 weights at [here](https://download.pytorch.org/models/resnet50-0676ba61.pth), and placed it under `data/pretrained_models/`.

Then, run feature extractions by the command: 

```bash
export CUDA_VISIBLE_DEVICES=0
python datasets/processing/extract_image_features.py \
    --config logs/best_run/config.yaml \
    --mode test
```

**⏱️ Expected Runtime**: This extraction step may take **40–60 minutes on GPU** (or longer on CPU), depending on the number of spots and hardware.

**Step 2: Run inference on final test set**:
```bash
export CUDA_VISIBLE_DEVICES=0

python test_final.py \
    --config logs/best_run/config.yaml \
    --checkpoint logs/best_run/checkpoints/gnn_best.pt \
    --output logs/best_run/predictions_best.csv \
    --exp_name best_run
```

The generated prediction on the test data will be saved in `logs/best_run/predictions_best.csv`.

**⏱️ Expected Runtime**: Inference takes around 1 minute.


## 🛠️ Training from Scratch

You can follow the steps below if you want to train the model from scratch or experiment with different configurations.

### Step 1: Extract H&E Image Features

First, place the ImageNet pretrained ResNet50 weights (`resnet50-0676ba61.pth`) under `data/pretrained_models/`.

Then, edit `scripts/data_processing.sh` to comment out the test mode command, keeping only the train mode command.

Run the following command to extract robust multi-view features from H&E images using foundation models:

``` bash
bash scripts/data_processing.sh
```

The H&E features will be saved to `data/extracted_features/`.

### Step 2: Train the Model

Train the multimodal GNN with all features:

```bash
bash scripts/train.sh
```

Training outputs are saved to `logs/run_01/`:

- `checkpoints/gnn_best.pt` - Best model checkpoint
- `checkpoints/mlp_reducer_best.pt` - Trained MLP reducer
- `config.yaml` - Configuration used for training
- `metrics.json` - Training metrics and validation scores
- `he_scaler.joblib` - H&E feature scaler
- `cell_mean_scaler.joblib` - Cell abundance scaler
- `training_history.png` - Loss curves
- `spearman_history.png` - Correlation progression

### Step 3: Generate Predictions

Run inference on the final test set (Slice B):

```bash
bash scripts/test_final.sh
```

The prediction will be saved to `logs/run_01/predictions_run_01_final.csv`.