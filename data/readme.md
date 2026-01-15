# Data Directory Structure

This directory should contain all required data files for the STP Challenge.

## Required Files (Download from Competition)

Download all files from the [competition website](https://www.codabench.org/competitions/10696/)

### Core Data Files
Place directly in this `data/` directory:

- `train_rna.h5ad` - Training RNA expression (required)
- `train_pro.h5ad` - Training protein expression (required)  
- `valid_rna.h5ad` - Validation RNA (internal split, required)
- `test_rna.h5ad` - Final test RNA (external slice, required)
- `HE_image_full_resolution.tif` - H&E image for training slice (~2-5GB)
- `test_HE_image_full_resolution.tif` - H&E image for test slice (~2-5GB)

### Expected Directory Structure
```
data/
├── README.md                          (this file)
├── train_rna.h5ad                    
├── train_pro.h5ad                    
├── test_rna.h5ad                     
├── final_test_rna.h5ad               
├── HE_image_full_resolution.tif      
├── test_HE_image_full_resolution.tif 
│
├── bio_info/                         (biological priors)
│   ├── README.md
│   ├── cell2location_predicted_cell_abundance_mean_train.csv
│   ├── cell2location_predicted_cell_abundance_mean_valid.csv
│   ├── cell2location_predicted_cell_abundance_mean_test.csv
│   └── protein_coexpression_matrix.txt
│
├── pretrained_models/                (image encoder weights)
│   ├── README.md
│   ├── resnet50-0676ba61.pth
│   └── ctranspath.pth
│
└── extracted_features/               (auto-generated)
    ├── he_features_robust_all_uni2-h.npy
    └── test_he_features_robust_all_uni2-h.npy
```

## Optional Files

### 1. Cell Abundance Features (`bio_info/`)
Cell type deconvolution results from cell2location or similar tools.  
**Status**: Recommended (improves performance ~2-5%)  
**Source**: Run cell2location on your own, or request from competition organizers

### 2. Protein Co-expression Matrix (`bio_info/`)
Protein-protein interaction prior from STRING database.  
**Status**: Optional (provides biological regularization)  
**Source**: [STRING database](https://string-db.org/) or download our processed version

### 3. Pretrained Image Encoders (`pretrained_models/`)
Required only if using specific H&E encoders:

| Model | Required For | Download Link |
|-------|-------------|---------------|
| ResNet50 | `he_encoder_name: "resnet50"` | [PyTorch Hub](https://download.pytorch.org/models/resnet50-0676ba61.pth) |
| CTransPath | `he_encoder_name: "ctranspath"` | [GitHub Release](https://github.com/Xiyue-Wang/TransPath) |
| UNI/UNI2-h | `he_encoder_name: "uni*"` | Auto-downloaded via Hugging Face |
| GigaPath | `he_encoder_name: "gigapath"` | Auto-downloaded via Hugging Face |

## Auto-Generated Directories

These will be created automatically during runtime:
- `extracted_features/` - Extracted H&E features (created by `extract_image_features.py`)
- `../logs/` - Training logs and checkpoints

## Quick Start Checklist

- [ ] Downloaded all 6 core data files from competition
- [ ] Placed them in `data/` directory
- [ ] Check cell abundance CSVs in `data/bio_info/`
- [ ] Downloaded pretrained model weights to `data/pretrained_models/`
- [ ] Updated `cfgs/gnn.yaml` with correct file paths
- [ ] Ready to run: `bash scripts/data_processing.sh`