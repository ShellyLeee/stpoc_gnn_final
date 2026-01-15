# Pretrained Image Encoder Weights

This directory stores pretrained weights for H&E image feature extraction.

## Environment Note (Important)

When configuring the environment:

- **For CTransPath only** — you MUST install `timm` exactly as required in the official GitHub repo:
```bash
pip install /mnt/sharedata/ssd_large/users/liyx/timm-0.5.4.tar   # ctranspath
```

For all other encoders, you can simply use:

```bash
pip install "timm==1.0.7"
```

Failing to follow this may cause import or checkpoint loading issues.


## Available Encoders

### 1. ResNet50 (Baseline)
- **File**: `resnet50-0676ba61.pth`
- **Source**: PyTorch ImageNet pretrained
- **Download**: 
```bash
  wget https://download.pytorch.org/models/resnet50-0676ba61.pth \
    -O data/pretrained_models/resnet50-0676ba61.pth
```
- **Feature Dim**: 2048
- **Config**: `he_encoder_name: "resnet50"`

### 2. CTransPath (Pathology Foundation Model)
- **File**: `ctranspath.pth`
- **Source**: [GitHub - Xiyue-Wang/TransPath](https://github.com/Xiyue-Wang/TransPath)
- **Download**: Follow instructions in the GitHub repo
- **Feature Dim**: 768
- **Config**: `he_encoder_name: "ctranspath"`
- **Note**: Requires manual download and accepts license agreement

### 3. UNI (Auto-downloaded)
- **No manual download required** - uses Hugging Face Hub
- **Feature Dim**: 1024
- **Config**: `he_encoder_name: "uni"`
- **First run will auto-download**: `hf_hub:MahmoodLab/UNI`

### 4. UNI2-h (Auto-downloaded, RECOMMENDED)
- **No manual download required** - uses Hugging Face Hub
- **Feature Dim**: 1536
- **Config**: `he_encoder_name: "uni2-h"`
- **First run will auto-download**: `hf-hub:MahmoodLab/UNI2-h`
- **Best performance** in our experiments

### 5. GigaPath (Auto-downloaded)
- **No manual download required** - uses Hugging Face Hub
- **Feature Dim**: 1536
- **Config**: `he_encoder_name: "gigapath"`
- **First run will auto-download**: `hf_hub:prov-gigapath/prov-gigapath`

## Quick Setup

### Option 1: Use Auto-Downloaded Models (Recommended)
```yaml
# In cfgs/gnn.yaml
data:
  he_encoder_name: "uni2-h"  # No manual download needed!
  pretrained_path: ""  # Leave empty for auto-downloaded models
```

### Option 2: Use Local ResNet50
```bash
# Download manually
wget https://download.pytorch.org/models/resnet50-0676ba61.pth \
  -O data/pretrained_models/resnet50-0676ba61.pth

# Update config
# he_encoder_name: "resnet50"
# pretrained_path: "data/pretrained_models/resnet50-0676ba61.pth"
```

**Note**: Auto-downloaded models are cached in `~/.cache/huggingface/hub/`

**Recommendation**: Use `uni2-h` for best performance.