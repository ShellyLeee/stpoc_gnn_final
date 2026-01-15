import os
import yaml
import argparse
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import tifffile
import logging
import timm

try:
    from timm.models.layers.helpers import to_2tuple
except ImportError:
    from timm.layers import to_2tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 CTransPath Foundation Model
# =============================================================================

class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, **kwargs):
        super().__init__()
        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for _ in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def ctranspath():
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    return model


# =============================================================================
# Foundation Model Factory
# =============================================================================

class FoundationModelFactory:
    @staticmethod
    def get_model(config: dict, device: torch.device):
        """
        Return: model, transform, feature_dim, model_name
        """
        model_cfg = config.get("data", {})
        model_name = model_cfg.get("he_encoder_name", "resnet50").lower()

        pretrained_path = model_cfg.get("pretrained_path", None)

        logger.info(f"Initializing H&E Encoder: {model_name.upper()}...")

        # Normalization
        # if patch_size != 224, need additional resizing
        base_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 1) CTransPath
        if model_name == "ctranspath":
            if not pretrained_path or not os.path.exists(pretrained_path):
                raise ValueError(f"For CTransPath, you must provide a valid config['model']['pretrained_path']. Got: {pretrained_path}")

            logger.info(f"Loading CTransPath weights from: {pretrained_path}")
            model = ctranspath()
            model.head = nn.Identity()

            state_dict = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

            try:
                model.load_state_dict(state_dict, strict=True)
                logger.info("✅ CTransPath weights loaded successfully (Strict=True).")
            except Exception as e:
                logger.warning(f"⚠️ Strict loading failed: {e}")
                logger.warning("Trying strict=False ...")
                msg = model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded with strict=False. Missing keys: {len(msg.missing_keys)}")

            transform = base_norm
            feature_dim = 768

        # 2) UNI
        elif model_name == "uni":
            model = timm.create_model(
                "hf_hub:MahmoodLab/UNI",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True
            )
            transform = base_norm
            feature_dim = 1024

        # 3) UNI2-h
        elif model_name == "uni2-h":
            timm_kwargs = dict(
                img_size=224,
                patch_size=14,
                depth=24,
                num_heads=24,
                embed_dim=1536,
                mlp_ratio=2.66667 * 2,
                init_values=1e-5,
                reg_tokens=8,
                num_classes=0,
                no_embed_class=True,
                dynamic_img_size=True,
                mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
            model = timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h",
                pretrained=True,
                **timm_kwargs
            )
            transform = base_norm
            feature_dim = 1536

        # 4) GigaPath
        elif model_name == "gigapath":
            model = timm.create_model(
                "hf_hub:prov-gigapath/prov-gigapath",
                pretrained=True
            )
            transform = base_norm
            feature_dim = 1536

        # 5) ResNet50 (baseline)
        elif model_name == "resnet50":
            data_cfg = config.get("data", {})
            weights_path = model_cfg.get("pretrained_path", None) or data_cfg.get("pretrained_path", None)

            model = models.resnet50(weights=None)
            if weights_path and os.path.exists(weights_path):
                logger.info(f"Loading local ResNet weights from: {weights_path}")
                state_dict = torch.load(weights_path, map_location="cpu")
                new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)
            else:
                logger.info("Loading ResNet weights from TorchVision (Download)...")
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

            model.fc = nn.Identity()
            transform = base_norm
            feature_dim = 2048

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model = model.to(device)
        model.eval()
        return model, transform, feature_dim, model_name


# =============================================================================
# Dataset
# =============================================================================

class WSIPathPatchDataset(Dataset):
    def __init__(self, coords, image_path, patch_size=224, transforms=None):
        self.coords = coords
        self.patch_size = patch_size
        self.transforms = transforms
        self.half_size = patch_size // 2

        logger.info(f"Loading WSI from {image_path}...")
        self.image = tifffile.memmap(image_path)
        self.img_h, self.img_w = self.image.shape[:2]
        logger.info(f"WSI Shape: {self.image.shape}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        r, c = self.coords[idx]

        r_start = max(0, int(r) - self.half_size)
        r_end = min(self.img_h, int(r) + self.half_size)
        c_start = max(0, int(c) - self.half_size)
        c_end = min(self.img_w, int(c) + self.half_size)

        patch = self.image[r_start:r_end, c_start:c_end]

        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            pad_h = self.patch_size - patch.shape[0]
            pad_w = self.patch_size - patch.shape[1]
            patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

        patch = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0

        if self.transforms:
            patch = self.transforms(patch)

        return patch


# =============================================================================
# Extraction Runner (mode=train/test)
# =============================================================================

def run_extract_he(config_path: str, mode: str = "train"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    data_cfg = config.get("data", {})

    logger.info("Loading RNA data to retrieve coordinates...")

    if mode == "train":
        train_rna = sc.read_h5ad(data_cfg["train_rna_h5ad"])
        test_rna = sc.read_h5ad(data_cfg["test_rna_h5ad"])

        coords_train = np.column_stack([
            train_rna.obs["pxl_row_in_fullres"].values,
            train_rna.obs["pxl_col_in_fullres"].values
        ])
        coords_test = np.column_stack([
            test_rna.obs["pxl_row_in_fullres"].values,
            test_rna.obs["pxl_col_in_fullres"].values
        ])
        all_coords = np.vstack([coords_train, coords_test])

        image_path = data_cfg["he_image_path"]
        save_path = data_cfg["he_features_save_path"]

        logger.info(f"[mode=train] Total spots: {len(all_coords)} (Train: {len(coords_train)}, Test: {len(coords_test)})")
        logger.info(f"[mode=train] Using HE image: {image_path}")
        logger.info(f"[mode=train] Saving features to: {save_path}")

    else:
        final_test_rna = sc.read_h5ad(data_cfg["final_test_rna_h5ad"])
        all_coords = np.column_stack([
            final_test_rna.obs["pxl_row_in_fullres"].values,
            final_test_rna.obs["pxl_col_in_fullres"].values
        ])

        image_path = data_cfg["test_he_image_path"]
        save_path = data_cfg["test_he_features_path"]

        logger.info(f"[mode=test] Total spots: {len(all_coords)} (Final test only)")
        logger.info(f"[mode=test] Using HE image: {image_path}")
        logger.info(f"[mode=test] Saving features to: {save_path}")

    patch_size = data_cfg.get("patch_size", 224)

    # 1) Load foundation encoder + transform
    model, transform, feat_dim, model_name = FoundationModelFactory.get_model(config, device)
    logger.info(f"Encoder ready: {model_name}, feature_dim={feat_dim}")

    # 2) Dataset / Loader
    dataset = WSIPathPatchDataset(
        all_coords,
        image_path,
        patch_size=patch_size,
        transforms=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg.get("extract_batch_size", 128),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 8),
        pin_memory=True
    )

    # 3) TTA
    def apply_tta_and_extract(images):
        features_list = []
        for k in [0, 1, 2, 3]:
            img_rot = torch.rot90(images, k, [2, 3])

            feat = model(img_rot)
            features_list.append(feat)

            img_flip = torch.flip(img_rot, [3])
            feat_flip = model(img_flip)
            features_list.append(feat_flip)

        robust_feat = torch.stack(features_list).mean(dim=0)
        return robust_feat

    # 4) Extract
    logger.info("Starting feature extraction with multi-view TTA...")
    all_features = []
    with torch.no_grad():
        for batch_imgs in tqdm(dataloader, desc="Extracting"):
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            feats = apply_tta_and_extract(batch_imgs)
            all_features.append(feats.detach().cpu().numpy())

    final_features = np.vstack(all_features)
    logger.info(f"Extract done. Feature shape: {final_features.shape}")

    # 5) Save
    auto_suffix = config.get("data", {}).get("auto_suffix_modelname", True)
    if auto_suffix:
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}_{model_name}{ext}"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, final_features)

    logger.info(f"Done! Features saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="train mode uses he_image_path; test mode uses test_he_image_path (final_test only)"
    )
    args = parser.parse_args()
    run_extract_he(args.config, args.mode)
