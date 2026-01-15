#!/usr/bin/env python
"""
Mini-batch training for large-scale GNN (100k+ nodes) with Multimodal Fusion.
H&E Image + RNA -> Protein Prediction

Integrated Regularization Revised Version:
1. Attention Dropout: 0.2 (Passed to Model)
2. Edge Dropout: 0.15 (Passed to Model)
3. Smoothness Weight: 0.01 (Enhanced Logging)

⚠️ Prerequisite:
Before running this script, ensure H&E image features are extracted and saved.
Refer to: processing.extract_image_features

Usage Example:
nohup python -m train \
    --config gnn_official/cfgs/gnn_he.yaml \
    --exp_name multimodal_gat_run_08 \
    > log/train_multimodal_08.log 2>&1 &
"""

import random
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from scipy.stats import spearmanr

class BaseTrainer:

    def __init__(self, model, criterion, optimizer, device, config, rna_dim, logger=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.rna_dim = rna_dim
        self.logger = logger

        training_cfg = config.get("training", {})
        self.use_masking = training_cfg.get("use_modality_masking", False)
        self.noise_std = training_cfg.get("input_noise_std", 0.0)
        self.p_drop_rna = training_cfg.get("p_drop_rna", 0.0)
        self.p_drop_img = training_cfg.get("p_drop_img", 0.0)
        self.smoothness_w = training_cfg.get("smoothness_weight", 0.01)
        self.smoothness_mode = training_cfg.get("smoothness_mode", "edge")
        self.smoothness_gamma = float(training_cfg.get("smoothness_gamma", 2.0))
        self.smoothness_threshold = float(training_cfg.get("smoothness_threshold", 0.0))
        self.cell_dim = int(training_cfg.get("cell_dim", 20))
        self.coexp_regularizer = None
        self.coexp_weight = 0.0
        coexp_cfg = config.get('coexpression', {})
        if coexp_cfg.get('enabled', False):
            from common.coexpression import build_coexpression_regularizer
            self.coexp_weight = coexp_cfg.get('weight', 0.1)
            if logger:
                logger.info(f"Co-expression regularization ENABLED (weight={self.coexp_weight})")


    def train(self, loader, epoch):
        """Train one epoch with mini-batches and multimodal augmentation."""
        self.model.train()
        total_task_loss = 0.0
        total_smooth_loss = 0.0
        total_coexp_loss = 0.0
        total_combined_loss = 0.0
        total_samples = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        for batch in pbar:
            metrics = self.train_one_batch(batch)
            n_samples = metrics["n_samples"]
            total_samples += n_samples
            total_task_loss += metrics["task_loss"] * n_samples
            total_smooth_loss += metrics["smooth_loss"] * n_samples
            total_coexp_loss += metrics["coexp_loss"] * n_samples
            total_combined_loss += metrics["combined_loss"] * n_samples

            if total_samples > 0:
                pbar.set_postfix(
                    task=f"{total_task_loss/total_samples:.4f}",
                    smooth=f"{total_smooth_loss/total_samples:.4f}",
                    coexp=f"{total_coexp_loss/total_samples:.4f}",
                    comb=f"{total_combined_loss/total_samples:.4f}",
                )

        return total_combined_loss / max(total_samples, 1)

    def train_one_batch(self, batch):
        batch = batch.to(self.device)
        self.optimizer.zero_grad()

        x_input = self._apply_augmentation(batch.x)
        out = self.model(x_input, batch.edge_index, batch.edge_attr)

        train_mask = batch.train_mask[: batch.batch_size]
        n_samples = train_mask.sum().item()
        if n_samples == 0:
            return {"task_loss": 0.0, "smooth_loss": 0.0, "combined_loss": 0.0, "n_samples": 0}

        task_loss = self.criterion(out[: batch.batch_size], batch.y[: batch.batch_size], train_mask)
        smooth_loss = self._smoothness_loss(out, batch)
        coexp_loss = torch.tensor(0.0, device=self.device)
        if self.coexp_regularizer is not None:
            coexp_loss = self.coexp_regularizer(out[: batch.batch_size], train_mask)

        combined_loss = task_loss + self.smoothness_w * smooth_loss + self.coexp_weight * coexp_loss

        combined_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "task_loss": task_loss.item(),
            "smooth_loss": smooth_loss.item(),
            "coexp_loss": coexp_loss.item(),
            "combined_loss": combined_loss.item(),
            "n_samples": n_samples,
        }

    @torch.no_grad()
    def validate(self, loader, epoch):
        """Validate with mini-batches."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Val]  ", leave=False)
        for batch in pbar:
            loss, n_samples = self.validate_batch(batch)
            total_loss += loss * n_samples
            total_samples += n_samples

            if total_samples > 0:
                pbar.set_postfix(loss=f"{total_loss/total_samples:.5f}")

        return total_loss / max(total_samples, 1)

    @torch.no_grad()
    def validate_batch(self, batch):
        batch = batch.to(self.device)
        out = self.model(batch.x, batch.edge_index, batch.edge_attr)

        val_mask = batch.val_mask[: batch.batch_size]
        n_samples = val_mask.sum().item()
        if n_samples == 0:
            return 0.0, 0

        task_loss = self.criterion(out[: batch.batch_size], batch.y[: batch.batch_size], val_mask)
        smooth_loss = self._smoothness_loss(out, batch)
        coexp_loss = torch.tensor(0.0, device=self.device)
        if self.coexp_regularizer is not None:
            coexp_loss = self.coexp_regularizer(out[: batch.batch_size], val_mask)

        combined_loss = task_loss + self.smoothness_w * smooth_loss + self.coexp_weight * coexp_loss
        return combined_loss.item(), n_samples

    def _apply_augmentation(self, x_input):
        if not self.use_masking:
            return x_input

        if self.noise_std > 0:
            noise = torch.randn_like(x_input) * self.noise_std
            x_input = x_input + noise

        rand_val = random.random()
        mask = torch.ones_like(x_input)
        if rand_val < self.p_drop_rna:
            mask[:, : self.rna_dim] = 0.0
        elif rand_val < (self.p_drop_rna + self.p_drop_img):
            mask[:, self.rna_dim :] = 0.0

        return x_input * mask

    def _smoothness_loss(self, out, batch):
        """
        Smoothness regularizer.
        mode="edge": original unweighted smoothness on edges
        mode="cell_cosine": cell-aware weighted smoothness using cosine similarity of cell abundance vectors
        """
        if self.smoothness_w <= 0:
            return torch.tensor(0.0, device=self.device)

        row, col = batch.edge_index  # [2, E]
        diff = out[row] - out[col]
        diff2 = torch.sum(diff ** 2, dim=1)  # [E]

        # ---- (A) Original edge smoothness ----
        if self.smoothness_mode == "edge":
            return torch.mean(diff2)

        # ---- (B) Cell-aware smoothness ----
        if self.smoothness_mode == "cell_cosine":
            # If cell features are not used, fallback to edge mode
            if not self.config.get("data", {}).get("use_cell_features", False):
                return torch.mean(diff2)

            # Cell features are concatenated at the END in datasets/utils.py (after HE if present)
            # Use batch.x (NOT augmented x_input) to compute stable weights
            x = batch.x
            if x.size(1) < self.cell_dim:
                # Fallback if something is wrong with dims
                return torch.mean(diff2)

            cell = x[:, -self.cell_dim:]  # [N, cell_dim]

            # L2 normalize
            cell = torch.nn.functional.normalize(cell, p=2, dim=1, eps=1e-8)

            ci = cell[row]  # [E, cell_dim]
            cj = cell[col]  # [E, cell_dim]

            # cosine similarity in [-1, 1]
            cos = torch.sum(ci * cj, dim=1)  # [E]

            # only regularize "similar" edges: cos > threshold
            # weight = max(0, cos - threshold)
            w = torch.clamp(cos - self.smoothness_threshold, min=0.0)

            # gamma sharpening (1~4)
            if self.smoothness_gamma != 1.0:
                w = w ** self.smoothness_gamma

            # Normalize by total weight to stabilize scale across batches
            denom = w.sum() + 1e-8
            if denom.item() <= 1e-8:
                # no effective edges
                return torch.tensor(0.0, device=self.device)

            return torch.sum(w * diff2) / denom

        # Unknown mode -> fallback
        return torch.mean(diff2)



@torch.no_grad()
def evaluate_spearman(model, loader, device, C_out, mask_name, epoch):
    """Evaluate mean Spearman correlation using mini-batches."""
    model.eval()
    all_preds = []
    all_true = []
    
    desc = f"Epoch {epoch:02d} [{mask_name.split('_')[0].capitalize()} Eval]"
    pbar = tqdm(loader, desc=desc, leave=False)

    for batch in pbar:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        mask_batch = getattr(batch, mask_name)[:batch.batch_size]
        
        if mask_batch.sum() > 0:
            all_preds.append(out[:batch.batch_size][mask_batch].cpu().numpy())
            all_true.append(batch.y[:batch.batch_size][mask_batch].cpu().numpy())
    
    if len(all_preds) == 0: return 0.0

    all_preds = np.vstack(all_preds)
    all_true = np.vstack(all_true)

    spearman_rhos = []
    for j in range(C_out):
        try:
            rho, _ = spearmanr(all_true[:, j], all_preds[:, j])
            spearman_rhos.append(rho)
        except:
            spearman_rhos.append(np.nan)

    return np.nanmean(spearman_rhos)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-5, verbose=True, logger=None):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.logger = logger
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose and self.logger:
                self.logger.debug(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

