"""
Protein Co-expression Regularization Module.

Loads co-expression matrix and computes regularization loss to constrain
predictions based on biological prior knowledge.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, Optional


class CoexpressionRegularizer(nn.Module):
    """
    Regularization based on protein co-expression relationships.
    
    Encourages model predictions to respect known biological co-expression patterns.
    """
    
    def __init__(
        self,
        coexp_file_path: str,
        protein_names: list,
        device: str = "cuda",
        threshold: float = 0.0,
        logger=None,
        mode: str = "pearson"
    ):
        """
        Args:
            coexp_file_path: Path to protein_coexpression_matrix.txt
            protein_names: List of protein names in the model output order
            device: Device to store tensors
            threshold: Minimum coexpression value to consider (filter weak correlations)
            logger: Logger instance
        """
        super().__init__()
        self.logger = logger
        self.device = device
        self.protein_names = protein_names
        self.num_proteins = len(protein_names)
        self.mode = mode
        
        # Load and build co-expression matrix
        self.coexp_matrix, self.valid_pairs = self._load_coexpression_matrix(
            coexp_file_path, threshold
        )
        
        if self.logger:
            self.logger.info(f"Loaded {len(self.valid_pairs)} valid co-expression pairs "
                           f"(threshold >= {threshold})")
    
    def _load_coexpression_matrix(
        self, 
        file_path: str, 
        threshold: float
    ) -> tuple:
        """
        Load co-expression data and build matrix.
        
        Returns:
            coexp_matrix: (C, C) tensor of co-expression values
            valid_pairs: List of (i, j, weight) tuples for non-zero pairs
        """
        if not os.path.exists(file_path):
            if self.logger:
                self.logger.warning(f"Co-expression file not found: {file_path}")
            return torch.zeros(self.num_proteins, self.num_proteins), []
        
        # Read co-expression data
        df = pd.read_csv(file_path, sep='\t')
        
        # Create protein name to index mapping
        protein_to_idx = {name: idx for idx, name in enumerate(self.protein_names)}
        
        # Initialize matrix
        coexp_matrix = torch.zeros(self.num_proteins, self.num_proteins)
        valid_pairs = []
        
        skipped = 0
        for _, row in df.iterrows():
            prot1 = row['match_protein_names1']
            prot2 = row['match_protein_names2']
            coexp_val = float(row['coexpression'])
            
            # Filter by threshold
            if abs(coexp_val) < threshold:
                continue
            
            # Check if both proteins are in our model
            if prot1 in protein_to_idx and prot2 in protein_to_idx:
                idx1 = protein_to_idx[prot1]
                idx2 = protein_to_idx[prot2]
                
                # Symmetric matrix
                coexp_matrix[idx1, idx2] = coexp_val
                coexp_matrix[idx2, idx1] = coexp_val
                
                valid_pairs.append((idx1, idx2, coexp_val))
                if idx1 != idx2:  # Add reverse pair if not self-loop
                    valid_pairs.append((idx2, idx1, coexp_val))
            else:
                skipped += 1
        
        if self.logger and skipped > 0:
            self.logger.info(f"Skipped {skipped} co-expression pairs (proteins not in model)")
        
        return coexp_matrix.to(self.device), valid_pairs
    
    def forward(
        self, 
        predictions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute co-expression regularization loss.
        
        Loss encourages predictions of co-expressed proteins to be correlated.
        
        Args:
            predictions: (N, C) predicted protein expressions
            mask: (N,) boolean mask for valid samples
            
        Returns:
            Regularization loss (scalar)
        """
        if len(self.valid_pairs) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Apply mask if provided
        if mask is not None:
            predictions = predictions[mask]
        
        if predictions.size(0) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Compute correlation-based loss
        loss = 0.0
        count = 0
        
        for idx1, idx2, coexp_weight in self.valid_pairs:
            # Get predictions for the two proteins
            pred1 = predictions[:, idx1]  # (N,)
            pred2 = predictions[:, idx2]  # (N,)
            
            # Compute Pearson correlation between predictions
            pred_corr = self._pearson_correlation(pred1, pred2)
            
            # Loss: penalize deviation from expected co-expression
            # If coexp_weight > 0, encourage positive correlation
            # If coexp_weight < 0, encourage negative correlation
            loss += (pred_corr - coexp_weight) ** 2
            count += 1
        
        return loss / max(count, 1)
    
    def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute differentiable Pearson correlation."""
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
        
        return numerator / (denominator + 1e-8)
    
    def _kernel_alignment(self, K1: torch.Tensor, K2: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
        a = K1[mask]
        b = K2[mask]
        num = torch.sum(a * b)
        denom = torch.sqrt(torch.sum(a * a) * torch.sum(b * b)) + eps
        return num / denom


class CoexpressionRegularizerV2(CoexpressionRegularizer):
    """
    Alternative implementation using matrix operations (faster for many pairs).
    """
    
    def forward(
        self, 
        predictions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute co-expression regularization using matrix operations.
        
        Args:
            predictions: (N, C) predicted protein expressions
            mask: (N,) boolean mask
            
        Returns:
            Regularization loss (scalar)
        """
        if len(self.valid_pairs) == 0:
            return torch.tensor(0.0, device=self.device)
        
        if mask is not None:
            predictions = predictions[mask]
        
        if predictions.size(0) < 2:
            return torch.tensor(0.0, device=self.device)
        
        valid_mask = self.coexp_matrix != 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        mode = getattr(self, "mode", "pearson")

        if mode == "pearson":
            pred_corr_matrix = self._compute_correlation_matrix(predictions)  # (C,C)
            return ((pred_corr_matrix - self.coexp_matrix) ** 2)[valid_mask].mean()

        elif mode == "kernel":
            # --- Prediction-side kernel: distance -> RBF -> [0,1] (matches STRING prior construction) ---
            F = predictions  # (B, C)
            F = F - F.mean(dim=0, keepdim=True)  # optional but stabilizes distances

            # Pairwise squared distance between protein vectors (columns of F)
            # D2_ij = ||F[:,i] - F[:,j]||^2
            G = F.T @ F                          # (C, C)
            sq = torch.diag(G).unsqueeze(1)      # (C, 1)
            D2 = sq - 2.0 * G + sq.T             # (C, C)
            D2 = torch.clamp(D2, min=0.0)

            # Use mask-selected median heuristic for sigma^2 (robust, scale-adaptive)
            # valid_mask selects only edges kept by threshold in prior
            sigma2 = torch.median(D2[valid_mask]).clamp_min(1e-6)

            K_pred = torch.exp(-D2 / (2.0 * sigma2))  # (C, C) in (0,1]

            K0 = self.coexp_matrix  # (C, C) prior kernel in [0,1]

            # Option A: MSE in kernel space (most direct)
            return ((K_pred - K0) ** 2)[valid_mask].mean()

            # Option B (if you prefer alignment objective instead of MSE):
            # return -self._kernel_alignment(K_pred, K0, valid_mask)

        else:
            raise ValueError(f"Unknown coexpression.mode: {mode}")

    
    def _compute_correlation_matrix(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Pearson correlation matrix.
        
        Args:
            predictions: (N, C)
            
        Returns:
            corr_matrix: (C, C)
        """
        # Center predictions
        pred_centered = predictions - predictions.mean(dim=0, keepdim=True)  # (N, C)
        
        # Compute covariance matrix
        cov_matrix = torch.mm(pred_centered.T, pred_centered) / (predictions.size(0) - 1)  # (C, C)
        
        # Compute standard deviations
        std_devs = torch.sqrt(torch.diag(cov_matrix))  # (C,)
        
        # Compute correlation matrix
        corr_matrix = cov_matrix / (std_devs.unsqueeze(1) * std_devs.unsqueeze(0) + 1e-8)
        
        return corr_matrix


def build_coexpression_regularizer(
    config: dict,
    protein_names: list,
    device: str,
    logger=None
) -> Optional[CoexpressionRegularizer]:
    """
    Factory function to create co-expression regularizer.
    
    Args:
        config: Configuration dictionary
        protein_names: List of protein names
        device: Device string
        logger: Logger instance
        
    Returns:
        Regularizer instance or None if disabled
    """
    coexp_cfg = config.get('coexpression', {})
    
    if not coexp_cfg.get('enabled', False):
        if logger:
            logger.info("Co-expression regularization: DISABLED")
        return None
    
    mode = coexp_cfg.get("mode", "pearson")
    file_path = coexp_cfg.get('matrix_path')
    threshold = coexp_cfg.get('threshold', 0.0)
    use_matrix_version = coexp_cfg.get('use_matrix_version', True)
    
    if not file_path:
        if logger:
            logger.warning("Co-expression enabled but matrix_path not provided")
        return None
    
    if logger:
        logger.info("=" * 60)
        logger.info("Building Co-expression Regularizer")
        logger.info("=" * 60)
        logger.info(f"Matrix path: {file_path}")
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Version: {'Matrix (V2)' if use_matrix_version else 'Pairwise (V1)'}")
    
    RegularizerClass = CoexpressionRegularizerV2 if use_matrix_version else CoexpressionRegularizer
    
    regularizer = RegularizerClass(
        coexp_file_path=file_path,
        protein_names=protein_names,
        device=device,
        threshold=threshold,
        logger=logger,
        mode=mode,
    )
    
    return regularizer