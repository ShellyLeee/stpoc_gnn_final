"""
Trainable multimodal fusion modules for use inside the model.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CellAffine(nn.Module):
    """
    Per-cell-type affine calibration: c' = gamma * c + beta
    Keeps semantics (20 dims stay 20 dims), only re-weights each cell type.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma + self.beta


class FusionStrategy(nn.Module):
    """Base class for fusion strategies."""

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger

    def log(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def fuse(self, rna_feat: torch.Tensor, he_feat: Optional[torch.Tensor] = None, cell_feat: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def forward(
        self, rna_feat: torch.Tensor, he_feat: Optional[torch.Tensor] = None, cell_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Alias to support torch-style calling."""
        return self.fuse(rna_feat, he_feat, cell_feat)

    def fuse_with_slices(
        self, features: torch.Tensor, slices: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Convenience wrapper that splits a combined feature tensor into modalities
        using provided slices before calling the underlying fuse implementation.
        """
        if not slices:
            return self.fuse(features, None, None)

        rna_slice = slices.get("rna")
        he_slice = slices.get("he")
        cell_slice = slices.get("cell")

        rna = features[:, rna_slice[0] : rna_slice[1]] if rna_slice else features

        he = None
        if he_slice:
            he_dim = he_slice[1] - he_slice[0]
            if he_dim > 0:
                he = features[:, he_slice[0]:he_slice[1]]
        
        cell = None
        if cell_slice:
            cell_dim = cell_slice[1] - cell_slice[0]
            if cell_dim > 0:
                cell = features[:, cell_slice[0]:cell_slice[1]]

        return self.fuse(rna, he, cell)
    

class ConcatMlpFusion(FusionStrategy):
    """
    Feature-wise concat-MLP fusion:
    Each modality -> MLP -> (same hidden_dim) -> concat.
    Supports 2 or 3 modalities (RNA/HE/CELL).
    """

    def __init__(
        self,
        rna_dim: int,
        he_dim: int = 0,
        cell_dim: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        logger=None,
    ):
        super().__init__(logger=logger)

        self.hidden_dim = hidden_dim
        self.has_he = he_dim > 0
        self.has_cell = cell_dim > 0

        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.he_encoder = None
        if self.has_he:
            self.he_encoder = nn.Sequential(
                nn.Linear(he_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.cell_dim = cell_dim
        self.cell_affine = CellAffine(cell_dim) if self.has_cell else None

        self.num_modalities = 1 + int(self.has_he)
        self.out_channels = hidden_dim * self.num_modalities + (cell_dim if self.has_cell else 0)

        self.log(
            f"Initializing ConcatMlpFusion: RNA({rna_dim}) + HE({he_dim}) + CELL({cell_dim}) "
            f"-> hidden={hidden_dim}, out={self.out_channels}"
        )

    def fuse(self, rna_feat, he_feat=None, cell_feat=None):
        parts = []
        names = []

        h_rna = self.rna_encoder(rna_feat)
        parts.append(h_rna); names.append("rna")

        if self.has_he and he_feat is not None and he_feat.numel() > 0:
            h_he = self.he_encoder(he_feat)
            parts.append(h_he); names.append("he")

        if self.has_cell:
            if cell_feat is not None and cell_feat.numel() > 0:
                h_cell = self.cell_affine(cell_feat)  # [N, cell_dim]
            else:
                h_cell = torch.zeros((h_rna.size(0), self.cell_dim), device=h_rna.device, dtype=h_rna.dtype)
            parts.append(h_cell); names.append("cell")

        if len(parts) == 1:
            return parts[0], {"modalities": names, "fused_dim": parts[0].shape[1]}

        h = torch.cat(parts, dim=1)
        return h, {"modalities": names, "fused_dim": h.shape[1]}


def build_fusion(fusion_cfg: dict, fusion_slices: Optional[dict] = None, model_hidden_dim: int = 160, model_dropout: float = 0.1, logger=None) -> FusionStrategy:
    """
    Factory for fusion strategies.
    
    Accept 'fusion_slices' to automatically calculate input dimensions for learnable modules.
    """
    fusion_type = fusion_cfg.get("type", "concat")
    
    if fusion_type == "concat_mlp":
        if fusion_slices is None:
            raise ValueError("ConcatMlpFusion requires 'fusion_slices' to determine input dimensions.")

        rna_slice = fusion_slices.get("rna", (0, 0))
        he_slice = fusion_slices.get("he", (0, 0))
        cell_slice = fusion_slices.get("cell", (0, 0))

        rna_dim = rna_slice[1] - rna_slice[0]
        he_dim = he_slice[1] - he_slice[0]
        cell_dim = cell_slice[1] - cell_slice[0]

        hidden_dim = fusion_cfg.get("hidden_dim", model_hidden_dim)
        dropout = fusion_cfg.get("dropout", model_dropout)

        return ConcatMlpFusion(
            rna_dim=rna_dim,
            he_dim=he_dim,
            cell_dim=cell_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            logger=logger,
        )


    raise ValueError(f"Unknown fusion type: {fusion_type}")