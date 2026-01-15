"""
GNN Model Overview:
GAT are the "message passing" layers.
Their job is to update the features of a central node (you) by aggregating features from your neighbors. 
The difference lies in *how* they aggregate.
[Note: The weight learning process occurs during training mini-batches]

Model: GAT - During graph construction, it computes an attention score (learned) for the edge between every neighbor and the central node.

Common Tricks:
- Group/Batch Normalization
- Edge Dropout & Feature Dropout
- Add Self-Loops
- Smoothness Regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dropout_edge
from typing import Optional, Union

from models.fusion import build_fusion

class GNNModel(nn.Module):
    """
    Standard GNN Model (Baseline Compatible).
    
    Features:
    - Allows dimension expansion (e.g., GAT concat=True).
    - Automatically adjusts Normalization layers to match expanded dimensions.
    - Does NOT support Skip Connections.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 256,
        num_layers: int = 2,
        conv_type: str = "GAT",
        # Tricks
        add_self_loops: bool = True,
        norm_type: str = "batch",
        dropout: float = 0.5,
        edge_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_groups: int = 32,
        # GAT Specifics
        gat_heads: int = 4,
        gat_concat: bool = True,  # Default True for baseline
        # Fusion
        fusion_cfg: Optional[dict] = None,
        fusion_slices: Optional[dict] = None,
        **kwargs
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.norm_type = norm_type
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.add_self_loops = add_self_loops
        
        self.fusion_slices = fusion_slices
        self.fusion = None
        current_dim = in_channels

        def _slice_dim(key: str) -> int:
            if not self.fusion_slices:
                return 0
            sl = self.fusion_slices.get(key)
            if not sl:
                return 0
            return max(0, sl[1] - sl[0])

        he_dim = _slice_dim("he")
        cell_dim = _slice_dim("cell")

        # If other modalities exist: conduct fusion
        if self.fusion_slices and (he_dim > 0 or cell_dim > 0):
            self.fusion = build_fusion(
                fusion_cfg or {"type": "concat_mlp"},
                fusion_slices=self.fusion_slices,
                model_hidden_dim=hidden_channels,
                model_dropout=dropout,
                logger=kwargs.get("logger"),
            )

            if hasattr(self.fusion, "out_channels"):
                print(f"[Model] Fusion Strategy detected: Output dim changed from {current_dim} to {self.fusion.out_channels}")
                current_dim = self.fusion.out_channels


        # 1. Encoder (Input Projection)
        self.encoder = nn.Identity()
        
        # 2. Convolutional Layers & Normalization
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # A. Convolution
            if conv_type == "GAT":
                conv = GATv2Conv(
                    current_dim, 
                    hidden_channels, 
                    heads=gat_heads, 
                    concat=gat_concat, 
                    dropout=attention_dropout,
                    add_self_loops=add_self_loops, 
                    edge_dim=1
                )
                # Calculate next dimension: if concat=True, dim expands
                next_dim = hidden_channels * gat_heads if gat_concat else hidden_channels
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            
            self.convs.append(conv)
            
            # B. Normalization (Must match next_dim)
            if norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(next_dim))
            elif norm_type == "group":
                # Ensure valid number of groups
                groups = num_groups if next_dim % num_groups == 0 else 8
                self.norms.append(nn.GroupNorm(groups, next_dim))
            elif norm_type == "layer":
                self.norms.append(nn.LayerNorm(next_dim))
            else:
                self.norms.append(nn.Identity())
            
            # Update dimension for the next layer's input
            current_dim = next_dim

        # 3. Decoder (Output Projection)
        self.decoder = nn.Linear(current_dim, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_smoothness_loss: bool = False,
        return_fusion_info: bool = False
    ) -> torch.Tensor:
        
        fusion_info = {}

        if self.fusion is not None and self.fusion_slices is not None:
            x, fusion_info = self.fusion.fuse_with_slices(x, self.fusion_slices)
        
        # Input Encoding
        x = self.encoder(x)
        
        # Edge Dropout (Structure Regularization)
        if self.training and self.edge_dropout > 0:
            edge_index, edge_mask = dropout_edge(edge_index, p=self.edge_dropout, force_undirected=False)
            if edge_attr is not None:
                edge_attr = edge_attr[edge_mask]

        # Message Passing
        for conv, norm in zip(self.convs, self.norms):
            # Conv -> Norm -> Activation -> Dropout
            if self.conv_type == "GAT":
                x = conv(x, edge_index, edge_attr=edge_attr)
            
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # No Skip Connections here
        
        out = self.decoder(x)

        ret = [out]

        if return_smoothness_loss:
            smoothness_loss = self.compute_smoothness_loss(out, edge_index)
            ret.append(smoothness_loss)
        
        if return_fusion_info:
            ret.append(fusion_info)
        
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)

    def compute_smoothness_loss(self, predictions, edge_index):
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=predictions.device)
        src_pred = predictions[edge_index[0]]
        dst_pred = predictions[edge_index[1]]
        return F.mse_loss(src_pred, dst_pred)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_gnn_model(config: dict, in_channels: int, out_channels: int):
    """
    Handles parameter mapping (YAML 'fusion' -> Class 'fusion_cfg') automatically.
    """
    cfg = config.copy()
    
    if 'fusion' in cfg and 'fusion_cfg' not in cfg:
        cfg['fusion_cfg'] = cfg.pop('fusion')
    
    skip_type = cfg.get('skip_type', 'none')
    
    model_args = {
        'in_channels': in_channels,
        'out_channels': out_channels,
        **cfg
    }
    
    if skip_type == 'none':
        print(f"Building Standard GNNModel (skip_type='none', Variable Width allowed).")
        return GNNModel(**model_args)
    else:
        raise ValueError("Only skip_type='none' is supported in submission version.")