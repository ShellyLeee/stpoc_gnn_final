"""
MLP-based dimensionality reduction for RNA features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLPReducer(nn.Module):
    """
    Multi-layer Perceptron for dimensionality reduction.
    
    Architecture:
        Input -> Linear -> BatchNorm -> ReLU -> Dropout -> 
        Linear -> BatchNorm -> ReLU -> Dropout ->
        Linear -> Output
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = None,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_batchnorm: bool = True
    ):
        """
        Args:
            in_features: Input dimension (e.g., number of genes)
            out_features: Output dimension (e.g., 2048, 4096)
            hidden_dim: Hidden layer dimension (default: mean of in and out)
            num_layers: Number of layers (2-4 recommended)
            dropout: Dropout rate
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Default hidden dim: geometric mean of input and output
        if hidden_dim is None:
            hidden_dim = int((in_features * out_features) ** 0.5)
        
        # Build layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(in_features, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(hidden_dim, out_features))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, in_features)
            
        Returns:
            Reduced features (batch_size, out_features)
        """
        return self.network(x)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_reducer(
    reducer_type: str,
    in_features: int,
    out_features: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create dimensionality reducer.
    
    Args:
        reducer_type: 'mlp'
        in_features: Input dimension
        out_features: Output dimension
        **kwargs: Additional arguments for the reducer
        
    Returns:
        Reducer module
    """
    if reducer_type == 'mlp':
        return MLPReducer(
            in_features=in_features,
            out_features=out_features,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown reducer type: {reducer_type}")