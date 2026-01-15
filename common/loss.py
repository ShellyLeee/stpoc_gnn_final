import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedHuber(nn.Module):
    """Masked Huber loss - robust to outliers."""
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.huber = nn.SmoothL1Loss(reduction='none', beta=delta)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted values (N, C)
            target: Target values (N, C)
            mask: Boolean mask (N,)
            
        Returns:
            Masked Huber loss (scalar)
        """
        loss_map = self.huber(pred, target)
        loss_map = loss_map[mask]
        
        if loss_map.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        return loss_map.mean()


def build_criterion(config: dict) -> nn.Module:

    loss_type = config['loss']['type']
    
    if loss_type == 'huber':
        delta = config['loss'].get('huber_delta', 1.0)
        return MaskedHuber(delta=delta)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")