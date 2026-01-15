import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Tuple, List


def compute_correlations_from_nodes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    protein_names: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute per-protein correlations on masked nodes.
    
    Args:
        y_true: Ground truth (N, C)
        y_pred: Predictions (N, C)
        mask: Boolean mask (N,)
        protein_names: List of protein names
        
    Returns:
        spearman_rhos, pearson_rhos, mean_spearman, mean_pearson
    """
    C_out = y_true.shape[1]
    
    if protein_names is None:
        protein_names = [f"protein_{i}" for i in range(C_out)]
    
    spearman_rhos = np.full(C_out, np.nan, dtype=np.float64)
    pearson_rhos = np.full(C_out, np.nan, dtype=np.float64)
    
    for j in range(C_out):
        true_vals = y_true[mask, j]
        pred_vals = y_pred[mask, j]
        
        n = true_vals.size
        if n < 3 or np.std(true_vals) < 1e-8 or np.std(pred_vals) < 1e-8:
            continue
        
        try:
            rho_s, _ = spearmanr(true_vals, pred_vals)
            rho_p, _ = pearsonr(true_vals, pred_vals)
            spearman_rhos[j] = rho_s
            pearson_rhos[j] = rho_p
        except:
            pass
    
    mean_spearman = np.nanmean(spearman_rhos)
    mean_pearson = np.nanmean(pearson_rhos)
    
    return spearman_rhos, pearson_rhos, mean_spearman, mean_pearson