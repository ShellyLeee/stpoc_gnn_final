import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_correlation_bar(
    correlations: np.ndarray,
    protein_names: List[str],
    mean_value: float,
    metric_name: str = "Spearman",
    figsize: tuple = None,
    save_path: str = None
):
    """Plot bar chart of per-protein correlations."""
    C_out = len(protein_names)
    
    # Sort by correlation
    sorted_idx = np.argsort(-np.nan_to_num(correlations, nan=-999))
    sorted_corr = correlations[sorted_idx]
    sorted_names = [protein_names[i] for i in sorted_idx]
    
    if figsize is None:
        figsize = (max(8, C_out * 0.3), 5)
    
    plt.figure(figsize=figsize)
    plt.bar(np.arange(C_out), np.nan_to_num(sorted_corr, nan=0.0))
    plt.axhline(mean_value, linestyle="--", linewidth=1.5, color='r',
                label=f'Mean = {mean_value:.4f}')
    plt.xlabel(f"Protein channel (sorted by {metric_name})")
    plt.ylabel(f"{metric_name} ρ")
    plt.title(f"Per-protein {metric_name} correlation")
    plt.legend()
    
    if C_out <= 60:
        plt.xticks(np.arange(C_out), sorted_names, rotation=90)
    else:
        step = max(1, C_out // 40)
        sel = np.arange(C_out)[::step]
        plt.xticks(sel, [sorted_names[i] for i in sel], rotation=90)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str = None
):
    """Plot training and validation loss curves."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_metric_history(
        train_history, 
        val_history, 
        save_path, 
        metric_name="Spearman"
    ):
    
    """Plots training and validation metric history."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label=f'Train {metric_name}')
    plt.plot(val_history, label=f'Validation {metric_name}')
    plt.title(f'Model {metric_name} History')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

