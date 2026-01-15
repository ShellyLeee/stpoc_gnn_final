import numpy as np
import scanpy as sc
import logging
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
from tqdm.auto import tqdm
from typing import Union
import copy
import sys
import os

# Used for efficient batch loading of data from sparse matrices to avoid memory overflow.
from torch.utils.data import Dataset, DataLoader

class SparseAnnDataDataset(Dataset):
    """
    An efficient Dataset for loading data directly from AnnData.X (sparse matrix).
    It converts required data to dense format only during __getitem__.
    """
    def __init__(self, adata: sc.AnnData):
        # Keep data in sparse format
        self.X = adata.X.astype(np.float32) 
        self.n_samples = adata.n_obs

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Called by DataLoader. idx can be a single index or a list/slice.
        """
        if isinstance(idx, (list, np.ndarray, slice)):
            # If batch, return a [batch_size, n_features] array
            return self.X[idx].toarray()
        else:
            # If single index, return a [n_features] array
            return self.X[idx].toarray()[0]

def load_and_preprocess_data(
    config: dict,
    logger=None,
    load_test: bool = True
) -> Tuple:
    """
    Load RNA and Protein data and perform preprocessing.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance (optional)
        load_test: Whether to load test RNA data
        
    Returns:
        Tuple of (rna_proc, pro_proc, rna_raw, pro_raw, test_rna_proc, test_rna_raw)
        If load_test=False, test_rna_proc and test_rna_raw will be None
    """
    log = logger.info if logger else print
    
    # Load training data
    log("Loading training RNA data...")
    rna = sc.read_h5ad(config['data']['train_rna_h5ad'])
    log(f"Training RNA data: {rna}")
    
    log("Loading training Protein data...")
    pro = sc.read_h5ad(config['data']['train_pro_h5ad'])
    log(f"Training Protein data: {pro}")
    
    # Synchronize RNA and Protein
    if not (rna.obs_names == pro.obs_names).all():
        log("WARNING: RNA and Protein cell names don't match! Synchronizing...")
        common_cells = rna.obs_names.intersection(pro.obs_names)
        rna = rna[common_cells].copy()
        pro = pro[common_cells].copy()
        log(f"After synchronization: {len(common_cells)} common cells")
    
    # Load test data if specified
    test_rna = None
    test_rna_proc = None
    if load_test and 'test_rna_h5ad' in config['data']:
        test_rna_path = config['data']['test_rna_h5ad']
        try:
            log(f"Loading test RNA data from: {test_rna_path}")
            test_rna = sc.read_h5ad(test_rna_path)
            log(f"Test RNA data: {test_rna}")
        except FileNotFoundError:
            log(f"Test RNA file not found: {test_rna_path}. Skipping.")
            test_rna = None
    
    # Make copies for processing
    rna_proc = rna.copy()
    pro_proc = pro.copy()
    if test_rna is not None:
        test_rna_proc = test_rna.copy()
    
    # RNA preprocessing
    if config['preprocessing'].get('normalize_rna', False):
        log("Applying normalize_total + log1p to training RNA...")
        sc.pp.normalize_total(rna_proc, target_sum=1e4)
        sc.pp.log1p(rna_proc)
        
        if test_rna_proc is not None:
            log("Applying normalize_total + log1p to test RNA...")
            sc.pp.normalize_total(test_rna_proc, target_sum=1e4)
            sc.pp.log1p(test_rna_proc)
    
    # Protein preprocessing
    if config['preprocessing'].get('zscore_protein', False):
        log("Applying log1p + z-score to Protein...")
        pro_X = np.asarray(pro_proc.X)
        pro_X = np.log1p(pro_X)
        scaler = StandardScaler(with_mean=True, with_std=True)
        pro_X = scaler.fit_transform(pro_X)
        pro_proc.X = pro_X
    
    return rna_proc, pro_proc, rna, pro, test_rna_proc, test_rna

def create_combined_masks(
    train_rna: sc.AnnData,
    test_rna: Optional[sc.AnnData],
    grid_h: int,
    grid_w: int,
    split_ratio: float = 0.9,
    random_seed: int = 42,
    logger=None
) -> Tuple:
    """
    Create combined train/val/test masks for graph construction.
    
    Args:
        train_rna: Training RNA data
        test_rna: Test RNA data (optional)
        grid_h: Grid height
        grid_w: Grid width
        split_ratio: Train/val split ratio (only applies to train_rna)
        random_seed: Random seed
        logger: Logger instance
        
    Returns:
        Tuple of:
        - all_rows: Combined row positions
        - all_cols: Combined column positions
        - train_spot_mask: Grid mask for train spots
        - val_spot_mask: Grid mask for val spots
        - test_spot_mask: Grid mask for test spots
        - node_train_mask: Boolean array indicating train nodes
        - node_val_mask: Boolean array indicating val nodes
        - node_test_mask: Boolean array indicating test nodes
    """
    log = logger.info if logger else print
    
    # Get train positions
    rows_train = train_rna.obs["array_row"].to_numpy().astype(int)
    cols_train = train_rna.obs["array_col"].to_numpy().astype(int)
    n_train = len(rows_train)
    
    log(f"Training spots: {n_train}")
    
    # Get test positions if available
    if test_rna is not None:
        rows_test = test_rna.obs["array_row"].to_numpy().astype(int)
        cols_test = test_rna.obs["array_col"].to_numpy().astype(int)
        n_test = len(rows_test)
        log(f"Test spots: {n_test}")
        
        # Combine positions
        all_rows = np.concatenate([rows_train, rows_test])
        all_cols = np.concatenate([cols_train, cols_test])
    else:
        all_rows = rows_train
        all_cols = cols_train
        n_test = 0
    
    n_total = len(all_rows)
    log(f"Total spots: {n_total}")
    
    # Create grid masks
    train_spot_mask = np.zeros((grid_h, grid_w), dtype=bool)
    test_spot_mask = np.zeros((grid_h, grid_w), dtype=bool)
    
    for r, c in zip(rows_train, cols_train):
        train_spot_mask[r, c] = True
    
    if test_rna is not None:
        for r, c in zip(rows_test, cols_test):
            test_spot_mask[r, c] = True
    
    # Split train into train/val
    train_indices = np.arange(n_train)
    train_idx, val_idx = train_test_split(
        train_indices,
        train_size=split_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    log(f"Train/Val split: {len(train_idx)}/{len(val_idx)}")
    
    # Create node-level masks
    node_train_mask = np.zeros(n_total, dtype=bool)
    node_val_mask = np.zeros(n_total, dtype=bool)
    node_test_mask = np.zeros(n_total, dtype=bool)
    
    node_train_mask[train_idx] = True
    node_val_mask[val_idx] = True
    
    if test_rna is not None:
        test_node_indices = np.arange(n_train, n_total)
        node_test_mask[test_node_indices] = True
    
    # Create val grid mask
    val_spot_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx in val_idx:
        r, c = rows_train[idx], cols_train[idx]
        val_spot_mask[r, c] = True
    
    log(f"Grid masks - Train: {train_spot_mask.sum()}, Val: {val_spot_mask.sum()}, Test: {test_spot_mask.sum()}")
    
    return (
        all_rows, all_cols,
        train_spot_mask, val_spot_mask, test_spot_mask,
        node_train_mask, node_val_mask, node_test_mask
    )

# --- Merged Dimensionality Reduction Function (Keeping only one version) ---

def _normalize_device(device):
    """
    Accepts various forms like str / torch.device / accelerator.device,
    uniformly returns a torch.device object.
    """
    if isinstance(device, torch.device):
        normalized = device
    else:
        normalized = torch.device(str(device))

    if normalized.type == "cuda":
        available = torch.cuda.device_count()
        # When CUDA_VISIBLE_DEVICES masks GPUs, torch sees contiguous ordinals starting at 0.
        if normalized.index is not None and (available == 0 or normalized.index >= available):
            # Fallback to first visible GPU or CPU to avoid invalid device ordinal.
            normalized = torch.device("cuda:0" if available > 0 else "cpu")
    return normalized

def apply_dimensionality_reduction(
    train_rna_proc: sc.AnnData,
    test_rna_proc: Optional[sc.AnnData],
    config: dict,
    device: Union[str, torch.device],
    random_seed: int = 42,
    logger=None
) -> Tuple[np.ndarray, float]:
    
    log = logger.info if logger else print

    device = _normalize_device(device)
    
    # Get reduction config
    reduction_method = config['preprocessing'].get('reduction_method', 'mlp')

    k_reduced = config['preprocessing'].get('k_pca', 512)
    
    log(f"Dimensionality reduction method: {reduction_method}")
    log(f"Training data shape: {train_rna_proc.X.shape}")
    if test_rna_proc is not None:
        log(f"Test data shape: {test_rna_proc.X.shape}")
    log(f"Target dimension: {k_reduced}")
    

    if reduction_method == 'mlp':
        # MLP reduction requires device parameter
        return _apply_mlp_reduction(
            train_rna_proc, test_rna_proc, k_reduced, config, random_seed, device, log
        )
    else:
        # Update error message
        raise ValueError(f"Unknown reduction method: {reduction_method}. Please enter a valid reduction methos.")

def _apply_mlp_reduction(
    train_rna_proc: sc.AnnData,
    test_rna_proc: Optional[sc.AnnData],
    k_reduced: int,
    config: dict,
    random_seed: int,
    device: torch.device, # <--- Passed from upper level
    log
) -> Tuple[np.ndarray, float]:
    """
    Apply MLP-based learnable reduction (Memory-Efficient Version).
    Uses DataLoader to process data in batches.
    """
    log("Applying MLP-based reduction (Memory-Efficient)...")

    device = _normalize_device(device)
    device_type = device.type

    # Dynamic import
    try:
        from models.mlp_reducer import MLPReducer
    except ImportError:
        # If file is not in parent directory, try importing from current directory
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from models.mlp_reducer import MLPReducer
    
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Get MLP config
    mlp_config = config['preprocessing'].get('mlp_reducer', {})
    use_all_rna = bool(mlp_config.get("use_all_rna", False))
    hidden_dim = mlp_config.get('hidden_dim', None)
    num_layers = mlp_config.get('num_layers', 3)
    dropout = mlp_config.get('dropout', 0.1)
    use_batchnorm = mlp_config.get('use_batchnorm', True)
    
    # Training config
    num_epochs = mlp_config.get('num_epochs', 50)
    batch_size = mlp_config.get('batch_size', 512)
    learning_rate = mlp_config.get('learning_rate', 0.001)
    
    # --- Memory Optimization: Create Dataset ---
    log("Creating sparse dataset for MLP training...")

    adata_fit = train_rna_proc  # default: train-only

    if use_all_rna:
        if test_rna_proc is None:
            raise ValueError("[use_all_rna=True] requires test_rna_proc (slice-internal unlabeled split).")

        final_path = config.get("data", {}).get("final_test_rna_h5ad", None)
        if final_path is None:
            raise ValueError("[use_all_rna=True] config['data']['final_test_rna_h5ad'] is missing.")
        if not os.path.exists(final_path):
            raise FileNotFoundError(f"[use_all_rna=True] final_test_rna_h5ad not found: {final_path}")

        log(f"[use_all_rna] Loading final_test RNA from: {final_path}")
        final_test_rna_proc = sc.read_h5ad(final_path)

        # apply the SAME preprocessing as train/test (normalize_total + log1p) if enabled
        if config.get("preprocessing", {}).get("normalize_rna", False):
            sc.pp.normalize_total(final_test_rna_proc, target_sum=1e4)
            sc.pp.log1p(final_test_rna_proc)

        # hard safety checks: gene names AND order must match
        if not (train_rna_proc.var_names.equals(test_rna_proc.var_names)):
            raise ValueError("[use_all_rna=True] train_rna_proc.var_names != test_rna_proc.var_names (order mismatch).")
        if not (train_rna_proc.var_names.equals(final_test_rna_proc.var_names)):
            raise ValueError("[use_all_rna=True] train_rna_proc.var_names != final_test_rna_proc.var_names (order mismatch).")

        # concat for fitting reducer ONLY
        adata_fit = sc.concat(
            [train_rna_proc, test_rna_proc, final_test_rna_proc],
            axis=0,
            join="inner",
            merge="same",
        )
        log(f"[use_all_rna] Fitting MLP reducer on concat RNA: n_obs={adata_fit.n_obs:,}, n_vars={adata_fit.n_vars:,}")

    fit_dataset = SparseAnnDataDataset(adata_fit)
    in_features = fit_dataset.X.shape[1]  # Get features from sparse matrix
    
    log(f"MLP config: hidden_dim={hidden_dim}, layers={num_layers}, dropout={dropout}")
    log(f"Training: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    
    # Create model
    model = MLPReducer(
        in_features=in_features,
        out_features=k_reduced,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_batchnorm=use_batchnorm
    ).to(device)
    
    log(f"Model parameters: {model.count_parameters():,}")
    
    # Train the reducer
    log("Training MLP reducer...")
    model, best_loss = _train_mlp_reducer(
        model,
        fit_dataset,
        num_epochs,
        batch_size,
        learning_rate,
        device,
        log
    )

    # --- Save the trained MLP reducer (best model already loaded inside _train_mlp_reducer) ---
    save_cfg = config.get("preprocessing", {}).get("mlp_reducer", {})
    save_best = save_cfg.get("save_best", True)

    run_dir = config.get("run_dir") or os.path.abspath(config.get("logging", {}).get("log_dir", "logs"))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    mlp_ckpt_path = save_cfg.get("ckpt_path") or os.path.join(ckpt_dir, "mlp_reducer_best.pt")

    if save_best:
        payload = {
            "state_dict": model.state_dict(),
            "in_features": in_features,
            "out_features": k_reduced,
            "mlp_config": mlp_config,
            "best_mse": float(best_loss),
            "random_seed": int(random_seed),
            # helpful for safety checks later:
            "normalize_rna": bool(config.get("preprocessing", {}).get("normalize_rna", False)),
            "reduction_method": "mlp",
        }
        torch.save(payload, mlp_ckpt_path)
        log(f"[MLPReducer] Saved best reducer checkpoint to: {mlp_ckpt_path}")
    
    log("Estimating data variance for 'Explained Variance' metric...")
    n_sample_var = min(10000, train_rna_proc.n_obs)
    sample_indices = np.random.choice(train_rna_proc.n_obs, n_sample_var, replace=False)
    X_sample_var = train_rna_proc.X[sample_indices].toarray().var()
    
    explained_variance_sum = 0.0
    if X_sample_var > 1e-8:
        explained_variance_sum = 1.0 - (best_loss / X_sample_var)
    
    log(f"MLP Best MSE (from Early Stopping): {best_loss:.6f}")
    log(f"Estimated Data Variance (from {n_sample_var} samples): {X_sample_var:.6f}")
    log(f"Explained variance (1 - MSE/Est.Var): {explained_variance_sum:.4f}")
    
    
    # --- Memory Optimization: Batch-wise Inference (Transform) ---
    
    log("Transforming training data (batch-wise)...")
    model.eval()
    
    # 1. Create DataLoader for training data
    train_dataset_inf = SparseAnnDataDataset(train_rna_proc)
    train_loader_inf = DataLoader(
        train_dataset_inf,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=(device_type == "cuda"),
    )
    
    Z_train_list = []
    with torch.no_grad():
        for batch_X in tqdm(train_loader_inf, desc="Transforming Train"):
            batch = batch_X.to(device)
            Z_batch = model(batch)
            Z_train_list.append(Z_batch.cpu().numpy())
    Z_train = np.vstack(Z_train_list)
    
    # 2. Create DataLoader for test data (if exists)
    if test_rna_proc is not None:
        log("Transforming test data (batch-wise)...")
        test_dataset = SparseAnnDataDataset(test_rna_proc)
        test_loader_inf = DataLoader(
            test_dataset, 
            batch_size=batch_size * 2, 
            shuffle=False,
            num_workers=4,
            pin_memory = (device_type == "cuda")
        )
        
        Z_test_list = []
        with torch.no_grad():
            for batch_X in tqdm(test_loader_inf, desc="Transforming Test"):
                batch = batch_X.to(device)
                Z_batch = model(batch)
                Z_test_list.append(Z_batch.cpu().numpy())
        Z_test = np.vstack(Z_test_list)
        
        Z_all = np.vstack([Z_train, Z_test])
    else:
        Z_all = Z_train
    
    Z_all = np.asarray(Z_all, dtype=np.float32)
    log(f"Final shape: {Z_all.shape}")
    
    return Z_all, explained_variance_sum


def _train_mlp_reducer(
    model: torch.nn.Module,
    train_dataset: SparseAnnDataDataset, # <--- Memory Optimization: Accept Dataset
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    log
) -> Tuple[torch.nn.Module, float]:
    """
    (Improved) Train MLP reducer with reconstruction loss.
    
    - Added Learning Rate Scheduler (ReduceLROnPlateau) to handle unstable training.
    - Added Early Stopping mechanism, returns only the model with the lowest loss.
    - Memory Optimization: Uses DataLoader to load data from Dataset.
    """
    import torch.optim as optim

    device = _normalize_device(device)
    device_type = device.type
    
    # Add a reconstruction head
    reconstruction_head = torch.nn.Linear(
        model.out_features, model.in_features
    ).to(device)
    
    # Optimizer
    params = list(model.parameters()) + list(reconstruction_head.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    
    # --- LR Scheduler ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=False
    )
    
    # --- Memory Optimization: Use DataLoader ---
    # num_workers > 0 needs care on Windows, but is generally safe on Linux servers
    num_loader_workers = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_loader_workers,
        pin_memory = (device_type == "cuda")
    )
    
    # --- Early Stopping & Save Best Model ---
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 15 # Stop if loss doesn't improve for 15 consecutive epochs
    
    # Training loop
    model.train()
    reconstruction_head.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # --- Memory Optimization: Iterate over DataLoader ---
        pbar = tqdm(train_loader, 
                   desc=f"Epoch {epoch+1}/{num_epochs}",
                   leave=False)
        
        for batch_X in pbar:
            # Data is already a batch, move directly to device
            batch = batch_X.to(device)
            
            optimizer.zero_grad()
            
            reduced = model(batch)
            reconstructed = reconstruction_head(reduced)
            
            loss = torch.nn.functional.mse_loss(reconstructed, batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        # Divide by number of batches
        avg_loss = epoch_loss / len(train_loader)
        
        # --- Scheduler and Early Stopping Logic ---
        scheduler.step(avg_loss) # Scheduler monitors avg_loss
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save the *state* of the model with the lowest loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0 # Reset patience counter
            log_msg = f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f} (New Best) | LR = {current_lr:.1e}"
        else:
            patience_counter += 1
            log_msg = f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f} (Patience: {patience_counter}/{early_stopping_patience}) | LR = {current_lr:.1e}"
        
        # Print only on rank 0 (if using DDP)
        log(log_msg)

        if patience_counter >= early_stopping_patience:
            log(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    log("MLP reducer training completed!")
    
    # --- Critical: Load Best Model ---
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    
    # Return best loss, not the loss from the last epoch
    return model, best_loss
