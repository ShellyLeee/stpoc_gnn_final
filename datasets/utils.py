"""Dataset builders and utilities."""

import os
import numpy as np
import torch
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from datasets.graph_dataset import GraphDataset
from datasets.processing.preprocessing import load_and_preprocess_data, create_combined_masks
from datasets.reduction import build_reducer


def build_dataset(config: dict, logger, for_eval: bool = False):
    """Load data, preprocess, fuse H&E, build graph, and loaders."""
    log = logger.info
    base_dir = config.get("run_dir") or os.path.abspath(config.get("logging", {}).get("log_dir", "logs"))
    os.makedirs(base_dir, exist_ok=True)
    
    # NEW: write back run_dir so reducers can save alongside the run
    config["run_dir"] = base_dir

    reduction_method = config["preprocessing"].get("reduction_method", "svd")
    first_layer_reduction = config["model"].get("first_layer_reduction", False)
    strategy = (
        "GNN_DIRECT_REDUCTION" if reduction_method == "none" and first_layer_reduction else "TRADITIONAL"
    )

    log("Loading and preprocessing data ...")
    rna_proc, pro_proc, rna, pro, test_rna_proc, test_rna = load_and_preprocess_data(
        config, logger=logger, load_test=True
    )

    (
        all_rows,
        all_cols,
        train_spot_mask,
        val_spot_mask,
        test_spot_mask,
        node_train_mask,
        node_val_mask,
        node_test_mask,
    ) = create_combined_masks(
        rna,
        test_rna,
        config["data"]["grid_h"],
        config["data"]["grid_w"],
        config["data"]["split_ratio"],
        config["random_seed"],
        logger,
    )

    # RNA features + reduction
    if strategy == "GNN_DIRECT_REDUCTION":
        Z_train = np.asarray(rna_proc.X.toarray(), dtype=np.float32)
        if test_rna_proc is not None:
            Z_test = np.asarray(test_rna_proc.X.toarray(), dtype=np.float32)
            Z_all = np.vstack([Z_train, Z_test])
        else:
            Z_all = Z_train
        explained_var = 1.0
    else:
        reducer = build_reducer(reduction_method, config, logger=logger)
        Z_all, explained_var = reducer.fit_transform(rna_proc, test_rna_proc)
        log(f"Dimensionality reduction explained {explained_var*100:.2f}% of variance")

    rna_dim = Z_all.shape[1]

    # Multimodal fusion
    he_features = None
    
    use_he = config["data"].get("use_he_features", True) # Use H&E features or not

    if use_he:
        he_path = config["data"].get("he_features_save_path")
        if he_path and os.path.exists(he_path):
            log(f"Loading robust H&E features from {he_path}")
            he_features = np.load(he_path)
        else:
            log("⚠️ H&E enabled but features not found. Proceeding with RNA ONLY.")
    else:
        log("🔵 RNA-only mode: Skipping H&E features.")

    slices = {"rna": (0, Z_all.shape[1])}

    # ---- (1) H&E concat ----
    if he_features is not None and he_features.size > 0:
        he = he_features.astype(np.float32)
        he_scaler = StandardScaler()
        he = he_scaler.fit_transform(he).astype(np.float32)
        # Save HE scaler for final_test reuse
        if config["data"].get("save_he_scaler", True):
            he_scaler_path = os.path.join(base_dir, "he_scaler.joblib")
            joblib.dump(he_scaler, he_scaler_path)
            log(f"Saved H&E scaler to: {he_scaler_path}")

        slices["he"] = (Z_all.shape[1], Z_all.shape[1] + he.shape[1])
        Z_all = np.hstack([Z_all, he])
        log(f"Concatenated H&E with RNA -> dim {Z_all.shape[1]}")
    else:
        log("H&E features not provided; using RNA only.")

    # ---- (2) Cell abundance mean concat ----
    use_cell = config["data"].get("use_cell_features", False)
    if use_cell:
        cell_dir = config["data"].get("cell_features_dir", "/mnt/sharedata/ssd_large/users/liyx/bio_info/")
        train_name = config["data"].get("cell_abundance_mean_train")
        test_name  = config["data"].get("cell_abundance_mean_test")

        if not train_name or not test_name:
            raise ValueError("use_cell_features=True but cell_abundance_mean_train / cell_abundance_mean_test not set in config.")

        train_path = os.path.join(cell_dir, train_name)
        test_path  = os.path.join(cell_dir, test_name)

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Cell mean train csv not found: {train_path}")
        if test_rna_proc is not None and not os.path.exists(test_path):
            raise FileNotFoundError(f"Cell mean test csv not found: {test_path}")

        log(f"Loading cell abundance (mean) train from: {train_path}")
        df_train = pd.read_csv(train_path, index_col=0)
        # Align to spot in rna_proc orderly
        train_ids = list(rna_proc.obs_names)
        missing = [i for i in train_ids if i not in df_train.index]
        if len(missing) > 0:
            raise ValueError(f"Cell train csv missing {len(missing)} spots. Example: {missing[0]}")
        cell_train = df_train.loc[train_ids].to_numpy(dtype=np.float32)

        cell_test = None
        if test_rna_proc is not None:
            log(f"Loading cell abundance (mean) test from: {test_path}")
            df_test = pd.read_csv(test_path, index_col=0)
            test_ids = list(test_rna_proc.obs_names)
            missing_t = [i for i in test_ids if i not in df_test.index]
            if len(missing_t) > 0:
                raise ValueError(f"Cell test csv missing {len(missing_t)} spots. Example: {missing_t[0]}")
            cell_test = df_test.loc[test_ids].to_numpy(dtype=np.float32)

        if cell_test is not None:
            cell_all = np.vstack([cell_train, cell_test])
        else:
            cell_all = cell_train

        cell_scaler = StandardScaler()
        cell_scaler.fit(cell_train)
        cell_all = cell_scaler.transform(cell_all).astype(np.float32)

        if config["data"].get("save_cell_scaler", True):
            scaler_path = os.path.join(base_dir, "cell_mean_scaler.joblib")
            joblib.dump(cell_scaler, scaler_path)
            log(f"Saved cell mean scaler to: {scaler_path}")

        # concat & slices
        slices["cell"] = (Z_all.shape[1], Z_all.shape[1] + cell_all.shape[1])
        Z_all = np.hstack([Z_all, cell_all])
        log(f"Concatenated Cell(mean) -> dim {Z_all.shape[1]}")
    else:
        log("Cell features disabled: skipping cell abundance.")

    fusion_meta = {"slices": slices, "fused_dim": Z_all.shape[1]}

    # Protein labels
    pro_X_train = np.asarray(pro_proc.X, dtype=np.float32)
    C_out = pro_X_train.shape[1]
    if test_rna is not None:
        n_test = len(test_rna)
        pro_X_test = np.zeros((n_test, C_out), dtype=np.float32)
        pro_X_all = np.vstack([pro_X_train, pro_X_test])
    else:
        pro_X_all = pro_X_train

    pos_array = np.column_stack([all_rows, all_cols])

    graph_data = GraphDataset.build_graph_from_config(
        pos_array,
        Z_all,
        pro_X_all,
        node_train_mask,
        node_val_mask,
        node_test_mask,
        config,
        base_dir,
        for_eval=for_eval,
    )

    train_loader = None
    val_loader = None
    if not for_eval:
        train_indices = torch.where(graph_data.train_mask)[0]
        val_indices = torch.where(graph_data.val_mask)[0]
        train_loader = GraphDataset.create_neighbor_loader(graph_data, config, train_indices, shuffle=True)
        val_loader = GraphDataset.create_neighbor_loader(graph_data, config, val_indices, shuffle=False)

    meta = {
        "rna_dim": rna_dim,
        "C_out": C_out,
        "C_in": Z_all.shape[1],
        "strategy": strategy,
        "first_layer_reduction": first_layer_reduction,
        "all_rows": all_rows,
        "all_cols": all_cols,
        "node_test_mask": node_test_mask,
        "test_rna": test_rna,
        "pro": pro,
        "pos_array": pos_array,
        "fusion": fusion_meta if 'fusion_meta' in locals() else {},
    }

    return graph_data, train_loader, val_loader, meta
