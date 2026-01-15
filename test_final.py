#!/usr/bin/env python
"""
Final test inference entrypoint (external slice).

This script:
1) Loads final_test_rna.h5ad and applies the same RNA preprocessing as training (transform-only).
2) Loads the saved MLP reducer checkpoint (mlp_reducer_best.pt) and transforms RNA to reduced embeddings (NO re-training).
3) Loads pre-extracted robust H&E features (.npy), standardizes them, and concatenates with reduced RNA.
4) Builds a graph (all nodes treated as "test" nodes) and runs inference with the trained GNN checkpoint.
5) Writes the final prediction CSV into the run directory.
"""

import argparse
import os
import joblib

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml

from common.logging import setup_logging
from datasets.graph_dataset import GraphDataset
from trainers.predict import infer_full_graph, maybe_reverse_preprocessing


def _load_mlp_reducer_from_ckpt(mlp_ckpt_path: str, device: torch.device):
    """
    Load the MLP reducer model + metadata from a saved checkpoint.
    The checkpoint is expected to store:
      - state_dict
      - in_features
      - out_features
      - mlp_config (hidden_dim, num_layers, dropout, use_batchnorm)
    """
    ckpt = torch.load(mlp_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    in_features = int(ckpt["in_features"])
    out_features = int(ckpt["out_features"])
    mlp_config = ckpt.get("mlp_config", {})

    # Import the same model class used during training reduction
    from models.mlp_reducer import MLPReducer

    model = MLPReducer(
        in_features=in_features,
        out_features=out_features,
        hidden_dim=mlp_config.get("hidden_dim", None),
        num_layers=mlp_config.get("num_layers", 3),
        dropout=mlp_config.get("dropout", 0.1),
        use_batchnorm=mlp_config.get("use_batchnorm", True),
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, in_features, out_features


def _transform_rna_with_mlp_reducer(rna_adata_proc: sc.AnnData, reducer, batch_size: int, device: torch.device):
    """
    Transform sparse AnnData.X into reduced embeddings using the loaded MLP reducer (batch-wise).
    This avoids densifying the full matrix at once.
    """
    # Reuse the exact Dataset implementation you already use in preprocessing.py
    # (it converts sparse rows to dense only when indexing).
    from datasets.processing.preprocessing import SparseAnnDataDataset
    from torch.utils.data import DataLoader

    ds = SparseAnnDataDataset(rna_adata_proc)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )

    z_list = []
    with torch.no_grad():
        for batch_X in loader:
            batch = batch_X.to(device)
            z = reducer(batch)
            z_list.append(z.detach().cpu().numpy())
    return np.vstack(z_list).astype(np.float32)


def _get_pos_array_for_graph(final_test_rna: sc.AnnData):
    """
    Return a (N, 2) coordinate array for building the graph.
    Prefer array_row/array_col if present (Visium grid coords),
    otherwise fallback to pxl_row_in_fullres/pxl_col_in_fullres.
    """
    if "array_row" in final_test_rna.obs.columns and "array_col" in final_test_rna.obs.columns:
        rows = final_test_rna.obs["array_row"].to_numpy().astype(np.float32)
        cols = final_test_rna.obs["array_col"].to_numpy().astype(np.float32)
        return np.column_stack([rows, cols]), rows, cols

    if "pxl_row_in_fullres" in final_test_rna.obs.columns and "pxl_col_in_fullres" in final_test_rna.obs.columns:
        rows = final_test_rna.obs["pxl_row_in_fullres"].to_numpy().astype(np.float32)
        cols = final_test_rna.obs["pxl_col_in_fullres"].to_numpy().astype(np.float32)
        return np.column_stack([rows, cols]), rows, cols

    raise ValueError("Cannot find coordinates in obs: need (array_row,array_col) or (pxl_row_in_fullres,pxl_col_in_fullres).")


def run_prediction_final(
    config_path: str,
    gnn_checkpoint_path: str,
    output_path: str,
    exp_name: str,
):
    # -----------------------------
    # Load config and set run_dir
    # -----------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging_cfg = config.get("logging", {})
    base_log_dir = os.path.abspath(logging_cfg.get("log_dir", "logs"))
    run_dir = os.path.join(base_log_dir, exp_name) if exp_name else base_log_dir
    os.makedirs(run_dir, exist_ok=True)
    config["run_dir"] = run_dir

    logger = setup_logging(os.path.join(run_dir, "test_final.log"))
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Default paths (follow the same naming convention as your run folder)
    gnn_checkpoint_path = gnn_checkpoint_path or os.path.join(run_dir, "checkpoints", "gnn_best.pt")
    output_path = output_path or os.path.join(run_dir, f"predictions_{exp_name}.csv")

    # MLP reducer checkpoint path (saved during training run)
    mlp_ckpt_path = os.path.join(run_dir, "checkpoints", "mlp_reducer_best.pt")

    if not os.path.exists(gnn_checkpoint_path):
        raise FileNotFoundError(f"GNN checkpoint not found: {gnn_checkpoint_path}")
    if not os.path.exists(mlp_ckpt_path):
        raise FileNotFoundError(f"MLP reducer checkpoint not found: {mlp_ckpt_path}")

    # -----------------------------
    # Load trained GNN checkpoint
    # -----------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Loading trained GNN checkpoint")
    logger.info("=" * 60)
    gnn_ckpt = torch.load(gnn_checkpoint_path, map_location="cpu")
    ckpt_config = gnn_ckpt.get("config", {})

    # -----------------------------
    # Load final test RNA + protein names
    # -----------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Loading final test RNA and protein names")
    logger.info("=" * 60)

    final_test_rna_path = config["data"]["final_test_rna_h5ad"]
    he_features_path = config["data"]["test_he_features_path"]
    train_pro_path = config["data"]["train_pro_h5ad"]

    if not os.path.exists(final_test_rna_path):
        raise FileNotFoundError(f"final_test_rna_h5ad not found: {final_test_rna_path}")
    if not os.path.exists(he_features_path):
        raise FileNotFoundError(f"test_he_features_path not found: {he_features_path}")
    if not os.path.exists(train_pro_path):
        raise FileNotFoundError(f"train_pro_h5ad not found (needed for protein names): {train_pro_path}")

    final_test_rna = sc.read_h5ad(final_test_rna_path)
    pro = sc.read_h5ad(train_pro_path)
    protein_names = list(map(str, pro.var_names))
    C_out = len(protein_names)

    logger.info(f"Final test RNA: {final_test_rna}")
    logger.info(f"Protein dim (C_out): {C_out}")

    # -----------------------------
    # Apply RNA preprocessing (same flags as training)
    # -----------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: RNA preprocessing (transform-only)")
    logger.info("=" * 60)

    # Mirror load_and_preprocess_data behavior for RNA normalization
    # (normalize_total + log1p applied consistently in training).
    final_test_rna_proc = final_test_rna.copy()
    if ckpt_config.get("preprocessing", {}).get("normalize_rna", False) or config.get("preprocessing", {}).get("normalize_rna", False):
        logger.info("Applying normalize_total + log1p to final test RNA.")
        sc.pp.normalize_total(final_test_rna_proc, target_sum=1e4)
        sc.pp.log1p(final_test_rna_proc)

    # -----------------------------
    # Load MLP reducer and transform RNA (NO training)
    # -----------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Loading MLP reducer and transforming RNA")
    logger.info("=" * 60)

    reducer, in_features_expected, rna_dim = _load_mlp_reducer_from_ckpt(mlp_ckpt_path, device)

    if final_test_rna_proc.n_vars != in_features_expected:
        raise ValueError(
            f"Gene dimension mismatch: final_test_rna_proc.n_vars={final_test_rna_proc.n_vars} "
            f"but reducer expects in_features={in_features_expected}. "
            f"Ensure the same gene set / order as training."
        )

    # Use the same batch_size used in reducer config if present; otherwise a safe default.
    reducer_bs = ckpt_config.get("preprocessing", {}).get("mlp_reducer", {}).get("batch_size", 512)
    Z_rna = _transform_rna_with_mlp_reducer(final_test_rna_proc, reducer, batch_size=reducer_bs * 2, device=device)
    logger.info(f"Reduced RNA shape: {Z_rna.shape}")

    # -----------------------------
    # Load HE / Cell features and concatenate
    # -----------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: Loading extra modalities (H&E / Cell) and concatenating")
    logger.info("=" * 60)

    use_he = config["data"].get("use_he_features", True)
    use_cell = config["data"].get("use_cell_features", False)

    # Start from RNA-only
    Z_all = Z_rna.astype(np.float32)
    slices = {"rna": (0, Z_rna.shape[1])}
    cursor = Z_rna.shape[1]

    # ---- (A) H&E ----
    if use_he:
        he_features_path = config["data"].get("test_he_features_path")
        if he_features_path and os.path.exists(he_features_path):
            logger.info(f"Loading H&E features from: {he_features_path}")
            he = np.load(he_features_path).astype(np.float32)

            if he.shape[0] != final_test_rna.n_obs:
                raise ValueError(
                    f"HE features N mismatch: he.shape[0]={he.shape[0]} vs final_test_rna.n_obs={final_test_rna.n_obs}"
                )

            he_scaler_path = os.path.join(run_dir, "he_scaler.joblib")
            if not os.path.exists(he_scaler_path):
                raise FileNotFoundError(
                    f"H&E scaler not found: {he_scaler_path}. "
                    "Make sure you saved it during training (datasets/utils.py)."
                )
            he_scaler = joblib.load(he_scaler_path)
            he = he_scaler.transform(he).astype(np.float32)

            slices["he"] = (cursor, cursor + he.shape[1])
            cursor += he.shape[1]
            Z_all = np.hstack([Z_all, he]).astype(np.float32)

            logger.info(f"✅ Added H&E | HE dim={he.shape[1]} | Total dim={Z_all.shape[1]}")
        else:
            logger.warning(
                f"⚠️ H&E enabled but features not found at: {he_features_path}. Proceeding without H&E."
            )
    else:
        logger.info("🔵 H&E disabled (use_he_features=false).")

    # ---- (B) Cell abundance (mean) for FINAL TEST ----
    if use_cell:
        cell_dir = config["data"].get("cell_features_dir", "/mnt/sharedata/ssd_large/users/liyx/bio_info/")

        # You said you changed the config key for final test:
        #   cell_abundance_mean_test_final
        # We'll also keep a fallback to cell_abundance_mean_test to be safe.
        final_test_name = config["data"].get("cell_abundance_mean_test_final")

        if not final_test_name:
            raise ValueError("use_cell_features=True but missing data.cell_abundance_mean_test_final in config.")

        final_test_path = os.path.join(cell_dir, final_test_name)

        if not os.path.exists(final_test_path):
            raise FileNotFoundError(f"Cell mean FINAL TEST csv not found: {final_test_path}")

        logger.info(f"Loading Cell abundance (mean) FINAL TEST from: {final_test_path}")
        df_cell_test = pd.read_csv(final_test_path, index_col=0)

        # Align to final_test_rna spot order
        test_ids = list(final_test_rna.obs_names)
        missing = [i for i in test_ids if i not in df_cell_test.index]
        if len(missing) > 0:
            raise ValueError(f"Cell FINAL TEST csv missing {len(missing)} spots. Example: {missing[0]}")

        cell_test = df_cell_test.loc[test_ids].to_numpy(dtype=np.float32)

        # Standardization: prefer loading the scaler saved during training in run_dir
        scaler_path = os.path.join(run_dir, "cell_mean_scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Cell scaler not found: {scaler_path}. "
                "Make sure you saved it during training (datasets/utils.py)."
            )

        logger.info(f"Loading saved cell scaler from: {scaler_path}")
        cell_scaler = joblib.load(scaler_path)
        cell_test = cell_scaler.transform(cell_test).astype(np.float32)

        # Concat cell (kept semantic: 20 dims)
        slices["cell"] = (cursor, cursor + cell_test.shape[1])
        cursor += cell_test.shape[1]
        Z_all = np.hstack([Z_all, cell_test]).astype(np.float32)

        logger.info(f"✅ Added Cell(mean) | Cell dim={cell_test.shape[1]} | Total dim={Z_all.shape[1]}")
    else:
        logger.info("🔵 Cell features disabled (use_cell_features=false).")

    C_in = Z_all.shape[1]
    logger.info(f"Final input dimension (C_in): {C_in}")


    # -----------------------------
    # Build graph (all nodes are test nodes)
    # -----------------------------
    logger.info("=" * 60)
    logger.info("STEP 6: Building graph for final test")
    logger.info("=" * 60)

    pos_array, all_rows, all_cols = _get_pos_array_for_graph(final_test_rna)

    n_total = final_test_rna.n_obs
    node_train_mask = np.zeros(n_total, dtype=bool)
    node_val_mask = np.zeros(n_total, dtype=bool)
    node_test_mask = np.ones(n_total, dtype=bool)

    # No labels for final test; create dummy protein matrix with correct shape.
    pro_X_all = np.zeros((n_total, C_out), dtype=np.float32)

    graph_data = GraphDataset.build_graph_from_config(
        pos_array,
        Z_all,
        pro_X_all,
        node_train_mask,
        node_val_mask,
        node_test_mask,
        config,
        run_dir,
        for_eval=True,
    )

    logger.info(f"Graph built | Nodes: {graph_data.num_nodes:,} | Edges: {graph_data.num_edges:,}")

    # -----------------------------
    # Build model and load weights
    # -----------------------------
    logger.info("=" * 60)
    logger.info("STEP 7: Building model and loading GNN weights")
    logger.info("=" * 60)

    from models.utils import build_model

    meta = {
        "rna_dim": rna_dim,
        "C_out": C_out,
        "C_in": C_in,
        "all_rows": all_rows,
        "all_cols": all_cols,
        "node_test_mask": node_test_mask,
        "test_rna": final_test_rna,
        "pro": pro,
        "pos_array": pos_array,
        "fusion": {"slices": slices, "fused_dim": C_in},
    }

    model = build_model(config, C_in, C_out, meta, logger=logger).to(device)
    model.load_state_dict(gnn_ckpt["model_state_dict"], strict=True)
    model.eval()
    logger.info("Model loaded successfully.")

    # -----------------------------
    # Inference
    # -----------------------------
    logger.info("=" * 60)
    logger.info("STEP 8: Inference on full final test graph")
    logger.info("=" * 60)

    batch_size_nodes = config.get("graph", {}).get("batch_size_nodes", 2048)
    fanout = config.get("graph", {}).get("fanout", [10, 10])

    logits_all, sec = infer_full_graph(model, graph_data, device, config, batch_size_nodes, fanout)
    logger.info(f"Inference done in {sec:.2f}s | Logits shape: {logits_all.shape}")

    preds = logits_all[node_test_mask]
    preds = maybe_reverse_preprocessing(preds, config, logger)

    # -----------------------------
    # Save CSV
    # -----------------------------
    logger.info("=" * 60)
    logger.info("STEP 9: Saving predictions to CSV")
    logger.info("=" * 60)

    df = pd.DataFrame()
    df["barcode"] = final_test_rna.obs_names.values

    if "pxl_row_in_fullres" in final_test_rna.obs.columns:
        df["pxl_row_in_fullres"] = final_test_rna.obs["pxl_row_in_fullres"].values
    else:
        df["pxl_row_in_fullres"] = all_rows.astype(float)

    if "pxl_col_in_fullres" in final_test_rna.obs.columns:
        df["pxl_col_in_fullres"] = final_test_rna.obs["pxl_col_in_fullres"].values
    else:
        df["pxl_col_in_fullres"] = all_cols.astype(float)

    for i, pname in enumerate(protein_names):
        df[pname] = preds[:, i]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved final test predictions to: {output_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Final test inference (slice2) with saved MLP reducer + trained GNN")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to gnn_best.pt")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()

    return run_prediction_final(
        config_path=args.config,
        gnn_checkpoint_path=args.checkpoint,
        output_path=args.output,
        exp_name=args.exp_name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
