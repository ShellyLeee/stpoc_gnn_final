#!/usr/bin/env python
"""Prediction helpers (no top-level run logic)."""

import time
import copy
import numpy as np
import torch
import scanpy as sc
from sklearn.preprocessing import StandardScaler

from datasets.graph_dataset import GraphDataset


@torch.no_grad()
def infer_full_graph(model, graph_data, device, config, batch_size_nodes, fanout):
    """Perform mini-batch inference on the full graph using NeighborLoader."""
    model.eval()
    all_indices = torch.arange(graph_data.num_nodes)

    cfg_for_loader = copy.deepcopy(config)
    cfg_for_loader.setdefault("graph", {})
    cfg_for_loader["graph"]["batch_size_nodes"] = batch_size_nodes
    cfg_for_loader["graph"]["fanout"] = fanout

    is_cuda = (isinstance(device, str) and device.startswith("cuda")) or (
        hasattr(device, "type") and device.type == "cuda"
    )
    cfg_for_loader.setdefault("training", {})
    cfg_for_loader["training"].setdefault("num_workers", 0 if not is_cuda else 4)
    cfg_for_loader["training"].setdefault("pin_memory", bool(is_cuda))

    loader = GraphDataset.create_neighbor_loader(graph_data, cfg_for_loader, all_indices, shuffle=False)

    C_out = graph_data.y.shape[1]
    preds = torch.zeros(graph_data.num_nodes, C_out, device="cpu")

    t0 = time.time()
    from tqdm import tqdm

    for batch in tqdm(loader, desc="Inference"):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        bs = batch.batch_size
        out_target = out[:bs]
        global_ids = batch.n_id[:bs].cpu()
        preds[global_ids] = out_target.detach().cpu()
    t1 = time.time()
    return preds.numpy(), (t1 - t0)


def maybe_reverse_preprocessing(test_pred, config, logger):
    """Reverse log1p + z-score transformations (to match training consistency)."""
    if not config["preprocessing"].get("zscore_protein", False):
        return test_pred

    logger.info("Reversing z-score and log1p on protein predictions ...")
    train_pro = sc.read_h5ad(config["data"]["train_pro_h5ad"])
    pro_X_raw = np.asarray(train_pro.X)
    pro_X_processed = np.log1p(pro_X_raw)

    scaler = StandardScaler()
    scaler.fit(pro_X_processed)

    test_log1p = test_pred * scaler.scale_ + scaler.mean_
    test_orig = np.exp(test_log1p) - 1.0
    test_orig = np.maximum(test_orig, 0.0)
    return test_orig
