#!/usr/bin/env python
"""Training entrypoint."""

import argparse
import json
import os
import shutil
from datetime import datetime
import scanpy as sc

import numpy as np
import torch
from tqdm.auto import tqdm
import yaml
from scipy.stats import spearmanr

from trainers import (
    EarlyStopping,
    evaluate_spearman,
    build_trainer,
)
from common.utils import set_seed
from models.utils import build_model
from common.loss import build_criterion
from common.optim import build_optimizer
from datasets.utils import build_dataset
from common.visualization import plot_training_history, plot_metric_history
from common.logging import setup_logging



def train(config_path: str, exp_name: str | None = None):

    # basic setup
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging_cfg = config.get("logging", {})
    base_log_dir = os.path.abspath(logging_cfg.get("log_dir", "logs"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = exp_name if exp_name else f"run_{timestamp}"
    run_dir = os.path.join(base_log_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)
    config["run_dir"] = run_dir

    log_file = os.path.join(run_dir, "train.log")

    logger = setup_logging(log_file)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    config_save_path = os.path.join(run_dir, "config.yaml")
    shutil.copy(config_path, config_save_path)

    set_seed(config["random_seed"])
    device = config["device"]

    logger.info("=" * 60)
    logger.info("LARGE-SCALE MULTIMODAL GNN TRAINING (WITH REGULARIZATION)")
    logger.info("=" * 60)

    # dataset
    graph_data, train_loader, val_loader, meta = build_dataset(config, logger, for_eval=False)

    strategy = meta["strategy"]
    rna_dim = meta["rna_dim"]
    C_out = meta["C_out"]
    C_in = meta["C_in"]
    first_layer_reduction = meta["first_layer_reduction"]

    logger.info("🔥 REGULARIZATION CONFIG:")
    logger.info(f"   Attention Dropout: {config['model'].get('attention_dropout', 0.2)}")
    logger.info(f"   Edge Dropout:      {config['model'].get('edge_dropout', 0.15)}")
    logger.info(f"   Smoothness Weight: {config['training'].get('smoothness_weight', 0.01)}")

    # Model
    model = build_model(config, C_in, C_out, meta).to(device)
    logger.info(f"Model created. Parameters: {model.count_parameters():,}")

    # loss
    criterion = build_criterion(config)

    # optimizer 
    optimizer, scheduler = build_optimizer(config, model)

    # Get protein names for co-expression regularizer
    pro = sc.read_h5ad(config['data']['train_pro_h5ad'])
    protein_names = list(map(str, pro.var_names))
    logger.info(f"Loaded {len(protein_names)} protein names for co-expression regularizer")


    # trainer
    trainer = build_trainer(model, criterion, optimizer, device, config, rna_dim, logger)

    # Build and attach co-expression regularizer
    if config.get('coexpression', {}).get('enabled', False):
        from common.coexpression import build_coexpression_regularizer
        trainer.coexp_regularizer = build_coexpression_regularizer(
            config, protein_names, device, logger
        )

    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
        min_delta=config["training"]["early_stopping_min_delta"],
        verbose=True,
        logger=logger,
    )

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_spearman_history = []
    val_spearman_history = []
    best_path = os.path.join(checkpoint_dir, "gnn_best.pt")
    stopped_epoch = None

    epoch_pbar = tqdm(range(1, config["training"]["epochs"] + 1), desc="Training")
    for epoch in epoch_pbar:
        
        # forward
        train_loss = trainer.train(train_loader, epoch)
        val_loss = trainer.validate(val_loader, epoch)

        if scheduler:
            scheduler.step()

        train_spearman = evaluate_spearman(model, train_loader, device, C_out, "train_mask", epoch)
        val_spearman = evaluate_spearman(model, val_loader, device, C_out, "val_mask", epoch)

        # logging
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_spearman_history.append(train_spearman)
        val_spearman_history.append(val_spearman)

        logger.info(
            f"Epoch {epoch:02d} - Train L: {train_loss:.5f}, Val L: {val_loss:.5f}, "
            f"Tr Sp: {train_spearman:.4f}, Val Sp: {val_spearman:.4f}"
        )
        epoch_pbar.set_postfix(
            tr_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}",
            tr_sp=f"{train_spearman:.4f}", val_sp=f"{val_spearman:.4f}"
        )

        if epoch % config["logging"]["save_interval"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                os.path.join(checkpoint_dir, f"gnn_epoch{epoch:02d}.pt"),
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "strategy": strategy,
                },
                best_path,
            )

        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch}")
            stopped_epoch = epoch
            break

    # Final evaluation & plotting
    logger.info("=" * 60)
    logger.info("STEP 6: Final evaluation (on best model)")
    logger.info("=" * 60)

    plot_training_history(train_losses, val_losses, os.path.join(run_dir, "training_history.png"))
    plot_metric_history(train_spearman_history, val_spearman_history, os.path.join(run_dir, "spearman_history.png"),metric_name="Spearman",)

    if os.path.exists(best_path):
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best model from epoch {checkpoint.get('epoch', 'N/A')}")
    else:
        logger.warning("Best model checkpoint not found! Using last epoch.")

    logger.info("Predicting on validation set...")
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation prediction"):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            val_mask_batch = batch.val_mask[: batch.batch_size]
            if val_mask_batch.sum() > 0:
                val_preds.append(out[: batch.batch_size][val_mask_batch].cpu().numpy())
                val_true.append(batch.y[: batch.batch_size][val_mask_batch].cpu().numpy())
    if len(val_preds) > 0:
        val_preds = np.vstack(val_preds)
        val_true = np.vstack(val_true)

    logger.info("Predicting on training set...")
    train_preds, train_true = [], []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Training prediction"):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            train_mask_batch = batch.train_mask[: batch.batch_size]
            if train_mask_batch.sum() > 0:
                train_preds.append(out[: batch.batch_size][train_mask_batch].cpu().numpy())
                train_true.append(batch.y[: batch.batch_size][train_mask_batch].cpu().numpy())
    if len(train_preds) > 0:
        train_preds = np.vstack(train_preds)
        train_true = np.vstack(train_true)

    # 5. Calculate Correlation (Spearman)
    # Validation
    val_spearman_rhos = []
    for j in range(C_out):
        try:
            rho, _ = spearmanr(val_true[:, j], val_preds[:, j])
            val_spearman_rhos.append(rho)
        except:
            val_spearman_rhos.append(np.nan)
    mean_val_spearman = np.nanmean(val_spearman_rhos)
    logger.info(f"Validation Mean Spearman (Best Model): {mean_val_spearman:.4f}")

    # Training
    train_spearman_rhos = []
    for j in range(C_out):
        try:
            rho, _ = spearmanr(train_true[:, j], train_preds[:, j])
            train_spearman_rhos.append(rho)
        except:
            train_spearman_rhos.append(np.nan)
    mean_train_spearman = np.nanmean(train_spearman_rhos)
    logger.info(f"Training Mean Spearman (Best Model): {mean_train_spearman:.4f}")

    metrics = {
        "strategy": strategy,
        "training": {
            "epochs_trained": stopped_epoch or config["training"]["epochs"],
            "best_val_loss": float(best_val_loss),
            "best_model_epoch": int(checkpoint.get("epoch", 0)) if os.path.exists(best_path) else 0,
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) for x in val_losses],
            "train_spearman_history": [float(x) for x in train_spearman_history],
            "val_spearman_history": [float(x) for x in val_spearman_history],
        },
        "evaluation_best_model": {
            "train_mean_spearman": float(mean_train_spearman),
            "val_mean_spearman": float(mean_val_spearman),
            "train_spearman_per_protein": [float(x) for x in train_spearman_rhos],
            "val_spearman_per_protein": [float(x) for x in val_spearman_rhos],
        },
        "model": {
            "parameters": model.count_parameters(),
            "input_dim": C_in,
            "first_layer_reduction": first_layer_reduction,
        },
        "graph": {"num_nodes": int(graph_data.num_nodes), "num_edges": int(graph_data.num_edges)},
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train multimodal GNN")
    default_cfg = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "cfgs", "gnn.yaml")
    )
    parser.add_argument("--config", type=str, default=default_cfg)
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()
    train(args.config, args.exp_name)


if __name__ == "__main__":
    main()
