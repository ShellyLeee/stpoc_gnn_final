"""Optimizer builders."""

import torch


def build_optimizer(config: dict, model: torch.nn.Module):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = None
    if config["scheduler"]["type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["epochs"]
        )
    return optimizer, scheduler
