"""
Modular dimensionality reduction strategies.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from datasets.processing import preprocessing as pp


class BaseReducer:
    """Interface for reducers."""

    def fit_transform(self, train_rna_proc, test_rna_proc=None) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

class MLPReducer(BaseReducer):
    def __init__(self, k_reduced: int, full_config: dict, random_seed: int, device, logger=None):
        self.k_reduced = k_reduced
        self.full_config = full_config
        self.random_seed = random_seed
        self.device = pp._normalize_device(device)
        self.logger = logger

    def fit_transform(self, train_rna_proc, test_rna_proc=None):
        return pp._apply_mlp_reduction(
            train_rna_proc,
            test_rna_proc,
            self.k_reduced,
            self.full_config,
            self.random_seed,
            self.device,
            self.logger.info if self.logger else print,
        )


def build_reducer(reducer_name: str, config: dict, logger=None) -> BaseReducer:

    preprocess_cfg = config.get("preprocessing", config)
    k_reduced = preprocess_cfg.get("k_pca", 512)
    random_seed = config.get("random_seed", preprocess_cfg.get("random_seed", 42))
    device = config.get("device", preprocess_cfg.get("device"))

    if reducer_name == "mlp":
        if device is None:
            raise ValueError("MLP reducer requires 'device' in config.")
        return MLPReducer(k_reduced, config, random_seed, device, logger=logger)

    raise ValueError(f"Unknown reduction method: {reducer_name}")
