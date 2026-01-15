from trainers.base_trainer import BaseTrainer, EarlyStopping, evaluate_spearman


def _resolve_trainer_type(config: dict) -> str:
    trainer_cfg = config.get("trainer", "baseline")
    if isinstance(trainer_cfg, str):
        return trainer_cfg.lower()
    return str(trainer_cfg.get("type", "baseline")).lower()


def build_trainer(model, criterion, optimizer, device, config, rna_dim, logger=None):
    """Factory to construct trainer implementations based on config."""
    trainer_type = _resolve_trainer_type(config)
    if trainer_type == "baseline":
        return BaseTrainer(model, criterion, optimizer, device, config, rna_dim, logger)
    raise ValueError(f"Unknown trainer type: {trainer_type}")
