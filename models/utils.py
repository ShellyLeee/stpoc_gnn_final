"""Model builders."""

from models.gnn import build_gnn_model


def build_model(config: dict, C_in: int, C_out: int, meta: dict, logger=None):
    """
    Instantiate GNN model using the factory function.
    Automatically selects GNNModel or GNNWithSkipConnections based on config.
    """
    
    model_cfg = config["model"].copy()
    
    model_args = {
        **model_cfg, 
        "in_channels": C_in,
        "out_channels": C_out,
        "fusion_slices": meta.get("fusion", {}).get("slices"),
        "fusion_cfg": config.get("fusion", {"type": "concat_mlp"}),
        "use_edge_weights": config["graph"].get("use_edge_attr", False),
        "logger": logger
    }

    return build_gnn_model(model_args, C_in, C_out)
