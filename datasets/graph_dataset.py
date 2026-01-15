"""
Graph dataset utilities and builders.

Provides a simple BaseDataset for extension plus GraphDataset static methods
that wrap graph construction and neighbor loading helpers.
"""

import os
import pickle
import time
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial import KDTree
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm.auto import tqdm


class BaseDataset:
    """Lightweight base class for dataset modules."""

    def __init__(self, config: dict):
        self.config = config


class GraphDataset(BaseDataset):
    """Static helpers for building and sampling large graphs."""

    @staticmethod
    def build_graph_from_config(
        pos_array: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        test_mask: np.ndarray,
        config: dict,
        base_dir: str,
        for_eval: bool = False,
    ) -> Data:
        """
        Convenience wrapper to build a graph using config-driven parameters and caching.
        Keeps dataset orchestration light by consolidating graph construction concerns.
        """
        cache_name = "graph_cache.pkl" if not for_eval else "graph_cache_pred.pkl"
        cache_path = None
        if config["graph"].get("cache_graph", True):
            cache_path = os.path.join(base_dir, cache_name)

        return GraphDataset.build_graph_chunked(
            pos_array,
            features,
            targets,
            train_mask,
            val_mask,
            test_mask,
            radius=config["graph"]["radius"],
            max_neighbors=config["graph"].get("max_neighbors", 6),
            use_edge_attr=config["graph"].get("use_edge_attr", False),
            self_loops=config["graph"].get("self_loops", True),
            chunk_size=config["graph"].get("build_batch_size", 10000),
            cache_path=cache_path,
        )

    @staticmethod
    def build_graph_chunked(
        pos_array: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        test_mask: np.ndarray,
        radius: float,
        max_neighbors: int = 6,
        use_edge_attr: bool = False,
        self_loops: bool = True,
        chunk_size: int = 10000,
        cache_path: str = None,
    ) -> Data:
        """Build graph in chunks for memory efficiency."""
        N = len(pos_array)

        if cache_path and os.path.exists(cache_path):
            if os.path.getsize(cache_path) > 0:
                print(f"✓ Loading cached graph from {cache_path}...")
                start = time.time()
                try:
                    with open(cache_path, "rb") as f:
                        data = pickle.load(f)
                    print(f"  Loaded in {time.time()-start:.2f}s")
                    print(f"  Nodes: {data.num_nodes:,}, Edges: {data.num_edges:,}")
                    return data
                except (EOFError, pickle.UnpicklingError, AttributeError, ImportError) as e:
                    print(
                        f"  Cache file at {cache_path} is invalid or corrupted. Rebuilding... (Error: {e})"
                    )
            else:
                print(f"  Empty cache file found at {cache_path}. Rebuilding...")

        print(f"Building large-scale graph: {N:,} nodes, radius={radius}")
        print(f"Parameters: max_neighbors={max_neighbors}, edge_attr={use_edge_attr}")
        print(f"Processing in chunks of {chunk_size:,} nodes...")

        # Build KDTree once
        print("Building KDTree index...", end=" ", flush=True)
        tree_start = time.time()
        tree = KDTree(pos_array)
        print(f"Done in {time.time()-tree_start:.2f}s")

        # Preallocate with estimated size
        estimated_edges = N * max_neighbors
        edge_list = np.zeros((estimated_edges, 2), dtype=np.int64)
        edge_attr_list = np.zeros((estimated_edges, 1), dtype=np.float32) if use_edge_attr else None
        edge_count = 0

        # Process in chunks with vectorized operations
        num_chunks = (N + chunk_size - 1) // chunk_size

        graph_start = time.time()
        for chunk_idx in tqdm(range(num_chunks), desc="Building graph"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, N)

            # Query neighbors for all nodes in chunk
            for i in range(start_idx, end_idx):
                indices = tree.query_ball_point(pos_array[i], r=radius)

                # Handle self-loops
                if self_loops and i not in indices:
                    indices.append(i)
                elif not self_loops:
                    indices = [idx for idx in indices if idx != i]

                selected_dists = None

                # Limit neighbors and compute distances
                if len(indices) > max_neighbors:
                    neighbor_positions = pos_array[indices]
                    dists = np.linalg.norm(neighbor_positions - pos_array[i], axis=1)
                    k_th = min(max_neighbors, len(dists) - 1)
                    top_k_idx = np.argpartition(dists, k_th)[:max_neighbors]
                    indices = [indices[idx] for idx in top_k_idx]
                    if use_edge_attr:
                        selected_dists = dists[top_k_idx]
                else:
                    if use_edge_attr and len(indices) > 0:
                        selected_dists = np.linalg.norm(pos_array[indices] - pos_array[i], axis=1)
                    elif use_edge_attr:
                        selected_dists = np.array([], dtype=np.float32)

                # Add edges (batch)
                n_edges = len(indices)
                if n_edges == 0:
                    continue

                if edge_count + n_edges > len(edge_list):
                    new_size = max(len(edge_list) * 2, edge_count + n_edges)
                    edge_list = np.resize(edge_list, (new_size, 2))
                    if use_edge_attr:
                        edge_attr_list = np.resize(edge_attr_list, (new_size, 1))

                edge_list[edge_count : edge_count + n_edges, 0] = i
                edge_list[edge_count : edge_count + n_edges, 1] = indices

                if use_edge_attr:
                    if selected_dists is not None:
                        edge_attr_list[edge_count : edge_count + n_edges, 0] = selected_dists
                    else:
                        edge_attr_list[edge_count : edge_count + n_edges, 0] = 0.0

                edge_count += n_edges

        graph_time = time.time() - graph_start
        print(f"Graph construction: {graph_time:.2f}s ({edge_count:,} edges)")

        edge_list = edge_list[:edge_count]
        if use_edge_attr:
            edge_attr_list = edge_attr_list[:edge_count]

        print(f"Total edges: {len(edge_list):,}")
        print("Converting to PyTorch tensors...", end=" ", flush=True)
        tensor_start = time.time()

        edge_index = torch.from_numpy(edge_list.T).contiguous().long()
        edge_attr = None
        if use_edge_attr and edge_attr_list is not None:
            edge_attr = torch.from_numpy(edge_attr_list).float()

        x = torch.from_numpy(features).float()
        y = torch.from_numpy(targets).float()
        pos = torch.from_numpy(pos_array).float()
        train_mask_t = torch.from_numpy(train_mask).bool()
        val_mask_t = torch.from_numpy(val_mask).bool()
        test_mask_t = torch.from_numpy(test_mask).bool()

        print(f"Done in {time.time()-tensor_start:.2f}s")

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos,
            train_mask=train_mask_t,
            val_mask=val_mask_t,
            test_mask=test_mask_t,
        )

        print(f"Graph summary: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
        if data.num_nodes > 0:
            print(f"Avg degree: {data.num_edges / data.num_nodes:.2f}")
        else:
            print("Avg degree: N/A (0 nodes)")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            print(f"Caching graph to {cache_path}...", end=" ", flush=True)
            cache_start = time.time()
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                cache_size_mb = os.path.getsize(cache_path) / 1024 / 1024
                print(f"Done in {time.time()-cache_start:.2f}s ({cache_size_mb:.1f} MB)")
            except Exception as e:
                print(f"Failed to cache graph: {e}")

        return data

    @staticmethod
    def create_neighbor_loader(
        data: Data,
        config: dict,
        indices: torch.Tensor = None,
        shuffle: bool = False,
    ) -> NeighborLoader:
        """Create NeighborLoader for mini-batch training on large graphs."""
        if indices is None:
            indices = torch.arange(data.num_nodes)

        batch_size = config["graph"].get("batch_size_nodes", 2048)
        num_neighbors = config["graph"].get("fanout", [10, 10])
        num_workers = config["training"].get("num_workers", 4)

        loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=indices,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
        )
        return loader

    @staticmethod
    def estimate_graph_memory(
        num_nodes: int, num_edges: int, feature_dim: int, output_dim: int
    ) -> dict:
        """Estimate memory requirements for graph."""
        bytes_per_float = 4
        bytes_per_long = 8
        bytes_per_bool = 1

        features_mb = num_nodes * feature_dim * bytes_per_float / 1024 / 1024
        targets_mb = num_nodes * output_dim * bytes_per_float / 1024 / 1024
        pos_mb = num_nodes * 2 * bytes_per_float / 1024 / 1024
        masks_mb = num_nodes * 3 * bytes_per_bool / 1024 / 1024
        edge_index_mb = num_edges * 2 * bytes_per_long / 1024 / 1024
        total_mb = features_mb + targets_mb + pos_mb + masks_mb + edge_index_mb

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "features_mb": features_mb,
            "targets_mb": targets_mb,
            "edge_index_mb": edge_index_mb,
            "total_mb": total_mb,
            "total_gb": total_mb / 1024,
        }
