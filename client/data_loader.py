"""
Loads the Breast Cancer Wisconsin (Diagnostic) dataset from a CSV file
(standard WDBC schema: id, diagnosis, 30 feature columns), partitions it
across NUM_CLIENTS federated clients, and builds a per-client k-NN
similarity graph (nodes = patient records, edges = feature-space k-nearest-
neighbors) for use with GraphSAGE.
"""
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

import config

# Canonical WDBC feature column order (30 features, matches UCI/Kaggle
# `data.csv`). id and diagnosis are metadata, not model inputs.
WDBC_FEATURE_COLUMNS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
    "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst",
    "symmetry_worst", "fractal_dimension_worst",
]


def load_wdbc_csv(csv_path: str = config.DATASET_CSV_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Reads the raw WDBC CSV and returns (X, y) where y=0 malignant, y=1 benign
    (matches sklearn's load_breast_cancer label convention for drop-in
    compatibility with the rest of the pipeline)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset CSV not found at {csv_path}. Place the WDBC CSV "
            f"(id, diagnosis, 30 feature columns) at this path, or update "
            f"config.DATASET_CSV_PATH."
        )
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")

    missing = [c for c in WDBC_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected WDBC feature columns: {missing}")

    X = df[WDBC_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df["diagnosis"].map({"M": 0, "B": 1}).to_numpy()
    if pd.isna(y).any():
        raise ValueError("diagnosis column contains values other than 'M'/'B'.")
    return X, y.astype(np.int64)


def _build_knn_graph(X: np.ndarray, y: np.ndarray, k: int) -> Data:
    """Build an undirected k-NN graph (by Euclidean distance in feature
    space) over a set of patient records and wrap it as a PyG `Data` object.
    """
    k_eff = max(1, min(k, len(X) - 1))
    adj = kneighbors_graph(X, n_neighbors=k_eff, mode="connectivity", include_self=False)
    adj = adj.maximum(adj.T)  # symmetrize -> undirected graph
    coo = adj.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)

    x = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y_t)


def _iid_partition(X: np.ndarray, y: np.ndarray, num_clients: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    shards = np.array_split(idx, num_clients)
    return shards


def _non_iid_partition(X: np.ndarray, y: np.ndarray, num_clients: int, seed: int, alpha: float = 0.5):
    """Dirichlet-based label-skewed partition (optional, config.NON_IID)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    client_idx = [[] for _ in range(num_clients)]
    for c in classes:
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        splits = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
        for cid, part in enumerate(np.split(c_idx, splits)):
            client_idx[cid].extend(part.tolist())
    return [np.array(sorted(idx)) for idx in client_idx]


def load_and_partition(
    num_clients: int = config.NUM_CLIENTS,
    test_split: float = config.DATASET_TEST_SPLIT,
    k: int = config.KNN_NEIGHBORS,
    non_iid: bool = config.NON_IID,
    seed: int = config.RANDOM_SEED,
    val_split: float = config.DATASET_VAL_SPLIT,
    holdout: str = "test",
) -> Dict[int, Tuple[Data, Data]]:
    """Returns {client_id: (train_graph, holdout_graph)}.

    Per client the records are split three ways with strict index
    separation:  test (``test_split`` of the client, chosen exactly as in the
    original pipeline, so test indices are unchanged), validation
    (``val_split`` of the remaining non-test records) and train.

    holdout="test": train graph = train + validation records (the full
        non-test data, identical to the original pipeline); second graph =
        test graph.  Used ONLY for final reporting.
    holdout="val":  train graph = train records only; second graph =
        validation graph.  Used for hyperparameter selection, so the test
        records are never seen (not even by the scaler) during tuning.

    Standardisation uses ONLY the client's own training-graph records, and
    each k-NN graph is built independently per split (no cross-split edges).
    """
    if holdout not in ("test", "val"):
        raise ValueError("holdout must be 'test' or 'val'")
    X, y = load_wdbc_csv()  # 569 samples, 30 features, {0: malignant, 1: benign}

    shards = (
        _non_iid_partition(X, y, num_clients, seed)
        if non_iid
        else _iid_partition(X, y, num_clients, seed)
    )

    client_graphs: Dict[int, Tuple[Data, Data]] = {}
    for cid, idx in enumerate(shards):
        cid_rng = np.random.default_rng(seed + cid)
        perm = cid_rng.permutation(len(idx))
        n_test = max(1, int(len(idx) * test_split))
        test_local = perm[:n_test]
        nontest_local = perm[n_test:]

        if holdout == "test":
            train_local, eval_local = nontest_local, test_local
        else:
            vrng = np.random.default_rng(10_000 + seed * 100 + cid)
            vperm = vrng.permutation(nontest_local)
            n_val = max(2, int(round(len(nontest_local) * val_split)))
            eval_local, train_local = vperm[:n_val], vperm[n_val:]

        client_X, client_y = X[idx], y[idx]
        scaler = StandardScaler().fit(client_X[train_local])
        client_X_scaled = scaler.transform(client_X)

        train_graph = _build_knn_graph(client_X_scaled[train_local], client_y[train_local], k)
        eval_graph = _build_knn_graph(client_X_scaled[eval_local], client_y[eval_local], k)
        client_graphs[cid] = (train_graph, eval_graph)

    return client_graphs


def split_indices(seed: int = config.RANDOM_SEED, num_clients: int = config.NUM_CLIENTS,
                  test_split: float = config.DATASET_TEST_SPLIT,
                  val_split: float = config.DATASET_VAL_SPLIT):
    """Global row indices of the train / val / test records per client
    (used by tests to prove the splits are disjoint)."""
    X, y = load_wdbc_csv()
    shards = _iid_partition(X, y, num_clients, seed)
    out = {}
    for cid, idx in enumerate(shards):
        perm = np.random.default_rng(seed + cid).permutation(len(idx))
        n_test = max(1, int(len(idx) * test_split))
        test_local, nontest_local = perm[:n_test], perm[n_test:]
        vperm = np.random.default_rng(10_000 + seed * 100 + cid).permutation(nontest_local)
        n_val = max(2, int(round(len(nontest_local) * val_split)))
        out[cid] = {"train": idx[vperm[n_val:]], "val": idx[vperm[:n_val]], "test": idx[test_local]}
    return out


if __name__ == "__main__":
    graphs = load_and_partition()
    for cid, (tr, te) in graphs.items():
        print(
            f"Client {cid}: train nodes={tr.num_nodes} edges={tr.num_edges} "
            f"| test nodes={te.num_nodes} edges={te.num_edges}"
        )
