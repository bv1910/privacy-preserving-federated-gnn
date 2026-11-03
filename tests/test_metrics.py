import torch
from torch_geometric.data import Data

from evaluation.metrics import compute_metrics, aggregate_client_metrics
from models.graphsage_model import GraphSAGENet


def _toy_graph(n=20, in_channels=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_channels, generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    # simple ring graph so every node has >=2 neighbors
    src = torch.arange(n)
    dst = (src + 1) % n
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    return Data(x=x, edge_index=edge_index, y=y)


def test_compute_metrics_returns_all_fields():
    model = GraphSAGENet()
    graph = _toy_graph()
    result = compute_metrics(model, graph)
    d = result.as_dict()
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp"]:
        assert key in d

    assert 0.0 <= d["accuracy"] <= 1.0
    assert 0.0 <= d["precision"] <= 1.0
    assert 0.0 <= d["recall"] <= 1.0
    assert d["tn"] + d["fp"] + d["fn"] + d["tp"] == graph.y.shape[0]


def test_single_class_graph_does_not_crash():
    model = GraphSAGENet()
    graph = _toy_graph()
    graph.y = torch.zeros_like(graph.y)  # only one class present
    result = compute_metrics(model, graph)
    import math
    # single-class shard -> P/R/F1/AUC undefined (NaN), excluded from macro-avg
    assert math.isnan(result.precision) and math.isnan(result.recall)
    assert math.isnan(result.f1) and math.isnan(result.roc_auc)


def test_aggregate_client_metrics_averages():
    model = GraphSAGENet()
    graphs = [_toy_graph(seed=i) for i in range(3)]
    per_client = [compute_metrics(model, g) for g in graphs]
    agg = aggregate_client_metrics(per_client)
    manual_acc = sum(m.accuracy for m in per_client) / len(per_client)
    assert abs(agg["accuracy"] - manual_acc) < 1e-9
