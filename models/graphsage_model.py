"""GraphSAGE model for per-client breast-cancer graph classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

import config


class GraphSAGENet(nn.Module):
    """2-layer GraphSAGE for node-level benign/malignant classification.

    Each client holds a k-NN similarity graph built over its own patient
    records (nodes = patients, edges = feature-similarity links). The model
    predicts a label per node (patient).
    """

    def __init__(
        self,
        in_channels: int = config.IN_CHANNELS,
        hidden_channels: int = config.HIDDEN_CHANNELS,
        out_channels: int = config.OUT_CHANNELS,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        out = self.classifier(h)
        return out  # raw logits, shape [num_nodes, 2]

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return the penultimate-layer node embeddings (used by the
        Explainability agent for feature attribution)."""
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return F.relu(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = GraphSAGENet()
    print(m)
    print("Trainable params:", count_parameters(m))
