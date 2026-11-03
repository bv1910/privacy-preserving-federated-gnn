import numpy as np
import torch

import config
from client.data_loader import load_and_partition, load_wdbc_csv
from client.federated_client import FederatedClient
from models.graphsage_model import GraphSAGENet


def test_wdbc_csv_loads_expected_shape():
    X, y = load_wdbc_csv()
    assert X.shape == (569, 30)
    assert set(np.unique(y)) <= {0, 1}


def test_partition_no_overlap_and_split_ratio():
    graphs = load_and_partition()
    assert len(graphs) == config.NUM_CLIENTS
    for cid, (train_g, test_g) in graphs.items():
        total = train_g.num_nodes + test_g.num_nodes
        ratio = test_g.num_nodes / total
        assert 0.1 <= ratio <= 0.35  # ~20% target, tolerant of rounding on small shards


def test_local_train_produces_valid_delta():
    graphs = load_and_partition()
    train_g, test_g = graphs[0]
    client = FederatedClient(0, train_g, test_g, GraphSAGENet)
    global_model = GraphSAGENet()
    delta = client.local_train(global_model.state_dict())
    for name, d in delta.items():
        assert isinstance(d, torch.Tensor)
        assert not torch.isnan(d).any()


def test_apply_global_update_decrypts_hyperparams():
    from crypto.fpe_engine import FPEEngine

    graphs = load_and_partition()
    train_g, test_g = graphs[0]
    client = FederatedClient(0, train_g, test_g, GraphSAGENet)
    global_model = GraphSAGENet()

    fpe = FPEEngine()
    encrypted = fpe.encrypt_hyperparams(learning_rate=0.02, local_epochs=5)
    client.apply_global_update(global_model.state_dict(), encrypted)

    assert abs(client.lr - 0.02) < 1e-9
    assert client.epochs == 5


if __name__ == "__main__":
    test_wdbc_csv_loads_expected_shape()
    test_partition_no_overlap_and_split_ratio()
    test_local_train_produces_valid_delta()
    test_apply_global_update_decrypts_hyperparams()
    print("test_client.py: all tests passed")
