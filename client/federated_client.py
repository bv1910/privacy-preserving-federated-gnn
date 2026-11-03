"""FederatedClient: local GraphSAGE training on one hospital's k-NN graph."""
from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

import config
from crypto.fpe_engine import FPEEngine
from attacks.attack_types import apply_attack, label_flip_graph


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        train_graph: Data,
        test_graph: Data,
        model_class: Type[nn.Module],
        is_malicious: bool = False,
        attack_type: str = "scale_gaussian",
        attack_kwargs: Optional[Dict] = None,
    ):
        """
        attack_type: which malicious-update strategy this client uses if
        is_malicious=True. Defaults to "scale_gaussian", which reproduces
        this pipeline's original (pre-experiments) attack exactly, so
        main.py / existing tests are unaffected. Other options
        ("sign_flip", "scale", "gaussian", "adaptive", "label_flip") are
        used by experiments/run_experiments.py to evaluate the defense
        pipeline against a range of attack families, not just one.
        """
        self.client_id = client_id
        self.train_graph = train_graph
        self.test_graph = test_graph
        self.model = model_class()
        self.is_malicious = is_malicious  # simulates a Byzantine / adversarial client
        self.attack_type = attack_type
        self.attack_kwargs = dict(attack_kwargs or {})
        self.last_train_loss = float("nan")  # final local-epoch training loss
        self.fpe = FPEEngine()

        # Active training hyperparameters — start at config defaults, then
        # updated each round from the server's FPE-encrypted broadcast.
        self.lr = config.LEARNING_RATE
        self.epochs = config.LOCAL_EPOCHS

    def local_train(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Loads the global weights, trains locally, returns the weight
        delta Δw_k = w_local - w_global for each parameter. Uses this
        client's currently active (decrypted) hyperparameters unless
        overridden explicitly."""
        epochs = epochs if epochs is not None else self.epochs
        lr = lr if lr is not None else self.lr

        self.model.load_state_dict(global_state_dict)
        global_flat = {k: v.clone() for k, v in global_state_dict.items()}

        train_graph = self.train_graph
        if self.is_malicious and self.attack_type == "label_flip":
            # Data-poisoning attack happens before training, not on the
            # resulting delta -- flip labels, then train normally.
            train_graph = label_flip_graph(train_graph)

        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        for _ in range(epochs):
            optimizer.zero_grad()
            out = self.model(train_graph.x, train_graph.edge_index)
            loss = F.cross_entropy(out, train_graph.y)
            loss.backward()
            optimizer.step()
        self.last_train_loss = float(loss.detach())

        delta = {
            name: (param.detach() - global_flat[name])
            for name, param in self.model.state_dict().items()
        }

        if self.is_malicious and self.attack_type != "label_flip":
            # Model-poisoning attack: corrupt the delta itself according to
            # this client's configured attack_type (default "scale_gaussian"
            # reproduces the pipeline's original fixed attack).
            delta = apply_attack(delta, self.attack_type, **self.attack_kwargs)

        return delta

    def apply_global_update(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        encrypted_hyperparams: Optional[Dict[str, str]] = None,
    ) -> None:
        """Loads the new global weights. If the server sent FPE-encrypted
        training hyperparameters alongside the update, decrypts them
        client-side and adopts them for the next local_train() call."""
        self.model.load_state_dict(global_state_dict)
        if encrypted_hyperparams is not None:
            decrypted = self.fpe.decrypt_hyperparams(encrypted_hyperparams)
            self.lr = decrypted["learning_rate"]
            self.epochs = int(decrypted["local_epochs"])

    @torch.no_grad()
    def evaluate(self, model: nn.Module = None) -> Tuple[float, float]:
        """Runs on this client's held-out test graph. Returns (accuracy, loss)."""
        m = model if model is not None else self.model
        m.eval()
        out = m(self.test_graph.x, self.test_graph.edge_index)
        loss = F.cross_entropy(out, self.test_graph.y).item()
        preds = out.argmax(dim=1)
        acc = (preds == self.test_graph.y).float().mean().item()
        return acc, loss
