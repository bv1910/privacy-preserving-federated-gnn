"""
Full classification metrics for the global model, computed on each client's
held-out test graph and pooled across clients.

Deliberately independent of accuracy alone: a paper comparing an encrypted
pipeline against a plaintext baseline needs precision/recall/F1/AUC and a
confusion matrix, since accuracy alone can hide class-imbalance effects
(WDBC is ~63% benign / 37% malignant) and is not sufficient evidence of
equivalence for a publication-grade comparison.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)


@dataclass
class MetricResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_dict(self) -> Dict:
        return asdict(self)


@torch.no_grad()
def compute_metrics(model, graph) -> MetricResult:
    """Computes the full metric suite for one client's test graph.

    Positive class = malignant (y=0 in this pipeline's convention -> we
    treat label 0 as the positive/"disease present" class for precision/
    recall, matching standard breast-cancer-diagnosis reporting).
    """
    model.eval()
    logits = model(graph.x, graph.edge_index)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()
    y_true = graph.y.numpy()

    # Flip convention so "malignant" (0) is the positive class for
    # precision/recall/F1, which is the clinically meaningful direction
    # (missing a malignant case = false negative).
    y_true_pos = 1 - y_true
    preds_pos = 1 - preds
    prob_pos = probs[:, 0]

    acc = accuracy_score(y_true, preds)

    # Guard against a test shard containing only one class (small per-client
    # splits can do this) -- precision/recall/AUC are undefined in that case.
    if len(np.unique(y_true_pos)) < 2:
        # Undefined for a single-class shard: use NaN (not 0.0) so the
        # shard is excluded from the P/R/F1 macro-average exactly like it
        # already is for ROC-AUC, instead of dragging the mean down.
        precision = recall = f1 = float("nan")
        auc = float("nan")
        tn = fp = fn = tp = 0
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_pos, preds_pos, average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(y_true_pos, prob_pos)
        except ValueError:
            auc = float("nan")
        cm = confusion_matrix(y_true_pos, preds_pos, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

    return MetricResult(
        accuracy=float(acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(auc),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def aggregate_client_metrics(per_client: List[MetricResult]) -> Dict[str, float]:
    """Macro-averages metrics across clients (each client weighted equally,
    matching how the federated average itself treats clients)."""
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    out = {}
    for k in keys:
        vals = [getattr(m, k) for m in per_client if not np.isnan(getattr(m, k))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


# ---------------------------------------------------------------------------
# Pooled (micro) evaluation over the union of all clients' held-out nodes.
#
# Why: each client's test graph holds only ~11 patients (~4 malignant).
# Per-shard precision/recall/F1 on that few positives is extremely noisy
# (one shard with a single malignant case contributes F1 in {0, 1}), shards
# with a single class are dropped, and zero_division=0 penalises shards with
# no positive prediction.  The pooled metrics below score every held-out
# patient exactly once with the same global model -- the standard estimate
# of the global model's performance.  The original macro metrics are still
# computed and stored next to them (``client_macro_*``) for transparency.
#
# Decision threshold: argmax over the 2 softmax outputs, i.e. "malignant"
# iff P(malignant) > 0.5.  This is NOT tuned anywhere.
# Positive class: malignant (dataset label 0).
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_graph(model, graph):
    """Returns (y_true_malignant {0,1}, p_malignant, pred_malignant {0,1}, ce_loss_sum)."""
    model.eval()
    logits = model(graph.x, graph.edge_index)
    probs = torch.softmax(logits, dim=1)
    ce_sum = float(torch.nn.functional.cross_entropy(logits, graph.y, reduction="sum"))
    y_pos = (1 - graph.y).numpy().astype(int)
    pred_pos = (1 - logits.argmax(dim=1)).numpy().astype(int)
    return y_pos, probs[:, 0].numpy().astype(float), pred_pos, ce_sum


def binary_metrics(y_true_pos, prob_pos, pred_pos) -> Dict[str, float]:
    """Accuracy / precision / recall / F1 / ROC-AUC + confusion matrix with
    malignant as the positive class."""
    y_true_pos = np.asarray(y_true_pos).astype(int)
    pred_pos = np.asarray(pred_pos).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_pos, pred_pos, average="binary", pos_label=1, zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_true_pos, prob_pos))
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true_pos, pred_pos, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true_pos, pred_pos)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "roc_auc": auc, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true_pos)),
    }


@torch.no_grad()
def evaluate_federation(model, graphs) -> Dict:
    """Scores the global model on a list of per-client held-out graphs.
    Returns pooled metrics, the original per-client macro metrics
    (prefixed ``client_macro_``), pooled cross-entropy loss, and the pooled
    labels/probabilities (for ROC / PR / confusion-matrix plots)."""
    ys, ps, preds, ce, n = [], [], [], 0.0, 0
    for g in graphs:
        y, p, pr, c = predict_graph(model, g)
        ys.append(y); ps.append(p); preds.append(pr); ce += c; n += len(y)
    y, p, pr = np.concatenate(ys), np.concatenate(ps), np.concatenate(preds)
    out = binary_metrics(y, p, pr)
    out["loss"] = ce / max(n, 1)
    macro = aggregate_client_metrics([compute_metrics(model, g) for g in graphs])
    for k, v in macro.items():
        out["client_macro_" + k] = v
    out["_y_true"] = y.tolist()
    out["_p_malignant"] = [round(float(v), 6) for v in p]
    return out
