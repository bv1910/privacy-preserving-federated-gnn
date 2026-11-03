"""
Generates every figure from the measured runs (results/raw/runs.jsonl) via
experiments/tables.py.  Nothing is typed in by hand.

    python -m experiments.make_plots      -> results/figures/*.png
"""
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,  # noqa: E402
                             precision_recall_curve, roc_curve)

import config  # noqa: E402
from experiments import results_io as rio  # noqa: E402
from experiments import tables as T  # noqa: E402
from experiments.pipeline_runner import RunConfig, original_config  # noqa: E402

OUT = config.FIGURES_DIR
M, ML = rio.METRICS, rio.METRIC_LABELS
C_CKKS, C_PLAIN, C_BASE = "#1f77b4", "#ff7f0e", "#7f7f7f"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 9, "figure.dpi": 110})
MADE = []


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    MADE.append(name)


def _sel_note(cfg):
    return (f"rounds={cfg.rounds}, local epochs={cfg.local_epochs}, lr={cfg.lr}, k={cfg.knn_k}, "
            f"hidden={cfg.hidden}, dropout={cfg.dropout}")


# --------------------------------------------------------------------------
def fig_rounds(cfg, nseeds):
    rc = T.round_curves(cfg)
    base = T.round_curves(original_config())
    # 01 accuracy vs rounds
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for df, lab, col, ls in ((rc, "Selected", None, "-"), (base, "Original config", None, "--")):
        for arm, c in (("CKKS", C_CKKS), ("Plaintext", C_PLAIN)):
            d = df[(df.quantity == "accuracy") & (df.aggregation == arm)]
            if d.empty:
                continue
            ax.plot(d["round"], d["mean"], ls, color=c if lab == "Selected" else (C_BASE if arm == "CKKS" else "#bcbd22"),
                    marker="o" if arm == "CKKS" else None, ms=3, label=f"{lab} - {arm}")
            ax.fill_between(d["round"], d["mean"] - d["std"], d["mean"] + d["std"], alpha=.12,
                            color=c if lab == "Selected" else C_BASE)
    ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy (pooled)")
    ax.set_title(f"Test accuracy vs communication rounds (mean ± std, {nseeds} seeds)")
    ax.grid(alpha=.3); ax.legend(); save(fig, "01_accuracy_vs_rounds.png")

    # 02 loss vs rounds
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for q, lab, ls in (("loss", "Test cross-entropy", "-"), ("train_loss", "Honest-client training loss", "--")):
        for arm, c in (("CKKS", C_CKKS), ("Plaintext", C_PLAIN)):
            d = rc[(rc.quantity == q) & (rc.aggregation == arm)]
            ax.plot(d["round"], d["mean"], ls, color=c, label=f"{lab} - {arm}")
            ax.fill_between(d["round"], d["mean"] - d["std"], d["mean"] + d["std"], alpha=.1, color=c)
    ax.set_xlabel("Communication round"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Loss vs communication rounds (selected config, {nseeds} seeds)")
    ax.grid(alpha=.3); ax.legend(); save(fig, "02_loss_vs_rounds.png")

    # 17 CKKS numerical error per round
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for q, lab in (("ckks_max_abs_err", "max |error|"), ("ckks_rmse", "RMSE")):
        d = rc[(rc.quantity == q) & (rc.aggregation == "CKKS")]
        ax.errorbar(d["round"], d["mean"], yerr=d["std"], marker="o", ms=3, capsize=2, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("Communication round")
    ax.set_ylabel("|CKKS aggregate − plaintext mean|")
    ax.set_title("CKKS numerical error per round (aggregated update, all parameters)")
    ax.grid(alpha=.3, which="both"); ax.legend(); save(fig, "17_ckks_error_per_round.png")


def fig_ckks_vs_plain(nseeds):
    ba = T.before_after()
    labels = list(ba["configuration"])
    colors = [C_BASE if "Original" in l and "CKKS" in l else "#c7c7c7" if "Original" in l else
              C_CKKS if "CKKS" in l else C_PLAIN for l in labels]
    for i, m in enumerate(M, start=3):  # 03..07
        fig, ax = plt.subplots(figsize=(7, 4.3))
        ax.bar(range(len(ba)), ba[m + "_mean"], yerr=ba[m + "_std"], capsize=4, color=colors)
        ps = T.per_seed_final()
        for j, l in enumerate(labels):
            model = "Original baseline" if "Original" in l else "Selected"
            arm = "CKKS" if "CKKS" in l else "Plaintext"
            v = ps[(ps.model == model) & (ps.aggregation == arm)][m]
            ax.scatter(np.full(len(v), j) + np.random.default_rng(0).uniform(-.15, .15, len(v)), v,
                       s=10, color="k", alpha=.5, zorder=3)
        for j, (v, e) in enumerate(zip(ba[m + "_mean"], ba[m + "_std"])):
            ax.text(j + 0.22, v, f"{v:.3f}", ha="left", va="center", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(ba))); ax.set_xticklabels([l.replace(" - ", "\n") for l in labels], fontsize=9)
        lo = max(0, float((ba[m + "_mean"] - ba[m + "_std"]).min()) - 0.1)
        ax.set_ylim(lo, 1.02); ax.set_ylabel(f"Test {ML[m]}")
        ax.set_title(f"{ML[m]}: CKKS vs plaintext FedAvg (mean ± std, dots = seeds, n={nseeds})")
        ax.grid(axis="y", alpha=.3); save(fig, f"{i:02d}_{m}_ckks_vs_plaintext.png")

    # 08 complete comparison
    cv = T.ckks_vs_plaintext("selected")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(M)); w = .38
    ax.bar(x - w / 2, cv.ckks_mean, w, yerr=cv.ckks_std, capsize=3, label="CKKS aggregation", color=C_CKKS)
    ax.bar(x + w / 2, cv.plaintext_mean, w, yerr=cv.plaintext_std, capsize=3, label="Plaintext FedAvg", color=C_PLAIN)
    for xi, a, b, ea, eb in zip(x, cv.ckks_mean, cv.plaintext_mean, cv.ckks_std, cv.plaintext_std):
        ax.text(xi - w / 2, a + ea + .008, f"{a:.3f}", ha="center", fontsize=8, rotation=90, va="bottom")
        ax.text(xi + w / 2, b + eb + .008, f"{b:.3f}", ha="center", fontsize=8, rotation=90, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in M]); ax.set_ylim(0.5, 1.12)
    ax.set_ylabel("Score (test, pooled)"); ax.legend(loc="lower right")
    ax.set_title(f"Selected config: all metrics, CKKS vs plaintext (same {nseeds} seeds)")
    ax.grid(axis="y", alpha=.3); save(fig, "08_all_metrics_ckks_vs_plaintext.png")


def fig_ablation(nseeds):
    ab = T.ablation()
    if ab.empty:
        return
    names = [n.split(". ", 1)[1] if ". " in n else n for n in ab.configuration]
    for i, m in enumerate(M, start=9):  # 09..13
        fig, ax = plt.subplots(figsize=(8, 4.5))
        cols = [C_CKKS] + ["#9ecae1"] * (len(ab) - 1)
        ax.bar(range(len(ab)), ab[m + "_mean"], yerr=ab[m + "_std"], capsize=4, color=cols)
        for j, (v, e) in enumerate(zip(ab[m + "_mean"], ab[m + "_std"])):
            ax.text(j, min(v + e + .004, 1.0), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(ab))); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        lo = max(0, float((ab[m + "_mean"] - ab[m + "_std"]).min()) - .1)
        ax.set_ylim(lo, 1.02); ax.set_ylabel(f"Test {ML[m]}")
        ax.set_title(f"Ablation: {ML[m]} (mean ± std, {nseeds} seeds)")
        ax.grid(axis="y", alpha=.3); save(fig, f"{i:02d}_ablation_{m}.png")


def fig_attacks(nseeds):
    at = T.attacks()
    if at.empty:
        return
    atks = list(dict.fromkeys(at.attack))
    for m, num in (("accuracy", 14), ("f1", 15), ("roc_auc", 16)):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(atks)); w = .38
        for k, (d, col) in enumerate((("ON", "#2ca02c"), ("OFF", "#d62728"))):
            sub = at[at.defense == d].set_index("attack").loc[atks]
            ax.bar(x + (k - .5) * w, sub[m + "_mean"], w, yerr=sub[m + "_std"], capsize=3,
                   color=col, label=f"Defense {d}")
        ax.set_xticks(x); ax.set_xticklabels(["no attacker" if a == "none" else a for a in atks], rotation=15)
        ax.set_ylabel(f"Test {ML[m]}"); ax.set_ylim(0, 1.05); ax.legend()
        ax.set_title(f"Attack vs defense: {ML[m]} (2 malicious clients of 10, {nseeds} seeds)")
        ax.grid(axis="y", alpha=.3); save(fig, f"{num}_attack_defense_{m}.png")
    # extra: degradation vs clean
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sub_on = at[(at.defense == "ON") & (at.attack != "none")].set_index("attack")
    sub_off = at[(at.defense == "OFF") & (at.attack != "none")].set_index("attack")
    a2 = list(sub_on.index); x = np.arange(len(a2)); w = .38
    ax.bar(x - w / 2, 100 * sub_on.loc[a2, "accuracy_drop_vs_clean"], w, color="#2ca02c", label="Defense ON")
    ax.bar(x + w / 2, 100 * sub_off.loc[a2, "accuracy_drop_vs_clean"], w, color="#d62728", label="Defense OFF")
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels(a2, rotation=15)
    ax.set_ylabel("Accuracy drop vs no-attacker run (percentage points)")
    ax.set_title("Attack impact: accuracy degradation relative to a clean federation")
    ax.legend(); ax.grid(axis="y", alpha=.3); save(fig, "28_attack_accuracy_degradation.png")


def fig_timing():
    tm = T.timing()
    if tm.empty:
        return
    lab = [f"{a}\n{b}" for a, b in zip(tm.model, tm.aggregation)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(tm))
    for k, col in (("local_train_s", "#8c564b"), ("encrypt_s", "#1f77b4"), ("aggregate_s", "#17becf"),
                   ("decrypt_s", "#9467bd"), ("eval_s", "#bcbd22")):
        ax.bar(range(len(tm)), tm[k + "_mean"], bottom=bottom, color=col, label=k.replace("_s", "").replace("_", " "))
        bottom += tm[k + "_mean"].values
    ax.errorbar(range(len(tm)), tm.total_s_mean, yerr=tm.total_s_std, fmt="none", ecolor="k", capsize=4,
                label="total wall-clock (± std)")
    ax.set_xticks(range(len(tm))); ax.set_xticklabels(lab, fontsize=9)
    ax.set_ylabel("Seconds per complete federated run (CPU)")
    ax.set_title("Training time per run, broken down by stage")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3); save(fig, "18_training_time_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sel = tm[tm.model == "Selected"]
    stages = ["encrypt", "aggregate", "decrypt"]
    x = np.arange(len(stages)); w = .38
    for k, (arm, col) in enumerate((("CKKS", C_CKKS), ("Plaintext", C_PLAIN))):
        r = sel[sel.aggregation == arm].iloc[0]
        ax.bar(x + (k - .5) * w, [r[s + "_per_round_ms"] for s in stages], w, color=col, label=arm)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xticks(x); ax.set_xticklabels(["Encrypt (10 clients)", "Aggregate", "Decrypt"])
    ax.set_ylabel("Milliseconds per round (symlog)")
    ax.set_title("Aggregation-path cost per round: CKKS vs plaintext FedAvg")
    ax.legend(); ax.grid(axis="y", alpha=.3); save(fig, "19_aggregation_time_comparison.png")


STAGE1_HIDDEN, STAGE1_DROPOUT = 32, 0.3  # stage-1 grid ran at the original model size


def fig_sweep(sel_cfg):
    sw = T.sweep()
    if sw.empty:
        return
    base = sw[(sw.anomaly_threshold == config.ANOMALY_THRESHOLD) & (sw.clip_threshold == config.CLIP_THRESHOLD)
              & (sw.noise_scale == config.NOISE_SCALE) & (sw.hidden == STAGE1_HIDDEN)
              & (sw.dropout == STAGE1_DROPOUT)]
    g = base[(base.knn_k == sel_cfg.knn_k) & (base.rounds == sel_cfg.rounds)]
    # 20 learning rate
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for ep in sorted(g.local_epochs.unique()):
        d = g[g.local_epochs == ep].sort_values("lr")
        ax.errorbar(d.lr, d.ckks_accuracy_mean, yerr=d.ckks_accuracy_std, marker="o", capsize=3, label=f"{ep} local epochs")
    ax.set_xscale("log"); ax.set_xlabel("Learning rate (log scale)"); ax.set_ylabel("Validation accuracy (CKKS)")
    ax.set_title(f"Learning-rate sweep (stage 1: k={sel_cfg.knn_k}, {sel_cfg.rounds} rounds, hidden=32, 3 val seeds)", fontsize=10)
    ax.grid(alpha=.3, which="both"); ax.legend(); save(fig, "20_learning_rate_comparison.png")
    # 21 local epochs
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for lr in sorted(g.lr.unique()):
        d = g[g.lr == lr].sort_values("local_epochs")
        ax.errorbar(d.local_epochs, d.ckks_accuracy_mean, yerr=d.ckks_accuracy_std, marker="o", capsize=3, label=f"lr={lr}")
    ax.set_xlabel("Local epochs per round"); ax.set_ylabel("Validation accuracy (CKKS)")
    ax.set_title(f"Local-epoch sweep (stage 1: k={sel_cfg.knn_k}, {sel_cfg.rounds} rounds, hidden=32, 3 val seeds)", fontsize=10)
    ax.grid(alpha=.3); ax.legend(); save(fig, "21_local_epoch_comparison.png")
    # 22 rounds
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    g2 = base[(base.knn_k == sel_cfg.knn_k) & (base.local_epochs == sel_cfg.local_epochs)]
    for lr in sorted(g2.lr.unique()):
        d = g2[g2.lr == lr].sort_values("rounds")
        ax.errorbar(d.rounds, d.ckks_accuracy_mean, yerr=d.ckks_accuracy_std, marker="o", capsize=3, label=f"lr={lr}")
    ax.set_xlabel("Communication rounds"); ax.set_ylabel("Validation accuracy (CKKS)")
    ax.set_title(f"Communication-round sweep (stage 1: {sel_cfg.local_epochs} local epochs, k={sel_cfg.knn_k}, hidden=32)", fontsize=10)
    ax.grid(alpha=.3); ax.legend(); save(fig, "22_communication_round_comparison.png")
    # 29 k / hidden / dropout (stage 2)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, p in zip(axes, ("knn_k", "hidden", "dropout")):
        others = {"knn_k": sel_cfg.knn_k, "hidden": STAGE1_HIDDEN, "dropout": STAGE1_DROPOUT}
        others.pop(p)
        d = sw[(sw.lr == sel_cfg.lr) & (sw.local_epochs == sel_cfg.local_epochs) & (sw.rounds == sel_cfg.rounds)
               & (sw.anomaly_threshold == config.ANOMALY_THRESHOLD) & (sw.clip_threshold == config.CLIP_THRESHOLD)
               & (sw.noise_scale == config.NOISE_SCALE)]
        for k, v in others.items():
            d = d[d[k] == v]
        d = d.sort_values(p)
        ax.errorbar(d[p].astype(str), d.ckks_accuracy_mean, yerr=d.ckks_accuracy_std, marker="o", capsize=3)
        ax.set_xlabel({"knn_k": "k (k-NN graph)", "hidden": "Hidden dimension", "dropout": "Dropout"}[p])
        ax.grid(alpha=.3)
    axes[0].set_ylabel("Validation accuracy (CKKS)")
    fig.suptitle(f"Stage 2: one-factor variations around the stage-1 winner (lr={sel_cfg.lr}, {sel_cfg.local_epochs} epochs, k={sel_cfg.knn_k}, hidden=32, dropout=0.3; 3 val seeds)", fontsize=10)
    save(fig, "29_graph_model_hyperparameters.png")

    # 30 defense sensitivity
    ds = T.defense_sensitivity()
    if not ds.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
        for ax, p in zip(axes, ("anomaly_threshold", "clip_threshold", "noise_scale")):
            d = ds[ds.parameter == p].sort_values("value")
            ax.errorbar(d.value.astype(str), d.ckks_accuracy_mean, yerr=d.ckks_accuracy_std, marker="o", capsize=3)
            for _, r in d[d.is_default].iterrows():
                ax.scatter([str(r.value)], [r.ckks_accuracy_mean], s=120, facecolors="none", edgecolors="r",
                           label="default (kept)")
            ax.set_xlabel(p); ax.grid(alpha=.3); ax.legend(fontsize=8)
        axes[0].set_ylabel("Validation accuracy (CKKS)")
        fig.suptitle("Defense-parameter sensitivity (reported only; defaults kept in final model)")
        save(fig, "30_defense_parameter_sensitivity.png")


def fig_distribution(nseeds):
    ps = T.per_seed_final()
    if ps.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    groups = [("Original baseline", "CKKS"), ("Original baseline", "Plaintext"), ("Selected", "CKKS"), ("Selected", "Plaintext")]
    for ax, m in zip(axes, ("accuracy", "f1")):
        data = [ps[(ps.model == a) & (ps.aggregation == b)][m].values for a, b in groups]
        ax.boxplot(data, widths=.5)
        for i, v in enumerate(data, 1):
            ax.scatter(np.full(len(v), i) + np.random.default_rng(1).uniform(-.12, .12, len(v)), v, s=14, alpha=.6)
        ax.set_xticks(range(1, 5)); ax.set_xticklabels([f"{a.split()[0]}\n{b}" for a, b in groups], fontsize=9)
        ax.set_ylabel(f"Test {ML[m]}"); ax.grid(axis="y", alpha=.3)
        ax.set_title(f"Per-seed {ML[m]} distribution ({nseeds} test seeds)")
    save(fig, "23_per_seed_distribution.png")

    ba = T.before_after()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(M)); w = .2
    cols = [C_BASE, "#c7c7c7", C_CKKS, C_PLAIN]
    for i, (_, r) in enumerate(ba.iterrows()):
        ax.bar(x + (i - 1.5) * w, [r[m + "_mean"] for m in M], w, yerr=[r[m + "_std"] for m in M],
               capsize=2, color=cols[i], label=r.configuration)
    ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in M]); ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Test score (mean ± std)"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"Mean ± std: original vs selected configuration ({nseeds} seeds)")
    ax.grid(axis="y", alpha=.3); save(fig, "24_mean_std_comparison.png")


def fig_final_model(cfg, nseeds):
    y, p, rs = T.final_pooled_predictions(cfg)
    if len(y) == 0:
        return
    pred = (p > 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])  # rows: true malignant, benign
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Malignant", "Benign"]); ax.set_yticks([0, 1])
    ax.set_yticklabels(["Malignant", "Benign"]); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (CKKS, pooled over {nseeds} test seeds)\nthreshold: argmax (P(malignant) > 0.5)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.046); save(fig, "25_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(5.8, 5))
    for r in rs:
        fpr, tpr, _ = roc_curve(r["predictions"]["y_true"], r["predictions"]["p_malignant"])
        ax.plot(fpr, tpr, color=C_CKKS, alpha=.15, lw=1)
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=C_CKKS, lw=2.2, label=f"Pooled ROC (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8, label="Chance")
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate (recall, malignant)")
    ax.set_title(f"ROC curve: selected CKKS model\n(thin lines = individual seeds, n={nseeds})", fontsize=10)
    ax.legend(loc="lower right"); ax.grid(alpha=.3); save(fig, "26_roc_curve.png")

    fig, ax = plt.subplots(figsize=(5.8, 5))
    for r in rs:
        pr, rc, _ = precision_recall_curve(r["predictions"]["y_true"], r["predictions"]["p_malignant"])
        ax.plot(rc, pr, color="#2ca02c", alpha=.15, lw=1)
    pr, rc, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    ax.plot(rc, pr, color="#2ca02c", lw=2.2, label=f"Pooled PR (AP = {ap:.3f})")
    ax.axhline(y.mean(), color="k", ls="--", lw=.8, label=f"Prevalence = {y.mean():.3f}")
    ax.set_xlabel("Recall (malignant)"); ax.set_ylabel("Precision (malignant)")
    ax.set_title(f"Precision-recall curve: selected CKKS model\n(thin lines = individual seeds, n={nseeds})", fontsize=10)
    ax.set_ylim(0, 1.03); ax.legend(loc="lower left"); ax.grid(alpha=.3); save(fig, "27_precision_recall_curve.png")


def fig_homophily():
    h = T.graph_homophily()
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for split, col in (("train", "#8c564b"), ("test", "#e377c2")):
        d = h[h.graph == split]
        ax.errorbar(d.knn_k, d.edge_homophily_mean, yerr=d.edge_homophily_std, marker="o", capsize=3, color=col,
                    label=f"{split} graphs (~{46 if split == 'train' else 11} nodes/client)")
    ax.set_xlabel("k (k-NN neighbours)"); ax.set_ylabel("Edge homophily (same-diagnosis edge fraction)")
    ax.set_title("Graph construction: class homophily vs k (data property, test seeds)")
    ax.grid(alpha=.3); ax.legend(); save(fig, "31_graph_homophily_vs_k.png")


def main():
    for f in os.listdir(OUT):
        if f.endswith(".png"):
            os.remove(os.path.join(OUT, f))
    cfg = T.selected_cfg()
    nseeds = len(T._final_seeds())
    fig_sweep(cfg)
    fig_homophily()
    if nseeds:
        fig_rounds(cfg, nseeds)
        fig_ckks_vs_plain(nseeds)
        fig_ablation(nseeds)
        fig_attacks(nseeds)
        fig_timing()
        fig_distribution(nseeds)
        fig_final_model(cfg, nseeds)
    print(f"{len(MADE)} figures written to results/figures/:")
    for m in sorted(MADE):
        print("  ", m)


if __name__ == "__main__":
    main()
