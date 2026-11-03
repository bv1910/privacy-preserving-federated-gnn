"""
Streamlit dashboard -- Privacy-Preserving Federated GraphSAGE (WDBC).

Every number shown here is computed at load time from the measured run store
results/raw/runs.jsonl through experiments/tables.py -- the same functions
that write results/summary/*.csv and draw results/figures/*.png.  Nothing is
hard-coded.

    streamlit run dashboard.py
"""
import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve

import config
from experiments import results_io as rio
from experiments import tables as T
from experiments.pipeline_runner import RunConfig, original_config

st.set_page_config(page_title="Federated GraphSAGE · CKKS · Breast Cancer", page_icon="🔒", layout="wide")

M, ML = rio.METRICS, rio.METRIC_LABELS
BLUE, ORANGE, GREY, GREEN, RED = "#1f77b4", "#ff7f0e", "#9e9e9e", "#2ca02c", "#d62728"

st.markdown("""
<style>
div[data-testid="stMetric"] {background: rgba(127,127,127,0.06); border: 1px solid rgba(127,127,127,0.18);
  border-radius: 10px; padding: 10px 14px;}
.badge {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600;}
.b-final {background:#e3f2e6; color:#1b5e20;} .b-single {background:#fff3e0; color:#e65100;}
.b-meas {background:#e3edf7; color:#0d47a1;}
.small {font-size:0.85rem; opacity:0.8;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
def _store_stamp():
    p = rio.RUNS_PATH
    return (os.path.getmtime(p), os.path.getsize(p)) if os.path.exists(p) else None


@st.cache_data(show_spinner="Loading measured results…")
def load_all(stamp):
    cfg = T.selected_cfg()
    d = {"selection": rio.load_selection(), "n_runs": len(rio.load_runs())}
    d["final_seeds"] = T._final_seeds()
    d["ckks_vs_plain"] = T.ckks_vs_plaintext("selected")
    d["baseline_cvp"] = T.ckks_vs_plaintext("baseline")
    d["per_seed"] = T.per_seed_final()
    d["before_after"] = T.before_after()
    d["ablation"] = T.ablation()
    d["attacks"] = T.attacks()
    d["attack_gain"] = T.attack_defense_gain()
    d["sweep"] = T.sweep()
    from experiments.sweep import S3_SEEDS
    d["sweep3"] = T.sweep(S3_SEEDS)
    d["defense_sens"] = T.defense_sensitivity()
    d["timing"] = T.timing()
    d["ckks_err"] = T.ckks_error_summary()
    d["curves"] = T.round_curves(cfg) if cfg else pd.DataFrame()
    d["curves_base"] = T.round_curves(original_config())
    y, p, rs = T.final_pooled_predictions(cfg) if cfg else (np.array([]), np.array([]), [])
    d["y"], d["p"] = y, p
    d["final_runs_preds"] = [(r["seed"], r["predictions"]) for r in rs]
    d["all_runs"] = rio.runs_dataframe(rio.load_runs())
    d["homophily"] = pd.read_csv(os.path.join(config.SUMMARY_DIR, "graph_homophily.csv")) \
        if os.path.exists(os.path.join(config.SUMMARY_DIR, "graph_homophily.csv")) else pd.DataFrame()
    dp = os.path.join(config.SUMMARY_DIR, "dp_budget.csv")
    d["dp"] = pd.read_csv(dp) if os.path.exists(dp) else pd.DataFrame()
    return d


stamp = _store_stamp()
if stamp is None or rio.load_selection() is None:
    st.title("🔒 Federated GraphSAGE — Breast Cancer")
    st.warning("No measured results found. Run `python -m experiments.run_experiments` first.")
    st.stop()
D = load_all(stamp)
cfg = T.selected_cfg()
sel = D["selection"]
NS = len(D["final_seeds"])


def pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.2f}%"


def fmt_ms(row, m):
    return f"{100 * row[m + '_mean']:.2f} ± {100 * row[m + '_std']:.2f}"


def provenance(text):
    st.markdown(f"<span class='badge b-meas'>MEASURED</span> <span class='small'>{text}</span>", unsafe_allow_html=True)


def metric_table(df, name_col, extra_cols=()):
    out = pd.DataFrame({name_col: df[name_col]})
    for m in M:
        out[ML[m] + " (%)"] = [fmt_ms(r, m) for _, r in df.iterrows()]
    for c in extra_cols:
        out[c] = df[c].values
    out["Seeds"] = df["n_seeds"].values
    return out


# --------------------------------------------------------------------------- sidebar
SECTIONS = ["Overview", "Final Model Performance", "CKKS vs Plaintext", "Communication Rounds",
            "Ablation Study", "Attack & Defense", "Hyperparameter Search", "Privacy / CKKS Analysis",
            "Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Experimental Logs / Raw Results"]
with st.sidebar:
    st.markdown("### 🔒 Federated GraphSAGE")
    st.caption("CKKS · FPE · 4-agent defense · WDBC")
    section = st.radio("Section", SECTIONS, label_visibility="collapsed")
    st.divider()
    st.caption(f"Run store: `results/raw/runs.jsonl`  \n{D['n_runs']} measured runs  \n"
               f"Final test seeds: {NS} ({D['final_seeds'][0]}–{D['final_seeds'][-1]})" if NS else "")
    st.caption("Device: CPU only · CKKS backend: " + config.CKKS_BACKEND)

cvp = D["ckks_vs_plain"].set_index("metric") if not D["ckks_vs_plain"].empty else None

# =========================================================================== OVERVIEW
if section == "Overview":
    st.title("Privacy-Preserving Federated GraphSAGE — Breast Cancer Diagnosis")
    st.markdown("10 hospitals (clients) each train a **GraphSAGE** model on a k-NN patient-similarity graph. "
                "Client weight updates are **CKKS-encrypted** and averaged homomorphically; the decrypted average "
                "is screened by a **4-agent defense** (anomaly scoring → risk threshold → L2 clipping + Gaussian "
                "sanitisation → audit log). Training hyperparameters are broadcast **FPE-encrypted**. "
                "Two of the ten clients are malicious in the default threat model (scale ×6 + Gaussian noise).")

    st.markdown("<span class='badge b-final'>FINAL VALIDATED RESULT</span> "
                f"<span class='small'>CKKS model, selected on validation data only, evaluated on "
                f"{NS} fresh test seeds never used during tuning — mean ± std</span>", unsafe_allow_html=True)
    cols = st.columns(5)
    for c, m in zip(cols, M):
        c.metric(ML[m], pct(cvp.loc[m, "ckks_mean"]), help=f"std across {NS} seeds: ±{100 * cvp.loc[m, 'ckks_std']:.2f} pp")
        c.caption(f"± {100 * cvp.loc[m, 'ckks_std']:.2f} pp")

    ps = D["per_seed"]
    best = ps[(ps.model == "Selected") & (ps.aggregation == "CKKS")].sort_values(["accuracy", "f1"], ascending=False).iloc[0]
    st.markdown("<span class='badge b-single'>BEST SINGLE EXPERIMENT</span> "
                "<span class='small'>one seed only — shown for transparency, NOT the reproducible result</span>",
                unsafe_allow_html=True)
    cols = st.columns(5)
    for c, m in zip(cols, M):
        c.metric(f"{ML[m]} (single seed)", pct(best[m]))
    st.caption(f"Seed {int(best.seed)} of the final test seeds.")

    st.divider()
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Best validated configuration")
        bc = sel["selected_config"]
        cfg_df = pd.DataFrame([
            ("Communication rounds", bc["rounds"]), ("Local epochs", bc["local_epochs"]),
            ("Learning rate (SGD, momentum 0.9)", bc["lr"]), ("k-NN neighbours (graph)", bc["knn_k"]),
            ("GraphSAGE hidden dim", bc["hidden"]), ("Dropout", bc["dropout"]),
            ("Clients", config.NUM_CLIENTS), ("Malicious clients", str(config.MALICIOUS_CLIENT_IDS) + " (scale_gaussian)"),
            ("Anomaly threshold (default kept)", bc["anomaly_threshold"]),
            ("L2 clip threshold (default kept)", bc["clip_threshold"]),
            ("Gaussian noise σ (default kept)", bc["noise_scale"]),
            ("CKKS backend / N / scale", f"{config.CKKS_BACKEND} / {config.POLY_MODULUS_DEGREE} / 2^{int(np.log2(config.CKKS_SCALE))}"),
            ("Validation seeds (selection)", ", ".join(map(str, sel["selection_seeds"]))),
            ("Test seeds (final)", f"{NS}: {D['final_seeds'][0]}–{D['final_seeds'][-1]}"),
        ], columns=["Setting", "Value"])
        cfg_df["Value"] = cfg_df["Value"].astype(str)
        st.dataframe(cfg_df, hide_index=True, use_container_width=True)
    with right:
        st.subheader("Original vs selected configuration")
        ba = D["before_after"]
        fig = go.Figure()
        for _, r in ba.iterrows():
            fig.add_bar(name=r.configuration, x=[ML[m] for m in M], y=[r[m + "_mean"] for m in M],
                        error_y=dict(type="data", array=[r[m + "_std"] for m in M]))
        fig.update_layout(barmode="group", yaxis=dict(title="Test score (mean ± std)", range=[0.5, 1.02]),
                          height=430, legend=dict(orientation="h", y=-0.2), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        provenance(f"Both configurations on the same {NS} test seeds.")

    st.subheader("The experimental story")
    st.markdown(f"""
1. **What was tested** — {D['n_runs']} federated runs: a staged hyperparameter search on a *validation* split,
   then the selected configuration on {NS} fresh test seeds with CKKS and plaintext aggregation, a 6-way component
   ablation, and {len(D['attacks']['attack'].unique()) - 1 if not D['attacks'].empty else 0} attack types with defense on/off.
2. **CKKS vs plaintext** — identical seeds, splits, initial weights and training; only the aggregation differs.
3. **Defense** — attack runs isolate what the 4-agent pipeline changes.
4. **Reproducibility** — every number on this page is recomputed from `results/raw/runs.jsonl`.
""")

# =========================================================================== FINAL
elif section == "Final Model Performance":
    st.title("Final model performance")
    provenance(f"Selected configuration, CKKS aggregation, {NS} fresh test seeds. Pooled metrics over all "
               "clients' held-out patients; malignant = positive class; threshold = argmax (P>0.5).")
    ps = D["per_seed"]
    sub = ps[(ps.model == "Selected")]
    st.subheader("Per-seed results")
    view = sub[["aggregation", "seed"] + M + ["tp", "fp", "fn", "tn", "n", "full_defense_rounds"]].copy()
    st.dataframe(view.style.format({m: "{:.4f}" for m in M}), hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        m = st.selectbox("Metric", M, format_func=lambda x: ML[x])
        fig = px.box(ps, x="model", y=m, color="aggregation", points="all",
                     color_discrete_map={"CKKS": BLUE, "Plaintext": ORANGE},
                     labels={m: f"Test {ML[m]}", "model": "Configuration"})
        fig.update_layout(height=420, title=f"Per-seed {ML[m]} distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Pooled vs per-client-macro metrics")
        st.caption("The original project macro-averaged metrics over 10 client test shards of ~11 patients. "
                   "Both are shown; the headline uses pooled metrics (each test patient counted once).")
        ba = D["before_after"]
        rows = []
        for _, r in ba.iterrows():
            rows.append({"Configuration": r.configuration, **{f"{ML[m]} pooled": f"{100*r[m+'_mean']:.2f}" for m in M},
                         **{f"{ML[m]} client-macro": f"{100*r['client_macro_'+m+'_mean']:.2f}" for m in M}})
        st.dataframe(pd.DataFrame(rows).T, use_container_width=True)

# =========================================================================== CKKS vs PLAIN
elif section == "CKKS vs Plaintext":
    st.title("CKKS encrypted aggregation vs plaintext FedAvg")
    st.info("CKKS protects client model updates *during aggregation*: the server only adds ciphertexts and "
            "never sees an individual client's update. It is an approximate scheme, so it can at best match "
            "plaintext accuracy — any difference is numerical noise, not a learning benefit.")
    provenance(f"Identical split, clients, graphs, initial weights, seeds ({NS}), local training and evaluation; "
               "only the aggregation step differs.")
    df = D["ckks_vs_plain"]
    cols = st.columns(5)
    for c, m in zip(cols, M):
        r = cvp.loc[m]
        c.metric(f"{ML[m]} — CKKS", pct(r.ckks_mean), delta=f"{100 * r.mean_paired_diff:+.4f} pp vs plaintext",
                 delta_color="off")
        c.metric(f"{ML[m]} — plaintext", pct(r.plaintext_mean))
    fig = go.Figure()
    fig.add_bar(name="CKKS", x=[ML[m] for m in M], y=df.ckks_mean, error_y=dict(type="data", array=df.ckks_std), marker_color=BLUE)
    fig.add_bar(name="Plaintext FedAvg", x=[ML[m] for m in M], y=df.plaintext_mean,
                error_y=dict(type="data", array=df.plaintext_std), marker_color=ORANGE)
    fig.update_layout(barmode="group", yaxis=dict(range=[0.5, 1.02], title="Test score"), height=420,
                      title="Selected configuration: CKKS vs plaintext (mean ± std)")
    st.plotly_chart(fig, use_container_width=True)
    show = df.copy()
    show["metric"] = show.metric.map(ML)
    st.dataframe(show.style.format({c: "{:.6f}" for c in show.columns if show[c].dtype == float}),
                 hide_index=True, use_container_width=True)
    st.caption("`seeds_with_any_difference` counts seeds whose final metric differs at all between the two arms; "
               "p-value is NaN when the paired differences are exactly zero (t-test undefined).")
    e = D["ckks_err"].iloc[0]
    tm = D["timing"]; s = tm[tm.model == "Selected"].set_index("aggregation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max |CKKS error| (any round)", f"{e.ckks_max_abs_err_max:.2e}")
    c2.metric("Mean CKKS RMSE / round", f"{e.ckks_rmse_mean:.2e}")
    c3.metric("Encrypt+aggregate+decrypt / round",
              f"{s.loc['CKKS', 'encrypt_per_round_ms'] + s.loc['CKKS', 'aggregate_per_round_ms'] + s.loc['CKKS', 'decrypt_per_round_ms']:.1f} ms")
    c4.metric("Plaintext mean / round", f"{s.loc['Plaintext', 'aggregate_per_round_ms']:.3f} ms")
    with st.expander("Original (pre-tuning) configuration: CKKS vs plaintext"):
        b = D["baseline_cvp"].copy(); b["metric"] = b.metric.map(ML)
        st.dataframe(b, hide_index=True, use_container_width=True)

# =========================================================================== ROUNDS
elif section == "Communication Rounds":
    st.title("Communication rounds")
    provenance(f"Per-round test metrics recorded during the final runs ({NS} seeds); mean ± std band.")
    q = st.selectbox("Quantity", M + ["loss", "train_loss"],
                     format_func=lambda x: {**ML, "loss": "Test cross-entropy", "train_loss": "Honest-client training loss"}[x])
    fig = go.Figure()
    for df, lab, dash in ((D["curves"], "Selected", "solid"), (D["curves_base"], "Original", "dash")):
        for arm, col in (("CKKS", BLUE), ("Plaintext", ORANGE)):
            d = df[(df.quantity == q) & (df.aggregation == arm)]
            if d.empty:
                continue
            fig.add_scatter(x=d["round"], y=d["mean"], name=f"{lab} – {arm}", line=dict(color=col, dash=dash),
                            error_y=dict(type="data", array=d["std"], visible=arm == "CKKS", thickness=0.8))
    fig.update_layout(xaxis_title="Communication round", yaxis_title=q, height=480)
    st.plotly_chart(fig, use_container_width=True)
    sw = D["sweep"]
    st.subheader("Validation accuracy at each round budget (selected lr / epochs / graph)")
    g = sw[(sw.lr == cfg.lr) & (sw.local_epochs == cfg.local_epochs) & (sw.knn_k == cfg.knn_k) & (sw.hidden == cfg.hidden)
           & (sw.dropout == cfg.dropout) & (sw.anomaly_threshold == config.ANOMALY_THRESHOLD)
           & (sw.clip_threshold == config.CLIP_THRESHOLD) & (sw.noise_scale == config.NOISE_SCALE)].sort_values("rounds")
    st.dataframe(g[["rounds", "ckks_accuracy_mean", "ckks_accuracy_std", "ckks_f1_mean", "plain_accuracy_mean"]],
                 hide_index=True, use_container_width=True)

# =========================================================================== ABLATION
elif section == "Ablation Study":
    st.title("Component ablation")
    provenance(f"Selected configuration, {NS} test seeds, CKKS on unless removed. Each row disables exactly one component.")
    ab = D["ablation"]
    st.dataframe(metric_table(ab, "configuration", ["full_defense_rounds_mean"]).rename(
        columns={"configuration": "Configuration", "full_defense_rounds_mean": "Full-defense rounds / run"}),
        hide_index=True, use_container_width=True)
    m = st.selectbox("Metric to plot", M, format_func=lambda x: ML[x])
    fig = px.bar(ab, x="configuration", y=m + "_mean", error_y=m + "_std", text=ab[m + "_mean"].map(lambda v: f"{v:.3f}"),
                 labels={"configuration": "", m + "_mean": f"Test {ML[m]}"})
    fig.update_traces(marker_color=[BLUE] + ["#9ecae1"] * (len(ab) - 1))
    fig.update_layout(yaxis_range=[max(0, ab[m + "_mean"].min() - 0.15), 1.02], height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("L2 clipping and Gaussian sanitisation act only on rounds the agents route to Full Defense; "
               "FPE encrypts the hyperparameter broadcast and is lossless, so removing it cannot change accuracy.")

# =========================================================================== ATTACK
elif section == "Attack & Defense":
    st.title("Attacks and the multi-agent defense")
    provenance(f"Selected configuration, CKKS on, 2 malicious clients (ids {config.MALICIOUS_CLIENT_IDS}), "
               f"{NS} test seeds. Defense OFF = 4-agent pipeline bypassed. 'none' = no malicious clients.")
    at = D["attacks"]
    t = at[["attack", "defense"] + [m + "_mean" for m in M] + [m + "_std" for m in M] + ["full_defense_rounds_mean", "accuracy_drop_vs_clean"]].copy()
    st.dataframe(t.style.format({c: "{:.4f}" for c in t.columns if c not in ("attack", "defense")}),
                 hide_index=True, use_container_width=True)
    m = st.selectbox("Metric", M, format_func=lambda x: ML[x])
    fig = px.bar(at, x="attack", y=m + "_mean", color="defense", barmode="group", error_y=m + "_std",
                 color_discrete_map={"ON": GREEN, "OFF": RED}, labels={m + "_mean": f"Test {ML[m]}", "attack": "Attack"})
    fig.update_layout(height=450, yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)
    g = D["attack_gain"]
    fig2 = px.bar(g, x="attack", y="accuracy_gain_defense_on_minus_off",
                  labels={"accuracy_gain_defense_on_minus_off": "Accuracy: defense ON − OFF"},
                  title="How much the defense changes accuracy under each attack")
    fig2.add_hline(y=0, line_color="black")
    st.plotly_chart(fig2, use_container_width=True)

# =========================================================================== SWEEP
elif section == "Hyperparameter Search":
    st.title("Hyperparameter search (validation split only)")
    st.markdown(f"**Protocol** — {sel['selection_metric']}. Stage 1/2 seeds: {sel['stage1_seeds']}; "
                f"finalists re-run on fresh validation seeds {sel['selection_seeds']}; the test split was never used.")
    fin = pd.DataFrame(sel["finalists"])
    st.subheader("Stage 3 finalists (fresh validation seeds)")
    st.dataframe(fin[["lr", "local_epochs", "knn_k", "hidden", "dropout", "rounds", "stage1_accuracy_mean",
                      "accuracy_mean", "accuracy_std", "f1_mean", "roc_auc_mean"]], hide_index=True, use_container_width=True)
    sw = D["sweep"]
    st.subheader(f"All validation configurations × round budgets ({len(sw)} rows, CKKS and plaintext)")
    st.dataframe(sw, hide_index=True, use_container_width=True, height=320)
    c1, c2 = st.columns(2)
    base = sw[(sw.anomaly_threshold == config.ANOMALY_THRESHOLD) & (sw.clip_threshold == config.CLIP_THRESHOLD)
              & (sw.noise_scale == config.NOISE_SCALE) & (sw.hidden == cfg.hidden) & (sw.dropout == cfg.dropout)]
    with c1:
        k = st.selectbox("k-NN k", sorted(base.knn_k.unique()), index=sorted(base.knn_k.unique()).index(cfg.knn_k))
        R = st.selectbox("Rounds", sorted(base.rounds.unique()), index=sorted(base.rounds.unique()).index(cfg.rounds))
        h = base[(base.knn_k == k) & (base.rounds == R)].pivot_table(index="local_epochs", columns="lr", values="ckks_accuracy_mean")
        fig = px.imshow(h, text_auto=".3f", aspect="auto", color_continuous_scale="Blues",
                        labels=dict(x="Learning rate", y="Local epochs", color="Val acc"),
                        title=f"Validation accuracy (CKKS), k={k}, {R} rounds")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(sw, x="ckks_accuracy_mean", y="plain_accuracy_mean", hover_data=["lr", "local_epochs", "knn_k", "rounds"],
                         labels={"ckks_accuracy_mean": "Val accuracy — CKKS", "plain_accuracy_mean": "Val accuracy — plaintext"},
                         title="Every tested configuration: CKKS vs plaintext")
        lo = float(min(sw.ckks_accuracy_mean.min(), sw.plain_accuracy_mean.min()))
        fig.add_shape(type="line", x0=lo, y0=lo, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Defense-parameter sensitivity (reported only — defaults kept)"):
        st.dataframe(D["defense_sens"], hide_index=True, use_container_width=True)
    with st.expander("Graph homophily vs k (why k matters)"):
        st.dataframe(D["homophily"], hide_index=True, use_container_width=True)
        st.caption("Each client's held-out graph has only ~11 patients; with large k most neighbours are other-class "
                   "patients, which the mean-aggregating GraphSAGE layer then mixes into each patient's representation.")

# =========================================================================== PRIVACY
elif section == "Privacy / CKKS Analysis":
    st.title("Privacy / CKKS analysis")
    st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f5f5f5", fontsize=10];
 c [label="10 clients\\nlocal GraphSAGE training"]; e [label="CKKS encrypt\\nΔw (per client)", fillcolor="#e3edf7"];
 a [label="Server: homomorphic\\nsum × 1/N (ciphertexts)", fillcolor="#e3edf7"]; d [label="Decrypt\\naggregate only", fillcolor="#e3edf7"];
 g [label="4-agent defense\\nscore → threshold →\\nclip + noise → log", fillcolor="#e3f2e6"]; u [label="Global model\\nupdate"];
 f [label="FPE-encrypted\\nlr / epochs broadcast", fillcolor="#fff3e0"];
 c->e->a->d->g->u; u->f->c; }""")
    e = D["ckks_err"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max |error|, any round", f"{e.ckks_max_abs_err_max:.2e}")
    c2.metric("Mean max |error| / round", f"{e.ckks_max_abs_err_mean:.2e}")
    c3.metric("Mean relative L2 error", f"{e.ckks_rel_l2_err_mean:.2e}")
    c4.metric("Ciphertexts / client / round", int(e.ciphertexts_per_client))
    st.caption(f"Parameters: N={int(e.poly_modulus_degree)}, scale=2^{int(e.scale_bits)}, q_i={e.qi_sizes}, backend={e.backend}. "
               f"{int(e.n_round_measurements)} round-level measurements.")
    cu = D["curves"]
    fig = go.Figure()
    for q, lab in (("ckks_max_abs_err", "max |error|"), ("ckks_rmse", "RMSE")):
        d = cu[(cu.quantity == q) & (cu.aggregation == "CKKS")]
        fig.add_scatter(x=d["round"], y=d["mean"], name=lab, error_y=dict(type="data", array=d["std"]))
    fig.update_layout(yaxis_type="log", xaxis_title="Communication round",
                      yaxis_title="CKKS aggregate − plaintext mean", title="CKKS numerical error per round", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Timing (CPU)")
    st.dataframe(D["timing"], hide_index=True, use_container_width=True)
    if not D["dp"].empty:
        st.subheader("Differential-privacy accounting of the Gaussian sanitisation")
        st.dataframe(D["dp"], hide_index=True, use_container_width=True)
        st.warning("With σ = 0.01 and clip = 1.0 the ε is very large: the noise step is a robustness/sanitisation "
                   "measure, not a meaningful formal DP guarantee.")

# =========================================================================== CM
elif section == "Confusion Matrix":
    st.title("Confusion matrix — final validated CKKS model")
    y, p = D["y"], D["p"]
    seeds = [s for s, _ in D["final_runs_preds"]]
    choice = st.selectbox("Seed", ["Pooled over all seeds"] + seeds)
    if choice != "Pooled over all seeds":
        pr = dict(D["final_runs_preds"])[choice]
        y, p = np.array(pr["y_true"]), np.array(pr["p_malignant"])
    cm = confusion_matrix(y, (p > 0.5).astype(int), labels=[1, 0])
    fig = px.imshow(cm, text_auto=True, x=["Pred. malignant", "Pred. benign"], y=["True malignant", "True benign"],
                    color_continuous_scale="Blues")
    fig.update_layout(height=430, width=520)
    st.plotly_chart(fig)
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    st.markdown(f"TP={tp} · FN={fn} · FP={fp} · TN={tn} · sensitivity={tp/(tp+fn):.3f} · specificity={tn/(tn+fp):.3f}")
    provenance("Threshold: argmax of the 2-class softmax (P(malignant) > 0.5); not tuned.")

# =========================================================================== ROC
elif section == "ROC Curve":
    st.title("ROC curve — final validated CKKS model")
    fig = go.Figure()
    for s, pr in D["final_runs_preds"]:
        f, t, _ = roc_curve(pr["y_true"], pr["p_malignant"])
        fig.add_scatter(x=f, y=t, mode="lines", line=dict(width=1, color="rgba(31,119,180,0.25)"), name=f"seed {s}", showlegend=False)
    f, t, _ = roc_curve(D["y"], D["p"])
    fig.add_scatter(x=f, y=t, mode="lines", line=dict(width=3, color=BLUE), name="Pooled")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="Chance")
    fig.update_layout(xaxis_title="False-positive rate", yaxis_title="True-positive rate", height=520)
    st.plotly_chart(fig, use_container_width=True)
    provenance(f"Pooled AUC = {auc(f, t):.4f}; mean per-seed AUC = {cvp.loc['roc_auc','ckks_mean']:.4f} ± {cvp.loc['roc_auc','ckks_std']:.4f}.")

# =========================================================================== PR
elif section == "Precision-Recall Curve":
    st.title("Precision-recall curve — final validated CKKS model")
    fig = go.Figure()
    for s, pr in D["final_runs_preds"]:
        P, R, _ = precision_recall_curve(pr["y_true"], pr["p_malignant"])
        fig.add_scatter(x=R, y=P, mode="lines", line=dict(width=1, color="rgba(44,160,44,0.25)"), showlegend=False)
    P, R, _ = precision_recall_curve(D["y"], D["p"])
    fig.add_scatter(x=R, y=P, mode="lines", line=dict(width=3, color=GREEN), name="Pooled")
    fig.add_hline(y=float(np.mean(D["y"])), line_dash="dash", annotation_text="prevalence")
    fig.update_layout(xaxis_title="Recall (malignant)", yaxis_title="Precision (malignant)", height=520, yaxis_range=[0, 1.03])
    st.plotly_chart(fig, use_container_width=True)
    provenance(f"Pooled average precision = {average_precision_score(D['y'], D['p']):.4f}.")

# =========================================================================== LOGS
elif section == "Experimental Logs / Raw Results":
    st.title("Experimental logs & raw results")
    st.markdown("**Single source of truth:** `results/raw/runs.jsonl` — one JSON line per federated run "
                "(config, seed, split, per-round history, final metrics, predictions, timing, CKKS error, timestamp).")
    st.code("CUDA_VISIBLE_DEVICES= python -m experiments.run_experiments   # reproduce everything\n"
            "python -m pytest tests -q\nstreamlit run dashboard.py", language="bash")
    df = D["all_runs"]
    c1, c2, c3 = st.columns(3)
    hold = c1.multiselect("Split", sorted(df.holdout.unique()), default=sorted(df.holdout.unique()))
    stg = c2.multiselect("Stage", sorted(df.stage.dropna().unique()), default=sorted(df.stage.dropna().unique()))
    ck = c3.multiselect("CKKS", [True, False], default=[True, False])
    f = df[df.holdout.isin(hold) & df.stage.isin(stg) & df.use_ckks.isin(ck)]
    st.caption(f"{len(f)} of {len(df)} runs · timestamps {df.timestamp.min()} → {df.timestamp.max()}")
    st.dataframe(f, hide_index=True, use_container_width=True, height=420)
    st.download_button("Download filtered runs (CSV)", f.to_csv(index=False), "runs_filtered.csv", "text/csv")
    with st.expander("Summary CSVs in results/summary/"):
        for fn in sorted(os.listdir(config.SUMMARY_DIR)):
            if fn.endswith(".csv") and fn != "all_runs.csv":
                st.markdown(f"**{fn}**")
                st.dataframe(pd.read_csv(os.path.join(config.SUMMARY_DIR, fn)), hide_index=True, use_container_width=True)
    with st.expander("Reference single run (main.py): per-round agent decisions"):
        if os.path.exists(config.ROUND_RESULTS_PATH):
            rr = json.load(open(config.ROUND_RESULTS_PATH))
            st.dataframe(pd.DataFrame(rr)[["round", "avg_accuracy", "avg_loss", "anomaly_score", "risk_level", "path_taken"]],
                         hide_index=True, use_container_width=True)
        else:
            st.info("Run `python main.py` to produce logs/round_results.json and the audit log.")
