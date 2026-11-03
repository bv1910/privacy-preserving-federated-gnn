# Privacy-Preserving Federated GraphSAGE — Breast Cancer Diagnosis

Federated learning system where **10 clients each run GraphSAGE** on a
per-client k-NN similarity graph built from their own patients' Breast
Cancer Wisconsin (Diagnostic) records. Weight updates are CKKS-encrypted
before leaving each client, homomorphically aggregated, then passed through
a 4-agent defense pipeline before being applied as the new global model.

## What changed vs. the original plan

| | Original plan | This build |
|---|---|---|
| Dataset | MNIST (images) | **Breast Cancer Wisconsin (WDBC)**, read from `data/wdbc.csv` — standard schema: `id`, `diagnosis`, 30 feature columns |
| Model | CNN | **GraphSAGE** (2-layer, `torch_geometric.nn.SAGEConv`) |
| Graph structure | n/a | Per-client **k-NN similarity graph** (k=6) over patient feature vectors |
| Hyperparameter broadcast | n/a | `learning_rate` + `local_epochs` are **FPE-encrypted** before being sent to clients alongside each round's global model update, decrypted client-side, and logged/explained |

Everything else (CKKS/Pyfhel, FF3-1 FPE for metrics, 4-agent pipeline, SHAP +
rule-based explainability, Streamlit dashboard) follows the plan as written.

## Tuning study (CPU only) — how the numbers below were produced

**Every number in this section, in `results/summary/*.csv`, in `results/figures/*.png`
and in the dashboard is computed from one file: `results/raw/runs.jsonl`** (one JSON
line per federated run: config, seed, split, per-round history, final metrics,
predictions, timing, CKKS error, timestamp). Nothing is typed in by hand.

### Protocol
* **Splits.** Per client: test 20% (the original test indices, unchanged), validation
  20% of the rest, train the remainder. Tuning used validation only; the final
  model is retrained on train+validation and scored once on test.
* **Metrics.** Pooled over every held-out patient (malignant = positive class,
  threshold = argmax, i.e. P(malignant) > 0.5, never tuned). The original
  per-client macro averages over ~11-patient shards are still stored as
  `client_macro_*`.
* **Threat model unchanged.** Clients 3 and 7 are malicious (`scale_gaussian`) in
  every headline run.
* **Search (validation).** Stage 1: lr {0.003,0.005,0.01,0.02,0.05} × local epochs
  {3,5,8,10} × k {3,6}, rounds {8,15,20,30,40}, 3 seeds. Stage 2: k {2,4,8}, hidden
  {16,64}, dropout {0,0.5}. Stage 3: top 6 re-run on 5 fresh validation seeds; the
  winner is chosen on those only. Every configuration was run with CKKS and with
  plaintext FedAvg. Defense parameters were analysed for sensitivity but kept at
  their defaults.
* **Final.** 10 test seeds (100–109) that were never used during selection.

### Selected configuration
40 rounds, 8 local epochs, lr 0.05 (SGD, momentum 0.9), hidden 64, dropout 0.3,
**k = 3** (was 6). Defense parameters unchanged (0.7 / 1.0 / 0.01).

### Why the original model plateaued around 85%
1. Too little training: 8 rounds × 3 epochs.
2. Graph construction: each client's test graph has ~11 patients. With k = 6 only
   ~64% of its edges join same-diagnosis patients, versus ~87% in the training
   graphs (`results/summary/graph_homophily.csv`, figure 31).
3. Two malicious clients inject noise every round. This is deliberately kept.

### Important reproducibility note on CKKS
CKKS encryption is randomised by design, so SEAL draws fresh encryption noise in
every run. The per-round error is only ~3×10⁻⁶, but it is amplified over 40 rounds:
the **same config and seed can differ by one test patient** between two CKKS runs.
Plaintext runs are bit-identical. CKKS results are therefore reproducible as
mean ± std over seeds, not bit-for-bit.

See `results/summary/` for every table and `results/figures/` for 31 figures.

### Reproduce
```bash
pip install -r requirements.txt
python -m pytest tests -q                                   # 56 tests
CUDA_VISIBLE_DEVICES= python -m experiments.run_experiments # everything (resumable)
streamlit run dashboard.py
python main.py --selected                                   # single reference run + audit log
```
To recompute from scratch, delete `results/raw/runs.jsonl`. This takes about 2.5 h
on one CPU core.

## Project structure

```
federated_gnn_breast_cancer/
├── config.py                    # all hyperparameters
├── main.py                      # orchestrator — runs the full FL pipeline
├── dashboard.py                 # Streamlit dashboard (4 tabs)
├── models/graphsage_model.py    # 2-layer GraphSAGE (~4K params)
├── client/
│   ├── data_loader.py           # breast cancer partition + kNN graph build
│   └── federated_client.py      # local training, weight delta, malicious sim
├── crypto/
│   ├── ckks_engine.py           # CKKS encrypt/aggregate/decrypt (Pyfhel)
│   └── fpe_engine.py            # FF3-1 format-preserving encryption
├── agents/
│   ├── base_agent.py
│   ├── classifier_agent.py      # Agent 1 — anomaly scoring
│   ├── threshold_agent.py       # Agent 2 — risk-level routing
│   ├── coordinator_agent.py     # Agent 3 — fast path / full defense
│   └── logger_agent.py          # Agent 4 — append-only audit log
├── defense/defense_module.py    # gradient clipping + Gaussian noise
├── explainability/explainer.py  # SHAP + rule-based explanations
├── testing/testing_loop.py      # per-client eval + FPE + explain + log
├── evaluation/metrics.py        # accuracy/precision/recall/F1/ROC-AUC/confusion matrix
├── attacks/attack_types.py      # sign_flip, scale, gaussian, adaptive, label_flip attacks
├── privacy/dp_accountant.py     # (epsilon, delta)-DP budget for the defense noise
├── experiments/
│   ├── pipeline_runner.py       # RunConfig + run_config(): one parameterised federated run
│   ├── results_io.py            # single source of truth: results/raw/runs.jsonl
│   ├── sweep.py                 # staged hyperparameter search (validation only)
│   ├── final_experiments.py     # test seeds: CKKS vs plaintext, ablation, attacks
│   ├── run_ablation.py          # ablation only
│   ├── tables.py                # all derived tables -> results/summary/*.csv
│   ├── make_plots.py            # 31 figures -> results/figures/
│   └── run_experiments.py       # one-command reproduction of everything
├── logs/                        # audit_log.jsonl, round_results.json, *_comparison.csv (generated)
└── tests/                       # test_client, test_crypto, test_agents, test_pipeline,
                                  # test_metrics, test_attacks, test_dp_accountant, test_pipeline_runner
```

## Data flow: what's encrypted, where

| Data | Encryption | Direction | Purpose |
|---|---|---|---|
| Client weight deltas | CKKS (Pyfhel) | client → server | homomorphic aggregation across 10 clients without server seeing plaintext gradients |
| `learning_rate`, `local_epochs` | FPE (FF3-1) | server → client (broadcast) | hyperparameters stay encrypted in transit alongside the global model update, decrypted only inside each client process |
| Per-client `accuracy`, `loss` | FPE (FF3-1) | client → dashboard/log | format-preserving masking of test metrics for audit/display |
| Model architecture, other config | none | n/a | local-only, never transmitted |

## Setup (Windows)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> **Pyfhel build requirement**: `pip install Pyfhel` compiles from source on
> first install (~3–5 min) and needs **MSVC C++17 Build Tools**. Install via
> Visual Studio Installer → "Desktop development with C++" if the install
> fails. If you'd rather avoid the compiler entirely, run this project
> inside **WSL2** and set `CKKS_BACKEND = "tenseal"` in `config.py`
> (TenSEAL only ships Linux wheels — no native Windows install path).

## Run

```bash
# Unit + integration tests
python -m pytest tests/ -v

# Full federated training run (original config); --selected uses the tuned config
python main.py

# Dashboard (after main.py has produced logs/)
streamlit run dashboard.py
```

## Verified in this environment

- CKKS encrypt → homomorphic aggregate → decrypt roundtrip: max abs error
  **5e-6** (well under the 0.01 tolerance)
- FF3-1 FPE metric encrypt/decrypt roundtrip: **exact match**
- FF3-1 FPE hyperparameter encrypt/decrypt roundtrip: **exact match**
  (`learning_rate=0.05` → ciphertext `'159780'` → decrypts back to `0.05`)
- 8-round run on the real WDBC CSV: avg accuracy climbs from ~0.62 → **~0.85**,
  with both the Fast Path and Full Defense routes exercised by the 2
  simulated malicious clients (ids 3, 7 in `config.MALICIOUS_CLIENT_IDS`)
- Every round's hyperparameter broadcast is logged in `logs/audit_log.jsonl`
  under `agent_name: "hyperparam_broadcast"` — only ciphertext values appear,
  never plaintext `learning_rate`/`local_epochs`
- `pytest tests/` — **56/56 passing** (incl. headless dashboard rendering)

## Key design notes

- **Non-IID option**: `config.NON_IID = True` switches from an IID shuffle
  split to a Dirichlet label-skewed partition, if you want to demonstrate
  robustness under heterogeneous client data distributions.
- **No fabricated metrics**: every number shown in the console output, the
  audit log, and the dashboard is computed from a real forward/backward
  pass or a real CKKS/FPE operation — nothing is randomly generated for
  display purposes (malicious-client gradient corruption is the one
  intentional synthetic perturbation, and it's clearly logged as such).
- **Graph leakage**: k-NN graphs for train and test are built independently
  per client and per split — test node features never inform the train
  graph's edges, and vice versa.
