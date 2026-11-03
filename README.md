# Privacy-Preserving Federated GraphSAGE for Breast Cancer Diagnosis

A privacy-preserving machine learning framework that combines **Federated Learning, Graph Neural Networks (GraphSAGE), and Homomorphic Encryption** to perform breast cancer classification without requiring centralized sharing of client training data.

## Overview

This project implements a federated learning system in which multiple clients collaboratively train a global GraphSAGE model while keeping their local patient data decentralized.

Model updates are protected using CKKS homomorphic encryption, while additional defense mechanisms help detect and mitigate malicious client updates.

An interactive Streamlit dashboard provides model evaluation, experiment visualization, and privacy-related insights.

## Key Features

- **Federated Learning:** Collaborative model training across 10 independent clients.
- **GraphSAGE:** Graph-based learning using k-nearest-neighbor patient similarity graphs.
- **Homomorphic Encryption:** CKKS encryption for protected model-update aggregation.
- **Format-Preserving Encryption:** FF3-1 encryption for selected metrics and hyperparameters.
- **Privacy Protection:** Differential privacy accounting and gradient clipping.
- **Malicious Client Detection:** Simulated adversarial clients and defense mechanisms.
- **Multi-Agent Architecture:** Four agents coordinate classification, risk assessment, defense routing, and audit logging.
- **Explainable AI:** SHAP-based and rule-based explanations.
- **Interactive Dashboard:** Streamlit interface for exploring model performance and experiments.
- **Automated Testing:** Unit and integration tests.

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Graph Learning | PyTorch Geometric, GraphSAGE |
| Federated Learning | Custom federated learning pipeline |
| Encryption | CKKS, Pyfhel, FF3-1 |
| Privacy | Differential Privacy |
| Explainable AI | SHAP |
| Dashboard | Streamlit, Plotly |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Testing | Pytest |

## System Architecture

1. The Breast Cancer Wisconsin Diagnostic dataset is distributed among 10 clients.
2. Each client constructs a local k-nearest-neighbor similarity graph.
3. Clients independently train their GraphSAGE models.
4. Model updates are encrypted using CKKS.
5. The server aggregates encrypted updates.
6. A multi-agent defense pipeline evaluates and handles suspicious updates.
7. The updated global model is distributed to clients.
8. Performance metrics and experiment results are visualized through the dashboard.

## Privacy and Security

The framework incorporates multiple protection mechanisms:

- CKKS homomorphic encryption for model-update aggregation.
- FF3-1 format-preserving encryption for selected values.
- Gradient clipping and Gaussian noise-based defense.
- Differential privacy budget accounting.
- Malicious-client simulation and anomaly detection.
- Audit logging of federated operations.

**Note:** These mechanisms demonstrate privacy-oriented design. They do not constitute a guarantee of complete privacy or security against every attack.

## Project Structure

```text
federated_gnn_breast_cancer/
│
├── agents/              # Multi-agent coordination
├── attacks/             # Adversarial attack simulations
├── baseline/            # Baseline implementations
├── checkpoints/         # Saved model checkpoints
├── client/              # Federated clients and data loading
├── crypto/              # CKKS and FPE encryption
├── defense/             # Defense mechanisms
├── evaluation/          # Model evaluation metrics
├── explainability/      # SHAP and rule-based explanations
├── experiments/         # Experiment pipeline and analysis
├── models/              # GraphSAGE architecture
├── privacy/             # Differential privacy accounting
├── results/
│   ├── figures/         # Experiment visualizations
│   ├── raw/             # Raw experiment results
│   └── summary/         # Result tables
├── logs/                # Audit and training logs
├── testing/             # Testing utilities
├── tests/               # Automated tests
│
├── data/
│   └── wdbc.csv
│
├── config.py
├── dashboard.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/privacy-preserving-federated-gnn.git
cd privacy-preserving-federated-gnn
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS / WSL:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Note: Pyfhel may require additional native build dependencies. On Windows, MSVC C++17 Build Tools may be needed. Refer to the installation notes in the project configuration before switching encryption backends.

## Running the Project

### Run automated tests

```bash
python -m pytest tests -q
```

### Run federated training

```bash
python main.py
```

### Launch the dashboard

```bash
streamlit run dashboard.py
```

Open the local Streamlit URL displayed in the terminal.

## Experiments and Results

The project includes experimental comparisons and visualizations covering:

- CKKS-encrypted versus plaintext aggregation.
- Accuracy, precision, recall, F1-score, and ROC-AUC.
- Training and communication performance.
- Ablation studies.
- Malicious-client attack and defense performance.
- Graph homophily analysis.
- Privacy budget analysis.
- Encryption approximation error.

See `results/summary/` for experiment tables and `results/figures/` for visualizations.

## Dataset

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset (WDBC), containing patient diagnostic features for benign and malignant classification.

The dataset is used for research and educational experimentation.

## Disclaimer

This project is intended for academic research and educational purposes only. It is not a clinically validated diagnostic system and must not be used for medical diagnosis or treatment decisions.

## Future Improvements

- Explore stronger privacy-preserving aggregation strategies.
- Improve robustness against advanced adversarial attacks.
- Evaluate performance on larger and more diverse datasets.
- Extend explainability and privacy auditing.
- Explore deployment in distributed healthcare environments.



⭐ If you find this project interesting, consider giving the repository a star!
