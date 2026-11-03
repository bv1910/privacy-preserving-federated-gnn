"""
Central configuration for the Privacy-Preserving Federated GraphSAGE system.
Dataset: Breast Cancer Wisconsin (Diagnostic) — sklearn.datasets.load_breast_cancer
Model:   GraphSAGE (per-client k-NN similarity graph over patient records)
"""
import os

# ---------------------------------------------------------------------------
# CPU ONLY: hide any GPU from torch before it is imported anywhere.
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Federated learning params
# ---------------------------------------------------------------------------
NUM_CLIENTS = 10
NUM_ROUNDS = 8
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.05
NUM_MALICIOUS_CLIENTS = 2          # simulate Byzantine/adversarial clients
MALICIOUS_CLIENT_IDS = [3, 7]      # fixed for reproducibility

# ---------------------------------------------------------------------------
# Dataset / graph construction params
# ---------------------------------------------------------------------------
DATASET_TEST_SPLIT = 0.2           # per-client held-out test fraction
DATASET_VAL_SPLIT = 0.2            # fraction of the NON-test records held out for validation (tuning only)
KNN_NEIGHBORS = 6                  # k for the per-client similarity graph
RANDOM_SEED = 42
NON_IID = False                    # True -> Dirichlet partition by label

DATASET_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "wdbc.csv"
)  # standard WDBC schema: id, diagnosis, 30 feature columns

# ---------------------------------------------------------------------------
# GraphSAGE model params
# ---------------------------------------------------------------------------
IN_CHANNELS = 30                   # breast cancer feature count
HIDDEN_CHANNELS = 32
OUT_CHANNELS = 2                   # benign / malignant
NUM_SAGE_LAYERS = 2
DROPOUT = 0.3

# ---------------------------------------------------------------------------
# CKKS params (Pyfhel)
# ---------------------------------------------------------------------------
# NOTE: Pyfhel requires an MSVC C++17 compiler (Visual Studio Build Tools,
# "Desktop development with C++" workload) to build on Windows the first
# time `pip install Pyfhel` runs from source. If Build Tools are not
# available, run this project inside WSL2 and swap CKKS_BACKEND to
# "tenseal" (TenSEAL has native Linux/WSL2 wheels; no Windows wheels exist).
CKKS_BACKEND = "pyfhel"            # "pyfhel" | "tenseal"
POLY_MODULUS_DEGREE = 8192
CKKS_SCALE = 2 ** 30
QI_SIZES = [60, 30, 30, 60]
CKKS_SLOTS = POLY_MODULUS_DEGREE // 2   # 4096 usable slots per ciphertext

# ---------------------------------------------------------------------------
# Agent thresholds
# ---------------------------------------------------------------------------
ANOMALY_THRESHOLD = 0.7            # score >= this -> flagged "high" risk
CONFIDENCE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Defense params (Full Defense path)
# ---------------------------------------------------------------------------
CLIP_THRESHOLD = 1.0
NOISE_SCALE = 0.01

# ---------------------------------------------------------------------------
# FPE (format-preserving encryption of test metrics)
# ---------------------------------------------------------------------------
FPE_KEY = os.environ.get("FPE_KEY", "EF4359D8D580AA4F7F036D6F04FC6A94")
FPE_TWEAK = os.environ.get("FPE_TWEAK", "D8E7920AFA330A73")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
AUDIT_LOG_PATH = os.path.join(LOG_DIR, "audit_log.jsonl")
ROUND_RESULTS_PATH = os.path.join(LOG_DIR, "round_results.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
for _d in (RESULTS_DIR, RAW_RESULTS_DIR, FIGURES_DIR, SUMMARY_DIR):
    os.makedirs(_d, exist_ok=True)
