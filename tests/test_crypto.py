import numpy as np

from crypto.ckks_engine import CKKSEngine
from crypto.fpe_engine import FPEEngine
from models.graphsage_model import GraphSAGENet


def test_ckks_roundtrip():
    engine = CKKSEngine()
    m1, m2 = GraphSAGENet(), GraphSAGENet()
    flat1, shapes = engine.flatten_state_dict(m1.state_dict())
    flat2, _ = engine.flatten_state_dict(m2.state_dict())

    ct1, ct2 = engine.encrypt_vector(flat1), engine.encrypt_vector(flat2)
    agg = engine.aggregate_encrypted([ct1, ct2])
    decrypted = engine.decrypt_vector(agg, len(flat1))

    expected = (flat1 + flat2) / 2
    assert np.abs(decrypted - expected).max() < 0.01


def test_fpe_roundtrip_exact():
    engine = FPEEngine()
    for val in [0.9523, 0.1847, 0.0001, 0.9999]:
        enc = engine.encrypt_metric(val)
        dec = engine.decrypt_metric(enc)
        assert abs(dec - val) < 1e-6


def test_fpe_hyperparam_roundtrip_exact():
    engine = FPEEngine()
    enc = engine.encrypt_hyperparams(learning_rate=0.05, local_epochs=3)
    dec = engine.decrypt_hyperparams(enc)
    assert abs(dec["learning_rate"] - 0.05) < 1e-9
    assert dec["local_epochs"] == 3
    # ciphertext must not leak plaintext values
    assert enc["learning_rate"] != "050000"
    assert enc["local_epochs"] != "000003"


if __name__ == "__main__":
    test_ckks_roundtrip()
    test_fpe_roundtrip_exact()
    test_fpe_hyperparam_roundtrip_exact()
    print("test_crypto.py: all tests passed")
