"""Headless render test of every dashboard section (skipped if no results)."""
import os

import pytest

from experiments import results_io as rio

pytestmark = pytest.mark.skipif(not os.path.exists(rio.RUNS_PATH) or rio.load_selection() is None,
                                reason="no measured results yet")
SECTIONS = ["Overview", "Final Model Performance", "CKKS vs Plaintext", "Communication Rounds",
            "Ablation Study", "Attack & Defense", "Hyperparameter Search", "Privacy / CKKS Analysis",
            "Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Experimental Logs / Raw Results"]


@pytest.mark.parametrize("section", SECTIONS)
def test_dashboard_section_renders(section):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../dashboard.py", default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(section).run()
    assert not at.exception, [e.value for e in at.exception]


def test_dashboard_numbers_match_summary_csv():
    import pandas as pd
    from streamlit.testing.v1 import AppTest
    import config
    at = AppTest.from_file("../dashboard.py", default_timeout=120)
    at.run()
    cvp = pd.read_csv(os.path.join(config.SUMMARY_DIR, "final_ckks_vs_plaintext.csv")).set_index("metric")
    shown = {}
    for m in at.metric:
        shown.setdefault(m.label, m.value)
    assert shown["Accuracy"] == f"{100 * cvp.loc['accuracy', 'ckks_mean']:.2f}%"
    assert shown["F1"] == f"{100 * cvp.loc['f1', 'ckks_mean']:.2f}%"
    assert shown["Precision"] == f"{100 * cvp.loc['precision', 'ckks_mean']:.2f}%"
    assert shown["Recall"] == f"{100 * cvp.loc['recall', 'ckks_mean']:.2f}%"
    assert shown["ROC-AUC"] == f"{100 * cvp.loc['roc_auc', 'ckks_mean']:.2f}%"
