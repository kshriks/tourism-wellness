"""
Lightweight smoke tests for the pipeline scripts. Not wired into the CI
workflow by default, but can be added as an extra job
(`pip install pytest && pytest tests/`) for an additional quality gate.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.data_validation import validate
from scripts.data_prep import clean


def test_validate_returns_dataframe():
    df = validate()
    assert df.shape[0] > 0
    assert "ProdTaken" in df.columns


def test_clean_fixes_known_typos():
    import pandas as pd
    df = pd.DataFrame({
        "Gender": ["Fe Male", "Male"],
        "MaritalStatus": ["Unmarried", "Married"],
        "ProdTaken": [0, 1],
    })
    cleaned = clean(df)
    assert "Fe Male" not in cleaned["Gender"].values
    assert "Unmarried" not in cleaned["MaritalStatus"].values
