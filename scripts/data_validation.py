"""
data_validation.py
-------------------
Registers the raw dataset for the "Wellness Tourism Package" MLOps pipeline.

What it does:
1. Loads data/tourism.csv
2. Checks that every expected column is present (and flags any unexpected ones)
3. Checks basic data-health conditions (row count, duplicate CustomerIDs, target
   column present and binary)
4. Prints a summary of the dataset (shape, dtypes, missing values, target balance)

Used as the first job of the GitHub Actions pipeline (pipeline.yml -> job: data-registration).
Exits with a non-zero status code if validation fails, which fails the CI job.
"""

import sys
import pandas as pd

DATA_PATH = "data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def validate(path: str = DATA_PATH) -> pd.DataFrame:
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    # Drop the stray index column some exports of this dataset carry
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    assert not missing_cols, f"Missing expected columns: {missing_cols}"
    assert df.shape[0] > 0, "Dataset has zero rows."
    assert "ProdTaken" in df.columns, "Target column 'ProdTaken' not found."
    assert set(df["ProdTaken"].unique()).issubset({0, 1}), \
        "Target column 'ProdTaken' must be binary (0/1)."
    assert df["CustomerID"].is_unique, "Duplicate CustomerID values found."

    print("\nVALIDATION PASSED")
    if extra_cols:
        print(f"Note: extra columns present (kept as-is): {extra_cols}")

    print("\n----- Dataset Summary -----")
    print(f"Rows x Columns : {df.shape}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nTarget distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts())
    print(df["ProdTaken"].value_counts(normalize=True).round(3))

    return df


if __name__ == "__main__":
    try:
        validate()
    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}")
        sys.exit(1)
