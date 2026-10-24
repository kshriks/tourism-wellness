"""
data_prep.py
------------
Loads the raw dataset directly from the repository's data/ folder, cleans it,
removes unnecessary columns, and splits it into train and test sets.

Used as the second job of the GitHub Actions pipeline (pipeline.yml -> job: data-preparation).
The resulting train.csv / test.csv are saved locally under data/ and are also
uploaded as a workflow artifact ("train-test-data") so the next job (model
training) can download them without re-running this job.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/tourism.csv"
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

# Columns that are identifiers / leak no predictive signal and should be dropped
DROP_COLS = ["Unnamed: 0", "CustomerID"]

TARGET = "ProdTaken"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop identifier / index columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Fix inconsistent category labels found during EDA
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Drop exact duplicate rows, if any
    df = df.drop_duplicates()

    # Impute any missing numeric values with the median, categorical with the mode
    # (the pipeline is written defensively in case a future data refresh has gaps,
    # even though the current snapshot has none)
    num_cols = df.select_dtypes(include="number").columns.drop(TARGET, errors="ignore")
    # Selected by elimination (rather than naming "object"/"string" dtypes
    # explicitly) so this works the same whether pandas is storing text
    # columns as the legacy "object" dtype or the newer "string" dtype.
    cat_cols = df.columns.difference(num_cols.tolist() + [TARGET])

    for c in num_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    return df


def prepare(raw_path: str = RAW_PATH, test_size: float = 0.2, random_state: int = 1):
    df = pd.read_csv(raw_path)
    df = clean(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = X_train.copy()
    train_df[TARGET] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Cleaned rows: {df.shape[0]} (from raw {pd.read_csv(raw_path).shape[0]})")
    print(f"Train set : {train_df.shape} -> {TRAIN_PATH}")
    print(f"Test set  : {test_df.shape} -> {TEST_PATH}")
    print("\nTrain target balance:")
    print(train_df[TARGET].value_counts(normalize=True).round(3))
    print("\nTest target balance:")
    print(test_df[TARGET].value_counts(normalize=True).round(3))

    return train_df, test_df


if __name__ == "__main__":
    prepare()
