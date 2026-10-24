"""
train.py
--------
Loads the train/test split produced by data_prep.py (downloaded as a workflow
artifact in CI), builds a preprocessing + model pipeline, tunes six candidate
classifiers with GridSearchCV, logs every run (parameters + metrics) to MLflow,
evaluates each tuned model on the held-out test set, and saves the best model
(by test-set recall on the positive/"purchased" class, since missing a likely
buyer is costlier to the business than one extra follow-up call) to
model/best_model.joblib so the pipeline can commit it to the repository.

Used as the third job of the GitHub Actions pipeline (pipeline.yml -> job: model-training).

MLflow is run with a local file-store tracking URI (./mlruns) so no external
tracking server or credentials are required inside CI; the mlruns folder (or
just the metrics summary CSV) can be inspected directly from the repo / artifact.
"""

import json
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
MODEL_PATH = "model/best_model.joblib"
METRICS_PATH = "model/metrics_summary.csv"
TARGET = "ProdTaken"

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# Model + hyperparameter grid registry.
# Allowed families per the project rubric: Decision Tree, Bagging, Random
# Forest, AdaBoost, Gradient Boosting, XGBoost.
MODEL_REGISTRY = {
    "DecisionTree": (
        DecisionTreeClassifier(random_state=1, class_weight="balanced"),
        {"model__max_depth": [4, 6, 8], "model__min_samples_leaf": [5, 10]},
    ),
    "Bagging": (
        BaggingClassifier(random_state=1),
        {"model__n_estimators": [50, 100], "model__max_samples": [0.7, 1.0]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=1, class_weight="balanced"),
        {"model__n_estimators": [100, 200], "model__max_depth": [6, 10, None]},
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=1),
        {"model__n_estimators": [50, 100], "model__learning_rate": [0.5, 1.0]},
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=1),
        {"model__n_estimators": [100, 150], "model__learning_rate": [0.05, 0.1]},
    ),
    "XGBoost": (
        XGBClassifier(random_state=1, eval_metric="logloss"),
        {"model__n_estimators": [100, 200], "model__max_depth": [3, 5]},
    ),
}


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def train_and_track(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    # Local SQLite-backed MLflow tracking store — no external server or
    # credentials required inside CI. mlflow.db (plus the artifact store
    # under mlruns/) is written to the repo working directory.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("wellness-tourism-package")

    results = []
    best_score, best_name, best_pipeline = -1, None, None

    for name, (estimator, param_grid) in MODEL_REGISTRY.items():
        pipe = Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ])
        search = GridSearchCV(
            pipe, param_grid, scoring="recall", cv=5, n_jobs=-1,
        )

        with mlflow.start_run(run_name=name):
            search.fit(X_train, y_train)
            tuned_model = search.best_estimator_

            y_pred = tuned_model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
            }

            # Log every tuned hyperparameter and every evaluation metric
            mlflow.log_params({k.replace("model__", ""): v
                                for k, v in search.best_params_.items()})
            mlflow.log_param("cv_best_score_recall", search.best_score_)
            mlflow.log_metrics(metrics)

            # Log the tuned pipeline as a plain joblib artifact (avoids the
            # extra untrusted-type checks MLflow's sklearn flavor performs
            # when *loading* models, since this run only needs to persist it).
            tmp_path = f"model/_tmp_{name}.joblib"
            joblib.dump(tuned_model, tmp_path)
            mlflow.log_artifact(tmp_path, artifact_path="model")
            os.remove(tmp_path)

            print(f"[{name}] best_params={search.best_params_} "
                  f"test_metrics={ {k: round(v, 4) for k, v in metrics.items()} }")

            results.append({"model": name, **search.best_params_, **metrics})

            if metrics["recall"] > best_score:
                best_score = metrics["recall"]
                best_name = name
                best_pipeline = tuned_model

    results_df = pd.DataFrame(results).sort_values("recall", ascending=False)
    results_df.to_csv(METRICS_PATH, index=False)

    print(f"\nBest model: {best_name} (test recall = {best_score:.4f})")
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Saved best model pipeline -> {MODEL_PATH}")

    with open("model/best_model_info.json", "w") as f:
        json.dump({"best_model": best_name, "test_recall": best_score}, f, indent=2)

    print("\nClassification report for the best model:")
    print(classification_report(y_test, best_pipeline.predict(X_test)))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, best_pipeline.predict(X_test)))

    return results_df, best_name, best_pipeline


if __name__ == "__main__":
    train_and_track()
