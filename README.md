# Wellness Tourism Package — MLOps Pipeline

Predicts whether a customer will purchase "Visit with Us"'s new Wellness
Tourism Package, so sales can target the right customers before contacting
them. Built as an end-to-end MLOps pipeline: data registration → data
preparation → model training with experiment tracking → Streamlit deployment
→ GitHub Actions CI/CD.

## Repository structure
```
tourism_mlops/
├── data/
│   ├── tourism.csv              # raw dataset
│   ├── train.csv                # created by scripts/data_prep.py
│   └── test.csv                 # created by scripts/data_prep.py
├── scripts/
│   ├── data_validation.py       # dataset registration / validation
│   ├── data_prep.py             # cleaning + train/test split
│   ├── train.py                 # model tuning, MLflow tracking, best-model selection
│   ├── deploy_github.sh         # pushes this folder to GitHub (triggers CI + Streamlit redeploy)
│   └── check_streamlit_app.py   # polls the live app URL to confirm the redeploy is up
├── model/
│   ├── best_model.joblib        # committed by the pipeline after each run
│   ├── metrics_summary.csv      # test-set metrics for every candidate model
│   └── best_model_info.json     # which model won and why
├── app.py                       # Streamlit inference app
├── requirements.txt
├── .github/workflows/pipeline.yml
│   ├── job: data-registration
│   ├── job: data-preparation
│   └── job: model-training
└── README.md
```

## Pipeline stages
1. **Data registration** (`scripts/data_validation.py`) — checks the dataset
   has the expected columns and a binary target, prints a summary.
2. **Data preparation** (`scripts/data_prep.py`) — drops ID columns, fixes
   inconsistent category labels (`Fe Male` → `Female`, `Unmarried` → `Single`),
   drops duplicate rows, splits 80/20 train/test (stratified on the target),
   saves both locally and passes them to the next job as a workflow artifact.
3. **Model training** (`scripts/train.py`) — tunes Decision Tree, Bagging,
   Random Forest, AdaBoost, Gradient Boosting and XGBoost with
   `GridSearchCV`, logs every tuned parameter set and test metric to MLflow
   (local SQLite tracking store, `mlflow.db`), selects the best model by
   test-set recall on the "purchased" class, and saves it to
   `model/best_model.joblib`.
4. **Deployment** (`app.py`) — a Streamlit app that loads
   `model/best_model.joblib`, collects a customer profile through form
   inputs, and returns a purchase-likelihood prediction.
5. **CI/CD** (`.github/workflows/pipeline.yml`) — runs stages 1–3 on every
   push to `main` that touches the data or pipeline code, and commits the
   retrained model back to the repo. Streamlit Community Cloud is connected
   to this repo's `main` branch and redeploys automatically whenever it
   changes — no separate deploy step needed in the workflow.

## One-time setup for the GitHub Actions pipeline
- **Settings → Actions → General → Workflow permissions** → "Read and
  write permissions" (lets the `model-training` job's `git push` work).

## Running locally
```bash
pip install -r requirements.txt
python scripts/data_validation.py
python scripts/data_prep.py
python scripts/train.py
streamlit run app.py
```

## Deploying
Streamlit Community Cloud has no separate push step or deploy API — once
an app is connected to this repo (one-time, at share.streamlit.io), the
`git push` below is what deploys it.

```bash
# 1. Push this folder to your existing GitHub repo — this is what deploys
#    both the model (via CI) and the Streamlit app (via its GitHub
#    connection, redeploying automatically on push to main)
./scripts/deploy_github.sh <your-existing-repo-url>

# 2. Confirm the Streamlit redeploy is actually live (and wake it if it
#    was hibernating) — this is a readiness check, not a deploy trigger
python scripts/check_streamlit_app.py <your-app>.streamlit.app
```

## Deployed links
- **GitHub repository:** <ADD_LINK_HERE>
- **Streamlit Community Cloud app:** <ADD_LINK_HERE>

## Model summary
See `model/metrics_summary.csv` for the full comparison; the pipeline
currently selects **XGBoost** (test accuracy ≈ 0.94, recall ≈ 0.81 on
customers who purchased the package).
