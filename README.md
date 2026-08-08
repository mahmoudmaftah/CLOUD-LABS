# Cloud & MLOps Labs

Coursework labs on serving machine-learning models, tracking experiments and deploying to the cloud. Each top-level folder is a self-contained lab with its own dependencies.

## Contents

### `Lab Folder - Flask`

Flask fundamentals — routing, Jinja templates and form handling — building up to a wine-quality prediction web app. The assignment folder holds two versions: a minimal one (`SIMPLE TEST (WINE)`) and an extended one (`ADVANCED TEST (WINE QUALITY DATASET)`) with a fuller preprocessing pipeline and a saved `joblib` artifact.

### `Lab folder - FastAPI Pydantic`

Three related exercise sets:

- **`fastapi/`** — prediction endpoints backed by scikit-learn models (iris classification, furniture price).
- **`pydantic/`** — schema and validation exercises (`ex1.py` … `ex3.py`).
- **`Pytest/`** — unit-testing exercises covering fixtures and shared setup via `conftest.py`.
- **`todo/`** — a small app serving both the wine and furniture models behind HTML forms.

### `Lab Folder MLOps`

MLflow experiment tracking across three backends — scikit-learn, TensorFlow/Keras and PySpark — plus local model serving and a Databricks-hosted MLflow run. Notebooks are numbered in the order they were worked through. Output screenshots for each lab are under `screens/`.

### `6. Azure MLflow Deploy`

Training a Ridge regression model on the diabetes dataset and deploying it to Azure ML:

- `Train & Deploy local and ACI.ipynb` — local training then deployment to Azure Container Instances
- `register.py` — model registration against the workspace
- `score.py` — scoring entry point used by the deployed endpoint
- `EDA/` — exploratory figures (correlation matrix, PCA projection, model comparison)
- `model/` — serialised model, scaler and metadata (test R² ≈ 0.46)

## Running a lab

Each folder is independent. Where a `requirements.txt` exists, install it into a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r "<lab folder>/requirements.txt"
```

### Azure configuration

The Azure lab needs a workspace configuration file, which is **not** committed. Copy the template and fill in your own values:

```bash
cp "6. Azure MLflow Deploy/config.json.example" "6. Azure MLflow Deploy/config.json"
```

`config.json` is listed in `.gitignore` — it identifies an Azure subscription and should not be published.

## Reports

Written lab reports are in `REPORT 1.pdf`, `REPORT 2.pdf` and the corresponding `.docx` files.

## Status

University coursework, kept for reference. Not actively maintained.
