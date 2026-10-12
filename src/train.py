import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import RandomOverSampler
from mlflow.tracking import MlflowClient
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# --- settings ---
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "diabetes-risk-classifier"
MIN_F1_TO_PROMOTE = 0.90

FAKE_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
MIN_ZERO_COLS = 4  # drop rows missing at least this many of those columns

features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


def extract():
    """Copy raw CSV to DATA_DIR/extracted.csv (no transformation)."""
    df = pd.read_csv(DATA_DIR / "dataset-diabete.csv")
    df.to_csv(DATA_DIR / "extracted.csv", index=False)
    print(f"extract: {len(df)} rows -> {DATA_DIR / 'extracted.csv'}")


def preprocess():
    """Clean and impute; write DATA_DIR/processed.csv."""
    df = pd.read_csv(DATA_DIR / "extracted.csv")

    # treat zeros in clinical columns as missing
    for col in FAKE_ZERO_COLS:
        df[col] = df[col].replace(0, np.nan)

    # drop rows missing >= MIN_ZERO_COLS of those columns
    missing_counts = df[FAKE_ZERO_COLS].isna().sum(axis=1)
    df = df[missing_counts < MIN_ZERO_COLS].copy()
    print(f"preprocess: {len(df)} rows after dropping sparse rows")

    # MICE imputation on all 8 columns
    imputer = IterativeImputer(random_state=42)
    df[features] = imputer.fit_transform(df[features])

    # round integer columns back; clip negatives produced by MICE
    df["Pregnancies"] = df["Pregnancies"].round().astype(int)
    df["Age"] = df["Age"].round().astype(int)
    for col in FAKE_ZERO_COLS + ["DiabetesPedigreeFunction"]:
        df[col] = df[col].clip(lower=0)

    df.to_csv(DATA_DIR / "processed.csv", index=False)
    print(f"preprocess: saved {DATA_DIR / 'processed.csv'}")


def retrain():
    """Cluster -> label -> classify -> log to MLflow -> promote if f1 >= threshold."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("diabetes-risk-prediction")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_DIR / "processed.csv")

    # --- clustering ---
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    with mlflow.start_run(run_name="kmeans_clustering"):
        mlflow.log_param("n_clusters", 2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_init", 10)
        mlflow.log_metric("inertia", kmeans.inertia_)
        cluster_sizes = df["Cluster"].value_counts().to_dict()
        mlflow.log_param("cluster_sizes", str(cluster_sizes))
        scaler_params = {
            "center": scaler.center_.tolist(),
            "scale": scaler.scale_.tolist(),
        }
        mlflow.log_dict(scaler_params, "scaler_params.json")
        mlflow.log_param("features", features)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("pandas_version", pd.__version__)

    # --- risk labeling: cluster whose means cross the most thresholds is high risk ---
    means = df.groupby("Cluster")[["Glucose", "BMI", "DiabetesPedigreeFunction"]].mean()
    score = ((means["Glucose"] > 126).astype(int)
             + (means["BMI"] > 30).astype(int)
             + (means["DiabetesPedigreeFunction"] > 0.5).astype(int)
             + means["Glucose"] / 1000)  # tiebreaker
    high_cluster = score.idxmax()
    df["risk_category"] = (df["Cluster"] == high_cluster).astype(int)

    # --- classification ---
    X, y = df[features], df["risk_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_res, y_res = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)

    grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000),
                        {"C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
    grid.fit(X_res, y_res)

    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000, **grid.best_params_)),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    joblib.dump(pipeline, MODELS_DIR / "diabetes_risk_pipeline.joblib")

    with mlflow.start_run(run_name="logistic_regression_tuned"):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({"accuracy": accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred),
                            "recall": recall_score(y_test, y_pred),
                            "f1": f1})
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        cm_path = MODELS_DIR / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(str(cm_path))
        mlflow.log_param("features", features)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("pandas_version", pd.__version__)
        mlflow.log_dict({"n_rows": len(df), "n_cols": len(features), "columns": features,
                         "dtypes": {c: str(t) for c, t in X.dtypes.items()}},
                        "dataset_signature.json")
        mlflow.sklearn.log_model(
            pipeline, name="pipeline", registered_model_name=MODEL_NAME,
            pip_requirements=[f"scikit-learn=={sklearn.__version__}"])

    client = MlflowClient()
    version = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
    if f1 >= MIN_F1_TO_PROMOTE:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=version, stage="Production",
            archive_existing_versions=True)
        print(f"version {version} registered and promoted to Production (f1={f1:.3f})")
    else:
        print(f"version {version} registered but NOT promoted (f1={f1:.3f} < {MIN_F1_TO_PROMOTE})")


if __name__ == "__main__":
    extract()
    preprocess()
    retrain()
