import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no popup windows (headless)
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
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE / "models"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "diabetes-risk-classifier"
MIN_F1_TO_PROMOTE = 0.90

FAKE_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
MIN_ZERO_COLS = FAKE_ZERO_COLS + ["DiabetesPedigreeFunction"]


def extract():
    df = pd.read_csv(DATA_DIR / "dataset-diabete.csv")
    df.to_csv(DATA_DIR / "extracted.csv", index=False)
    print("extracted", df.shape)


def preprocess():
    df = pd.read_csv(DATA_DIR / "extracted.csv")
    df[FAKE_ZERO_COLS] = df[FAKE_ZERO_COLS].replace(0, np.nan)
    df = df[df[FAKE_ZERO_COLS].isnull().sum(axis=1) < 4].reset_index(drop=True)

    cols = df.columns.tolist()
    df[cols] = IterativeImputer(random_state=42).fit_transform(df[cols])
    df["Pregnancies"] = df["Pregnancies"].round().astype(int)
    df["Age"] = df["Age"].round().astype(int)
    df[MIN_ZERO_COLS] = df[MIN_ZERO_COLS].clip(lower=0)

    df.to_csv(DATA_DIR / "processed.csv", index=False)
    print("processed", df.shape)


def retrain():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("diabetes-risk-prediction")
    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_DIR / "processed.csv")
    features = df.columns.tolist()

    # --- clustering ---
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=features)
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(df_scaled)

    with mlflow.start_run(run_name="kmeans_clustering"):
        mlflow.log_param("k", 2)
        mlflow.log_param("algorithm", "KMeans")
        mlflow.log_param("scaler", "RobustScaler")
        mlflow.log_metric("inertia", km.inertia_)
        mlflow.log_metric("silhouette_score", silhouette_score(df_scaled, df["Cluster"]))
        mlflow.log_dict({"columns": features,
                         "center_median": scaler.center_.tolist(),
                         "scale_iqr": scaler.scale_.tolist()}, "scaler_params.json")
        joblib.dump(km, MODELS_DIR / "kmeans_model.joblib")
        joblib.dump(scaler, MODELS_DIR / "robust_scaler.joblib")
        mlflow.log_artifact(str(MODELS_DIR / "kmeans_model.joblib"))
        mlflow.log_artifact(str(MODELS_DIR / "robust_scaler.joblib"))

    # --- risk labeling: the cluster whose means cross the most thresholds is "high risk" ---
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

        # reproducibility
        mlflow.log_param("features", features)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("pandas_version", pd.__version__)
        mlflow.log_dict({"n_rows": len(df), "n_cols": len(features), "columns": features,
                         "dtypes": {c: str(t) for c, t in X.dtypes.items()}},
                        "dataset_signature.json")

        mlflow.sklearn.log_model(
            pipeline, name="pipeline", registered_model_name=MODEL_NAME,
            pip_requirements=[f"scikit-learn=={sklearn.__version__}"])

    # --- promote only if the new model is good enough ---
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