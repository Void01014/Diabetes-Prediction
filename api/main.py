import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.sklearn.load_model("models:/diabetes-risk-classifier/Production")

app = FastAPI()


class Patient(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


@app.post("/predict")
def predict(patient: Patient):
    df = pd.DataFrame([patient.model_dump()])
    pred = int(model.predict(df)[0])
    return {"risk_category": pred, "risk_label": "high" if pred == 1 else "low"}