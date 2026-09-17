import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "models/churn_model.pkl")
SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

app = FastAPI(title="Churn Prediction API")
artifact = joblib.load(MODEL_PATH)


class ChurnRequest(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


def engineer_features(df):
    df = df.copy()
    df["NumServices"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)
    df["AvgMonthlyCharge"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    return df


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: ChurnRequest):
    df = pd.DataFrame([request.model_dump()])
    df = engineer_features(df)

    X = artifact["preprocessor"].transform(df)
    proba = float(artifact["model"].predict_proba(X)[:, 1][0])
    churn = bool(proba >= artifact["threshold"])

    return {"churn_probability": proba, "churn": churn, "threshold": artifact["threshold"]}


@app.post("/reload")
def reload_model():
    global artifact
    artifact = joblib.load(MODEL_PATH)
    return {"status": "reloaded"}
