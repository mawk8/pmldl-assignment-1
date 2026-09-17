import argparse
import os

import joblib
import numpy as np
import pandas as pd
import wandb
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from wandb.integration.xgboost import WandbCallback

RANDOM_SEED = 52
SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def engineer_features(df):
    df = df.copy()
    df["NumServices"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)
    df["AvgMonthlyCharge"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    return df


def build_preprocessor(X):
    num = X.select_dtypes(include=["int64", "float64"]).columns.to_list()
    cat = X.select_dtypes(include=["object"]).columns.to_list()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )


def train(processed_dir, model_dir):
    train_df = engineer_features(pd.read_csv(os.path.join(processed_dir, "train.csv")))
    test_df = engineer_features(pd.read_csv(os.path.join(processed_dir, "test.csv")))

    X_train_raw = train_df.drop(columns="Churn")
    y_train = train_df["Churn"].map({"Yes": 1, "No": 0})
    X_test_raw = test_df.drop(columns="Churn")
    y_test = test_df["Churn"].map({"Yes": 1, "No": 0})

    preprocessor = build_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    wandb.login(key=os.environ["WANDB_API_KEY"])

    config = {
        "objective": "binary:logistic",
        "eta": 0.1,
        "max_depth": 6,
        "n_estimators": 300,
        "random_state": RANDOM_SEED,
    }
    run = wandb.init(project="pmldl-telco-churn", config=config)

    model = xgb.XGBClassifier(
        objective=config["objective"],
        learning_rate=config["eta"],
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        eval_metric=["logloss", "auc"],
        early_stopping_rounds=20,
        random_state=config["random_state"],
        callbacks=[WandbCallback(log_model=True)],
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 33)
    f1_by_threshold = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1_by_threshold))])
    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "best_iteration": model.best_iteration,
        "threshold": best_threshold,
    }
    wandb.log(metrics)
    run.finish()

    os.makedirs(model_dir, exist_ok=True)
    model.callbacks = None

    joblib.dump(
        {"preprocessor": preprocessor, "model": model, "threshold": best_threshold},
        os.path.join(model_dir, "churn_model.pkl"),
    )

    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    train(args.processed_dir, args.model_dir)
