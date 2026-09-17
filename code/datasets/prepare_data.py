import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 52
OUTLIER_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def remove_outliers_iqr(data, columns):
    mask = pd.Series(True, index=data.index)
    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= data[col].between(lower, upper)
    return data[mask]


def prepare_data(raw_path, processed_dir):
    df = pd.read_csv(raw_path)
    df = df.drop(columns="customerID")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    df = remove_outliers_iqr(df, OUTLIER_COLUMNS)

    train_df, test_df = train_test_split(
        df, train_size=0.8, stratify=df["Churn"], random_state=RANDOM_SEED
    )

    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    print(f"train: {train_df.shape}, test: {test_df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    prepare_data(args.raw_path, args.processed_dir)
