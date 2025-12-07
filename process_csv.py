import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV into pandas DataFrame."""
    return pd.read_csv(path)


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a dictionary with useful summary statistics."""
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "numeric_describe": df.select_dtypes(include=[np.number]).describe().to_dict(),
    }
    return summary


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Perform common cleaning steps on the Telco dataset.

    - Strip string columns
    - Convert TotalCharges to numeric (coerce errors)
    - Fill or infer missing TotalCharges
    - Normalize boolean-like columns
    """
    df = df.copy()

    # Strip whitespace from object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # TotalCharges sometimes has spaces, coerce to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace("", np.nan), errors="coerce")
        # If TotalCharges missing, infer from MonthlyCharges * tenure when possible
        if "MonthlyCharges" in df.columns and "tenure" in df.columns:
            mask = df["TotalCharges"].isna() & df["MonthlyCharges"].notna() & df["tenure"].notna()
            df.loc[mask, "TotalCharges"] = (df.loc[mask, "MonthlyCharges"].astype(float) * df.loc[mask, "tenure"].astype(float))
        # If still missing, fill with 0
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Ensure SeniorCitizen is integer
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # Normalize common Yes/No columns to binary where applicable
    yes_no = [c for c in df.columns if df[c].dropna().isin(["Yes", "No"]).all()]
    for c in yes_no:
        df[c] = df[c].map({"Yes": 1, "No": 0}).astype(int)

    return df


def prepare_features(df: pd.DataFrame, drop_cols: list = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Encode categorical variables and return feature matrix X and target y (Churn).

    By default this function will drop `customerID` if present and will treat `Churn` as the target.
    """
    df = df.copy()
    if drop_cols is None:
        drop_cols = []
    if "customerID" in df.columns:
        drop_cols = drop_cols + ["customerID"]

    target = None
    if "Churn" in df.columns:
        # Support both string Yes/No and already-numeric targets
        if pd.api.types.is_numeric_dtype(df["Churn"]):
            target = df["Churn"].astype(int)
        else:
            target = df["Churn"].map({"Yes": 1, "No": 0})

    # Select features
    features = df.drop(columns=[c for c in (drop_cols + (["Churn"] if "Churn" in df.columns else []))], errors="ignore")

    # Convert numeric-ish columns
    for c in features.columns:
        if features[c].dtype == object:
            # small optimization: if column looks numeric, convert
            try:
                features[c] = pd.to_numeric(features[c])
            except Exception:
                pass

    # One-hot encode remaining categoricals
    X = pd.get_dummies(features, drop_first=True)

    if target is None:
        raise ValueError("Target column 'Churn' not found in dataframe")

    return X, target


def save_cleaned(df: pd.DataFrame, out_path: str) -> None:
    dirpath = os.path.dirname(out_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process Telco CSV")
    parser.add_argument("input", help="input CSV path")
    parser.add_argument("--out", help="clean CSV output path", default="cleaned_telco.csv")
    args = parser.parse_args()

    df = load_csv(args.input)
    print("Loaded", df.shape)
    dfc = clean(df)
    save_cleaned(dfc, args.out)
    print("Saved cleaned CSV to", args.out)
