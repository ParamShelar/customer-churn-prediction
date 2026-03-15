import pandas as pd


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Drop ID column
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing TotalCharges
    df = df.dropna(subset=["TotalCharges"])

    # Map target to 0/1
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y
