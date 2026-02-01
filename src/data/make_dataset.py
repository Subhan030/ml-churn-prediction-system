import pandas as pd

def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df
