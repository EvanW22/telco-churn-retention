import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.copy()

    remove_cols = [
        "CustomerID", "Count", "Country", "State", "City", "Zip Code",
        "Lat Long", "Latitude", "Longitude",
        "Churn Label", "Churn Score", "CLTV", "Churn Reason"
    ]

    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df.loc[df["Tenure Months"] == 0, "Total Charges"] = 0
    df.drop(columns=remove_cols, inplace=True)

    return df
