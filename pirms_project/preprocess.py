import pandas as pd

def preprocess_data(df):
    # Currently no major cleaning needed
    df = df.copy()

    # Ensure correct types
    df["injured"] = df["injured"].astype(int)

    return df
