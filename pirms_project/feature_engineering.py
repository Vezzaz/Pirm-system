import pandas as pd

def engineer_features(df):
    df = df.sort_values(["player_id", "season", "week"])

    # -----------------------
    # BASIC FEATURES
    # -----------------------
    df["touches"] = df["carries"] + df["targets"]
    df["workload"] = df["carries"] + df["targets"] + df["routes"]

    df["rolling_touches"] = (
        df.groupby("player_id")["touches"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["delta_touches"] = df.groupby("player_id")["touches"].diff().fillna(0)

    df["rolling_routes"] = (
        df.groupby("player_id")["routes"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # -----------------------
    # ML FEATURES
    # -----------------------

    df["games_played_recent"] = (
        df.groupby("player_id")["injured"]
        .apply(lambda x: (1 - x).rolling(3, min_periods=1).sum())
        .reset_index(level=0, drop=True)
    )

    df["games_missed_recent"] = (
        df.groupby("player_id")["injured"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["injury_history_score"] = (
        df.groupby("player_id")["injured"].cumsum()
    )

    return df
