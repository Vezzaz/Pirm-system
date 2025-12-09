from data_loader import generate_players, generate_weekly_stats, generate_injuries
from preprocess import preprocess_data
from feature_engineering import engineer_features
from config import OUTPUT_DATASET

def build_pirms():
    print("Generating player roster...")
    players = generate_players()

    print("Generating synthetic workload stats...")
    stats = generate_weekly_stats(players)

    print("Generating synthetic injury data...")
    injuries = generate_injuries(players)

    print("Merging datasets...")
    df = stats.merge(injuries, on=["player_id", "season", "week"], how="left")

    print("Preprocessing...")
    df = preprocess_data(df)

    print("Engineering features...")
    df = engineer_features(df)

    print(f"Saving dataset to {OUTPUT_DATASET}...")
    df.to_csv(OUTPUT_DATASET, index=False)

    print("Pipeline complete. Final dataset shape:", df.shape)
    return df
