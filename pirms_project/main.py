from pipeline import build_pirms
from preprocess import preprocess_data
from feature_engineering import engineer_features
from visualizations import run_eda
from ml_models import train_models

def main():
    print("=== Running PIRMS Data Product ===")

    df = build_pirms()

    df = preprocess_data(df)

    df = engineer_features(df)

    run_eda(df)

    train_models(df)

    print("\n=== PIRMS Pipeline Complete! ===")

if __name__ == "__main__":
    main()
