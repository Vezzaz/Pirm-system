import streamlit as st
import pandas as pd
import os

from pipeline import build_pirms
from preprocess import preprocess_data
from feature_engineering import engineer_features
from visualizations import run_eda
from ml_models import train_models

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="PIRMS Injury Risk System",
    layout="wide"
)

st.title("🏈 Player Injury Risk Monitoring System (PIRMS)")
st.write("Synthetic-data machine learning system for NFL-style injury analysis.")

# ============================================
# OPTIONAL: USER-UPLOADED NFL DATA
# ============================================
st.header("📤 Upload Your Own NFL Data (Optional)")

st.write("""
If you have real NFL datasets (e.g., weekly stats, injury logs, player rosters),
upload them here and PIRMS will attempt to use them instead of synthetic data.
Supported formats: **CSV files**.
""")

uploaded_stats = st.file_uploader("Upload Weekly Player Stats CSV", type=["csv"])
uploaded_injuries = st.file_uploader("Upload Injury Log CSV", type=["csv"])

use_real_data = False
user_stats_df = None
user_injury_df = None

if uploaded_stats is not None:
    try:
        user_stats_df = pd.read_csv(uploaded_stats)
        st.success("Player stats file uploaded successfully!")
        st.write(user_stats_df.head())
        use_real_data = True
    except Exception as e:
        st.error(f"Failed to load stats file: {e}")

if uploaded_injuries is not None:
    try:
        user_injury_df = pd.read_csv(uploaded_injuries)
        st.success("Injury log file uploaded successfully!")
        st.write(user_injury_df.head())
        use_real_data = True
    except Exception as e:
        st.error(f"Failed to load injury file: {e}")


# ============================================
# RUN PIPELINE BUTTON (Supports Real or Synthetic Data)
# ============================================
st.header("🏈 Run PIRMS Pipeline")

if st.button("Run Pipeline"):
    st.write("🔄 Starting pipeline...")

    # ------------------------------
    # OPTION A: Use user-uploaded data
    # ------------------------------
    if use_real_data and (user_stats_df is not None) and (user_injury_df is not None):
        st.write("📡 Using uploaded NFL datasets...")

        from preprocess import merge_data
        from feature_engineering import engineer_features

        try:
            # Merge the real data
            df = merge_data(user_stats_df, user_injury_df)

            # Preprocess
            df = preprocess_data(df)

            # Feature engineering
            df = engineer_features(df)

            st.success("Real data processed successfully!")
            st.write(df.head())

        except Exception as e:
            st.error(f"Error processing uploaded data: {e}")
            st.stop()

    else:
        # ------------------------------
        # OPTION B: Use synthetic data
        # ------------------------------
        st.write("🧪 No real data detected — generating synthetic dataset...")
        df = build_pirms()
        df = preprocess_data(df)
        df = engineer_features(df)

    # ------------------------------
    # Continue with EDA and Modeling
    # ------------------------------
    st.write("📊 Running EDA...")
    run_eda(df)

    st.write("🤖 Training ML Models...")
    train_models(df)

    st.success("🎉 Pipeline finished successfully!")


# ============================================
# SECTION: VIEW GENERATED PLOTS
# ============================================
st.header("📈 View EDA Plots")

plot_list = [
    "missing_heatmap.png",
    "histograms.png",
    "correlation_heatmap.png",
    "injury_by_position.png",
    "workload_vs_injury.png"
]

for p in plot_list:
    path = os.path.join("plots", p)
    if os.path.exists(path):
        st.subheader(p.replace(".png", "").replace("_", " ").title())
        st.image(path)
    else:
        st.write(f"Plot not found: {p}")


# ============================================
# SECTION: VIEW MODEL OUTPUT
# ============================================
st.header("🤖 View Model Reports")

if os.path.exists("models/logreg_report.txt"):
    with open("models/logreg_report.txt") as f:
        st.subheader("Logistic Regression Report")
        st.text(f.read())

if os.path.exists("models/rf_report.txt"):
    with open("models/rf_report.txt") as f:
        st.subheader("Random Forest Report")
        st.text(f.read())

# ROC Curves and Feature Importance
st.header("📊 ML Visualizations")

model_images = [
    "logreg_roc.png",
    "rf_roc.png",
    "rf_feature_importance.png"
]

for img in model_images:
    path = os.path.join("models", img)
    if os.path.exists(path):
        st.subheader(img.replace(".png", "").replace("_", " ").title())
        st.image(path)
