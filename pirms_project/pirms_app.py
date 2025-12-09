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
# RUN PIPELINE BUTTON
# ============================================
if st.button("Run Full Pipeline (Generate Data → EDA → Model Training)"):
    st.write("🔄 Running synthetic data pipeline...")

    df = build_pirms()
    st.success("Dataset generated!")

    st.write(df.head())

    st.write("🔄 Preprocessing...")
    df = preprocess_data(df)

    st.write("🔄 Feature engineering...")
    df = engineer_features(df)

    st.write("📊 Running EDA...")
    run_eda(df)
    st.success("EDA Complete! Plots saved to /plots")

    st.write("🤖 Training ML models...")
    train_models(df)
    st.success("Models trained! Results saved to /models")

    st.write("🎉 Pipeline Finished!")


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
