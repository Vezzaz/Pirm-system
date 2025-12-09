import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(df, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    print("Running EDA and saving plots...")

    plot_missing(df, save_dir)
    plot_histograms(df, save_dir)
    plot_correlation(df, save_dir)
    plot_injury_by_position(df, save_dir)
    plot_workload_vs_injury(df, save_dir)

    print("EDA complete. Plots saved to /plots/")


def plot_missing(df, save_dir):
    plt.figure(figsize=(10, 4))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Value Heatmap")
    plt.savefig(os.path.join(save_dir, "missing_heatmap.png"))
    plt.close()


def plot_histograms(df, save_dir):
    df.hist(figsize=(12, 10))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "histograms.png"))
    plt.close()


def plot_correlation(df, save_dir):
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    plt.figure(figsize=(10, 7))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_heatmap.png"))
    plt.close()


def plot_injury_by_position(df, save_dir):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="position", hue="injured")
    plt.title("Injuries by Position")
    plt.savefig(os.path.join(save_dir, "injury_by_position.png"))
    plt.close()


def plot_workload_vs_injury(df, save_dir):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="injured", y="workload")
    plt.title("Workload vs Injury")
    plt.savefig(os.path.join(save_dir, "workload_vs_injury.png"))
    plt.close()
