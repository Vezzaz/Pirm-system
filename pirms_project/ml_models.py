import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

def prepare_features(df):
    features = [
        "season", "week", "workload", "games_played_recent",
        "games_missed_recent", "injury_history_score"
    ]
    X = df[features]
    y = df["injured"]
    return X, y


def train_models(df, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    y_proba_lr = logreg.predict_proba(X_test)[:, 1]

    lr_report = evaluate_model("Logistic Regression", y_test, y_pred_lr, y_proba_lr)

    with open(os.path.join(save_dir, "logreg_report.txt"), "w") as f:
        f.write(lr_report)

    plot_roc(y_test, y_proba_lr, "logreg_roc.png", save_dir)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    rf_report = evaluate_model("Random Forest", y_test, y_pred_rf, y_proba_rf)

    with open(os.path.join(save_dir, "rf_report.txt"), "w") as f:
        f.write(rf_report)

    plot_roc(y_test, y_proba_rf, "rf_roc.png", save_dir)
    plot_feature_importance(rf, X.columns, "rf_feature_importance.png", save_dir)

    print("Modeling complete. Reports and plots saved to /models/")
    return logreg, rf


def evaluate_model(name, y_true, y_pred, y_proba):
    report = (
        f"===== {name} =====\n"
        f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n"
        f"Precision: {precision_score(y_true, y_pred):.4f}\n"
        f"Recall: {recall_score(y_true, y_pred):.4f}\n"
        f"F1 Score: {f1_score(y_true, y_pred):.4f}\n"
        f"ROC-AUC: {roc_auc_score(y_true, y_proba):.4f}\n\n"
        f"Classification Report:\n{classification_report(y_true, y_pred)}\n"
    )
    print(report)
    return report


def plot_roc(y_true, y_proba, filename, save_dir):
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def plot_feature_importance(model, feature_names, filename, save_dir):
    importance = model.feature_importances_
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importance, y=feature_names)
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
