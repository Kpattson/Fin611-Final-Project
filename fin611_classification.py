"""
Classifying Financial Time Series: What Statistical Features of Return Series
Can Identify Asset Class?

FIN-611: Introduction to Topics in FinTech
Kevin Patterson | New Jersey Institute of Technology | May 2025

This script reproduces all data processing, feature engineering, model training,
and evaluation steps described in the final paper. Results are printed to the
console and saved to a /results folder as CSV files.

Requirements:
    pip install yfinance pandas numpy scikit-learn matplotlib seaborn

Usage:
    python fin611_classification.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ETFS = {
    "SPY": "Equities",
    "QQQ": "Equities",
    "IWM": "Equities",
    "AGG": "Bonds",
    "TLT": "Bonds",
    "GLD": "Commodities",
    "DBC": "Commodities",
}

START_DATE      = "2010-01-01"
END_DATE        = "2024-12-31"
WINDOW          = 252          # rolling window in trading days (~1 year)

TRAIN_END       = "2018-12-31"
VAL_END         = "2021-12-31"
# Test period: 2022-01-01 through END_DATE

RESULTS_DIR     = "results"
RANDOM_STATE    = 42

CLASS_ORDER     = ["Equities", "Bonds", "Commodities"]

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def make_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to: ./{RESULTS_DIR}/\n")


def section(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_data():
    """
    Download daily adjusted closing prices from Yahoo Finance for all seven
    ETFs and compute daily log returns. Returns are aligned to common trading
    dates; rows with any missing value are dropped.
    """
    section("STEP 1: Downloading Price Data")

    tickers = list(ETFS.keys())
    raw = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    # Keep closing prices only
    prices = raw["Close"][tickers].copy()
    prices.dropna(how="all", inplace=True)

    print(f"  ETFs downloaded : {tickers}")
    print(f"  Date range      : {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Trading days    : {len(prices)}")

    # Daily log returns
    returns = np.log(prices / prices.shift(1)).dropna()

    print(f"  Return rows     : {len(returns)}")
    return returns


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def max_drawdown(series):
    """Maximum peak-to-trough cumulative return decline over a return series."""
    cumulative = (1 + series).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()


def compute_features(window_returns):
    """
    Compute eight statistical features over a single rolling window.
    Returns a dict of feature name -> value.
    """
    r = window_returns.dropna()
    if len(r) < 30:
        return None

    mean_ret        = r.mean()
    volatility      = r.std() * np.sqrt(252)          # annualized
    skewness        = r.skew()
    kurtosis        = r.kurtosis()                     # excess kurtosis
    mdd             = max_drawdown(r)
    autocorr        = r.autocorr(lag=1)
    vol_persistence = (r ** 2).autocorr(lag=1)         # GARCH-type clustering
    tail_q05        = r.quantile(0.05)
    tail_q95        = r.quantile(0.95)

    return {
        "mean_return":       mean_ret,
        "volatility":        volatility,
        "skewness":          skewness,
        "kurtosis":          kurtosis,
        "max_drawdown":      mdd,
        "autocorrelation":   autocorr,
        "vol_persistence":   vol_persistence,
        "tail_q05":          tail_q05,
        "tail_q95":          tail_q95,
    }


def build_feature_dataset(returns):
    """
    Roll a WINDOW-day window over each ETF's return series and compute
    features for every valid window. Returns a tidy DataFrame with one
    row per (date, ETF) observation, labeled by asset class.
    """
    section("STEP 2: Feature Engineering")
    print(f"  Rolling window  : {WINDOW} trading days")

    records = []
    for ticker, asset_class in ETFS.items():
        series = returns[ticker].dropna()
        total_windows = len(series) - WINDOW + 1
        print(f"  {ticker:<5} ({asset_class:<12}): {total_windows} windows")

        for i in range(WINDOW, len(series) + 1):
            window_data = series.iloc[i - WINDOW : i]
            date        = series.index[i - 1]
            feats       = compute_features(window_data)
            if feats is None:
                continue
            feats["date"]        = date
            feats["ticker"]      = ticker
            feats["asset_class"] = asset_class
            records.append(feats)

    df = pd.DataFrame(records).set_index("date").sort_index()
    print(f"\n  Total observations: {len(df)}")
    print(f"  Features computed : {[c for c in df.columns if c not in ['ticker','asset_class']]}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN / VALIDATION / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "mean_return", "volatility", "skewness", "kurtosis",
    "max_drawdown", "autocorrelation", "vol_persistence",
    "tail_q05", "tail_q95",
]


def temporal_split(df):
    """
    Strict time-based split — no shuffling.
    Train: 2010-2018 | Validation: 2019-2021 | Test: 2022-2024
    """
    section("STEP 3: Temporal Train / Validation / Test Split")

    train = df[df.index <= TRAIN_END]
    val   = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
    test  = df[df.index > VAL_END]

    for name, subset in [("Train", train), ("Validation", val), ("Test", test)]:
        print(f"  {name:<12}: {subset.index.min().date()} to {subset.index.max().date()}"
              f"  |  {len(subset):>5} rows")

    X_train = train[FEATURE_COLS]
    y_train = train["asset_class"]
    X_val   = val[FEATURE_COLS]
    y_val   = val["asset_class"]
    X_test  = test[FEATURE_COLS]
    y_test  = test["asset_class"]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MODELS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(name, y_true, y_pred, split_label):
    """Print and return accuracy + macro F1 for one split."""
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro")
    print(f"\n  [{name}] {split_label}")
    print(f"    Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"    Macro F1 : {f1:.4f}")
    return acc, f1


def run_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Logistic Regression — interpretable linear baseline.
    Features are standardized; L2 regularization with C=1.0.
    """
    section("STEP 4a: Logistic Regression")

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_va_sc = scaler.transform(X_val)
    X_te_sc = scaler.transform(X_test)

    lr = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    lr.fit(X_tr_sc, y_train)

    # Training performance
    y_train_pred = lr.predict(X_tr_sc)
    evaluate("Logistic Regression", y_train, y_train_pred, "TRAIN")

    # Validation performance
    y_val_pred = lr.predict(X_va_sc)
    evaluate("Logistic Regression", y_val, y_val_pred, "VALIDATION")

    # Test performance
    y_test_pred = lr.predict(X_te_sc)
    evaluate("Logistic Regression", y_test, y_test_pred, "TEST")

    # Classification report — validation
    print("\n  Classification Report (Validation):")
    print(classification_report(y_val, y_val_pred, target_names=CLASS_ORDER, digits=3))

    # Confusion matrix — validation
    cm = confusion_matrix(y_val, y_val_pred, labels=CLASS_ORDER)
    cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
    cm_df.to_csv(f"{RESULTS_DIR}/lr_confusion_matrix_val.csv")
    print("  Confusion matrix (Validation):")
    print(cm_df)

    # Feature coefficients
    coef_df = pd.DataFrame(
        lr.coef_,
        index=lr.classes_,
        columns=FEATURE_COLS,
    ).T
    coef_df.to_csv(f"{RESULTS_DIR}/lr_coefficients.csv")
    print("\n  Coefficients saved to results/lr_coefficients.csv")
    print(coef_df.round(4))

    return lr, scaler, y_val_pred, y_test_pred


def run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Random Forest — nonlinear benchmark.
    Hyperparameters are set conservatively to reduce overfitting.
    """
    section("STEP 4b: Random Forest Classifier")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    evaluate("Random Forest", y_train, y_train_pred, "TRAIN")

    y_val_pred = rf.predict(X_val)
    evaluate("Random Forest", y_val, y_val_pred, "VALIDATION")

    y_test_pred = rf.predict(X_test)
    evaluate("Random Forest", y_test, y_test_pred, "TEST")

    print("\n  Classification Report (Validation):")
    print(classification_report(y_val, y_val_pred, target_names=CLASS_ORDER, digits=3))

    # Feature importance
    imp_df = pd.DataFrame({
        "feature":   FEATURE_COLS,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(f"{RESULTS_DIR}/rf_feature_importance.csv", index=False)
    print("\n  Feature Importances:")
    print(imp_df.to_string(index=False))

    return rf, y_val_pred, y_test_pred


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "Equities":    "#5C7AEA",
    "Bonds":       "#E8A838",
    "Commodities": "#A0624A",
}


def plot_feature_distributions(df):
    """Box plots of each feature by asset class — exploratory analysis."""
    section("STEP 5a: Feature Distribution Plots")

    features_to_plot = ["volatility", "max_drawdown", "kurtosis",
                        "skewness", "vol_persistence", "tail_q05"]
    labels = {
        "volatility":      "Annualized Volatility",
        "max_drawdown":    "Maximum Drawdown",
        "kurtosis":        "Excess Kurtosis",
        "skewness":        "Skewness",
        "vol_persistence": "Volatility Persistence",
        "tail_q05":        "5th Percentile Return",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, feat in zip(axes, features_to_plot):
        data = [df[df["asset_class"] == cls][feat].dropna().values for cls in CLASS_ORDER]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, cls in zip(bp["boxes"], CLASS_ORDER):
            patch.set_facecolor(PALETTE[cls])
            patch.set_alpha(0.75)
        ax.set_xticklabels(CLASS_ORDER, fontsize=10)
        ax.set_title(labels[feat], fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Feature Distributions by Asset Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{RESULTS_DIR}/feature_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, model_name, split_label):
    """Heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrBr",
        xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"{model_name} — Confusion Matrix ({split_label})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    path  = f"{RESULTS_DIR}/{fname}_confusion_{split_label.lower()}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(rf):
    """Horizontal bar chart of random forest feature importances."""
    imp_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": rf.feature_importances_,
    }).sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#A0624A" if imp > imp_df["importance"].median() else "#D6C4A8"
              for imp in imp_df["importance"]]
    ax.barh(imp_df["feature"], imp_df["importance"], color=colors)
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
    ax.set_title("Random Forest — Feature Importance", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = f"{RESULTS_DIR}/rf_feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_lr_coefficients(lr):
    """Grouped bar chart of logistic regression coefficients per class."""
    coef_df = pd.DataFrame(
        lr.coef_,
        index=lr.classes_,
        columns=FEATURE_COLS,
    )

    x     = np.arange(len(FEATURE_COLS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, cls in enumerate(CLASS_ORDER):
        ax.bar(x + i * width, coef_df.loc[cls], width,
               label=cls, color=PALETTE[cls], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(FEATURE_COLS, rotation=30, ha="right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Coefficient", fontsize=11)
    ax.set_title("Logistic Regression Coefficients by Asset Class", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = f"{RESULTS_DIR}/lr_coefficients.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_accuracy_comparison(results_dict):
    """
    Bar chart comparing accuracy and macro F1 across models and splits.
    results_dict: { "Model (Split)": {"accuracy": x, "f1": y} }
    """
    labels  = list(results_dict.keys())
    acc     = [v["accuracy"] for v in results_dict.values()]
    f1      = [v["f1"] for v in results_dict.values()]
    x       = np.arange(len(labels))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, acc, width, label="Accuracy", color="#5C7AEA", alpha=0.85)
    ax.bar(x + width / 2, f1,  width, label="Macro F1",  color="#A0624A", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Score")
    ax.set_title("Model Performance: Accuracy and Macro F1", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = f"{RESULTS_DIR}/model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — ROBUSTNESS: WINDOW SIZE SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def window_sensitivity(returns):
    """
    Re-run logistic regression for window sizes 63, 126, 189, 252, 504 days.
    Reports validation accuracy and macro F1 for each.
    """
    section("STEP 6: Window Size Sensitivity Analysis")

    windows = [63, 126, 189, 252, 504]
    rows    = []

    for w in windows:
        records = []
        for ticker, asset_class in ETFS.items():
            series = returns[ticker].dropna()
            for i in range(w, len(series) + 1):
                window_data = series.iloc[i - w : i]
                date        = series.index[i - 1]
                feats       = compute_features(window_data)
                if feats is None:
                    continue
                feats["date"]        = date
                feats["asset_class"] = asset_class
                records.append(feats)

        if not records:
            continue

        df_w = pd.DataFrame(records).set_index("date").sort_index()

        train_w = df_w[df_w.index <= TRAIN_END]
        val_w   = df_w[(df_w.index > TRAIN_END) & (df_w.index <= VAL_END)]

        if len(train_w) < 50 or len(val_w) < 10:
            continue

        X_tr = train_w[FEATURE_COLS]
        y_tr = train_w["asset_class"]
        X_va = val_w[FEATURE_COLS]
        y_va = val_w["asset_class"]

        sc   = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_va_sc = sc.transform(X_va)

        lr = LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            C=1.0, max_iter=1000, random_state=RANDOM_STATE
        )
        lr.fit(X_tr_sc, y_tr)
        y_pred = lr.predict(X_va_sc)

        acc = accuracy_score(y_va, y_pred)
        f1  = f1_score(y_va, y_pred, average="macro")
        rows.append({"window_days": w, "val_accuracy": acc, "val_macro_f1": f1})
        print(f"  Window {w:>4}d  |  Val Accuracy: {acc:.4f}  |  Val Macro F1: {f1:.4f}")

    sens_df = pd.DataFrame(rows)
    sens_df.to_csv(f"{RESULTS_DIR}/window_sensitivity.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sens_df["window_days"], sens_df["val_accuracy"], marker="o",
            label="Accuracy", color="#5C7AEA")
    ax.plot(sens_df["window_days"], sens_df["val_macro_f1"], marker="s",
            label="Macro F1",  color="#A0624A")
    ax.set_xlabel("Rolling Window (trading days)", fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("Validation Performance vs. Window Size", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = f"{RESULTS_DIR}/window_sensitivity.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    return sens_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    make_results_dir()

    # 1. Data
    returns = download_data()

    # 2. Features
    df = build_feature_dataset(returns)
    df.to_csv(f"{RESULTS_DIR}/feature_dataset.csv")
    print(f"\n  Feature dataset saved to results/feature_dataset.csv")

    # 3. Split
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(df)

    # 4a. Logistic Regression
    lr_model, scaler, lr_val_pred, lr_test_pred = run_logistic_regression(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # 4b. Random Forest
    rf_model, rf_val_pred, rf_test_pred = run_random_forest(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # 5. Visualisations
    section("STEP 5: Generating Plots")
    plot_feature_distributions(df)
    plot_confusion_matrix(y_val,  lr_val_pred,  "Logistic Regression", "Validation")
    plot_confusion_matrix(y_test, lr_test_pred, "Logistic Regression", "Test")
    plot_confusion_matrix(y_val,  rf_val_pred,  "Random Forest",       "Validation")
    plot_confusion_matrix(y_test, rf_test_pred, "Random Forest",       "Test")
    plot_feature_importance(rf_model)
    plot_lr_coefficients(lr_model)

    # Summary comparison chart
    results_dict = {
        "LR — Train":  {"accuracy": accuracy_score(y_train, lr_model.predict(scaler.transform(X_train))),
                        "f1":       f1_score(y_train, lr_model.predict(scaler.transform(X_train)), average="macro")},
        "LR — Val":    {"accuracy": accuracy_score(y_val,  lr_val_pred),
                        "f1":       f1_score(y_val,  lr_val_pred, average="macro")},
        "LR — Test":   {"accuracy": accuracy_score(y_test, lr_test_pred),
                        "f1":       f1_score(y_test, lr_test_pred, average="macro")},
        "RF — Train":  {"accuracy": accuracy_score(y_train, rf_model.predict(X_train)),
                        "f1":       f1_score(y_train, rf_model.predict(X_train), average="macro")},
        "RF — Val":    {"accuracy": accuracy_score(y_val,  rf_val_pred),
                        "f1":       f1_score(y_val,  rf_val_pred, average="macro")},
        "RF — Test":   {"accuracy": accuracy_score(y_test, rf_test_pred),
                        "f1":       f1_score(y_test, rf_test_pred, average="macro")},
    }
    plot_accuracy_comparison(results_dict)

    # 6. Robustness
    window_sensitivity(returns)

    section("DONE")
    print(f"  All outputs saved to ./{RESULTS_DIR}/")
    print("""
  Output files:
    feature_dataset.csv           - Full feature matrix with labels
    lr_confusion_matrix_val.csv   - LR confusion matrix (validation)
    lr_coefficients.csv           - LR coefficients by class
    rf_feature_importance.csv     - RF feature importances
    window_sensitivity.csv        - Robustness results by window size

    feature_distributions.png     - Box plots by asset class
    lr_confusion_*.png            - LR confusion matrix heatmaps
    rf_confusion_*.png            - RF confusion matrix heatmaps
    lr_coefficients.png           - LR coefficient bar chart
    rf_feature_importance.png     - RF importance bar chart
    model_comparison.png          - Accuracy + F1 across models/splits
    window_sensitivity.png        - Performance vs. window size
""")


if __name__ == "__main__":
    main()
