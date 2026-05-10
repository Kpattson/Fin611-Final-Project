import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

ETFS = {
    "SPY": "Equities",
    "QQQ": "Equities",
    "IWM": "Equities",
    "AGG": "Bonds",
    "TLT": "Bonds",
    "GLD": "Commodities",
    "DBC": "Commodities",
}

START_DATE   = "2010-01-01"
END_DATE     = "2024-12-31"
WINDOW       = 252
TRAIN_END    = "2018-12-31"
VAL_END      = "2021-12-31"
RESULTS_DIR  = "results"
RANDOM_STATE = 42
CLASS_ORDER  = ["Equities", "Bonds", "Commodities"]

FEATURE_COLS = [
    "mean_return", "volatility", "skewness", "kurtosis",
    "max_drawdown", "autocorrelation", "vol_persistence",
    "tail_q05", "tail_q95",
]

os.makedirs(RESULTS_DIR, exist_ok=True)

raw     = yf.download(list(ETFS.keys()), start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
prices  = raw["Close"][list(ETFS.keys())].dropna(how="all")
returns = np.log(prices / prices.shift(1)).dropna()


def max_drawdown(series):
    cumulative  = (1 + series).cumprod()
    rolling_max = cumulative.cummax()
    return ((cumulative - rolling_max) / rolling_max).min()


def compute_features(window_data):
    r = window_data.dropna()
    if len(r) < 30:
        return None
    return {
        "mean_return":     r.mean(),
        "volatility":      r.std() * np.sqrt(252),
        "skewness":        r.skew(),
        "kurtosis":        r.kurtosis(),
        "max_drawdown":    max_drawdown(r),
        "autocorrelation": r.autocorr(lag=1),
        "vol_persistence": (r**2).autocorr(lag=1),
        "tail_q05":        r.quantile(0.05),
        "tail_q95":        r.quantile(0.95),
    }


records = []
for ticker, asset_class in ETFS.items():
    series = returns[ticker].dropna()
    for i in range(WINDOW, len(series) + 1):
        feats = compute_features(series.iloc[i - WINDOW: i])
        if feats is None:
            continue
        feats["date"]        = series.index[i - 1]
        feats["ticker"]      = ticker
        feats["asset_class"] = asset_class
        records.append(feats)

df = pd.DataFrame(records).set_index("date").sort_index()
df.to_csv(f"{RESULTS_DIR}/feature_dataset.csv")

train = df[df.index <= TRAIN_END]
val   = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
test  = df[df.index > VAL_END]

X_train, y_train = train[FEATURE_COLS], train["asset_class"]
X_val,   y_val   = val[FEATURE_COLS],   val["asset_class"]
X_test,  y_test  = test[FEATURE_COLS],  test["asset_class"]

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_va_sc = scaler.transform(X_val)
X_te_sc = scaler.transform(X_test)

lr = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                        C=1.0, max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_tr_sc, y_train)

results_rows = []
for split_name, X_sc, y_true in [
    ("Train",      X_tr_sc, y_train),
    ("Validation", X_va_sc, y_val),
    ("Test",       X_te_sc, y_test),
]:
    y_pred = lr.predict(X_sc)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    results_rows.append({"Model": "Logistic Regression", "Split": split_name,
                         "Accuracy": round(acc, 4), "Macro_F1": round(f1, 4)})

lr_val_pred  = lr.predict(X_va_sc)
lr_test_pred = lr.predict(X_te_sc)

print(classification_report(y_val, lr_val_pred, target_names=CLASS_ORDER))

cm_lr = confusion_matrix(y_val, lr_val_pred, labels=CLASS_ORDER)
pd.DataFrame(cm_lr, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
    f"{RESULTS_DIR}/lr_confusion_matrix.csv")

coef_df = pd.DataFrame(lr.coef_, index=lr.classes_, columns=FEATURE_COLS).T
coef_df.to_csv(f"{RESULTS_DIR}/lr_coefficients.csv")

rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                            min_samples_leaf=20, max_features="sqrt",
                            class_weight="balanced",
                            random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)

for split_name, X_in, y_true in [
    ("Train",      X_train, y_train),
    ("Validation", X_val,   y_val),
    ("Test",       X_test,  y_test),
]:
    y_pred = rf.predict(X_in)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    results_rows.append({"Model": "Random Forest", "Split": split_name,
                         "Accuracy": round(acc, 4), "Macro_F1": round(f1, 4)})

rf_val_pred  = rf.predict(X_val)
rf_test_pred = rf.predict(X_test)

print(classification_report(y_val, rf_val_pred, target_names=CLASS_ORDER))

imp_df = pd.DataFrame({"Feature": FEATURE_COLS,
                        "Importance": rf.feature_importances_}
                      ).sort_values("Importance", ascending=False)
imp_df.to_csv(f"{RESULTS_DIR}/rf_feature_importance.csv", index=False)

cm_rf = confusion_matrix(y_val, rf_val_pred, labels=CLASS_ORDER)
pd.DataFrame(cm_rf, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
    f"{RESULTS_DIR}/rf_confusion_matrix.csv")

results_df = pd.DataFrame(results_rows)
results_df.to_csv(f"{RESULTS_DIR}/model_performance_summary.csv", index=False)

sensitivity_rows = []
for w in [63, 126, 189, 252, 504]:
    recs = []
    for ticker, asset_class in ETFS.items():
        series = returns[ticker].dropna()
        for i in range(w, len(series) + 1):
            feats = compute_features(series.iloc[i - w: i])
            if feats is None:
                continue
            feats["date"]        = series.index[i - 1]
            feats["asset_class"] = asset_class
            recs.append(feats)
    if not recs:
        continue
    df_w    = pd.DataFrame(recs).set_index("date").sort_index()
    train_w = df_w[df_w.index <= TRAIN_END]
    val_w   = df_w[(df_w.index > TRAIN_END) & (df_w.index <= VAL_END)]
    if len(train_w) < 50 or len(val_w) < 10:
        continue
    sc_w = StandardScaler()
    lr_w = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                               C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    lr_w.fit(sc_w.fit_transform(train_w[FEATURE_COLS]), train_w["asset_class"])
    y_pred_w = lr_w.predict(sc_w.transform(val_w[FEATURE_COLS]))
    acc = accuracy_score(val_w["asset_class"], y_pred_w)
    f1  = f1_score(val_w["asset_class"], y_pred_w, average="macro")
    sensitivity_rows.append({"Window_Days": w, "Val_Accuracy": round(acc, 4),
                              "Val_Macro_F1": round(f1, 4)})

sens_df = pd.DataFrame(sensitivity_rows)
sens_df.to_csv(f"{RESULTS_DIR}/window_sensitivity.csv", index=False)

PALETTE = {"Equities": "#5C7AEA", "Bonds": "#E8A838", "Commodities": "#A0624A"}

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
plot_features = ["volatility", "max_drawdown", "kurtosis",
                 "skewness", "vol_persistence", "tail_q05"]
feat_labels = {
    "volatility":      "Annualized Volatility",
    "max_drawdown":    "Maximum Drawdown",
    "kurtosis":        "Excess Kurtosis",
    "skewness":        "Skewness",
    "vol_persistence": "Volatility Persistence",
    "tail_q05":        "5th Percentile Return",
}
for ax, feat in zip(axes, plot_features):
    data = [df[df["asset_class"] == c][feat].dropna().values for c in CLASS_ORDER]
    bp   = ax.boxplot(data, patch_artist=True,
                      medianprops=dict(color="black", linewidth=2))
    for patch, cls in zip(bp["boxes"], CLASS_ORDER):
        patch.set_facecolor(PALETTE[cls])
        patch.set_alpha(0.75)
    ax.set_xticklabels(CLASS_ORDER, fontsize=9)
    ax.set_title(feat_labels[feat], fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.suptitle("Feature Distributions by Asset Class", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/plot_1_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="YlOrBr",
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
            linewidths=0.5, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Logistic Regression - Confusion Matrix (Validation)", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/plot_2_lr_confusion_matrix.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="YlOrBr",
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
            linewidths=0.5, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Random Forest - Confusion Matrix (Validation)", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/plot_3_rf_confusion_matrix.png", dpi=150)
plt.close()

imp_sorted = imp_df.sort_values("Importance")
fig, ax    = plt.subplots(figsize=(8, 5))
colors     = ["#A0624A" if v > imp_sorted["Importance"].median()
              else "#D6C4A8" for v in imp_sorted["Importance"]]
ax.barh(imp_sorted["Feature"], imp_sorted["Importance"], color=colors)
ax.set_xlabel("Mean Decrease in Impurity")
ax.set_title("Random Forest - Feature Importance", fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/plot_4_rf_feature_importance.png", dpi=150)
plt.close()

coef_plot = pd.DataFrame(lr.coef_, index=lr.classes_, columns=FEATURE_COLS)
x, width  = np.arange(len(FEATURE_COLS)), 0.25
fig, ax   = plt.subplots(figsize=(13, 5))
for i, cls in enumerate(CLASS_ORDER):
    ax.bar(x + i * width, coef_plot.loc[cls], width,
           label=cls, color=PALETTE[cls], alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(FEATURE_COLS, rotation=30, ha="right", fontsize=9)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Coefficient")
ax.set_title("Logistic Regression Coefficients by Asset Class", fontweight="bold")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/plot_5_lr_coefficients.png", dpi=150)
plt.close()

summary  = results_df[results_df["Split"] == "Validation"].copy()
x, width = np.arange(len(summary)), 0.35
fig, ax  = plt.subplots(figsize=(7, 4))
ax.bar(x - width/2, summary["Accuracy"],  width, label="Accuracy",  color="#5C7AEA", alpha=0.85)
ax.bar(x + width/2, summary["Macro_F1"], width, label="Macro F1", color="#A0624A", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(summary["Model"])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Validation Performance: LR vs Random Forest", fontweight="bold")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/plot_6_model_comparison.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sens_df["Window_Days"], sens_df["Val_Accuracy"],
        marker="o", label="Val Accuracy", color="#5C7AEA")
ax.plot(sens_df["Window_Days"], sens_df["Val_Macro_F1"],
        marker="s", label="Val Macro F1", color="#A0624A")
ax.set_xlabel("Rolling Window (trading days)")
ax.set_ylabel("Score")
ax.set_title("Validation Performance vs Window Size", fontweight="bold")
ax.legend()
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/plot_7_window_sensitivity.png", dpi=150)
plt.close()
