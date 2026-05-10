# Classifying Financial Time Series

**FIN-611: Introduction to Topics in FinTech**  
Kevin Patterson | New Jersey Institute of Technology | May 2025

---

## Research Question

Can statistical features of financial return time series — computed with no external labels or market context — accurately identify whether an asset belongs to the equity, bond, or commodity class?

---

## What This Code Does

This script reproduces all steps from the final paper:

1. Downloads daily ETF price data from Yahoo Finance (2010–2024)
2. Computes daily log returns and aligns across all seven instruments
3. Engineers eight rolling statistical features per asset per 252-day window
4. Trains a logistic regression (interpretable baseline) and random forest (nonlinear benchmark)
5. Evaluates on a strict time-based split — no shuffling, no look-ahead
6. Generates all plots and results tables used in the paper
7. Runs a window-size sensitivity analysis as a robustness check

---

## ETFs

| Asset Class | Tickers |
|-------------|---------|
| Equities | SPY, QQQ, IWM |
| Bonds | AGG, TLT |
| Commodities | GLD, DBC |

---

## Features

| Feature | Description |
|---------|-------------|
| Mean Return | Average daily log return over the window |
| Volatility | Annualized standard deviation |
| Skewness | Asymmetry of the return distribution |
| Kurtosis | Tail heaviness (excess kurtosis) |
| Max Drawdown | Largest peak-to-trough decline |
| Autocorrelation | First-order autocorrelation of returns |
| Vol. Persistence | Autocorrelation of squared returns (GARCH proxy) |
| Tail Quantiles | 5th and 95th percentile returns |

---

## Data Split

| Split | Period |
|-------|--------|
| Train | 2010–2018 |
| Validation | 2019–2021 |
| Test | 2022–2024 |

No random shuffling is applied at any stage.

---

## Installation

```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn
```

Requires Python 3.8 or later.

---

## Usage

```bash
python fin611_classification.py
```

All output files are saved to a `/results` folder that is created automatically.

---

## Output Files

**CSVs**
- `feature_dataset.csv` — full feature matrix with labels
- `lr_confusion_matrix_val.csv` — logistic regression confusion matrix (validation)
- `lr_coefficients.csv` — logistic regression coefficients by class
- `rf_feature_importance.csv` — random forest feature importances
- `window_sensitivity.csv` — robustness results across window sizes

**Plots**
- `feature_distributions.png` — box plots of each feature by asset class
- `lr_confusion_validation.png` / `lr_confusion_test.png` — LR confusion matrix heatmaps
- `rf_confusion_validation.png` / `rf_confusion_test.png` — RF confusion matrix heatmaps
- `lr_coefficients.png` — coefficient bar chart by class
- `rf_feature_importance.png` — feature importance bar chart
- `model_comparison.png` — accuracy and macro F1 across models and splits
- `window_sensitivity.png` — validation performance vs. rolling window size

---

## Key Results

| Model | Split | Accuracy | Macro F1 |
|-------|-------|----------|----------|
| Logistic Regression | Validation | ~81% | ~0.79 |
| Random Forest | Validation | TBD (tuning) | TBD |

Bonds are the most cleanly separated class. The most challenging boundary is equities vs. commodities, which exhibit overlapping statistical behavior during market stress periods.

---

## References

- Cont, R. (2001). Empirical properties of asset returns. *Quantitative Finance*, 1(2), 223–236.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Lubba, C. H., et al. (2019). catch22: CAnonical Time-series CHaracteristics. *Data Mining and Knowledge Discovery*, 33(6), 1821–1852.
- Tsay, R. S. (2005). *Analysis of Financial Time Series* (2nd ed.). Wiley.
