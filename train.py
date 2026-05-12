"""
Phase 4: Model Training & Comparison
======================================
Train and compare 5 models: Linear Regression, Ridge, Lasso,
Elastic Net, and XGBoost. Includes cross-validation, residual
analysis, and learning curves.

Model progression (FAANG interview answer for "walk me through your approach"):
  1. Linear Regression → baseline, fully interpretable
  2. Ridge (L2)        → handles correlated features (VIF > 5)
  3. Lasso (L1)        → feature selection built in (coeff → 0)
  4. Elastic Net       → best of Ridge + Lasso, good default
  5. XGBoost           → non-linear benchmark, production target

Evaluation beyond R² and MSE:
  - RMSPE (Root Mean Squared Percentage Error): Rossmann competition metric
  - Cross-validation with TimeSeriesSplit (not KFold!)
  - Residual plots: detect heteroscedasticity
  - Learning curves: diagnose overfitting vs underfitting
  - Feature importance: model interpretability

FAANG note on RMSPE:
  RMSPE = sqrt(mean((y_true - y_pred)^2 / y_true^2))
  Used instead of RMSE because a $1000 error on a $2000 store
  is worse than the same error on a $10,000 store.
  Always use business-aligned metrics, not just statistical ones.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline as SKPipeline
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR  = ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Metrics ───────────────────────────────────────────────────────────────────

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Percentage Error.
    Rossmann competition primary metric. Business-friendly.
    Penalises proportional errors — a 10% miss on a small store
    counts as much as a 10% miss on a large store.
    """
    mask = y_true != 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> dict:
    """Compute full evaluation metrics."""
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    rmspe_val = rmspe(y_true, y_pred)
    mape  = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)

    return {
        "model":  name,
        "RMSE":   round(rmse, 2),
        "MAE":    round(mae, 2),
        "R2":     round(r2, 4),
        "RMSPE":  round(rmspe_val, 4),
        "MAPE%":  round(mape, 2),
    }


# ── Model Definitions ─────────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Return dict of model name → sklearn Pipeline (scaler + model).

    Why wrap in Pipeline?
      Scaler must be fit only on train folds during cross-validation.
      Wrapping in Pipeline ensures this automatically.

    XGBoost doesn't need scaling but we include it for consistency.
    """
    return {
        "LinearRegression": SKPipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
        "Ridge (L2)": SKPipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=10.0)),
        ]),
        "Lasso (L1)": SKPipeline([
            ("scaler", StandardScaler()),
            ("model",  Lasso(alpha=1.0, max_iter=5000)),
        ]),
        "ElasticNet": SKPipeline([
            ("scaler", StandardScaler()),
            ("model",  ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000)),
        ]),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        ),
    }


# ── Cross-Validation ──────────────────────────────────────────────────────────

def time_series_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """
    Cross-validate using TimeSeriesSplit — NOT KFold.

    FAANG interview note:
      KFold randomly shuffles folds → test set can precede train set in time.
      TimeSeriesSplit always trains on past, validates on future.
      For time-series data, KFold gives optimistically biased CV scores.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

    return {
        "cv_rmse_mean":  round(float(np.mean(rmse_scores)), 2),
        "cv_rmse_std":   round(float(np.std(rmse_scores)), 2),
        "cv_rmse_folds": [round(r, 2) for r in rmse_scores],
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list = None,
) -> pd.DataFrame:
    """
    Train all models, evaluate, and return comparison DataFrame.
    """
    models = get_models()
    results = []

    for name, model in models.items():
        print(f"[train] {name}...", end=" ", flush=True)

        # Cross-validation
        cv_result = time_series_cv(model, X_train, y_train)

        # Final fit on full train set
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        # Metrics
        train_metrics = evaluate(y_train, y_pred_train, name)
        test_metrics  = evaluate(y_test,  y_pred_test,  name)

        result = {
            "model":           name,
            "train_R2":        train_metrics["R2"],
            "test_R2":         test_metrics["R2"],
            "test_RMSE":       test_metrics["RMSE"],
            "test_MAE":        test_metrics["MAE"],
            "test_RMSPE":      test_metrics["RMSPE"],
            "test_MAPE%":      test_metrics["MAPE%"],
            "cv_RMSE_mean":    cv_result["cv_rmse_mean"],
            "cv_RMSE_std":     cv_result["cv_rmse_std"],
            "overfit_gap":     round(train_metrics["R2"] - test_metrics["R2"], 4),
        }
        results.append(result)

        # Save model
        joblib.dump(model, MODEL_DIR / f"{name.replace(' ', '_')}.pkl")
        print(f"R²={test_metrics['R2']:.4f}  RMSPE={test_metrics['RMSPE']:.4f}")

    df_results = pd.DataFrame(results).sort_values("test_RMSPE")
    return df_results


# ── Residual Analysis ─────────────────────────────────────────────────────────

def plot_residuals(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str = "Model") -> None:
    """
    Four residual plots for diagnosing model quality.

    1. Residuals vs Fitted: detect heteroscedasticity (fan shape = bad)
    2. Q-Q plot: check if residuals are normally distributed
    3. Residuals histogram: distribution of errors
    4. Actual vs Predicted: overall fit quality

    FAANG interview note:
      "What is heteroscedasticity and why does it matter for linear regression?"
      → It means the variance of errors is not constant across predictions.
        LR assumes constant variance (homoscedasticity).
        If fan-shaped residuals appear, log-transform the target or use WLS.
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=14, fontweight="bold")

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.3, s=10, color="#378ADD")
    axes[0, 0].axhline(0, color="#D85A30", linewidth=2, linestyle="--")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Fitted\n(fan shape = heteroscedasticity)")

    # 2. Q-Q Plot
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    axes[0, 1].scatter(osm, osr, alpha=0.3, s=8, color="#7F77DD")
    axes[0, 1].plot(osm, slope * np.array(osm) + intercept, color="#D85A30", linewidth=2)
    axes[0, 1].set_xlabel("Theoretical Quantiles")
    axes[0, 1].set_ylabel("Sample Quantiles")
    axes[0, 1].set_title(f"Q-Q Plot (R={r:.3f})\n(points on line = normal residuals)")

    # 3. Residual histogram
    axes[1, 0].hist(residuals, bins=50, color="#1D9E75", edgecolor="white", alpha=0.8)
    axes[1, 0].axvline(0, color="#D85A30", linewidth=2, linestyle="--")
    axes[1, 0].set_xlabel("Residual")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Residual Distribution\n(should be centred at 0)")

    # 4. Actual vs Predicted
    axes[1, 1].scatter(y_test, y_pred, alpha=0.3, s=8, color="#BA7517")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[1, 1].plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")
    axes[1, 1].set_xlabel("Actual Sales")
    axes[1, 1].set_ylabel("Predicted Sales")
    r2 = r2_score(y_test, y_pred)
    axes[1, 1].set_title(f"Actual vs Predicted (R²={r2:.4f})")
    axes[1, 1].legend()

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(OUTPUT_DIR / f"residuals_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[residuals] Saved for {model_name}")


# ── Learning Curves ───────────────────────────────────────────────────────────

def plot_learning_curve(model, X: np.ndarray, y: np.ndarray,
                        model_name: str = "Model") -> None:
    """
    Plot learning curve to diagnose bias vs variance.

    High train score, low val score → overfitting (high variance)
    Both scores low → underfitting (high bias)
    Both scores converge → good fit

    FAANG interview note:
      "How do you know if your model is overfitting?"
      → Learning curve: if val score plateaus far below train score,
        you're overfitting. Add regularisation, reduce features, or get more data.
    """
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_rmse, val_rmse = [], []
    tscv = TimeSeriesSplit(n_splits=3)

    n = len(X)
    for size in train_sizes:
        end = max(int(n * size), 100)
        X_sub, y_sub = X[:end], y[:end]

        fold_train, fold_val = [], []
        for tr_idx, val_idx in tscv.split(X_sub):
            if len(tr_idx) < 10 or len(val_idx) < 5:
                continue
            model.fit(X_sub[tr_idx], y_sub[tr_idx])
            fold_train.append(np.sqrt(mean_squared_error(
                y_sub[tr_idx], model.predict(X_sub[tr_idx]))))
            fold_val.append(np.sqrt(mean_squared_error(
                y_sub[val_idx], model.predict(X_sub[val_idx]))))

        if fold_train:
            train_rmse.append(np.mean(fold_train))
            val_rmse.append(np.mean(fold_val))

    plt.figure(figsize=(10, 5))
    x = np.arange(len(train_rmse))
    plt.plot(x, train_rmse, "o-", color="#378ADD", label="Train RMSE", linewidth=2)
    plt.plot(x, val_rmse,   "s-", color="#D85A30", label="Val RMSE",   linewidth=2)
    plt.fill_between(x, train_rmse, val_rmse, alpha=0.1, color="gray")
    plt.xlabel("Training Set Size (relative)")
    plt.ylabel("RMSE")
    plt.title(f"Learning Curve — {model_name}\n"
              "converging = good | gap = overfitting | both high = underfitting")
    plt.legend()
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(OUTPUT_DIR / f"learning_curve_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[learning_curve] Saved for {model_name}")


# ── Feature Importance ────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, model_name: str = "XGBoost",
                             top_n: int = 20) -> None:
    """Plot top-N feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        inner = model.named_steps.get("model")
        if inner and hasattr(inner, "coef_"):
            importances = np.abs(inner.coef_)
        elif inner and hasattr(inner, "feature_importances_"):
            importances = inner.feature_importances_
        else:
            print(f"[feature_importance] {model_name} has no importances.")
            return
    else:
        return

    n = min(top_n, len(feature_names))
    idx = np.argsort(importances)[-n:]

    plt.figure(figsize=(10, max(6, n * 0.35)))
    colors = ["#1D9E75" if importances[i] > np.median(importances) else "#B5D4F4"
              for i in idx]
    plt.barh([feature_names[i] for i in idx], importances[idx], color=colors)
    plt.xlabel("Importance")
    plt.title(f"Top {n} Feature Importances — {model_name}")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(OUTPUT_DIR / f"importance_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[importance] Saved for {model_name}")


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))

    from preprocessing.pipeline import build_preprocessing_pipeline, time_based_split, get_X_y
    from features.engineer import apply_feature_engineering

    df = pd.read_csv(ROOT / "data" / "raw" / "train.csv", parse_dates=["Date"])

    # Feature engineering first (needs Sales column for lags)
    print("[main] Building features...")
    df_feat = apply_feature_engineering(df)

    # Time-based split
    train_raw, test_raw = time_based_split(df_feat, test_months=3)

    # Preprocessing pipeline
    pipeline = build_preprocessing_pipeline()
    train_input = train_raw.drop(columns=["Sales", "Customers"], errors="ignore")
    test_input  = test_raw.drop(columns=["Sales", "Customers"],  errors="ignore")

    X_train_df = pipeline.fit_transform(train_input, train_raw["Sales"].values)
    X_test_df  = pipeline.transform(test_input)

    # Fill NaNs from lag features (first few rows)
    X_train_df = X_train_df.fillna(X_train_df.median())
    X_test_df  = X_test_df.fillna(X_train_df.median())

    X_train = X_train_df.values.astype(float)
    X_test  = X_test_df.values.astype(float)
    y_train = train_raw.loc[train_raw.index.isin(
        train_raw[train_raw["Open"] == 1].index), "Sales"].values
    y_test  = test_raw.loc[test_raw.index.isin(
        test_raw[test_raw["Open"] == 1].index), "Sales"].values

    # Align lengths (drop_closed removes rows)
    min_train = min(len(X_train), len(y_train))
    min_test  = min(len(X_test),  len(y_test))
    X_train, y_train = X_train[:min_train], y_train[:min_train]
    X_test,  y_test  = X_test[:min_test],   y_test[:min_test]

    feature_names = list(X_train_df.columns)

    print(f"\n[main] X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Train and compare all models
    print("\n[main] Training all models...")
    results = train_all_models(X_train, y_train, X_test, y_test, feature_names)
    print("\n── Model Comparison ──")
    print(results.to_string(index=False))

    # Residual analysis for best model
    best_name = results.iloc[0]["model"]
    print(f"\n[main] Running residual analysis for best model: {best_name}")
    best_model = joblib.load(MODEL_DIR / f"{best_name.replace(' ', '_')}.pkl")
    plot_residuals(best_model, X_test, y_test, best_name)
    plot_learning_curve(best_model, X_train, y_train, best_name)
    plot_feature_importance(best_model, feature_names, best_name)

    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\n[done] Phase 4 complete.")
