"""
Phase 3: Feature Engineering
==============================
Time-series feature engineering for sales forecasting.

Features built:
  - Lag features    : sales 7, 14, 30 days ago (same store)
  - Rolling features: 7/30-day rolling mean and std
  - Trend features  : linear trend slope over 30 days
  - Interaction     : Promo × IsWeekend, Promo × Month
  - VIF analysis    : detect multicollinearity

FAANG interview note:
  "Why lag features and not just use raw date features?"
  → Lag features capture autocorrelation in sales.
    Yesterday's sales is the best predictor of today's.
    Raw date features only capture seasonality, not momentum.

  "What lag distance do you choose?"
  → Domain-driven: 7 = same weekday last week (strong weekly cycle),
    14 = two-week cycle, 30 = monthly promotion cycles.
    Validate by checking autocorrelation function (ACF plot).

Leakage warning:
  Lag features MUST be computed before the train/test split target date.
  A lag-7 feature for Oct 1 uses Sep 24 sales → safe.
  Never use future values.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore")


# ── Lag & Rolling Features ────────────────────────────────────────────────────

class LagFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Build lag features: sales N days ago for the same store.

    Why per-store? Store 1's sales 7 days ago is meaningful.
    Cross-store lags are noise.
    """
    def __init__(self, lags: list = None, target_col: str = "Sales",
                 store_col: str = "Store", date_col: str = "Date"):
        self.lags = lags or [7, 14, 30]
        self.target_col = target_col
        self.store_col = store_col
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values([self.store_col, self.date_col])

        for lag in self.lags:
            col_name = f"lag_{lag}"
            df[col_name] = (
                df.groupby(self.store_col)[self.target_col]
                .shift(lag)
            )

        return df


class RollingFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Build rolling mean and std features per store.

    Rolling mean: captures recent trend level
    Rolling std:  captures recent volatility (useful for uncertainty models)

    min_periods=1 prevents NaN for the first few rows.
    """
    def __init__(self, windows: list = None, target_col: str = "Sales",
                 store_col: str = "Store"):
        self.windows = windows or [7, 30]
        self.target_col = target_col
        self.store_col = store_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        df = df.sort_values([self.store_col, "Date"] if "Date" in df.columns else self.store_col)

        for w in self.windows:
            grp = df.groupby(self.store_col)[self.target_col]
            df[f"rolling_mean_{w}"] = grp.transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
            )
            df[f"rolling_std_{w}"] = grp.transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).std().fillna(0)
            )

        return df


class TrendFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Compute linear trend slope over a rolling window per store.

    Slope > 0: store is on an upward trajectory
    Slope < 0: store is declining

    Used as a signal for the drift detection module (Phase 6).
    """
    def __init__(self, window: int = 30, target_col: str = "Sales",
                 store_col: str = "Store"):
        self.window = window
        self.target_col = target_col
        self.store_col = store_col

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _slope(series: pd.Series) -> float:
        """Compute slope of linear regression over a series."""
        y = series.values
        if len(y) < 2:
            return 0.0
        x = np.arange(len(y))
        try:
            slope = np.polyfit(x, y, 1)[0]
        except Exception:
            slope = 0.0
        return slope

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        df = df.sort_values([self.store_col, "Date"] if "Date" in df.columns else self.store_col)

        df[f"trend_slope_{self.window}d"] = (
            df.groupby(self.store_col)[self.target_col]
            .transform(lambda x: x.shift(1)
                                   .rolling(window=self.window, min_periods=5)
                                   .apply(self._slope, raw=True)
                                   .fillna(0))
        )
        return df


# ── Interaction Features ──────────────────────────────────────────────────────

class InteractionFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Build interaction features that capture non-linear relationships.

    Promo × IsWeekend: promos on weekends are stronger than either alone
    Promo × Month: promotions in Dec are more effective than in Sep

    FAANG note: Tree models learn these automatically. For Linear Regression,
    you must add them explicitly. These interactions are how you close the
    gap between LR and tree models on structured data.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if "Promo" in df.columns and "IsWeekend" in df.columns:
            df["Promo_x_Weekend"] = df["Promo"] * df["IsWeekend"]

        if "Promo" in df.columns and "Month" in df.columns:
            df["Promo_x_Month"] = df["Promo"] * df["Month"]

        if "Promo" in df.columns and "SchoolHoliday" in df.columns:
            df["Promo_x_SchoolHoliday"] = df["Promo"] * df["SchoolHoliday"]

        if "rolling_mean_7" in df.columns and "lag_7" in df.columns:
            # Momentum: current rolling mean relative to last week
            df["sales_momentum"] = df["rolling_mean_7"] - df["lag_7"]

        return df


# ── VIF Analysis ──────────────────────────────────────────────────────────────

def compute_vif(df: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for each feature.

    VIF = 1        : no correlation
    VIF = 1–5      : moderate correlation (acceptable)
    VIF > 5        : high multicollinearity (concern)
    VIF > 10       : severe multicollinearity (drop or combine features)

    FAANG interview note:
      "How do you detect multicollinearity?"
      → VIF is the standard answer. Correlation matrix only shows pairwise
        relationships. VIF captures the combined effect of ALL other features
        on one feature — much more powerful.

    Args:
        df: feature dataframe
        feature_cols: columns to check (default: all numeric)

    Returns:
        DataFrame with feature and VIF columns, sorted descending
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    X = df[feature_cols].dropna()
    # Add constant for intercept
    X_const = np.column_stack([np.ones(len(X)), X.values])

    vif_data = []
    for i, col in enumerate(feature_cols):
        try:
            vif = variance_inflation_factor(X_const, i + 1)
        except Exception:
            vif = np.nan
        vif_data.append({"feature": col, "VIF": round(vif, 2)})

    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False).reset_index(drop=True)


def drop_high_vif_features(df: pd.DataFrame, threshold: float = 10.0,
                            feature_cols: list = None) -> tuple:
    """
    Iteratively drop features with VIF > threshold.
    Remove one at a time (highest VIF first) and recompute.

    Returns (cleaned_df, list_of_dropped_cols)
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    dropped = []
    cols = feature_cols.copy()

    while True:
        vif_df = compute_vif(df[cols], cols)
        max_vif = vif_df.iloc[0]["VIF"]
        if max_vif <= threshold or len(cols) <= 2:
            break
        drop_col = vif_df.iloc[0]["feature"]
        print(f"[vif] Dropping '{drop_col}' (VIF={max_vif:.1f})")
        cols.remove(drop_col)
        dropped.append(drop_col)

    return df[cols], dropped


# ── Full Feature Engineering Pipeline ────────────────────────────────────────

def build_feature_pipeline(
    lag_days: list = None,
    rolling_windows: list = None,
    trend_window: int = 30,
) -> list:
    """
    Return ordered list of feature engineering transformers.
    Applied BEFORE the preprocessing pipeline transformers.

    Note: Lag/rolling features require the raw Sales column,
    so they must be applied BEFORE dropping Sales from X.
    """
    return [
        LagFeatureBuilder(lags=lag_days or [7, 14, 30]),
        RollingFeatureBuilder(windows=rolling_windows or [7, 30]),
        TrendFeatureBuilder(window=trend_window),
        InteractionFeatureBuilder(),
    ]


def apply_feature_engineering(df: pd.DataFrame, lag_days=None,
                               rolling_windows=None) -> pd.DataFrame:
    """Apply all feature engineering steps to a dataframe."""
    transformers = build_feature_pipeline(lag_days, rolling_windows)
    result = df.copy()
    for transformer in transformers:
        result = transformer.transform(result)
    return result


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent.parent
    df = pd.read_csv(ROOT / "data" / "raw" / "train.csv", parse_dates=["Date"])

    # Apply feature engineering
    print("[feature_eng] Building lag, rolling, trend, interaction features...")
    df_feat = apply_feature_engineering(df)

    print(f"\n[feature_eng] Shape before: {df.shape}, after: {df_feat.shape}")
    new_cols = [c for c in df_feat.columns if c not in df.columns]
    print(f"[feature_eng] New features added: {new_cols}")

    # Sample of engineered features
    sample = df_feat[df_feat["Store"] == 1].sort_values("Date").head(10)
    print("\n[feature_eng] Sample (Store 1):")
    print(sample[["Date", "Sales", "lag_7", "lag_14", "rolling_mean_7",
                   "rolling_std_7", "trend_slope_30d"]].to_string())

    # VIF analysis
    print("\n[vif] Computing VIF on numeric features...")
    numeric_cols = ["Promo", "SchoolHoliday", "DayOfWeek", "CompetitionDistance",
                    "lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_30"]
    available = [c for c in numeric_cols if c in df_feat.columns]
    vif_df = compute_vif(df_feat.dropna(subset=available), available)
    print(vif_df.to_string())

    print("\n[done] Phase 3 complete.")
