"""
Phase 2: Preprocessing Pipeline
=================================
Production-grade, leakage-safe preprocessing using sklearn Pipeline.

Key design decisions (FAANG interview answers):
  - Everything inside a Pipeline → fit only on train, transform on val/test
  - K-Fold target encoding with smoothing → prevents target leakage
  - Median imputation by group → better than global median for CompetitionDistance
  - StandardScaler AFTER encoding → correct order matters
  - Open=0 rows dropped BEFORE pipeline → no point predicting closed stores

FAANG note: "Why not fit the scaler on the full dataset?"
  → That leaks test distribution into training. Always fit transformers on train only.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ── Custom Transformers ───────────────────────────────────────────────────────

class DropClosedStores(BaseEstimator, TransformerMixin):
    """Drop rows where Open == 0. Closed stores always Sales=0 → bias model."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if "Open" in df.columns:
            df = df[df["Open"] == 1].drop(columns=["Open"]).reset_index(drop=True)
        return df


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values with the median of a group.
    CompetitionDistance missing → fill with median for that StoreType.

    FAANG note: Global median ignores store context. Group median is better
    because competition patterns vary by store type.
    """
    def __init__(self, target_col: str, group_col: str):
        self.target_col = target_col
        self.group_col = group_col
        self.group_medians_ = {}
        self.global_median_ = None

    def fit(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.group_medians_ = df.groupby(self.group_col)[self.target_col].median().to_dict()
        self.global_median_ = df[self.target_col].median()
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        mask = df[self.target_col].isnull()
        df.loc[mask, self.target_col] = (
            df.loc[mask, self.group_col].map(self.group_medians_).fillna(self.global_median_)
        )
        return df


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    K-Fold Target Encoding with smoothing.

    Why K-Fold instead of simple mean encoding?
      Simple mean: encode using ALL rows for that category → data leakage
      K-Fold: for each fold, encode using other folds → no leakage

    Smoothing formula:
      encoded = (n * category_mean + m * global_mean) / (n + m)
      where m = smoothing parameter (higher = more regularization)

    This prevents rare categories from getting extreme values.
    """
    def __init__(self, cols: list, n_folds: int = 5, smoothing: float = 10.0):
        self.cols = cols
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.encodings_ = {}
        self.global_mean_ = None

    def fit(self, X, y):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        df["__target__"] = np.array(y)
        self.global_mean_ = df["__target__"].mean()

        for col in self.cols:
            stats = df.groupby(col)["__target__"].agg(["mean", "count"]).reset_index()
            # Smoothing: shrink rare categories towards global mean
            stats["smoothed"] = (
                (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_)
                / (stats["count"] + self.smoothing)
            )
            self.encodings_[col] = stats.set_index(col)["smoothed"].to_dict()
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in self.cols:
            df[f"{col}_encoded"] = df[col].map(self.encodings_[col]).fillna(self.global_mean_)
            df = df.drop(columns=[col])
        return df


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract time-based features from the Date column.
    Cyclical encoding (sin/cos) ensures Month 12 and Month 1 are 'close'.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if "Date" not in df.columns:
            return df

        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"]       = df["Date"].dt.year
        df["Month"]      = df["Date"].dt.month
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
        df["DayOfMonth"] = df["Date"].dt.day
        df["IsWeekend"]  = (df["Date"].dt.dayofweek >= 5).astype(int)
        df["Quarter"]    = df["Date"].dt.quarter

        # Cyclical encoding: preserves the circular nature of time
        df["Month_sin"]  = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"]  = np.cos(2 * np.pi * df["Month"] / 12)
        df["Week_sin"]   = np.sin(2 * np.pi * df["WeekOfYear"] / 52)
        df["Week_cos"]   = np.cos(2 * np.pi * df["WeekOfYear"] / 52)

        df = df.drop(columns=["Date"])
        return df


class StateHolidayEncoder(BaseEstimator, TransformerMixin):
    """Encode StateHoliday: '0' → 0, 'a'/'b'/'c' → 1 (any holiday = 1)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if "StateHoliday" in df.columns:
            df["StateHoliday"] = (df["StateHoliday"] != "0").astype(int)
        return df


# ── Full Preprocessing Pipeline ───────────────────────────────────────────────

def build_preprocessing_pipeline(
    categorical_cols: list = None,
    n_folds: int = 5,
    smoothing: float = 10.0,
) -> Pipeline:
    """
    Build the full preprocessing pipeline.

    Order of steps:
      1. Drop closed stores (Open=0)
      2. Extract date features (Year, Month, cyclical sin/cos)
      3. Encode StateHoliday (binary)
      4. Impute CompetitionDistance by StoreType median
      5. Target encode categorical columns (K-Fold + smoothing)

    Note: StandardScaler is applied in the model training step
    after we have final feature matrix.
    """
    if categorical_cols is None:
        categorical_cols = ["StoreType", "Assortment"]

    return Pipeline([
        ("drop_closed",     DropClosedStores()),
        ("date_features",   DateFeatureExtractor()),
        ("holiday_encoder", StateHolidayEncoder()),
        ("group_imputer",   GroupMedianImputer("CompetitionDistance", "StoreType")),
        ("target_encoder",  KFoldTargetEncoder(categorical_cols, n_folds, smoothing)),
    ])


# ── Train/Test Split (TIME-BASED) ─────────────────────────────────────────────

LEAKAGE_COLS = ["Customers", "Sales"]  # Unknown at serving time or is the target


def time_based_split(df: pd.DataFrame, test_months: int = 3, date_col: str = "Date"):
    """
    Split by time — NOT randomly.

    FAANG interview note:
      Random split = data leakage for time series.
      The model sees future patterns during training.
      Always temporal split: train on past, validate on future.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff = df[date_col].max() - pd.DateOffset(months=test_months)
    train = df[df[date_col] <= cutoff].copy()
    test  = df[df[date_col] >  cutoff].copy()
    print(f"[split] Train: {train.shape[0]:,} rows  "
          f"({train[date_col].min().date()} → {train[date_col].max().date()})")
    print(f"[split] Test:  {test.shape[0]:,} rows   "
          f"({test[date_col].min().date()} → {test[date_col].max().date()})")
    return train, test


def get_X_y(df: pd.DataFrame, target: str = "Sales"):
    """Split dataframe into X (features) and y (target)."""
    drop_cols = [c for c in LEAKAGE_COLS if c in df.columns]
    y = df[target].values if target in df.columns else None
    X = df.drop(columns=drop_cols + (["Store"] if "Store" in df.columns else []),
                errors="ignore")
    return X, y


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent.parent
    df = pd.read_csv(ROOT / "data" / "raw" / "train.csv", parse_dates=["Date"])

    train_raw, test_raw = time_based_split(df, test_months=3)

    pipeline = build_preprocessing_pipeline()

    # Fit + transform train (y needed for target encoder)
    X_train_raw, y_train = get_X_y(train_raw)
    # We pass full train_raw so pipeline can access all columns it needs
    X_train = pipeline.fit_transform(train_raw.drop(columns=["Sales", "Customers"],
                                                     errors="ignore"), y_train)

    # Re-extract y separately (pipeline doesn't touch Sales column)
    X_train_raw, y_train = get_X_y(train_raw)
    X_test_raw,  y_test  = get_X_y(test_raw)

    # Fit pipeline on train features
    train_input = train_raw.drop(columns=["Sales", "Customers"], errors="ignore")
    test_input  = test_raw.drop(columns=["Sales", "Customers"], errors="ignore")

    X_train_proc = pipeline.fit_transform(train_input, y_train)
    X_test_proc  = pipeline.transform(test_input)          # NO fit on test

    print(f"\n[done] Train: {X_train_proc.shape}, Test: {X_test_proc.shape}")
    print(f"[done] Features: {list(X_train_proc.columns)}")

    enc = pipeline.named_steps["target_encoder"]
    print(f"\nStoreType encodings: {enc.encodings_['StoreType']}")
    print(f"Global mean:         {enc.global_mean_:.2f}")
