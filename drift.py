"""
Phase 6: Drift Detection & Monitoring
=======================================
Detect covariate drift and concept drift in production.

Types of drift:
  Covariate drift  : P(X) changes — input feature distribution shifts
                     e.g., more promotions than expected, different seasonality
  Concept drift    : P(Y|X) changes — the relationship between features and target shifts
                     e.g., promo effect weakens due to promotion fatigue

Detection methods:
  1. PSI (Population Stability Index)
     → Measures shift in feature distributions between reference and current
     → PSI < 0.10: no drift  |  0.10-0.25: moderate  |  > 0.25: significant drift

  2. Residual mean tracking
     → Monitor mean residual over time. If mean ≠ 0, model is systematically biased.
     → Use CUSUM (Cumulative Sum) for sequential drift detection

  3. KS Test (Kolmogorov-Smirnov)
     → Statistical test for distribution shift
     → p-value < 0.05 → distribution changed significantly

FAANG interview note — "How would you monitor an ML model in production?":
  Three layers:
    Data layer:  PSI on input features (detect covariate drift)
    Model layer: Residual mean/std tracking over time windows
    Business layer: downstream KPIs (revenue, conversion) correlated with predictions
  Alert when any layer crosses threshold → trigger retraining pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "outputs"


# ── PSI (Population Stability Index) ─────────────────────────────────────────

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute PSI between reference and current distributions.

    PSI = sum((current% - reference%) * ln(current% / reference%))

    Thresholds:
      < 0.10  : no significant change
      0.10-0.25: moderate change, monitor
      > 0.25  : significant change, retrain

    Args:
        reference: baseline distribution (train data)
        current:   new distribution (recent production data)
        n_bins:    number of equal-frequency bins
        epsilon:   small constant to prevent log(0)

    Returns:
        PSI value
    """
    # Create bins based on reference distribution
    breakpoints = np.nanpercentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    ref_counts = np.histogram(reference, bins=breakpoints)[0] + epsilon
    cur_counts = np.histogram(current,   bins=breakpoints)[0] + epsilon

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_psi_report(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Compute PSI for all features. Return sorted report.

    FAANG note: Run this daily/weekly in production.
    Alert on features with PSI > 0.25.
    """
    results = []
    for col in feature_cols:
        ref = df_reference[col].dropna().values
        cur = df_current[col].dropna().values
        if len(ref) < 10 or len(cur) < 10:
            continue
        psi_val = compute_psi(ref, cur)
        status = (
            "CRITICAL" if psi_val > 0.25 else
            "WARNING"  if psi_val > 0.10 else
            "OK"
        )
        results.append({
            "feature": col,
            "PSI":     round(psi_val, 4),
            "status":  status,
        })

    return pd.DataFrame(results).sort_values("PSI", ascending=False).reset_index(drop=True)


# ── KS Test for Feature Drift ─────────────────────────────────────────────────

def ks_drift_test(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    feature_cols: list,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Kolmogorov-Smirnov test for each feature.
    H0: distributions are the same.
    Reject H0 (p < alpha) → drift detected.
    """
    results = []
    for col in feature_cols:
        ref = df_reference[col].dropna().values
        cur = df_current[col].dropna().values
        if len(ref) < 10 or len(cur) < 10:
            continue
        ks_stat, p_value = stats.ks_2samp(ref, cur)
        results.append({
            "feature":  col,
            "KS_stat":  round(ks_stat, 4),
            "p_value":  round(p_value, 4),
            "drift":    p_value < alpha,
        })

    return pd.DataFrame(results).sort_values("KS_stat", ascending=False).reset_index(drop=True)


# ── Residual Monitoring ───────────────────────────────────────────────────────

class ResidualMonitor:
    """
    Monitor prediction residuals over time.
    Detect systematic bias using CUSUM.

    CUSUM (Cumulative Sum Control Chart):
      - Designed for sequential change detection
      - Accumulates deviations from expected mean
      - Triggers when cumulative sum exceeds threshold
      - Much faster at detecting small persistent shifts than threshold-on-mean

    FAANG scenario:
      The notes describe an inventory shortage scenario where sales drop
      because products are unavailable, not because demand dropped.
      CUSUM would detect this within 3-5 days.
      Simple threshold-on-residual would take much longer.
    """

    def __init__(self, cusum_threshold: float = 5.0, drift_window: int = 7):
        self.cusum_threshold = cusum_threshold
        self.drift_window = drift_window
        self.residuals_history = []
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.alerts = []

    def update(self, y_true: np.ndarray, y_pred: np.ndarray,
               timestamp: str = None) -> dict:
        """
        Update monitor with new batch of predictions.

        Returns alert dict if drift detected.
        """
        residuals = y_true - y_pred
        mean_res  = float(np.mean(residuals))
        std_res   = float(np.std(residuals))

        self.residuals_history.append({
            "timestamp":    timestamp or str(len(self.residuals_history)),
            "mean_residual": round(mean_res, 2),
            "std_residual":  round(std_res, 2),
            "n_samples":    len(y_true),
        })

        # CUSUM update
        k = std_res * 0.5  # allowance
        self.cusum_pos = max(0, self.cusum_pos + mean_res - k)
        self.cusum_neg = max(0, self.cusum_neg - mean_res - k)

        alert = None
        if self.cusum_pos > self.cusum_threshold:
            alert = {
                "type":    "POSITIVE_DRIFT",
                "message": f"Model underpredicting. CUSUM_pos={self.cusum_pos:.2f}",
                "timestamp": timestamp,
                "cusum_pos": round(self.cusum_pos, 2),
            }
            self.alerts.append(alert)
            self.cusum_pos = 0  # reset after alert
        elif self.cusum_neg > self.cusum_threshold:
            alert = {
                "type":    "NEGATIVE_DRIFT",
                "message": f"Model overpredicting. CUSUM_neg={self.cusum_neg:.2f}",
                "timestamp": timestamp,
                "cusum_neg": round(self.cusum_neg, 2),
            }
            self.alerts.append(alert)
            self.cusum_neg = 0

        return alert

    def get_summary(self) -> dict:
        """Return monitoring summary."""
        if not self.residuals_history:
            return {}
        df = pd.DataFrame(self.residuals_history)
        return {
            "total_batches": len(df),
            "overall_mean_residual": round(df["mean_residual"].mean(), 2),
            "recent_mean_residual":  round(df["mean_residual"].tail(self.drift_window).mean(), 2),
            "total_alerts":          len(self.alerts),
            "last_alert":            self.alerts[-1] if self.alerts else None,
        }


# ── Simulate Drift ────────────────────────────────────────────────────────────

def simulate_concept_drift(
    df: pd.DataFrame,
    drift_start_month: str = "2023-10",
    promo_effectiveness_drop: float = 0.3,
) -> pd.DataFrame:
    """
    Simulate concept drift: promo effectiveness drops after a date.
    This mimics promotion fatigue or a competitor launching a campaign.

    In real production, you would NOT simulate this — you'd detect it.
    This is for demonstrating that your monitoring catches it.
    """
    df_drift = df.copy()
    df_drift["Date"] = pd.to_datetime(df_drift["Date"])
    drift_mask = (
        (df_drift["Date"] >= drift_start_month) & (df_drift["Promo"] == 1)
    )
    # Reduce promo-driven sales by the specified fraction
    df_drift.loc[drift_mask, "Sales"] *= (1 - promo_effectiveness_drop)
    print(f"[drift_sim] Applied concept drift from {drift_start_month}: "
          f"promo effectiveness ↓{promo_effectiveness_drop*100:.0f}%")
    return df_drift


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_drift_monitoring(
    residuals_by_month: pd.DataFrame,
    psi_report: pd.DataFrame,
) -> None:
    """Plot monitoring dashboard: residuals over time + PSI bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Production Monitoring Dashboard", fontsize=14, fontweight="bold")

    # Residual mean over time
    months = residuals_by_month["month"].astype(str)
    x = np.arange(len(months))
    axes[0].bar(x, residuals_by_month["mean_residual"],
                color=["#D85A30" if abs(r) > 500 else "#1D9E75"
                       for r in residuals_by_month["mean_residual"]])
    axes[0].axhline(0, color="black", linewidth=1.5, linestyle="--")
    axes[0].axhline(500,  color="#D85A30", linewidth=1, linestyle=":", alpha=0.6,
                    label="Alert threshold")
    axes[0].axhline(-500, color="#D85A30", linewidth=1, linestyle=":", alpha=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Mean Residual Over Time\n(red = bias alert)")
    axes[0].set_ylabel("Mean Residual (Actual - Predicted)")
    axes[0].legend()

    # PSI report
    top_psi = psi_report.head(10)
    colors = ["#D85A30" if s == "CRITICAL" else "#BA7517" if s == "WARNING" else "#1D9E75"
              for s in top_psi["status"]]
    axes[1].barh(top_psi["feature"], top_psi["PSI"], color=colors)
    axes[1].axvline(0.10, color="#BA7517", linewidth=1.5, linestyle="--", label="Warning (0.10)")
    axes[1].axvline(0.25, color="#D85A30", linewidth=1.5, linestyle="--", label="Critical (0.25)")
    axes[1].set_title("Feature Drift (PSI)\nGreen < 0.10 | Yellow 0.10-0.25 | Red > 0.25")
    axes[1].set_xlabel("PSI")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monitoring_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] Monitoring dashboard saved.")


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))

    from preprocessing.pipeline import build_preprocessing_pipeline, time_based_split
    from features.engineer import apply_feature_engineering

    df = pd.read_csv(ROOT / "data" / "raw" / "train.csv", parse_dates=["Date"])

    # Simulate drift in second year
    df_drifted = simulate_concept_drift(df, drift_start_month="2023-09",
                                         promo_effectiveness_drop=0.3)

    df_feat        = apply_feature_engineering(df)
    df_feat_drifted = apply_feature_engineering(df_drifted)

    train_raw, test_raw = time_based_split(df_feat, test_months=3)
    _, test_drift = time_based_split(df_feat_drifted, test_months=3)

    pipeline = build_preprocessing_pipeline()
    train_input = train_raw.drop(columns=["Sales", "Customers"], errors="ignore")
    test_input  = test_raw.drop(columns=["Sales", "Customers"],  errors="ignore")
    test_drift_input = test_drift.drop(columns=["Sales", "Customers"], errors="ignore")

    X_train_df = pipeline.fit_transform(train_input, train_raw["Sales"].values)
    X_train_df = X_train_df.fillna(X_train_df.median())
    X_test_df  = pipeline.transform(test_input).fillna(X_train_df.median())
    X_drift_df = pipeline.transform(test_drift_input).fillna(X_train_df.median())

    # PSI: compare train vs test feature distributions
    numeric_cols = [c for c in X_train_df.select_dtypes(include=np.number).columns]
    print("\n[drift] PSI Report (train vs test):")
    psi_report = compute_psi_report(X_train_df, X_test_df, numeric_cols)
    print(psi_report.head(10).to_string())

    # KS test
    print("\n[drift] KS Test (train vs test):")
    ks_report = ks_drift_test(X_train_df, X_test_df, numeric_cols[:8])
    print(ks_report.to_string())

    # Residual monitor: simulate monthly batches
    import joblib
    xgb_path = ROOT / "outputs" / "models" / "XGBoost.pkl"
    if xgb_path.exists():
        model = joblib.load(xgb_path)
        monitor = ResidualMonitor(cusum_threshold=5.0)
        test_drift["Date"] = pd.to_datetime(test_drift["Date"])
        test_drift["YearMonth"] = test_drift["Date"].dt.to_period("M")

        monthly_residuals = []
        for month, grp in test_drift.groupby("YearMonth"):
            grp_input = grp.drop(columns=["Sales", "Customers"], errors="ignore")
            X_m = pipeline.transform(grp_input).fillna(X_train_df.median()).values.astype(float)
            y_m = grp["Sales"].values[:len(X_m)]
            y_p = model.predict(X_m[:len(y_m)])
            alert = monitor.update(y_m, y_p, timestamp=str(month))
            mean_res = float(np.mean(y_m - y_p))
            monthly_residuals.append({"month": month, "mean_residual": mean_res})
            if alert:
                print(f"[ALERT] {alert['message']}")

        summary = monitor.get_summary()
        print(f"\n[monitor] Summary: {summary}")

        monthly_df = pd.DataFrame(monthly_residuals)
        plot_drift_monitoring(monthly_df, psi_report)

    with open(OUTPUT_DIR / "psi_report.json", "w") as f:
        json.dump(psi_report.to_dict(orient="records"), f, indent=2)

    print("\n[done] Phase 6 complete.")
