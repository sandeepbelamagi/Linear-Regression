"""
Phase 5: Uncertainty Estimation & Model Routing
================================================
Bootstrap ensemble for uncertainty quantification and a
tiered routing layer — fast model vs accurate model.

Why uncertainty estimation?
  In production, a model that says "I predict $8,000 ± $200"
  is far more useful than one that just says "$8,000".
  High uncertainty = flag for review or route to a better model.

Bootstrap ensemble approach:
  Train B models on B bootstrap samples (random samples with replacement).
  Prediction = mean of B predictions.
  Uncertainty = std of B predictions.
  → Wide std = high uncertainty → route to accurate (slow) model.

Tiered routing:
  Fast model (LinearRegression)  → low-uncertainty predictions at low cost
  Accurate model (XGBoost)       → high-uncertainty predictions, more compute

FAANG system design note:
  This is a classic ML serving tradeoff: latency vs accuracy.
  The routing layer sits in your inference service. In practice:
    - p50 requests → fast model (< 5ms)
    - p95 uncertain requests → accurate model (< 50ms)
  Netflix, Uber, and Amazon all use this pattern.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline
import xgboost as xgb
import joblib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR  = ROOT / "outputs" / "models"


# ── Bootstrap Ensemble ────────────────────────────────────────────────────────

class BootstrapEnsemble:
    """
    Bootstrap ensemble for uncertainty quantification.

    Attributes:
        n_bootstrap: number of bootstrap models
        base_model_fn: callable returning a fresh model instance
        models_: list of fitted models
        prediction_intervals_: (lower, upper) bounds on last predict call
    """

    def __init__(self, n_bootstrap: int = 50, base_model_fn=None):
        self.n_bootstrap = n_bootstrap
        self.base_model_fn = base_model_fn or self._default_model
        self.models_ = []

    @staticmethod
    def _default_model():
        return SKPipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=10.0)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BootstrapEnsemble":
        """Fit n_bootstrap models on bootstrap samples."""
        self.models_ = []
        n = len(X)
        print(f"[bootstrap] Training {self.n_bootstrap} models...", end=" ")
        for i in range(self.n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            model = self.base_model_fn()
            model.fit(X[idx], y[idx])
            self.models_.append(model)
        print("done.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return mean prediction across all bootstrap models."""
        preds = np.stack([m.predict(X) for m in self.models_], axis=1)
        return preds.mean(axis=1)

    def predict_with_uncertainty(
        self, X: np.ndarray, ci: float = 0.90
    ) -> tuple:
        """
        Return (mean, std, lower_bound, upper_bound) for each prediction.

        Args:
            X: feature matrix
            ci: confidence interval (0.90 = 90% PI)

        Returns:
            mean_pred, std_pred, lower, upper
        """
        preds = np.stack([m.predict(X) for m in self.models_], axis=1)
        alpha = (1 - ci) / 2
        mean  = preds.mean(axis=1)
        std   = preds.std(axis=1)
        lower = np.percentile(preds, alpha * 100, axis=1)
        upper = np.percentile(preds, (1 - alpha) * 100, axis=1)
        return mean, std, lower, upper

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "BootstrapEnsemble":
        return joblib.load(path)


# ── Tiered Routing Layer ──────────────────────────────────────────────────────

class TieredRouter:
    """
    Routes predictions to fast or accurate model based on uncertainty.

    Architecture:
        Input → Bootstrap Ensemble → uncertainty estimate
                                          ↓
                              uncertainty > threshold?
                                  YES               NO
                                  ↓                  ↓
                           Accurate model       Fast model
                           (XGBoost)            (Ridge)

    FAANG interview note:
      This is essentially a confidence-based ensemble.
      Key hyperparameter: uncertainty_threshold
        - Too low  → everything goes to slow model → defeats purpose
        - Too high → uncertain predictions use fast model → bad accuracy
      Tune by minimising weighted_cost = accuracy_loss + latency_cost
    """

    def __init__(
        self,
        fast_model,
        accurate_model,
        ensemble: BootstrapEnsemble,
        uncertainty_threshold: float = None,
    ):
        self.fast_model = fast_model
        self.accurate_model = accurate_model
        self.ensemble = ensemble
        self.uncertainty_threshold = uncertainty_threshold
        self._threshold_fitted = False

    def fit_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        percentile: float = 80.0,
    ) -> float:
        """
        Set threshold at the Nth percentile of validation uncertainty.
        Top (100 - percentile)% most uncertain samples go to accurate model.

        Args:
            percentile: e.g. 80 means top 20% most uncertain → accurate model
        """
        _, std_val, _, _ = self.ensemble.predict_with_uncertainty(X_val)
        self.uncertainty_threshold = float(np.percentile(std_val, percentile))
        self._threshold_fitted = True
        routed_pct = 100 - percentile
        print(f"[router] Threshold: {self.uncertainty_threshold:.2f}  "
              f"({routed_pct:.0f}% of predictions → accurate model)")
        return self.uncertainty_threshold

    def predict(self, X: np.ndarray) -> tuple:
        """
        Route each sample to fast or accurate model.

        Returns:
            predictions: final predictions
            routing_mask: True where accurate model was used
            uncertainties: std for each prediction
        """
        if self.uncertainty_threshold is None:
            raise RuntimeError("Call fit_threshold() first.")

        _, std, _, _ = self.ensemble.predict_with_uncertainty(X)
        use_accurate = std > self.uncertainty_threshold

        predictions = np.zeros(len(X))
        if use_accurate.any():
            predictions[use_accurate]  = self.accurate_model.predict(X[use_accurate])
        if (~use_accurate).any():
            predictions[~use_accurate] = self.fast_model.predict(X[~use_accurate])

        return predictions, use_accurate, std

    def routing_stats(self, X: np.ndarray) -> dict:
        """Return statistics about routing decisions."""
        _, use_accurate, std = self.predict(X)
        return {
            "total_samples":    len(X),
            "fast_model_pct":   round(float((~use_accurate).mean() * 100), 1),
            "accurate_model_pct": round(float(use_accurate.mean() * 100), 1),
            "mean_uncertainty": round(float(std.mean()), 2),
            "p90_uncertainty":  round(float(np.percentile(std, 90)), 2),
        }


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_uncertainty(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    routing_mask: np.ndarray = None,
    n_samples: int = 200,
) -> None:
    """
    Plot prediction intervals and routing decisions.
    """
    idx = np.argsort(y_true)[:n_samples]
    x = np.arange(n_samples)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Uncertainty Estimation & Routing", fontsize=14, fontweight="bold")

    # Prediction intervals
    axes[0].fill_between(x, y_lower[idx], y_upper[idx], alpha=0.25,
                          color="#7F77DD", label="90% Prediction Interval")
    axes[0].plot(x, y_true[idx],  ".", color="#1D9E75", markersize=3, label="Actual", alpha=0.7)
    axes[0].plot(x, y_mean[idx],  "-", color="#D85A30", linewidth=1.5, label="Predicted mean")
    axes[0].set_xlabel("Sample (sorted by actual)")
    axes[0].set_ylabel("Sales")
    axes[0].set_title("Prediction Intervals (90% CI)")
    axes[0].legend()

    # Routing visualisation
    if routing_mask is not None:
        colors = ["#D85A30" if routing_mask[i] else "#1D9E75" for i in idx]
        axes[1].scatter(x, y_true[idx], c=colors, s=10, alpha=0.7)
        axes[1].set_xlabel("Sample")
        axes[1].set_ylabel("Sales")
        fast_pct = (~routing_mask[idx]).mean() * 100
        axes[1].set_title(f"Routing: Green=fast model ({fast_pct:.0f}%), "
                          f"Red=accurate model ({100-fast_pct:.0f}%)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "uncertainty_routing.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] Uncertainty routing plot saved.")


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))

    from preprocessing.pipeline import build_preprocessing_pipeline, time_based_split
    from features.engineer import apply_feature_engineering

    df = pd.read_csv(ROOT / "data" / "raw" / "train.csv", parse_dates=["Date"])
    df_feat = apply_feature_engineering(df)
    train_raw, test_raw = time_based_split(df_feat, test_months=3)

    pipeline = build_preprocessing_pipeline()
    train_input = train_raw.drop(columns=["Sales", "Customers"], errors="ignore")
    test_input  = test_raw.drop(columns=["Sales", "Customers"],  errors="ignore")

    X_train_df = pipeline.fit_transform(train_input, train_raw["Sales"].values)
    X_test_df  = pipeline.transform(test_input)
    X_train_df = X_train_df.fillna(X_train_df.median())
    X_test_df  = X_test_df.fillna(X_train_df.median())

    X_train = X_train_df.values.astype(float)
    X_test  = X_test_df.values.astype(float)

    open_train_mask = train_raw["Open"] == 1
    open_test_mask  = test_raw["Open"] == 1
    y_train = train_raw.loc[open_train_mask, "Sales"].values[:len(X_train)]
    y_test  = test_raw.loc[open_test_mask,  "Sales"].values[:len(X_test)]

    # Bootstrap ensemble (fast model)
    ensemble = BootstrapEnsemble(n_bootstrap=30)
    ensemble.fit(X_train, y_train)

    mean_pred, std_pred, lower, upper = ensemble.predict_with_uncertainty(X_test)
    print(f"\n[uncertainty] Mean prediction std: {std_pred.mean():.2f}")
    print(f"[uncertainty] P90 uncertainty:     {np.percentile(std_pred, 90):.2f}")

    # Load or train accurate model
    xgb_path = MODEL_DIR / "XGBoost.pkl"
    if xgb_path.exists():
        accurate_model = joblib.load(xgb_path)
    else:
        accurate_model = xgb.XGBRegressor(n_estimators=200, max_depth=6,
                                           learning_rate=0.05, random_state=42)
        accurate_model.fit(X_train, y_train)

    fast_model = SKPipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
    fast_model.fit(X_train, y_train)

    # Routing
    router = TieredRouter(fast_model, accurate_model, ensemble)
    router.fit_threshold(X_test, y_test, percentile=80)
    final_preds, routing_mask, uncertainties = router.predict(X_test)

    stats = router.routing_stats(X_test)
    print(f"\n[router] Routing stats: {stats}")

    plot_uncertainty(y_test, mean_pred, lower, upper, routing_mask)

    ensemble.save(MODEL_DIR / "bootstrap_ensemble.pkl")
    joblib.dump(router, MODEL_DIR / "tiered_router.pkl")

    print("\n[done] Phase 5 complete.")
