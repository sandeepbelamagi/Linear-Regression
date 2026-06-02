"""
Phase 7: Production API — FastAPI Inference Service
=====================================================
REST API for real-time sales prediction with:
  - /predict        : single or batch prediction
  - /predict/batch  : batch with uncertainty
  - /health         : health check
  - /model/info     : model metadata
  - /monitor/psi    : latest PSI report
  - /monitor/alerts : recent drift alerts

Production design decisions (FAANG system design):
  - Pydantic for request/response validation
  - Async endpoints for non-blocking I/O
  - Model loaded once at startup (not per request)
  - Input validation with business-logic checks
  - Structured logging (JSON) for observability
  - Prediction with uncertainty (not just point estimate)
  - Version header in every response

FAANG interview note — "How would you serve this model in production?":
  1. FastAPI → Gunicorn workers → Docker container
  2. Behind a load balancer (nginx or AWS ALB)
  3. Model artifact in S3 / GCS, loaded at container startup
  4. Feature store (Feast / Tecton) for pre-computed lag features
  5. Request → Feature store → Model server → Response
  6. Async logging to Kafka → Spark → drift detection pipeline
  7. SLA: p50 < 5ms, p99 < 50ms for the fast model

To run:
  uvicorn src.api.serve:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Setup ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent.parent
MODEL_DIR  = ROOT / "outputs" / "models"
OUTPUT_DIR = ROOT / "outputs"
MPLCONFIGDIR = ROOT / ".tmp" / "mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

app = FastAPI(
    title="Sales Forecasting API",
    description="Production-grade sales prediction with uncertainty estimation",
    version="1.0.0",
)

# ── Global model state (loaded once at startup) ───────────────────────────────

class ModelRegistry:
    """
    Holds loaded models and preprocessors.
    Loaded once at startup → not per request (would be very slow).
    """
    pipeline      = None
    fast_model    = None
    accurate_model = None
    ensemble      = None
    router        = None
    feature_names = None
    loaded_at     = None
    train_median  = None  # for filling NaN from lag features
    raw_history   = None

registry = ModelRegistry()


@app.on_event("startup")
async def load_models():
    """Load all models at startup."""
    logger.info("Loading models...")
    try:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from features.engineer import apply_feature_engineering
        from preprocessing.pipeline import build_preprocessing_pipeline, time_based_split

        registry.raw_history = pd.read_csv(
            ROOT / "data" / "raw" / "train.csv",
            parse_dates=["Date"],
        ).sort_values(["Store", "Date"]).reset_index(drop=True)

        # Recreate the same preprocessing fit used during training so
        # inference feature names and target encodings line up with the models.
        engineered = apply_feature_engineering(registry.raw_history)
        train_raw, _ = time_based_split(engineered, test_months=3)
        train_input = train_raw.drop(columns=["Sales", "Customers"], errors="ignore")
        y_train = train_raw.loc[train_raw["Open"] == 1, "Sales"].values

        registry.pipeline = build_preprocessing_pipeline()
        X_train_df = registry.pipeline.fit_transform(train_input, y_train)
        registry.feature_names = list(X_train_df.columns)
        registry.train_median = X_train_df.median(numeric_only=True)

        import __main__
        from models.uncertainty import BootstrapEnsemble, TieredRouter

        setattr(__main__, "BootstrapEnsemble", BootstrapEnsemble)
        setattr(__main__, "TieredRouter", TieredRouter)

        for model_name, attr in [
            ("XGBoost.pkl",            "accurate_model"),
            ("Ridge_(L2).pkl",         "fast_model"),
            ("bootstrap_ensemble.pkl", "ensemble"),
            ("tiered_router.pkl",      "router"),
        ]:
            path = MODEL_DIR / model_name
            if path.exists():
                try:
                    setattr(registry, attr, joblib.load(path))
                except Exception as model_exc:
                    logger.warning(f"Could not load {model_name}: {model_exc}")

        registry.loaded_at = datetime.utcnow().isoformat()
        logger.info("Models loaded successfully.")

    except Exception as e:
        logger.error(f"Model load failed: {e}")
        # API still starts — returns 503 on predict if model missing


# ── Pydantic Models ───────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Single store prediction request."""
    store_id:            int   = Field(..., ge=1,    description="Store ID")
    date:                str   = Field(...,           description="Prediction date (YYYY-MM-DD)")
    day_of_week:         int   = Field(..., ge=1, le=7)
    promo:               int   = Field(..., ge=0, le=1)
    state_holiday:       str   = Field(default="0")
    school_holiday:      int   = Field(default=0, ge=0, le=1)
    store_type:          str   = Field(default="a")
    assortment:          str   = Field(default="basic")
    competition_distance: Optional[float] = Field(default=None, ge=0)
    promo2:              int   = Field(default=0, ge=0, le=1)

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date must be YYYY-MM-DD format")
        return v

    @field_validator("store_type")
    @classmethod
    def validate_store_type(cls, v):
        if v not in {"a", "b", "c", "d"}:
            raise ValueError("store_type must be a/b/c/d")
        return v


class PredictionResponse(BaseModel):
    store_id:           int
    date:               str
    predicted_sales:    float
    lower_bound:        Optional[float]
    upper_bound:        Optional[float]
    uncertainty:        Optional[float]
    model_used:         str
    latency_ms:         float
    api_version:        str = "1.0.0"


class BatchRequest(BaseModel):
    requests: List[PredictionRequest]


class BatchResponse(BaseModel):
    predictions:    List[PredictionResponse]
    total_latency_ms: float
    model_version:  str = "1.0.0"


# ── Feature Preparation ───────────────────────────────────────────────────────

def request_to_dataframe(req: PredictionRequest) -> pd.DataFrame:
    """Convert a PredictionRequest to a one-row DataFrame for preprocessing."""
    return pd.DataFrame([{
        "Store":               req.store_id,
        "Date":                req.date,
        "DayOfWeek":           req.day_of_week,
        "Open":                1,
        "Promo":               req.promo,
        "StateHoliday":        req.state_holiday,
        "SchoolHoliday":       req.school_holiday,
        "StoreType":           req.store_type,
        "Assortment":          req.assortment,
        "CompetitionDistance": req.competition_distance,
        "Promo2":              req.promo2,
    }])


def _fallback_feature_matrix() -> np.ndarray:
    """Return a safe fallback matrix with the expected feature width."""
    if isinstance(registry.train_median, pd.Series) and registry.feature_names:
        fallback = registry.train_median.reindex(registry.feature_names).fillna(0)
        return pd.DataFrame([fallback]).values.astype(float)

    width = len(registry.feature_names) if registry.feature_names else 29
    return np.zeros((1, width), dtype=float)


def _build_request_history_frame(req: PredictionRequest) -> pd.DataFrame:
    """Build a short per-store time series so lag/rolling features exist."""
    if registry.raw_history is None:
        raise RuntimeError("Raw history not loaded.")

    request_date = pd.Timestamp(req.date)
    store_history = registry.raw_history[registry.raw_history["Store"] == req.store_id].copy()
    if store_history.empty:
        raise RuntimeError(f"No historical data available for store {req.store_id}.")

    store_history["Date"] = pd.to_datetime(store_history["Date"])
    store_history = store_history.sort_values("Date")
    history_min = store_history["Date"].min()
    history_max = store_history["Date"].max()

    if request_date < history_min:
        window_start = request_date - pd.Timedelta(days=90)
    elif request_date <= history_max:
        window_start = history_min
    else:
        window_start = max(history_min, request_date - pd.Timedelta(days=90))

    calendar = pd.DataFrame({"Date": pd.date_range(window_start, request_date, freq="D")})
    calendar["Store"] = req.store_id
    calendar["DayOfWeek"] = calendar["Date"].dt.dayofweek + 1
    calendar["Open"] = (calendar["Date"].dt.dayofweek != 6).astype(int)
    calendar["Promo"] = 0
    calendar["StateHoliday"] = "0"
    calendar["SchoolHoliday"] = 0
    calendar["StoreType"] = req.store_type
    calendar["Assortment"] = req.assortment
    calendar["CompetitionDistance"] = req.competition_distance
    calendar["Promo2"] = req.promo2
    calendar["Sales"] = np.nan
    calendar["Customers"] = np.nan

    history_idx = store_history.set_index("Date")
    reindexed = history_idx.reindex(calendar["Date"])
    for col in [
        "Sales",
        "Customers",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "Promo2",
        "DayOfWeek",
    ]:
        if col in reindexed.columns:
            calendar[col] = reindexed[col].combine_first(calendar[col])

    latest = store_history.iloc[-1]
    request_mask = calendar["Date"] == request_date
    calendar.loc[request_mask, "Store"] = req.store_id
    calendar.loc[request_mask, "DayOfWeek"] = req.day_of_week
    calendar.loc[request_mask, "Open"] = 1
    calendar.loc[request_mask, "Promo"] = req.promo
    calendar.loc[request_mask, "StateHoliday"] = req.state_holiday
    calendar.loc[request_mask, "SchoolHoliday"] = req.school_holiday
    calendar.loc[request_mask, "StoreType"] = req.store_type
    calendar.loc[request_mask, "Assortment"] = req.assortment
    calendar.loc[request_mask, "CompetitionDistance"] = (
        req.competition_distance
        if req.competition_distance is not None
        else latest.get("CompetitionDistance")
    )
    calendar.loc[request_mask, "Promo2"] = req.promo2
    calendar.loc[request_mask, "Sales"] = np.nan
    calendar.loc[request_mask, "Customers"] = latest.get("Customers", 0)

    sales_proxy = float(store_history["Sales"].tail(30).mean())
    if np.isnan(sales_proxy):
        sales_proxy = float(registry.raw_history["Sales"].median())

    future_mask = (calendar["Date"] < request_date) & calendar["Sales"].isna()
    calendar.loc[future_mask, "Sales"] = sales_proxy
    calendar["Customers"] = calendar["Customers"].fillna(
        float(store_history["Customers"].median()) if "Customers" in store_history.columns else 0
    )
    calendar["CompetitionDistance"] = calendar["CompetitionDistance"].fillna(
        latest.get("CompetitionDistance")
    )
    calendar["Promo2"] = calendar["Promo2"].fillna(int(latest.get("Promo2", 0)))
    calendar["StoreType"] = calendar["StoreType"].fillna(req.store_type)
    calendar["Assortment"] = calendar["Assortment"].fillna(req.assortment)

    return calendar.sort_values("Date").reset_index(drop=True)


def preprocess_request(req: PredictionRequest) -> np.ndarray:
    """Build request features, apply pipeline, and return feature array."""
    if registry.pipeline is None:
        raise RuntimeError("Preprocessing pipeline not loaded.")

    try:
        history_frame = _build_request_history_frame(req)
        engineered = history_frame.copy()

        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from features.engineer import apply_feature_engineering

        engineered = apply_feature_engineering(engineered)
        request_row = engineered[engineered["Date"] == pd.Timestamp(req.date)].copy()
        if request_row.empty:
            raise RuntimeError("Could not construct request feature row.")

        X = registry.pipeline.transform(
            request_row.drop(columns=["Sales", "Customers"], errors="ignore")
        )
        if isinstance(X, pd.DataFrame):
            X = X.fillna(registry.train_median if registry.train_median is not None else 0)
            return X.values.astype(float)

        return np.asarray(X, dtype=float)
    except Exception as exc:
        logger.warning(f"Falling back to default feature matrix: {exc}")
        return _fallback_feature_matrix()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — used by load balancer."""
    model_ready = registry.accurate_model is not None or registry.fast_model is not None
    return {
        "status":      "healthy" if model_ready else "degraded",
        "loaded_at":   registry.loaded_at,
        "models_ready": model_ready,
    }


@app.get("/model/info")
async def model_info():
    """Model metadata endpoint."""
    return {
        "api_version":    "1.0.0",
        "models_loaded":  {
            "accurate_model": registry.accurate_model is not None,
            "fast_model":     registry.fast_model is not None,
            "ensemble":       registry.ensemble is not None,
            "router":         registry.router is not None,
        },
        "loaded_at":      registry.loaded_at,
        "description":    "XGBoost + Ridge tiered routing with bootstrap uncertainty",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """
    Single prediction with uncertainty.

    Routing:
      If router is available → tiered routing (fast vs accurate)
      Else if accurate model → use XGBoost
      Else if fast model → use Ridge
      Else → 503
    """
    start = time.perf_counter()

    try:
        X = preprocess_request(req)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    lower = upper = uncertainty = None
    model_used = "unknown"

    if registry.router is not None and registry.ensemble is not None:
        try:
            mean, std, lb, ub = registry.ensemble.predict_with_uncertainty(X)
            preds, use_accurate, stds = registry.router.predict(X)
            prediction = float(preds[0])
            uncertainty = float(stds[0])
            lower = float(lb[0])
            upper = float(ub[0])
            model_used = "accurate (XGBoost)" if use_accurate[0] else "fast (Ridge)"
        except Exception:
            prediction, model_used = _fallback_predict(X)
    else:
        prediction, model_used = _fallback_predict(X)

    prediction = max(0.0, round(prediction, 2))
    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    logger.info(f"predict store={req.store_id} date={req.date} "
                f"pred={prediction} model={model_used} latency={latency_ms}ms")

    return PredictionResponse(
        store_id=req.store_id,
        date=req.date,
        predicted_sales=prediction,
        lower_bound=round(lower, 2) if lower else None,
        upper_bound=round(upper, 2) if upper else None,
        uncertainty=round(uncertainty, 2) if uncertainty else None,
        model_used=model_used,
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(batch: BatchRequest):
    """Batch prediction endpoint."""
    start = time.perf_counter()

    if len(batch.requests) > 1000:
        raise HTTPException(status_code=400,
                            detail="Batch size exceeds maximum (1000)")

    results = []
    for req in batch.requests:
        resp = await predict(req)
        results.append(resp)

    total_ms = round((time.perf_counter() - start) * 1000, 3)
    return BatchResponse(predictions=results, total_latency_ms=total_ms)


@app.get("/monitor/psi")
async def get_psi_report():
    """Return latest PSI drift report."""
    psi_path = OUTPUT_DIR / "psi_report.json"
    if not psi_path.exists():
        return {"message": "No PSI report available. Run Phase 6 first."}
    with open(psi_path) as f:
        return {"psi_report": json.load(f), "generated_at": "see file metadata"}


@app.get("/monitor/alerts")
async def get_alerts():
    """Return recent drift alerts."""
    alerts_path = OUTPUT_DIR / "drift_alerts.json"
    if not alerts_path.exists():
        return {"alerts": [], "message": "No alerts recorded."}
    with open(alerts_path) as f:
        return {"alerts": json.load(f)}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fallback_predict(X: np.ndarray) -> tuple:
    """Fallback: use best available model."""
    if registry.accurate_model is not None:
        return float(registry.accurate_model.predict(X)[0]), "accurate (XGBoost)"
    elif registry.fast_model is not None:
        return float(registry.fast_model.predict(X)[0]), "fast (Ridge)"
    else:
        raise HTTPException(status_code=503, detail="No model loaded.")


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.serve:app", host="0.0.0.0", port=8000, reload=True)
