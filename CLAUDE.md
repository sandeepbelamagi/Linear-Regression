# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end sales forecasting system built on Rossmann-style retail data. Covers the full ML lifecycle across 7 phases: EDA, preprocessing, feature engineering, model training, uncertainty estimation, drift detection, and a FastAPI serving layer. Containerized with Docker for production deployment.

## Directory Structure

```
├── data/raw/train.csv                  ← Raw dataset (50 stores, 2 years)
├── src/
│   ├── eda.py                          ← Phase 1: EDA
│   ├── preprocessing/pipeline.py       ← Phase 2: sklearn Pipeline, target encoding
│   ├── features/engineer.py            ← Phase 3: Lag, rolling, trend features
│   ├── models/train.py                 ← Phase 4: LR/Ridge/Lasso/ElasticNet/XGBoost
│   ├── models/uncertainty.py           ← Phase 5: Bootstrap ensemble, tiered routing
│   ├── monitoring/drift.py             ← Phase 6: PSI, KS test, CUSUM
│   └── api/serve.py                    ← Phase 7: FastAPI inference service
├── run_pipeline.py                     ← Entry point for training phases
├── Dockerfile                          ← Multi-stage: training + serving targets
├── docker-compose.yml                  ← train + api services
└── outputs/models/                     ← Saved .pkl model artifacts
```

## Running Locally

```bash
pip install -r requirements.txt

# Full pipeline
python run_pipeline.py

# Specific phases
python run_pipeline.py --phase 1 2 3

# API server (after training)
uvicorn src.api.serve:app --host 0.0.0.0 --port 8000 --reload
```

## Docker Deployment

```bash
# Train models (artifacts written to model-artifacts volume)
docker compose up train

# Start API (reads from model-artifacts volume)
docker compose up api -d

# Or build targets directly
docker build --target training -t sales-forecast-train .
docker build --target serving -t sales-forecast-api .
```

## Architecture

The pipeline flows sequentially — each phase produces artifacts consumed by later phases:

1. **eda.py** — Generates EDA plots to `outputs/`.
2. **preprocessing/pipeline.py** — Custom sklearn transformers (`DropClosedStores`, `GroupMedianImputer`, `KFoldTargetEncoder`, `DateFeatureExtractor`, `StateHolidayEncoder`) composed via `build_preprocessing_pipeline()`. Time-based train/test split (never random). Fit only on train data.
3. **features/engineer.py** — Lag (7/14/30-day), rolling (mean/std), trend (slope), and interaction features, all computed per-store. VIF analysis for multicollinearity. Must run **before** preprocessing since lag/rolling features need the raw `Sales` column.
4. **models/train.py** — Trains 5 models with `TimeSeriesSplit` CV. Primary metric: RMSPE. Saves `.pkl` files to `outputs/models/`.
5. **models/uncertainty.py** — `BootstrapEnsemble` for prediction intervals. `TieredRouter` routes uncertain predictions to XGBoost, confident ones to Ridge.
6. **monitoring/drift.py** — PSI, KS test, and CUSUM `ResidualMonitor` for covariate and concept drift detection.
7. **api/serve.py** — FastAPI with `/predict`, `/predict/batch`, `/health`, `/model/info`, `/monitor/psi`, `/monitor/alerts`. Models loaded once at startup via `ModelRegistry`. Production serving uses gunicorn with uvicorn workers.

## Key Constraints

- **Leakage prevention**: `Customers` and `Sales` are in `LEAKAGE_COLS` and dropped from features. Target encoding uses K-Fold with smoothing. All splits are temporal.
- **Feature engineering ordering**: Lag/rolling features require the `Sales` column, so `apply_feature_engineering()` must be called before `get_X_y()` separates features from target.
- **Preprocessing fit discipline**: `pipeline.fit_transform()` on train only; `pipeline.transform()` on test. `KFoldTargetEncoder` requires `y` during `fit`.
- **Import pattern**: Cross-module imports (e.g. `from preprocessing.pipeline import ...`) rely on `sys.path.insert(0, str(ROOT / "src"))` being called first. This is done in each module's `__main__` block and in `run_pipeline.py`.
