# Sales Forecasting — Production-Grade ML Project

A FAANG-level end-to-end sales forecasting system built on Rossmann-style retail data.
Covers the full ML lifecycle: EDA → preprocessing → feature engineering → model training
→ uncertainty estimation → drift detection → production API.

---

## Folder Structure

```
sales-forecasting/
│
├── data/
│   ├── raw/
│   │   └── train.csv              ← Raw dataset (50 stores, 2 years)
│   └── processed/                 ← Preprocessed data (generated at runtime)
│
├── src/
│   ├── eda.py                     ← Phase 1: EDA & findings
│   │
│   ├── preprocessing/
│   │   └── pipeline.py            ← Phase 2: Leakage-safe sklearn Pipeline
│   │                                  DropClosedStores, GroupMedianImputer,
│   │                                  KFoldTargetEncoder, DateFeatureExtractor
│   │
│   ├── features/
│   │   └── engineer.py            ← Phase 3: Lag, rolling, trend, interaction features
│   │                                  VIF analysis for multicollinearity
│   │
│   ├── models/
│   │   ├── train.py               ← Phase 4: LR / Ridge / Lasso / ElasticNet / XGBoost
│   │   │                              TimeSeriesSplit CV, RMSPE, residual plots
│   │   └── uncertainty.py         ← Phase 5: Bootstrap ensemble + tiered routing
│   │
│   ├── monitoring/
│   │   └── drift.py               ← Phase 6: PSI, KS test, CUSUM residual monitor
│   │
│   └── api/
│       └── serve.py               ← Phase 7: FastAPI inference service
│
├── outputs/                        ← All plots, model .pkl files, reports
│   └── models/                    ← Saved model artifacts
│
├── tests/                         ← Unit tests (add as you build)
├── run_pipeline.py                ← Entry point: run all or specific phases
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline
python run_pipeline.py

# 3. Run specific phases
python run_pipeline.py --phase 1 2 3   # EDA + preprocessing + features
python run_pipeline.py --phase 4       # Model training only

# 4. Start API server (after running phases 1-6)
uvicorn src.api.serve:app --host 0.0.0.0 --port 8000 --reload

# 5. Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"store_id": 1, "date": "2024-01-15", "day_of_week": 1,
       "promo": 1, "store_type": "a", "assortment": "basic"}'
```

---

## Phase-by-Phase Overview

| Phase | File | What it does | Key concepts |
|-------|------|-------------|--------------|
| 1 | `src/eda.py` | EDA, distributions, seasonality, correlations | Leakage risk identification |
| 2 | `src/preprocessing/pipeline.py` | sklearn Pipeline, K-Fold target encoding, time split | Leakage prevention, GroupMedianImputer |
| 3 | `src/features/engineer.py` | Lag/rolling/trend features, VIF analysis | Autocorrelation, multicollinearity |
| 4 | `src/models/train.py` | 5 models, TimeSeriesSplit CV, RMSPE, residual analysis | Model selection, RMSPE, heteroscedasticity |
| 5 | `src/models/uncertainty.py` | Bootstrap ensemble, prediction intervals, tiered router | Uncertainty quantification, latency/accuracy tradeoff |
| 6 | `src/monitoring/drift.py` | PSI, KS test, CUSUM residual monitor | Covariate drift, concept drift, CUSUM |
| 7 | `src/api/serve.py` | FastAPI REST endpoint with routing and uncertainty | Production serving, Pydantic, async |

---

## Key Design Decisions (FAANG Interview Ready)

**Why time-based split, not random split?**
Random split leaks future information into training. For time-series data, always split temporally: train on past, validate on future.

**Why K-Fold target encoding with smoothing?**
Simple mean encoding leaks target information. K-Fold encodes each fold using other folds. Smoothing `(n * cat_mean + m * global_mean) / (n + m)` prevents rare categories from getting extreme values.

**Why drop Customers feature?**
Customers has 0.924 correlation with Sales but is unknown at real-time serving. Using it creates data leakage — you'd need to predict customers first, which is circular.

**Why RMSPE instead of RMSE?**
RMSPE penalises proportional errors. A 1000-unit error on a 2000-unit store (50%) is worse than the same error on a 10000-unit store (10%). Business-aligned metrics matter.

**Why bootstrap ensemble for uncertainty?**
Confidence intervals from a single model are unreliable. Bootstrap ensemble captures epistemic uncertainty (model uncertainty) by training on random subsets and measuring prediction variance.

**How does tiered routing work?**
Uncertainty > threshold → route to accurate (slow) XGBoost model.
Uncertainty ≤ threshold → route to fast Ridge model.
Reduces p50 latency while maintaining accuracy for uncertain predictions.

**What is PSI and when do you use it?**
Population Stability Index measures how much a feature distribution has shifted.
PSI < 0.10: stable. 0.10–0.25: moderate drift. > 0.25: retrain.
Run daily on all input features in production.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check for load balancer |
| GET | `/model/info` | Model metadata and versions |
| POST | `/predict` | Single prediction with uncertainty |
| POST | `/predict/batch` | Batch predictions (max 1000) |
| GET | `/monitor/psi` | Latest PSI drift report |
| GET | `/monitor/alerts` | Recent CUSUM drift alerts |

Interactive docs: `http://localhost:8000/docs`
