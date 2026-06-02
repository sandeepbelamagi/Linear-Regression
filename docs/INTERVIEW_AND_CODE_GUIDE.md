# Sales Forecasting Project Interview Guide and Code Walkthrough

Draft.ipynb is intentionally excluded from this guide.

This document is written for two use cases:
- Interview prep: how to explain the project like a senior candidate.
- Code reference: what each file, function, class, and major execution block does.

It is not a literal punctuation-by-punctuation annotation of every source line. Instead, it covers every function and every meaningful execution block in code order so you can explain the repository confidently in an interview.

## 30-Second Pitch

This project is an end-to-end sales forecasting system built on retail time-series data. It starts with EDA, then leakage-safe preprocessing, feature engineering, model comparison, uncertainty estimation, drift monitoring, and a FastAPI serving layer. The repository also includes a Streamlit dashboard that visualizes the outputs already produced by the pipeline.

## Senior Recruiter Interview Questions and Answers

### 1. What problem does this project solve?
This project predicts store sales from historical retail data. The business value is not just the point forecast - it also includes uncertainty, drift detection, and a production API so the model can be operationalized rather than treated as a notebook-only experiment.

### 2. Why did you split the work into phases?
Each phase maps to a real ML lifecycle step: understanding data, preventing leakage, engineering features, training and comparing models, quantifying uncertainty, monitoring drift, and serving predictions. That structure makes the project easier to reason about and easier to present in an interview.

### 3. Why is time-based splitting important here?
Because the data is temporal. Random splitting would leak future patterns into training and make validation unrealistically optimistic. Time-based splitting preserves causality: train on the past and evaluate on the future.

### 4. Why did you drop `Customers`?
`Customers` is highly correlated with `Sales`, but it is not available in a real-time prediction context. Using it would create leakage because the model would depend on a value that is only known after the sale happens.

### 5. Why did you engineer lag and rolling features?
Sales forecasting is driven by autocorrelation and recent momentum. Lag features capture what happened in previous periods, while rolling features capture local trend and volatility. Those features are especially important for linear models, which do not discover them automatically.

### 6. Why compare linear models and XGBoost?
The linear models are interpretable baselines. Ridge, Lasso, and Elastic Net help with multicollinearity and regularization. XGBoost is the non-linear benchmark and, in this project, it gives the strongest test performance.

### 7. What metric matters most?
RMSPE matters most because it measures proportional error. In retail, a 1000-unit miss on a small store is more harmful than the same absolute miss on a larger store. RMSPE is closer to the business problem than RMSE alone.

### 8. What is the purpose of uncertainty estimation?
Point forecasts are not enough in production. Uncertainty helps decide whether a fast model is good enough or whether the request should be routed to a more accurate model. It also gives the business a confidence signal.

### 9. How does the routing layer work?
The bootstrap ensemble estimates predictive uncertainty. If uncertainty is low, the request is served by the fast model. If uncertainty is high, it is routed to the more accurate model. That is a latency-versus-accuracy tradeoff.

### 10. What does drift monitoring catch?
It catches both input drift and residual drift. PSI and KS test check whether input distributions changed. CUSUM on residuals checks whether the model has become systematically biased over time.

### 11. What is PSI and how do you interpret it?
PSI compares the reference distribution with the current one. Values below 0.10 are usually stable, 0.10 to 0.25 indicate moderate drift, and above 0.25 usually justify investigation or retraining.

### 12. Why did you build an API?
Because a model is only useful when it can be consumed. The FastAPI layer turns the trained pipeline into an application that can answer predictions, return model metadata, and expose monitoring reports.

### 13. What MLOps pieces are already included?
The project includes reproducible training phases, saved model artifacts, a serving API, monitoring outputs, Docker support, and a lightweight dashboard. It is not a full enterprise MLOps platform, but it includes the core patterns.

### 14. What is missing for a production-grade MLOps stack?
CI/CD, automated retraining orchestration, model registry integration, feature store integration, and alerting infrastructure such as Prometheus or Grafana. Those would be the next hardening steps.

### 15. How would you improve this project next?
I would add automated tests, true out-of-fold target encoding, proper feature store support for lag features, API request validation against feature schemas, and scheduled retraining triggered by drift thresholds.

### 16. How do you explain this project to a non-technical stakeholder?
I would say it forecasts future store sales, tells you how confident the system is, warns when the world changes enough that the model may be stale, and exposes everything through a service that can be used by other applications.

### 17. What is the strongest technical decision in the repo?
The strongest decision is the time-aware design: temporal splitting, lag-based feature engineering, drift monitoring, and route-to-model logic all respect the fact that this is a time-series business problem.

### 18. What is the biggest risk in the current implementation?
The biggest risk is that the project is still more demo-oriented than fully production-hardened. The model and API work, but there is still room for stronger testing, orchestration, and schema enforcement.

## End-to-End Data Flow

1. `data/raw/train.csv` is loaded.
2. `src/eda.py` profiles the data and writes EDA plots.
3. `src/features/engineer.py` adds lag, rolling, trend, and interaction features.
4. `src/preprocessing/pipeline.py` removes leakage, imputes missing values, encodes categoricals, and extracts date features.
5. `src/models/train.py` compares multiple models and saves the best ones.
6. `src/models/uncertainty.py` builds bootstrap uncertainty and routing logic.
7. `src/monitoring/drift.py` computes PSI, KS test, and residual monitoring outputs.
8. `src/api/serve.py` loads the trained artifacts and serves predictions.
9. `frontend/app.py` visualizes the artifacts already produced by the pipeline.

## Code Walkthrough

### `run_pipeline.py`

Purpose: single entry point that runs phases 1 to 7 in order or runs a selected subset.

| Block | What it does | Why it matters |
| --- | --- | --- |
| Imports and `ROOT` setup | Adds `src` to `sys.path`, sets a writable matplotlib cache path, and forces UTF-8 output on Windows. | Makes the repo runnable from the command line without import or encoding issues. |
| `run_phase1()` to `run_phase6()` | Uses `runpy.run_path(..., run_name="__main__")` so each script executes its own main block. | Ensures each phase behaves exactly like a standalone script. |
| `run_phase7()` | Prints the FastAPI startup instructions instead of running a server inside the pipeline. | Keeps the pipeline runner focused on the offline phases while still documenting serving. |
| `PHASES` dictionary | Maps phase numbers to names and functions. | Makes `--phase 1 2 3` style execution easy. |
| Main argument parsing | Reads the selected phases and defaults to all phases when no argument is given. | Makes the script flexible for partial reruns. |
| Phase execution loop | Runs each phase, times it, prints success or failure, and continues to the next phase. | Good for demos because one failing phase does not block later phases. |

Important lines to understand:
- `sys.path.insert(0, str(ROOT / "src"))` allows local imports from the repository source tree.
- `run_name="__main__"` is what makes the phase scripts actually execute their main blocks.
- The `try/except` around each phase is intentionally forgiving for a demo workflow.

### `src/eda.py`

Purpose: summarize the data, visualize patterns, and print modeling implications.

| Function or block | What it does | Why it matters |
| --- | --- | --- |
| Imports and `matplotlib.use("Agg")` | Loads pandas, numpy, seaborn, and configures headless plotting. | Allows plots to be saved in environments without a display. |
| `ROOT`, `DATA_PATH`, `OUTPUT_DIR` | Defines input and output locations. | Keeps paths centralized and portable. |
| `load_data()` | Reads the CSV, parses `Date`, prints shape and date range. | Gives an immediate sanity check on the dataset. |
| `print_summary()` | Prints dtypes, missing values, and sales statistics on open days. | Surfaces data quality issues and target distribution. |
| `plot_sales_distribution()` | Plots raw and log-transformed sales distributions. | Shows skew and motivates `log1p` for linear models. |
| `plot_temporal_patterns()` | Plots monthly trend, day-of-week behavior, and seasonality. | Shows why temporal features are needed. |
| `plot_feature_analysis()` | Plots promo effect, store type comparison, correlation heatmap, and assortment distribution. | Links the data story to feature engineering decisions. |
| `print_eda_findings()` | Prints the interview-ready findings and modeling implications. | Turns EDA into a narrative you can explain in an interview. |
| Main block | Calls the functions in order and writes EDA PNGs to `outputs/`. | Produces the artifacts that the dashboard later displays. |

Key implementation notes:
- The EDA only looks at open days when it is analyzing sales behavior.
- The correlation with `Customers` is explicitly called out as leakage risk.
- The plots are saved, not shown interactively, so they can be reused by Streamlit.

### `src/preprocessing/pipeline.py`

Purpose: create a leakage-safe preprocessing pipeline for training and inference.

| Class or function | What it does | Why it matters |
| --- | --- | --- |
| `DropClosedStores` | Removes rows where `Open == 0` and drops the `Open` column. | Closed stores always have zero sales and can distort training. |
| `GroupMedianImputer` | Fills missing numeric values using the median by group, with a global fallback. | Better than a global constant when store context matters. |
| `KFoldTargetEncoder` | Encodes categorical columns using smoothed target means. | Converts high-cardinality categories into numeric signals. |
| `DateFeatureExtractor` | Adds year, month, week, quarter, weekend flag, and cyclical encodings. | Extracts time structure that plain regression would not see. |
| `StateHolidayEncoder` | Converts `StateHoliday` to binary. | Simplifies a categorical holiday indicator into one numeric flag. |
| `build_preprocessing_pipeline()` | Chains the custom transformers in the right order. | Centralizes preprocessing logic in one reusable pipeline. |
| `time_based_split()` | Splits data by date, not randomly. | Prevents future leakage in time series evaluation. |
| `get_X_y()` | Separates features from target and removes leakage columns. | Keeps training and inference feature sets consistent. |

Important line-level ideas:
- The pipeline is ordered so date extraction happens before target encoding.
- `LEAKAGE_COLS` explicitly excludes `Customers` and `Sales`.
- The split cutoff is based on `max(Date) - test_months`, which gives a future holdout window.
- In the main block, the pipeline is fit on open-store targets only. That alignment is critical after dropping closed stores.

Note on interview honesty:
- The class is named `KFoldTargetEncoder`, but the implementation is a smoothed target encoder fit on the training data. If asked, say that the production-safe version should be a true out-of-fold encoder or be wrapped inside a proper CV routine.

### `src/features/engineer.py`

Purpose: create lag, rolling, trend, and interaction features before preprocessing.

| Class or function | What it does | Why it matters |
| --- | --- | --- |
| `LagFeatureBuilder` | Sorts by store and date, then adds lagged sales columns. | Captures autocorrelation and weekly/monthly memory. |
| `RollingFeatureBuilder` | Adds rolling mean and rolling std per store using shifted history. | Captures recent trend and volatility without leakage. |
| `TrendFeatureBuilder` | Fits a local slope over a rolling window. | Gives the model a trend signal beyond raw lags. |
| `InteractionFeatureBuilder` | Creates promo x weekend, promo x month, promo x holiday, and momentum features. | Adds non-linear business relationships explicitly for linear models. |
| `compute_vif()` | Calculates variance inflation factor for numeric features. | Helps identify multicollinearity. |
| `drop_high_vif_features()` | Iteratively removes high-VIF columns. | Useful when linear coefficients become unstable. |
| `build_feature_pipeline()` | Returns the list of feature transformers in order. | Centralizes engineering logic. |
| `apply_feature_engineering()` | Applies all feature transformers sequentially. | Gives a simple one-call interface for downstream phases. |

Important line-level ideas:
- Every lag or rolling transform sorts by `Store` and `Date` before shifting.
- The rolling features use `shift(1)` so the current row does not leak into the statistic.
- `sales_momentum` is defined as recent rolling mean minus the last lag value.
- `compute_vif()` uses numeric columns only and drops missing rows before calculation.

Main block behavior:
- Loads the raw CSV.
- Builds engineered features.
- Prints a sample for store 1.
- Computes VIF on selected numeric columns.
- Saves no model, only prints diagnostics for the phase.

### `src/models/train.py`

Purpose: compare multiple regression models, evaluate them with time-aware validation, and save artifacts.

| Function or block | What it does | Why it matters |
| --- | --- | --- |
| `rmspe()` | Computes root mean squared percentage error. | Matches the retail forecasting objective better than plain RMSE. |
| `evaluate()` | Returns RMSE, MAE, R2, RMSPE, and MAPE. | Gives a broader evaluation than a single metric. |
| `get_models()` | Builds Linear Regression, Ridge, Lasso, Elastic Net, and XGBoost. | Creates the model comparison set. |
| `time_series_cv()` | Uses `TimeSeriesSplit` to cross-validate models. | Prevents future leakage in cross-validation. |
| `train_all_models()` | Trains each model, evaluates it, and saves it as a pickle. | Produces a comparable leaderboard and persisted artifacts. |
| `plot_residuals()` | Draws residual plots, QQ plot, histogram, and actual vs predicted. | Diagnoses heteroscedasticity and fit quality. |
| `plot_learning_curve()` | Measures train and validation RMSE across increasing data sizes. | Helps identify bias versus variance. |
| `plot_feature_importance()` | Visualizes coefficients or feature importances. | Explains what the model is using. |

Important line-level ideas:
- All models except XGBoost are wrapped in a scaler plus estimator pipeline.
- `TimeSeriesSplit` is used instead of KFold to preserve chronology.
- `train_all_models()` sorts results by `test_RMSPE`, not by R2.
- Models are saved under `outputs/models/` using a sanitized name.
- The residual plot and learning curve are saved as PNGs for later inspection.

Main block behavior:
- Loads raw data.
- Runs feature engineering first because lag features need `Sales`.
- Splits train and test by date.
- Fits the preprocessing pipeline.
- Aligns feature arrays and label vectors after dropping closed stores.
- Trains all models and writes `outputs/model_comparison.csv`.
- Runs residual, learning curve, and feature importance plots for the best model.

### `src/models/uncertainty.py`

Purpose: estimate predictive uncertainty and route requests to a fast or accurate model.

| Class or function | What it does | Why it matters |
| --- | --- | --- |
| `BootstrapEnsemble` | Trains many bootstrap replicas of a base model. | Produces a distribution of predictions instead of a single number. |
| `BootstrapEnsemble._default_model()` | Defines a Ridge-based pipeline used by default. | Gives a lightweight uncertainty model. |
| `BootstrapEnsemble.fit()` | Fits each bootstrap model on a resampled dataset. | Captures model variance. |
| `BootstrapEnsemble.predict()` | Returns the mean prediction. | Provides a stable point estimate. |
| `BootstrapEnsemble.predict_with_uncertainty()` | Returns mean, std, and percentile bounds. | Supplies confidence-style outputs for routing and UI display. |
| `TieredRouter` | Uses uncertainty to choose between fast and accurate models. | Implements the latency-versus-accuracy tradeoff. |
| `TieredRouter.fit_threshold()` | Chooses an uncertainty threshold from validation predictions. | Controls how many requests go to the accurate model. |
| `TieredRouter.predict()` | Routes each sample to the fast or accurate model. | Makes inference adaptive. |
| `TieredRouter.routing_stats()` | Summarizes the routing decision distribution. | Useful for reporting and monitoring. |
| `plot_uncertainty()` | Visualizes prediction intervals and routing decisions. | Makes uncertainty easier to communicate. |

Important line-level ideas:
- The ensemble trains on bootstrap samples with replacement.
- Routing is based on uncertainty, not on the target value itself.
- `fit_threshold()` uses a percentile to decide the split between fast and accurate paths.
- The main block loads the XGBoost model if available, otherwise trains one.
- The router and ensemble are saved to `outputs/models/`.

### `src/monitoring/drift.py`

Purpose: detect covariate drift and residual drift, then create a monitoring dashboard.

| Function or class | What it does | Why it matters |
| --- | --- | --- |
| `compute_psi()` | Computes Population Stability Index for one feature. | Measures how much a distribution shifted. |
| `compute_psi_report()` | Runs PSI across multiple columns and labels the severity. | Gives a compact drift report. |
| `ks_drift_test()` | Runs a Kolmogorov-Smirnov test for each feature. | Adds a statistical drift check alongside PSI. |
| `ResidualMonitor` | Tracks residual batches and runs CUSUM-based drift detection. | Detects systematic bias in model predictions. |
| `ResidualMonitor.update()` | Updates residual history, CUSUM, and raises alerts when thresholds are crossed. | Core runtime monitoring logic. |
| `ResidualMonitor.get_summary()` | Returns a compact summary of residual monitoring. | Useful for reporting and logging. |
| `simulate_concept_drift()` | Artificially weakens promo effect after a chosen date. | Useful for demonstrating that monitoring works. |
| `plot_drift_monitoring()` | Draws residual trend and PSI dashboard plots. | Produces the main monitoring visual. |

Important line-level ideas:
- PSI is computed against reference bins derived from the training distribution.
- The KS test compares the full distribution shape, not just summary stats.
- CUSUM accumulates persistent bias instead of reacting to a single noisy batch.
- The main block creates both normal and drifted datasets, then compares them.
- `psi_report.json` is saved for the API and dashboard to read later.

Main block behavior:
- Loads raw data.
- Simulates concept drift in a second copy of the dataset.
- Applies feature engineering to both versions.
- Fits the preprocessing pipeline on the training slice.
- Computes PSI and KS reports.
- If the XGBoost model exists, runs monthly residual monitoring and saves `monitoring_dashboard.png`.

### `src/api/serve.py`

Purpose: expose the trained models and monitoring outputs through a FastAPI service.

| Function or block | What it does | Why it matters |
| --- | --- | --- |
| Logging and UTF-8 setup | Forces UTF-8 output and a writable matplotlib cache. | Prevents Windows console and cache issues. |
| `ModelRegistry` | Stores loaded models, the preprocessing pipeline, and metadata. | Avoids reloading models on every request. |
| `load_models()` | Startup hook that rebuilds preprocessing, loads models, and registers pickle aliases. | Makes the API ready to serve real requests. |
| `PredictionRequest` | Validates input JSON for a single prediction. | Prevents malformed requests from reaching the model. |
| `PredictionResponse` | Defines the response schema for predictions. | Makes the API response consistent. |
| `BatchRequest` and `BatchResponse` | Define the batch prediction payloads. | Supports bulk inference. |
| `request_to_dataframe()` | Converts a request into the feature layout expected by preprocessing. | Bridges API input and model input. |
| `_fallback_feature_matrix()` | Builds a safe fallback matrix with the expected width. | Prevents hard crashes when a transform path fails. |
| `_build_request_history_frame()` | Reconstructs a short per-store history so lag features can be computed for inference. | This is the critical step that makes phase 7 functional. |
| `preprocess_request()` | Applies feature engineering and preprocessing to a single request. | Turns API input into model-ready features. |
| `/health` endpoint | Returns service health and model readiness. | Used by load balancers and humans. |
| `/model/info` endpoint | Returns model loading status and metadata. | Useful for debugging and observability. |
| `/predict` endpoint | Returns a single prediction with uncertainty and model choice. | Core inference endpoint. |
| `/predict/batch` endpoint | Runs multiple predictions with a batch size guard. | Useful for bulk scoring. |
| `/monitor/psi` endpoint | Serves the latest PSI report from disk. | Exposes monitoring to the API. |
| `/monitor/alerts` endpoint | Serves drift alerts from disk. | Lets consumers inspect alert history. |
| `_fallback_predict()` | Uses the best available loaded model if routing is unavailable. | Keeps the service usable even when some artifacts are missing. |

Important line-level ideas:
- Startup loads raw historical data and rebuilds the preprocessing pipeline from the same data context used in training.
- The code registers `BootstrapEnsemble` and `TieredRouter` on `__main__` so old pickles created in script mode can still be loaded.
- The request-time feature builder is what makes lag-based modeling work in the API.
- If the router is available, the API returns both prediction and uncertainty details.
- The batch endpoint simply loops over the single-request prediction path.

### `frontend/app.py`

Purpose: visualize the artifacts already produced by the pipeline. It does not generate new model outputs.

| Function or block | What it does | Why it matters |
| --- | --- | --- |
| `show_missing()` | Displays a warning when a file is absent. | Makes the dashboard robust when a phase has not been run yet. |
| `load_data()` | Loads `data/raw/train.csv` with parsed dates. | Powers the sample data viewer. |
| `load_model_comparison()` | Reads `outputs/model_comparison.csv`. | Shows phase 4 metrics in the UI. |
| `load_json()` | Reads a JSON artifact from disk. | Used for PSI and drift alerts. |
| `show_image()` | Displays a PNG if it exists. | Lets the dashboard reuse saved plots. |
| `show_image_grid()` | Renders a set of images in a two-column layout. | Keeps the UI compact and readable. |
| Streamlit tab layout | Splits the UI into sample data, metrics, visuals, and monitoring. | Keeps the dashboard simple and practical. |

Important line-level ideas:
- The dashboard reads from `outputs/` and `data/raw/` only.
- It uses caching for faster reloads.
- It is intentionally artifact-driven so it stays in sync with the pipeline outputs.

## Support Files

### `requirements.txt`
- Lists the Python dependencies needed for data processing, modeling, monitoring, API serving, and the Streamlit UI.
- The important packages are `pandas`, `numpy`, `scikit-learn`, `xgboost`, `fastapi`, `uvicorn`, `joblib`, `scipy`, `gunicorn`, and `streamlit`.

### `Dockerfile`
- Defines a shared base layer for dependencies.
- Has a training stage that runs `run_pipeline.py`.
- Has a serving stage that runs `gunicorn` with Uvicorn workers.
- Includes a health check endpoint.

### `docker-compose.yml`
- Creates a training service and an API service.
- Shares the model artifact volume between the two services.
- Exposes port `8000` for the API.

### `.dockerignore`
- Excludes local files and generated artifacts from Docker build context.
- Keeps the image smaller and avoids copying `outputs/`, virtual environments, and editor noise.

### `.gitignore`
- Ignores bytecode, virtual environments, tool caches, generated outputs, and model pickle artifacts.
- Prevents local execution artifacts from being committed.

### `README.md`
- Describes the project structure, the phase-based pipeline, and the basic run commands.
- Useful as a short external-facing summary, while this document is the deeper internal guide.

## What To Say If Asked About MLOps

Included already:
- Reproducible pipeline phases
- Artifact persistence
- Serving API
- Monitoring artifacts
- Docker support
- Streamlit visualization layer

Not yet fully included:
- CI/CD
- Orchestration for retraining
- Feature store
- Model registry
- Formal automated testing
- Production alerting stack

## Best Interview Framing

If you want to sound strong in an interview, use this structure:
1. Problem and business value.
2. Data leakage risk and how you prevented it.
3. Feature engineering choices.
4. Model comparison and metric choice.
5. Uncertainty and routing.
6. Monitoring and drift detection.
7. Serving and dashboarding.
8. What you would harden next.

## Quick Run Sequence

```powershell
.\.venv313\Scripts\python.exe run_pipeline.py --phase 1 2 3 4 5 6
.\.venv313\Scripts\python.exe -m uvicorn src.api.serve:app --host 0.0.0.0 --port 8000 --reload
streamlit run frontend/app.py
```

## Final Interview One-Liner

I built a time-series sales forecasting system that goes from EDA to leakage-safe preprocessing, feature engineering, model comparison, uncertainty-aware routing, drift monitoring, and production serving, with a lightweight dashboard to surface the outputs.
