from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_CSV = OUTPUT_DIR / "model_comparison.csv"
PSI_JSON = OUTPUT_DIR / "psi_report.json"
ALERTS_JSON = OUTPUT_DIR / "drift_alerts.json"


def show_missing(path: Path) -> None:
    st.warning(f"Missing: {path}")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    return pd.read_csv(MODEL_CSV)


@st.cache_data(show_spinner=False)
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        show_missing(path)


def show_image_grid(paths: list[Path], title: str) -> None:
    st.subheader(title)
    if not paths:
        st.info("No files found yet.")
        return
    cols = st.columns(2)
    for i, path in enumerate(paths):
        with cols[i % 2]:
            show_image(path, path.name)


st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("Sales Forecasting Dashboard")
st.caption("Visualizes pipeline outputs from phases 1 to 6. No new plots are generated here.")

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

tabs = st.tabs(
    [
        "Sample Data",
        "Model Metrics",
        "Generated Visuals",
        "Monitoring",
    ]
)


with tabs[0]:
    st.header("Raw Sample Data")
    if not DATA_PATH.exists():
        show_missing(DATA_PATH)
    else:
        df = load_data()

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Stores", int(df["Store"].nunique()))
        c3.metric(
            "Date Range",
            f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        )

        store_ids = sorted(df["Store"].unique().tolist())
        selected_stores = st.multiselect(
            "Filter Store IDs",
            store_ids,
            default=store_ids[:3],
        )

        date_min = df["Date"].min().date()
        date_max = df["Date"].max().date()
        date_range = st.date_input(
            "Filter Date Range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        )

        filtered = df.copy()
        if selected_stores:
            filtered = filtered[filtered["Store"].isin(selected_stores)]

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered = filtered[
                (filtered["Date"].dt.date >= start_date)
                & (filtered["Date"].dt.date <= end_date)
            ]

        st.dataframe(filtered.head(200), use_container_width=True)
        st.caption("Showing up to first 200 filtered rows.")


with tabs[1]:
    st.header("Model Comparison")
    if MODEL_CSV.exists():
        metrics_df = load_model_comparison()
        st.dataframe(metrics_df, use_container_width=True)
        if "test_RMSPE" in metrics_df.columns:
            best = metrics_df.sort_values("test_RMSPE").iloc[0]
            st.success(
                f"Best by test_RMSPE: {best['model']} | "
                f"RMSPE={best['test_RMSPE']} | R2={best.get('test_R2', 'NA')}"
            )
    else:
        show_missing(MODEL_CSV)
        st.info("Run Phase 4 first to generate model comparison.")


with tabs[2]:
    st.header("Generated PNG Visuals")

    eda_files = [
        OUTPUT_DIR / "eda_sales_distribution.png",
        OUTPUT_DIR / "eda_temporal_patterns.png",
        OUTPUT_DIR / "eda_feature_analysis.png",
    ]
    show_image_grid(eda_files, "Phase 1: EDA")

    residual_files = sorted(OUTPUT_DIR.glob("residuals_*.png"))
    learning_files = sorted(OUTPUT_DIR.glob("learning_curve_*.png"))
    importance_files = sorted(OUTPUT_DIR.glob("importance_*.png"))

    show_image_grid(residual_files, "Phase 4: Residual Plots")
    show_image_grid(learning_files, "Phase 4: Learning Curves")
    show_image_grid(importance_files, "Phase 4: Feature Importances")

    st.subheader("Phase 5 and 6")
    show_image(OUTPUT_DIR / "uncertainty_routing.png", "uncertainty_routing.png")
    show_image(OUTPUT_DIR / "monitoring_dashboard.png", "monitoring_dashboard.png")


with tabs[3]:
    st.header("Monitoring Reports")

    if PSI_JSON.exists():
        psi = load_json(PSI_JSON)
        psi_df = pd.DataFrame(psi)
        if not psi_df.empty:
            st.subheader("PSI Report")
            if "status" in psi_df.columns:
                selected_status = st.multiselect(
                    "Filter Status",
                    sorted(psi_df["status"].dropna().unique().tolist()),
                    default=sorted(psi_df["status"].dropna().unique().tolist()),
                )
                if selected_status:
                    psi_df = psi_df[psi_df["status"].isin(selected_status)]
            st.dataframe(psi_df, use_container_width=True)
        else:
            st.info("PSI report is empty.")
    else:
        show_missing(PSI_JSON)
        st.info("Run Phase 6 first to generate PSI report.")

    st.subheader("Drift Alerts")
    if ALERTS_JSON.exists():
        alerts = load_json(ALERTS_JSON)
        alerts_df = pd.DataFrame(alerts)
        st.dataframe(alerts_df, use_container_width=True)
    else:
        st.info("No drift_alerts.json found yet.")
