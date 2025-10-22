# app.py
# Remade safe Streamlit app for FX Flow Monitoring
# - Uses safe try/except imports to avoid ModuleNotFoundError crashes
# - Gives friendly UI error messages if dependencies or data files are missing
# - Expects CSV data files in repo root:
#     - inflow_outflow_long_cleaned.csv
#     - inflow_outflow_quarterly_agg.csv
#
# Run: streamlit run app.py

import streamlit as st

# ---------------------------
# Safe import block
# ---------------------------
missing_core = []
_plotly_ok = False
_pandas_ok = False
_numpy_ok = False
_sklearn_ok = False

# pandas
try:
    import pandas as pd
    _pandas_ok = True
except Exception as e:
    missing_core.append(("pandas", str(e)))

# numpy
try:
    import numpy as np
    _numpy_ok = True
except Exception as e:
    missing_core.append(("numpy", str(e)))

# plotly (required for charts)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _plotly_ok = True
except Exception as e:
    # Capture the exception message for display (helpful for debugging)
    _plotly_err_msg = str(e)
    _plotly_ok = False

# sklearn for simple linear regression forecasting (optional fallback implemented)
try:
    from sklearn.linear_model import LinearRegression
    _sklearn_ok = True
except Exception as e:
    _sklearn_ok = False
    _sklearn_err_msg = str(e)

# If any core libs missing, show friendly message and stop.
if not (_pandas_ok and _numpy_ok and _plotly_ok):
    st.set_page_config(page_title="FX Flow Monitoring - Error", layout="wide")
    st.title("FX Flow Monitoring — startup error")
    if not _pandas_ok or not _numpy_ok:
        st.error("Core Python packages missing or failed to import.")
        st.write("Missing or failed imports:")
        if not _pandas_ok:
            st.write("- pandas (required)")
        if not _numpy_ok:
            st.write("- numpy (required)")
        st.write("\nMake sure your `requirements.txt` includes the packages and redeploy.")
        st.write("Recommended minimal `requirements.txt` entries:")
        st.code("streamlit==1.38.0\npandas==2.2.2\nnumpy==1.26.4\nplotly==5.24.1\nscikit-learn==1.5.2")
        st.stop()
    # plotly missing
    if not _plotly_ok:
        st.error("plotly failed to import — plotting library is required for this app.")
        st.write("Possible causes:")
        st.write("• `plotly` missing from requirements.txt; or")
        st.write("• installation failed during build (check build logs in Streamlit Cloud -> Manage app -> Logs).")
        st.write("Captured import error (for debugging):")
        st.code(_plotly_err_msg)
        st.write("\nRecommended action: ensure `plotly==5.24.1` is present in `requirements.txt`, commit & redeploy.")
        st.stop()

# ---------------------------
# Normal imports after safe check
# ---------------------------
from datetime import datetime
from pathlib import Path

st.set_page_config(layout="wide", page_title="FX Flow Monitoring", initial_sidebar_state="expanded")

# ---------------------------
# Data loading with friendly messages
# ---------------------------
DATA_LONG = "inflow_outflow_long_cleaned.csv"
DATA_Q = "inflow_outflow_quarterly_agg.csv"

@st.cache_data
def load_data(path_long: str, path_q: str):
    errors = []
    d_long = pd.DataFrame()
    d_q = pd.DataFrame()
    try:
        d_long = pd.read_csv(path_long, parse_dates=["period_ts"])
    except FileNotFoundError:
        errors.append(f"File not found: {path_long}")
    except Exception as e:
        errors.append(f"Failed to read {path_long}: {e}")

    try:
        d_q = pd.read_csv(path_q, parse_dates=["period_ts"])
    except FileNotFoundError:
        errors.append(f"File not found: {path_q}")
    except Exception as e:
        errors.append(f"Failed to read {path_q}: {e}")

    # Attempt minimal sanitization if data loaded
    if not d_long.empty:
        if "value_signed_musd" in d_long.columns:
            d_long["value_signed_musd"] = pd.to_numeric(d_long["value_signed_musd"], errors="coerce")
        # Ensure item_simple and flow_class exist
        if "item_simple" not in d_long.columns:
            d_long["item_simple"] = d_long.columns[0] if len(d_long.columns) > 0 else "unknown"
        if "flow_class" not in d_long.columns:
            d_long["flow_class"] = "unknown"

    if not d_q.empty:
        for col in d_q.columns:
            if col != "period_ts":
                d_q[col] = pd.to_numeric(d_q[col], errors="coerce")

    return d_long, d_q, errors

d_long, d_q, load_errors = load_data(DATA_LONG, DATA_Q)

if load_errors:
    st.title("FX Flow Monitoring — data load error")
    st.error("One or more data files failed to load.")
    for e in load_errors:
        st.write("- " + e)
    st.write("\nMake sure the CSV files are committed to the repository and paths are correct.")
    st.write("Expected filenames (in repo root):")
    st.code(f"{DATA_LONG}\n{DATA_Q}")
    st.stop()

# Ensure we have at least minimal data structure to proceed
if d_long.empty and d_q.empty:
    st.title("FX Flow Monitoring — no data available")
    st.error("Both datasets are empty. Please provide the required CSV files and redeploy.")
    st.stop()

# ---------------------------
# Helper functions
# ---------------------------
def rolling_mean(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def linear_forecast_one_step(df_q, target_col="total_inflow_musd"):
    # Simple linear trend model (requires scikit-learn)
    df = df_q.dropna(subset=[target_col]).sort_values("period_ts").reset_index(drop=True)
    if df.shape[0] < 3 or not _sklearn_ok:
        # fallback: return last value as naive forecast
        last = df[target_col].iloc[-1] if not df.empty else 0.0
        return float(last), float(last), float(last), None
    X = np.arange(len(df)).reshape(-1,1)
    y = df[target_col].values.reshape(-1,1)
    model = LinearRegression().fit(X, y)
    next_X = np.array([[len(df)]])
    pred = float(model.predict(next_X)[0,0])
    resid = y.flatten() - model.predict(X).flatten()
    se = resid.std(ddof=1)/np.sqrt(len(resid)) if len(resid) > 1 else resid.std(ddof=0)
    ci = 1.96 * se
    return pred, pred - ci, pred + ci, model

# ---------------------------
# App layout & pages
# ---------------------------
st.sidebar.title("FX Flow Monitoring")
page = st.sidebar.radio("Pilih halaman:", ["Overview", "Breakdown", "Compliance", "Forecast", "Data Preview"])

# Prepare quarterly df sorted
if not d_q.empty and "period_ts" in d_q.columns:
    d_q = d_q.sort_values("period_ts").reset_index(drop=True)

latest_q = d_q["period_ts"].max() if not d_q.empty else None
latest_row = d_q.loc[d_q["period_ts"] == latest_q].iloc[0] if (latest_q is not None and not d_q.empty) else None

# ---------- Overview ----------
if page == "Overview":
    st.title("Overview — Foreign Exchange Inflow / Outflow")
    if latest_row is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest quarter", latest_q.strftime("%Y-%m-%d"))
        c2.metric("Total Inflow (latest)", f"{int(latest_row.get('total_inflow_musd', 0)):,} USD")
        c3.metric("Total Outflow (latest)", f"{int(latest_row.get('total_outflow_musd', 0)):,} USD")
        c4.metric("Net Flow (latest)", f"{int(latest_row.get('net_flow_signed_musd', 0)):,} USD")
    else:
        st.warning("Quarterly aggregated dataset tersedia tapi tidak ada baris dengan nilai valid.")

    st.markdown("### Quarterly time series")
    if not d_q.empty:
        fig = go.Figure()
        if "total_inflow_musd" in d_q.columns:
            fig.add_trace(go.Scatter(x=d_q["period_ts"], y=d_q["total_inflow_musd"], mode="lines+markers", name="Total Inflow"))
        if "total_outflow_musd" in d_q.columns:
            fig.add_trace(go.Scatter(x=d_q["period_ts"], y=d_q["total_outflow_musd"], mode="lines+markers", name="Total Outflow"))
        fig.update_layout(legend=dict(y=0.99, x=0.01))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aggregate quarterly dataset kosong — tidak ada timeseries yang dapat ditampilkan.")

    st.markdown("### Net flow and rolling average (4 quarters)")
    if "net_flow_signed_musd" in d_q.columns:
        d_q["rolling4_net"] = d_q["net_flow_signed_musd"].rolling(window=4, min_periods=1).mean()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=d_q["period_ts"], y=d_q["net_flow_signed_musd"], name="Net Flow"))
        fig2.add_trace(go.Line(x=d_q["period_ts"], y=d_q["rolling4_net"], name="Rolling4 Avg", line=dict(width=3)))
        st.plotly_chart(fig2, use_container_width=True)

    if "net_flow_signed_musd" in d_q.columns:
        d_q["net_yoy_pct"] = d_q["net_flow_signed_musd"].pct_change(periods=4)
        st.markdown("### YoY Growth of Net Flow (percent)")
        st.line_chart(d_q.set_index("period_ts")["net_yoy_pct"].dropna())

# ---------- Breakdown ----------
elif page == "Breakdown":
    st.title("Breakdown — by sector and flow_class")
    st.markdown("Filter the time range and flow direction to inspect components.")

    min_date = st.date_input("Start date", d_long["period_ts"].min().date() if not d_long.empty else datetime(2010,1,1).date())
    max_date = st.date_input("End date", d_long["period_ts"].max().date() if not d_long.empty else datetime.today().date())
    flow_options = d_long["flow_class"].unique().tolist() if "flow_class" in d_long.columns else []
    flow_sel = st.multiselect("Flow class", options=flow_options, default=flow_options)

    mask = (d_long["period_ts"].dt.date >= min_date) & (d_long["period_ts"].dt.date <= max_date)
    if flow_sel:
        mask &= d_long["flow_class"].isin(flow_sel)
    dff = d_long.loc[mask].copy()

    if dff.empty:
        st.info("Tidak ada data untuk filter yang dipilih.")
    else:
        st.markdown("#### Stacked contributions by item")
        stacked = dff.groupby(["period_ts","item_simple"], as_index=False)["value_signed_musd"].sum()
        fig = px.bar(stacked, x="period_ts", y="value_signed_musd", color="item_simple", title="Contributions by Item (stacked)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top items (aggregate over selected period)")
        agg = dff.groupby("item_simple", as_index=False)["value_signed_musd"].sum().sort_values("value_signed_musd", ascending=False).head(20)
        fig2 = px.bar(agg, x="value_signed_musd", y="item_simple", orientation="h", title="Top items by sum(value)")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Matrix (pivot-like)")
        pivot = dff.pivot_table(index="item_simple", columns="period_ts", values="value_signed_musd", aggfunc="sum").fillna(0)
        st.dataframe(pivot.style.format("{:,.0f}"))

# ---------- Compliance ----------
elif page == "Compliance":
    st.title("Compliance — missing data & anomaly detection")
    st.markdown("Flags for missing data and z-score based anomaly detection by item")

    if "value_signed_musd" not in d_long.columns:
        st.warning("Kolom 'value_signed_musd' tidak ditemukan di dataset long. Pastikan file yang benar di-upload.")
    else:
        missing_summary = d_long[d_long["value_signed_musd"].isna()].groupby("item_simple").size().reset_index(name="missing_count").sort_values("missing_count", ascending=False)
        st.subheader("Missing data summary (per item)")
        if not missing_summary.empty:
            st.dataframe(missing_summary)
        else:
            st.write("No missing values found in `value_signed_musd`.")

        st.subheader("Anomaly detection (z-score within each item)")
        dfz = d_long.dropna(subset=["value_signed_musd"]).copy()
        dfz["zscore"] = dfz.groupby("item_simple")["value_signed_musd"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0))
        threshold = st.slider("Z-score threshold", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
        anomalies = dfz.loc[dfz["zscore"].abs() > threshold].sort_values("zscore", ascending=False)
        st.write(f"Detected {len(anomalies)} anomalies (|z| > {threshold})")
        if not anomalies.empty:
            st.dataframe(anomalies[["period_ts","item_simple","flow_class","value_signed_musd","zscore"]].reset_index(drop=True))
            top_anom_items = anomalies["item_simple"].value_counts().head(6).index.tolist()
            for item in top_anom_items:
                subset = dfz[dfz["item_simple"]==item].sort_values("period_ts")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=subset["period_ts"], y=subset["value_signed_musd"], mode="lines+markers", name=item))
                anom_pts = anomalies[anomalies["item_simple"]==item]
                fig.add_trace(go.Scatter(x=anom_pts["period_ts"], y=anom_pts["value_signed_musd"], mode="markers", marker=dict(color="red", size=10), name="Anomaly"))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No anomalies detected with the current threshold.")

# ---------- Forecast ----------
elif page == "Forecast":
    st.title("Forecast — simple 1-quarter-ahead forecast (linear trend)")
    st.markdown("Lightweight linear-trend forecast with approximate 95% CI. For production, use Prophet/ETS/ARIMA.")
    metric_options = [c for c in ["total_inflow_musd","total_outflow_musd","net_flow_signed_musd"] if c in d_q.columns]
    if not metric_options:
        st.warning("Aggregate metrics for forecasting not found in quarterly dataset.")
    else:
        metric_choice = st.selectbox("Forecast target", options=metric_options, index=0)
        pred, low, high, model = linear_forecast_one_step(d_q, target_col=metric_choice)
        last_val = d_q[metric_choice].iloc[-1] if not d_q.empty else np.nan
        st.metric(f"Forecast 1-quarter ahead for {metric_choice}", f"{pred:,.0f} USD", delta=f"{(pred - last_val):,.0f}")
        st.write(f"Approx. 95% CI: [{low:,.0f}, {high:,.0f}]")
        # Plot
        hist = d_q.sort_values("period_ts").reset_index(drop=True).copy()
        if not hist.empty:
            next_period = hist["period_ts"].max() + pd.offsets.QuarterEnd(0)
            plot_df = hist.copy()
            fig = px.line(plot_df, x="period_ts", y=metric_choice, title=f"{metric_choice} history + 1q forecast")
            fig.add_trace(go.Scatter(x=[next_period], y=[pred], mode="markers", marker=dict(size=10), name="Forecast"))
            fig.add_trace(go.Scatter(x=[next_period, next_period], y=[low, high], mode="lines", line=dict(width=3, dash="dash"), name="CI"))
            st.plotly_chart(fig, use_container_width=True)

# ---------- Data Preview ----------
elif page == "Data Preview":
    st.title("Data preview")
    st.subheader("Long cleaned dataset (head)")
    st.dataframe(d_long.head(200))
    st.subheader("Quarterly aggregated dataset (head)")
    st.dataframe(d_q.head(200))

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("Prototype — Monitoring + Compliance + Forecast\nReplace forecasting with more advanced models for production.")
