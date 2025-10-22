# streamlit_app.py
# FX Flow Monitoring — Streamlit prototype
# Expects CSVs:
# - /mnt/data/inflow_outflow_long_cleaned.csv
# - /mnt/data/inflow_outflow_quarterly_agg.csv
#
# Run: pip install streamlit pandas numpy plotly scikit-learn
#      streamlit run streamlit_app.py

# di paling atas file streamlit_app.py
import importlib
import streamlit as st

if importlib.util.find_spec("plotly.express") is None:
    st.error("Package `plotly` belum terinstall pada runtime. "
             "Pastikan `plotly` tercantum pada requirements.txt dan redeploy. "
             "Untuk debug lokal: pip install plotly")
    st.stop()

# jika ada, import normal
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide", page_title="FX Flow Monitoring", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    # Update paths if necessary
    d_long = pd.read_csv("/mnt/data/inflow_outflow_long_cleaned.csv", parse_dates=["period_ts"])
    d_q = pd.read_csv("/mnt/data/inflow_outflow_quarterly_agg.csv", parse_dates=["period_ts"])
    # ensure numerics
    if "value_signed_musd" in d_long.columns:
        d_long["value_signed_musd"] = pd.to_numeric(d_long["value_signed_musd"], errors="coerce")
    for col in d_q.columns:
        if col != "period_ts":
            d_q[col] = pd.to_numeric(d_q[col], errors="coerce")
    return d_long, d_q

d_long, d_q = load_data()

# Helper functions
def linear_forecast_one_step(df_q, target_col="total_inflow_musd"):
    df = df_q.dropna(subset=[target_col]).sort_values("period_ts").reset_index(drop=True)
    if df.shape[0] < 3:
        # fallback: naive forecast (last value)
        last = df[target_col].iloc[-1] if df.shape[0] > 0 else 0.0
        return last, last, last, None
    X = np.arange(len(df)).reshape(-1,1)
    y = df[target_col].values.reshape(-1,1)
    m = LinearRegression().fit(X, y)
    next_X = np.array([[len(df)]])
    pred = float(m.predict(next_X)[0,0])
    resid = y.flatten() - m.predict(X).flatten()
    se = resid.std(ddof=1)/np.sqrt(len(resid)) if len(resid)>1 else resid.std(ddof=0)
    ci = 1.96 * se
    return pred, pred - ci, pred + ci, m

# Sidebar
st.sidebar.title("FX Flow Monitoring")
page = st.sidebar.radio("Pilih halaman:", ["Overview", "Breakdown", "Compliance", "Forecast", "Data Preview"])

# Latest metrics
d_q = d_q.sort_values("period_ts").reset_index(drop=True)
latest_q = d_q["period_ts"].max()
latest_row = d_q.loc[d_q["period_ts"] == latest_q].iloc[0] if not d_q.empty else None

# Overview
if page == "Overview":
    st.title("Overview — Foreign Exchange Inflow / Outflow")
    if latest_row is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest quarter", latest_q.strftime("%Y-%m-%d"))
        c2.metric("Total Inflow (latest)", f"{int(latest_row['total_inflow_musd']):,} USD")
        c3.metric("Total Outflow (latest)", f"{int(latest_row['total_outflow_musd']):,} USD")
        c4.metric("Net Flow (latest)", f"{int(latest_row['net_flow_signed_musd']):,} USD")

    st.markdown("### Quarterly time series")
    if not d_q.empty:
        fig = go.Figure()
        if "total_inflow_musd" in d_q.columns:
            fig.add_trace(go.Scatter(x=d_q["period_ts"], y=d_q["total_inflow_musd"], mode="lines+markers", name="Total Inflow"))
        if "total_outflow_musd" in d_q.columns:
            fig.add_trace(go.Scatter(x=d_q["period_ts"], y=d_q["total_outflow_musd"], mode="lines+markers", name="Total Outflow"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Net flow and rolling average (4 quarters)")
        if "net_flow_signed_musd" in d_q.columns:
            d_q["rolling4_net"] = d_q["net_flow_signed_musd"].rolling(window=4, min_periods=1).mean()
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=d_q["period_ts"], y=d_q["net_flow_signed_musd"], name="Net Flow"))
            fig2.add_trace(go.Line(x=d_q["period_ts"], y=d_q["rolling4_net"], name="Rolling4 Avg", line=dict(width=3)))
            st.plotly_chart(fig2, use_container_width=True)

        # YoY
        if "net_flow_signed_musd" in d_q.columns:
            d_q["net_yoy_pct"] = d_q["net_flow_signed_musd"].pct_change(periods=4)
            st.markdown("### YoY Growth of Net Flow (percent)")
            st.line_chart(d_q.set_index("period_ts")["net_yoy_pct"].dropna())
    else:
        st.write("Quarterly aggregate dataset kosong atau tidak tersedia.")

# Breakdown
elif page == "Breakdown":
    st.title("Breakdown — by sector and flow_class")
    st.markdown("Filter time range and flow direction to inspect components.")
    min_date = st.date_input("Start date", d_long["period_ts"].min().date() if not d_long.empty else pd.Timestamp("2010-01-01").date())
    max_date = st.date_input("End date", d_long["period_ts"].max().date() if not d_long.empty else pd.Timestamp.today().date())
    flow_options = d_long["flow_class"].unique().tolist() if "flow_class" in d_long.columns else []
    flow_sel = st.multiselect("Flow class", options=flow_options, default=flow_options)

    mask = (d_long["period_ts"].dt.date >= min_date) & (d_long["period_ts"].dt.date <= max_date)
    if flow_sel:
        mask &= d_long["flow_class"].isin(flow_sel)
    dff = d_long.loc[mask].copy()

    st.markdown("#### Stacked contributions by item")
    if not dff.empty:
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
    else:
        st.write("No data for selected filters.")

# Compliance
elif page == "Compliance":
    st.title("Compliance — missing data & anomaly detection")
    st.markdown("Flags for missing data and z-score based anomaly detection by item")
    full = d_long.copy()
    if "value_signed_musd" in full.columns:
        missing_summary = full[full["value_signed_musd"].isna()].groupby("item_simple").size().reset_index(name="missing_count").sort_values("missing_count", ascending=False)
        st.subheader("Missing data summary (per item)")
        if not missing_summary.empty:
            st.dataframe(missing_summary)
        else:
            st.write("No missing values found in `value_signed_musd`.")
        st.subheader("Anomaly detection (z-score within each item)")
        dfz = full.dropna(subset=["value_signed_musd"]).copy()
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
    else:
        st.write("Dataset does not contain `value_signed_musd` column.")

# Forecast
elif page == "Forecast":
    st.title("Forecast — simple 1-quarter-ahead forecast (linear trend)")
    st.markdown("Lightweight linear-trend forecast with approximate 95% CI. For production, use Prophet/ETS/ARIMA.")
    metric_choice = st.selectbox("Forecast target", options=[col for col in ["total_inflow_musd","total_outflow_musd","net_flow_signed_musd"] if col in d_q.columns], index=0)
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

# Data preview
elif page == "Data Preview":
    st.title("Data preview")
    st.subheader("Long cleaned dataset (head)")
    st.dataframe(d_long.head(200))
    st.subheader("Quarterly aggregated dataset (head)")
    st.dataframe(d_q.head(200))

# sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("Prototype — Monitoring + Compliance + Forecast\nReplace forecasting with more advanced models for production.")
