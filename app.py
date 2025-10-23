# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# forecasting library (statsmodels)
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATS = True
except Exception as e:
    HAS_STATS = False

st.set_page_config(layout="wide", page_title="FX Flow Monitoring (Streamlit)", initial_sidebar_state="expanded")

#### Utility functions ####
@st.cache_data
def load_quarterly(path="/mnt/data/inflow_outflow_quarterly_agg.csv"):
    df = pd.read_csv(path)
    # normalize date
    df['period_ts'] = pd.to_datetime(df['period_ts'])
    df = df.sort_values('period_ts').reset_index(drop=True)
    # ensure numeric
    numeric_cols = [c for c in df.columns if c not in ['period_ts']]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # computed fields
    if 'net_flow_signed_musd' not in df.columns:
        df['net_flow_signed_musd'] = df.get('total_inflow_musd',0) - df.get('total_outflow_musd',0)
    return df

@st.cache_data
def load_long(path="/mnt/data/inflow_outflow_long_cleaned.csv"):
    df = pd.read_csv(path)
    # normalize
    if 'period_ts' in df.columns:
        df['period_ts'] = pd.to_datetime(df['period_ts'])
    else:
        # try period_raw like "2010q1"
        if 'period_raw' in df.columns:
            df['period_ts'] = df['period_raw'].apply(lambda x: pd.Period(x).end_time if isinstance(x,str) else pd.NaT)
    # ensure numeric value column names (common names in your data)
    for c in ['value_signed_musd','value_musd','value_raw','value']:
        if c in df.columns:
            df['value_musd'] = pd.to_numeric(df[c], errors='coerce')
            break
    # standardize columns
    if 'flow_class' not in df.columns:
        # try to infer from sign
        if 'value_musd' in df.columns:
            df['flow_class'] = df['value_musd'].apply(lambda x: 'inflow' if x>=0 else 'outflow')
    # optionally create item_simple
    if 'item' in df.columns:
        df['item_simple'] = df['item'].astype(str)
    return df

def compute_anomalies(df_quarterly, z_thresh=3.0):
    # anomaly detection on net flow signed using z-score
    s = df_quarterly['net_flow_signed_musd'].dropna()
    mean = s.mean(); std = s.std()
    df = df_quarterly.copy()
    df['net_zscore'] = (df['net_flow_signed_musd'] - mean) / std
    df['anomaly_flag'] = df['net_zscore'].abs() > z_thresh
    return df, mean, std

def forecast_series(series, periods=2, seasonal_periods=4):
    """
    Forecast using ExponentialSmoothing if available.
    Returns forecast_index (DatetimeIndex), forecast (array), lower, upper
    If statsmodels not available, simple naive forecast (last value) with +/- std as CI.
    """
    series = series.dropna()
    if len(series) < 4 or not HAS_STATS:
        last = series.iloc[-1] if len(series)>0 else 0.0
        forecast = np.array([last]*periods)
        resid_std = series.diff().std() if len(series)>1 else 0.0
        lower = forecast - 1.96*resid_std
        upper = forecast + 1.96*resid_std
        last_idx = series.index[-1]
        # assume quarterly frequency
        try:
            freq = pd.infer_freq(series.index)
        except:
            freq = None
        if hasattr(last_idx, 'to_period'):
            idx = pd.period_range(start=last_idx.to_period('Q') + 1, periods=periods, freq='Q').to_timestamp(how='end')
        else:
            idx = pd.date_range(start=series.index[-1] + pd.offsets.QuarterEnd(), periods=periods, freq='Q')
        return idx, forecast, lower, upper
    # Use Holt-Winters
    try:
        model = ExponentialSmoothing(series, seasonal='add', seasonal_periods=seasonal_periods, trend='add', damped_trend=False)
        fit = model.fit(optimized=True)
        pred = fit.forecast(periods)
        resid = fit.resid
        resid_std = resid.std() if resid is not None else 0.0
        lower = pred - 1.96*resid_std
        upper = pred + 1.96*resid_std
        # forecast index
        last_ts = series.index[-1]
        if isinstance(last_ts, pd.Timestamp):
            idx = pd.date_range(start=last_ts + pd.offsets.QuarterEnd(), periods=periods, freq='Q')
        else:
            idx = pd.RangeIndex(start=len(series), stop=len(series)+periods)
        return idx, pred.values, lower.values, upper.values
    except Exception as e:
        # fallback naive
        last = series.iloc[-1] if len(series)>0 else 0.0
        forecast = np.array([last]*periods)
        resid_std = series.diff().std() if len(series)>1 else 0.0
        lower = forecast - 1.96*resid_std
        upper = forecast + 1.96*resid_std
        last_idx = series.index[-1]
        try:
            idx = pd.date_range(start=series.index[-1] + pd.offsets.QuarterEnd(), periods=periods, freq='Q')
        except:
            idx = np.arange(len(series), len(series)+periods)
        return idx, forecast, lower, upper

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

#### Load data ####
st.sidebar.title("FX Flow Monitoring")
st.sidebar.markdown("Data source (auto-loaded from `/mnt/data`)")
df_q = load_quarterly()
df_l = load_long()

# Sidebar controls
st.sidebar.subheader("Controls")
z_threshold = st.sidebar.slider("Anomaly z-threshold (net flow)", 1.5, 5.0, 3.0, step=0.1)
forecast_periods = st.sidebar.selectbox("Forecast horizon (quarters)", [1,2,4], index=1)
selected_start = st.sidebar.date_input("Start date", value=df_q['period_ts'].min().date())
selected_end = st.sidebar.date_input("End date", value=df_q['period_ts'].max().date())

# Page navigation
page = st.sidebar.radio("Select page", ["Overview","Breakdown","Compliance","Forecast","Data"])

#### Overview Page ####
if page == "Overview":
    st.title("Overview — Foreign Exchange Flow Monitoring")
    st.markdown("High-level summary of inflow, outflow, net flow and reserve changes (quarterly).")
    mask = (df_q['period_ts'].dt.date >= selected_start) & (df_q['period_ts'].dt.date <= selected_end)
    df_view = df_q.loc[mask].copy()
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total_inflow = df_view['total_inflow_musd'].sum()
    total_outflow = df_view['total_outflow_musd'].sum()
    total_net = df_view['net_flow_signed_musd'].sum()
    reserve_change = df_view['reserve_change_musd'].iloc[-1] if 'reserve_change_musd' in df_view.columns else np.nan

    col1.metric("Total Inflow (selected)", f"${total_inflow:,.0f} M")
    col2.metric("Total Outflow (selected)", f"${total_outflow:,.0f} M")
    col3.metric("Net Flow (selected)", f"${total_net:,.0f} M")
    col4.metric("Reserve change (last)", f"${reserve_change:,.0f} M")

    # Time series
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_view['period_ts'], y=df_view['total_inflow_musd'],
                             mode='lines+markers', name='Inflow', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df_view['period_ts'], y=df_view['total_outflow_musd'],
                             mode='lines+markers', name='Outflow', line=dict(color='red')))
    fig.update_layout(title="Quarterly Inflow & Outflow", xaxis_title="Quarter", yaxis_title="USD (M)")
    st.plotly_chart(fig, use_container_width=True)

    # Net flow area
    fig2 = px.area(df_view, x='period_ts', y='net_flow_signed_musd', title="Net Flow (Inflow - Outflow) (M USD)")
    st.plotly_chart(fig2, use_container_width=True)

    # YoY growth (inflow)
    df_view = df_view.set_index('period_ts').sort_index()
    yoy = df_view['total_inflow_musd'].pct_change(4) * 100
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=yoy.index, y=yoy.values, name='YoY Inflow %', marker_color='blue'))
    fig3.update_layout(title="YoY Growth in Inflow (%)", xaxis_title="Quarter")
    st.plotly_chart(fig3, use_container_width=True)

    # Rolling 4-quarter average for net flow
    df_view['rolling_4q'] = df_view['net_flow_signed_musd'].rolling(window=4).mean()
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df_view.index, y=df_view['net_flow_signed_musd'], mode='lines', name='Net Flow'))
    fig4.add_trace(go.Scatter(x=df_view.index, y=df_view['rolling_4q'], mode='lines', name='Rolling 4Q Avg', line=dict(dash='dash')))
    fig4.update_layout(title="Net Flow & Rolling 4-Quarter Average")
    st.plotly_chart(fig4, use_container_width=True)

#### Breakdown Page ####
elif page == "Breakdown":
    st.title("Breakdown — Component & Country Analysis")
    st.markdown("Use the long (item-level) dataset for sectoral and country breakdowns.")
    # basic cleaning
    dfl = df_l.copy()
    # ensure period_ts present
    if 'period_ts' not in dfl.columns:
        st.warning("Long file doesn't have `period_ts`. Please check `period_raw` or `period` column.")
    # filters
    items = dfl['item_simple'].dropna().unique().tolist() if 'item_simple' in dfl.columns else dfl['item'].unique().tolist()
    chosen_items = st.multiselect("Select items (components)", options=sorted(items), default=items[:6])
    chosen_flow = st.selectbox("Flow class", options=['all','inflow','outflow'], index=0)
    if 'value_musd' not in dfl.columns:
        st.error("Long file does not contain a numeric 'value_musd' column. Check your file.")
    # filter
    mask = dfl['item_simple'].isin(chosen_items) if 'item_simple' in dfl.columns else dfl['item'].isin(chosen_items)
    if chosen_flow!='all':
        mask = mask & (dfl['flow_class']==chosen_flow)
    dff = dfl.loc[mask].copy()
    # aggregate per quarter & item
    agg = dff.groupby([pd.Grouper(key='period_ts', freq='Q'), 'item_simple' if 'item_simple' in dff.columns else 'item'])['value_musd'].sum().reset_index()
    agg = agg.rename(columns={'period_ts':'period_ts','value_musd':'amount_musd'})

    st.subheader("Stacked area / column by quarter")
    fig = px.bar(agg, x='period_ts', y='amount_musd', color='item_simple' if 'item_simple' in agg.columns else 'item',
                 title="Component contributions per quarter (selected items)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Treemap: Share by component (selected period)")
    # select single quarter
    quarter = st.selectbox("Select quarter (for treemap)", options=sorted(agg['period_ts'].dt.to_period('Q').astype(str).unique()), index=-1)
    q_ts = pd.Period(quarter).end_time
    treedf = agg[agg['period_ts']==q_ts]
    if treedf.empty:
        st.info("No data for selected quarter.")
    else:
        fig2 = px.treemap(treedf, path=[treedf.columns[1]], values='amount_musd', title=f"Share by component — {quarter}")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Raw table (filtered)")
    st.dataframe(dff.sort_values('period_ts', ascending=False).reset_index(drop=True))

#### Compliance Page ####
elif page == "Compliance":
    st.title("Compliance & Anomaly Detection")
    st.markdown("Monitor `errors & omissions`, flags for anomalies, and missing data metrics.")

    df_q_anom, mean_net, std_net = compute_anomalies(df_q, z_thresh=z_threshold)
    st.metric("Net flow mean (all)", f"${mean_net:,.0f} M")
    st.metric("Net flow std (all)", f"${std_net:,.0f} M")
    st.markdown(f"Using z-threshold = **{z_threshold}** → anomalies flagged when |z| > {z_threshold}")

    # anomalies table
    anomalies = df_q_anom[df_q_anom['anomaly_flag']]
    st.subheader("Anomalous quarters (by net flow z-score)")
    st.dataframe(anomalies[['period_ts','net_flow_signed_musd','net_zscore','errors_omissions_musd']].sort_values('period_ts', ascending=False))

    # errors & omissions over time
    if 'errors_omissions_musd' in df_q.columns:
        fig = px.line(df_q, x='period_ts', y='errors_omissions_musd', title="Errors & Omissions (M USD)")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No `errors_omissions_musd` column in quarterly data.")

    # missing data stats
    st.subheader("Missing data summary (quarterly file)")
    missing = df_q.isna().sum().reset_index()
    missing.columns = ['column','n_missing']
    st.table(missing)

#### Forecast Page ####
elif page == "Forecast":
    st.title("Forecast — Simple Time Series Forecast")
    st.markdown("Forecasting using Holt-Winters (additive seasonal) when available. If `statsmodels` not installed or data short, falls back to naive forecast.")
    periods = forecast_periods
    # choose series to forecast
    series_choice = st.selectbox("Series to forecast", ['total_inflow_musd','total_outflow_musd','net_flow_signed_musd'])
    s = df_q.set_index('period_ts')[series_choice]
    idx, pred, lower, upper = forecast_series(s, periods=periods, seasonal_periods=4)
    # plot historic + forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines+markers', name='Historical'))
    # forecast
    fig.add_trace(go.Scatter(x=idx, y=pred, mode='lines+markers', name='Forecast', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=idx, y=upper, mode='lines', name='Upper CI', line=dict(dash='dot'), marker_color='lightgrey', showlegend=False))
    fig.add_trace(go.Scatter(x=idx, y=lower, mode='lines', name='Lower CI', line=dict(dash='dot'), marker_color='lightgrey', showlegend=False))
    # fill between
    fig.add_traces([go.Scatter(
        x = list(idx) + list(idx[::-1]),
        y = list(upper) + list(lower[::-1]),
        fill='toself',
        fillcolor='rgba(255,165,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    )])
    fig.update_layout(title=f"Forecast for {series_choice} ({periods} quarters)", xaxis_title="Quarter", yaxis_title="USD (M)")
    st.plotly_chart(fig, use_container_width=True)

    # show forecast table
    fc_df = pd.DataFrame({
        'period': idx,
        'forecast_musd': pred,
        'lower_ci': lower,
        'upper_ci': upper
    })
    st.subheader("Forecast numbers")
    st.dataframe(fc_df)

#### Data Page ####
elif page == "Data":
    st.title("Data Explorer & Download")
    st.markdown("Preview and download the cleaned datasets used in the app.")
    st.subheader("Quarterly aggregated file")
    st.dataframe(df_q)
    st.download_button("Download quarterly CSV", data=df_to_csv_bytes(df_q), file_name="inflow_outflow_quarterly_agg_cleaned.csv", mime="text/csv")

    st.subheader("Long (item-level) file")
    st.dataframe(df_l)
    st.download_button("Download long CSV", data=df_to_csv_bytes(df_l), file_name="inflow_outflow_long_cleaned_export.csv", mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.markdown("Made for: Daffa — FX Flow Monitoring prototype")
st.sidebar.caption("Streamlit app generated by ChatGPT — edit & expand as needed.")
