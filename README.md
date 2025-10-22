# Dashboard-Lalu-Lintas-Devisa-Foreign-Exchange-Flow-Monitoring-System-
Membuat dashboard internal untuk memantau arus masuk dan keluar devisa (export receipts, import payments, foreign investment, tourism receipts)

FX Flow Monitoring — Streamlit prototype

Files expected:
- /mnt/data/inflow_outflow_long_cleaned.csv
- /mnt/data/inflow_outflow_quarterly_agg.csv

Install minimal dependencies:
pip install streamlit pandas numpy plotly scikit-learn

Run:
streamlit run streamlit_app.py

Notes:
- Forecast: simple linear trend (one-quarter). For better accuracy use Prophet/ETS/ARIMA.
- Anomaly detection: per-item z-score; tune threshold in the Compliance page.
- Replace file paths if your files are in another folder.
