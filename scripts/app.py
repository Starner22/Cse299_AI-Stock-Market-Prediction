# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Stock Models + Risk Dashboard")

# =========================
# BASE PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))  # project root
PREDICTIONS_BASE = os.path.join(BASE_DIR, "data", "predictions")
PLOTS_BASE = os.path.join(BASE_DIR, "plots")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_BASE = os.path.join(BASE_DIR, "models")

MODEL_LIST = ["LR", "LSTM", "GRU", "NBEATS", "TFT", "SVM"]

os.makedirs(PREDICTIONS_BASE, exist_ok=True)
os.makedirs(PLOTS_BASE, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_BASE, exist_ok=True)

# =========================
# UTILITY FUNCTIONS
# =========================
def try_variants(base, model_name, kind="predictions"):
    """Return folder path for predictions/plots/models with common naming variants."""
    candidates, base_dir = [], ""
    if kind == "predictions":
        candidates = [f"{model_name}_predictions", f"{model_name}-predictions", model_name]
        base_dir = PREDICTIONS_BASE
    elif kind == "plots":
        candidates = [f"{model_name}_plots", f"{model_name}-plots", model_name]
        base_dir = PLOTS_BASE
    elif kind == "models":
        candidates = [f"{model_name}_models", f"{model_name}-models", model_name]
        base_dir = MODELS_BASE
    else:
        return None
    for c in candidates:
        p = os.path.join(base_dir, c)
        if os.path.isdir(p):
            return p
    return os.path.join(base_dir, model_name)

def find_prediction_and_metric_file(pred_dir, ticker):
    """Return paths for prediction CSV and metric CSV if they exist."""
    pred_path = metrics_path = None
    try:
        files = os.listdir(pred_dir)
    except Exception:
        files = []
    for f in files:
        if f.lower().startswith(ticker.lower()) and ("pred" in f.lower() or "prediction" in f.lower()):
            pred_path = os.path.join(pred_dir, f)
            break
    for f in files:
        if f.lower().startswith(ticker.lower()) and "metric" in f.lower():
            metrics_path = os.path.join(pred_dir, f)
            break
    return pred_path, metrics_path

def load_raw_csv_flexible(path):
    """Load raw CSV robustly: skip metadata, set Date as index."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    if "Close" not in df.columns:
        raise ValueError(f"No 'Close' column found. Columns: {df.columns.tolist()}")
    return df

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Settings")
st.sidebar.markdown("Select a model and a stock to inspect predictions & risk metrics")

available_models = []
model_pred_map, model_plot_map = {}, {}
for model in MODEL_LIST:
    pred_dir = try_variants(PREDICTIONS_BASE, model, kind="predictions")
    plot_dir = try_variants(PLOTS_BASE, model, kind="plots")
    if pred_dir:
        available_models.append(model)
        model_pred_map[model] = pred_dir
        model_plot_map[model] = plot_dir

if not available_models:
    st.error("No prediction model subfolders found in data/predictions/.")
    st.stop()

selected_model = st.sidebar.selectbox("Model", available_models)
PRED_DIR = model_pred_map[selected_model]
PLOT_DIR = model_plot_map.get(selected_model) or os.path.join(PLOTS_BASE, f"{selected_model}_plots")

# =========================
# STOCK SELECTION
# =========================
all_files = os.listdir(PRED_DIR)
tickers_set = {f.split("_")[0].upper() for f in all_files if f.lower().endswith(".csv")}
tickers = sorted(list(tickers_set))
if not tickers:
    st.error(f"No CSV predictions found in {PRED_DIR}")
    st.stop()

selected_stock = st.sidebar.selectbox("Stock (Ticker)", tickers)
st.title(f"Stock Dashboard — {selected_stock} · Model: {selected_model}")

# =========================
# PREDICTIONS & METRICS
# =========================
pred_file, metrics_file = find_prediction_and_metric_file(PRED_DIR, selected_stock)

st.subheader("Model Predictions")
if pred_file and os.path.exists(pred_file):
    preds_df = pd.read_csv(pred_file)
    st.dataframe(preds_df.head(200))
else:
    st.info("Predictions CSV not found.")

if metrics_file and os.path.exists(metrics_file):
    metrics_df = pd.read_csv(metrics_file)
    st.markdown("**Metrics**")
    st.table(metrics_df)
else:
    st.info("Metrics CSV not found.")

# =========================
# MODEL PLOT
# =========================
plot_candidates = [
    f"{selected_stock}_plot.png",
    f"{selected_stock}_{selected_model}_plot.png",
    f"{selected_stock}.png",
    f"{selected_stock}_pred_plot.png",
]
found_plot = None
if os.path.isdir(PLOT_DIR):
    for pc in plot_candidates:
        p = os.path.join(PLOT_DIR, pc)
        if os.path.exists(p):
            found_plot = p
            break

if found_plot:
    st.subheader("Model Plot")
    img = Image.open(found_plot)
    st.image(img, width=700)
else:
    st.info(f"No plot found in {PLOT_DIR}")

col1, col2 = st.columns(2)
with col1:
    if pred_file:
        st.download_button("Download Predictions", data=open(pred_file, "rb"), file_name=os.path.basename(pred_file))
with col2:
    if metrics_file:
        st.download_button("Download Metrics", data=open(metrics_file, "rb"), file_name=os.path.basename(metrics_file))

# =========================
# RISK ANALYSIS — HISTORICAL & MODEL-BASED
# =========================
st.header("Risk Analysis — Historical vs. Model-Based")

# Load raw historical CSV
raw_candidates = [
    os.path.join(RAW_DATA_DIR, f"{selected_stock}.csv"),
    os.path.join(RAW_DATA_DIR, f"{selected_stock}_raw.csv"),
    os.path.join(RAW_DATA_DIR, f"{selected_stock.lower()}.csv"),
]
raw_file = next((r for r in raw_candidates if os.path.exists(r)), None)
if raw_file is None:
    st.error(f"Raw CSV not found for {selected_stock}.")
    st.stop()

df_raw = load_raw_csv_flexible(raw_file)
hist_close = df_raw["Close"]
hist_returns = hist_close.pct_change().dropna()
hist_vol = hist_returns.std() * np.sqrt(252)
confidence = st.sidebar.slider("VaR confidence level (%)", 90, 99, 95)
conf = confidence / 100
hist_var = hist_returns.quantile(1 - conf)
hist_es = hist_returns[hist_returns <= hist_var].mean()
risk_free_rate = st.sidebar.number_input("Risk-free rate (annual, e.g. 0.03)", value=0.03, step=0.005)
hist_annual_return = hist_returns.mean() * 252
hist_sharpe = (hist_annual_return - risk_free_rate) / (hist_vol if hist_vol != 0 else np.nan)

# Model-based
if pred_file and os.path.exists(pred_file):
    model_df = pd.read_csv(pred_file)
    if "Predicted_Close" in model_df.columns:
        model_returns = model_df["Predicted_Close"].pct_change().dropna()
        model_vol = model_returns.std() * np.sqrt(252)
        model_var = model_returns.quantile(1 - conf)
        model_es = model_returns[model_returns <= model_var].mean()
        model_annual_return = model_returns.mean() * 252
        model_sharpe = (model_annual_return - risk_free_rate) / (model_vol if model_vol != 0 else np.nan)
    else:
        model_returns = model_vol = model_var = model_es = model_annual_return = model_sharpe = None
else:
    model_returns = model_vol = model_var = model_es = model_annual_return = model_sharpe = None

# Display metrics side by side
c1, c2 = st.columns(2)
with c1:
    st.subheader("Historical Risk Metrics")
    st.metric("Annual Volatility", f"{hist_vol:.2%}")
    st.metric(f"VaR ({confidence}%)", f"{hist_var:.2%}")
    st.metric("Expected Shortfall (ES)", f"{hist_es:.2%}")
    st.metric("Sharpe Ratio", f"{hist_sharpe:.2f}")
with c2:
    st.subheader("Model-Based Risk Metrics")
    if model_returns is not None:
        st.metric("Annual Volatility", f"{model_vol:.2%}")
        st.metric(f"VaR ({confidence}%)", f"{model_var:.2%}")
        st.metric("Expected Shortfall (ES)", f"{model_es:.2%}")
        st.metric("Sharpe Ratio", f"{model_sharpe:.2f}")
    else:
        st.info("Model predictions not found or invalid.")

# Plot histogram comparison
st.subheader("Returns Distribution Comparison")
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(hist_returns, bins=60, alpha=0.6, label="Historical", edgecolor="black")
if model_returns is not None:
    ax.hist(model_returns, bins=60, alpha=0.6, label="Model-Based", edgecolor="red")
ax.set_xlabel("Daily Returns")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

st.success("Dashboard loaded successfully!")
