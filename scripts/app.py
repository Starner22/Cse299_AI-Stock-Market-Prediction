import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    layout="wide",
    page_title="AI Stock Forecast & Investment Compass"
)

# =========================
# BASE PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

PRED_BASE = os.path.join(BASE_DIR, "data", "predictions")
RAW_BASE = os.path.join(BASE_DIR, "data", "raw")
PLOT_BASE = os.path.join(BASE_DIR, "plots")

MODEL_LIST = ["LR", "RIDGE", "SVR", "PROPHET", "LSTM", "GRU", "NBEATS", "TFT"]

# =========================
# UTILITY FUNCTIONS
# =========================
def find_folder(base, name, folder_type="any"):
    """
    Flexible folder finder.
    folder_type: "predictions" (prioritize _predictions and _recursive),
                 "plots" (prioritize _plots over _recursive),
                 "any" (original behavior)
    """
    if folder_type == "predictions":
        suffixes = ["_predictions", "_recursive", "_plots", ""]
    elif folder_type == "plots":
        suffixes = ["_plots", "_recursive", "_predictions", ""]
    else:
        suffixes = ["_predictions", "_recursive", "_plots", ""]

    for suffix in suffixes:
        path = os.path.join(base, f"{name}{suffix}")
        if os.path.isdir(path):
            return path
    return None


def find_csv(folder, ticker, keyword=None, exact_suffix=None):
    """Find CSV starting with ticker, optionally with keyword or exact suffix"""
    if not folder or not os.path.isdir(folder):
        return None
    try:
        files = os.listdir(folder)
    except:
        return None

    matching = []
    for f in files:
        if f.lower().startswith(ticker.lower()) and f.endswith(".csv"):
            if exact_suffix and f.endswith(exact_suffix):
                return os.path.join(folder, f)  # highest priority
            if keyword is None or keyword in f.lower():
                matching.append(f)

    # Return first match if any
    if matching:
        return os.path.join(folder, matching[0])
    return None


def load_raw_csv(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(
            path, skiprows=3,
            names=["Date", "Close", "High", "Low", "Open", "Volume"]
        )
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
        return df
    except:
        return None


# =========================
# INVESTMENT ANALYSIS LOGIC
# =========================
def analyze_future(df):
    prices = df["Predicted_Close"].values
    returns = pd.Series(prices).pct_change().dropna()

    slope = np.polyfit(range(len(prices)), prices, 1)[0]
    expected_return = (prices[-1] - prices[0]) / prices[0]
    volatility = returns.std()

    trend_score = np.clip(slope / prices.mean(), -1, 1)
    return_score = np.clip(expected_return, -1, 1)
    risk_score = np.clip(1 - volatility * 10, 0, 1)

    score = 0.4 * trend_score + 0.3 * return_score + 0.3 * risk_score
    return score, slope, expected_return, volatility


def investment_label(score):
    if score >= 0.7:
        return "🟢 STRONG BUY", "green"
    elif score >= 0.5:
        return "🟡 BUY", "gold"
    elif score >= 0.3:
        return "🟠 HOLD", "orange"
    else:
        return "🔴 AVOID", "red"


# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")

available_models = []
model_map = {}

for model in MODEL_LIST:
    std = find_folder(PRED_BASE, f"{model}_predictions", folder_type="predictions")  # doesn't exist, falls to _recursive if present
    rec = find_folder(PRED_BASE, f"{model}_recursive", folder_type="predictions")
    plots = find_folder(PLOT_BASE, model, folder_type="plots")  # KEY: prioritize _plots
    if std or rec:
        available_models.append(model)
        model_map[model] = {"std": std, "rec": rec, "plots": plots}

if not available_models:
    st.error("No model folders found in data/predictions/")
    st.stop()

selected_model = st.sidebar.selectbox("Select Model", available_models)
model_dirs = model_map[selected_model]

# =========================
# STOCK SELECTION
# =========================
files = []
for folder in [model_dirs["std"], model_dirs["rec"]]:
    if folder and os.path.isdir(folder):
        try:
            files += [f for f in os.listdir(folder) if f.endswith(".csv")]
        except:
            pass

tickers = sorted({f.split("_")[0].upper() for f in files})
if not tickers:
    st.error("No ticker CSVs found.")
    st.stop()

selected_stock = st.sidebar.selectbox("Select Stock", tickers)
st.title(f"📈 {selected_stock} — {selected_model}")

# =========================
# STANDARD (HISTORICAL/TEST) PREDICTIONS
# =========================
st.header("📊 Historical / Test Predictions")

# Prioritize exact match: AAPL_predictions.csv
std_csv = find_csv(model_dirs["std"], selected_stock, exact_suffix="_predictions.csv")
# Fallback: any CSV starting with ticker
if not std_csv:
    std_csv = find_csv(model_dirs["std"], selected_stock)

if std_csv and os.path.exists(std_csv):
    df_std = pd.read_csv(std_csv)
    if "Date" in df_std.columns:
        df_std["Date"] = pd.to_datetime(df_std["Date"])
    st.dataframe(df_std.head(200), use_container_width=True)
    st.download_button(
        "Download Historical Predictions",
        data=open(std_csv, "rb").read(),
        file_name=os.path.basename(std_csv),
        mime="text/csv"
    )
else:
    st.info("No standard (historical/test) prediction CSV found.")

# Also show metrics if present
metrics_csv = find_csv(model_dirs["std"], selected_stock, exact_suffix="_metrics.csv")
if metrics_csv and os.path.exists(metrics_csv):
    st.subheader("Model Metrics")
    metrics_df = pd.read_csv(metrics_csv)
    st.table(metrics_df)
    st.download_button(
        "Download Metrics",
        data=open(metrics_csv, "rb").read(),
        file_name=os.path.basename(metrics_csv),
        mime="text/csv"
    )

# =========================
# HISTORICAL PREDICTION PLOT
# =========================
st.header("📉 Historical Actual vs Predicted Plot")

plots_folder = model_dirs["plots"]
if plots_folder and os.path.isdir(plots_folder):
    possible_plots = [
    f"{selected_stock}_{selected_model.lower()}_plot.png",  # Primary: AAPL_lstm_plot.png, AAPL_tft_plot.png, etc.
    f"{selected_stock}_{selected_model.lower()}_lstm_plot.png",  # Extra safety for LSTM
    f"{selected_stock}_lstm_plot.png",
    f"{selected_stock}_gru_plot.png",
    f"{selected_stock}_tft_plot.png",
    f"{selected_stock}_nbeats_plot.png",
    f"{selected_stock}_plot.png",
    f"{selected_stock}.png",
    ]
    plot_path = None
    for name in possible_plots:
        candidate = os.path.join(plots_folder, name)
        if os.path.exists(candidate):
            plot_path = candidate
            break

    if plot_path:
        st.image(plot_path, caption=f"{selected_stock} — Actual vs Predicted (Test Period)", use_column_width=None)
    else:
        st.info("No historical prediction plot found.")
else:
    st.info("No plots folder found for this model.")

# =========================
# FUTURE (RECURSIVE) FORECAST
# =========================
st.header("🔮 Future Forecast")

rec_csv = find_csv(model_dirs["rec"], selected_stock, keyword="forecast")
if not rec_csv:
    rec_csv = find_csv(model_dirs["rec"], selected_stock)  # fallback

df_future = None
if rec_csv and os.path.exists(rec_csv):
    df_future = pd.read_csv(rec_csv)
    df_future["Date"] = pd.to_datetime(df_future["Date"])

    st.dataframe(df_future, use_container_width=True)
    st.download_button(
        "Download Future Forecast",
        data=open(rec_csv, "rb").read(),
        file_name=os.path.basename(rec_csv),
        mime="text/csv"
    )

    # Plot future forecast
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_future["Date"], df_future["Predicted_Close"], marker="o", color="purple", linewidth=2)
    ax.set_title(f"{selected_stock} — 30-Day Future Price Forecast ({selected_model})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Close Price")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
else:
    st.warning("No recursive future forecast found.")

# =========================
# INVESTMENT COMPASS
# =========================
st.header("🧭 Investment Compass")

if df_future is not None:
    score, slope, exp_ret, vol = analyze_future(df_future)
    label, color = investment_label(score)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(
            f"""
            <div style="text-align:center; padding:40px; border-radius:50%; 
                        border:8px solid {color}; background:#f9f9f9; 
                        font-size:26px; font-weight:bold; width:200px; height:200px; 
                        display:flex; align-items:center; justify-content:center; margin:auto;">
                {label}
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.subheader("Analysis Details")
        st.write(f"""
        • **Trend Direction**: {"Upward 📈" if slope > 0 else "Downward 📉"}  
        • **Expected Return (30 days)**: {exp_ret:+.2%}  
        • **Forecast Volatility**: {vol:.2%}  
        • **Overall Score**: {score:.2f} → **{label.split(maxsplit=1)[1] if ' ' in label else label}**  
        • **Confidence**: {"High" if score > 0.6 else "Moderate" if score > 0.4 else "Low"}
        """)
else:
    st.info("Future forecast required for investment recommendation.")

# =========================
# RISK ANALYSIS
# =========================
st.header("⚠️ Risk Metrics & Returns Distribution")

possible_raw = [
    f"{selected_stock}.csv",
    f"{selected_stock}_raw.csv",
    f"{selected_stock.lower()}.csv",
    f"{selected_stock.lower()}_raw.csv"
]
raw_file = next((os.path.join(RAW_BASE, n) for n in possible_raw if os.path.exists(os.path.join(RAW_BASE, n))), None)

risk_free_rate = st.sidebar.number_input("Risk-free rate (annual)", value=0.04, step=0.005, format="%.3f")
conf_level = st.sidebar.slider("VaR Confidence (%)", 90, 99, 95)
conf = conf_level / 100

if raw_file:
    df_raw = load_raw_csv(raw_file)
    if df_raw is not None and "Close" in df_raw.columns:
        hist_returns = df_raw["Close"].pct_change().dropna()
        hist_vol = hist_returns.std() * np.sqrt(252)
        hist_var = hist_returns.quantile(1 - conf)
        hist_es = hist_returns[hist_returns <= hist_var].mean()
        hist_ret_annual = hist_returns.mean() * 252
        hist_sharpe = (hist_ret_annual - risk_free_rate) / hist_vol if hist_vol > 0 else 0

        forecast_returns = None
        if df_future is not None:
            forecast_returns = df_future["Predicted_Close"].pct_change().dropna()
            fore_vol = forecast_returns.std() * np.sqrt(252)
            fore_var = forecast_returns.quantile(1 - conf)
            fore_es = forecast_returns[forecast_returns <= fore_var].mean()
            fore_ret_annual = forecast_returns.mean() * 252
            fore_sharpe = (fore_ret_annual - risk_free_rate) / fore_vol if fore_vol > 0 else 0

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Historical Risk")
            st.metric("Annual Volatility", f"{hist_vol:.2%}")
            st.metric(f"VaR ({conf_level}%)", f"{hist_var:.2%}")
            st.metric("Expected Shortfall", f"{hist_es:.2%}")
            st.metric("Sharpe Ratio", f"{hist_sharpe:.2f}")

        with c2:
            st.subheader("Forecast Risk (30-day)")
            if forecast_returns is not None:
                st.metric("Annualized Volatility", f"{fore_vol:.2%}")
                st.metric(f"VaR ({conf_level}%)", f"{fore_var:.2%}")
                st.metric("Expected Shortfall", f"{fore_es:.2%}")
                st.metric("Sharpe Ratio", f"{fore_sharpe:.2f}")
            else:
                st.info("No forecast available")

        # Returns distribution
        st.subheader("Returns Distribution Comparison")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(hist_returns, bins=60, alpha=0.7, label="Historical", color="skyblue", edgecolor="black")
        if forecast_returns is not None:
            ax.hist(forecast_returns, bins=40, alpha=0.6, label="Forecast", color="orange", edgecolor="red")
        ax.set_xlabel("Daily Returns")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.set_title("Historical vs Forecast Returns Distribution")
        st.pyplot(fig)
    else:
        st.error("Failed to load or parse raw historical data.")
else:
    st.warning("Raw historical data not found.")

st.success("✅ Dashboard loaded successfully!")