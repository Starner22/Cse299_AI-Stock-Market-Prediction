import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from data_loader import load_stock
import warnings
warnings.filterwarnings("ignore")

# ---- Directories ----
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

predictions_folder = os.path.join(BASE_DIR, "data/predictions/LSTM_recursive")
plots_folder = os.path.join(BASE_DIR, "plots/LSTM_recursive")
models_folder = os.path.join(BASE_DIR, "models/LSTM_recursive")

for folder in [predictions_folder, plots_folder, models_folder]:
    os.makedirs(folder, exist_ok=True)

# ---- Stocks ----
stocks = [
    "HD","PG","DIS","MA","BAC","NFLX","ADBE","PYPL","CMCSA","XOM",
    "PFE","KO","INTC","CSCO","NKE","MRK","PEP","ABBV","ABT","CRM",
    "ORCL","T","VZ","CVX","MCD","WMT","DHR","ACN","LLY","AVGO",
    "QCOM","TXN","NEE","COST","TMUS","IBM","SBUX","MDT","HON","AMD"
]

SEQ_LENGTH = 60
FUTURE_DAYS = 30
NOISE_FACTOR = 1.0  # Adjust to increase/decrease injected volatility (1.0 = realistic)

# Optional: Set seed for reproducibility across runs
np.random.seed(42)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # Predict 'Close'
    return np.array(X), np.array(y)


def recursive_forecast_lstm(model, last_sequence, scaler, n_steps, daily_vol):
    """
    Perform recursive forecasting with proper stochastic noise based on historical volatility.
    """
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(n_steps):
        # Predict next step (scaled)
        pred_scaled = model.predict(current_seq.reshape(1, SEQ_LENGTH, 1), verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0, 0]

        # Inject stochastic noise: multiplicative Gaussian noise scaled by recent daily volatility
        noise = np.random.normal(0, daily_vol * NOISE_FACTOR)
        noisy_price = pred_price * (1 + noise)

        predictions.append(noisy_price)

        # Update sequence: shift and append new noisy scaled value
        noisy_scaled = scaler.transform([[noisy_price]])[0, 0]
        current_seq = np.append(current_seq[1:], [[noisy_scaled]], axis=0)

    return predictions


# ---- Main Loop ----
print("Starting Recursive LSTM Forecasting with Stochastic Noise...\n")

for symbol in stocks:
    print(f"--- Processing {symbol} ---")

    try:
        df = load_stock(symbol)
        if df is None or "Close" not in df.columns or len(df) < SEQ_LENGTH + 50:
            print(f"{symbol}: Insufficient or missing data. Skipping.\n")
            continue

        close_prices = df["Close"].values.reshape(-1, 1)

        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)

        # Create sequences
        X, y = create_sequences(scaled_data, SEQ_LENGTH)
        if len(X) == 0:
            print(f"{symbol}: Not enough sequences. Skipping.\n")
            continue

        # Train-test split (85% train)
        train_size = int(len(X) * 0.85)
        X_train, y_train = X[:train_size], y[:train_size]

        # Build LSTM model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
            LSTM(64),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")

        # Train with early stopping
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0
        )

        # Estimate recent daily volatility (std of log returns is better, but pct_change is fine)
        returns = pd.Series(close_prices.flatten()).pct_change().dropna()
        recent_daily_vol = returns.tail(60).std()  # Use last 60 days for stability

        # Use last available sequence for forecasting
        last_sequence = X[-1]

        # Generate future prices with noise
        future_prices = recursive_forecast_lstm(
            model, last_sequence, scaler, FUTURE_DAYS, recent_daily_vol
        )

        # Generate future business dates
        last_date = df.index[-1]
        future_dates = pd.bdate_range(start=last_date, periods=FUTURE_DAYS + 1)[1:]

        # Save forecast
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Close": future_prices
        })

        forecast_csv_path = os.path.join(predictions_folder, f"{symbol}_recursive_forecast.csv")
        forecast_df.to_csv(forecast_csv_path, index=False)

        # Save model and scaler
        model.save(os.path.join(models_folder, f"{symbol}_recursive_lstm.h5"))
        joblib.dump(scaler, os.path.join(models_folder, f"{symbol}_scaler.pkl"))

        # Create combined plot: last 120 days historical + 30-day forecast
        plt.figure(figsize=(14, 7))
        historical_slice = df["Close"].iloc[-120:]
        plt.plot(historical_slice.index, historical_slice.values, label="Historical Close", color="blue", linewidth=2)
        plt.plot(future_dates, future_prices, "--o", label="Recursive Forecast (with noise)", color="purple", alpha=0.8)

        plt.title(f"{symbol} — 30-Day Recursive LSTM Forecast (with Stochastic Volatility)", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(plots_folder, f"{symbol}_recursive_lstm.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"{symbol}: Forecast completed and saved.\n")

    except Exception as e:
        print(f"{symbol}: ERROR → {str(e)}\n")
        continue

print("Recursive LSTM forecasting with stochastic noise completed for all stocks!")