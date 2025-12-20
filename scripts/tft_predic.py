import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from data_loader import load_stock
from tensorflow.keras.models import Sequential  
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dropout, Dense  # ← This was missing

# ---- Directories ----
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

predictions_folder = os.path.join(BASE_DIR, "data/predictions/TFT_recursive")
plots_folder = os.path.join(BASE_DIR, "plots/TFT_recursive")
models_folder = os.path.join(BASE_DIR, "models/TFT_recursive")

for folder in [predictions_folder, plots_folder, models_folder]:
    os.makedirs(folder, exist_ok=True)

# ---- Stocks (top 10 for consistency with LSTM recursive) ----
stocks = [
    "HD","PG","DIS","MA","BAC","NFLX","ADBE","PYPL","CMCSA","XOM",
    "PFE","KO","INTC","CSCO","NKE","MRK","PEP","ABBV","ABT","CRM",
    "ORCL","T","VZ","CVX","MCD","WMT","DHR","ACN","LLY","AVGO",
    "QCOM","TXN","NEE","COST","TMUS","IBM","SBUX","MDT","HON","AMD"
]

SEQ_LENGTH = 60
FUTURE_DAYS = 30
NOISE_FACTOR = 1.0  # Adjust multiplier for more/less volatility in forecast (1.0 = realistic)

# For reproducibility
np.random.seed(42)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def recursive_forecast_tft(model, last_sequence, scaler, n_steps, daily_vol):
    """
    Recursive multi-step forecasting with stochastic noise injection.
    Identical logic to LSTM version — works perfectly for this encoder-decoder style.
    """
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(n_steps):
        # Predict next scaled price
        pred_scaled = model.predict(current_seq.reshape(1, SEQ_LENGTH, 1), verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0, 0]

        # Add multiplicative Gaussian noise based on recent volatility
        noise = np.random.normal(0, daily_vol * NOISE_FACTOR)
        noisy_price = pred_price * (1 + noise)

        predictions.append(noisy_price)

        # Scale noisy price back and append to sequence
        noisy_scaled = scaler.transform([[noisy_price]])[0, 0]
        current_seq = np.append(current_seq[1:], [[noisy_scaled]], axis=0)

    return predictions


# ---- Main Loop ----
print("Starting Recursive TFT Forecasting with Stochastic Noise...\n")

for symbol in stocks:
    print(f"--- Processing {symbol} ---")

    try:
        df = load_stock(symbol)
        if df is None or "Close" not in df.columns or len(df) < SEQ_LENGTH + 100:
            print(f"{symbol}: Insufficient data. Skipping.\n")
            continue

        close_prices = df["Close"].values.reshape(-1, 1)

        # Scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Create sequences
        X, y = create_sequences(scaled_data, SEQ_LENGTH)
        if len(X) < 50:
            print(f"{symbol}: Not enough sequences. Skipping.\n")
            continue

        # Train on 85% (like LSTM version)
        train_size = int(len(X) * 0.85)
        X_train, y_train = X[:train_size], y[:train_size]

        # Re-build and retrain the same TFT-style model
        inputs = (SEQ_LENGTH, 1)
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=inputs),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")

        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0
        )

        # Estimate recent daily volatility
        returns = pd.Series(close_prices.flatten()).pct_change().dropna()
        recent_daily_vol = returns.tail(60).std()

        # Get last sequence for recursive start
        last_sequence = X[-1]

        # Generate future prices
        future_prices = recursive_forecast_tft(
            model=model,
            last_sequence=last_sequence,
            scaler=scaler,
            n_steps=FUTURE_DAYS,
            daily_vol=recent_daily_vol
        )

        # Future dates (business days)
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
        model.save(os.path.join(models_folder, f"{symbol}_tft_recursive.h5"))
        joblib.dump(scaler, os.path.join(models_folder, f"{symbol}_scaler.pkl"))

        # Plot: Historical + Forecast
        plt.figure(figsize=(14, 7))
        historical_slice = df["Close"].iloc[-120:]
        plt.plot(historical_slice.index, historical_slice.values, label="Historical Close", color="blue", linewidth=2)
        plt.plot(future_dates, future_prices, "--o", label="Recursive TFT Forecast (with noise)", color="teal", alpha=0.8)

        plt.title(f"{symbol} — 30-Day Recursive TFT Forecast (LSTM Encoder-Decoder)", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(plots_folder, f"{symbol}_recursive_tft.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"{symbol}: Recursive forecast completed and saved.\n")

    except Exception as e:
        print(f"{symbol}: ERROR → {str(e)}\n")
        continue

print("Recursive TFT forecasting with stochastic noise completed for all stocks!")
