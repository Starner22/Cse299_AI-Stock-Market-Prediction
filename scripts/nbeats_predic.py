import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from data_loader import load_processed
import warnings
warnings.filterwarnings("ignore")

# ---- Directories ----
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

predictions_folder = os.path.join(BASE_DIR, "data/predictions/NBEATS_recursive")
plots_folder = os.path.join(BASE_DIR, "plots/NBEATS_recursive")
models_folder = os.path.join(BASE_DIR, "models/NBEATS_recursive")

for folder in [predictions_folder, plots_folder, models_folder]:
    os.makedirs(folder, exist_ok=True)

# ---- Stocks (top 10 for consistency) ----
stocks = [
    "HD","PG","DIS","MA","BAC","NFLX","ADBE","PYPL","CMCSA","XOM",
    "PFE","KO","INTC","CSCO","NKE","MRK","PEP","ABBV","ABT","CRM",
    "ORCL","T","VZ","CVX","MCD","WMT","DHR","ACN","LLY","AVGO",
    "QCOM","TXN","NEE","COST","TMUS","IBM","SBUX","MDT","HON","AMD"
]

TIME_STEPS = 60
FUTURE_DAYS = 30
NOISE_FACTOR = 1.0  # Adjust to increase/decrease forecast volatility
np.random.seed(42)  # Reproducibility

# ============================
# N-BEATS BLOCK & MODEL
# ============================
def nbeats_block(inputs, units=128, name="block"):
    x = Dense(units, activation='relu')(inputs)
    x = Dense(units, activation='relu')(x)
    backcast = Dense(inputs.shape[1])(x)  # No activation
    forecast = Dense(1)(x)
    return backcast, forecast

def build_nbeats(input_dim, blocks=4, units=128, lr=0.001):
    inputs = Input(shape=(input_dim,))
    backcast, forecast = nbeats_block(inputs, units, name="block_1")

    for i in range(2, blocks + 1):
        residual = Add()([inputs, -backcast])  # Subtract backcast
        backcast_i, forecast_i = nbeats_block(residual, units, name=f"block_{i}")
        backcast = Add()([backcast, backcast_i])
        forecast = Add()([forecast, forecast_i])

    model = Model(inputs=inputs, outputs=forecast)
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model

# ============================
# RECURSIVE FORECAST FUNCTION
# ============================
def recursive_forecast_nbeats(model, last_sequence_scaled, scaler, n_steps, daily_vol):
    """
    Recursively generate future prices using N-BEATS.
    N-BEATS takes a fixed-length window and outputs one step → perfect for recursion.
    """
    predictions = []
    current_seq = last_sequence_scaled.copy().flatten()  # Shape: (TIME_STEPS,)

    for _ in range(n_steps):
        # Predict next step (scaled)
        pred_scaled = model.predict(current_seq.reshape(1, -1), verbose=0)[0, 0]
        pred_price = scaler.inverse_transform([[pred_scaled]])[0, 0]

        # Add stochastic noise
        noise = np.random.normal(0, daily_vol * NOISE_FACTOR)
        noisy_price = pred_price * (1 + noise)

        predictions.append(noisy_price)

        # Shift sequence: drop oldest, append new scaled noisy value
        noisy_scaled = scaler.transform([[noisy_price]])[0, 0]
        current_seq = np.append(current_seq[1:], noisy_scaled)

    return predictions

# ============================
# MAIN LOOP
# ============================
print("Starting Recursive N-BEATS Forecasting with Stochastic Noise...\n")

for symbol in stocks:
    print(f"--- Processing {symbol} ---")

    try:
        df = load_processed(symbol)
        if df is None or "Close" not in df.columns or len(df) < TIME_STEPS + 100:
            print(f"{symbol}: Insufficient data. Skipping.\n")
            continue

        close_prices = df["Close"].values.reshape(-1, 1)

        # Scale
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)

        # Create sequences
        X, y = [], []
        for i in range(TIME_STEPS, len(scaled_data)):
            X.append(scaled_data[i-TIME_STEPS:i].flatten())
            y.append(scaled_data[i])
        X = np.array(X)
        y = np.array(y)

        if len(X) < 50:
            print(f"{symbol}: Not enough sequences. Skipping.\n")
            continue

        # Train on 85%
        train_size = int(len(X) * 0.85)
        X_train, y_train = X[:train_size], y[:train_size]

        # Build & train N-BEATS
        model = build_nbeats(input_dim=TIME_STEPS, blocks=4, units=128, lr=0.001)
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0
        )

        # Recent volatility
        returns = pd.Series(close_prices.flatten()).pct_change().dropna()
        recent_daily_vol = returns.tail(60).std()

        # Last input sequence
        last_sequence_scaled = scaled_data[-TIME_STEPS:]

        # Generate 30-day forecast
        future_prices = recursive_forecast_nbeats(
            model=model,
            last_sequence_scaled=last_sequence_scaled,
            scaler=scaler,
            n_steps=FUTURE_DAYS,
            daily_vol=recent_daily_vol
        )

        # Future dates
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
        symbol_model_dir = os.path.join(models_folder, symbol)
        os.makedirs(symbol_model_dir, exist_ok=True)
        model.save(os.path.join(symbol_model_dir, "nbeats_recursive.h5"))
        joblib.dump(scaler, os.path.join(symbol_model_dir, "scaler.pkl"))

        # Plot
        plt.figure(figsize=(14, 7))
        historical_slice = df["Close"].iloc[-120:]
        plt.plot(historical_slice.index, historical_slice.values, label="Historical Close", color="blue", linewidth=2)
        plt.plot(future_dates, future_prices, "--o", label="Recursive N-BEATS Forecast (with noise)", color="darkorange", alpha=0.9)

        plt.title(f"{symbol} — 30-Day Recursive N-BEATS Forecast", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(plots_folder, f"{symbol}_recursive_nbeats.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"{symbol}: Recursive forecast completed and saved.\n")

    except Exception as e:
        print(f"{symbol}: ERROR → {str(e)}\n")
        continue

print("Recursive N-BEATS forecasting with stochastic noise completed!")
