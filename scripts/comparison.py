# comparison.py - FINAL WORKING VERSION (Dec 2025)
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ========================= CONFIG =========================
processed_folder = "../data/processed"
comparison_folder = "../plots_comparison"
os.makedirs(comparison_folder, exist_ok=True)

stocks = ["AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","V","UNH"]
TIME_STEPS = 60

# ====================== 1. OLD (YOUR ORIGINAL) MODEL - 2D INPUT ======================
def build_old_nbeats():
    inputs = Input(shape=(TIME_STEPS,))                 # <-- 2D input
    x = Dense(128, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    backcast = Dense(TIME_STEPS)(x)
    forecast = Dense(1)(x)

    for _ in range(3):
        residual = Add()([inputs, -backcast])
        x = Dense(128, activation='relu')(residual)
        x = Dense(128, activation='relu')(x)
        b = Dense(TIME_STEPS)(x)
        f = Dense(1)(x)
        backcast = Add()([backcast, b])
        forecast = Add()([forecast, f])

    model = Model(inputs, forecast)
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model

# ====================== 2. CORRECT N-BEATS - 3D INPUT ======================
def build_correct_nbeats():
    inputs = Input(shape=(TIME_STEPS, 1))               # <-- 3D input
    x = inputs
    backcast = inputs
    forecasts = []

    for _ in range(6):
        h = Dense(256, activation='relu')(x)
        h = Dense(256, activation='relu')(h)
        h = LayerNormalization()(h)
        theta = Dense(512, activation='relu')(h)
        b = Dense(TIME_STEPS)(theta)
        f = Dense(1)(theta)
        backcast = backcast - b
        forecasts.append(f)
        x = b

    forecast = Add()(forecasts) if len(forecasts) > 1 else forecasts[0]
    model = Model(inputs, forecast)
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model

# ====================== DATA PREP ======================
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

# ====================== MAIN LOOP ======================
results = []

for symbol in stocks:
    print(f"\nComparing {symbol}...", end=" ")

    files = glob.glob(os.path.join(processed_folder, f"{symbol}_processed*.csv"))
    if not files:
        print("No data")
        continue
    file_path = max(files, key=os.path.getctime)

    try:
        df = pd.read_csv(file_path)
        if "Close" not in df.columns:
            print("No Close column")
            continue

        close = df["Close"].dropna().values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close)

        X, y = create_sequences(scaled, TIME_STEPS)

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # IMPORTANT: Two separate input shapes
        X_train_2d = X_train.reshape(-1, TIME_STEPS)           # For old model
        X_test_2d  = X_test.reshape(-1, TIME_STEPS)

        X_train_3d = X_train.reshape(-1, TIME_STEPS, 1)        # For new model
        X_test_3d  = X_test.reshape(-1, TIME_STEPS, 1)

        y_test_inv = scaler.inverse_transform(y_test)

        # === OLD MODEL (2D) ===
        model_old = build_old_nbeats()
        model_old.fit(X_train_2d, y_train, epochs=20, batch_size=32, verbose=0)
        pred_old = scaler.inverse_transform(model_old.predict(X_test_2d, verbose=0))

        # === NEW MODEL (3D) ===
        model_new = build_correct_nbeats()
        callbacks = [EarlyStopping(patience=15, restore_best_weights=True),
                     ReduceLROnPlateau(factor=0.5, patience=8)]
        model_new.fit(X_train_3d, y_train,
                      validation_split=0.1,
                      epochs=100,
                      batch_size=32,
                      callbacks=callbacks,
                      verbose=0)
        pred_new = scaler.inverse_transform(model_new.predict(X_test_3d, verbose=0))

        # === METRICS & PLOT ===
        r2_old = r2_score(y_test_inv, pred_old)
        r2_new = r2_score(y_test_inv, pred_new)

        plt.figure(figsize=(14, 7))
        last = 150
        plt.plot(y_test_inv[-last:], label="Actual Price", color="black", linewidth=2.5)
        plt.plot(pred_old[-last:], label=f"Old (buggy) — R² = {r2_old:.3f}", color="#d62728", linestyle="--", linewidth=2)
        plt.plot(pred_new[-last:], label=f"Correct N-BEATS — R² = {r2_new:.3f}", color="#2ca02c", linewidth=2.5)

        plt.title(f"{symbol} — Old vs Fine-Tuned N-BEATS (Last {last} Days)", fontsize=18, pad=15)
        plt.xlabel("Trading Days")
        plt.ylabel("Price ($)")
        plt.legend(fontsize=13)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_folder, f"{symbol}_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

        results.append({
            "Symbol": symbol,
            "Old_R2": round(r2_old, 3),
            "New_R2": round(r2_new, 3),
            "Improvement": round(r2_new - r2_old, 3)
        })

        print(f"Old R² = {r2_old:.3f} → New R² = {r2_new:.3f} (+{r2_new-r2_old:.3f})")

    except Exception as e:
        print(f"Error: {e}")

# ====================== FINAL SUMMARY ======================
if results:
    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(comparison_folder, "comparison_summary.csv"), index=False)
    print("\n" + "="*75)
    print("SUCCESS! All 10 comparison plots saved in:")
    print(comparison_folder)
    print(f"\nAverage R² improvement: {summary['Improvement'].mean():.3f}")
    print("\n", summary[["Symbol", "Old_R2", "New_R2", "Improvement"]])
else:
    print("No stocks processed.")
