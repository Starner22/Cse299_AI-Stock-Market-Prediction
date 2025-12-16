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


# ---- Set Project Root ----
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
print("Project root:", BASE_DIR)

# ---- Folder Setup ----
predictions_folder = os.path.join(BASE_DIR, "data/predictions/LSTM_predictions")
plots_folder = os.path.join(BASE_DIR, "plots/LSTM_plots")
models_folder = os.path.join(BASE_DIR, "models/LSTM_models")

for folder in [predictions_folder, plots_folder, models_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")

# ---- Stock List ----
stocks = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","V","UNH",
    "HD","PG","DIS","MA","BAC","NFLX","ADBE","PYPL","CMCSA","XOM",
    "PFE","KO","INTC","CSCO","NKE","MRK","PEP","ABBV","ABT","CRM",
    "ORCL","T","VZ","CVX","MCD","WMT","DHR","ACN","LLY","AVGO",
    "QCOM","TXN","NEE","COST","TMUS","IBM","SBUX","MDT","HON","AMD"
]

processed_stocks = []
skipped_stocks = []

# ---- Helper function to create sequences ----
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # predict 'Close' price
    return np.array(X), np.array(y)

# ---- Main Loop ----
SEQ_LENGTH = 60  # lookback window

for symbol in stocks:
    print(f"\n--- Processing {symbol} ---")
    try:
        df = load_stock(symbol)  # Your custom function
        if df is None or 'Close' not in df.columns:
            print(f"{symbol}: data not found or missing 'Close'. Skipping.")
            skipped_stocks.append(symbol)
            continue

        # Use only Close price for simplicity
        close_data = df['Close'].values.reshape(-1, 1)

        # Scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_data)

        # Create sequences
        X, y = create_sequences(scaled_data, SEQ_LENGTH)

        # Split train/val/test (70/15/15)
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                  callbacks=[early_stop], verbose=0)

        # Predict
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Metrics
        mse = np.mean((y_test_actual - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test_actual - y_pred) / np.where(y_test_actual==0, 1e-8, y_test_actual))) * 100
        r2 = 1 - np.sum((y_test_actual - y_pred)**2)/np.sum((y_test_actual - np.mean(y_test_actual))**2)

        print(f"{symbol} | RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")

        # Save predictions
        predictions_df = pd.DataFrame({
            'Date': df.index[-len(y_test_actual):],
            'Actual_Close': y_test_actual.flatten(),
            'Predicted_Close': y_pred.flatten()
        })
        predictions_df.to_csv(os.path.join(predictions_folder, f"{symbol}_predictions.csv"), index=False)

        # Save metrics
        metrics_df = pd.DataFrame([{
            'Symbol': symbol,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }])
        metrics_df.to_csv(os.path.join(predictions_folder, f"{symbol}_metrics.csv"), index=False)

        # Save model
        model.save(os.path.join(models_folder, f"{symbol}_lstm.h5"))
        joblib.dump(scaler, os.path.join(models_folder, f"{symbol}_scaler.pkl"))

        # Plot
        plt.figure(figsize=(12,6))
        plt.plot(predictions_df['Date'], predictions_df['Actual_Close'], label='Actual')
        plt.plot(predictions_df['Date'], predictions_df['Predicted_Close'], linestyle='--', label='Predicted')
        plt.title(f"{symbol} LSTM: Actual vs Predicted")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.savefig(os.path.join(plots_folder, f"{symbol}_lstm_plot.png"))
        plt.close()

        processed_stocks.append(symbol)

    except Exception as e:
        print(f"{symbol}: ERROR → {e}")
        skipped_stocks.append(symbol)
        continue

print("\nLSTM processing complete.")
print("Processed stocks:", processed_stocks)
print("Skipped stocks:", skipped_stocks)
