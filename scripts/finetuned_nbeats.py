import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Add, Layer
from tensorflow.keras.optimizers import Adam

from data_loader import load_processed  # Your existing data loader


plots_folder = "../plots_nbeats"
model_folder = "../saved_models/nbeats"
metrics_output = "../data/nbeats_finetuned_metrics.csv"

os.makedirs(plots_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)


stocks = ["AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","V","UNH"]


class Negative(Layer):
    def call(self, inputs):
        return -inputs


def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)


def nbeats_block(inputs, units=128, name="block"):
    x = Dense(units, activation="relu")(inputs)
    x = Dense(units, activation="relu")(x)
    backcast = Dense(inputs.shape[1], name=f"{name}_backcast")(x)
    forecast = Dense(1, name=f"{name}_forecast")(x)
    return backcast, forecast

def build_nbeats(input_dim, blocks=4, units=128, lr=0.001):
    inputs = Input(shape=(input_dim,))
    backcast, forecast = nbeats_block(inputs, units, name="block_1")
    for i in range(2, blocks + 1):
        residual = Add()([inputs, -backcast])
        backcast_i, forecast_i = nbeats_block(residual, units, name=f"block_{i}")
        backcast = Add()([backcast, backcast_i])
        forecast = Add()([forecast, forecast_i])
    model = Model(inputs=inputs, outputs=forecast)
    model.compile(optimizer=Adam(lr), loss="mse")
    return model


results = []

for symbol in stocks:
    print(f"\n=== Fine-tuning {symbol} ===")
    try:
        df = load_processed(symbol)
        if df is None or df.empty:
            print(f"No data for {symbol}, skipping.")
            continue
        df.dropna(subset=["Close"], inplace=True)
        
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df["Close"].values.reshape(-1,1))

        time_steps = 60
        X, y = create_sequences(data, time_steps)
        X = X.reshape(X.shape[0], X.shape[1])

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        
        symbol_model_dir = os.path.join(model_folder, symbol)
        pretrained_model_path = os.path.join(symbol_model_dir, "model.h5")

        if os.path.exists(pretrained_model_path):
            print(f"Loading pretrained model for {symbol}...")
            model = load_model(pretrained_model_path, compile=False, custom_objects={"Negative": Negative})
            model.compile(optimizer=Adam(0.0005), loss="mse")  
        else:
            print(f"No pretrained model found for {symbol}. Training new model...")
            model = build_nbeats(input_dim=time_steps, blocks=4, units=128, lr=0.001)

        
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        
        os.makedirs(symbol_model_dir, exist_ok=True)
        model.save(os.path.join(symbol_model_dir, "model_finetuned.h5"))

        
        y_pred = model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        accuracy = 100 * (1 - (mae / np.mean(y_test_inv)))

        results.append({
            "Symbol": symbol,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Accuracy(%)": accuracy
        })

        print(f"Metrics for {symbol}: RMSE={rmse:.2f}, R2={r2:.3f}, Acc={accuracy:.2f}%")

       
        plt.figure(figsize=(10,5))
        plt.plot(y_test_inv, label="Actual")
        plt.plot(y_pred_inv, label="Predicted", linestyle="--")
        plt.title(f"{symbol} - Fine-tuned NBEATS Prediction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"{symbol}_nbeats_finetuned.png"))
        plt.close()

    except Exception as e:
        print(f"Error fine-tuning {symbol}: {e}")
        continue


pd.DataFrame(results).to_csv(metrics_output, index=False)
print(f"\nSaved fine-tuned metrics to {metrics_output}")
