import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.optimizers import Adam

from data_loader import load_processed 

# ============================
# PATHS
# ============================
plots_folder = "../plots_nbeats"
model_folder = "../saved_models/nbeats"
predictions_folder = "../data/predictions/NBEATS_predictions"

os.makedirs(plots_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)
os.makedirs(predictions_folder, exist_ok=True)


# ============================
# STOCK LIST
# ============================
stocks = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA",
    "META","NVDA","JPM","V","UNH",
]


# ============================
# SEQUENCE CREATION
# ============================
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)


# ============================
# N-BEATS BLOCK
# ============================
def nbeats_block(inputs, units=128, name="block"):
    x = Dense(units, activation='relu')(inputs)
    x = Dense(units, activation='relu')(x)

    backcast = Dense(inputs.shape[1], name=f"{name}_backcast")(x)
    forecast = Dense(1, name=f"{name}_forecast")(x)

    return backcast, forecast


# ============================
# N-BEATS MODEL BUILDER
# ============================
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


# ============================
# TRAINING LOOP
# ============================
for symbol in stocks:
    print(f"\n============================")
    print(f"   PROCESSING {symbol}")
    print(f"============================")

    try:
        df = load_processed(symbol)
        df.dropna(subset=["Close"], inplace=True)

        # Scaling
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

        time_steps = 60
        X, y = create_sequences(data, time_steps)
        X = X.reshape(X.shape[0], X.shape[1])

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # ----------------------------
        # Build & Train Model
        # ----------------------------
        model = build_nbeats(input_dim=time_steps, blocks=4, units=128, lr=0.001)
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

        # ----------------------------
        # Save Model
        # ----------------------------
        symbol_model_dir = os.path.join(model_folder, symbol)
        os.makedirs(symbol_model_dir, exist_ok=True)
        model.save(os.path.join(symbol_model_dir, "model.h5"))
        print(f"Saved model for {symbol}")

        # ----------------------------
        # Predictions
        # ----------------------------
        y_pred = model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Save predictions per stock
        pred_df = pd.DataFrame({
            "Actual": y_test_inv.flatten(),
            "Predicted": y_pred_inv.flatten()
        })
        pred_file = os.path.join(predictions_folder, f"{symbol}_predictions.csv")
        pred_df.to_csv(pred_file, index=False)
        print(f"Saved predictions to {pred_file}")

        # ----------------------------
        # Metrics
        # ----------------------------
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        accuracy = 100 * (1 - (mae / np.mean(y_test_inv)))

        metrics_df = pd.DataFrame([{
            "Symbol": symbol,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Accuracy(%)": accuracy
        }])
        metrics_file = os.path.join(predictions_folder, f"{symbol}_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Saved metrics to {metrics_file}")

        # ----------------------------
        # Save Plot
        # ----------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(y_test_inv, label="Actual")
        plt.plot(y_pred_inv, label="Predicted", linestyle="--")
        plt.title(f"{symbol} - NBEATS Prediction vs Actual")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"{symbol}_nbeats.png"))
        plt.close()

    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        continue

print("\nAll stocks processed successfully!")
