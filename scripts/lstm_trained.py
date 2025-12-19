# train_lstm.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
from data_loader import load_stock, get_features_targets

MODEL_DIR = "../outputs/models/lstm"
SCALER_DIR = MODEL_DIR
PRED_DIR = "../outputs/predictions"
PLOT_DIR = "../outputs/plots"
HPARAMS_FILE = "../outputs/hparams/lstm_best_params.json"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Example hyperparameters for stocks (can tune later)
best_params_all = {
    "AAPL": {"seq_len": 10, "hidden_units": 100, "dropout": 0.1},
    "MSFT": {"seq_len": 10, "hidden_units": 100, "dropout": 0.1}
}

# Save hyperparameters
with open(HPARAMS_FILE, 'w') as f:
    json.dump(best_params_all, f, indent=4)

stocks = list(best_params_all.keys())

for stock in stocks:
    print(f"\n--- Training LSTM for {stock} ---")
    
    df = load_stock(stock)
    feature_cols = ['Close', 'MA_20', 'Volatility', 'Volume']
    df_cols = [col for col in feature_cols if col in df.columns]
    
    X, y = get_features_targets(df, features=df_cols)
    
    # Scaling
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    # Save scalers
    joblib.dump(scaler_X, f"{SCALER_DIR}/{stock}_scaler_X.pkl")
    joblib.dump(scaler_y, f"{SCALER_DIR}/{stock}_scaler_y.pkl")
    
    seq_len = best_params_all[stock]["seq_len"]
    
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq.append(y_scaled[i+seq_len])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Build LSTM
    hidden_units = best_params_all[stock]["hidden_units"]
    dropout = best_params_all[stock]["dropout"]
    
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=(seq_len, X_seq.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_seq, y_seq, epochs=20, batch_size=32, callbacks=[es], verbose=1)
    
    # Save model
    model.save(f"{MODEL_DIR}/{stock}_lstm.h5")
    
    # Predictions (before fine-tuning)
    y_pred_scaled = model.predict(X_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Save CSV
    np.savetxt(f"{PRED_DIR}/{stock}_before.csv",
               np.hstack([y[seq_len:].values.reshape(-1,1), y_pred]),
               delimiter=",", header="Actual,Predicted", comments="")
    
    # Save plot
    plt.figure(figsize=(10,5))
    plt.plot(y[seq_len:].values, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title(f"{stock} - Before Fine-Tuning")
    plt.legend()
    plt.savefig(f"{PLOT_DIR}/{stock}_before.png")
    plt.close()

print("\n✅ Training complete. Models, predictions, and plots saved.")
