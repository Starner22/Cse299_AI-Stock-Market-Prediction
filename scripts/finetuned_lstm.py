
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from data_loader import load_stock, get_features_targets
from sklearn.preprocessing import MinMaxScaler
import joblib


OLD_MODEL_DIR = "../models/LSTM_models"
MODEL_DIR = "../outputs/models/lstm_finetuned"
SCALER_DIR = "../outputs/models/lstm"
HPARAMS_FILE = "../outputs/hparams/lstm_best_params.json"
CSV_DIR = "../outputs/csv"
PLOT_DIR = "../outputs/plots"

for d in [MODEL_DIR, CSV_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)


with open(HPARAMS_FILE, 'r') as f:
    best_params_all = json.load(f)

stocks = list(best_params_all.keys())

for stock in stocks:
    print(f"\n--- Fine-tuning LSTM for {stock} ---")
    
    
    df = load_stock(stock)
    feature_cols = ['Close', 'MA_20', 'Volatility', 'Volume']
    df_cols = [col for col in feature_cols if col in df.columns]
    
    X, y = get_features_targets(df, features=df_cols)
    
    
    scaler_X = joblib.load(f"{SCALER_DIR}/{stock}_scaler_X.pkl")
    scaler_y = joblib.load(f"{SCALER_DIR}/{stock}_scaler_y.pkl")
    
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.values.reshape(-1,1))
    
    seq_len = best_params_all[stock]["seq_len"]
    
    
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq.append(y_scaled[i+seq_len])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    
    model = load_model(f"{OLD_MODEL_DIR}/{stock}_lstm.h5", compile=False)
    model.compile(optimizer='adam', loss='mse')  # compile to allow fine-tuning
    
    
    y_pred_before = model.predict(X_seq, verbose=0)
    
    
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_seq, y_seq, epochs=10, batch_size=32, callbacks=[es], verbose=1)
    
    
    y_pred_after = model.predict(X_seq, verbose=0)
    
    
    model.save(f"{MODEL_DIR}/{stock}_lstm_finetuned.h5")
    
    
    y_pred_before_inv = scaler_y.inverse_transform(y_pred_before)
    y_pred_after_inv = scaler_y.inverse_transform(y_pred_after)
    y_actual_inv = scaler_y.inverse_transform(y_seq)
    
    
    df_save = pd.DataFrame({
        "Actual": y_actual_inv.flatten(),
        "Predicted_Before": y_pred_before_inv.flatten(),
        "Predicted_After": y_pred_after_inv.flatten()
    })
    df_save.to_csv(f"{CSV_DIR}/{stock}_predictions.csv", index=False)
    
    
    plt.figure(figsize=(12,6))
    plt.plot(df.index[seq_len:seq_len+len(df_save)], df_save["Actual"], label="Actual", color="blue")
    plt.plot(df.index[seq_len:seq_len+len(df_save)], df_save["Predicted_Before"], label="Before Fine-tune", color="orange")
    plt.plot(df.index[seq_len:seq_len+len(df_save)], df_save["Predicted_After"], label="After Fine-tune", color="green")
    plt.title(f"{stock} LSTM Predictions Comparison")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{stock}_comparison.png")
    plt.close()
    
    print(f"✅ {stock} fine-tuned. CSV & plot saved.")

print("\n🎉 Fine-tuning complete for all stocks.")
