# week3_xgboost_latest.py

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


processed_folder = "../data/processed"
metrics_output = "../data/xgboost_metrics.csv"
plots_folder = "../plots_xgboost"
os.makedirs(plots_folder, exist_ok=True)


stocks = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","V","UNH",
    "HD","PG","DIS","MA","BAC","NFLX","ADBE","PYPL","CMCSA","XOM",
    "PFE","KO","INTC","CSCO","NKE","MRK","PEP","ABBV","ABT","CRM",
    "ORCL","T","VZ","CVX","MCD","WMT","DHR","ACN","LLY","AVGO",
    "QCOM","TXN","NEE","COST","TMUS","IBM","SBUX","MDT","HON","AMD"
]

results = []

def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

for symbol in stocks:
    print(f"\n--- Processing {symbol} ---")

    
    files = glob.glob(os.path.join(processed_folder, f"{symbol}_processed*.csv"))
    if not files:
        print(f"Processed file for {symbol} not found. Skipping.")
        continue

    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=["Close"], inplace=True)

        close_prices = df["Close"].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        time_steps = 60
        X, y = create_sequences(scaled_data, time_steps)

      
        X = X.reshape((X.shape[0], X.shape[1]))

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        accuracy = 100 * (1 - (mae / np.mean(y_test_inv)))

        print(f"{symbol}: RMSE={rmse:.2f}, R²={r2:.3f}, Accuracy={accuracy:.2f}%")

        results.append({
            "Symbol": symbol,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Accuracy(%)": accuracy
        })

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(y_test_inv, label="Actual")
        plt.plot(y_pred_inv, label="Predicted", linestyle='--')
        plt.title(f"{symbol} - XGBoost Prediction vs Actual")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"{symbol}_xgboost_plot.png"))
        plt.close()

    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        continue


if results:
    pd.DataFrame(results).to_csv(metrics_output, index=False)
    print(f"\nMetrics saved to {metrics_output}")
else:
    print("\nNo results to save. Check data files.")

print("\nXGBoost training completed!")