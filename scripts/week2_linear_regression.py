import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from data_loader import load_stock, get_features_targets, train_val_test_split_time_series
import joblib

# ---- Set Project Root ----
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))  # Go up one level from scripts/
print("Project root:", BASE_DIR)

# ---- Folder Setup ----
predictions_folder = os.path.join(BASE_DIR, "data/predictions/LR_predictions")
plots_folder = os.path.join(BASE_DIR, "plots/LR_plots")    # LR_plots subfolder
models_folder = os.path.join(BASE_DIR, "models/LR_models")

# Create folders if they don't exist
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

# ---- Main Loop ----
for symbol in stocks:
    print(f"\n--- Processing {symbol} ---")

    try:
        df = load_stock(symbol)
        if df is None:
            print(f"{symbol}: processed CSV not found. Skipping.")
            skipped_stocks.append(symbol)
            continue

        # Features
        feature_cols = ['Close', 'MA_20', 'Volatility', 'Volume']
        df_cols = [col for col in feature_cols if col in df.columns]

        if len(df_cols) == 0:
            print(f"{symbol}: no valid features. Skipping.")
            skipped_stocks.append(symbol)
            continue

        X, y = get_features_targets(df, features=df_cols)

        # Split
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_time_series(X, y)

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Metrics
        y_test_safe = np.where(y_test == 0, 1e-8, y_test)  # Avoid division by zero
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test_safe)) * 100
        r2 = r2_score(y_test, y_pred)

        print(f"{symbol} | RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")

        # Predictions DataFrame
        predictions_df = pd.DataFrame({
            'Date': df.index[-len(y_test):],
            'Actual_Close': y_test,
            'Predicted_Close': y_pred
        })
        predictions_path = os.path.join(predictions_folder, f"{symbol}_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)

        # Metrics CSV
        metrics_df = pd.DataFrame([{
            'Symbol': symbol,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }])
        metrics_path = os.path.join(predictions_folder, f"{symbol}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

        # Save Model + Scaler
        model_path = os.path.join(models_folder, f"{symbol}_lr.pkl")
        scaler_path = os.path.join(models_folder, f"{symbol}_scaler.pkl")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(predictions_df['Date'], predictions_df['Actual_Close'], label='Actual')
        plt.plot(predictions_df['Date'], predictions_df['Predicted_Close'], linestyle='--', label='Predicted')
        plt.title(f"{symbol} Linear Regression: Actual vs Predicted")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plot_path = os.path.join(plots_folder, f"{symbol}_lr_plot.png")
        plt.savefig(plot_path)
        plt.close()

        processed_stocks.append(symbol)

    except Exception as e:
        print(f"{symbol}: ERROR → {e}")
        skipped_stocks.append(symbol)
        continue

print("\nLinear Regression processing complete.")
print("Processed stocks:", processed_stocks)
print("Skipped stocks:", skipped_stocks)