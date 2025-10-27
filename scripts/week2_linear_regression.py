# week1_stock_linear_regression_auto.py

import pandas as pd
import numpy as np
import os
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# List of 50 stock symbols
stocks = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","V","UNH",
    "HD","PG","DIS","MA","BAC","NFLX","ADBE","PYPL","CMCSA","XOM",
    "PFE","KO","INTC","CSCO","NKE","MRK","PEP","ABBV","ABT","CRM",
    "ORCL","T","VZ","CVX","MCD","WMT","DHR","ACN","LLY","AVGO",
    "QCOM","TXN","NEE","COST","TMUS","IBM","SBUX","MDT","HON","AMD"
]

# Folders
processed_folder = "../data/processed"
predictions_folder = "../data/predictions"
plots_folder = "../plots/linear_regression_auto"
os.makedirs(predictions_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)

processed_stocks = []
skipped_stocks = []

for symbol in stocks:
    print(f"\n--- Processing {symbol} ---")
    
    # Find latest processed CSV for the symbol
    files = glob.glob(f"{processed_folder}/{symbol}_processed*.csv")
    if files:
        processed_file = sorted(files)[-1]  # pick the latest file
    else:
        print(f"Processed file for {symbol} not found. Skipping.")
        skipped_stocks.append(symbol)
        continue
    
    # Load processed data
    data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
    
    # Ensure Close column exists
    if 'Close' not in data.columns or data['Close'].isna().all():
        print(f"No valid Close data for {symbol}. Skipping.")
        skipped_stocks.append(symbol)
        continue
    
    # Prepare target: next day Close
    data['Close_next'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    
    # Features selection
    feature_cols = ['Close']
    for col in ['MA_20', 'Volatility', 'Volume']:
        if col in data.columns:
            feature_cols.append(col)
    
    X = data[feature_cols]
    y = data['Close_next']
    
    # Split data (80% train, 20% test, no shuffle for time series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{symbol} MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Save predictions
    predictions = pd.DataFrame({
        'Actual_Close': y_test,
        'Predicted_Close': y_pred
    })
    predictions_file = os.path.join(predictions_folder, f"{symbol}_predictions.csv")
    predictions.to_csv(predictions_file)
    
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(predictions.index, predictions['Actual_Close'], label="Actual Close")
    plt.plot(predictions.index, predictions['Predicted_Close'], linestyle='--', label="Predicted Close")
    plt.title(f"{symbol} Linear Regression: Actual vs Predicted Close (Multi-feature)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plot_file = os.path.join(plots_folder, f"{symbol}_lr_plot.png")
    plt.savefig(plot_file)
    plt.close()
    
    processed_stocks.append(symbol)

print("\nLinear Regression (multi-feature) processing complete.")
print("Processed stocks:", processed_stocks)
print("Skipped stocks:", skipped_stocks)
