# week1_stock_analysis_50.py

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
import warnings
import time
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

# List of 50 stock symbols
stocks = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","V","UNH",
    "HD","PG","DIS","MA","BAC","NFLX","ADBE","PYPL","CMCSA","XOM",
    "PFE","KO","INTC","CSCO","NKE","MRK","PEP","ABBV","ABT","CRM",
    "ORCL","T","VZ","CVX","MCD","WMT","DHR","ACN","LLY","AVGO",
    "QCOM","TXN","NEE","COST","TMUS","IBM","SBUX","MDT","HON","AMD"
]

# Create necessary folders
os.makedirs("../data/raw", exist_ok=True)
os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../plots", exist_ok=True)

processed_stocks = []
skipped_stocks = []

# Timestamp for unique file names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for symbol in stocks:
    print(f"\n--- Processing {symbol} ---")
    
    raw_file = f"../data/raw/{symbol}_raw.csv"
    processed_file = f"../data/processed/{symbol}_processed_{timestamp}.csv"
    plot_file = f"../plots/{symbol}_price_trend_{timestamp}.png"
    
    # 1. Load or download data
    if os.path.exists(raw_file):
        print(f"{symbol} raw data already exists. Loading from file.")
        data = pd.read_csv(raw_file, index_col=0, parse_dates=True)
    else:
        try:
            data = yf.download(symbol, start="2015-01-01", end="2025-01-01")
            if data.empty:
                print(f"Warning: {symbol} data is empty. Skipping.")
                skipped_stocks.append(symbol)
                continue
            data.to_csv(raw_file)
            time.sleep(1)  # Delay to avoid Yahoo blocking
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            skipped_stocks.append(symbol)
            continue
    
    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Convert numeric columns safely
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in numeric_cols:
        if col in data.columns and isinstance(data[col], pd.Series):
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Use Adj Close if Close is missing
    if 'Close' not in data.columns and 'Adj Close' in data.columns:
        data['Close'] = data['Adj Close']
    
    if 'Close' not in data.columns or data['Close'].isna().all():
        print(f"Warning: {symbol} has no valid Close data. Skipping.")
        skipped_stocks.append(symbol)
        continue
    
    
    # 1. Drop rows with missing Close
    data.dropna(subset=['Close'], inplace=True)
    
    # 2. Forward-fill other numeric columns
    for col in ['Open', 'High', 'Low', 'Adj Close', 'Volume']:
        if col in data.columns and isinstance(data[col], pd.Series):
            data[col].fillna(method='ffill', inplace=True)
    
    # 3. Remove duplicate dates
    data = data[~data.index.duplicated(keep='first')]
    
    # 4. Sort by date
    data.sort_index(inplace=True)
    
    # Feature Engineering
    data['Return'] = data['Close'].pct_change()
    data['MA_20'] = data['Close'].rolling(20).mean()
    data['Volatility'] = data['Return'].rolling(20).std()
    
    # Save processed data
    data.to_csv(processed_file)
    
    # Plot closing price and MA
    plt.figure(figsize=(12,6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['MA_20'], label='20-day MA', linestyle='--')
    plt.title(f"{symbol} Stock Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(plot_file)
    plt.close()
    
    processed_stocks.append(symbol)

print("\nWeek 1: Stocks processed successfully.")
print("Processed stocks:", processed_stocks)
print("Skipped stocks:", skipped_stocks)
