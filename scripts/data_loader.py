# training/data_loader.py

import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler

def load_stock(symbol, processed_folder="../data/processed"):
    """
    Load the latest processed CSV for a stock symbol.
    Returns a pandas DataFrame.
    """
    files = glob.glob(f"{processed_folder}/{symbol}_processed*.csv")
    if not files:
        return None
    df = pd.read_csv(sorted(files)[-1], parse_dates=['Date'], index_col='Date')
    return df

def get_features_targets(df, target_col='Close', features=['Close', 'MA_20', 'Volatility', 'Volume']):
    """
    Prepare feature matrix X and target vector y.
    Align target with features (next day prediction).
    """
    df = df.dropna()
    y = df[target_col].shift(-1).dropna()
    X = df[features].iloc[:-1]  # align X with y
    return X, y

def train_val_test_split_time_series(X, y, train_ratio=0.8, val_ratio=0.1):
    """
    Split features and target into train, validation, and test sets for time series.
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test

def create_lstm_sequences(data, seq_len=60):
    """
    Convert feature array into sequences for LSTM input.
    data: numpy array with shape (n_samples, n_features)
    Returns X_seq, y_seq
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])  # predict first column (Close)
    return np.array(X), np.array(y)
def load_processed(symbol, processed_folder="../data/processed"):
    """
    Compatibility wrapper for scripts expecting load_processed().
    Loads the latest processed CSV for a given stock.
    """
    return load_stock(symbol, processed_folder)
