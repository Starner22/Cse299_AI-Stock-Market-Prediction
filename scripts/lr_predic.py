import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
from data_loader import load_stock
import warnings
warnings.filterwarnings("ignore")

# ---- Directories ----
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

predictions_folder = os.path.join(BASE_DIR, "data/predictions/LR_recursive")
plots_folder = os.path.join(BASE_DIR, "plots/LR_recursive")
models_folder = os.path.join(BASE_DIR, "models/LR_recursive")

for folder in [predictions_folder, plots_folder, models_folder]:
    os.makedirs(folder, exist_ok=True)

# ---- Stocks (same as LSTM recursive for consistency) ----
stocks = [
    "HD","PG","DIS","MA","BAC","NFLX","ADBE","PYPL","CMCSA","XOM",
    "PFE","KO","INTC","CSCO","NKE","MRK","PEP","ABBV","ABT","CRM",
    "ORCL","T","VZ","CVX","MCD","WMT","DHR","ACN","LLY","AVGO",
    "QCOM","TXN","NEE","COST","TMUS","IBM","SBUX","MDT","HON","AMD"
]

FUTURE_DAYS = 30
NOISE_FACTOR = 1.0  # Multiply volatility noise (1.0 = realistic level)
np.random.seed(42)  # For reproducibility

# Required features from your original LR script
FEATURE_COLS = ['Close', 'MA_20', 'Volatility', 'Volume']

def add_features(df):
    """Add rolling features used in original LR training"""
    df = df.copy()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
    df['Volume'] = df.get('Volume', pd.Series(np.nan, index=df.index))  # Keep if exists
    return df

def recursive_forecast_lr(model, scaler, last_row, recent_vol, n_steps=30):
    """
    Recursively forecast using Linear Regression by generating future features.
    """
    predictions = []
    current_features = last_row.copy()  # Last known feature row (as DataFrame)

    for _ in range(n_steps):
        # Scale features and predict
        scaled_features = scaler.transform(current_features[FEATURE_COLS].values.reshape(1, -1))
        pred_price = model.predict(scaled_features)[0]

        # Add stochastic noise based on recent volatility
        noise = np.random.normal(0, recent_vol * NOISE_FACTOR)
        noisy_price = pred_price * (1 + noise)

        predictions.append(noisy_price)

        # === Update features for next step ===
        new_row = current_features.iloc[0].copy()
        new_row['Close'] = noisy_price

        # Shift rolling windows forward
        # We'll approximate: use last 20 known Closes + new prediction
        # This is lightweight and works well in practice
        closes_history = list(current_features['Close'].tail(19).values) + [noisy_price]
        new_row['MA_20'] = np.mean(closes_history[-20:])

        returns_history = np.diff(closes_history) / closes_history[:-1]
        if len(returns_history) >= 20:
            new_row['Volatility'] = np.std(returns_history[-20:])
        else:
            new_row['Volatility'] = recent_vol  # fallback

        # Volume: keep last known or NaN (not critical)
        new_row['Volume'] = current_features['Volume'].iloc[-1]

        # Update current_features
        current_features = pd.DataFrame([new_row], columns=FEATURE_COLS)

    return predictions


# ---- Main Loop ----
print("Starting Recursive Linear Regression Forecasting with Stochastic Noise...\n")

for symbol in stocks:
    print(f"--- Processing {symbol} ---")

    try:
        df = load_stock(symbol)
        if df is None or "Close" not in df.columns or len(df) < 100:
            print(f"{symbol}: Insufficient data. Skipping.\n")
            continue

        # Add features
        df_featured = add_features(df)
        df_featured = df_featured.dropna()  # Drop rows where MA_20/Volatility are NaN

        if len(df_featured) < 50:
            print(f"{symbol}: Not enough data after feature engineering. Skipping.\n")
            continue

        # Prepare features and target
        X = df_featured[FEATURE_COLS]
        y = df_featured['Close']

        # Use last ~15% as "test" to validate, but train on full for better future forecast
        train_size = int(len(X) * 0.85)
        X_train, y_train = X[:train_size], y[:train_size]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train Linear Regression
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Estimate recent daily volatility (from actual returns)
        returns = df['Close'].pct_change().dropna()
        recent_daily_vol = returns.tail(60).std()

        # Get last known feature row for starting recursion
        last_feature_row = df_featured[FEATURE_COLS].iloc[-1:]

        # Generate 30-day future prices
        future_prices = recursive_forecast_lr(
            model=model,
            scaler=scaler,
            last_row=last_feature_row,
            recent_vol=recent_daily_vol,
            n_steps=FUTURE_DAYS
        )

        # Generate future business dates
        last_date = df.index[-1]
        future_dates = pd.bdate_range(start=last_date, periods=FUTURE_DAYS + 1)[1:]

        # Save forecast CSV
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Close": future_prices
        })
        forecast_csv_path = os.path.join(predictions_folder, f"{symbol}_recursive_forecast.csv")
        forecast_df.to_csv(forecast_csv_path, index=False)

        # Save model and scaler
        joblib.dump(model, os.path.join(models_folder, f"{symbol}_lr_recursive.pkl"))
        joblib.dump(scaler, os.path.join(models_folder, f"{symbol}_scaler.pkl"))

        # Plot: Last 120 days historical + 30-day forecast
        plt.figure(figsize=(14, 7))
        historical_slice = df["Close"].iloc[-120:]
        plt.plot(historical_slice.index, historical_slice.values, label="Historical Close", color="blue", linewidth=2)
        plt.plot(future_dates, future_prices, "--o", label="Recursive LR Forecast (with noise)", color="green", alpha=0.8)

        plt.title(f"{symbol} — 30-Day Recursive Linear Regression Forecast", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(plots_folder, f"{symbol}_recursive_lr.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"{symbol}: Recursive forecast completed and saved.\n")

    except Exception as e:
        print(f"{symbol}: ERROR → {str(e)}\n")
        continue

print("Recursive Linear Regression forecasting completed for all stocks!")