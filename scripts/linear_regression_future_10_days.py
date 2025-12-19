import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta, datetime

os.makedirs("../data/future_predictions", exist_ok=True)
os.makedirs("../plots/future_predictions", exist_ok=True)

stocks = ["AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","V","UNH"]

processed_dir = "../data/processed"
future_dir = "../data/future_predictions"

processed_stocks = []
skipped_stocks = []

for symbol in stocks:
    print(f"\n--- Processing {symbol} ---")

    matching_files = [f for f in os.listdir(processed_dir) if f.startswith(f"{symbol}_processed")]
    if not matching_files:
        print(f"Processed file for {symbol} not found. Skipping.")
        skipped_stocks.append(symbol)
        continue

    latest_file = max(matching_files, key=lambda f: os.path.getmtime(os.path.join(processed_dir, f)))
    file_path = os.path.join(processed_dir, latest_file)

    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    data.dropna(inplace=True)

    required_cols = ["Close", "MA_20", "Volatility", "Volume"]
    if not all(col in data.columns for col in required_cols):
        skipped_stocks.append(symbol)
        continue

    # ---------------- TRAIN MODEL ----------------
    X = data[["Close", "MA_20", "Volatility", "Volume"]].shift(1).dropna()
    y = data["Close"].iloc[1:]
    y = y.loc[X.index]

    model = LinearRegression()
    model.fit(X, y)

    last_data = data.copy()
    last_date = data.index[-1].date()
    today = datetime.now().date()

    predictions = []

    # -------- BRIDGE: LAST DATASET DATE → TODAY --------
    while last_date < today:
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        features = last_data[["Close", "MA_20", "Volatility", "Volume"]].iloc[-1].values.reshape(1, -1)
        predicted_close = model.predict(features)[0]

        predictions.append({"Date": next_date, "Predicted_Close": predicted_close})

        new_row = last_data.iloc[-1].copy()
        new_row["Close"] = predicted_close
        last_data = pd.concat([last_data, pd.DataFrame([new_row], index=[next_date])])

        last_data["MA_20"] = last_data["Close"].rolling(20).mean()
        last_data["Volatility"] = last_data["Close"].pct_change().rolling(20).std()

        last_date = next_date

    # -------- FUTURE: NEXT 10 DAYS --------
    for _ in range(10):
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        features = last_data[["Close", "MA_20", "Volatility", "Volume"]].iloc[-1].values.reshape(1, -1)
        predicted_close = model.predict(features)[0]

        predictions.append({"Date": next_date, "Predicted_Close": predicted_close})

        new_row = last_data.iloc[-1].copy()
        new_row["Close"] = predicted_close
        last_data = pd.concat([last_data, pd.DataFrame([new_row], index=[next_date])])

        last_data["MA_20"] = last_data["Close"].rolling(20).mean()
        last_data["Volatility"] = last_data["Close"].pct_change().rolling(20).std()

        last_date = next_date

    # ---------------- SAVE + PLOT ----------------
    pred_df = pd.DataFrame(predictions).set_index("Date")
    pred_df.to_csv(os.path.join(future_dir, f"{symbol}_predictions_from_lastdate.csv"))

    plt.figure(figsize=(12,6))
    plt.plot(data.index[-100:], data["Close"].iloc[-100:], label="Actual Close", color="blue")
    plt.plot(pred_df.index, pred_df["Predicted_Close"],
             label="Predicted (From Last Dataset → Today → +10 Days)",
             color="red", linestyle="--", marker="o")

    plt.title(f"{symbol} - Continuous Prediction from Last Dataset Date")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../plots/future_predictions/{symbol}_continuous_prediction.png")
    plt.close()

    processed_stocks.append(symbol)

print("\n✅ Prediction graph now starts from LAST DATASET DATE till TODAY (+10 days)")
print("Processed stocks:", processed_stocks)
print("Skipped stocks:", skipped_stocks)
