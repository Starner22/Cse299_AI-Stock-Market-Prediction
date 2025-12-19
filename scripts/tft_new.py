import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import math
warnings.filterwarnings('ignore')
import shutil

# --- Optimization Constants ---
PRETRAIN_EPOCHS = 30
PRETRAIN_BATCH_SIZE = 128
NUM_WORKERS = 4 # Use 0 if the training speed is still an issue on CPU
# ------------------------------

# ------------------ Folders ------------------
processed_folder = "../data/processed"
pretrained_model_folder = "../saved_models/tft_pretrained"
fine_tuned_model_folder = "../saved_models/tft_finetuned"
plots_folder = "../plots_tft"
metrics_output = "../data/tft_multistock_metrics.csv"

os.makedirs(pretrained_model_folder, exist_ok=True)
os.makedirs(fine_tuned_model_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)

# ------------------ Stocks ------------------
stocks = ["TSLA"]

# ------------------ TFT Model ------------------
class TemporalFusionModel(nn.Module):
    # Note: Feature size is the number of input features (not including the target)
    def __init__(self, sequence_length=90, feature_size=8, hidden_size=128, num_heads=8, dropout=0.1, num_layers=2):
        super().__init__()
        self.encoder_lstm = nn.LSTM(feature_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size*2, num_heads, dropout=dropout, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_size*2, hidden_size, num_layers=num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        enc_out,_ = self.encoder_lstm(x)
        attn_out, _ = self.attention(enc_out, enc_out, enc_out)
        dec_out,_ = self.decoder_lstm(attn_out)
        last_out = dec_out[:, -1, :]
        out = self.output_layer(self.dropout(self.layer_norm(last_out)))
        return out

# ------------------ Feature Engineering ------------------
def prepare_features(df):
    df = df.copy()
    # Log Returns: The new target variable and a highly useful feature
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df['price_change'] = df['Close'].pct_change()
    df['high_low_ratio'] = df['High'] / df['Low']
    df['price_range'] = df['High'] - df['Low']
    
    if 'Volume' in df.columns:
        df['volume_scaled'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_change'] = df['Volume'].pct_change()
    
    # Moving averages
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_ratio'] = df['sma_5'] / df['sma_20']
    
    # EMA
    df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ema_ratio'] = df['ema_10'] / df['ema_20']
    
    # Bollinger Bands
    df['bb_upper'] = df['sma_20'] + 2*df['Close'].rolling(20).std()
    df['bb_lower'] = df['sma_20'] - 2*df['Close'].rolling(20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1*delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # NOTE: The first row (index 0) will have NaNs due to shift(1) and diff()
    # We drop the first row to ensure clean data for sequencing.
    df = df.dropna().reset_index(drop=True) 
    return df

def create_sequences(df, feature_columns, time_steps=90):
    X, y = [], []
    # Store the actual close prices for inverse transformation
    y_actual_prices = []
    
    # We now predict the log_return (y) based on the features up to time t-1 (X)
    for i in range(len(df) - time_steps):
        # X: Features from time i to i+time_steps - 1
        X.append(df[feature_columns].iloc[i:i+time_steps].values)
        
        # y: Log Return at time i+time_steps (the prediction target)
        y.append(df['log_return'].iloc[i+time_steps])
        
        # We need the previous day's close price to reconstruct the actual price
        y_actual_prices.append(df['Close'].iloc[i+time_steps])

    return np.array(X), np.array(y), np.array(y_actual_prices)

def compute_metrics(y_true, y_pred):
    # This function now uses the actual (unscaled, reconstructed) prices
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

# ------------------ Main Execution Block (Fix for DataLoader on Windows) ------------------
if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    results = []

    # ------------------ Main Loop ------------------
    for symbol in stocks:
        print(f"\n=== Processing {symbol} ===")

        files = glob.glob(os.path.join(processed_folder, f"{symbol}_processed*.csv"))
        if not files:
            print(f"No data for {symbol}, skipping.")
            continue
        file_path = max(files, key=os.path.getctime)
        df = pd.read_csv(file_path)
        df.dropna(subset=['Close','High','Low','Open'], inplace=True)
        
        # The target variable is now 'log_return'
        df = prepare_features(df)

        feature_columns = ['log_return','price_change','high_low_ratio','price_range',
                           'sma_ratio','ema_ratio','bb_upper','rsi']
        if 'Volume' in df.columns: feature_columns += ['volume_scaled','volume_change']

        # Scaling: Only scale the features, NOT the target (log_return)
        scalers = {}
        for col in feature_columns:
            scaler = MinMaxScaler()
            # Fit and transform all features, including the log_return feature
            df[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler
        
        # Sequences
        time_steps = 90
        # X: Features (scaled); y: Log Return (unscaled); y_actual_prices: Close Price (unscaled)
        X, y, y_actual_prices = create_sequences(df, feature_columns, time_steps)

        # We must align the closing price for reconstruction correctly. 
        # y_actual_prices stores the actual price at time t. We need price at time t-1
        # for reconstruction: P_t = P_{t-1} * exp(y_pred_return).
        
        # Get the closing price from the day *before* the prediction
        P_prev_all = df['Close'].iloc[time_steps-1:-1].values

        split_idx = int(0.8 * len(X))
        X_pretrain, y_pretrain = X[:split_idx], y[:split_idx]
        X_recent, y_recent = X[split_idx:], y[split_idx:]
        
        P_prev_recent = P_prev_all[split_idx:]
        y_actual_recent = y_actual_prices[split_idx:] # Actual price at time t

        # Train/val/test split on recent
        val_size = int(0.15 * len(X_recent))
        test_size = int(0.15 * len(X_recent))
        train_size = len(X_recent) - val_size - test_size

        X_train = torch.FloatTensor(X_recent[:train_size])
        y_train = torch.FloatTensor(y_recent[:train_size])
        
        X_val = torch.FloatTensor(X_recent[train_size:train_size+val_size])
        y_val = torch.FloatTensor(y_recent[train_size:train_size+val_size])
        
        X_test = torch.FloatTensor(X_recent[train_size+val_size:])
        y_test_actual = y_actual_recent[train_size+val_size:] # Actual prices P_t for test set
        P_prev_test = P_prev_recent[train_size+val_size:] # Previous day prices P_{t-1} for test set

        # DataLoaders for Fine-tuning and Testing
        loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
        
        train_loader = DataLoader(TensorDataset(X_train,y_train), batch_size=32, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(TensorDataset(X_val,y_val), batch_size=32, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(TensorDataset(X_test,torch.FloatTensor(y_test_actual)), batch_size=32, shuffle=False, **loader_kwargs) # y here is placeholder

        # Model
        # Feature size is len(feature_columns)
        model = TemporalFusionModel(sequence_length=time_steps, feature_size=len(feature_columns)).to(device)
        pretrained_path = os.path.join(pretrained_model_folder,f"{symbol}_tft_return_model.pth")

        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
            print("Loaded log-return pretrained model")
        else:
            print(f"Pretraining log-return model for {PRETRAIN_EPOCHS} epochs...")
            
            # Optimized Pretrain DataLoader
            pretrain_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_pretrain), torch.FloatTensor(y_pretrain)), 
                batch_size=PRETRAIN_BATCH_SIZE, # Optimized batch size
                shuffle=True, 
                **loader_kwargs
            )
            
            # We still use MSE loss on the scaled log_returns
            criterion = nn.MSELoss() 
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(PRETRAIN_EPOCHS):
                model.train()
                train_loss = 0
                for bx, by in pretrain_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out.squeeze(), by)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                print(f"Pretrain Epoch {epoch+1}/{PRETRAIN_EPOCHS}, Loss: {train_loss/len(pretrain_loader):.6f}")

            torch.save(model.state_dict(), pretrained_path)
            print("Pretrained log-return model saved")

        # --- EVALUATION BEFORE FINE-TUNING ---
        model.eval()
        y_pred_log_return_list = []
        with torch.no_grad():
            for bx, by_dummy in test_loader:
                bx = bx.to(device)
                preds = model(bx)
                y_pred_log_return_list.extend(preds.cpu().numpy())
                
        y_pred_log_return = np.array(y_pred_log_return_list).flatten()
        
        # Reconstruction: P_t = P_{t-1} * exp(y_pred_log_return)
        y_pred_inv = P_prev_test * np.exp(y_pred_log_return)
        y_test_inv = y_test_actual # True prices P_t
        
        mse_before, rmse_before, mae_before, r2_before = compute_metrics(y_test_inv, y_pred_inv)
        print(f"Before FT → MSE: {mse_before:.3f}, RMSE: {rmse_before:.3f}, MAE: {mae_before:.3f}, R2: {r2_before:.3f}")

        # Fine-tuning
        # FREEZE ENCODER: Prevent core feature extractor weights from being ruined by small dataset
        for p in model.encoder_lstm.parameters(): p.requires_grad = False 
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        epochs = 30
        patience = 5
        best_val_loss = float('inf')
        epochs_no_improve = 0

        print("Starting Fine-tuning (Encoder Frozen)...")
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out.squeeze(), by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    val_loss += criterion(out.squeeze(), by).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.6f}, Val Loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # NOTE: Removed the 'unfreeze encoder' block to prevent catastrophic overfitting.

        model.load_state_dict(best_model_state)

        # --- EVALUATION AFTER FINE-TUNING ---
        model.eval()
        y_pred_log_return_list_after = []
        with torch.no_grad():
            for bx, by_dummy in test_loader:
                bx = bx.to(device)
                preds = model(bx)
                y_pred_log_return_list_after.extend(preds.cpu().numpy())
                
        y_pred_log_return_after = np.array(y_pred_log_return_list_after).flatten()
        
        # Reconstruction: P_t = P_{t-1} * exp(y_pred_log_return_after)
        y_pred_inv_after = P_prev_test * np.exp(y_pred_log_return_after)
        
        mse_after, rmse_after, mae_after, r2_after = compute_metrics(y_test_inv, y_pred_inv_after)
        print(f"After FT → MSE: {mse_after:.3f}, RMSE: {rmse_after:.3f}, MAE: {mae_after:.3f}, R2: {r2_after:.3f}")

        # Save fine-tuned model
        ft_path = os.path.join(fine_tuned_model_folder, f"{symbol}_tft_finetuned_return_{pd.Timestamp.today().strftime('%Y%m%d')}.pth")
        torch.save(model.state_dict(), ft_path)

        # Plot
        plt.figure(figsize=(12,6))
        plt.plot(y_test_inv, label='Actual')
        plt.plot(y_pred_inv, label='Before FT', linestyle='--')
        plt.plot(y_pred_inv_after, label='After FT', linestyle='--')
        plt.title(f"{symbol} - TFT Predictions (Log Return Based)")
        plt.legend()
        plt.savefig(os.path.join(plots_folder,f"{symbol}_tft_return_comparison.png"))
        plt.close()

        # Save metrics
        results.append({
            "Symbol": symbol,
            "Target": "Log Return",
            "MSE_before": mse_before, "RMSE_before": rmse_before, "MAE_before": mae_before, "R2_before": r2_before,
            "MSE_after": mse_after, "RMSE_after": rmse_after, "MAE_after": mae_after, "R2_after": r2_after
        })

    # Save CSV
    pd.DataFrame(results).to_csv(metrics_output,index=False)
    print("\nAll metrics saved.")