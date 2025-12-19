

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
warnings.filterwarnings('ignore')


processed_folder = "../data/processed"
metrics_output = "../data/tft_metrics.csv"
plots_folder = "../plots_tft"
os.makedirs(plots_folder, exist_ok=True)

stocks = [
    "TSLA","NVDA","AAPL","MSFT","GOOGL","AMZN","META","JPM","V","UNH","HD","PG","MA","DIS","BAC","XOM","PFE","KO"
]

results = []

class TemporalFusionModel(nn.Module):
    """Simplified Temporal Fusion Transformer implementation"""
    def __init__(self, sequence_length=60, feature_size=5, hidden_size=64, num_heads=4, dropout=0.1):
        super(TemporalFusionModel, self).__init__()
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        
        
        self.encoder_lstm = nn.LSTM(feature_size, hidden_size, batch_first=True, bidirectional=True)
        
        
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads, dropout=dropout, batch_first=True)
        
        
        self.decoder_lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)
        
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        
        batch_size = x.size(0)
        
        
        encoder_output, (hidden, cell) = self.encoder_lstm(x)  
        
        
        attn_output, attn_weights = self.attention(encoder_output, encoder_output, encoder_output)
        
        
        decoder_output, _ = self.decoder_lstm(attn_output)
        
        
        last_output = decoder_output[:, -1, :]  
        
        
        output = self.output_layer(self.dropout(self.layer_norm(last_output)))
        
        return output, attn_weights

def create_sequences(data, time_steps=60, feature_columns=None):
    """Create sequences for training"""
    if feature_columns is None:
        feature_columns = ['Close']
    
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[feature_columns].iloc[i:i+time_steps].values)
        y.append(data['Close'].iloc[i+time_steps])
    return np.array(X), np.array(y)

def prepare_features(df):
    """Prepare additional features for TFT"""
    df = df.copy()
    
    
    df['price_change'] = df['Close'].pct_change()
    df['high_low_ratio'] = df['High'] / df['Low']
    df['price_range'] = df['High'] - df['Low']
    
    if 'Volume' in df.columns:
        df['volume_scaled'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_change'] = df['Volume'].pct_change()
    
   
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_ratio'] = df['sma_5'] / df['sma_20']
    
   
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

for symbol in stocks:
    print(f"\n--- Processing {symbol} ---")

    files = glob.glob(os.path.join(processed_folder, f"{symbol}_processed*.csv"))
    if not files:
        print(f"Processed file for {symbol} not found. Skipping.")
        continue
    file_path = max(files, key=os.path.getctime)

    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=["Close", "High", "Low", "Open"], inplace=True)
        
        
        df = prepare_features(df)
        
        
        feature_columns = ['Close', 'price_change', 'high_low_ratio', 'price_range']
        if 'volume_scaled' in df.columns:
            feature_columns.extend(['volume_scaled', 'volume_change'])
        if 'sma_ratio' in df.columns:
            feature_columns.extend(['sma_ratio'])
        
        
        scalers = {}
        scaled_features = []
        for col in feature_columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_col = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
            scaled_features.append(scaled_col)
            scalers[col] = scaler
        
        
        scaled_df = pd.DataFrame(np.column_stack(scaled_features), columns=feature_columns)
        scaled_df['Close_original'] = df['Close'].values
        
       
        time_steps = 60
        X, y = create_sequences(scaled_df, time_steps, feature_columns)
        
        
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
       
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TemporalFusionModel(
            sequence_length=time_steps, 
            feature_size=len(feature_columns),
            hidden_size=64,
            num_heads=4,
            dropout=0.2
        )
        model.to(device)
        
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        
        print(f"Training TFT for {symbol}...")
        epochs = 100
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs, _ = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs, _ = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(test_loader)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        
        model.eval()
        all_predictions = []
        all_actuals = []
        attention_weights = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                preds, attn = model(batch_x)
                all_predictions.extend(preds.cpu().numpy())
                all_actuals.extend(batch_y.numpy())
                attention_weights.append(attn.cpu().numpy())
        
        
        y_pred = np.array(all_predictions).flatten()
        y_actual = np.array(all_actuals).flatten()
        
        
        close_scaler = scalers['Close']
        y_test_inv = close_scaler.inverse_transform(y_actual.reshape(-1, 1)).flatten()
        y_pred_inv = close_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        
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
        
        
        plt.figure(figsize=(12, 8))
        
        
        plt.subplot(2, 1, 1)
        plt.plot(y_test_inv, label="Actual", alpha=0.7, linewidth=2)
        plt.plot(y_pred_inv, label="Predicted", linestyle='--', alpha=0.7, linewidth=2)
        plt.title(f"{symbol} - TFT Prediction vs Actual")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"{symbol}_tft_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f"{symbol} - TFT Training History")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"{symbol}_tft_training.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plots for {symbol}")
            
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        continue


if results:
    pd.DataFrame(results).to_csv(metrics_output, index=False)
    print(f"\nMetrics saved to {metrics_output}")
    
    
    df_results = pd.DataFrame(results)
    print(f"\nTFT Model Summary:")
    print(f"Average RMSE: {df_results['RMSE'].mean():.2f}")
    print(f"Average R²: {df_results['R2'].mean():.3f}")
    print(f"Average Accuracy: {df_results['Accuracy(%)'].mean():.2f}%")
else:
    print("\nNo results to save. Check data files.")

print("\nTFT training completed!")