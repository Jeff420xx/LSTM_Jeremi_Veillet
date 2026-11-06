import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import os

# ==========================
# 1. Configuration Dynamique
# ==========================
TARGET_TICKER = "ARM"  
def get_data(ticker, start="2020-01-01", end="2026-06-05"):
    print(f" Téléchargement de {ticker} ...")
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[["Close"]].dropna()
    return data

data = get_data(TARGET_TICKER)

# Log-returns pour la stabilité
data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
data = data.dropna()

log_returns = data[['Log_Ret']].values
close_prices = data[["Close"]].values

# ==========================
# 2. Fenêtre glissante & Séquences
# ==========================
window_size = 2000   
seq_length = 180      

if len(log_returns) > window_size:
    log_returns = log_returns[-window_size:]

test_horizon = 21 
split_idx = len(log_returns) - test_horizon

scaler = StandardScaler()
scaler.fit(log_returns[:split_idx])         
scaled_data = scaler.transform(log_returns)

def create_sequences(dataset, seq_length):
    X, y = [], []
    for i in range(len(dataset) - seq_length):
        X.append(dataset[i:i+seq_length])
        y.append(dataset[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, seq_length)

train_size = len(X) - test_horizon
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test   = torch.tensor(X_test, dtype=torch.float32),  torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# ==========================
# 3. Modèle & Early Stopping
# ==========================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, train_loss):
        if train_loss < self.best_loss - self.min_delta:
            self.best_loss = train_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

criterion = nn.L1Loss()


# ==========================
# 4. Entraînement
# ==========================
epochs = 20     # Augmenté car l'Early Stopping gérera l'arrêt optimal
n_runs = 5      
results_rmse = []

checkpoint_path = f"models/last_model_{TARGET_TICKER}.pth"
os.makedirs("models", exist_ok=True)

model = LSTMModel()

for run in range(n_runs):
    print(f"\n=== Run {run+1}/{n_runs} ===")
    torch.manual_seed(run)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    early_stopper = EarlyStopping(patience=3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad() 
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
        
        early_stopper(avg_loss)
        if early_stopper.early_stop:
            print(" Early Stopping activé.")
            break

    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()
        preds_inv = scaler.inverse_transform(preds)
        y_test_inv = scaler.inverse_transform(y_test.numpy())
        rmse_val = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
        results_rmse.append(rmse_val) 
        print(f"Run {run+1} terminé — RMSE : {rmse_val:.4f}")

    torch.save(model.state_dict(), checkpoint_path)

# ======================================================
# 5. Prédiction Récursive du Futur
# ======================================================
n_future = 10 
model.eval()

last_seq = torch.tensor(scaled_data[-seq_length:], dtype=torch.float32).view(1, seq_length, 1)
future_log_rets = []
current_seq = last_seq.clone()

with torch.no_grad():
    for _ in range(n_future):
        pred = model(current_seq)
        future_log_rets.append(pred.item())
        new_entry = pred.view(1, 1, 1)
        current_seq = torch.cat((current_seq[:, 1:, :], new_entry), dim=1)

future_log_rets = scaler.inverse_transform(np.array(future_log_rets).reshape(-1, 1))

last_price = data['Close'].values[-1]
future_prices = []
for lr in future_log_rets:
    next_price = last_price * np.exp(lr[0])
    future_prices.append(next_price)
    last_price = next_price

future_dates = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_future)

# ======================================================
# 6. Visualisation Finale
# ======================================================
plt.figure(figsize=(16, 7))

# --- Graph 1 : RMSE ---
plt.subplot(1, 2, 1)
plt.plot(range(1, len(results_rmse) + 1), results_rmse, marker='o', color='green')
plt.title(f"Convergence de l'erreur ({TARGET_TICKER})")
plt.grid(True, alpha=0.3)

# --- Graph 2 : Passé + Futur ---
plt.subplot(1, 2, 2)
with torch.no_grad():
    test_preds_scaled = model(X_test).numpy()
    test_preds_lr = scaler.inverse_transform(test_preds_scaled)
    prices_prev = data['Close'].values[-(test_horizon + 1) : -1]
    recon_prices = [prices_prev[i] * np.exp(test_preds_lr[i][0]) for i in range(test_horizon)]

test_dates = data.index[-test_horizon:]
plt.plot(data.index[-60:], data['Close'].values[-60:], label="Réel", color='blue', alpha=0.5)
plt.plot(test_dates, recon_prices, label="Reconstruction", color='red', linestyle='--')
plt.plot(future_dates, future_prices, label="PROJECTION (10j)", color='orange', marker='x', linewidth=2)
plt.axvline(x=data.index[-1], color='black', linestyle='-', alpha=0.2)

plt.title(f"Prédiction {TARGET_TICKER} : Reconstruction vs Futur")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()