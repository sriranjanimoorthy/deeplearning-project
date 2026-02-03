"""
Advanced Time Series Forecasting with Deep Learning and Attention
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import math
import random

# =============================
# 1. REPRODUCIBILITY
# =============================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# 2. SYNTHETIC DATA GENERATION
# =============================
def generate_multivariate_series(n_steps=1500):
    t = np.arange(n_steps)

    f1 = np.sin(2 * np.pi * t / 24)                     # daily seasonality
    f2 = np.sin(2 * np.pi * t / (24 * 7))                # weekly seasonality
    f3 = 0.01 * t                                        # trend
    f4 = np.random.normal(0, 0.5, n_steps)               # noise
    f5 = np.cos(2 * np.pi * t / 12) * 0.5                # short seasonal

    target = (
        0.4 * f1 +
        0.3 * f2 +
        0.2 * f3 +
        0.1 * f5 +
        np.random.normal(0, 0.2, n_steps)
    )

    data = np.vstack([f1, f2, f3, f4, f5, target]).T
    columns = ["f1", "f2", "f3", "f4", "f5", "target"]
    return pd.DataFrame(data, columns=columns)

df = generate_multivariate_series()

# =============================
# 3. DATA PREPROCESSING
# =============================
FEATURES = ["f1", "f2", "f3", "f4", "f5"]
TARGET = "target"

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_x.fit_transform(df[FEATURES])
y_scaled = scaler_y.fit_transform(df[[TARGET]])

SEQ_LEN = 48
FORECAST_HORIZON = 1

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)

split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# =============================
# 4. DATASET
# =============================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=64)

# =============================
# 5. BASELINE LSTM
# =============================
class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# =============================
# 6. ATTENTION MECHANISM
# =============================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):
        scores = self.attn(lstm_outputs)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_outputs, dim=1)
        return context, weights

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, attn_weights

# =============================
# 7. TRAINING FUNCTION
# =============================
def train_model(model, loader, epochs=20):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(Xb)
            if isinstance(preds, tuple):
                preds = preds[0]
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# =============================
# 8. EVALUATION
# =============================
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            out = model(Xb)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.cpu().numpy())
            trues.append(yb.numpy())

    preds = scaler_y.inverse_transform(np.vstack(preds))
    trues = scaler_y.inverse_transform(np.vstack(trues))

    rmse = math.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    mape = np.mean(np.abs((trues - preds) / trues)) * 100

    return rmse, mae, mape

# =============================
# 9. TRAIN BASELINE
# =============================
baseline = BaselineLSTM(len(FEATURES), 64).to(DEVICE)
train_model(baseline, train_loader)
baseline_metrics = evaluate(baseline, test_loader)

# =============================
# 10. TRAIN ATTENTION MODEL
# =============================
attn_model = LSTMAttentionModel(len(FEATURES), 64).to(DEVICE)
train_model(attn_model, train_loader)
attn_metrics = evaluate(attn_model, test_loader)

print("\n--- Performance Comparison ---")
print("Baseline LSTM  | RMSE:", baseline_metrics[0], "MAE:", baseline_metrics[1], "MAPE:", baseline_metrics[2])
print("LSTM+Attention| RMSE:", attn_metrics[0], "MAE:", attn_metrics[1], "MAPE:", attn_metrics[2])

# =============================
# 11. ATTENTION VISUALIZATION
# =============================
def visualize_attention(model, dataset, samples=3):
    model.eval()
    for i in range(samples):
        x, _ = dataset[i]
        x = x.unsqueeze(0).to(DEVICE)
        _, attn = model(x)
        attn = attn.squeeze().cpu().numpy()

        plt.figure(figsize=(8,3))
        plt.plot(attn)
        plt.title(f"Attention Weights – Test Sample {i+1}")
        plt.xlabel("Time Steps")
        plt.ylabel("Weight")
        plt.show()

visualize_attention(attn_model, TimeSeriesDataset(X_test, y_test))
