# =========================
# 1. IMPORTS & SETUP
# =========================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("results/attention_plots", exist_ok=True)

# =========================
# 2. DATASET GENERATION
# =========================
def generate_dataset(n_steps=1500):
    t = np.arange(n_steps)
    data = np.stack([
        np.sin(2*np.pi*t/50) + 0.01*t,
        np.cos(2*np.pi*t/30),
        np.sin(2*np.pi*t/100),
        0.005*t + np.random.normal(0, 0.2, n_steps),
        np.random.normal(0, 0.3, n_steps)
    ], axis=1)

    return pd.DataFrame(data, columns=[f"feat_{i}" for i in range(5)])

# =========================
# 3. SEQUENCE CREATION
# =========================
def create_sequences(data, input_len=30, output_len=1):
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len, 0])
    return np.array(X), np.array(y)

# =========================
# 4. DATASET CLASS
# =========================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =========================
# 5. MODELS
# =========================
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        score = torch.tanh(self.attn(x))
        weights = torch.softmax(self.context(score), dim=1)
        context = (weights * x).sum(dim=1)
        return context, weights

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        context, attn_weights = self.attention(enc_out)
        output = self.fc(context)
        return output, attn_weights

class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =========================
# 6. TRAINING FUNCTION
# =========================
def train_model(model, loader, epochs=20):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            outputs = model(X_batch)
            preds = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = criterion(preds.squeeze(), y_batch.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

# =========================
# 7. EVALUATION
# =========================
def evaluate_model(model, loader):
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            y_pred = outputs[0] if isinstance(outputs, tuple) else outputs

            preds.extend(y_pred.cpu().numpy())
            actuals.extend(y_batch.numpy())

    preds = np.array(preds).flatten()
    actuals = np.array(actuals).flatten()

    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    epsilon = 1e-8
    mape = np.mean(np.abs((actuals - preds) / (actuals + epsilon))) * 100

    return mae, rmse, mape

# =========================
# 8. DISPLAY METRICS (COLAB)
# =========================
def display_metrics(base_metrics, attn_metrics):
    print("\n===== MODEL PERFORMANCE =====\n")
    print("Baseline LSTM")
    print(f"MAE  : {base_metrics[0]:.4f}")
    print(f"RMSE : {base_metrics[1]:.4f}")
    print(f"MAPE : {base_metrics[2]:.2f}%\n")

    print("Attention-based LSTM")
    print(f"MAE  : {attn_metrics[0]:.4f}")
    print(f"RMSE : {attn_metrics[1]:.4f}")
    print(f"MAPE : {attn_metrics[2]:.2f}%")

def show_metrics_table(base_metrics, attn_metrics):
    df = pd.DataFrame({
        "Model": ["Baseline LSTM", "Attention LSTM"],
        "MAE": [base_metrics[0], attn_metrics[0]],
        "RMSE": [base_metrics[1], attn_metrics[1]],
        "MAPE (%)": [base_metrics[2], attn_metrics[2]]
    })
    display(df)

# =========================
# 9. ATTENTION VISUALIZATION
# =========================
def show_and_save_attention_plots(model, loader):
    model.eval()
    X_sample, _ = next(iter(loader))
    X_sample = X_sample.to(device)

    _, attn_weights = model(X_sample)

    for i in range(3):
        plt.figure(figsize=(8, 3))
        plt.imshow(attn_weights[i].cpu().numpy().T, aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.title(f"Attention Weights - Sequence {i+1}")
        plt.xlabel("Time Steps")
        plt.ylabel("Attention")

        plt.show()
        plt.savefig(f"results/attention_plots/attention_seq_{i+1}.png")
        plt.close()

# =========================
# 10. MAIN EXECUTION
# =========================
def main():
    df = generate_dataset()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)

    X, y = create_sequences(scaled_data)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=32)

    baseline = BaselineLSTM(5, 64)
    attention = LSTMAttentionModel(5, 64)

    print("\nTraining Baseline LSTM")
    train_model(baseline, train_loader)

    print("\nTraining Attention LSTM")
    train_model(attention, train_loader)

    base_metrics = evaluate_model(baseline, test_loader)
    attn_metrics = evaluate_model(attention, test_loader)

    display_metrics(base_metrics, attn_metrics)
    show_metrics_table(base_metrics, attn_metrics)

    with open("results/metrics.txt", "w") as f:
        f.write("Baseline LSTM:\n")
        f.write(f"MAE: {base_metrics[0]:.4f}\nRMSE: {base_metrics[1]:.4f}\nMAPE: {base_metrics[2]:.2f}%\n\n")
        f.write("Attention LSTM:\n")
        f.write(f"MAE: {attn_metrics[0]:.4f}\nRMSE: {attn_metrics[1]:.4f}\nMAPE: {attn_metrics[2]:.2f}%\n")

    show_and_save_attention_plots(attention, test_loader)

    print("\nExecution completed successfully.")
    print("Metrics saved to results/metrics.txt")

main()
