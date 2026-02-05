"""
Advanced Time Series Forecasting with LSTM + Attention
Author: <Your Name>

This script fulfills all Tasks to Complete and Expected Deliverables:
- Data acquisition & preprocessing
- Feature engineering
- Baseline LSTM
- Seq2Seq LSTM with Attention
- Training & evaluation
- Visualization
"""

import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Configuration
# =========================
SYMBOL = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"

SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
HIDDEN_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Data Utilities
# =========================
def load_data() -> pd.DataFrame:
    df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)
    df = df[["Close", "Volume"]]
    df.dropna(inplace=True)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["rolling_mean_5"] = df["Close"].rolling(5).mean()
    df["rolling_std_5"] = df["Close"].rolling(5).std()
    df["rolling_mean_10"] = df["Close"].rolling(10).mean()
    df.dropna(inplace=True)
    return df

def train_val_test_split(
    data: np.ndarray, train=0.7, val=0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(data)
    train_end = int(n * train)
    val_end = int(n * (train + val))
    return data[:train_end], data[train_end:val_end], data[val_end:]

# =========================
# Dataset
# =========================
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx : idx + self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx + self.seq_len], dtype=torch.float32),
        )

# =========================
# Models
# =========================
class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(-1)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs, hidden):
        hidden = hidden.unsqueeze(1)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden)))
        weights = torch.softmax(score, dim=1)
        context = torch.sum(weights * encoder_outputs, dim=1)
        return context, weights

class Seq2SeqAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = BahdanauAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        encoder_outputs, (h_n, _) = self.encoder(x)
        context, _ = self.attention(encoder_outputs, h_n[-1])
        combined = torch.cat((context, h_n[-1]), dim=1)
        return self.fc(combined).squeeze(-1)

# =========================
# Training & Evaluation
# =========================
def train_model(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            preds.append(model(X).cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)

def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))

# =========================
# Main Pipeline
# =========================
def main():
    df = engineer_features(load_data())

    features = df.drop(columns=["Close"]).values
    target = df["Close"].values

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    features = scaler_x.fit_transform(features)
    target = scaler_y.fit_transform(target.reshape(-1, 1)).ravel()

    X_train, X_val, X_test = train_val_test_split(features)
    y_train, y_val, y_test = train_val_test_split(target)

    train_ds = TimeSeriesDataset(X_train, y_train, SEQ_LEN)
    test_ds = TimeSeriesDataset(X_test, y_test, SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[1]

    baseline = LSTMBaseline(input_dim, HIDDEN_SIZE).to(DEVICE)
    attention_model = Seq2SeqAttention(input_dim, HIDDEN_SIZE).to(DEVICE)

    criterion = nn.MSELoss()

    for name, model in {
        "Baseline LSTM": baseline,
        "Attention Model": attention_model,
    }.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        for epoch in range(EPOCHS):
            loss = train_model(model, train_loader, optimizer, criterion)
        preds, trues = evaluate(model, test_loader)

        preds_inv = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
        trues_inv = scaler_y.inverse_transform(trues.reshape(-1, 1)).ravel()

        rmse = math.sqrt(mean_squared_error(trues_inv, preds_inv))
        mae = mean_absolute_error(trues_inv, preds_inv)
        da = directional_accuracy(trues_inv, preds_inv)

        print(f"\n{name}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"Directional Accuracy: {da:.2%}")

        plt.figure(figsize=(10, 4))
        plt.plot(trues_inv[:200], label="Actual")
        plt.plot(preds_inv[:200], label="Predicted")
        plt.title(name)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
