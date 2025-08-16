# Batch processing of the eeg data; Dataloader creates the batch

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
import mne
import pathlib
from torchviz import make_dot
from load_edf import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load and Preprocess Real EEG Signal ===


def bandpass_filter(x, fs, low, high):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x)

# === Laplace + RNN Model Definition ===
class LaplaceRNNForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, laplace_N=100):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.s_grid = torch.linspace(1, 10, laplace_N).unsqueeze(0).to(device)  # shape: (1, N)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + laplace_N, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1)
        )

    def forward(self, t, x):
        # x: (batch, seq_len, 1), t: (batch, seq_len)
        B = torch.exp(-t.unsqueeze(-1) * self.s_grid)  # shape: (batch, seq_len, N)
        rnn_out, _ = self.rnn(x)  # shape: (batch, seq_len, hidden)
        out = torch.cat([rnn_out, B], dim=-1)  # shape: (batch, seq_len, hidden + N)
        #print('data:',x.shape,'laplace_fun:', B.shape,'rnn_out:',rnn_out.shape,'rnn_out+lap',out.shape)
        
        return self.fc(out)  # shape: (batch, seq_len, 1)

# === Training Function ===
def train_laplace_rnn(model, dataloader, window_size=128, num_epochs=100, model_path='rnn_laplace_model_bath.pth'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for t_batch, x_batch in dataloader:
            t_batch = t_batch.to(device)  # (batch, seq_len)
            x_batch = x_batch.to(device).unsqueeze(-1)  # (batch, seq_len, 1)

            for i in range(0, x_batch.shape[1] - window_size):
                t_win = t_batch[:, i:i+window_size]  # (batch, window_size)
                x_win = x_batch[:, i:i+window_size, :]  # (batch, window_size, 1)
                y_win = x_batch[:, i+1:i+window_size+1, :]  # next step (target)

                optimizer.zero_grad()
                y_pred = model(t_win, x_win)  # predict
                loss = criterion(y_pred, y_win)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(),model_path)
            print('model saved epoch: ',epoch)
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= 5:
            print("Early stopping triggered.")
            break

# === Forecast Function ===
def forecast_laplace_rnn(model, x_input, t_input, forecast_len, fs=128):
    model=model.to(device)
    preds = []
    #print('true_input:',x_input.shape)
    x_input = x_input.to(device).unsqueeze(0).unsqueeze(-1)  # (1, len, 1)
    t_input = t_input.to(device).unsqueeze(0)  # (1, len)

    with torch.no_grad():
        for i in range(forecast_len):
            y_pred = model(t_input, x_input)  # (1, len, 1)
            next_val = y_pred[:, -1:, :]  # last prediction
            #print(y_pred.shape,next_val.shape);
            next_t = t_input[:, -1:] + 1.0/fs

            x_input = torch.cat([x_input[:, 1:], next_val], dim=1)
            t_input = torch.cat([t_input[:, 1:], next_t], dim=1)

            preds.append(next_val.squeeze().cpu().numpy())

    return np.array(preds)

# === Main ===
fs = 128
input_duration = 4
forecast_duration = 1
model_path="rnn_laplace_model_batch.pth"
# Replace with your EEG file
eeg_file = '../eeg_physionet/S001R14.edf'

# Load full EEG for reference
dataset = EEGDataset('../eeg_physionet', duration_sec=4)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

_, full_signal = load_eeg(eeg_file, channel=1, fs=fs, duration_sec=-1)

# Load partial signal for training and forecasting
t_input, x_input = load_eeg(eeg_file, channel=1, fs=fs, duration_sec=input_duration)

model = LaplaceRNNForecaster()
train=0
if train:
    train_laplace_rnn(model, dataloader, window_size=fs,model_path=model_path)
else:
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()

forecast_steps = int(forecast_duration * fs)
predicted = forecast_laplace_rnn(model, x_input, t_input, forecast_steps, fs=fs)


# === Plot Results ===
t_input_np = t_input.numpy()
t_forecast = np.linspace(t_input_np[-1], t_input_np[-1] + forecast_duration, forecast_steps)

t_combined = np.concatenate((t_input_np, t_forecast))
x_combined = np.concatenate((x_input.numpy(), predicted))
x_gt=full_signal[:len(t_combined)].cpu().numpy()

plt.figure(figsize=(12, 5))
#plt.plot(t_input_np, x_input.numpy(), label='Input EEG')
#plt.plot(t_forecast, predicted, label='Forecast EEG')
plt.plot(t_combined, x_combined,label='Input + forecasted')
plt.plot(t_combined, x_gt,label='ground truth')
plt.title("EEG Forecast using Laplace RNN")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rnn_laplace_batch.eps", format='eps')
plt.show()
def plot_psd_comparison(E, E_synth, fs=128):
    plt.figure(figsize=(10, 5))
    f, Pxx = welch(E, fs, nperseg=fs*2)  #compute mean of all channels and plot psd
    f_synth, Pxx_synth = welch(E_synth, fs, nperseg=fs*2)
    plt.semilogy(f, Pxx, label='Original EEG')
    plt.semilogy(f_synth, Pxx_synth, label='Synthesized EEG')
    plt.title('Power Spectral Density Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V^2/Hz)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rnn_laplace_psd_batch.eps", format='eps')
    plt.show()

plot_psd_comparison(x_gt, x_combined)
