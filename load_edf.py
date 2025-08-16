import os
import mne
import numpy as np
import torch
from scipy.signal import butter, filtfilt
import pathlib
from torch.utils.data import Dataset, DataLoader

def bandpass_filter(x, fs, low, high):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x)

def load_eeg(filepath='eeg.csv', channel=0, fs=128, duration_sec=0, bandpass=(4, 40)):
    basename = pathlib.Path(filepath).suffix
    if basename == '.edf':
        raw = mne.io.read_raw_edf(filepath, preload=True)
        raw.pick(['Fcz.'])
        raw.resample(fs)
        eeg_data = raw.get_data()[0]
        x = eeg_data[:fs * duration_sec]
    else:
        eeg_data = np.loadtxt(filepath, delimiter=',')
        x = eeg_data[:fs * duration_sec, channel]
    x = (x - np.mean(x)) / np.std(x)
    if bandpass:
        x = bandpass_filter(x, fs, *bandpass)
    x=x.copy()
    t = np.linspace(0, duration_sec, len(x))
    return torch.tensor(t, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)
    

class EEGDataset(Dataset):
    def __init__(self, edf_folder, fs=128, duration_sec=4, bandpass=(4, 40), channel='Fcz.'):
        self.fs = fs
        self.duration_sec = duration_sec
        self.bandpass = bandpass
        self.channel = channel
        self.file_paths = [os.path.join(edf_folder, f) for f in os.listdir(edf_folder) if f.endswith('.edf')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raw.pick([self.channel])
        raw.resample(self.fs)
        eeg_data = raw.get_data()[0]
        x = eeg_data[:self.fs * self.duration_sec]
        x = (x - np.mean(x)) / np.std(x)
        if self.bandpass:
            x = bandpass_filter(x, self.fs, *self.bandpass)
        x=x.copy()
        t = np.linspace(0, self.duration_sec, len(x))
        return torch.tensor(t, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)

# Usage:
dataset = EEGDataset('../eeg_physionet', duration_sec=4)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

for t_batch, x_batch in loader:
    print("Time shape:", t_batch.shape)   # Expected: [1, seq_len]
    print("EEG shape:", x_batch.shape)    # Expected: [1, seq_len]
    break  # Only one batch



