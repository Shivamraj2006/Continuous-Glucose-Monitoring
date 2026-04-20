import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn

# ---------- Define Model Architecture ----------
def get_cnn1d():
    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
            self.fc = nn.Sequential(
                nn.Linear(64*22, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    return CNN1D()

# ---------- Load TXT ----------
txt_data = []
with open('harshal1161402.txt', 'r') as f:
    for line in f:
        if ',' in line and line.strip()[0].isdigit():
            _, val = line.strip().split(',')
            txt_data.append(float(val))
txt_data = np.array(txt_data)

# ---------- Load NPZ ----------
npz = np.load('ppg_test.npz')
npz_data = npz[npz.files[0]].flatten()

# ---------- Remove transient ----------
txt_data = txt_data[50:]

# ---------- Bandpass Filter ----------
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)


def segment_signal(signal, window_size=100, step=50):
    segments = []
    for i in range(0, len(signal) - window_size, step):
        segments.append(signal[i:i+window_size])
    return np.array(segments)

fs = 100  # adjust if needed
filtered_txt = bandpass_filter(txt_data, 0.5, 5.0, fs)

# ---------- Normalize TXT ----------
filtered_txt = normalize_signal(filtered_txt)

# ---------- Get NPZ stats ----------
npz_mean = np.mean(npz_data)
npz_std = np.std(npz_data)

# ---------- Match amplitude to NPZ ----------
matched_txt = filtered_txt * npz_std + npz_mean

# Segment
segments = segment_signal(matched_txt)
segments = torch.tensor(segments).unsqueeze(1).float()

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_cnn1d()
model.load_state_dict(torch.load('ppg_cnn_model.pth'))
model.to(device)
model.eval()

# Predict
with torch.no_grad():
    segments = segments.to(device)
    preds = model(segments).detach().cpu().numpy()

# Final glucose
glucose = preds.mean()
print(f'Predicted Glucose: {glucose:.2f} mg/dL')

# # ---------- Plot ----------
# min_len = min(len(matched_txt), len(npz_data))

# plt.figure()
# plt.plot(npz_data[:min_len], label="Reference NPZ")
# plt.plot(matched_txt[:min_len], label="Processed TXT (matched)")
# plt.legend()
# plt.title("Amplitude Matched Signal")
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.show()