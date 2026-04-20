from fastapi import FastAPI, HTTPException
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "ppg_cnn_model.pth")
NPZ_PATH   = os.environ.get("NPZ_PATH", "ppg_test.npz")

FS          = 100
WINDOW_SIZE = 100
STEP        = 50
MIN_SAMPLES = 100

app = FastAPI()

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 22, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ─────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────
def bandpass_filter(signal):
    if len(signal) < 30:
        raise ValueError("Signal too short for filtering")
    nyquist = 0.5 * FS
    b, a = butter(4, [0.5 / nyquist, 5.0 / nyquist], btype="band")
    return filtfilt(b, a, signal)

def normalize_signal(signal):
    std = np.std(signal)
    return signal if std == 0 else (signal - np.mean(signal)) / std

def segment_signal(signal):
    return np.array([
        signal[i:i + WINDOW_SIZE]
        for i in range(0, len(signal) - WINDOW_SIZE + 1, STEP)
    ])

def run_pipeline(raw):
    data = np.array(raw, dtype=np.float64)

    if len(data) < MIN_SAMPLES:
        raise ValueError(f"Need ≥ {MIN_SAMPLES} samples")

    data = data[50:]
    data = bandpass_filter(data)
    data = normalize_signal(data)

    if npz_mean is not None and npz_std is not None:
        data = data * npz_std + npz_mean

    segments = segment_signal(data)
    if len(segments) == 0:
        raise ValueError("No segments generated")

    tensor = torch.tensor(segments, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        preds = model(tensor).cpu().numpy().flatten()

    return {
        "glucose_mg_dl": float(np.mean(preds)),
        "num_segments": int(len(preds)),
        "raw_predictions": preds.tolist(),
        "status": "ok",
    }

# ─────────────────────────────────────────────
# Load model at startup
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found: {MODEL_PATH}")

model = CNN1D()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

npz_mean = npz_std = None
if os.path.exists(NPZ_PATH):
    npz = np.load(NPZ_PATH)
    ref = npz[npz.files[0]].flatten()
    npz_mean = float(np.mean(ref))
    npz_std = float(np.std(ref))

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
def health():
    return {
        "status": "running",
        "device": str(device),
        "model": MODEL_PATH,
        "npz": NPZ_PATH if os.path.exists(NPZ_PATH) else "not found",
    }

@app.post("/predict")
def predict(body: dict):
    if "sensor_values" not in body:
        raise HTTPException(status_code=400, detail="Missing sensor_values")

    try:
        return run_pipeline(body["sensor_values"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))