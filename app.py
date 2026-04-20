"""
PPG Glucose Prediction — Standalone HTTP Server
================================================
Run:  python ppg_server.py
      (optional) MODEL_PATH=my_model.pth NPZ_PATH=ppg_test.npz python ppg_server.py

Endpoints
---------
GET  /              → health check
POST /predict       → JSON body  { "sensor_values": [float, ...] }
                    → JSON resp  { "glucose_mg_dl": float,
                                   "num_segments": int,
                                   "raw_predictions": [float, ...],
                                   "status": "ok" }
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import traceback

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt

# ─────────────────────────────────────────────
# Config  (override via environment variables)
# ─────────────────────────────────────────────
HOST        = os.environ.get("HOST", "0.0.0.0")
PORT        = int(os.environ.get("PORT", "8080"))
MODEL_PATH  = os.environ.get("MODEL_PATH", "ppg_cnn_model.pth")
NPZ_PATH    = os.environ.get("NPZ_PATH", "ppg_test.npz")
FS          = 100          # sampling frequency (Hz)
WINDOW_SIZE = 100          # samples per segment
STEP        = 50           # segment stride
MIN_SAMPLES = 100          # minimum raw samples required


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
def bandpass_filter(signal: np.ndarray, lowcut=0.5, highcut=5.0,
                    fs=FS, order=4) -> np.ndarray:
    if len(signal) < 30:
        raise ValueError("Signal too short for filtering (need ≥ 30 samples)")
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    std = np.std(signal)
    return signal if std == 0 else (signal - np.mean(signal)) / std


def segment_signal(signal: np.ndarray, window_size=WINDOW_SIZE,
                   step=STEP) -> np.ndarray:
    segs = [signal[i:i + window_size]
            for i in range(0, len(signal) - window_size + 1, step)]
    return np.array(segs)


def run_pipeline(raw: list, model: nn.Module, device: torch.device,
                 npz_mean=None, npz_std=None) -> dict:
    data = np.array(raw, dtype=np.float64)

    if len(data) < MIN_SAMPLES:
        raise ValueError(f"Need ≥ {MIN_SAMPLES} samples, got {len(data)}")

    data = data[50:]                          # drop transient
    data = bandpass_filter(data)
    data = normalize_signal(data)

    if npz_mean is not None and npz_std is not None:
        data = data * npz_std + npz_mean      # amplitude match

    segments = segment_signal(data)
    if len(segments) == 0:
        raise ValueError("No segments generated — signal may be too short")

    tensor = torch.tensor(segments, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        preds = model(tensor).cpu().numpy().flatten()

    return {
        "glucose_mg_dl":  float(np.mean(preds)),
        "num_segments":   int(len(preds)),
        "raw_predictions": preds.tolist(),
        "status": "ok",
    }


# ─────────────────────────────────────────────
# HTTP handler
# ─────────────────────────────────────────────
class PPGHandler(BaseHTTPRequestHandler):

    # silence default request logging — replace with custom
    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}")

    # ── helpers ──────────────────────────────
    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            raise ValueError("Empty request body")
        raw = self.rfile.read(length)
        return json.loads(raw)

    # ── routes ───────────────────────────────
    def do_GET(self):
        if self.path in ("/", "/health"):
            self._send_json(200, {
                "status": "running",
                "device": str(self.server.device),
                "model":  MODEL_PATH,
                "npz":    NPZ_PATH if os.path.exists(NPZ_PATH) else "not found",
            })
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/predict":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            body = self._read_json_body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"Bad JSON: {exc}"})
            return

        if "sensor_values" not in body:
            self._send_json(400, {"error": "Missing field: sensor_values"})
            return

        try:
            result = run_pipeline(
                raw      = body["sensor_values"],
                model    = self.server.model,
                device   = self.server.device,
                npz_mean = self.server.npz_mean,
                npz_std  = self.server.npz_std,
            )
            self._send_json(200, result)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception:
            self._send_json(500, {"error": traceback.format_exc()})


# ─────────────────────────────────────────────
# Server bootstrap
# ─────────────────────────────────────────────
def build_server() -> HTTPServer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}")

    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = CNN1D()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device,
                                     weights_only=True))
    model.to(device)
    model.eval()
    print(f"[init] model loaded from {MODEL_PATH}")

    # Load NPZ reference stats (optional)
    npz_mean = npz_std = None
    if os.path.exists(NPZ_PATH):
        npz      = np.load(NPZ_PATH)
        ref      = npz[npz.files[0]].flatten()
        npz_mean = float(np.mean(ref))
        npz_std  = float(np.std(ref))
        print(f"[init] NPZ stats — mean={npz_mean:.4f}  std={npz_std:.4f}")
    else:
        print(f"[init] NPZ not found ({NPZ_PATH}), amplitude matching disabled")

    server = HTTPServer((HOST, PORT), PPGHandler)
    server.device   = device
    server.model    = model
    server.npz_mean = npz_mean
    server.npz_std  = npz_std
    return server


if __name__ == "__main__":
    server = build_server()
    print(f"\n✓ PPG server listening on http://{HOST}:{PORT}")
    print("  POST /predict   { \"sensor_values\": [...] }")
    print("  GET  /health\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[shutdown] bye!")
        server.server_close()