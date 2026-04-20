import os
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt, find_peaks
import random

# =========================
# PATHS
# =========================
RAW_PATH = 'PPG_Dataset/RawData'
LABEL_PATH = 'PPG_Dataset/Labels'

# =========================
# STEP 1: LOAD DATA
# =========================
def load_patient_data():
    signals = {}
    labels = {}

    for fname in os.listdir(RAW_PATH):
        if fname.endswith('.csv') and fname.startswith('signal_'):
            pid = fname.replace('signal_', '').replace('.csv', '')

            signal = np.loadtxt(os.path.join(RAW_PATH, fname))
            signals[pid] = signal

            label_file = f'label_{pid}.csv'
            label_path = os.path.join(LABEL_PATH, label_file)

            if os.path.exists(label_path):
                df = pd.read_csv(label_path)
                labels[pid] = df['Glucose'].values[0]
            else:
                labels[pid] = None

    return signals, labels


# =========================
# STEP 2: DOWNSAMPLE
# =========================
def downsample_signal(signal, orig_fs=2175, target_fs=100):
    new_length = int(len(signal) * target_fs / orig_fs)
    return resample(signal, new_length)


# =========================
# STEP 3: BANDPASS FILTER
# =========================
def bandpass_filter(signal, fs=100, low=0.5, high=8, order=4):
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq

    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, signal)


# =========================
# STEP 4: PEAK DETECTION
# =========================
def detect_peaks(signal, fs=100):
    peaks, _ = find_peaks(
        signal,
        distance=int(fs * 0.5),      # minimum 0.5 sec gap
        height=np.mean(signal)       # simple threshold
    )
    return peaks


# =========================
# STEP 5: SEGMENT AROUND PEAKS
# =========================
def segment_around_peaks(signal, peaks, fs=100, window_sec=1):
    window_size = int(fs * window_sec)
    half_window = window_size // 2

    segments = []

    for p in peaks:
        start = p - half_window
        end = p + half_window

        if start >= 0 and end < len(signal):
            segment = signal[start:end]
            segments.append(segment)

    return np.array(segments)


# =========================
# STEP 6: NORMALIZE EACH SEGMENT
# =========================
def normalize_segments(segments):
    norm_segments = []

    for seg in segments:
        mean = np.mean(seg)
        std = np.std(seg)

        if std == 0:
            norm = seg - mean
        else:
            norm = (seg - mean) / std

        norm_segments.append(norm)

    return np.array(norm_segments)


# =========================
# MAIN PIPELINE
# =========================
if __name__ == '__main__':
    signals, labels = load_patient_data()
    print(f'Loaded {len(signals)} patients')

    # Patient-wise split
    patient_ids = list(signals.keys())
    random.seed(42)
    random.shuffle(patient_ids)

    split = int(0.7 * len(patient_ids))
    train_ids = patient_ids[:split]
    test_ids = patient_ids[split:]

    print(f'Train: {len(train_ids)}, Test: {len(test_ids)}')

    X_train, y_train = [], []
    X_test, y_test = [], []

    # =========================
    # PROCESS FUNCTION
    # =========================
    def process_patient(pid):
        signal = signals[pid]

        # Pipeline
        signal = downsample_signal(signal)
        signal = bandpass_filter(signal)
        peaks = detect_peaks(signal)
        segments = segment_around_peaks(signal, peaks)
        segments = normalize_segments(segments)

        return segments

    # =========================
    # TRAIN SET
    # =========================
    for pid in train_ids:
        if labels[pid] is None:
            continue

        segments = process_patient(pid)

        X_train.append(segments)
        y_train.extend([labels[pid]] * len(segments))

        print(f'[TRAIN] {pid}: {len(segments)} segments')

    # =========================
    # TEST SET
    # =========================
    for pid in test_ids:
        if labels[pid] is None:
            continue

        segments = process_patient(pid)

        X_test.append(segments)
        y_test.extend([labels[pid]] * len(segments))

        print(f'[TEST] {pid}: {len(segments)} segments')

    # =========================
    # FINAL DATA
    # =========================
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Final shapes:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    # Save
    np.savez('ppg_train.npz', X=X_train, y=y_train)
    np.savez('ppg_test.npz', X=X_test, y=y_test)

    print("Saved successfully")