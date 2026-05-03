"""
score_collected_data.py
─────────────────────────────────────────────────────────────
Step 2 of the Judge training pipeline.

Run AFTER data_collector_fft-3.py has produced fft_judge_ready.csv.

What this does:
  - Loads fft_judge_ready.csv (price_snapshot + vol_snapshot per row)
  - Rebuilds the GAF image for each window → runs through ViT → vit_prob
  - Rebuilds FFT features + LSTM sequence → runs through LSTM → lstm_prob
  - Appends vit_prob, lstm_prob, signal_strength to every row
  - Saves fft_judge_scored.csv — the true Judge training dataset

Why this matters:
  Without vit_prob and lstm_prob, the Judge learns FFT features as proxies
  for what the models think. With them, the Judge learns to combine all
  signals directly — which is what it actually does in live trading.

Usage:
  python score_collected_data.py

Output:
  fft_judge_scored.csv  — feed this into train_judge.py
"""

import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd
import os
import json
from PIL import Image
from torchvision import transforms
from scipy.signal import welch, coherence
from scipy.stats import entropy
from collections import deque

# ─────────────────────────────────────────────────────────────
# 1. CONFIG — match your watchdog paths exactly
# ─────────────────────────────────────────────────────────────
# 1. Get the current directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Point to a local 'data' and 'models' folder within your project
INPUT_CSV        = os.path.join(BASE_DIR, "data", "fft_judge_ready.csv")
OUTPUT_CSV       = os.path.join(BASE_DIR, "data", "fft_judge_scored.csv")

# 3. Point to the 'models' folder for your weights and scalers
BEST_MODEL       = os.path.join(BASE_DIR, "models", "best.pth")
LSTM_MODEL_PATH  = os.path.join(BASE_DIR, "models", "lstm_model.pt")
LSTM_SCALER_PATH = os.path.join(BASE_DIR, "models", "lstm_scaler.json")

# 4. Standard parameters (no changes needed)
JUDGE_THRESHOLD  = 0.10
LOOKBACK         = 100
LSTM_SEQ_LEN     = 10

# ─────────────────────────────────────────────────────────────
# 2. LOAD MODELS
# ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {device}")

# ViT
print("  Loading ViT...", end=" ", flush=True)
vit_model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=2)
in_features      = vit_model.head.in_features
vit_model.head   = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 2))
vit_model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
vit_model        = vit_model.to(device)
vit_model.eval()
print("✅")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# LSTM
print("  Loading LSTM...", end=" ", flush=True)
with open(LSTM_SCALER_PATH, 'r') as f:
    _scaler = json.load(f)
LSTM_FEATURE_COLS = _scaler['features']
LSTM_MEANS        = np.array(_scaler['means'])
LSTM_STDS         = np.array(_scaler['stds'])
print(f"  Scaler: {len(LSTM_FEATURE_COLS)} features, {len(LSTM_MEANS)} means, {len(LSTM_STDS)} stds")

class MasterLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=48, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)

lstm_model = MasterLSTM(input_size=len(LSTM_FEATURE_COLS) + 5)
lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
lstm_model = lstm_model.to(device)
lstm_model.eval()
print("✅\n")

# ─────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS — copied from watchdog exactly
# ─────────────────────────────────────────────────────────────
def gramian_angular_field(series, method='summation'):
    s_min, s_max = series.min(), series.max()
    if s_max - s_min == 0:
        scaled = np.zeros_like(series)
    else:
        scaled = 2 * (series - s_min) / (s_max - s_min) - 1
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)
    if method == 'summation':
        return np.cos(phi[:, None] + phi[None, :])
    else:
        return np.sin(phi[:, None] - phi[None, :])


def build_gaf_from_arrays(prices: np.ndarray, volumes: np.ndarray):
    """Build GAF image directly from arrays (no yfinance call needed)."""
    window     = prices[-LOOKBACK:]
    vol_window = volumes[-LOOKBACK:]

    vol_min, vol_max = vol_window.min(), vol_window.max()
    if vol_max - vol_min < 1e-8:
        return None

    img_sum  = gramian_angular_field(window,   method='summation')
    img_diff = gramian_angular_field(window,   method='difference')
    vol_norm = (vol_window - vol_min) / (vol_max - vol_min)
    vol_gaf  = gramian_angular_field(vol_norm, method='summation')

    rgb = np.zeros((LOOKBACK, LOOKBACK, 3))
    rgb[:, :, 0] = (img_sum  + 1) / 2
    rgb[:, :, 1] = (img_diff + 1) / 2
    rgb[:, :, 2] = (vol_gaf  + 1) / 2
    rgb = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def get_vit_prob(prices: np.ndarray, volumes: np.ndarray) -> float | None:
    img = build_gaf_from_arrays(prices, volumes)
    if img is None:
        return None
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = vit_model(tensor)
        prob   = torch.softmax(logits, dim=1)[0, 1].item()
    return round(prob, 6)


def extract_fft_features(window: np.ndarray, vol_window: np.ndarray) -> dict:
    """Matches watchdog + collector exactly."""
    prices  = window.astype(float)
    volumes = vol_window.astype(float)

    _, psd      = welch(prices, nperseg=min(len(prices), 64))
    psd_norm    = psd / (np.sum(psd) + 1e-10)
    spec_ent    = entropy(psd_norm)

    _, coh      = coherence(prices, volumes, nperseg=32)
    avg_coh     = np.mean(coh)

    first_half  = prices[:70]
    last_half   = prices[-30:]
    p_first     = np.max(np.abs(np.fft.fft(first_half))[1:])
    p_last      = np.max(np.abs(np.fft.fft(last_half))[1:])
    fft_mom     = p_last / (p_first + 1e-9)

    fft_vals       = np.abs(np.fft.fft(prices))
    meaningful_fft = fft_vals[1:len(prices) // 2]
    sorted_fft     = np.sort(meaningful_fft)[::-1]
    lock_in        = sorted_fft[0] / (sorted_fft[1] + 1e-9) if len(sorted_fft) > 1 else 1.0

    volatility   = float(np.std(prices))
    trend        = float((prices[-1] - prices[0]) / (prices[0] + 1e-8))
    volume_spike = float(volumes[-1] / (np.mean(volumes) + 1e-8))

    return {
        'spectral_entropy': round(float(spec_ent),    4),
        'vp_coherence':     round(float(avg_coh),     4),
        'fft_momentum':     round(float(fft_mom),     4),
        'lock_in_ratio':    round(float(lock_in),     4),
        'total_energy':     round(float(np.sum(psd)), 4),
        'volatility':       round(volatility,          6),
        'trend':            round(trend,               6),
        'volume_spike':     round(volume_spike,        4),
    }


def get_lstm_prob(fft_features: dict, lstm_buffer: deque) -> float | None:
    feature_row = [fft_features.get(col, 0.0) for col in LSTM_FEATURE_COLS]
    # Only scale using the first 24 means/stds (price tail handled separately)
    feature_arr = (np.array(feature_row) - LSTM_MEANS[:24]) / (LSTM_STDS[:24] + 1e-8)
    # Pad with 5 zeros for price tail (last row only, not available here)
    feature_arr = np.append(feature_arr, np.zeros(5))
    feature_row = feature_arr.tolist()
    lstm_buffer.append(feature_row)

    if len(lstm_buffer) < LSTM_SEQ_LEN:
        return None

    seq    = torch.tensor([list(lstm_buffer)], dtype=torch.float32).to(device)
    with torch.no_grad():
        prob = lstm_model(seq).item()
    return round(prob, 6)


def signal_strength_label(vit_prob: float, judge_threshold: float = JUDGE_THRESHOLD) -> str:
    """
    Matches the A/B/C grading in watchdog run_draft().
    Uses ViT prob as proxy since Judge prob not yet available.
    """
    if vit_prob >= 0.82:
        return 'A'
    elif vit_prob >= 0.65:
        return 'B'
    else:
        return 'C'

# ─────────────────────────────────────────────────────────────
# 4. MAIN SCORING LOOP
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  SCORER — appending vit_prob + lstm_prob to collected data")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)
print(f"\n  Input rows : {len(df):,}")
print(f"  Symbols    : {df['symbol'].nunique()}")
print(f"  Outcomes   : {df['outcome'].value_counts().to_dict()}\n")

# LSTM buffers — one per symbol, warm up sequentially like the watchdog
# This is critical: feeding bars in order so the LSTM sees the same
# sequential context it would in live trading
lstm_buffers = {sym: deque(maxlen=LSTM_SEQ_LEN) for sym in df['symbol'].unique()}

vit_probs        = []
lstm_probs       = []
signal_strengths = []

skipped_vit  = 0
skipped_lstm = 0
total        = len(df)

for idx, row in df.iterrows():
    if idx % 500 == 0:
        print(f"  Scoring row {idx:,} / {total:,}...", flush=True)

    try:
        prices  = np.array([float(x) for x in row['price_snapshot'].split(',')])
        volumes = np.array([float(x) for x in row['vol_snapshot'].split(',')])
    except Exception:
        vit_probs.append(None)
        lstm_probs.append(None)
        signal_strengths.append(None)
        skipped_vit += 1
        continue

    # ── ViT prob ──
    vit_p = get_vit_prob(prices, volumes)
    if vit_p is None:
        skipped_vit += 1

    # ── FFT features (needed for LSTM input) ──
    fft_feats = extract_fft_features(prices, volumes)

    # ── LSTM prob — feed into per-symbol buffer in bar order ──
    symbol = row['symbol']
    lstm_p = get_lstm_prob(fft_feats, lstm_buffers[symbol])
    if lstm_p is None:
        skipped_lstm += 1

    # ── Signal strength — based on vit_prob ──
    strength = signal_strength_label(vit_p) if vit_p is not None else None

    vit_probs.append(vit_p)
    lstm_probs.append(lstm_p)
    signal_strengths.append(strength)

# ─────────────────────────────────────────────────────────────
# 5. APPEND & SAVE
# ─────────────────────────────────────────────────────────────
df['vit_prob']        = vit_probs
df['lstm_prob']       = lstm_probs
df['signal_strength'] = signal_strengths

# Drop rows where either model couldn't score
# (LSTM warmup rows at start of each symbol sequence)
before = len(df)
df_scored = df.dropna(subset=['vit_prob', 'lstm_prob']).copy()
dropped   = before - len(df_scored)

df_scored.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'='*60}")
print(f"  SCORING COMPLETE")
print(f"  Input rows    : {before:,}")
print(f"  Dropped       : {dropped:,}  (LSTM warmup + parse errors)")
print(f"  Scored rows   : {len(df_scored):,}")
print(f"  ViT skipped   : {skipped_vit:,}")
print(f"  LSTM warmup   : {skipped_lstm:,}")
print(f"\n  Outcome breakdown:")
print(df_scored['outcome'].value_counts().to_string())
print(f"\n  Signal strength breakdown:")
print(df_scored['signal_strength'].value_counts().to_string())
print(f"\n  ViT prob stats:")
print(df_scored['vit_prob'].describe().round(4).to_string())
print(f"\n  LSTM prob stats:")
print(df_scored['lstm_prob'].describe().round(4).to_string())
print(f"\n  Saved to : {OUTPUT_CSV}")
print(f"{'='*60}")
print(f"\n  NEXT STEP:")
print(f"  Update train_judge.py FEATURE_COLS to include:")
print(f"    vit_prob, lstm_prob, vol_regime, slope_regime, atr,")
print(f"    spectral_entropy, vp_coherence, fft_momentum,")
print(f"    lock_in_ratio, total_energy, sector_encoded, is_new_ticker")
print(f"  And set input file to: fft_judge_scored.csv")
