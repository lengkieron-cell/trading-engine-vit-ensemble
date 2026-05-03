import torch
import torch.nn as nn
import timm
import yfinance as yf
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
import xgboost as xgb
from scipy.signal import welch, coherence
from scipy.stats import entropy

# ─────────────────────────────────────────────────────────────
# 1. CONFIG  ← only section you ever need to touch
# ─────────────────────────────────────────────────────────────
JUDGE_MODEL_PATH     = r"C:\Users\Kylek\Downloads\Kieron Stuff\Trader-2026\MODELS\judge_model.json"
LSTM_MODEL_PATH      = r"C:\Users\Kylek\Downloads\Kieron Stuff\Trader-2026\MODELS\lstm_model.pt"
LSTM_SCALER_PATH     = r"C:\Users\Kylek\Downloads\Kieron Stuff\Trader-2026\MODELS\lstm_scaler.json"
JUDGE_THRESHOLD      = 0.51   # updated to match new meta-Judge optimal threshold
LSTM_SEQ_LEN         = 10     # must match train_lstm.py

BEST_MODEL           = r"C:\Users\Kylek\Downloads\Kieron Stuff\Trader-2026\MODELS\best.pth"
CONFIDENCE_THRESHOLD = 0.65
TOP_N                = 5
LOOKBACK             = 100
FORECAST             = 15
OUTPUT_DIR           = r"C:\Users\Kylek\Downloads\Kieron Stuff\Trader-2026"
LOGBOOK_PATH         = os.path.join(OUTPUT_DIR, "LOGBOOK.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Auto-Auditor settings
AUDIT_DURATION_MINS  = 75    # total watch window in minutes
POLL_INTERVAL_SECS   = 60    # check price every N seconds

# ─────────────────────────────────────────────────────────────
# 2. ALLOWED TICKERS
# ─────────────────────────────────────────────────────────────
ALLOWED_SECTORS = {
    'Financials': ['GS', 'JPM', 'MS', 'BAC', 'WFC', 'BLK', 'SCHW', 'V', 'MA', 'AXP', 'CB', 'ICE', 'FI', 'SPGI'],
    'Technology': ['TSLA', 'NVDA', 'AAPL', 'AMD', 'MSFT', 'GOOGL', 'META', 'AMZN', 
                   'NFLX', 'INTC', 'SOFI', 'PLTR', 'COIN', 'UBER', 'PYPL', 'SNAP', 
                   'SPOT', 'HOOD', 'RIVN', 'BABA', 'MSTR', 'MARA', 'SMCI', 'AVGO', 'ARM',
                   'ORCL', 'ADBE', 'ASML', 'CRM', 'ACN', 'CSCO', 'TXN', 'INTU', 'AMAT', 'SNPS', 'KLAC'],
    'Healthcare': ['LLY', 'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'AMGN', 'ISRG', 'VRTX', 'TMO', 'ABT', 'DHR', 'GILD', 'REGN', 'ZTS', 'BSX', 'MDT'],
    'Growth_Retail': ['DIS', 'SBUX', 'NKE', 'TSM', 'COST', 'WMT', 'TGT', 'PANW', 'SNOW', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'LOW', 'ORLY', 'TJX', 'MDLZ'],
    'Industrials_Energy': ['XOM', 'CVX', 'CAT', 'GE', 'UNP', 'HON', 'RTX', 'LMT', 'DE', 'BA', 'ETN', 'COP']
}

# This stays the same and will now automatically grab all ~100 tickers
ALL_TICKERS = [t for tickers in ALLOWED_SECTORS.values() for t in tickers]

def get_sector(symbol):
    for sector, tickers in ALLOWED_SECTORS.items():
        if symbol in tickers:
            return sector
    return None

# ─────────────────────────────────────────────────────────────
# 3. GAF BUILDER
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

def build_gaf_image(symbol):
    try:
        data    = yf.download(symbol, period='5d', interval='5m',
                              progress=False, auto_adjust=True)
        prices  = data['Close'].values.flatten()
        volumes = data['Volume'].values.flatten()

        if len(prices) < LOOKBACK:
            print(f"  ⚠️  {symbol}: only {len(prices)} bars, need {LOOKBACK}. Skipping.")
            return None

        window     = prices[-LOOKBACK:]
        vol_window = volumes[-LOOKBACK:]

        if np.isnan(window).any() or np.isnan(vol_window).any():
            print(f"  ⚠️  {symbol}: NaN values found. Skipping.")
            return None

        vol_min, vol_max = vol_window.min(), vol_window.max()
        if vol_max - vol_min < 1e-8:
            print(f"  ⚠️  {symbol}: flat volume. Skipping.")
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

        img_uint8 = (rgb * 255).astype(np.uint8)
        return Image.fromarray(img_uint8)

    except Exception as e:
        print(f"  ❌  {symbol}: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# 4. TRANSFORM
# ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ─────────────────────────────────────────────────────────────
# 5. LOAD MODEL
# ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=2)
in_features = model.head.in_features
model.head  = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 2))
model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
model = model.to(device)
model.eval()
print("✅ Model loaded\n")

# ─────────────────────────────────────────────────────────────
# 5b. LOAD JUDGE
# ─────────────────────────────────────────────────────────────
judge = xgb.XGBClassifier()
judge.load_model(JUDGE_MODEL_PATH)
print("✅ Judge loaded\n")

# ── Load LSTM ──────────────────────────────────────────────
import json as _json

with open(LSTM_SCALER_PATH, 'r') as f:
    _scaler = _json.load(f)

LSTM_FEATURE_COLS = _scaler['features']
LSTM_MEANS        = np.array(_scaler['means'])
LSTM_STDS         = np.array(_scaler['stds'])

class MasterLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.5):
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
print("✅ LSTM loaded\n")

# Rolling feature buffer — stores last SEQ_LEN feature rows per ticker
# Format: { 'NVDA': deque([row1, row2, ...], maxlen=SEQ_LEN) }
from collections import deque
lstm_buffers     = {ticker: deque(maxlen=LSTM_SEQ_LEN) for ticker in ALL_TICKERS}
lstm_fft_buffers = {ticker: deque(maxlen=10) for ticker in ALL_TICKERS}

# ─────────────────────────────────────────────────────────────
# 6. INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────
def get_win_probability(pil_image):
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob   = torch.softmax(output, dim=1)[0][1].item()
    return prob

# ─────────────────────────────────────────────────────────────
# 6b. FFT FEATURE EXTRACTOR  (matches data_collector_fft.py exactly)
# ─────────────────────────────────────────────────────────────
def extract_fft_features(window: np.ndarray, vol_window: np.ndarray) -> dict:
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

    return {
        'spectral_entropy': float(spec_ent),
        'vp_coherence':     float(avg_coh),
        'fft_momentum':     float(fft_mom),
        'lock_in_ratio':    float(lock_in),
        'total_energy':     float(np.sum(psd)),
    }

def get_lstm_prob(symbol: str, fft: dict, window: np.ndarray) -> float:
    """
    Gets LSTM win probability using rolling feature buffer with delta features
    and raw price tail — matches train_lstm.py exactly.
    """
    buf     = lstm_buffers[symbol]
    fft_buf = lstm_fft_buffers[symbol]  # second buffer for delta calculation

    # ── Base feature row ──
    base_row = np.array([
        fft['spectral_entropy'],
        fft['vp_coherence'],
        fft['fft_momentum'],
        fft['lock_in_ratio'],
        fft['total_energy'],
    ], dtype=np.float32)

    fft_buf.append(base_row)

    # ── Delta features (need at least 4 bars in fft_buf) ──
    if len(fft_buf) < 4:
        return None

    arr  = np.array(list(fft_buf))   # shape (buf_len, 5)
    d1   = arr[-1] - arr[-2]         # 1-bar change
    d3   = arr[-1] - arr[-4]         # 3-bar change

    # ── Raw price tail (last 5 bars normalised) ──
    price_tail      = window[-5:].astype(np.float32)
    price_tail_norm = (price_tail - price_tail[0]) / (price_tail[0] + 1e-8)

    # ── Full feature row: base + deltas + price tail ──
    full_row = np.concatenate([base_row, d1, d3, price_tail_norm])
    buf.append(full_row)

    if len(buf) < LSTM_SEQ_LEN:
        return None

    seq        = np.array(list(buf), dtype=np.float32)
    seq_scaled = (seq - LSTM_MEANS) / LSTM_STDS
    tensor     = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        return float(lstm_model(tensor).item())

def get_judge_verdict(vit_prob: float, lstm_prob: float,
                      vol: float, slope: float, atr: float,
                      fft: dict, sector: str) -> float:
    """
    Matches train_judge-5.py FEATURE_COLS exactly — 12 features.
    """
    SECTOR_MAP = {
        'Financials': 0, 'Technology': 1, 'Healthcare': 2,
        'Growth_Retail': 3, 'Industrials_Energy': 4
    }
    sector_idx = SECTOR_MAP.get(sector, 1)
    features = np.array([[
        vit_prob,
        lstm_prob,
        fft['spectral_entropy'],
        fft['vp_coherence'],
        fft['fft_momentum'],
        fft['lock_in_ratio'],
        fft['total_energy'],
        vol,
        slope,
        atr,
        sector_idx,
        1,   # is_new_ticker — always 1 for live trading
    ]])
    return float(judge.predict_proba(features)[0, 1])


# ─────────────────────────────────────────────────────────────
# 6c. REGIME FEATURES
# ─────────────────────────────────────────────────────────────
def get_regime_features(prices, window_size=20):
    returns = np.diff(np.log(prices[-window_size:]))
    vol     = np.std(returns) * np.sqrt(252 * 78)  # annualised for 5m bars
    y       = prices[-10:]
    x       = np.arange(len(y))
    slope   = np.polyfit(x, y, 1)[0]
    return float(vol), float(slope)

# ─────────────────────────────────────────────────────────────
# 7. LIVE PRICE FETCH
# ─────────────────────────────────────────────────────────────
def get_live_price(symbol):
    """Fetch the latest price for a symbol using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d', interval='1m')
        if data.empty:
            return None
        return float(data['Close'].iloc[-1])
    except Exception as e:
        print(f"  ⚠️  Price fetch error for {symbol}: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# 8. LOGBOOK WRITER
# ─────────────────────────────────────────────────────────────
def write_to_logbook(entry: str):
    """Append a single entry to the LOGBOOK.txt file."""
    with open(LOGBOOK_PATH, 'a', encoding='utf-8') as f:
        f.write(entry + "\n")

def log_session_header(num_signals):
    """Write a session header to the logbook."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header = (
        f"\n{'='*60}\n"
        f"  SESSION START: {now}\n"
        f"  Signals to audit: {num_signals}\n"
        f"  Watch window: {AUDIT_DURATION_MINS} minutes\n"
        f"  Stop-loss polling: every {POLL_INTERVAL_SECS} seconds\n"
        f"{'='*60}"
    )
    write_to_logbook(header)
    print(header)

# ─────────────────────────────────────────────────────────────
# 9. AUTO-AUDITOR (Shadow Logger)
# ─────────────────────────────────────────────────────────────
STATS = {"wins": 0, "losses": 0, "stops": 0}
def audit_trade(signal: dict):
    """
    Watches a single trade for AUDIT_DURATION_MINS minutes.
    - Polls price every POLL_INTERVAL_SECS seconds.
    - Logs STOPPED OUT if price hits or drops below stop_price.
    - Logs WIN or LOSS at the 75-minute mark.
    Runs in its own thread so all signals are watched concurrently.
    """
    symbol        = signal['symbol']
    sector        = signal['sector']
    win_prob      = signal['win_prob']
    entry_price   = signal.get('current_price')
    stop_price    = signal.get('stop_price')
    vol_regime    = signal.get('vol_regime')
    slope_regime  = signal.get('slope_regime')
    entry_time    = datetime.now()

    if entry_price is None:
        msg = f"[{symbol}] Could not get entry price — skipping audit."
        print(f"  ⚠️  {msg}")
        write_to_logbook(msg)
        return

    header = (
        f"\n  {'─'*50}\n"
        f"  🔍 AUDITING: {symbol} ({sector})\n"
        f"     Entry Time  : {entry_time.strftime('%H:%M:%S')}\n"
        f"     Entry Price : ${entry_price:.2f}\n"
        f"     Stop Loss   : ${stop_price:.2f if stop_price else 'N/A'}\n"
        f"     Win Prob    : {win_prob:.1%}\n"
        f"     Watch Until : {(entry_time.replace(second=0, microsecond=0))}\n"
        f"  {'─'*50}"
    )
    write_to_logbook(header)
    print(header)

    end_time     = time.time() + (AUDIT_DURATION_MINS * 60)
    minute_count = 0
    outcome      = None

    while time.time() < end_time:
        time.sleep(POLL_INTERVAL_SECS)
        minute_count += 1
        current_price = get_live_price(symbol)

        if current_price is None:
            tick_msg = f"  [{symbol}] Min {minute_count:02d}: Price unavailable, retrying next poll."
            print(tick_msg)
            continue

        elapsed = minute_count
        print(f"  [{symbol}] Min {elapsed:02d}: Current = ${current_price:.2f} | Entry = ${entry_price:.2f}"
              + (f" | Stop = ${stop_price:.2f}" if stop_price else ""))

# Check stop-loss hit
        if stop_price and current_price <= stop_price:
            outcome = "STOPPED OUT"
            STATS["stops"] += 1
            result_line = (
                f"\n  🛑 STOPPED OUT — {symbol}\n"
                f"     Time    : {datetime.now().strftime('%H:%M:%S')} (min {elapsed})\n"
                f"     Price   : ${current_price:.2f}  (hit stop ${stop_price:.2f})\n"
                f"     Entry   : ${entry_price:.2f}  |  Loss: ${current_price - entry_price:.2f} per share\n"
                f"     Sector  : {sector}\n"
            )
            print(result_line)
            write_to_logbook(result_line)

            # ── Judge training data ──
            judge_row = {
                'scan_time':    entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                'symbol':       symbol,
                'sector':       sector,
                'model1_prob':  win_prob,
                'entry_price':  entry_price,
                'stop_price':   stop_price,
                'final_price':  round(float(current_price), 2),
                'outcome':      outcome,
                'minutes_held': elapsed,
                'vol_regime':   vol_regime,
                'slope_regime': slope_regime,
            }
            judge_csv = os.path.join(OUTPUT_DIR, "judge_training_data.csv")
            pd.DataFrame([judge_row]).to_csv(judge_csv, mode='a', header=not os.path.exists(judge_csv), index=False)
            return

# 75-minute final check
    final_price = get_live_price(symbol)
    if final_price is None:
        final_price = current_price  # use last known if fetch fails

    if final_price is None:
        outcome = "UNKNOWN"
        result_line = (
            f"\n  ❓ UNKNOWN — {symbol}\n"
            f"     Could not fetch final price at {AUDIT_DURATION_MINS} minutes.\n"
        )
        print(result_line)
        write_to_logbook(result_line)

    elif final_price > entry_price:
        outcome = "WIN"
        STATS["wins"] += 1
        pnl = final_price - entry_price
        pct = ((final_price - entry_price) / entry_price) * 100
        result_line = (
            f"\n  ✅ WIN — {symbol}\n"
            f"     Time     : {datetime.now().strftime('%H:%M:%S')} ({AUDIT_DURATION_MINS} min mark)\n"
            f"     Entry    : ${entry_price:.2f}  →  Exit: ${final_price:.2f}\n"
            f"     Gain     : +${pnl:.2f} per share  (+{pct:.2f}%)\n"
            f"     Win Prob : {win_prob:.1%}  |  Sector: {sector}\n"
        )
        print(result_line)
        write_to_logbook(result_line)

        # ── Judge training data ──
        judge_row = {
            'scan_time':    entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'symbol':       symbol,
            'sector':       sector,
            'model1_prob':  win_prob,
            'entry_price':  entry_price,
            'stop_price':   stop_price,
            'final_price':  round(float(final_price), 2),
            'outcome':      outcome,
            'minutes_held': minute_count,
            'vol_regime':   vol_regime,
            'slope_regime': slope_regime,
        }
        judge_csv = os.path.join(OUTPUT_DIR, "judge_training_data.csv")
        pd.DataFrame([judge_row]).to_csv(judge_csv, mode='a', header=not os.path.exists(judge_csv), index=False)

    else:
        outcome = "LOSS"
        STATS["losses"] += 1
        pnl = final_price - entry_price
        pct = ((final_price - entry_price) / entry_price) * 100
        result_line = (
            f"\n  ❌ LOSS — {symbol}\n"
            f"     Time     : {datetime.now().strftime('%H:%M:%S')} ({AUDIT_DURATION_MINS} min mark)\n"
            f"     Entry    : ${entry_price:.2f}  →  Exit: ${final_price:.2f}\n"
            f"     Loss     : ${pnl:.2f} per share  ({pct:.2f}%)\n"
            f"     Win Prob : {win_prob:.1%}  |  Sector: {sector}\n"
        )
        print(result_line)
        write_to_logbook(result_line)

        # ── Judge training data ──
        judge_row = {
            'scan_time':    entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'symbol':       symbol,
            'sector':       sector,
            'model1_prob':  win_prob,
            'entry_price':  entry_price,
            'stop_price':   stop_price,
            'final_price':  round(float(final_price), 2),
            'outcome':      outcome,
            'minutes_held': minute_count,
            'vol_regime':   vol_regime,
            'slope_regime': slope_regime,
        }
        judge_csv = os.path.join(OUTPUT_DIR, "judge_training_data.csv")
        pd.DataFrame([judge_row]).to_csv(judge_csv, mode='a', header=not os.path.exists(judge_csv), index=False)

def run_auditor(signals: list):
    """
    Launches a watcher thread for each signal concurrently.
    All threads run in the background — main thread waits for all to finish.
    """
    if not signals:
        print("\n  No signals to audit.")
        return

    print(f"\n{'='*55}")
    print(f"  🚀 AUTO-AUDITOR STARTED — watching {len(signals)} trade(s)")
    print(f"  All trades monitored concurrently for {AUDIT_DURATION_MINS} mins")
    print(f"{'='*55}\n")

    log_session_header(len(signals))

    threads = []
    for signal in signals:
        t = threading.Thread(target=audit_trade, args=(signal,), daemon=True)
        threads.append(t)
        t.start()

    # Wait for all watchers to finish

    for t in threads:
        t.join()

    total = STATS["wins"] + STATS["losses"] + STATS["stops"]
    win_pct = (STATS["wins"] / total * 100) if total > 0 else 0
    scoreboard = (
        f"\n  📊 LIVE SCOREBOARD (this session)\n"
        f"     ✅ Wins       : {STATS['wins']}\n"
        f"     ❌ Losses     : {STATS['losses']}\n"
        f"     🛑 Stopped    : {STATS['stops']}\n"
        f"     🎯 Win Rate   : {win_pct:.1f}%  ({total} trades)\n"
        + (f"     ⚠️  Small sample — need 20+ trades for reliable %\n" if total < 20 else "")
    )
    print(scoreboard)
    write_to_logbook(scoreboard)

    summary = (
        f"\n{'='*60}\n"
        f"  SESSION COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  All {len(signals)} trade(s) audited. See LOGBOOK.txt for full results.\n"
        f"{'='*60}\n"
    )
    print(summary)
    write_to_logbook(summary)

# ─────────────────────────────────────────────────────────────
# 10. DRAFT SYSTEM
# ─────────────────────────────────────────────────────────────
def run_draft(tickers=ALL_TICKERS,
              threshold=CONFIDENCE_THRESHOLD,
              top_n=TOP_N):
    print(f"{'='*55}")
    print(f"  DRAFT SYSTEM — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Threshold: {threshold} | Top-N: {top_n}")
    print(f"  Scanning {len(tickers)} tickers...")
    print(f"{'='*55}\n")

    all_signals = []

    for symbol in tickers:
        sector = get_sector(symbol)
        print(f"  Scanning {symbol} ({sector})...", end=" ")

        img = build_gaf_image(symbol)
        if img is None:
            continue

        prob = get_win_probability(img)
        print(f"Win prob: {prob:.3f}", end="")

        if prob >= threshold:
            # ── FFT + LSTM + Judge pipeline ───────────────────
            data       = yf.download(symbol, period='5d', interval='5m',
                                     progress=False, auto_adjust=True)
            prices     = data['Close'].values.flatten()
            volumes    = data['Volume'].values.flatten()
            window     = prices[-LOOKBACK:]
            vol_window = volumes[-LOOKBACK:]

            fft = extract_fft_features(window, vol_window)

            # ── Pre-warm LSTM buffer if not ready ──────────────
            if len(lstm_buffers[symbol]) < LSTM_SEQ_LEN:
                for i in range(15, 0, -1):
                    end_idx   = len(prices) - i
                    start_idx = end_idx - LOOKBACK
                    if start_idx < 0:
                        continue
                    sub_window = prices[start_idx:end_idx]
                    sub_vol    = volumes[start_idx:end_idx]
                    if len(sub_window) < LOOKBACK:
                        continue
                    sub_fft   = extract_fft_features(sub_window, sub_vol)
                    lstm_prob  = get_lstm_prob(symbol, sub_fft, sub_window)

            lstm_prob = get_lstm_prob(symbol, fft, window)

            if lstm_prob is None:
                print(f"  | LSTM: still warming ({len(lstm_buffers[symbol])}/{LSTM_SEQ_LEN})")
                continue

            vol, slope = get_regime_features(prices)

            # Dynamic threshold — raise bar in choppy markets
            if vol > 0.40:
                effective_threshold = JUDGE_THRESHOLD + 0.08
                print(f"  ⚠️ High vol ({vol:.2f}) — threshold raised to {effective_threshold:.2f}")
            else:
                effective_threshold = JUDGE_THRESHOLD

            # ── ViT Priority Tiered Logic ─────────────────────
            if prob >= 0.80:
                # ViT highly confident — trade directly, skip Judge
                print(f"  | LSTM: {lstm_prob:.3f} | Judge: BYPASSED (ViT≥0.80)", end="")
                all_signals.append({
                    'symbol':       symbol,
                    'sector':       sector,
                    'win_prob':     round(prob, 4),
                    'lstm_prob':    round(lstm_prob, 4),
                    'judge_prob':   None,
                    'vol_regime':   round(vol, 4),
                    'slope_regime': round(slope, 4),
                })
                print(f"  ✅ APPROVED (ViT Priority)")

            else:
                # ViT moderately confident (0.65–0.80) — ask Judge
                risk_data  = get_atr_stop(symbol)
                atr_val    = risk_data['atr'] if risk_data else 0.0
                judge_prob = get_judge_verdict(prob, lstm_prob, vol, slope, atr_val, fft, sector)
                print(f"  | LSTM: {lstm_prob:.3f} | Judge: {judge_prob:.3f}", end="")

                if judge_prob >= effective_threshold:
                    all_signals.append({
                        'symbol':       symbol,
                        'sector':       sector,
                        'win_prob':     round(prob, 4),
                        'lstm_prob':    round(lstm_prob, 4),
                        'judge_prob':   round(judge_prob, 4),
                        'vol_regime':   round(vol, 4),
                        'slope_regime': round(slope, 4),
                    })
                    print(f"  ✅ APPROVED")
                else:
                    print(f"  ⛔ VETOED by Judge")
        elif prob >= 0.50:
            
# Ghost signal — log for research but don't trade
            ghost_data   = yf.download(symbol, period='5d', interval='5m',
                                       progress=False, auto_adjust=True)
            ghost_prices = ghost_data['Close'].values.flatten()
            ghost_vol, ghost_slope = get_regime_features(ghost_prices) if len(ghost_prices) >= 20 else (None, None)
            ghost_row = {
                'timestamp':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol':       symbol,
                'sector':       sector,
                'signal_type':  'GHOST',
                'vit_prob':     round(prob, 4),
                'judge_prob':   None,
                'entry_price':  None,
                'tp_price':     None,
                'sl_price':     None,
                'atr_value':    None,
                'vol_regime':   round(ghost_vol, 4) if ghost_vol is not None else None,
                'slope_regime': round(ghost_slope, 4) if ghost_slope is not None else None,
            }
            perf_csv = os.path.join(OUTPUT_DIR, "RESEARCH_LOG.csv")
            pd.DataFrame([ghost_row]).to_csv(perf_csv, mode='a', header=not os.path.exists(perf_csv), index=False)
            print(f"  👻 GHOST")
        else:
            print(f"  — below threshold")

    all_signals.sort(key=lambda x: x['win_prob'], reverse=True)
    top_signals = all_signals[:top_n]

    return top_signals

# ─────────────────────────────────────────────────────────────
# 11. RISK SIZING (ATR-based)
# ─────────────────────────────────────────────────────────────
def get_atr_stop(symbol, period=14, portfolio_value=10000, risk_pct=0.02):
    try:
        data   = yf.download(symbol, period='5d', interval='5m',
                              progress=False, auto_adjust=True)
        highs  = data['High'].values.flatten()
        lows   = data['Low'].values.flatten()
        closes = data['Close'].values.flatten()

        if len(closes) < period + 1:
            return None

        tr = np.maximum(highs[1:] - lows[1:],
             np.maximum(np.abs(highs[1:] - closes[:-1]),
                        np.abs(lows[1:]  - closes[:-1])))
        atr        = tr[-period:].mean()
        price      = closes[-1]
        atr_pct    = atr / price
        stop_price = price - (1.5 * atr)
        risk_amt   = portfolio_value * risk_pct
        position   = int(risk_amt / (1.5 * atr))

        return {
            'current_price': round(float(price), 2),
            'atr':           round(float(atr), 4),
            'atr_pct':       round(float(atr_pct * 100), 2),
            'stop_price':    round(float(stop_price), 2),
            'position_size': position,
            'risk_amount':   round(risk_amt, 2),
        }
    except Exception as e:
        print(f"  ATR error for {symbol}: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# 12. SCHEDULER & AUTONOMOUS SESSION
# ─────────────────────────────────────────────────────────────
PORTFOLIO_VALUE = 10000   # ← set your actual portfolio size here

# Power windows in 24h EST — the three times the bot will wake up and scan
POWER_WINDOWS = [15, 18, 21]

def get_seconds_until_next_window():
    """Returns seconds to sleep until the next power window (10:00, 13:00, 15:00)."""
    now = datetime.now()
    for hour in POWER_WINDOWS:
        target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if target > now:
            delta = (target - now).total_seconds()
            return delta, target.strftime('%H:%M')
    # All windows passed today — sleep until 10:00 tomorrow
    tomorrow_target = (now + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
    delta = (tomorrow_target - now).total_seconds()
    return delta, tomorrow_target.strftime('%Y-%m-%d 10:00')

def run_one_batch():
    """Runs a full scan + ATR sizing + audit for one power window."""
    signals = run_draft()

    print(f"\n{'='*55}")
    print(f"  RESULTS: {len(signals)} signal(s) above threshold {CONFIDENCE_THRESHOLD}")
    print(f"{'='*55}\n")

    if not signals:
        print("  No signals this window. Model says: sit on hands. ✋")
        write_to_logbook(f"\n[{datetime.now().strftime('%H:%M')}] No signals this window.\n")
        return

    rows = []
    for s in signals:
        judge_conf = s.get('judge_prob', 0) or 0
        risk_pct   = 0.03 if judge_conf >= 0.65 else 0.02 if judge_conf >= 0.55 else 0.01
        risk = get_atr_stop(s['symbol'], portfolio_value=PORTFOLIO_VALUE, risk_pct=risk_pct)
        row  = {**s, **(risk if risk else {})}
        rows.append(row)

        print(f"  {'─'*50}")
        print(f"  🎯 {s['symbol']}  ({s['sector']})")
        print(f"     Win probability : {s['win_prob']:.1%}")
        if risk:
            print(f"     Entry price     : ${risk['current_price']}")
            print(f"     Stop loss       : ${risk['stop_price']}  ({risk['atr_pct']}% ATR)")
            print(f"     Position size   : {risk['position_size']} shares")
            print(f"     Max risk        : ${risk['risk_amount']}")
            s.update(risk)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    csv_path  = os.path.join(OUTPUT_DIR, f"signals_{timestamp}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  💾 Signal sheet saved: {csv_path}")

        # ── Performance snapshot for research ─────────────────────
    for s in signals:
        risk  = next((r for r in rows if r.get('symbol') == s['symbol']), {})
        entry = risk.get('current_price')
        atr   = risk.get('atr')
        perf_row = {
            'timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol':      s['symbol'],
            'sector':      s['sector'],
            'signal_type': 'APPROVED',
            'vit_prob':    s.get('win_prob'),
            'judge_prob':  s.get('judge_prob'),
            'entry_price': entry,
            'tp_price':    round(entry * 1.01, 2) if entry else None,
            'sl_price':     risk.get('stop_price'),
            'atr_value':    atr,
            'vol_regime':   s.get('vol_regime'),
            'slope_regime': s.get('slope_regime'),
        }
        perf_csv = os.path.join(OUTPUT_DIR, "RESEARCH_LOG.csv")
        pd.DataFrame([perf_row]).to_csv(perf_csv, mode='a',
                      header=not os.path.exists(perf_csv), index=False)

    print(f"\n  ⏱️  Starting Auto-Auditor in 5 seconds...")
    print(f"     Logbook: {LOGBOOK_PATH}")
    time.sleep(5)
    run_auditor(signals)

def start_autonomous_session():
    """Main loop — runs forever, waking at each power window."""
    print(f"\n{'='*55}")
    print(f"  🤖 AUTONOMOUS MODE ACTIVE")
    print(f"  Power windows: {POWER_WINDOWS} (EST)")
    print(f"  Logbook: {LOGBOOK_PATH}")
    print(f"{'='*55}\n")

    while True:
        now = datetime.now()

        # Check if we're inside a power window (within first 2 mins of the hour)
        if now.hour in POWER_WINDOWS and now.minute < 2:
            print(f"\n⏰ POWER WINDOW: {now.strftime('%H:%M')} — launching scan...")
            run_one_batch()

        # Sleep until next window
        wait_secs, next_window = get_seconds_until_next_window()
        wait_hrs = wait_secs / 3600
        print(f"\n💤 Sleeping {wait_hrs:.2f}h until next window at {next_window}...")
        write_to_logbook(f"\n[{now.strftime('%H:%M')}] Sleeping {wait_hrs:.2f}h until {next_window}.\n")
        time.sleep(wait_secs)

# ─────────────────────────────────────────────────────────────
# ENTRY POINT — choose your mode
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"

    if mode == "once":
        # Run a single scan right now and exit — useful for testing
        run_one_batch()
    else:
        # Full autonomous mode — runs all day at your power windows
        start_autonomous_session()

print(f"\n{'='*55}")
print("  ⚠️  This is a research tool, not financial advice.")
print(f"{'='*55}")