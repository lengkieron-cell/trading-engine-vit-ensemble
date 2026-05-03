import yfinance as yf
import numpy as np
import os
import pandas as pd
from PIL import Image

# ─────────────────────────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────────────────────────
# 1. Dynamically find the project folder
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

# 2. Define training folders relative to the script location
TRAIN_WIN  = os.path.join(BASE_DIR, 'train', 'win')
TRAIN_LOSS = os.path.join(BASE_DIR, 'train', 'loss')

# 3. Ensure the folders exist so the script doesn't crash on first run
os.makedirs(TRAIN_WIN,  exist_ok=True)
os.makedirs(TRAIN_LOSS, exist_ok=True)



# ─────────────────────────────────────────────────────────────
# 2. CONFIG
# ─────────────────────────────────────────────────────────────

# Original ViT tickers (already have images — skip regenerating)
EXISTING_TICKERS = [
    'TSLA', 'NVDA', 'AAPL', 'AMD',  'MSFT',
    'GOOGL','META', 'AMZN', 'NFLX', 'INTC',
    'SOFI', 'PLTR', 'RIVN', 'COIN', 'UBER',
    'BABA', 'PYPL', 'SNAP', 'SPOT', 'HOOD',
]

# NEW tickers to add — financials + healthcare
NEW_TICKERS = [
    # Financials
    'JPM', 'BAC', 'WFC', 'SCHW', 'GS', 'MS', 'BLK',
    # Healthcare
    'UNH', 'JNJ', 'MRK', 'PFE', 'LLY', 'ABBV',
]

# Only collect new tickers — existing images stay untouched
TICKERS = EXISTING_TICKERS + NEW_TICKERS

LOOKBACK       = 100    # must match watchdog + scorer
FORECAST       = 75     # bars — matches watchdog's 75-min audit window
WIN_THRESHOLD  = 0.01   # 1% take-profit barrier
ATR_MULTIPLIER = 1.5    # matches watchdog's get_atr_stop()
ATR_PERIOD     = 14
INTERVAL_5M    = '5m'
INTERVAL_15M   = '15m'
INTERVAL_60M   = '60m'
PERIOD         = '60d'
MAX_PER_TICKER = 500    # cap per class per ticker

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


def build_multitf_gaf(window_5m: np.ndarray,
                      window_15m: np.ndarray,
                      window_60m: np.ndarray) -> np.ndarray | None:
    """
    Multi-timeframe GAF — each channel is a different timeframe.
    Red   = 5m  (micro price action)
    Green = 15m (trend context)
    Blue  = 60m (macro context)
    All resized to LOOKBACK x LOOKBACK so ViT sees aligned context.
    """
    from PIL import Image as PILImage

    def to_gaf_channel(series):
        if len(series) < 2:
            return None
        gaf = gramian_angular_field(series, method='summation')
        # Resize to LOOKBACK x LOOKBACK
        img = PILImage.fromarray(((gaf + 1) / 2 * 255).astype(np.uint8))
        img = img.resize((LOOKBACK, LOOKBACK), PILImage.BILINEAR)
        return np.array(img) / 255.0

    ch_r = to_gaf_channel(window_5m)
    ch_g = to_gaf_channel(window_15m)
    ch_b = to_gaf_channel(window_60m)

    if ch_r is None or ch_g is None or ch_b is None:
        return None

    rgb = np.stack([ch_r, ch_g, ch_b], axis=2)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────
# 4. TRIPLE BARRIER LABELER — matches fft_judge_ready pipeline
# ─────────────────────────────────────────────────────────────
def triple_barrier_label(window, future,
                          atr_period=ATR_PERIOD,
                          atr_mult=ATR_MULTIPLIER,
                          win_thresh=WIN_THRESHOLD):
    """
    Replicates the watchdog's actual trade logic exactly.
      WIN     : price hits +win_thresh% before stop or timeout
      LOSS    : price hits ATR stop-loss before take-profit
      TIMEOUT : neither barrier hit within FORECAST bars

    Timeouts are treated as losses for the ViT (label=loss folder)
    so the ViT learns "no follow-through = don't take it".
    """
    prices      = window.astype(float)
    entry_price = prices[-1]

    if len(prices) < atr_period + 1:
        atr = float(np.std(prices[-20:]) * 0.01)
    else:
        tr  = np.abs(np.diff(prices[-(atr_period + 1):]))
        atr = float(tr.mean())

    stop_price = entry_price - (atr_mult * atr)
    tp_price   = entry_price * (1 + win_thresh)

    label = 'timeout'
    for bar in future.astype(float):
        if bar >= tp_price:
            label = 'win'
            break
        if bar <= stop_price:
            label = 'loss'
            break

    return label

# ─────────────────────────────────────────────────────────────
# 5. MAIN COLLECTION LOOP
# ─────────────────────────────────────────────────────────────
print(f"{'='*60}")
print(f"  GAF COLLECTOR v2 — Triple Barrier + Sector Balanced")
print(f"  New tickers : {NEW_TICKERS}")
print(f"  Lookback    : {LOOKBACK} bars | Forecast: {FORECAST} bars")
print(f"  Labels      : Triple Barrier (win/loss/timeout→loss)")
print(f"  Cap         : {MAX_PER_TICKER} wins + {MAX_PER_TICKER} losses per ticker")
print(f"{'='*60}\n")

# Count existing images so we don't overwrite
existing_wins   = len(os.listdir(TRAIN_WIN))
existing_losses = len(os.listdir(TRAIN_LOSS))
print(f"  Existing images: {existing_wins} wins, {existing_losses} losses\n")

total_wins   = 0
total_losses = 0

for symbol in TICKERS:
    print(f"Processing {symbol}...", end=" ", flush=True)

    try:
        data_5m  = yf.download(symbol, period=PERIOD, interval=INTERVAL_5M,
                               progress=False, auto_adjust=True)
        data_15m = yf.download(symbol, period=PERIOD, interval=INTERVAL_15M,
                               progress=False, auto_adjust=True)
        data_60m = yf.download(symbol, period=PERIOD, interval=INTERVAL_60M,
                               progress=False, auto_adjust=True)

        # Align all timeframes by timestamp
        data_5m.index  = pd.to_datetime(data_5m.index).tz_localize(None)
        data_15m.index = pd.to_datetime(data_15m.index).tz_localize(None)
        data_60m.index = pd.to_datetime(data_60m.index).tz_localize(None)

        # Build timestamp lookup for 15m and 60m
        ts_5m  = data_5m.index.tolist()
        ts_15m = data_15m.index.tolist()
        ts_60m = data_60m.index.tolist()

        prices_5m  = data_5m['Close'].values.flatten()
        prices_15m = data_15m['Close'].values.flatten()
        prices_60m = data_60m['Close'].values.flatten()
        volumes    = data_5m['Volume'].values.flatten()

        if len(prices_5m) < LOOKBACK + FORECAST + 10:
            print(f"⚠️  Not enough 5m data, skipping.")
            continue

        wins     = 0
        losses   = 0
        timeouts = 0

        for i in range(len(prices_5m) - LOOKBACK - FORECAST):

            if wins >= MAX_PER_TICKER and losses >= MAX_PER_TICKER:
                break

            window     = prices_5m[i  : i + LOOKBACK]
            future     = prices_5m[i + LOOKBACK : i + LOOKBACK + FORECAST]
            vol_window = volumes[i : i + LOOKBACK]

            # Skip NaNs
            if np.isnan(window).any() or np.isnan(future).any() or np.isnan(vol_window).any():
                continue

            # Triple barrier label
            label = triple_barrier_label(window, future)

            # Timeouts → loss folder (ViT learns "no momentum = don't trade")
            gaf_label = 'win' if label == 'win' else 'loss'

            if gaf_label == 'win'  and wins   >= MAX_PER_TICKER:
                continue
            if gaf_label == 'loss' and losses >= MAX_PER_TICKER:
                continue

            # Build GAF image
            # Align 15m and 60m windows to same time point
            # Each 5m bar = 1/3 of a 15m bar = 1/12 of a 60m bar
            # Get the actual timestamp of the current 5m bar
            entry_ts = ts_5m[i + LOOKBACK - 1]

            # Find the closest 15m and 60m bar at or before this timestamp
            idx_15m = next((j for j in range(len(ts_15m)-1, -1, -1)
                           if ts_15m[j] <= entry_ts), None)
            idx_60m = next((j for j in range(len(ts_60m)-1, -1, -1)
                           if ts_60m[j] <= entry_ts), None)

            if idx_15m is None or idx_60m is None:
                continue
            if idx_15m < LOOKBACK or idx_60m < LOOKBACK:
                continue

            idx_15m = idx_15m - LOOKBACK + 1
            idx_60m = idx_60m - LOOKBACK + 1

            window_15m = prices_15m[idx_15m : idx_15m + LOOKBACK]
            window_60m = prices_60m[idx_60m : idx_60m + LOOKBACK]

            if len(window_15m) < LOOKBACK or len(window_60m) < LOOKBACK:
                continue

            img_arr = build_multitf_gaf(window, window_15m, window_60m)
            if img_arr is None:
                continue  # flat volume — skip

            # Save
            save_dir = TRAIN_WIN if gaf_label == 'win' else TRAIN_LOSS
            Image.fromarray(img_arr).save(
                os.path.join(save_dir, f"{symbol}_{i}.png")
            )

            if label == 'win':
                wins += 1
            elif label == 'loss':
                losses += 1
            else:
                timeouts += 1
                losses += 1  # counted in loss folder

        total_wins   += wins
        total_losses += losses
        print(f"✅  {wins}W / {losses}L / {timeouts}T")

    except Exception as e:
        import traceback
        print(f"❌  Error:")
        traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# 6. SUMMARY
# ─────────────────────────────────────────────────────────────
final_wins   = len(os.listdir(TRAIN_WIN))
final_losses = len(os.listdir(TRAIN_LOSS))

print(f"\n{'='*60}")
print(f"  COLLECTION COMPLETE")
print(f"  New wins added   : {total_wins:,}")
print(f"  New losses added : {total_losses:,}")
print(f"  Total wins now   : {final_wins:,}  (was {existing_wins:,})")
print(f"  Total losses now : {final_losses:,}  (was {existing_losses:,})")
print(f"  Imbalance ratio  : {final_losses / max(final_wins, 1):.2f}x")
print(f"{'='*60}")
print(f"\n  NEXT STEP:")
print(f"  Upload train/win + train/loss to Kaggle as new dataset version")
print(f"  Then retrain ViT on the full sector-balanced dataset")
