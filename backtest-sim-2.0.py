"""
backtest_simulator.py
─────────────────────────────────────────────────────────────
Threshold sweep simulator — answers the question:
"What ViT + Judge threshold combo maximises trades AND win rate?"

Uses your existing CSVs — no live data needed.
Runs your current Judge model against every historical window.

OUTPUT:
  - threshold_sweep_results.csv   — full grid, every combo
  - threshold_sweep_heatmap.png   — win rate heatmap
  - threshold_sweep_trades.png    — trade count heatmap
  - best_combos.txt               — top 10 combos ranked by F1
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────
# 1. CONFIG — update paths to match your machine
# ─────────────────────────────────────────────────────────────
BASE_DIR        = r"C:\Users\Kylek\Downloads\Kieron Stuff\Trader-2026"
SCORED_CSV      = os.path.join(BASE_DIR, "fft_judge_scored.csv")
JUDGE_MODEL     = os.path.join(BASE_DIR, "MODELS", "judge_model.json")
OUTPUT_DIR      = os.path.join(BASE_DIR, "backtest_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sweep ranges — edit if you want finer/coarser steps
VIT_THRESHOLDS   = np.arange(0.50, 0.91, 0.05).round(2)   # 0.50 → 0.90
JUDGE_THRESHOLDS = np.arange(0.05, 0.51, 0.05).round(2)   # 0.05 → 0.50

# Judge was trained on these 9 features — confirmed from model's own feature_names
FEATURE_COLS = [
    'vit_prob', 'lstm_prob',
    'spectral_entropy', 'vp_coherence', 'fft_momentum',
    'lock_in_ratio', 'total_energy',
    'vol_regime', 'slope_regime', 'atr',
    'sector_encoded', 'is_new_ticker',
]

SECTOR_MAP = {
    'GS': 'Financials', 'JPM': 'Financials', 'MS': 'Financials',
    'BAC': 'Financials', 'WFC': 'Financials', 'BLK': 'Financials',
    'SCHW': 'Financials', 'V': 'Financials', 'MA': 'Financials',
    'LLY': 'Healthcare', 'UNH': 'Healthcare', 'JNJ': 'Healthcare',
    'PFE': 'Healthcare', 'ABBV': 'Healthcare', 'MRK': 'Healthcare',
    'CVS': 'Healthcare', 'BMY': 'Healthcare',
    'TSLA': 'Technology', 'NVDA': 'Technology', 'AAPL': 'Technology',
    'AMD': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
    'META': 'Technology', 'AMZN': 'Technology', 'NFLX': 'Technology',
    'INTC': 'Technology', 'SOFI': 'Technology', 'PLTR': 'Technology',
    'COIN': 'Technology', 'UBER': 'Technology', 'PYPL': 'Technology',
    'SNAP': 'Technology', 'SPOT': 'Technology', 'HOOD': 'Technology',
    'RIVN': 'Technology', 'BABA': 'Technology',
}

# Tickers that were in scored_new (is_new_ticker=1)
NEW_TICKERS = set([
    'GS', 'JPM', 'MS', 'BAC', 'WFC', 'BLK', 'SCHW', 'V', 'MA',
    'UNH', 'JNJ', 'MRK', 'ABBV', 'LLY', 'PFE', 'CVS', 'BMY',
])

# ─────────────────────────────────────────────────────────────
# 2. LOAD & MERGE DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  BACKTEST THRESHOLD SWEEP SIMULATOR")
print("=" * 60)

df = pd.read_csv(SCORED_CSV)
print(f"\n  Loaded {len(df):,} total windows")

TRAIN_TICKERS = [
    'TSLA', 'NVDA', 'AAPL', 'AMD',  'MSFT',
    'META', 'AMZN', 'NFLX', 'INTC', 'SOFI',
    'PLTR', 'COIN', 'UBER', 'PYPL', 'GOOGL',
    'RIVN', 'BABA', 'SNAP', 'SPOT', 'HOOD',
]
df['is_new_ticker'] = df['symbol'].apply(lambda x: 0 if x in TRAIN_TICKERS else 1)

# Honest backtest — only unseen tickers
df = df[df['is_new_ticker'] == 1].copy()
print(f"  Filtered to unseen tickers: {len(df):,} rows ({df['symbol'].nunique()} tickers)")

# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
le = LabelEncoder()
df['sector']         = df['symbol'].map(SECTOR_MAP).fillna('Unknown')
df['sector_encoded'] = le.fit_transform(df['sector'])
df['label']          = (df['outcome'] == 'win').astype(int)

# Compute vol and slope from price_snapshot (replicates get_regime_features)
def get_vol_slope(price_str, window_size=20):
    try:
        prices  = np.array([float(x) for x in price_str.split(',')])
        returns = np.diff(np.log(prices[-window_size:]))
        vol     = float(np.std(returns) * np.sqrt(252 * 78))
        y       = prices[-10:]
        x       = np.arange(len(y))
        slope   = float(np.polyfit(x, y, 1)[0])
        return round(vol, 4), round(slope, 4)
    except:
        return None, None

print("  Computing vol + slope from price snapshots...")
vol_slope       = df['price_snapshot'].apply(get_vol_slope)
df['vol_regime']    = vol_slope.apply(lambda x: x[0])
df['slope_regime']  = vol_slope.apply(lambda x: x[1])

print(f"  Label balance: {df['label'].sum():,} wins / {(~df['label'].astype(bool)).sum():,} losses "
      f"({df['label'].mean()*100:.1f}% win rate)\n")

# ─────────────────────────────────────────────────────────────
# 4. LOAD JUDGE & SCORE ALL ROWS ONCE
# ─────────────────────────────────────────────────────────────
print("  Loading Judge model...")
judge = xgb.XGBClassifier()
judge.load_model(JUDGE_MODEL)

X = df[FEATURE_COLS]
df['judge_prob'] = judge.predict_proba(X)[:, 1]
print(f"  Judge scored {len(df):,} rows\n")

# ─────────────────────────────────────────────────────────────
# 5. THRESHOLD SWEEP
# ─────────────────────────────────────────────────────────────
print(f"  Sweeping {len(VIT_THRESHOLDS)} ViT thresholds × "
      f"{len(JUDGE_THRESHOLDS)} Judge thresholds = "
      f"{len(VIT_THRESHOLDS)*len(JUDGE_THRESHOLDS)} combos...\n")

results = []

for vit_t in VIT_THRESHOLDS:
    # Filter by ViT threshold first
    vit_mask = df['vit_prob'] >= vit_t
    vit_pool = df[vit_mask]

    for judge_t in JUDGE_THRESHOLDS:
        # Then filter by Judge threshold
        approved = vit_pool[vit_pool['judge_prob'] >= judge_t]

        n_trades    = len(approved)
        n_wins      = approved['label'].sum()
        win_rate    = n_wins / n_trades if n_trades > 0 else 0
        precision   = win_rate  # same thing in this context

        # Recall = of all real wins in the full dataset, how many did we catch?
        total_wins  = df['label'].sum()
        recall      = n_wins / total_wins if total_wins > 0 else 0

        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)

        # Average regime features of approved trades
        avg_vol   = approved['vol_regime'].mean()   if n_trades > 0 else None
        avg_slope = approved['slope_regime'].mean() if n_trades > 0 else None

        results.append({
            'vit_threshold':   vit_t,
            'judge_threshold': judge_t,
            'trades':          n_trades,
            'wins':            int(n_wins),
            'win_rate':        round(win_rate, 4),
            'recall':          round(recall, 4),
            'f1':              round(f1, 4),
            'avg_vol':         round(avg_vol, 4)   if avg_vol   is not None else None,
            'avg_slope':       round(avg_slope, 4) if avg_slope is not None else None,
        })

results_df = pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────
# 6. BASELINE — current live settings for comparison
# ─────────────────────────────────────────────────────────────
current = results_df[
    (results_df['vit_threshold']   == 0.65) &
    (results_df['judge_threshold'] == 0.10)
].iloc[0] if len(results_df[
    (results_df['vit_threshold']   == 0.65) &
    (results_df['judge_threshold'] == 0.10)
]) > 0 else None

print("=" * 60)
print("  CURRENT LIVE SETTINGS (ViT=0.65, Judge=0.51)")
print("=" * 60)
if current is not None:
    print(f"  Trades   : {int(current['trades']):,}")
    print(f"  Win rate : {current['win_rate']*100:.1f}%")
    print(f"  Recall   : {current['recall']*100:.1f}%")
    print(f"  F1       : {current['f1']:.4f}")

# Also show original 0.82 ViT-only baseline
vit_only = df[df['vit_prob'] >= 0.82]
print(f"\n  ORIGINAL (ViT=0.82, no Judge)")
print(f"  Trades   : {len(vit_only):,}")
print(f"  Win rate : {vit_only['label'].mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────
# 7. TOP 10 COMBOS
# ─────────────────────────────────────────────────────────────
# Rank by F1 but require at least 50 trades (statistical floor)
top = (results_df[results_df['trades'] >= 50]
       .sort_values('f1', ascending=False)
       .head(10))

print(f"\n{'='*60}")
print(f"  TOP 10 THRESHOLD COMBOS (min 50 trades, ranked by F1)")
print(f"{'='*60}")
print(f"  {'ViT':>5} {'Judge':>6} {'Trades':>7} {'WinRate':>8} {'Recall':>7} {'F1':>7}")
print(f"  {'-'*50}")
for _, row in top.iterrows():
    print(f"  {row['vit_threshold']:>5.2f} {row['judge_threshold']:>6.2f} "
          f"{int(row['trades']):>7,} {row['win_rate']*100:>7.1f}% "
          f"{row['recall']*100:>6.1f}% {row['f1']:>7.4f}")

# Save results
results_csv = os.path.join(OUTPUT_DIR, "threshold_sweep_results.csv")
results_df.to_csv(results_csv, index=False)
print(f"\n  💾 Full results: {results_csv}")

# Save top combos text
best_path = os.path.join(OUTPUT_DIR, "best_combos.txt")
with open(best_path, 'w') as f:
    f.write("TOP 10 THRESHOLD COMBOS\n")
    f.write("Ranked by F1 (precision × recall), min 50 trades\n\n")
    f.write(f"{'ViT':>5} {'Judge':>6} {'Trades':>7} {'WinRate':>8} {'Recall':>7} {'F1':>7}\n")
    f.write("-"*50 + "\n")
    for _, row in top.iterrows():
        f.write(f"{row['vit_threshold']:>5.2f} {row['judge_threshold']:>6.2f} "
                f"{int(row['trades']):>7,} {row['win_rate']*100:>7.1f}% "
                f"{row['recall']*100:>6.1f}% {row['f1']:>7.4f}\n")
print(f"  💾 Best combos:  {best_path}")

# ─────────────────────────────────────────────────────────────
# 8. HEATMAPS
# ─────────────────────────────────────────────────────────────
def make_heatmap(metric, title, filename, fmt='.1%', cmap='RdYlGn'):
    pivot = results_df.pivot(
        index='vit_threshold',
        columns='judge_threshold',
        values=metric
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    ax.set_xlabel('Judge Threshold', fontsize=12)
    ax.set_ylabel('ViT Threshold',   fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                txt = f"{val:{fmt[1:]}}" if fmt == '.1%' else f"{int(val):,}"
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=8, color='black')

    # Mark current live settings
    vit_vals   = list(pivot.index)
    judge_vals = list(pivot.columns)
    try:
        yi = vit_vals.index(min(vit_vals, key=lambda x: abs(x - 0.65)))
        xi = judge_vals.index(min(judge_vals, key=lambda x: abs(x - 0.10)))
        ax.add_patch(plt.Rectangle((xi-0.5, yi-0.5), 1, 1,
                     fill=False, edgecolor='blue', lw=3, label='Current settings'))
    except:
        pass

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 {filename}")

make_heatmap('win_rate', 'Win Rate by Threshold Combo',   '1_winrate_heatmap.png',  fmt='.1%')
make_heatmap('trades',   'Trade Count by Threshold Combo','2_trades_heatmap.png',   fmt=',',   cmap='Blues')
make_heatmap('f1',       'F1 Score by Threshold Combo',   '3_f1_heatmap.png',       fmt='.3f')

# ─────────────────────────────────────────────────────────────
# 9. ViT-ALONE CURVE (how much Judge actually adds)
# ─────────────────────────────────────────────────────────────
print(f"\n  ViT ALONE vs ViT+JUDGE at each ViT threshold:")
print(f"  {'ViT':>5} | {'ViT alone trades':>16} {'ViT alone WR':>13} | "
      f"{'+Judge trades':>14} {'+Judge WR':>10}")
print(f"  {'-'*65}")

for vit_t in VIT_THRESHOLDS:
    vit_only_sub = df[df['vit_prob'] >= vit_t]
    n_vit    = len(vit_only_sub)
    wr_vit   = vit_only_sub['label'].mean() if n_vit > 0 else 0

    # Judge at 0.10 (current threshold)
    judged   = vit_only_sub[vit_only_sub['judge_prob'] >= 0.51]
    n_judged = len(judged)
    wr_judged = judged['label'].mean() if n_judged > 0 else 0

    print(f"  {vit_t:>5.2f} | {n_vit:>16,} {wr_vit*100:>12.1f}% | "
          f"{n_judged:>14,} {wr_judged*100:>9.1f}%")

print(f"\n{'='*60}")
print(f"  SIMULATION COMPLETE")
print(f"  Results: {OUTPUT_DIR}")
print(f"{'='*60}")
print(f"\n  HOW TO READ THE RESULTS:")
print(f"  - Blue box on heatmaps = your current settings")
print(f"  - Best combo = highest F1 with >= 50 trades")
print(f"  - If Judge WR >> ViT alone WR → Judge is doing real work")
print(f"  - If similar → ViT threshold is the real bottleneck")