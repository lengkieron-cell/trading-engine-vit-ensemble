import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import os
import json

# ─────────────────────────────────────────────────────────────
# KAGGLE CONFIG — Adjusted for fft_judge_ready.csv
# ─────────────────────────────────────────────────────────────
# 1. INPUT: Where your uploaded CSV lives
# Replace 'your-dataset-name' with the actual name of the dataset you uploaded to Kaggle
# 1. Get the current directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define a clear folder for outputs
# We'll create a 'results' folder to keep things tidy
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 3. Update paths to live inside the 'data', 'models', and 'results' folders
INPUT_CSV   = os.path.join(BASE_DIR, "data", "fft_judge_ready.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "lstm_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "models", "lstm_scaler.json")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "lstm_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"  Training on Kaggle using: {INPUT_CSV}")
print(f"  Outputs will be saved to: /kaggle/working")

# Sequence settings
SEQ_LEN       = 10    # how many consecutive bars the LSTM looks back
BATCH_SIZE    = 64
EPOCHS        = 50
LR            = 1e-3
HIDDEN_SIZE   = 32
NUM_LAYERS    = 1
DROPOUT       = 0.5

# With this — remove model1_prob, add delta features flag:
FEATURE_COLS = [
    'spectral_entropy',  'spectral_entropy_d1',  'spectral_entropy_d3',
    'vp_coherence',      'vp_coherence_d1',      'vp_coherence_d3',
    'fft_momentum',      'fft_momentum_d1',      'fft_momentum_d3',
    'lock_in_ratio',     'lock_in_ratio_d1',     'lock_in_ratio_d3',
    'total_energy',      'total_energy_d1',      'total_energy_d3',
    'volatility',        'volatility_d1',        'volatility_d3',
    'trend',             'trend_d1',             'trend_d3',
    'volume_spike',      'volume_spike_d1',      'volume_spike_d3',
]
USE_RAW_PRICES = True   # feed last 20 bars of raw price to LSTM
ADD_DELTAS     = True   # add 1-bar and 3-bar change in each feature

# ─────────────────────────────────────────────────────────────
# 2. LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────
print(f"{'='*55}")
print(f"  LSTM TRAINING PIPELINE  (v6 — Triple Barrier labels)")
print(f"  Sequence length : {SEQ_LEN} consecutive bars")
print(f"  Features        : {len(FEATURE_COLS)}")
print(f"{'='*55}\n")

df_all = pd.read_csv(INPUT_CSV)

# Triple Barrier labels: win=1, loss/timeout=0 (already in meta_label column)
# outcome column preserved for backward compat with evaluate() below
df_all['outcome'] = df_all['outcome'].str.lower()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base_cols = [
        'spectral_entropy', 'vp_coherence',
        'fft_momentum', 'lock_in_ratio', 'total_energy',
        'volatility', 'trend', 'volume_spike',   # ← new context features
    ]
    for col in base_cols:
        df[f'{col}_d1'] = df.groupby('symbol')[col].diff(1)
        df[f'{col}_d3'] = df.groupby('symbol')[col].diff(3)
    df = df.dropna().reset_index(drop=True)
    return df
df_all = engineer_features(df_all)

# Train/test split by ticker sector — original Tech tickers train, Fin+Health test
# This replicates the is_new_ticker split from the old pipeline
all_symbols = df_all['symbol'].unique()
rng = np.random.default_rng(42)
test_symbols = set(rng.choice(all_symbols, size=int(len(all_symbols) * 0.2), replace=False))

df_orig = df_all[~df_all['symbol'].isin(test_symbols)].copy().reset_index(drop=True)
df_new  = df_all[ df_all['symbol'].isin(test_symbols)].copy().reset_index(drop=True)
df_orig['is_new_ticker'] = 0
df_new['is_new_ticker']  = 1

print(f"  Loaded {len(df_all):,} total rows from fft_judge_ready.csv")
print(f"  Train split : {len(df_orig):,} rows  ({df_orig['symbol'].nunique()} tickers)")
print(f"  Test split  : {len(df_new):,} rows  ({df_new['symbol'].nunique()} tickers)")
wins = int((df_orig['meta_label'] == 1).sum())
total = len(df_orig)
print(f"  Label balance : {wins:,} wins / {total-wins:,} neg  ({wins/total*100:.1f}% win rate)\n")

# ─────────────────────────────────────────────────────────────
# 3. SEQUENCE BUILDER
# ─────────────────────────────────────────────────────────────
def build_sequences(df: pd.DataFrame, seq_len: int, feature_cols: list):
    sequences = []
    labels    = []
    meta      = []

    # Calculate total features: base features (15) + price tail (5)
    total_features = len(feature_cols) + (5 if USE_RAW_PRICES else 0)

    for symbol, group in df.groupby('symbol'):
        group = group.sort_values('bar_index').reset_index(drop=True)
        feats = group[feature_cols].values
        labs  = group['meta_label'].astype(int).values
        bars  = group['bar_index'].values
        wids  = group['window_id'].values

        for i in range(len(group) - seq_len):
            slice_bars = bars[i : i + seq_len + 1]
            if not np.all(np.diff(slice_bars) == 1):
                continue  

            # 1. Create a zero-padded block of the correct FINAL shape (10, 20)
            seq = np.zeros((seq_len, total_features), dtype=np.float32)
            
            # 2. Fill in the 15 base features for all 10 bars
            seq[:, :len(feature_cols)] = feats[i : i + seq_len]
            
            label = labs[i + seq_len]             
            wid   = wids[i + seq_len]

            if np.isnan(seq).any():
                continue
            
            if USE_RAW_PRICES:
                # Parse last 20 prices from snapshot
                snapshot_str = group['price_snapshot'].iloc[i + seq_len - 1]
                prices_arr   = np.array([float(x) for x in str(snapshot_str).split(',')])
                
                # Normalise to pct change
                price_tail   = prices_arr[-20:]
                price_tail   = (price_tail - price_tail[0]) / (price_tail[0] + 1e-8)
                
                # 3. Fill the LAST 5 columns of the LAST row only
                seq[-1, len(feature_cols):] = price_tail[-5:]

            sequences.append(seq)
            labels.append(label)
            meta.append({
                'window_id': wid,
                'symbol':    symbol,
                'outcome':   'win' if label == 1 else 'loss',
            })

    return (np.array(sequences, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            meta)

print("  Building sequences from original tickers (train)...")
X_orig, y_orig, meta_orig = build_sequences(df_orig, SEQ_LEN, FEATURE_COLS)
print(f"  → {len(X_orig):,} sequences\n")

print("  Building sequences from new tickers (test)...")
X_new, y_new, meta_new = build_sequences(df_new, SEQ_LEN, FEATURE_COLS)
print(f"  → {len(X_new):,} sequences\n")

wins   = int(y_orig.sum())
losses = len(y_orig) - wins
print(f"  Train class balance:")
print(f"    Wins   : {wins:,}  ({wins/len(y_orig)*100:.1f}%)")
print(f"    Losses : {losses:,}  ({losses/len(y_orig)*100:.1f}%)\n")

# ─────────────────────────────────────────────────────────────
# 4. NORMALISE FEATURES
# ─────────────────────────────────────────────────────────────
# Fit scaler on training data only — flatten to 2D, scale, reshape back
n_train, sl, nf = X_orig.shape
scaler_means = X_orig.reshape(-1, nf).mean(axis=0)
scaler_stds  = X_orig.reshape(-1, nf).std(axis=0) + 1e-8

X_orig_scaled = ((X_orig.reshape(-1, nf) - scaler_means) / scaler_stds).reshape(n_train, sl, nf)
X_new_scaled  = ((X_new.reshape(-1, nf)  - scaler_means) / scaler_stds).reshape(len(X_new), sl, nf)

# Save scaler params for watchdog inference
scaler_params = {
    'means': scaler_means.tolist(),
    'stds':  scaler_stds.tolist(),
    'features': FEATURE_COLS,
    'seq_len':  SEQ_LEN,
}
with open(SCALER_PATH, 'w') as f:
    json.dump(scaler_params, f, indent=2)
print(f"  ✅ Scaler saved: {SCALER_PATH}\n")

# Train/val split on original tickers (80/20)
split = int(0.8 * len(X_orig_scaled))
X_train, X_val = X_orig_scaled[:split], X_orig_scaled[split:]
y_train, y_val = y_orig[:split],        y_orig[split:]

print(f"  Split:")
print(f"    Train : {len(X_train):,}")
print(f"    Val   : {len(X_val):,}")
print(f"    Test  : {len(X_new_scaled):,} (unseen tickers)\n")

# ─────────────────────────────────────────────────────────────
# 5. DATASET & DATALOADER
# ─────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SequenceDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SequenceDataset(X_val, y_val),
                          batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────────────────────
# 6. LSTM MODEL
# ─────────────────────────────────────────────────────────────
class MasterLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0,
            batch_first = True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step   = lstm_out[:, -1, :]  # take output at final timestep
        return self.head(last_step).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Training on: {device}\n")

lstm_model = MasterLSTM(
    input_size  = len(FEATURE_COLS) + 5,  # +5 for raw price tail
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    dropout     = DROPOUT,
).to(device)

# Class weight to handle imbalance
pos_weight = torch.tensor([losses / max(wins, 1)], dtype=torch.float32).to(device)
criterion  = nn.BCELoss()
optimizer  = torch.optim.Adam(lstm_model.parameters(), lr=LR, weight_decay=1e-4)
# NEW CODE (Fixed)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5)

# ─────────────────────────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────────────────────────


def get_lstm_probs(model, X, batch_size=256):
    model.eval()
    all_probs = []
    dataset   = DataLoader(
        SequenceDataset(X, np.zeros(len(X))),
        batch_size=batch_size, shuffle=False
    )
    with torch.no_grad():
        for Xb, _ in dataset:
            out = model(Xb.to(device))
            all_probs.extend(out.cpu().numpy())
    return np.array(all_probs)


print(f"{'='*55}")
print(f"  TRAINING LSTM ({EPOCHS} epochs)")
print(f"{'='*55}")

best_val_loss = float('inf')
best_weights  = None
no_improve = 0
for epoch in range(1, EPOCHS + 1):

    # Train
    lstm_model.train()
    train_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        # Use raw logits for BCEWithLogitsLoss
        logits = lstm_model.lstm(Xb)[0][:, -1, :]
        logits = lstm_model.head[1](lstm_model.head[0](logits))  # dropout + linear
        # Simpler: just use model output with sigmoid removed for loss
        out    = lstm_model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    # Validate
    lstm_model.eval()
    val_loss = 0
    val_probs_list = []
    val_labels_list = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out    = lstm_model(Xb)
            loss = criterion(out, yb)
            val_loss += loss.item()
            val_probs_list.extend(out.cpu().numpy())
            val_labels_list.extend(yb.cpu().numpy())

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    scheduler.step(val_loss)

    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights  = {k: v.clone() for k, v in lstm_model.state_dict().items()}
        no_improve    = 0
    else:
        no_improve += 1
        if no_improve >= 20:
            print(f"  Early stopping at epoch {epoch}")
            break

    if epoch % 5 == 0 or epoch == 1:
        val_preds = (np.array(val_probs_list) >= 0.5).astype(int)
        val_f1    = f1_score(val_labels_list, val_preds, zero_division=0)
        print(f"  Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.3f}")

# Restore best weights
lstm_model.load_state_dict(best_weights)
torch.save(lstm_model.state_dict(), MODEL_PATH)
print(f"\n  ✅ Best LSTM saved: {MODEL_PATH}\n")

# ── Save per-window LSTM probabilities IMMEDIATELY after training ──
all_probs_orig = get_lstm_probs(lstm_model, X_orig_scaled)
prob_rows = [{'window_id': m['window_id'], 'lstm_prob': float(p)}
             for m, p in zip(meta_orig, all_probs_orig)]

all_probs_new = get_lstm_probs(lstm_model, X_new_scaled)
prob_rows += [{'window_id': m['window_id'], 'lstm_prob': float(p)}
              for m, p in zip(meta_new, all_probs_new)]

lstm_probs_out = "/kaggle/working/lstm_window_probs_v2.csv"
pd.DataFrame(prob_rows).to_csv(lstm_probs_out, index=False)
print(f"  💾 LSTM window probs saved: {lstm_probs_out}\n")

# ─────────────────────────────────────────────────────────────
# 8. EVALUATE — THREE-WAY COMPARISON
# ─────────────────────────────────────────────────────────────


def evaluate(name, probs, labels, threshold=0.5):
    preds  = (probs >= threshold).astype(int)
    labels = np.array(labels).astype(int)
    prec   = precision_score(labels, preds, zero_division=0)
    rec    = recall_score(labels, preds, zero_division=0)
    f1     = f1_score(labels, preds, zero_division=0)
    auc    = roc_auc_score(labels, probs)
    trades = preds.sum()
    return {'name': name, 'precision': prec, 'recall': rec,
            'f1': f1, 'auc': auc, 'trades': trades}

# Get LSTM probabilities on test set (unseen tickers)
lstm_probs = get_lstm_probs(lstm_model, X_new_scaled)

# Rebuild test meta for model1_prob lookup
# Rebuild test meta for model1_prob lookup
meta_df     = pd.DataFrame(meta_new)
test_labels = np.array([m['outcome'] == 'win' for m in meta_new]).astype(int)
vit_available = False

# Load judge model for three-way comparison
try:
    import xgboost as xgb
    judge_model = xgb.XGBClassifier()
    judge_model.load_model("/kaggle/working/judge_model.json")

    from sklearn.preprocessing import LabelEncoder
    SECTOR_MAP = {
        'GS':'Financials','JPM':'Financials','MS':'Financials',
        'BAC':'Financials','WFC':'Financials','BLK':'Financials','SCHW':'Financials',
        'LLY':'Healthcare','UNH':'Healthcare','JNJ':'Healthcare',
        'PFE':'Healthcare','ABBV':'Healthcare','MRK':'Healthcare',
        'TSLA':'Technology','NVDA':'Technology','AAPL':'Technology',
        'AMD':'Technology','MSFT':'Technology','GOOGL':'Technology',
        'META':'Technology','AMZN':'Technology','NFLX':'Technology',
    }
    le = LabelEncoder()
    test_df = test_df.copy()
    test_df['sector']         = test_df['symbol'].map(SECTOR_MAP).fillna('Unknown')
    test_df['sector_encoded'] = le.fit_transform(test_df['sector'])
    test_df['lstm_prob']      = lstm_probs

    judge_feature_cols = [
        'model1_prob', 'lstm_prob', 'spectral_entropy', 'vp_coherence',
        'fft_momentum', 'lock_in_ratio', 'total_energy',
        'sector_encoded', 'is_new_ticker',
        # New features — only present after scorer runs
        # If missing, judge eval section will be skipped gracefully
    ]
    # Filter to only cols that exist in test_df
    judge_feature_cols = [c for c in judge_feature_cols if c in test_df.columns]
    judge_probs = judge_model.predict_proba(
        test_df[judge_feature_cols]
    )[:, 1]
    
    judge_available = True
except Exception as e:
    print(f"  ⚠️  Judge model not loaded: {e}")
    judge_available = False

# Combined score: average of Judge + LSTM
if judge_available:
    combo_probs = (judge_probs + lstm_probs) / 2
else:
    combo_probs = lstm_probs

print(f"\n{'='*65}")
print(f"  THREE-WAY BACKTEST — Unseen Tickers")
print(f"{'='*65}")
print(f"  {'Model':<28} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'Trades':>7}")
print(f"  {'-'*60}")

results = [
    evaluate("LSTM Alone (0.50)",     lstm_probs,  test_labels, 0.50),
    evaluate("LSTM Alone (0.60)",     lstm_probs,  test_labels, 0.60),
]
if vit_available:
    results.insert(0, evaluate("ViT Alone (0.82)", vit_probs, test_labels, 0.82))

if judge_available:
    results += [
        evaluate("Judge (0.82)",           judge_probs, test_labels, 0.82),
        evaluate("Judge+LSTM Combo (0.55)", combo_probs, test_labels, 0.55),
        evaluate("Judge+LSTM Combo (0.65)", combo_probs, test_labels, 0.65),
    ]

for r in results:
    print(f"  {r['name']:<28} {r['precision']:>6.3f} {r['recall']:>6.3f} "
          f"{r['f1']:>6.3f} {r['auc']:>6.3f} {r['trades']:>7}")

print(f"{'='*65}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "backtest_results.csv"), index=False)
print(f"\n  💾 Backtest results saved: lstm_results/backtest_results.csv")

# ─────────────────────────────────────────────────────────────
# 9. SAVE CONFIG FOR WATCHDOG INTEGRATION
# ─────────────────────────────────────────────────────────────
config = {
    'lstm_model_path':  MODEL_PATH,
    'scaler_path':      SCALER_PATH,
    'seq_len':          SEQ_LEN,
    'feature_cols':     FEATURE_COLS,
    'hidden_size':      HIDDEN_SIZE,
    'num_layers':       NUM_LAYERS,
    'trained_on':       pd.Timestamp.now().strftime('%Y-%m-%d'),
    'train_sequences':  len(X_train),
    'test_sequences':   len(X_new_scaled),
}
config_path = "/kaggle/working/lstm_config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"  💾 LSTM config saved: {config_path}")
print(f"\n{'='*55}")
print(f"  TRAINING COMPLETE")
print(f"  INTERPRETATION:")
print(f"  Judge+LSTM Prec > Judge alone → LSTM adds value ✅")
print(f"  Judge+LSTM Prec < Judge alone → LSTM not helping ⚠️")
print(f"{'='*55}")

