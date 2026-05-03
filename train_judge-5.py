import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ─────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\Kylek\Downloads\Kieron Stuff\Trader-2026"
INPUT_CSV  = os.path.join(BASE_DIR, "fft_judge_scored.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "judge_results")
MODEL_PATH = os.path.join(BASE_DIR, "MODELS", "judge_model.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Features the Judge will use
# model1_prob = ViT confidence
# FFT features = frequency domain signal
FEATURE_COLS = [
    'vit_prob',
    'lstm_prob',
    'spectral_entropy',
    'vp_coherence',
    'fft_momentum',
    'lock_in_ratio',
    'total_energy',
    'vol_regime',
    'slope_regime',
    'atr',
    'sector_encoded',
    'is_new_ticker',
]



# ─────────────────────────────────────────────────────────────
# 2. LOAD & MERGE DATA
# ─────────────────────────────────────────────────────────────
print(f"{'='*55}")
print(f"  JUDGE TRAINING PIPELINE")
print(f"{'='*55}\n")

df = pd.read_csv(INPUT_CSV)
print(f"  Loaded {len(df):,} rows from fft_judge_scored.csv")
print(f"  Symbols : {df['symbol'].nunique()}")
print(f"  Columns : {list(df.columns)}\n")

# Tag train vs test tickers
TRAIN_TICKERS = [
    'TSLA', 'NVDA', 'AAPL', 'AMD', 'MSFT',
    'META', 'AMZN', 'NFLX', 'INTC', 'SOFI',
    'PLTR', 'COIN', 'UBER', 'PYPL', 'GOOGL',
    'RIVN', 'BABA', 'SNAP', 'SPOT', 'HOOD',
]
df['is_new_ticker'] = df['symbol'].apply(lambda x: 0 if x in TRAIN_TICKERS else 1)
print(f"  Train tickers : {(df['is_new_ticker']==0).sum():,} rows")
print(f"  Test tickers  : {(df['is_new_ticker']==1).sum():,} rows\n")

# ── Encode sector ──
SECTOR_MAP = {
    'GS': 'Financials', 'JPM': 'Financials', 'MS': 'Financials',
    'BAC': 'Financials', 'WFC': 'Financials', 'BLK': 'Financials',
    'SCHW': 'Financials', 'LLY': 'Healthcare', 'UNH': 'Healthcare',
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
    'MRK': 'Healthcare', 'TSLA': 'Technology', 'NVDA': 'Technology',
    'AAPL': 'Technology', 'AMD': 'Technology', 'MSFT': 'Technology',
    'GOOGL': 'Technology', 'META': 'Technology', 'AMZN': 'Technology',
    'NFLX': 'Technology', 'INTC': 'Technology', 'SOFI': 'Technology',
    'PLTR': 'Technology', 'COIN': 'Technology', 'UBER': 'Technology',
    'PYPL': 'Technology', 'SNAP': 'Technology', 'SPOT': 'Technology',
    'HOOD': 'Technology', 'RIVN': 'Technology', 'BABA': 'Technology',
}
le = LabelEncoder()
df['sector'] = df['symbol'].map(SECTOR_MAP).fillna('Unknown')
df['sector_encoded'] = le.fit_transform(df['sector'])

# ─────────────────────────────────────────────────────────────
# 3. PREPARE LABELS
# ─────────────────────────────────────────────────────────────
df['label'] = (df['outcome'] == 'win').astype(int)  # win=1, loss=0

wins   = df['label'].sum()
losses = len(df) - wins
print(f"  Class balance:")
print(f"    Wins   : {wins:,}  ({wins/len(df)*100:.1f}%)")
print(f"    Losses : {losses:,}  ({losses/len(df)*100:.1f}%)")
print(f"    Ratio  : {losses/wins:.2f}x\n")

# ─────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
# Split by ticker to prevent data leakage — test on held-out tickers
# Use new tickers as test set (they're unseen, honest evaluation)
train_df = df[df['is_new_ticker'] == 0].copy()
test_df  = df[df['is_new_ticker'] == 1].copy()

# Also take 20% of original tickers for validation
train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=42,
    stratify=train_df['label']
)

X_train = train_df[FEATURE_COLS]
y_train = train_df['label']
X_val   = val_df[FEATURE_COLS]
y_val   = val_df['label']
X_test  = test_df[FEATURE_COLS]
y_test  = test_df['label']

print(f"  Split:")
print(f"    Train : {len(X_train):,} rows (original tickers, 80%)")
print(f"    Val   : {len(X_val):,} rows (original tickers, 20%)")
print(f"    Test  : {len(X_test):,} rows (unseen tickers — honest eval)\n")

# ─────────────────────────────────────────────────────────────
# 5. TRAIN XGBOOST JUDGE
# ─────────────────────────────────────────────────────────────
scale_pos_weight = losses / wins  # handles class imbalance

judge = xgb.XGBClassifier(
    n_estimators          = 500,   # fewer needed for 4 clean features
    max_depth             = 3,     # shallower — forces generalisation
    learning_rate         = 0.01,  # slower, more stable
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    scale_pos_weight      = scale_pos_weight,
    eval_metric           = 'logloss',
    early_stopping_rounds = 30,
    random_state          = 42,
    device                = 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
)
print(f"  Training XGBoost Judge...")
judge.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

# Save model
judge.save_model(MODEL_PATH)
print(f"\n  ✅ Judge model saved: {MODEL_PATH}\n")

# ─────────────────────────────────────────────────────────────
# 6. FIND OPTIMAL THRESHOLD (precision cliff for Judge)
# ─────────────────────────────────────────────────────────────
val_probs = judge.predict_proba(X_val)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)

# Find threshold where precision >= 0.65 with most recall
best_thresh = 0.5
best_f1     = 0
for p, r, t in zip(precisions, recalls, thresholds):
    if p >= 0.55:
        f1 = 2 * p * r / (p + r + 1e-8)
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = t

print(f"  Optimal Judge threshold : {best_thresh:.3f}")
print(f"  At this threshold F1    : {best_f1:.3f}\n")

# ─────────────────────────────────────────────────────────────
# 7. EVALUATE — THREE COMPARISONS
# ─────────────────────────────────────────────────────────────
def evaluate(name, probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    acc   = np.mean(preds == labels) * 100
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    f1    = f1_score(labels, preds, zero_division=0)
    auc   = roc_auc_score(labels, probs)
    trades = preds.sum()
    return {
        'name': name, 'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1': f1, 'auc': auc,
        'trades': trades, 'threshold': threshold
    }

# ViT alone on test set (using 0.82 threshold)
vit_alone   = evaluate("ViT Alone (0.82)",
                        test_df['vit_prob'].values, y_test.values,
                        threshold=0.82)

# Judge on test set (using optimal threshold)
judge_probs = judge.predict_proba(X_test)[:, 1]
judge_eval  = evaluate(f"Judge ({best_thresh:.2f})",
                        judge_probs, y_test.values,
                        threshold=best_thresh)

# Judge on test set at 0.82 for direct comparison
judge_82    = evaluate("Judge (0.82)",
                        judge_probs, y_test.values,
                        threshold=0.82)

results = [vit_alone, judge_eval, judge_82]

print(f"{'='*65}")
print(f"  RESULTS ON UNSEEN TICKERS (honest test)")
print(f"{'='*65}")
print(f"  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'Trades':>7}")
print(f"  {'-'*60}")
for r in results:
    print(f"  {r['name']:<22} {r['accuracy']:>5.1f}% {r['precision']:>6.3f} "
          f"{r['recall']:>6.3f} {r['f1']:>6.3f} {r['auc']:>6.3f} {r['trades']:>7}")
print(f"{'='*65}\n")

# ─────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
importance = dict(zip(FEATURE_COLS, judge.feature_importances_))
importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

print(f"  Feature Importance (what the Judge relies on):")
for feat, imp in importance.items():
    bar = '█' * int(imp * 50)
    print(f"    {feat:<20} {imp:.4f}  {bar}")

# ─────────────────────────────────────────────────────────────
# 9. SECTOR BREAKDOWN
# ─────────────────────────────────────────────────────────────


test_df = test_df.copy()
test_df['judge_prob'] = judge_probs
test_df['judge_pred'] = (judge_probs >= best_thresh).astype(int)
test_df['sector']     = test_df['symbol'].map(SECTOR_MAP)

print(f"\n  Sector Breakdown (Judge at {best_thresh:.2f}):")
print(f"  {'Sector':<14} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Trades':>7}")
print(f"  {'-'*42}")
for sector in test_df['sector'].dropna().unique():
    s = test_df[test_df['sector'] == sector]
    sp = precision_score(s['label'], s['judge_pred'], zero_division=0)
    sr = recall_score(s['label'], s['judge_pred'], zero_division=0)
    sf = f1_score(s['label'], s['judge_pred'], zero_division=0)
    st = s['judge_pred'].sum()
    print(f"  {sector:<14} {sp:>6.3f} {sr:>6.3f} {sf:>6.3f} {st:>7}")

# ─────────────────────────────────────────────────────────────
# 10. PLOTS
# ─────────────────────────────────────────────────────────────

# Confusion matrix — Judge vs ViT side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, probs, thresh) in zip(axes, [
    ("ViT Alone (0.82)", test_df['vit_prob'].values, 0.82),
    (f"Judge ({best_thresh:.2f})", judge_probs, best_thresh),
]):
    preds = (probs >= thresh).astype(int)
    cm    = confusion_matrix(y_test.values, preds)
    disp  = ConfusionMatrixDisplay(cm, display_labels=['Loss', 'Win'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name)

plt.suptitle('Judge vs ViT Alone — Unseen Tickers', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1_confusion_comparison.png'))
plt.close()
print(f"\n  💾 Saved: 1_confusion_comparison.png")

# Precision-Recall curve
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], label='Precision', color='green')
plt.plot(thresholds, recalls[:-1],    label='Recall',    color='blue')
plt.axvline(x=best_thresh, color='red', linestyle='--',
            label=f'Optimal threshold ({best_thresh:.2f})')
plt.axvline(x=0.82, color='orange', linestyle='--', label='ViT threshold (0.82)')
plt.title('Judge — Precision vs Recall by Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2_precision_recall_curve.png'))
plt.close()
print(f"  💾 Saved: 2_precision_recall_curve.png")

# Feature importance bar chart
plt.figure(figsize=(8, 5))
plt.barh(list(importance.keys()), list(importance.values()), color='steelblue')
plt.title('Judge Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '3_feature_importance.png'))
plt.close()
print(f"  💾 Saved: 3_feature_importance.png")

# Win prob distribution — Judge vs ViT
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, probs) in zip(axes, [
    ("ViT Alone", test_df['vit_prob'].values),
    ("Judge",     judge_probs),
]):
    ax.hist(probs[y_test.values == 0], bins=40, alpha=0.6,
            label='True Loss', color='red')
    ax.hist(probs[y_test.values == 1], bins=40, alpha=0.6,
            label='True Win',  color='green')
    ax.set_title(f'{name} — Probability Distribution')
    ax.set_xlabel('Win Probability')
    ax.set_ylabel('Count')
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4_prob_distributions.png'))
plt.close()
print(f"  💾 Saved: 4_prob_distributions.png")

# ─────────────────────────────────────────────────────────────
# 11. SAVE THRESHOLD TO CONFIG
# ─────────────────────────────────────────────────────────────
config = {
    'judge_threshold':  round(float(best_thresh), 4),
    'judge_model_path': MODEL_PATH,
    'trained_on':       pd.Timestamp.now().strftime('%Y-%m-%d'),
    'train_rows':       len(X_train),
    'test_rows':        len(X_test),
    'judge_precision':  round(float(judge_eval['precision']), 4),
    'judge_f1':         round(float(judge_eval['f1']), 4),
    'vit_precision':    round(float(vit_alone['precision']), 4),
    'vit_f1':           round(float(vit_alone['f1']), 4),
}
config_path = os.path.join(BASE_DIR, "judge_config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n  💾 Judge config saved: {config_path}")

print(f"\n{'='*55}")
print(f"  TRAINING COMPLETE")
print(f"  Judge model  : {MODEL_PATH}")
print(f"  Results dir  : {OUTPUT_DIR}")
print(f"\n  INTERPRETATION:")
print(f"  If Judge Precision > ViT Precision → synergy works ✅")
print(f"  If Judge F1 > ViT F1               → more trades found ✅")
print(f"  If Judge Precision < ViT Precision → FFT not adding value ⚠️")
print(f"{'='*55}")