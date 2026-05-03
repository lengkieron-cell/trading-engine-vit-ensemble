# trading-engine-vit-ensemble
Autonomous high-frequency trading pipeline using Vision Transformers and XGBoost ensemble validation.

1. data_collector4.py
Collects historical OHLCV data across 31 tickers (Tech, Financials, Healthcare) using yfinance. Applies Triple Barrier labelling to generate win/loss/timeout outcomes and extracts 8 frequency-domain and market context features per 100-bar window using FFT and Welch's method.
2. score-collector-2.py
Scores each collected window by rebuilding the GAF image and running ViT inference, then feeding FFT features sequentially into the LSTM. Appends vit_prob and lstm_prob to every row, producing the final Judge training dataset.
3. train-lstm-9-kaggle.py
Trains a 2-layer LSTM on sequential FFT feature windows to predict trade outcomes. Uses Triple Barrier labels and handles class imbalance via weighted loss. Outputs lstm_model.pt, lstm_scaler.json, and per-window probability scores for Judge training.
4. train_judge-5.py
Trains an XGBoost meta-classifier (the Judge) that combines ViT confidence, LSTM probability, FFT physics features, and market regime context to make final trade decisions. Evaluated on held-out unseen tickers for honest generalisation testing.
5. backtest-sim-2.0.py
Threshold sweep simulator that tests every combination of ViT and Judge thresholds (0.50→0.90) against historical data to find the optimal precision/recall tradeoff before live deployment.
6. watchdog9.0.py
Live trading signal scanner. Runs every evening, downloads recent price data, builds GAF images, runs the full ViT→LSTM→Judge pipeline with tiered confidence gating, and logs all signals and ghost trades to a research log for continuous improvement.

ADDED the training run code for the viit model as well , this would happen after STEP 1.
