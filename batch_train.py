# batch_train.py
import pandas as pd
from train import main as train_one

TICKERS = ["AAPL", "MSFT", "GOOGL"]  # start small

for t in TICKERS:
    print(f"\n===== TRAINING {t} =====")
    try:
        train_one(ticker=t)
    except Exception as e:
        print(f"FAILED {t}: {e}")
