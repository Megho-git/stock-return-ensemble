# src/predict.py
import argparse
import json
import pandas as pd
from .utils import load_pickle
from pathlib import Path
from datetime import datetime

def load_artifacts():
    # expected artifacts: scalers, models, meta, feature config. Adjust names as you used.
    artifacts = {}
    # these load calls should match the saved filenames in training
    try:
        artifacts["meta"] = load_pickle("meta_elasticnet.pkl")
    except Exception:
        artifacts["meta"] = None
    # load refit base models
    for name in ["elasticnet", "randomforest", "gbrt"]:
        try:
            artifacts[name] = load_pickle(f"model_{name}_refit.pkl")
        except Exception:
            artifacts[name] = None
    return artifacts

def predict_for_date(ticker, date_str, artifacts, features_df, price_df):
    """
    features_df: full-feature DataFrame for all dates & tickers constructed earlier (index Date)
    price_df: price Series for ticker
    """
    date = pd.to_datetime(date_str)
    if date not in features_df.index:
        raise KeyError(f"{date_str} not in feature index")
    X = features_df.loc[[date]].values  # single row
    # base preds: average of available base models
    base_preds = []
    for name in ["elasticnet","randomforest","gbrt"]:
        m = artifacts.get(name)
        if m is None:
            continue
        try:
            p = m.predict(X)[0]
            base_preds.append(p)
        except Exception:
            pass
    if len(base_preds) == 0:
        raise RuntimeError("No base models available.")
    base_mean = sum(base_preds) / len(base_preds)
    meta = artifacts.get("meta")
    if meta is not None:
        # stack: build array of base preds in consistent order for meta
        # NOTE: meta expects the same columns as OOF used; here we use [elasticnet,randomforest,gbrt]
        arr = []
        for name in ["elasticnet","randomforest","gbrt"]:
            if artifacts.get(name) is not None:
                try:
                    arr.append(artifacts[name].predict(X)[0])
                except Exception:
                    arr.append(base_mean)
            else:
                arr.append(base_mean)
        arr = pd.np.array(arr).reshape(1, -1)
        r_hat = meta.predict(arr)[0]
    else:
        r_hat = base_mean
    p_today = price_df.loc[date]
    p_hat = p_today * (1 + r_hat)
    # drivers: for lightweight explanation return top features by absolute value of linear model coef if meta is linear
    drivers = {}
    drivers["own"] = {}  # placeholder — you can return the most relevant features from features_df
    # For now return a few core fields
    drivers["own"]["r_t"] = float(features_df.loc[date].get("r_t", float("nan")))
    return {
        "date": date_str,
        "ticker": ticker,
        "r_hat_next": float(r_hat),
        "p_hat_next": float(p_hat),
        "decision": "Buy" if r_hat > 0.01 else ("Sell" if r_hat < -0.01 else "Hold"),
        "drivers": drivers
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--features", required=True, help="path to precomputed features (parquet/csv).")
    parser.add_argument("--prices", required=True, help="path to prices series (parquet/csv) for ticker")
    args = parser.parse_args()
    artifacts = load_artifacts()
    if args.features.endswith(".parquet"):
        features = pd.read_parquet(args.features)
    else:
        features = pd.read_csv(args.features, parse_dates=["Date"], index_col="Date")
    if args.prices.endswith(".parquet"):
        prices = pd.read_parquet(args.prices)
    else:
        prices = pd.read_csv(args.prices, parse_dates=["Date"], index_col="Date")
    # features expected to be DataFrame indexed by Date
    out = predict_for_date(args.ticker, args.date, artifacts, features, prices[args.ticker] if args.ticker in prices.columns else prices.squeeze())
    print(json.dumps(out, indent=2, default=str))

if __name__ == "__main__":
    main()
