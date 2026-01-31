# predict_live.py
import pandas as pd
import numpy as np
from src.data import load_universe
from src.features import compute_own_features, expanding_ols_features
from src.sector import peer_sector_index
from src.utils import load_pickle

TICKER = "AAPL"
MARKET_TICKER = "QQQ"
ART_DIR = "artifacts"
DATA_DIR = "data"

def main():
    # Load models
    models = {
        "elasticnet": load_pickle("model_elasticnet.pkl"),
        "randomforest": load_pickle("model_randomforest.pkl"),
        "gbrt": load_pickle("model_gbrt.pkl"),
    }
    meta = load_pickle("meta_elasticnet.pkl")

    # Load latest data
    univ = load_universe(DATA_DIR)
    prices = pd.DataFrame({t: df["Adj Close"] for t, df in univ.items()})
    returns = prices.pct_change(fill_method=None)

    # Load peers used during training
    peers = load_pickle("sector_peers_AAPL.pkl")  # we’ll save this once (see below)
    sector = peer_sector_index(returns, peers)

    # Build features
    own_feats = compute_own_features(univ[TICKER])
    exp_ols = expanding_ols_features(
        returns[TICKER],
        returns[MARKET_TICKER],
        sector,
    )

    feats = pd.concat(
        [
            own_feats,
            exp_ols,
            returns[MARKET_TICKER].rename("r_mkt"),
            sector.rename("r_sec"),
        ],
        axis=1,
    ).dropna()

    # Take the LAST available date (today)
    X_today = feats.iloc[[-1]]

    # Base predictions
    base_preds = np.column_stack([
        m.predict(X_today) for m in models.values()
    ])

    # Meta prediction
    r_hat = meta.predict(base_preds)[0]

    # Price today
    p_today = prices[TICKER].iloc[-1]
    p_hat = p_today * (1 + r_hat)

    print(f"Prediction date: {X_today.index[0].date()}")
    print(f"Predicted return (T+1): {r_hat:.4%}")
    print(f"Predicted price (T+1): {p_hat:.2f}")

if __name__ == "__main__":
    main()
