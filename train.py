# train.py
import pandas as pd
import numpy as np

from src.data import load_universe
from src.features import compute_own_features, expanding_ols_features
from src.sector import pick_peers_by_correlation, peer_sector_index
from src.models import default_models, fit_and_save
from src.stack import train_meta_model
from src.evaluate import metrics_df, save_predictions_csv
from src.utils import ensure_dir

DATA_DIR = "data"
ART_DIR = "artifacts"

ticker = "AAPL"
MARKET_TICKER = "QQQ"

TRAIN_YEARS = [2016, 2017, 2018]
VAL_YEAR = 2019
TEST_YEAR = 2020
TEST_MONTH_CUTOFF = 3


def main(ticker="AAPL"):
    ensure_dir(ART_DIR)

    print("Loading universe...")
    univ = load_universe(DATA_DIR)

    assert ticker in univ, "Target ticker not found"
    assert MARKET_TICKER in univ, "Market ticker not found"

    # --------------------------------------------------
    # Prices & returns
    # --------------------------------------------------
    prices = pd.DataFrame({
        t: df["Adj Close"] for t, df in univ.items()
    })

    returns = prices.pct_change(fill_method=None)

    # --------------------------------------------------
    # SECTOR PEERS (DATE-BASED, NO MASKS)
    # --------------------------------------------------
    print("Selecting sector peers...")
    train_returns = returns.loc[returns.index.year.isin(TRAIN_YEARS)]
    peers = pick_peers_by_correlation(
        train_returns,
        ticker,
        k=10
    )

    from src.utils import save_pickle
    save_pickle(peers, f"sector_peers_{ticker}.pkl")

    sector = peer_sector_index(returns, peers)

    # --------------------------------------------------
    # FEATURES
    # --------------------------------------------------
    print("Building features...")
    own_feats = compute_own_features(univ[ticker])

    exp_ols = expanding_ols_features(
        returns[ticker],
        returns[MARKET_TICKER],
        sector
    )

    feats = pd.concat(
        [
            own_feats,
            exp_ols,
            returns[MARKET_TICKER].rename("r_mkt"),
            sector.rename("r_sec"),
        ],
        axis=1,
    )

    # --------------------------------------------------
    # TARGET
    # --------------------------------------------------
    y = returns[ticker].shift(-1)

    data = feats.join(y.rename("y")).dropna()

    X = data.drop(columns="y")
    y = data["y"]

    # --------------------------------------------------
    # DATE SPLITS (ONLY AFTER X EXISTS)
    # --------------------------------------------------
    idx = X.index

    train_mask = idx.year.isin(TRAIN_YEARS)
    val_mask = idx.year == VAL_YEAR
    test_mask = (idx.year == TEST_YEAR) & (idx.month <= TEST_MONTH_CUTOFF)

    assert train_mask.any(), "No training data!"
    assert val_mask.any(), "No validation data!"
    assert test_mask.any(), "No test data!"

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    print(f"Train rows: {len(X_train)}")
    print(f"Val rows:   {len(X_val)}")
    print(f"Test rows:  {len(X_test)}")

    # --------------------------------------------------
    # BASE MODELS
    # --------------------------------------------------
    print("Training base models...")
    models = default_models()

    for name, model in models.items():
        fit_and_save(model, X_train, y_train, name)

    # ---- META MODEL (NO OOF, SAFE STACKING) ----
    print("Training meta-model...")

    # Base models already trained on TRAIN
    meta = train_meta_model(
        models,
        X_val,
        y_val,
    )


    # --------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------
    print("Evaluating on test set...")
    preds = []

    for name, model in models.items():
        model.fit(
            pd.concat([X_train, X_val]),
            pd.concat([y_train, y_val]),
        )
        preds.append(model.predict(X_test))

    base_preds = np.column_stack(preds)
    final_pred = meta.predict(base_preds)

    metrics = metrics_df(y_test, final_pred)
    print("TEST METRICS:", metrics)

    out = pd.DataFrame(
        {
            "Date": X_test.index,
            "Ticker": ticker,
            "y_true": y_test.values,
            "y_pred": final_pred,
        }
    )

    save_predictions_csv(
        out, f"{ART_DIR}/test_predictions_{ticker}.csv"
    )

    feats.to_parquet(f"{ART_DIR}/features_{ticker}.parquet")

    print("DONE.")


if __name__ == "__main__":
    main()
