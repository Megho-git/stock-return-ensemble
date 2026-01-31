# src/sector.py
import numpy as np
import pandas as pd

def compute_returns(price_series: pd.Series):
    return price_series.pct_change()

def pick_peers_by_correlation(train_returns: pd.DataFrame, target: str, k=10):
    """
    train_returns: DataFrame columns = tickers, index = dates (train period only)
    target: ticker string in columns
    Returns list of peer tickers (exclude target), sorted by correlation descending
    """
    corr = train_returns.corr()
    if target not in corr.columns:
        raise KeyError(f"{target} not in train_returns")
    target_corr = corr[target].drop(index=target).sort_values(ascending=False)
    peers = list(target_corr.index[:k])
    return peers

def peer_sector_index(all_returns: pd.DataFrame, peers: list):
    """
    all_returns: DataFrame (full time) columns = tickers
    peers: list of tickers
    Returns Series: mean return across peers for each date (NaN if peers missing)
    """
    sector = all_returns[peers].mean(axis=1)
    return sector
