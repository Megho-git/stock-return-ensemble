# src/data.py
import pandas as pd
import numpy as np
from pathlib import Path
from .utils import safe_read_csv

def load_universe(data_dir: str, tickers: list = None):
    """
    Load CSVs from data_dir. Each file should be <TICKER>.csv and contain Date,Adj Close,Volume.
    Returns dict[ticker] -> DataFrame indexed by Date with columns ['Adj Close','Volume', 'Open', ...]
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv"))
    res = {}
    for f in files:
        ticker = f.stem.upper()
        if tickers is not None and ticker not in [t.upper() for t in tickers]:
            continue
        df = safe_read_csv(f)
        df = df.sort_values("Date")
        df = df.set_index("Date")
        # Ensure Adj Close exists
        if "Adj Close" not in df.columns and "Adj_Close" in df.columns:
            df = df.rename(columns={"Adj_Close": "Adj Close"})
        # Basic cleaning
        if "Volume" in df.columns:
            df = df[df["Volume"].notna()]
            df = df[df["Volume"] != 0]
        df = df.dropna(subset=["Adj Close"])
        res[ticker] = df
    return res

def align_universe(univ_dict):
    """
    Align all tickers on union of dates. Returns DataFrame with MultiIndex columns (ticker, col).
    Missing trading days remain NaN for that ticker.
    """
    # collect outer join on dates
    dfs = []
    for t, df in univ_dict.items():
        df2 = df[["Adj Close"]].rename(columns={"Adj Close": t})
        dfs.append(df2)
    price_df = pd.concat(dfs, axis=1)
    return price_df
