# src/features.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def pct_change(series):
    return series.pct_change()

def add_lags(df, col, lags):
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

def rolling_stats(series, window, prefix):
    rmean = series.rolling(window=window, min_periods=1).mean()
    rstd = series.rolling(window=window, min_periods=1).std()
    return rmean, rstd

def compute_own_features(df):
    """
    df: DataFrame with columns ['Adj Close','High','Low','Open','Close','Volume'] and index Date
    Returns DataFrame of features with aligned index (same dates) where feature values at date t
    are computed using data up to t (no future leaks)
    """
    out = pd.DataFrame(index=df.index)
    price = df["Adj Close"]
    r = price.pct_change()
    out["r_t"] = r
    # lags r_t, up to 5
    for lag in range(1, 6):
        out[f"r_lag{lag}"] = r.shift(lag)
    # rolling means and vol
    for w in [5, 20, 63]:
        out[f"mom_{w}"] = r.rolling(window=w, min_periods=1).mean().shift(0)  # it's ok to use up to t
        out[f"vol_{w}"] = r.rolling(window=w, min_periods=1).std().shift(0)
    # H-L / C and rolling normalized
    if {"High", "Low", "Close"}.issubset(df.columns):
        hlc = (df["High"] - df["Low"]) / df["Close"]
        for w in [20, 63]:
            out[f"hlc_mean_{w}"] = hlc.rolling(window=w, min_periods=1).mean()
            out[f"hlc_pctile_{w}"] = hlc.rolling(window=w, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    # SMA gaps
    for w in [5, 20, 63]:
        sma = price.rolling(window=w, min_periods=1).mean()
        out[f"sma_gap_{w}"] = (price - sma) / sma
    # volume log zscore (21d)
    if "Volume" in df.columns:
        vol = df["Volume"].replace(0, np.nan)
        lv = np.log(vol)
        out["vol_log"] = lv
        out["vol21_z"] = (lv - lv.rolling(21, min_periods=1).mean()) / (lv.rolling(21, min_periods=1).std().replace(0, np.nan))
        # 5/20 day deltas of vol21_z
        out["vol21_z_delta5"] = out["vol21_z"] - out["vol21_z"].shift(5)
        out["vol21_z_delta20"] = out["vol21_z"] - out["vol21_z"].shift(20)
    return out

def compute_market_breadth(returns_df):
    """
    returns_df: DataFrame columns=tickers, index=dates, values = returns (pct)
    breadth_t = fraction of tickers with return > 0 on date t
    """
    breadth = (returns_df > 0).sum(axis=1) / returns_df.shape[1]
    return breadth

def expanding_ols_features(target_returns, mkt_returns, sec_returns, min_obs=10):
    """
    For each date t compute betas (beta_mkt, beta_sec) and idiosyncratic residual e_t
    using only past data (expanding OLS up to t).
    Returns DataFrame with columns ['beta_mkt','beta_sec','idio_resid'] and index aligned with returns.
    Implementation loops over dates; reasonably fast for daily data decently sized universe.
    """
    dates = target_returns.index
    out = pd.DataFrame(index=dates, columns=["beta_mkt", "beta_sec", "idio_resid"])
    # Prepare X_full with columns mkt, sec
    X_full = pd.concat([mkt_returns.rename("mkt"), sec_returns.rename("sec")], axis=1)
    for i in range(len(dates)):
        t = dates[i]
        # use data up to and including t
        y = target_returns.loc[:t].dropna()
        X = X_full.loc[y.index].dropna()
        # ensure aligned
        df = pd.concat([y, X], axis=1).dropna()
        if len(df) < min_obs:
            out.loc[t] = [np.nan, np.nan, np.nan]
            continue
        Y = df.iloc[:, 0].values
        Xmat = df.iloc[:, 1:].values
        Xmat_const = np.column_stack([np.ones(len(Xmat)), Xmat])
        # solve OLS: [const, mkt, sec]
        try:
            coef, _, _, _ = np.linalg.lstsq(Xmat_const, Y, rcond=None)
            beta_mkt = coef[1]
            beta_sec = coef[2]
            # residual at date t = observed y_t - predicted using coefs
            x_t = Xmat_const[-1]
            resid_t = Y[-1] - x_t.dot(coef)
            out.loc[t] = [beta_mkt, beta_sec, resid_t]
        except Exception:
            out.loc[t] = [np.nan, np.nan, np.nan]
    return out
