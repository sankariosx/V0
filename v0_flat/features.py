import numpy as np
import pandas as pd

def _sma(series, window):
    return series.rolling(window, min_periods=window).mean()

def _slope(series, window):
    return (series - series.shift(window)) / window

def _percentile_rank(series, window):
    def rank_func(x):
        if len(x) < window:
            return np.nan
        return (np.sum(x <= x[-1]) - 1) / (len(x)-1) if len(x)>1 else 0.5
    return series.rolling(window, min_periods=window).apply(rank_func, raw=True)

def compute_target(df, horizon, vol_window):
    close = df["close"]
    log_ret_forward = np.log(close.shift(-horizon) / close)
    past_ret = np.log(close / close.shift(1))
    sigma = past_ret.ewm(span=vol_window, min_periods=vol_window).std()
    sigma = sigma.replace(0, np.nan).bfill()
    y = log_ret_forward / (sigma * np.sqrt(horizon) + 1e-12)
    return past_ret, y

def build_features(df, horizon=16, vol_window=32):
    o = df["open"]; h = df["high"]; l = df["low"]; c = df["close"]
    feats = {}
    ret_1 = np.log(c / c.shift(1))
    ret_4 = np.log(c / c.shift(4))
    ret_16 = np.log(c / c.shift(16))
    ret_32 = np.log(c / c.shift(32))
    feats["ret_1"] = ret_1; feats["ret_4"] = ret_4; feats["ret_16"] = ret_16; feats["ret_32"] = ret_32
    range_ = (h - l) / c
    body = (c - o).abs() / c
    body_range = body / (range_ + 1e-9)
    rng = (h - l)
    upper_wick_ratio = (h - np.maximum(o,c)) / (rng + 1e-9)
    lower_wick_ratio = (np.minimum(o,c) - l) / (rng + 1e-9)
    feats["range_"] = range_
    feats["body_range_ratio"] = body_range.fillna(0)
    feats["upper_wick_ratio"] = upper_wick_ratio.fillna(0)
    feats["lower_wick_ratio"] = lower_wick_ratio.fillna(0)
    feats["body_direction"] = np.sign(c - o)
    direction = np.sign(ret_1.fillna(0))
    feats["consec_5"] = direction.rolling(5).sum()
    rv_20 = ret_1.rolling(20).std(); rv_100 = ret_1.rolling(100).std()
    feats["realized_vol_20"] = rv_20; feats["realized_vol_100"] = rv_100
    feats["vol_ratio_20_100"] = rv_20 / (rv_100 + 1e-9)
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr_20 = tr.rolling(20).mean() / c; atr_100 = tr.rolling(100).mean() / c
    feats["atr_20_norm"] = atr_20; feats["atr_100_norm"] = atr_100
    feats["vol_expansion"] = atr_20 / (atr_100 + 1e-9)
    vol_regime = rv_20.rolling(25000, min_periods=1000).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)
    feats["vol_regime_1y"] = vol_regime
    feats["range_zscore_20"] = (range_ - range_.rolling(20).mean()) / (range_.rolling(20).std() + 1e-9)
    sma_50 = _sma(c, 50); sma_200 = _sma(c, 200)
    feats["close_sma50_dist"] = (c - sma_50) / c
    feats["close_sma200_dist"] = (c - sma_200) / c
    feats["sma50_sma200_dist"] = (sma_50 - sma_200) / c
    feats["slope_20"] = _slope(c, 20) / c; feats["slope_100"] = _slope(c, 100) / c
    feats["dist_roll_mean_z_50"] = (c - sma_50) / (c.rolling(50).std() + 1e-9)
    feats["trend_persist_20"] = (c > sma_50).rolling(20).mean()
    roc_8 = (c / c.shift(8) - 1); roc_32 = (c / c.shift(32) - 1)
    feats["roc_8"] = roc_8; feats["roc_32"] = roc_32; feats["mom_accel"] = roc_8 - roc_32
    high_50 = h.rolling(50).max(); low_50 = l.rolling(50).min()
    feats["rel_pos_50"] = (c - low_50) / (high_50 - low_50 + 1e-9)
    high_100 = h.rolling(100).max(); low_100 = l.rolling(100).min()
    feats["rel_pos_100"] = (c - low_100) / (high_100 - low_100 + 1e-9)
    feats["rsi_proxy_14"] = ret_1.rolling(14).apply(lambda x: (x>0).mean(), raw=True)
    feats["dist_high_20"] = (h.rolling(20).max() - c) / c
    feats["dist_low_20"] = (c - l.rolling(20).min()) / c
    feats["dist_high_100"] = (h.rolling(100).max() - c) / c
    feats["dist_low_100"] = (c - l.rolling(100).min()) / c
    feats["breakout_up_20"] = (c >= h.rolling(20).max().shift(1)).astype(int)
    feats["breakout_down_20"] = (c <= l.rolling(20).min().shift(1)).astype(int)
    feats["range_comp_20_100"] = range_.rolling(20).mean() / (range_.rolling(100).mean() + 1e-9)
    feats["swing_high_age_50"] = c.rolling(50).apply(lambda x: 49 - np.argmax(x), raw=True)
    feats["swing_low_age_50"] = c.rolling(50).apply(lambda x: 49 - np.argmin(x), raw=True)
    feats["pct_rank_50"] = _percentile_rank(c, 50)
    feats["pct_rank_100"] = _percentile_rank(c, 100)
    feats["pct_rank_200"] = _percentile_rank(c, 200)
    feats["dist_extreme_100"] = (c - (high_100+low_100)/2) / (high_100 - low_100 + 1e-9)
    feats["dist_extreme_200"] = (c - (h.rolling(200).max()+l.rolling(200).min())/2) / (h.rolling(200).max() - l.rolling(200).min() + 1e-9)
    hour = df.index.hour + df.index.minute/60.0
    feats["hour_sin"] = np.sin(2*np.pi*hour/24)
    feats["hour_cos"] = np.cos(2*np.pi*hour/24)
    feats["is_london"] = ((df.index.hour >= 8) & (df.index.hour < 12)).astype(int)
    feats["is_ny"] = ((df.index.hour >= 13) & (df.index.hour < 17)).astype(int)
    feats["is_overlap"] = ((df.index.hour >= 13) & (df.index.hour < 16)).astype(int)
    feats["dow"] = df.index.dayofweek
    feature_df = pd.DataFrame(feats, index=df.index)
    _, target = compute_target(df, horizon, vol_window)
    min_valid = int(len(feature_df.columns)*0.8)
    feature_df = feature_df.dropna(thresh=min_valid)
    feature_df = feature_df.ffill().fillna(0)
    common = feature_df.index.intersection(target.index)
    feature_df = feature_df.loc[common]
    target = target.loc[common]
    mask = target.notna()
    feature_df = feature_df[mask]
    target = target[mask]
    return feature_df, target
