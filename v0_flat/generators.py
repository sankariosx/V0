import numpy as np
import pandas as pd
from config_final import N_BARS, INITIAL_PRICE, GARCH_OMEGA, GARCH_ALPHA, GARCH_BETA, T_DF, GAP_PROB, GAP_MULT, HORIZON, VOL_WINDOW

def _garch_returns(n, seed=None):
    rng = np.random.default_rng(seed)
    df = T_DF
    scale = np.sqrt((df-2)/df)
    sigma2 = np.zeros(n)
    r = np.zeros(n)
    sigma2[0] = GARCH_OMEGA / (1 - GARCH_ALPHA - GARCH_BETA)
    r[0] = np.sqrt(sigma2[0]) * rng.standard_t(df) * scale
    for i in range(1, n):
        sigma2[i] = GARCH_OMEGA + GARCH_ALPHA * (r[i-1]**2) + GARCH_BETA * sigma2[i-1]
        mult = GAP_MULT if rng.random() < GAP_PROB else 1.0
        r[i] = np.sqrt(sigma2[i]) * rng.standard_t(df) * scale * mult
    return r, np.sqrt(sigma2)

def generate_ohlc_from_returns(returns, initial_price=INITIAL_PRICE, seed=None):
    rng = np.random.default_rng(seed)
    n = len(returns)
    log_price = np.zeros(n)
    log_price[0] = np.log(initial_price)
    for i in range(1, n):
        log_price[i] = log_price[i-1] + returns[i]
    close = np.exp(log_price)
    open_ = np.zeros(n); open_[0]=initial_price; open_[1:]=close[:-1]
    high = np.zeros(n); low = np.zeros(n)
    for i in range(n):
        bar_range = abs(returns[i]) * close[i] * 0.3 + rng.uniform(0.0001, 0.0005)*close[i]
        high[i] = max(open_[i], close[i]) + bar_range * rng.uniform(0.2, 1.0)
        low[i] = min(open_[i], close[i]) - bar_range * rng.uniform(0.2, 1.0)
    idx = pd.date_range(start="2020-01-01", periods=n, freq="15min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "return": returns}, index=idx)

def generate_null_market(n_bars=N_BARS, seed=None):
    r, sigma = _garch_returns(n_bars, seed=seed)
    df = generate_ohlc_from_returns(r, seed=seed)
    df["sigma_true"] = sigma
    df["market_type"] = "null"
    return df

def verify_no_predictive_signal(df, feature_df=None, horizon=HORIZON, vol_window=VOL_WINDOW):
    from features import compute_target, build_features
    from validation import circular_block_bootstrap
    import numpy as np
    if feature_df is None:
        feature_df, target = build_features(df, horizon=horizon, vol_window=vol_window)
    else:
        _, target = compute_target(df, horizon, vol_window)
    y = target.dropna()
    common = feature_df.index.intersection(y.index)
    y = y.loc[common]; feature_df = feature_df.loc[common]
    checks = {}
    mean_y = y.mean(); std_y = y.std(); n = len(y)
    t_stat = mean_y / (std_y/np.sqrt(n) + 1e-12)
    checks["uncond_mean"] = float(mean_y); checks["uncond_t"] = float(t_stat); checks["uncond_pass"] = abs(t_stat) < 2.5
    cond_failures = 0; total_bins = 0; worst_bin_t = 0; worst_bin_p = 1.0
    for col in feature_df.columns[:15]:
        if feature_df[col].nunique() < 5: continue
        try:
            q = pd.qcut(feature_df[col], 5, duplicates="drop")
            for _, group in y.groupby(q, observed=True):
                if len(group) < 100: continue
                mask = y.index.isin(group.index)
                if mask.sum() < 100: continue
                res = circular_block_bootstrap(y.values, mask, block_len=64, n_boot=300, seed=42)
                t_val = abs(res["effect"]/(res["boot_std"]+1e-12))
                worst_bin_t = max(worst_bin_t, t_val)
                worst_bin_p = min(worst_bin_p, res["p_value"])
                if res["p_value"] < 0.01:
                    cond_failures += 1
                    break
            total_bins += 1
        except Exception:
            continue
    checks["cond_bins_tested"] = total_bins; checks["cond_failures"] = cond_failures
    checks["worst_bin_t"] = float(worst_bin_t); checks["worst_bin_p_boot"] = float(worst_bin_p)
    checks["cond_pass"] = cond_failures <= 2
    max_corr = 0; max_feat = None
    for col in feature_df.columns:
        corr = np.corrcoef(feature_df[col].fillna(0).values, y.values)[0,1]
        if not np.isnan(corr) and abs(corr) > max_corr:
            max_corr = abs(corr); max_feat = col
    checks["max_abs_corr"] = float(max_corr); checks["max_corr_feature"] = max_feat
    checks["construction_guarantee"] = "E[y|F]=0 by construction"
    checks["overall_pass"] = checks["uncond_pass"] and checks["cond_pass"]
    checks["overall_pass_strict"] = checks["overall_pass"] and (max_corr < 0.05)
    return checks

def generate_injected_market(n_bars=N_BARS, effect_type="1way", effect_size=0.3, seed=None):
    from features import build_features
    base_df = generate_null_market(n_bars=n_bars, seed=seed)
    feat, y_base = build_features(base_df, horizon=HORIZON, vol_window=VOL_WINDOW)
    common = feat.index.intersection(y_base.index)
    feat = feat.loc[common]; y_base = y_base.loc[common]; base_df_aligned = base_df.loc[common]
    import pandas as pd
    condition = pd.Series(False, index=feat.index)
    if effect_type == "1way":
        f = feat["ret_16"] if "ret_16" in feat.columns else feat.iloc[:,0]
        thresh = f.quantile(0.10)
        condition = f < thresh
        cond_desc = f"ret_16 < {thresh:.4f} (10th pct)"
    elif effect_type == "2way":
        f1 = feat["vol_expansion"] if "vol_expansion" in feat.columns else feat["vol_regime_1y"]
        f2 = feat["pct_rank_100"] if "pct_rank_100" in feat.columns else feat.iloc[:,2]
        t1 = f1.quantile(0.80); t2 = f2.quantile(0.20)
        condition = (f1 > t1) & (f2 < t2)
        cond_desc = f"vol_expansion > {t1:.3f} (80th) AND pct_rank_100 < {t2:.3f} (20th)"
    elif effect_type == "3way":
        f1 = feat["vol_regime_1y"] if "vol_regime_1y" in feat.columns else feat.iloc[:,1]
        f2 = feat["pct_rank_100"] if "pct_rank_100" in feat.columns else feat.iloc[:,2]
        f3 = feat["trend_persist_20"] if "trend_persist_20" in feat.columns else feat.iloc[:,3]
        t1 = f1.quantile(0.80); t2 = f2.quantile(0.20); t3 = f3.quantile(0.60)
        condition = (f1 > t1) & (f2 < t2) & (f3 > t3)
        cond_desc = f"vol_regime>{t1:.3f} AND pct_rank<{t2:.3f} AND trend>{t3:.3f}"
    else:
        raise ValueError("effect_type must be 1way,2way,3way")
    y_injected = y_base.copy()
    y_injected[condition] = y_injected[condition] + effect_size
    info = {"effect_type": effect_type, "effect_size": effect_size, "condition_desc": cond_desc, "n_condition": int(condition.sum()), "condition_rate": float(condition.mean()), "base_mean": float(y_base.mean()), "injected_mean_cond": float(y_injected[condition].mean()) if condition.sum()>0 else 0.0, "injected_mean_all": float(y_injected.mean())}
    return {"ohlc": base_df_aligned, "features": feat, "y_base": y_base, "y_injected": y_injected, "condition": condition, "info": info}

def generate_adversarial_null(n_bars=N_BARS, n_features_target=120, seed=None):
    from features import build_features
    import numpy as np
    rng = np.random.default_rng(seed)
    base_df = generate_null_market(n_bars=n_bars, seed=seed)
    feat_base, y_base = build_features(base_df, horizon=HORIZON, vol_window=VOL_WINDOW)
    common = feat_base.index.intersection(y_base.index)
    feat_base = feat_base.loc[common]; y_base = y_base.loc[common]
    n_base = feat_base.shape[1]
    expanded = {}
    for c in feat_base.columns:
        expanded[c] = feat_base[c].values
    idx = 0
    while len(expanded) < n_features_target - 30:
        base_col = feat_base.columns[idx % n_base]
        noise = rng.normal(0, 0.1 * feat_base[base_col].std() + 1e-8, size=len(feat_base))
        expanded[f"{base_col}_copy{idx}"] = feat_base[base_col].values + noise
        idx += 1
    for i in range(30):
        expanded[f"noise_{i}"] = rng.normal(0, 1, size=len(feat_base))
    for i in range(5):
        c1, c2 = rng.choice(feat_base.columns, 2, replace=False)
        expanded[f"combo_{i}"] = 0.6*feat_base[c1].values + 0.4*feat_base[c2].values + rng.normal(0,0.05,size=len(feat_base))
    import pandas as pd
    feat_expanded = pd.DataFrame(expanded, index=feat_base.index)
    cols = list(feat_expanded.columns)[:n_features_target]
    feat_expanded = feat_expanded[cols]
    return {"ohlc": base_df.loc[common], "features": feat_expanded, "y_base": y_base, "y_injected": y_base, "info": {"n_features": len(cols), "type": "adversarial_null"}}
