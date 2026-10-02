"""
Run11: frozen forward OOS test of the two Run10 direct-4H survivors.

No discovery, no refitting, no threshold selection. Rules are copied verbatim
from Run10 and are evaluated only on 4H decision bars strictly after 2026-09-01.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from features import build_features

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "xauusd_m15.csv"
OUT_PATH = ROOT / "results" / "run11_frozen_forward_oos.json"
CUTOFF = pd.Timestamp("2026-09-01 00:00:00", tz="UTC")

RULES = [
    {
        "name": "run10_2way_consec5_dist_high100",
        "type": "2way",
        "features": ["consec_5", "dist_high_100"],
        "thresholds": [1.0, 0.008003802742536977],
        "direction": "long",
    },
    {
        "name": "run10_1way_swing_high_age50",
        "type": "1way",
        "features": ["swing_high_age_50"],
        "thresholds": [1.0],
        "direction": "long",
    },
]

def load_m15(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    ts_col = "date" if "date" in df.columns else "datetime" if "datetime" in df.columns else "timestamp"
    if ts_col == "timestamp" and pd.api.types.is_numeric_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], unit="ms", utc=True)
    else:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index(ts_col).sort_index()
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df

def make_4h_bars(m15):
    groups = m15.groupby(m15.index.floor("4h"), sort=True)
    rows = []
    for start, g in groups:
        if start.dayofweek >= 5:
            continue
        expected = pd.date_range(start, periods=16, freq="15min", tz="UTC")
        if len(g) != 16 or not g.index.equals(expected):
            continue
        rows.append({
            "timestamp": start + pd.Timedelta(hours=4),
            "open": float(g["open"].iloc[0]),
            "high": float(g["high"].max()),
            "low": float(g["low"].min()),
            "close": float(g["close"].iloc[-1]),
        })
    return pd.DataFrame(rows).set_index("timestamp").sort_index()

def main():
    m15 = load_m15(DATA_PATH)
    bars = make_4h_bars(m15)
    feat, target_norm = build_features(bars, horizon=1, vol_window=16)

    # Rebuild the raw next-4H return on the exact feature index.
    raw_log_ret = np.log(bars["close"].shift(-1) / bars["close"]).reindex(feat.index)
    next_ts = pd.Series(feat.index, index=feat.index).shift(-1)
    contiguous = (next_ts - pd.Series(feat.index, index=feat.index)) == pd.Timedelta(hours=4)
    mask = contiguous.fillna(False).to_numpy()
    feat = feat.loc[mask]
    target_norm = target_norm.loc[mask]
    raw_log_ret = raw_log_ret.loc[mask]

    # True forward period: no observation at or before the Run10 data cutoff.
    oos = feat.index > CUTOFF
    feat = feat.loc[oos]
    target_norm = target_norm.loc[oos]
    raw_log_ret = raw_log_ret.loc[oos]

    results = []
    for rule in RULES:
        if rule["type"] == "2way":
            cond = (feat[rule["features"][0]] > rule["thresholds"][0]) & (feat[rule["features"][1]] < rule["thresholds"][1])
        else:
            cond = feat[rule["features"][0]] < rule["thresholds"][0]
        r = raw_log_ret[cond].dropna()
        yn = target_norm[cond].dropna()
        if len(r) == 0:
            results.append({"name": rule["name"], "n_signals": 0})
            continue
        results.append({
            "name": rule["name"],
            "direction": rule["direction"],
            "n_signals": int(len(r)),
            "signal_rate": float(len(r) / len(raw_log_ret)),
            "mean_raw_log_return": float(r.mean()),
            "median_raw_log_return": float(r.median()),
            "win_rate": float((r > 0).mean()),
            "mean_normalized_return": float(yn.mean()),
            "sum_raw_log_return": float(r.sum()),
            "cost_sensitivity": {
                f"{bps}bps": float(r.sum() - len(r) * bps / 10000.0)
                for bps in [0, 5, 10, 20, 30]
            },
            "first_signal": str(r.index.min()),
            "last_signal": str(r.index.max()),
        })

    report = {
        "dataset": "Dukascopy XAUUSD M15 -> completed UTC 4H bars",
        "run10_cutoff": str(CUTOFF),
        "oos_start": str(feat.index.min()) if len(feat) else None,
        "oos_end": str(feat.index.max()) if len(feat) else None,
        "oos_4h_rows": int(len(feat)),
        "rules_frozen_from_run10": True,
        "discovery_performed": False,
        "rules": RULES,
        "results": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
