import json
from pathlib import Path
import numpy as np
import pandas as pd
from features import build_features
from validation import split_discovery_validation, block_wild_null_test, replication_check

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "xauusd_m15.csv"
OUT = ROOT / "results" / "run6_targeted_diagnostic.json"
HORIZON = 16
PURGE = 16
DISCOVERY_FRAC = 0.60
BLOCK_LEN = 64

CANDIDATES = [
    ("range_ + hour_cos", [("range_", "<", 0.00014818282204814845), ("hour_cos", ">", 0.7933533402912352)]),
    ("dist_low_20 + hour_cos", [("dist_low_20", "<", 0.0003755636445740603), ("hour_cos", ">", 0.7933533402912352)]),
    ("rel_pos_50 + hour_cos", [("rel_pos_50", "<", 0.15929677521127186), ("hour_cos", ">", 0.7933533402912352)]),
    ("swing_high_age_50 + hour_cos", [("swing_high_age_50", ">", 47.0), ("hour_cos", ">", 0.7933533402912352)]),
]

def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    ts = "date" if "date" in df.columns else "datetime" if "datetime" in df.columns else "timestamp"
    if ts == "timestamp" and pd.api.types.is_numeric_dtype(df[ts]):
        df[ts] = pd.to_datetime(df[ts], unit="ms", utc=True)
    else:
        df[ts] = pd.to_datetime(df[ts], utc=True)
    df = df.set_index(ts).sort_index()
    return df

def condition(feat, rules):
    m = np.ones(len(feat), dtype=bool)
    for f, op, t in rules:
        x = feat[f].to_numpy()
        m &= (x < t) if op == "<" else (x > t)
    return m

def block_stats(y, cond):
    y = np.asarray(y, dtype=float)
    cond = np.asarray(cond, dtype=bool)
    edges = np.linspace(0, len(y), 4, dtype=int)
    blocks = []
    for i in range(3):
        yy, cc = y[edges[i]:edges[i+1]], cond[edges[i]:edges[i+1]]
        blocks.append({
            "n": int(len(yy)),
            "n_condition": int(cc.sum()),
            "effect": float(np.mean(yy[cc]) - np.mean(yy)) if cc.sum() else 0.0
        })
    return blocks

df = load_data(DATA_PATH)
feat, target = build_features(df, horizon=HORIZON, vol_window=32)
idx = pd.DatetimeIndex(df.index)
future = pd.Series(idx, index=idx).shift(-HORIZON)
valid = ((future - pd.Series(idx, index=idx)) == pd.Timedelta(minutes=15*HORIZON)).reindex(target.index).fillna(False).to_numpy()
feat, target = feat.loc[valid], target.loc[valid]

_, _, fv, yv = split_discovery_validation(feat, target, frac=DISCOVERY_FRAC, purge=PURGE)

results = []
for name, rules in CANDIDATES:
    full = condition(fv, rules)
    hour = condition(fv, [("hour_cos", ">", 0.7933533402912352)])
    full_eval = block_wild_null_test(yv.to_numpy(), full, block_len=BLOCK_LEN, n_perm=2000, seed=20261002)
    hour_eval = block_wild_null_test(yv.to_numpy(), hour, block_len=BLOCK_LEN, n_perm=2000, seed=20261002)
    rep, rep_effects = replication_check(yv.to_numpy(), full)
    hour_rep, hour_rep_effects = replication_check(yv.to_numpy(), hour)
    results.append({
        "candidate": name,
        "candidate_n": int(full.sum()),
        "candidate_effect": full_eval["effect"],
        "candidate_p": full_eval["p_value"],
        "candidate_replication_pass": bool(rep),
        "candidate_block_effects": rep_effects,
        "hour_only_n": int(hour.sum()),
        "hour_only_effect": hour_eval["effect"],
        "hour_only_p": hour_eval["p_value"],
        "hour_only_replication_pass": bool(hour_rep),
        "hour_only_block_effects": hour_rep_effects,
        "incremental_effect_vs_hour_only": float(full_eval["effect"] - hour_eval["effect"]),
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "dataset": "Dukascopy XAUUSD M15",
    "validation_rows": len(yv),
    "candidates": results
}, indent=2))
print(OUT.read_text())
