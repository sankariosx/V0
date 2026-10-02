import json
from pathlib import Path
import numpy as np
import pandas as pd
from features import build_features
from validation import block_wild_null_test, split_discovery_validation

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "xauusd_m15.csv"
OUT = ROOT / "results" / "run7_forensic.json"
HORIZON = 16
PURGE = 16
DISCOVERY_FRAC = 0.60
BLOCK_LEN = 64
SEED = 20261003

# These are frozen from the prior discovery run. Do not change them after seeing Run 7.
CANDIDATE_RULES = [
    ("dist_low_20 + hour_cos", [("dist_low_20", "<", 0.0003755636445740603), ("hour_cos", ">", 0.7933533402912352)]),
    ("range_ + hour_cos", [("range_", "<", 0.00014818282204814845), ("hour_cos", ">", 0.7933533402912352)]),
    ("rel_pos_50 + hour_cos", [("rel_pos_50", "<", 0.15929677521127186), ("hour_cos", ">", 0.7933533402912352)]),
]

def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    ts = "date" if "date" in df.columns else "datetime" if "datetime" in df.columns else "timestamp"
    if ts == "timestamp" and pd.api.types.is_numeric_dtype(df[ts]):
        df[ts] = pd.to_datetime(df[ts], unit="ms", utc=True)
    else:
        df[ts] = pd.to_datetime(df[ts], utc=True)
    return df.set_index(ts).sort_index()

def condition(feat, rules):
    m = np.ones(len(feat), dtype=bool)
    for f, op, t in rules:
        x = feat[f].to_numpy()
        m &= (x < t) if op == "<" else (x > t)
    return m

def stats(y, raw, cond):
    cond = np.asarray(cond, dtype=bool)
    y = np.asarray(y, dtype=float)
    raw = np.asarray(raw, dtype=float)
    if cond.sum() == 0:
        return {"n": 0, "effect_norm": 0.0, "mean_raw": 0.0, "median_raw": 0.0, "hit_rate": 0.0}
    return {
        "n": int(cond.sum()),
        "effect_norm": float(np.mean(y[cond]) - np.mean(y)),
        "mean_raw": float(np.mean(raw[cond])),
        "median_raw": float(np.median(raw[cond])),
        "hit_rate": float(np.mean(raw[cond] > 0)),
    }

def period_stats(y, raw, cond, stamps, parts=6):
    edges = np.linspace(0, len(y), parts + 1, dtype=int)
    out = []
    for i in range(parts):
        lo, hi = edges[i], edges[i+1]
        s = stats(y[lo:hi], raw[lo:hi], cond[lo:hi])
        s["start"] = str(stamps[lo])
        s["end"] = str(stamps[hi-1])
        out.append(s)
    return out

df = load_data(DATA_PATH)
feat, target = build_features(df, horizon=HORIZON, vol_window=32)
idx = pd.DatetimeIndex(df.index)
raw4 = np.log(df["close"].shift(-16) / df["close"])
raw8 = np.log(df["close"].shift(-32) / df["close"])
raw16 = np.log(df["close"].shift(-64) / df["close"])

future = pd.Series(idx, index=idx).shift(-HORIZON)
valid = ((future - pd.Series(idx, index=idx)) == pd.Timedelta(minutes=15*HORIZON)).reindex(target.index).fillna(False).to_numpy()
feat, target = feat.loc[valid], target.loc[valid]
raw4, raw8, raw16 = raw4.loc[feat.index], raw8.loc[feat.index], raw16.loc[feat.index]

fd, yd, fv, yv = split_discovery_validation(feat, target, frac=DISCOVERY_FRAC, purge=PURGE)
raw4v, raw8v, raw16v = raw4.loc[fv.index], raw8.loc[fv.index], raw16.loc[fv.index]
stamps = pd.DatetimeIndex(fv.index)

results = {}
for name, rules in CANDIDATE_RULES:
    cond = condition(fv, rules)
    hour = condition(fv, [("hour_cos", ">", 0.7933533402912352)])
    aligned4h = cond & (stamps.minute == 0) & ((stamps.hour % 4) == 0)

    # Main fixed-rule validation and a second half that is deliberately kept separate.
    mid = len(yv) // 2
    diagnostic = slice(0, mid)
    final = slice(mid, len(yv))

    main = stats(yv.to_numpy(), raw4v.to_numpy(), cond)
    main["p_value_norm"] = block_wild_null_test(yv.to_numpy(), cond, block_len=BLOCK_LEN, n_perm=4000, seed=SEED)["p_value"]
    main["periods_6"] = period_stats(yv.to_numpy(), raw4v.to_numpy(), cond, stamps, 6)
    main["horizon_8h"] = stats(yv.to_numpy(), raw8v.to_numpy(), cond)
    main["horizon_16h"] = stats(yv.to_numpy(), raw16v.to_numpy(), cond)

    main["hour_only"] = stats(yv.to_numpy(), raw4v.to_numpy(), hour)
    main["incremental_raw_vs_hour"] = float(main["mean_raw"] - main["hour_only"]["mean_raw"])

    # 4H-only decision points: fixed rule evaluated only at exact 4H closes.
    main["four_hour_aligned"] = stats(yv.to_numpy(), raw4v.to_numpy(), aligned4h)
    main["four_hour_aligned_p_value_norm"] = block_wild_null_test(
        yv.to_numpy(), aligned4h, block_len=BLOCK_LEN, n_perm=4000, seed=SEED+1
    )["p_value"]

    # Pre-registered-looking split: no refitting or threshold changes in the second half.
    main["diagnostic_half"] = stats(yv.to_numpy()[diagnostic], raw4v.to_numpy()[diagnostic], cond[diagnostic])
    main["late_fixed_half"] = stats(yv.to_numpy()[final], raw4v.to_numpy()[final], cond[final])

    results[name] = main

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "dataset": "Dukascopy XAUUSD M15",
    "validation_rows": int(len(yv)),
    "validation_start": str(stamps[0]),
    "validation_end": str(stamps[-1]),
    "candidate_rules_frozen_from_run6": True,
    "notes": {
        "target": "forward 16 x 15m normalized log return",
        "raw_return_horizons": "4h, 8h, 16h",
        "four_hour_alignment": "candidate must occur at an exact UTC 4H boundary (00/04/08/12/16/20); no rule changes",
        "late_half": "same frozen rule; no refit"
    },
    "results": results
}, indent=2))
print(OUT.read_text())
