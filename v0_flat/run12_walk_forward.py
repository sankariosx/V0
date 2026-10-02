"""
V0 finite walk-forward test.

Purpose:
- Stop the Run10/Run11 loop.
- Run the same direct-4H discovery machinery on earlier historical training
  periods, freeze the discovered rules, and evaluate them only on the next
  chronological block.
- No discovery is performed on any test block.

This is the single finite validation run; no follow-up diagnostic is planned unless the run itself exposes an implementation failure.\n\nTwo folds are used:
  Fold A: train through 2022-12-31, test 2023-01-01 -> 2024-12-31
  Fold B: train through 2024-12-31, test 2025-01-01 -> 2026-09-01

The discovery settings are kept identical to Run10. The test block is never
used to choose candidates or thresholds.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from config_final import (
    OUTER_BAGS, BAG_SAMPLE_FRAC, BOOST_ROUNDS, TREE_MAX_DEPTH,
    TOP_FEATURES_FOR_PAIRS, MAX_INTERACTIONS, MIN_SAMPLES_FOR_EDGE,
    EFFECT_SIZE_THRESH, N_BOOTSTRAP, ALPHA_FINAL, BLOCK_LENGTH,
    PURGE_GAP_BARS,
)
from features import build_features
from ebm_discovery import SimpleEBM
from validation import split_discovery_validation, holm_step_down
from tests import validate_candidates, DISCOVERY_FRAC

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sp500_m15.csv"
OUT_PATH = ROOT / "results" / "run12_walk_forward.json"

FOLDS = [
    {
        "name": "fold_A",
        "train_end": pd.Timestamp("2023-01-01", tz="UTC"),
        "test_end": pd.Timestamp("2025-01-01", tz="UTC"),
    },
    {
        "name": "fold_B",
        "train_end": pd.Timestamp("2025-01-01", tz="UTC"),
        "test_end": pd.Timestamp("2026-09-01", tz="UTC"),
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
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("M15 timestamps must be unique and chronological.")
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

def contiguous_target_mask(index):
    idx = pd.DatetimeIndex(index)
    future = pd.Series(idx, index=idx).shift(-1)
    return ((future - pd.Series(idx, index=idx)) == pd.Timedelta(hours=4)).fillna(False).to_numpy()

def discover(train_feat, train_target, seed):
    fd, yd, fv, yv = split_discovery_validation(
        train_feat, train_target, frac=DISCOVERY_FRAC, purge=PURGE_GAP_BARS
    )
    model = SimpleEBM(
        outer_bags=OUTER_BAGS,
        bag_frac=BAG_SAMPLE_FRAC,
        boost_rounds=BOOST_ROUNDS,
        max_depth=TREE_MAX_DEPTH,
        top_features_for_pairs=TOP_FEATURES_FOR_PAIRS,
        max_interactions=MAX_INTERACTIONS,
        seed=seed,
    )
    model.fit(fd, yd)
    candidates = model.generate_candidate_hypotheses(
        fd, yd,
        min_samples=max(MIN_SAMPLES_FOR_EDGE, 250),
        effect_thresh=EFFECT_SIZE_THRESH,
        max_candidates=100,
    )
    validated = validate_candidates(
        fv, yv, candidates,
        block_len=max(16, BLOCK_LENGTH // 4),
        n_boot=N_BOOTSTRAP,
        seed=seed,
    )
    pvals = [v["boot_p_value"] for v in validated]
    reject, cutoff = holm_step_down(pvals, alpha=ALPHA_FINAL)
    survivors = [validated[i] for i in range(len(validated)) if bool(reject[i])]
    return {
        "n_discovery": int(len(fd)),
        "n_internal_validation": int(len(fv)),
        "n_candidates": int(len(candidates)),
        "n_survivors": int(len(survivors)),
        "holm_cutoff": float(cutoff),
        "survivors": survivors,
    }

def rule_mask(feat, rule):
    if rule["type"] == "2way":
        return (
            (feat[rule["features"][0]] > rule["thresholds"][0]) &
            (feat[rule["features"][1]] < rule["thresholds"][1])
        )
    return feat[rule["features"][0]] < rule["thresholds"][0]

def evaluate_frozen(test_feat, test_target, rules):
    out = []
    for rule in rules:
        cond = rule_mask(test_feat, rule).fillna(False)
        y = test_target[cond].dropna()
        item = {
            "name": rule["name"],
            "direction": rule.get("direction", "long"),
            "condition_hits": int(cond.sum()),
            "rows": int(len(test_feat)),
            "condition_rate": float(cond.mean()) if len(test_feat) else 0.0,
            "n_signals": int(len(y)),
        }
        if len(y):
            item.update({
                "mean_normalized_return": float(y.mean()),
                "median_normalized_return": float(y.median()),
                "win_rate": float((y > 0).mean()),
                "sum_normalized_return": float(y.sum()),
                "first_signal": str(y.index.min()),
                "last_signal": str(y.index.max()),
            })
        out.append(item)
    return out

def main():
    m15 = load_m15(DATA_PATH)
    bars = make_4h_bars(m15)
    feat, target = build_features(bars, horizon=1, vol_window=16)
    valid = contiguous_target_mask(bars.index)
    valid = pd.Series(valid, index=bars.index).reindex(target.index).fillna(False).to_numpy()
    feat = feat.loc[valid]
    target = target.loc[valid]

    reports = []
    for i, fold in enumerate(FOLDS):
        train = feat.index < fold["train_end"]
        test = (feat.index >= fold["train_end"]) & (feat.index < fold["test_end"])
        train_feat, train_target = feat.loc[train], target.loc[train]
        test_feat, test_target = feat.loc[test], target.loc[test]

        if len(train_feat) < 5000 or len(test_feat) < 500:
            raise ValueError(
                f"{fold['name']} has insufficient rows: train={len(train_feat)}, test={len(test_feat)}"
            )

        discovery = discover(train_feat, train_target, seed=20261002 + i)
        frozen_rules = discovery["survivors"]
        test_results = evaluate_frozen(test_feat, test_target, frozen_rules)

        reports.append({
            "fold": fold["name"],
            "train_start": str(train_feat.index.min()),
            "train_end_exclusive": str(fold["train_end"]),
            "test_start": str(fold["train_end"]),
            "test_end_exclusive": str(fold["test_end"]),
            "train_rows": int(len(train_feat)),
            "test_rows": int(len(test_feat)),
            "discovery": discovery,
            "frozen_rules_tested": frozen_rules,
            "test_results": test_results,
        })
        print(json.dumps(reports[-1], indent=2, default=float))

    report = {
        "dataset": "Dukascopy S&P 500 (USA500.IDX/USD) M15 -> completed UTC 4H bars",
        "purpose": "finite chronological walk-forward validation",
        "rules_changed_after_test": False,
        "test_data_used_for_selection": False,
        "folds": reports,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))

if __name__ == "__main__":
    main()
