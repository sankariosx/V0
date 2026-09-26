"""
Real-data V0 discovery runner for Dukascopy XAUUSD M15.

This deliberately reuses the validated V0 discovery + held-out validation
functions. It does NOT modify the synthetic validation benchmark.

Real-data-specific handling:
- normalizes Dukascopy CSV column names
- requires chronological, unique timestamps
- target horizon is 16 actual consecutive 15-minute bars (4 clock hours)
- rows whose t+16 timestamp is not exactly t+4h are excluded from the target
- uses the same 60/20 time-ordered split, purge gap, EBM, candidate generation,
  Holm FWER validation, and replication gate as validated V0
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from config_final import HORIZON, VOL_WINDOW, PURGE_GAP_BARS, BLOCK_LENGTH
from config_final import OUTER_BAGS, BAG_SAMPLE_FRAC, BOOST_ROUNDS, TREE_MAX_DEPTH
from config_final import TOP_FEATURES_FOR_PAIRS, MAX_INTERACTIONS
from config_final import MIN_SAMPLES_FOR_EDGE, EFFECT_SIZE_THRESH, N_BOOTSTRAP
from config_final import ALPHA_FINAL

from features import build_features
from ebm_discovery import SimpleEBM
from validation import split_discovery_validation, holm_step_down
from tests import validate_candidates, DISCOVERY_FRAC


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "xauusd_m15.csv"
OUT_PATH = ROOT / "results" / "real_xauusd_discovery.json"


def load_data(path):
    df = pd.read_csv(path)
    rename = {c.lower(): c.lower() for c in df.columns}
    df = df.rename(columns=rename)

    if "date" in df.columns:
        ts_col = "date"
    elif "datetime" in df.columns:
        ts_col = "datetime"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    else:
        raise ValueError(f"No timestamp column found: {list(df.columns)}")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    # Dukascopy exports timestamp as Unix milliseconds. Without unit="ms",
    # pandas interprets it as nanoseconds, collapsing 2015-2026 into hours
    # and making the exact-4h horizon mask empty.
    if ts_col == "timestamp" and pd.api.types.is_numeric_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], unit="ms", utc=True)
    else:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index(ts_col).sort_index()

    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps found.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Timestamps are not chronological.")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="raise")

    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if bad.any():
        raise ValueError(f"Invalid OHLC rows: {int(bad.sum())}")

    return df


def apply_real_horizon_mask(feat, target, raw_index):
    idx = pd.DatetimeIndex(raw_index)
    future = pd.Series(idx, index=idx).shift(-HORIZON)
    valid = (future - pd.Series(idx, index=idx)) == pd.Timedelta(minutes=15 * HORIZON)
    valid = valid.reindex(target.index).fillna(False).to_numpy()
    return feat.loc[valid], target.loc[valid], int(valid.sum())


def discover(feat, target, seed=20260926):
    feat_disc, target_disc, feat_val, target_val = split_discovery_validation(
        feat, target, frac=DISCOVERY_FRAC, purge=PURGE_GAP_BARS
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
    model.fit(feat_disc, target_disc)

    candidates = model.generate_candidate_hypotheses(
        feat_disc,
        target_disc,
        min_samples=MIN_SAMPLES_FOR_EDGE,
        effect_thresh=EFFECT_SIZE_THRESH,
        max_candidates=50,
    )

    validated = validate_candidates(
        feat_val,
        target_val,
        candidates,
        block_len=BLOCK_LENGTH,
        n_boot=N_BOOTSTRAP,
        seed=seed,
    )

    pvals = [v["boot_p_value"] for v in validated]
    reject, cutoff = holm_step_down(pvals, alpha=ALPHA_FINAL)

    survivors = [
        validated[i] for i in range(len(validated)) if bool(reject[i])
    ]

    # Keep the report compact but preserve all surviving hypotheses.
    return {
        "seed": seed,
        "n_discovery": len(feat_disc),
        "n_validation": len(feat_val),
        "n_candidates": len(candidates),
        "n_survivors": len(survivors),
        "holm_cutoff": float(cutoff),
        "survivors": survivors,
        "top_candidates": validated[:20],
    }


def main():
    df = load_data(DATA_PATH)
    feat, target = build_features(
        df, horizon=HORIZON, vol_window=VOL_WINDOW
    )

    # Controlled real-data sensitivity test: remove all clock/session features.
    # This is the ONLY methodological change versus the validated #4 runner.
    # Synthetic V0 and the core discovery/validation machinery are untouched.
    session_features = ["hour_sin", "hour_cos", "is_london", "is_ny", "is_overlap"]
    feat = feat.drop(columns=[c for c in session_features if c in feat.columns])

    # The feature builder can produce a numeric target across a market closure.
    # Re-apply the target using the original timestamp index so 16 means
    # exactly 16 consecutive M15 observations / 4 clock hours.
    feat, target, n_valid_horizon = apply_real_horizon_mask(
        feat, target, df.index
    )

    if len(feat) < 5000:
        raise ValueError(f"Too few usable rows after preprocessing: {len(feat)}")

    report = {
        "dataset": "Dukascopy XAUUSD M15",
        "source_file": DATA_PATH.name,
        "rows_raw": len(df),
        "rows_after_feature_target": len(feat),
        "rows_with_true_4h_horizon": n_valid_horizon,
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "time_gap_policy": "require exact 16 consecutive 15-minute bars",
        "horizon_bars": HORIZON,
        "horizon_minutes": 15 * HORIZON,
        "discovery_fraction": DISCOVERY_FRAC,
        "purge_bars": PURGE_GAP_BARS,
        "model": "validated V0 SimpleEBM discovery + Holm FWER held-out validation",
        "sensitivity_test": "remove UTC clock/session features only",
        "removed_features": ["hour_sin", "hour_cos", "is_london", "is_ny", "is_overlap"],
    }
    report["discovery"] = discover(feat, target)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()

# Real-data pipeline: Dukascopy XAUUSD M15.
