"""
Direct 4H discovery on real Dukascopy XAUUSD.

This is intentionally separate from the M15 discovery funnel: the discovery
features and target are both defined on completed, UTC-aligned 4H bars.
A 4H bar is built only when all 16 underlying M15 bars are present and
strictly consecutive. The target is the next contiguous 4H close-to-close
return, so the model discovers rules at the same decision frequency the
strategy will actually trade.
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
DATA_PATH = ROOT / "data" / "xauusd_m15.csv"
OUT_PATH = ROOT / "results" / "run10_direct_4h_discovery.json"


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
    # M15 timestamps are bar-open times. A completed 4H candle beginning at
    # 00:00 is represented at its decision time 04:00.
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
    out = pd.DataFrame(rows).set_index("timestamp").sort_index()
    if out.empty:
        raise ValueError("No complete weekday 4H bars were constructed.")
    return out


def apply_contiguous_target(feat, target, index):
    idx = pd.DatetimeIndex(index)
    future = pd.Series(idx, index=idx).shift(-1)
    valid = ((future - pd.Series(idx, index=idx)) == pd.Timedelta(hours=4))
    valid = valid.reindex(target.index).fillna(False).to_numpy()
    return feat.loc[valid], target.loc[valid], int(valid.sum())


def main():
    m15 = load_m15(DATA_PATH)
    bars4 = make_4h_bars(m15)

    # One completed 4H bar ahead = the actual next 4H decision interval.
    feat, target = build_features(bars4, horizon=1, vol_window=16)
    feat, target, n_contig = apply_contiguous_target(feat, target, bars4.index)

    if len(feat) < 5000:
        raise ValueError(f"Too few usable 4H rows: {len(feat)}")

    fd, yd, fv, yv = split_discovery_validation(
        feat, target, frac=DISCOVERY_FRAC, purge=PURGE_GAP_BARS
    )

    model = SimpleEBM(
        outer_bags=OUTER_BAGS,
        bag_frac=BAG_SAMPLE_FRAC,
        boost_rounds=BOOST_ROUNDS,
        max_depth=TREE_MAX_DEPTH,
        top_features_for_pairs=TOP_FEATURES_FOR_PAIRS,
        max_interactions=MAX_INTERACTIONS,
        seed=20261002,
    )
    model.fit(fd, yd)

    # Give the direct 4H search a larger but still bounded hypothesis funnel.
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
        seed=20261002,
    )
    pvals = [v["boot_p_value"] for v in validated]
    reject, cutoff = holm_step_down(pvals, alpha=ALPHA_FINAL)
    survivors = [validated[i] for i in range(len(validated)) if bool(reject[i])]

    # A compact view of the strongest held-out candidates, including rules
    # that do not survive Holm, so we can diagnose whether discovery is barren.
    top = sorted(
        validated,
        key=lambda x: (float(x.get("boot_p_value", 1.0)),
                        -abs(float(x.get("boot_effect", 0.0)))),
    )[:30]

    report = {
        "dataset": "Dukascopy XAUUSD M15 -> direct completed 4H bars",
        "raw_m15_rows": int(len(m15)),
        "complete_weekday_4h_bars": int(len(bars4)),
        "contiguous_4h_target_rows": int(n_contig),
        "feature_rows": int(len(feat)),
        "discovery_rows": int(len(fd)),
        "validation_rows": int(len(fv)),
        "split": "60/40 chronological with purge",
        "decision_time": "UTC 4H boundaries; feature state is completed prior 4H candle",
        "target": "next contiguous 4H close-to-close normalized return",
        "n_candidates": int(len(candidates)),
        "n_survivors_holm": int(len(survivors)),
        "holm_cutoff": float(cutoff),
        "survivors": survivors,
        "top_validation_candidates": top,
        "top_discovery_features": [
            {"feature": n, "importance": float(v), "index": int(i)}
            for n, v, i in model.get_top_features(20)
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
