"""Statistical audit for the frozen XAUUSD user-edge test.

This script does not change the trading rules. It audits the already-generated
trade records from run_user_edge_xauusd.py and separates descriptive history
from the frozen 2025+ OOS holdout.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
IN_PATH = ROOT / "results" / "user_edge_xauusd_timeframes.json"
OUT_PATH = ROOT / "results" / "user_edge_xauusd_stats.json"
OOS_START = pd.Timestamp("2025-01-01", tz="UTC")
BOOTSTRAP_N = 100_000
SEED = 20261003


def r_values(records):
    return np.asarray([x["R"] for x in records if x.get("R") is not None], dtype=float)


def max_drawdown(r):
    if len(r) == 0:
        return 0.0
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(np.r_[0.0, eq])
    dd = np.r_[0.0, eq] - peak
    return float(dd.min())


def max_loss_streak(r):
    best = cur = 0
    for x in r:
        if x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def bootstrap_mean_ci(r, rng):
    if len(r) < 2:
        return {"low_95": None, "high_95": None}
    idx = rng.integers(0, len(r), size=(BOOTSTRAP_N, len(r)))
    means = r[idx].mean(axis=1)
    return {
        "low_95": float(np.quantile(means, 0.025)),
        "high_95": float(np.quantile(means, 0.975)),
    }


def sign_permutation_pvalue(r, rng):
    """Null: trade direction has no effect, conditional on observed |R|.

    This is intentionally a simple null diagnostic, not a proof of independence.
    """
    if len(r) == 0:
        return None
    observed = float(r.sum())
    abs_r = np.abs(r)
    chunk = 20_000
    extreme = 0
    done = 0
    while done < BOOTSTRAP_N:
        n = min(chunk, BOOTSTRAP_N - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(r)))
        sums = (signs * abs_r).sum(axis=1)
        extreme += int(np.sum(sums >= observed))
        done += n
    return float((extreme + 1) / (BOOTSTRAP_N + 1))


def audit(records):
    r = r_values(records)
    wins = int(np.sum(r > 0))
    losses = int(np.sum(r < 0))
    rng = np.random.default_rng(SEED)
    years = {}
    for x in records:
        if x.get("R") is None:
            continue
        y = pd.Timestamp(x["start"]).year
        years.setdefault(str(y), []).append(float(x["R"]))

    year_stats = {}
    for y, vals in sorted(years.items()):
        a = np.asarray(vals, dtype=float)
        year_stats[y] = {
            "trades": int(len(a)),
            "total_R": float(a.sum()),
            "expectancy_R": float(a.mean()),
            "win_rate": float(np.mean(a > 0)),
        }

    return {
        "trades_with_R": int(len(r)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / len(r)) if len(r) else None,
        "total_R": float(r.sum()) if len(r) else None,
        "expectancy_R": float(r.mean()) if len(r) else None,
        "gross_profit_R": float(r[r > 0].sum()) if wins else 0.0,
        "gross_loss_R": float(-r[r < 0].sum()) if losses else 0.0,
        "profit_factor": float(r[r > 0].sum() / -r[r < 0].sum()) if losses else None,
        "max_drawdown_R": max_drawdown(r),
        "max_loss_streak": max_loss_streak(r),
        "bootstrap_mean_R_95pct": bootstrap_mean_ci(r, rng),
        "sign_permutation_p_value": sign_permutation_pvalue(r, rng),
        "year_stats": year_stats,
    }


def main():
    data = json.loads(IN_PATH.read_text())
    out = {
        "instrument": data["instrument"],
        "source": data["source"],
        "rules_fixed": data.get("rules_fixed", True),
        "method": {
            "bootstrap_resamples": BOOTSTRAP_N,
            "bootstrap_seed": SEED,
            "sign_null": "randomize the sign of each observed |R| while preserving its magnitude",
            "important": "Historical statistics are descriptive and were not used to alter rules. The 2025+ OOS holdout is the decisive forward test.",
        },
        "timeframes": {},
    }

    for tf, block in data["timeframes"].items():
        records = block["setups"]
        development = [x for x in records if pd.Timestamp(x["start"]) < OOS_START]
        post = [x for x in records if pd.Timestamp(x["start"]) >= OOS_START]
        out["timeframes"][tf] = {
            "all_history": audit(records),
            "development_before_2025": audit(development),
            "post_2025_from_full_scan": audit(post),
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    for tf, x in out["timeframes"].items():
        print("\n", tf)
        for label in ("all_history", "development_before_2025", "post_2025_from_full_scan"):
            a = x[label]
            print(label, "trades=", a["trades_with_R"], "total_R=", a["total_R"],
                  "expectancy=", a["expectancy_R"], "PF=", a["profit_factor"],
                  "bootstrap95=", a["bootstrap_mean_R_95pct"],
                  "sign_p=", a["sign_permutation_p_value"])
    print("\nREPORT:", OUT_PATH)


if __name__ == "__main__":
    main()
