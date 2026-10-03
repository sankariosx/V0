"""
Deterministic test of the user's XAUUSD candle-pattern edge.

Tests the identical rules independently on D1, 4H, 1H and 15M.
No discovery, ML, optimization, threshold fitting, or V0 candidate selection.

Important:
- One setup at a time. While a setup is active, later setups are ignored.
- Target/failure are touch-based.
- If both target and failure are touched inside the same post-signal candle,
  intrabar order is unknowable from OHLC, so the outcome is recorded as
  ambiguous and excluded from win/loss rates.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "xauusd_m15.csv"
OUT_PATH = ROOT / "results" / "user_edge_xauusd_timeframes.json"

TIMEFRAMES = {
    "D1": "1D",
    "4H": "4h",
    "1H": "1h",
    "15M": "15min",
}

def load_m15(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    ts = "date" if "date" in df.columns else "datetime" if "datetime" in df.columns else "timestamp"
    if ts == "timestamp" and pd.api.types.is_numeric_dtype(df[ts]):
        df[ts] = pd.to_datetime(df[ts], unit="ms", utc=True)
    else:
        df[ts] = pd.to_datetime(df[ts], utc=True)
    df = df.set_index(ts).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df

def aggregate(m15, tf):
    if tf == "15min":
        return m15[["open","high","low","close"]].copy()
    rule = m15.resample(tf, label="right", closed="right", origin="epoch").agg(
        open=("open","first"),
        high=("high","max"),
        low=("low","min"),
        close=("close","last"),
        n=("close","count"),
    )
    expected = {"1D": 96, "4h": 16, "1h": 4}[tf]
    rule = rule[rule["n"] == expected].drop(columns="n")
    return rule

def add_ema(df):
    out = df.copy()
    out["ema21"] = out["close"].ewm(span=21, adjust=False, min_periods=21).mean()
    return out

def green(r): return r["close"] > r["open"]
def red(r): return r["close"] < r["open"]

def finish_record(direction, start_i, start_ts, signal_i, signal_ts, entry,
                  target, failure, outcome, outcome_i, outcome_ts, ambiguous=False):
    bars_to_signal = signal_i - start_i
    bars_to_outcome = None if outcome_i is None else outcome_i - signal_i
    # Directional R: risk must be measured from entry to the failure
    # level in the direction of the trade, and reward from entry to target.
    if ambiguous or outcome not in ("success", "failure"):
        r_value = None
    elif direction == "bull":
        risk = entry - failure
        reward = target - entry
        if risk > 0:
            r_value = (reward / risk) if reward > 0 else None
            if outcome == "failure":
                r_value = -1.0
        else:
            r_value = None
    else:
        risk = failure - entry
        reward = entry - target
        if risk > 0:
            r_value = (reward / risk) if reward > 0 else None
            if outcome == "failure":
                r_value = -1.0
        else:
            r_value = None

    return {
        "direction": direction,
        "start": str(start_ts),
        "signal": str(signal_ts),
        "entry": float(entry),
        "target": float(target),
        "failure_level": float(failure),
        "R": r_value,
        "outcome": outcome,
        "ambiguous_same_bar": bool(ambiguous),
        "bars_to_signal": int(bars_to_signal),
        "bars_to_outcome": None if bars_to_outcome is None else int(bars_to_outcome),
        "outcome_time": None if outcome_ts is None else str(outcome_ts),
        "immediate_success": bool(outcome == "success" and outcome_i == signal_i),
        "delayed_success": bool(outcome == "success" and outcome_i is not None and outcome_i > signal_i),
    }

def scan(df, direction):
    df = df.reset_index().rename(columns={"index":"timestamp"})
    rows = df.to_dict("records")
    n = len(rows)
    results = []
    i = 0

    while i <= n - 7:
        # Initial trend sequence.
        ok = True
        for k in range(4):
            r = rows[i+k]
            if direction == "bull":
                ok &= green(r) and r["close"] > r["ema21"]
            else:
                ok &= red(r) and r["close"] < r["ema21"]
        if not ok:
            i += 1
            continue

        # Immediate first pullback candle and immediate failed attempt.
        r1 = rows[i+4]
        g5 = rows[i+5]
        if direction == "bull":
            if not red(r1) or not green(g5) or g5["high"] >= rows[i+3]["high"]:
                i += 1
                continue
            target = rows[i+3]["high"]
            break_level = min(r1["low"], g5["low"])
        else:
            if not green(r1) or not red(g5) or g5["low"] <= rows[i+3]["low"]:
                i += 1
                continue
            target = rows[i+3]["low"]
            break_level = max(r1["high"], g5["high"])

        # Wait indefinitely for the first break. Any candle colors are allowed,
        # but the actual breaking candle must be opposite-colored.
        j = i + 6
        break_i = None
        while j < n:
            r = rows[j]
            if direction == "bull":
                if red(r) and r["low"] <= break_level:
                    break_i = j
                    break
            else:
                if green(r) and r["high"] >= break_level:
                    break_i = j
                    break
            j += 1

        if break_i is None:
            # No completed setup outcome; this unfinished setup blocks later
            # setups, matching the user's "stay with the original one" rule.
            break

        # After the break, wait for the signal candle. For bullish setup the
        # signal is the first green; bearish is the first red.
        signal_i = break_i + 1
        if signal_i >= n:
            break
        while signal_i < n:
            r = rows[signal_i]
            if (direction == "bull" and green(r)) or (direction == "bear" and red(r)):
                break
            signal_i += 1

        if signal_i >= n:
            break

        # Second-attempt failure level = extreme from break through the last
        # candle before the signal.
        pre_signal = rows[break_i:signal_i]
        if direction == "bull":
            failure = min(r["low"] for r in pre_signal)
        else:
            failure = max(r["high"] for r in pre_signal)

        # Race target vs failure from the signal candle onward.
        outcome = None
        outcome_i = None
        outcome_ts = None
        ambiguous = False
        k = signal_i
        while k < n:
            r = rows[k]
            if direction == "bull":
                hit_target = r["high"] >= target
                hit_failure = r["low"] <= failure
            else:
                hit_target = r["low"] <= target
                hit_failure = r["high"] >= failure

            if hit_target and hit_failure:
                outcome = "ambiguous_same_bar"
                ambiguous = True
                outcome_i = k
                outcome_ts = r["timestamp"]
                break
            if hit_target:
                outcome = "success"
                outcome_i = k
                outcome_ts = r["timestamp"]
                break
            if hit_failure:
                outcome = "failure"
                outcome_i = k
                outcome_ts = r["timestamp"]
                break
            k += 1

        # Entry convention: signal candle OPEN. This allows an immediate
        # same-candle target touch to count as a captured success.
        entry = rows[signal_i]["open"]
        results.append(finish_record(
            direction, i, rows[i]["timestamp"], signal_i, rows[signal_i]["timestamp"],
            entry, target, failure, outcome, outcome_i, outcome_ts, ambiguous
        ))

        # Current setup is finished. Resume scanning only after its outcome.
        if outcome_i is None:
            break
        i = outcome_i + 1

    return results

def scan_one_setup_at_a_time(df):
    """Scan both directions chronologically, enforcing one active setup globally."""
    df = df.reset_index().rename(columns={"index": "timestamp"})
    rows = df.to_dict("records")
    n = len(rows)
    results = []
    i = 0

    while i <= n - 7:
        direction = None
        for candidate in ("bull", "bear"):
            ok = True
            for k in range(4):
                r = rows[i+k]
                if candidate == "bull":
                    ok &= green(r) and r["close"] > r["ema21"]
                else:
                    ok &= red(r) and r["close"] < r["ema21"]
            if not ok:
                continue
            r1, g5 = rows[i+4], rows[i+5]
            if candidate == "bull":
                valid = red(r1) and green(g5) and g5["high"] < rows[i+3]["high"]
            else:
                valid = green(r1) and red(g5) and g5["low"] > rows[i+3]["low"]
            if valid:
                direction = candidate
                break

        if direction is None:
            i += 1
            continue

        one = scan(df.iloc[i:].copy(), direction)
        if not one:
            break
        result = one[0]
        results.append(result)

        if result["bars_to_outcome"] is None:
            break
        i = i + result["bars_to_signal"] + result["bars_to_outcome"] + 1

    return results


def summarize(results):
    total = len(results)
    wins = sum(x["outcome"] == "success" for x in results)
    losses = sum(x["outcome"] == "failure" for x in results)
    ambiguous = sum(x["outcome"] == "ambiguous_same_bar" for x in results)
    resolved = wins + losses
    immediate = sum(x["immediate_success"] for x in results)
    delayed = sum(x["delayed_success"] for x in results)
    rvals = [x["R"] for x in results if x["R"] is not None]
    mean_r = float(np.mean(rvals)) if rvals else None
    total_r = float(np.sum(rvals)) if rvals else None
    gross_profit = float(np.sum([r for r in rvals if r > 0])) if rvals else 0.0
    gross_loss = float(-np.sum([r for r in rvals if r < 0])) if rvals else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None
    invalid_r = sum(x["outcome"] in ("success", "failure") and x["R"] is None for x in results)
    r_sample = len(rvals)
    def mean(key):
        vals = [x[key] for x in results if x[key] is not None]
        return float(np.mean(vals)) if vals else None
    return {
        "setups": total,
        "wins": wins,
        "losses": losses,
        "ambiguous_same_bar": ambiguous,
        "resolved_setups": resolved,
        "success_rate_resolved": (wins / resolved) if resolved else None,
        "success_rate_all_setups": (wins / total) if total else None,
        "immediate_successes": immediate,
        "delayed_successes": delayed,
        "total_R": total_r,
        "R_calculated_setups": r_sample,
        "R_excluded_invalid_or_ambiguous": resolved - r_sample,
        "expectancy_R_per_R_calculated_setup": mean_r,
        "mean_win_R": (float(np.mean([x["R"] for x in results if x["R"] is not None and x["R"] > 0]))
                       if any(x["R"] is not None and x["R"] > 0 for x in results) else None),
        "gross_profit_R": gross_profit,
        "gross_loss_R": gross_loss,
        "profit_factor": profit_factor,
        "invalid_directional_R_cases": invalid_r,
        "mean_bars_to_signal": mean("bars_to_signal"),
        "mean_bars_to_outcome": mean("bars_to_outcome"),
    }

def main():
    m15 = load_m15(DATA_PATH)
    report = {
        "instrument": "XAUUSD",
        "source": "Dukascopy M15",
        "rules_fixed": True,
        "one_setup_at_a_time": True,
        "ema": "21 EMA on the tested timeframe",
        "R_entry": "signal candle open",
        "timeframes": {},
    }

    for name, tf in TIMEFRAMES.items():
        bars = add_ema(aggregate(m15, tf))
        # Only completed bars with an EMA are eligible.
        bars = bars.dropna(subset=["ema21"])
        all_results = scan_one_setup_at_a_time(bars)
        bull = [x for x in all_results if x["direction"] == "bull"]
        bear = [x for x in all_results if x["direction"] == "bear"]
        report["timeframes"][name] = {
            "bars": int(len(bars)),
            "start": str(bars.index.min()) if len(bars) else None,
            "end": str(bars.index.max()) if len(bars) else None,
            "bullish": summarize(bull),
            "bearish": summarize(bear),
            "combined": summarize(all_results),
            "setups": all_results,
        }
        print(name, json.dumps(report["timeframes"][name], indent=2, default=str))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print("\nREPORT:", OUT_PATH)

if __name__ == "__main__":
    main()
