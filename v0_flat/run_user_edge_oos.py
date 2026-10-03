"""Frozen out-of-sample test of the user's XAUUSD edge."""
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_user_edge_xauusd import load_m15, aggregate, add_ema, scan_one_setup_at_a_time, summarize, TIMEFRAMES, DATA_PATH

OOS_START = "2025-01-01T00:00:00+00:00"
OUT_PATH = Path("v0_flat/results/user_edge_xauusd_oos.json")

def main():
    m15 = load_m15(DATA_PATH)
    cutoff = pd.Timestamp(OOS_START)
    report = {
        "instrument": "XAUUSD",
        "source": "Dukascopy M15",
        "rules_frozen": True,
        "oos_start": OOS_START,
        "oos_definition": "Only setups whose initial 4-candle sequence starts on/after OOS_START are included. No rule fitting or optimization.",
        "ema": "21 EMA on the tested timeframe, calculated with full prior history",
        "R_entry": "signal candle open",
        "timeframes": {},
    }
    for name, tf in TIMEFRAMES.items():
        full_bars = add_ema(aggregate(m15, tf)).dropna(subset=["ema21"])
        # Run the frozen one-setup-at-a-time state machine from the full
        # history, then select only setups whose initial sequence starts on
        # or after the OOS cutoff. This preserves the real historical state
        # and prevents the OOS window from resetting an already-active setup.
        full_results, blocked = scan_one_setup_at_a_time(full_bars)
        all_results = [x for x in full_results if pd.Timestamp(x["start"]) >= cutoff]
        bars = full_bars[full_bars.index >= cutoff]
        bull = [x for x in all_results if x["direction"] == "bull"]
        bear = [x for x in all_results if x["direction"] == "bear"]
        report["timeframes"][name] = {
            "bars_in_oos": int(len(bars)),
            "start": str(bars.index.min()) if len(bars) else None,
            "end": str(bars.index.max()) if len(bars) else None,
            "bullish": summarize(bull),
            "bearish": summarize(bear),
            "combined": summarize(all_results),
            "setups": all_results,
            "full_scan_blocked_unresolved_setup": blocked,
        }
        print(name, json.dumps(report["timeframes"][name], indent=2, default=str))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print("\nREPORT:", OUT_PATH)

if __name__ == "__main__":
    main()
