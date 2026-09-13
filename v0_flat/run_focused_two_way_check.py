"""Focused smoke test for the directional two-way interaction repair.

This is deliberately not the full V0 validation. It replays four fixed,
previously observed two-way cases at effect size 0.3: two that passed and two
that failed before the repair. It uses the production worker unchanged so the
only variable is the interaction-ranking repair.
"""
import json
import sys
from pathlib import Path
from multiprocessing import Pool

from parallel_runner import _worker_test_B

CASES = [
    {"label": "previous_hit_0", "effect_type": "2way", "eff_size": 0.3, "j": 0, "seed": 7278},
    {"label": "previous_miss_1", "effect_type": "2way", "eff_size": 0.3, "j": 1, "seed": 37575},
    {"label": "previous_miss_2", "effect_type": "2way", "eff_size": 0.3, "j": 2, "seed": 19616},
    {"label": "previous_hit_9", "effect_type": "2way", "eff_size": 0.3, "j": 9, "seed": 51270},
]


def run_case(case):
    result = _worker_test_B(case)
    return {
        "label": case["label"],
        "seed": case["seed"],
        "detected": result["detected"],
        "n_candidates": result["n_candidates"],
        "n_surviving": result["n_surviving"],
        "true_info": result["true_info"],
        "surviving": result["surviving"],
    }


if __name__ == "__main__":
    with Pool(processes=2) as pool:
        results = pool.map(run_case, CASES)

    recovered_misses = sum(
        row["detected"] for row in results if row["label"].startswith("previous_miss")
    )
    summary = {
        "status": "complete",
        "purpose": "Focused four-case check of directional two-way interaction ranking",
        "cases": results,
        "previously_missed_cases_recovered": recovered_misses,
        "repair_signal": recovered_misses == 2,
    }
    out = Path(__file__).parent / "results" / "focused_two_way_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    if recovered_misses != 2:
        print("Focused repair check did not recover both prior misses.", file=sys.stderr)
        sys.exit(1)
