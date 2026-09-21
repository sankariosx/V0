"""Targeted B2 diagnostic: 10 markets at the locked 0.3 effect size.

This is intentionally NOT the 170-job validation. It traces the known injected
pair (vol_expansion, pct_rank_100) through every discovery stage so we can make
one evidence-based fix instead of repeatedly changing the search algorithm.
"""
import os
import sys
os.environ.setdefault("PYTHONHASHSEED", "0")

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config_final as CFG
from generators import generate_injected_market
from tests import run_discovery_on_dataset, validate_candidates, DISCOVERY_FRAC
from validation import split_discovery_validation, holm_step_down

def main():
    rows = []
    seed_base = 2000
    for j in range(CFG.TEST_B_N_MARKETS_PER_SIZE):
        effect_type = "2way"
        eff_size = 0.3
        seed = seed_base + abs(hash((effect_type, eff_size, j))) % 100000
        injected = generate_injected_market(
            n_bars=CFG.N_BARS,
            effect_type=effect_type,
            effect_size=eff_size,
            seed=seed,
        )
        feat = injected["features"]
        y = injected["y_injected"]
        xd, yd, xv, yv = split_discovery_validation(
            feat, y, frac=DISCOVERY_FRAC, purge=CFG.PURGE_GAP_BARS
        )
        model, candidates = run_discovery_on_dataset(xd, yd, seed=seed)
        validated = validate_candidates(xv, yv, candidates, seed=seed) if candidates else []
        if validated:
            pvals = [v["boot_p_value"] for v in validated]
            reject, _ = holm_step_down(pvals, alpha=CFG.ALPHA_FINAL)
            surviving = [validated[i] for i in range(len(validated)) if reject[i]]
        else:
            surviving = []
        diag = model.diagnose_pair("vol_expansion", "pct_rank_100", candidates)
        rows.append({
            "j": j,
            "seed": seed,
            "n_candidates": len(candidates),
            "n_surviving": len(surviving),
            "detected": any(
                set(v.get("features", [])) >= {"vol_expansion", "pct_rank_100"}
                for v in surviving
            ),
            "diagnostic": diag,
        })
        print(json.dumps(rows[-1], default=str), flush=True)

    out = {
        "effect_type": "2way",
        "effect_size": 0.3,
        "n_markets": len(rows),
        "rows": rows,
    }
    out_path = CFG.RESULTS_DIR / "b2_diagnostic_03.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print("\nWrote", out_path)

if __name__ == "__main__":
    main()
