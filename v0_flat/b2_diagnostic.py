"""Fast targeted B2 discovery-stage diagnostic.

Traces the exact 10 seeds from the failing B2 run through discovery only.
Validation/bootstrap is omitted because the diagnostic question is where the
true pair is lost before validation.
"""
import json

import config_final as CFG
from generators import generate_injected_market
from tests import run_discovery_on_dataset, DISCOVERY_FRAC
from validation import split_discovery_validation

LOCKED_B2_SEEDS = [
    88117, 91751, 65440, 93810, 24107,
    33084, 29450, 31025, 60097, 97094,
]

def main():
    rows = []
    for j, seed in enumerate(LOCKED_B2_SEEDS):
        injected = generate_injected_market(
            n_bars=CFG.N_BARS,
            effect_type="2way",
            effect_size=0.3,
            seed=seed,
        )
        feat = injected["features"]
        y = injected["y_injected"]
        xd, yd, _, _ = split_discovery_validation(
            feat, y, frac=DISCOVERY_FRAC, purge=CFG.PURGE_GAP_BARS
        )
        model, candidates = run_discovery_on_dataset(xd, yd, seed=seed)
        diag = model.diagnose_pair("vol_expansion", "pct_rank_100", candidates)
        candidate_hit = any(
            set(c.get("features", [])) >= {"vol_expansion", "pct_rank_100"}
            for c in candidates
        )
        row = {
            "j": j,
            "seed": seed,
            "n_candidates": len(candidates),
            "candidate_hit": candidate_hit,
            "diagnostic": diag,
        }
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)

    out = {
        "effect_type": "2way",
        "effect_size": 0.3,
        "n_markets": len(rows),
        "validation_omitted": True,
        "rows": rows,
    }
    out_path = CFG.RESULTS_DIR / "b2_diagnostic_03.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print("\nWrote", out_path)

if __name__ == "__main__":
    main()
