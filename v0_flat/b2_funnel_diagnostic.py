import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config_final as CFG
from generators import generate_injected_market
from tests import run_discovery_on_dataset, DISCOVERY_FRAC
from validation import split_discovery_validation

def main():
    rows = []
    effect_size = 0.3
    for j in range(10):
        seed = 2000 + abs(hash(("2way", effect_size, j))) % 100000
        injected = generate_injected_market(
            n_bars=CFG.N_BARS,
            effect_type="2way",
            effect_size=effect_size,
            seed=seed,
        )
        feat = injected["features"]
        y = injected["y_injected"]
        feat_disc, y_disc, _, _ = split_discovery_validation(
            feat, y, frac=DISCOVERY_FRAC, purge=CFG.PURGE_GAP_BARS
        )
        model, candidates = run_discovery_on_dataset(feat_disc, y_disc, seed=seed)
        diag = model.diagnose_pair("vol_expansion", "pct_rank_100", candidates)
        rows.append({
            "j": j,
            "seed": seed,
            "n_candidates": len(candidates),
            "n_two_way": sum(c.get("type") == "2way" for c in candidates),
            "true_pair_in_final_candidates": diag.get("final_candidate") is not None,
            "true_pair_final_rank": diag.get("final_candidate_rank"),
            "true_pair_purified_rank": diag.get("purified_rank"),
            "purified_pairs_total": diag.get("purified_pairs_total"),
            "generated_two_way_candidate_count": diag.get("generated_two_way_candidate_count"),
            "generated_two_way_rank": diag.get("generated_two_way_rank"),
        })
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    summary = {
        "effect_size": effect_size,
        "n_seeds": len(rows),
        "candidate_budget": CFG.MAX_DISCOVERY_CANDIDATES,
        "one_way_fraction": CFG.ONE_WAY_CANDIDATE_FRACTION,
        "two_way_budget": CFG.MAX_DISCOVERY_CANDIDATES - int(round(CFG.MAX_DISCOVERY_CANDIDATES * CFG.ONE_WAY_CANDIDATE_FRACTION)),
        "pair_recall": sum(r["true_pair_in_final_candidates"] for r in rows) / len(rows),
        "rows": rows,
    }
    out = ROOT / "results" / "b2_funnel_diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
