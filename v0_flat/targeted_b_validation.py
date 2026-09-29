import numpy as np
from generators import generate_injected_market
from validation import split_discovery_validation
from tests import run_discovery_on_dataset, validate_candidates

for effect_type, seed in (("1way", 2001), ("2way", 3001)):
    injected = generate_injected_market(n_bars=12000, effect_type=effect_type, effect_size=0.3, seed=seed)
    feat, y = injected["features"], injected["y_injected"]
    fd, yd, fv, yv = split_discovery_validation(feat, y, frac=0.6, purge=16)
    model, candidates = run_discovery_on_dataset(fd, yd, seed=seed)

    print(f"\n=== DIAG {effect_type} ===")
    print("candidates:", len(candidates))
    print("top_features:", model.get_top_features(20))
    print("top_interactions:", model.get_top_interactions(20))

    target = "ret_16" if effect_type == "1way" else "vol_expansion"
    ti = model.feature_names_.index(target)
    imp = model.feature_importances_[ti]
    rank = int(np.sum(model.feature_importances_ > imp)) + 1
    print(f"target {target}: importance={imp:.8f} rank={rank}")

    if effect_type == "2way":
        print("pair_diagnostic:", model.diagnose_pair("vol_expansion", "pct_rank_100", candidates))

    if candidates:
        validated = validate_candidates(fv, yv, candidates, n_boot=100, seed=seed)
        print("validated_top:", [
            (v["features"], round(v["boot_effect"], 4), round(v["boot_p_value"], 4), v["n_cond_val"])
            for v in validated[:15]
        ])

print("\nDISCOVERY DIAGNOSTIC COMPLETE")
