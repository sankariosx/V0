import numpy as np
from generators import generate_injected_market
from validation import split_discovery_validation
from tests import run_discovery_on_dataset, validate_candidates

def diagnose(effect_type, seed):
    injected = generate_injected_market(n_bars=12000, effect_type=effect_type, effect_size=0.3, seed=seed)
    feat, y = injected["features"], injected["y_injected"]
    fd, yd, fv, yv = split_discovery_validation(feat, y, frac=0.6, purge=16)
    model, candidates = run_discovery_on_dataset(fd, yd, seed=seed)
    print(f"DIAG {effect_type}: candidates={len(candidates)}")
    print("TOP FEATURES:", model.get_top_features(20))
    print("TOP INTERACTIONS:", model.get_top_interactions(20))
    target = "ret_16" if effect_type=="1way" else "vol_expansion"
    if target in model.feature_names_:
        ti=model.feature_names_.index(target)
        print(f"TARGET {target}: importance={model.feature_importances_[ti]:.8f} rank={int(np.sum(model.feature_importances_ > model.feature_importances_[ti]))+1}")
    if effect_type=="2way":
        print("PAIR DIAG:", model.diagnose_pair("vol_expansion","pct_rank_100",candidates))
    if candidates:
        validated=validate_candidates(fv,yv,candidates,n_boot=100,seed=seed)
        print("VALIDATED TOP:", [(v["features"], round(v["boot_effect"],4), round(v["boot_p_value"],4), v["n_cond_val"]) for v in validated[:10]])

diagnose("1way", 2001)
diagnose("2way", 3001)
print("DISCOVERY DIAGNOSTIC COMPLETE")
