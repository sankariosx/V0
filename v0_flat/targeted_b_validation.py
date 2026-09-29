import numpy as np
from generators import generate_injected_market
from validation import split_discovery_validation, replication_check
from config_final import N_BARS, PURGE_GAP_BARS

def preflight(effect_type, seed):
    injected = generate_injected_market(n_bars=N_BARS, effect_type=effect_type, effect_size=0.3, seed=seed)
    feat, y = injected["features"], injected["y_injected"]
    _, _, feat_val, y_val = split_discovery_validation(feat, y, frac=0.6, purge=PURGE_GAP_BARS)
    true_cond = injected["condition"].loc[feat_val.index].to_numpy(dtype=bool)
    effect = float(np.mean(y_val.to_numpy()[true_cond]) - np.mean(y_val.to_numpy()))
    replicated, effects = replication_check(y_val.to_numpy(), true_cond)
    print(f"PRECHECK {effect_type} seed={seed}: n_val={len(y_val)} n_cond={int(true_cond.sum())} effect={effect:.4f} replicated={replicated} blocks={[round(x,4) for x in effects]}")
    return effect, replicated

for effect_type, seeds in (("1way", (2001, 2002)), ("2way", (3001, 3002))):
    for seed in seeds:
        preflight(effect_type, seed)

print("PREFLIGHT COMPLETE")
