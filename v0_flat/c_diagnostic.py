import os, json
import numpy as np
from config_final import *
from generators import generate_adversarial_null
from features import build_features
from validation import block_wild_null_test, circular_shift_test, replication_check, split_discovery_validation
from ebm_discovery import SimpleEBM

SEEDS = [3014, 3019]

def run(seed):
    adv = generate_adversarial_null(n_bars=N_BARS, n_features_target=120, seed=seed)
    feat, y = adv["features"], adv["y_base"]
    fd, yd, fv, yv = split_discovery_validation(feat, y, frac=0.6, purge=PURGE_GAP_BARS)
    model = SimpleEBM(outer_bags=OUTER_BAGS, bag_frac=BAG_SAMPLE_FRAC,
        boost_rounds=BOOST_ROUNDS, max_depth=TREE_MAX_DEPTH,
        top_features_for_pairs=TOP_FEATURES_FOR_PAIRS,
        max_interactions=MAX_INTERACTIONS, seed=seed)
    model.fit(fd)
    candidates = model.generate_candidate_hypotheses(fd, yd, min_samples=MIN_SAMPLES_FOR_EDGE,
        effect_thresh=EFFECT_SIZE_THRESH, max_candidates=50)

    rows=[]
    for c in candidates:
        # Recompute fixed discovery rule on held-out data.
        from validation import recompute_condition
        cond = recompute_condition(c, fv)
        if int(cond.sum()) < 100:
            continue
        wild = block_wild_null_test(yv.values, cond, block_len=BLOCK_LENGTH, n_perm=N_BOOTSTRAP, seed=seed)
        shift = circular_shift_test(yv.values, cond, n_perm=N_BOOTSTRAP, seed=seed)
        repl, effects = replication_check(yv.values, cond)
        rows.append({
            "features": c["features"], "type": c["type"],
            "n_cond_val": int(cond.sum()),
            "effect": float(wild["effect"]),
            "wild_p": float(wild["p_value"]),
            "shift_p": float(shift["p_value"]),
            "replication_pass": bool(repl),
            "replication_effects": effects,
        })
    rows.sort(key=lambda x: x["wild_p"])
    return {"seed":seed, "n_candidates":len(candidates), "n_tested":len(rows), "top":rows[:15]}

out={"seeds":[run(s) for s in SEEDS]}
os.makedirs("results", exist_ok=True)
with open("results/c_diagnostic_3014_3019.json","w") as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
