import json
import os

import numpy as np

from config_final import *
from generators import generate_adversarial_null
from validation import (
    block_wild_null_test,
    circular_shift_test,
    holm_step_down,
    replication_check,
    recompute_condition,
    split_discovery_validation,
)
from ebm_discovery import SimpleEBM

# These are the ONLY two Test-C false-positive markets from the completed
# 170-job run. This diagnostic does not change production validation logic.
SEEDS = [3014, 3019]


def run(seed):
    adv = generate_adversarial_null(
        n_bars=N_BARS,
        n_features_target=TEST_C_N_FEATURES,
        seed=seed,
    )
    feat, y = adv["features"], adv["y_base"]

    fd, yd, fv, yv = split_discovery_validation(
        feat, y, frac=0.6, purge=PURGE_GAP_BARS
    )

    model = SimpleEBM(
        outer_bags=OUTER_BAGS,
        bag_frac=BAG_SAMPLE_FRAC,
        boost_rounds=BOOST_ROUNDS,
        max_depth=TREE_MAX_DEPTH,
        top_features_for_pairs=TOP_FEATURES_FOR_PAIRS,
        max_interactions=MAX_INTERACTIONS,
        seed=seed,
    )
    model.fit(fd, yd)
    candidates = model.generate_candidate_hypotheses(
        fd,
        yd,
        min_samples=MIN_SAMPLES_FOR_EDGE,
        effect_thresh=EFFECT_SIZE_THRESH,
        max_candidates=50,
    )

    rows = []
    for rank, c in enumerate(candidates, 1):
        cond = recompute_condition(c, fv)
        n_cond = int(cond.sum())
        if n_cond < 100:
            continue

        wild = block_wild_null_test(
            yv.values,
            cond,
            block_len=BLOCK_LENGTH,
            n_perm=N_BOOTSTRAP,
            seed=seed,
        )
        shift = circular_shift_test(
            yv.values,
            cond,
            n_perm=N_BOOTSTRAP,
            seed=seed,
        )
        repl, effects = replication_check(yv.values, cond)

        # This is the exact final Test-C gate used by tests.py:
        # block-wild p-value + Holm FWER + replication hard gate.
        original_p = float(wild["p_value"]) if repl else 1.0

        rows.append(
            {
                "candidate_rank": rank,
                "features": c["features"],
                "type": c["type"],
                "condition_str": c.get("condition_str"),
                "n_cond_val": n_cond,
                "validation_rate": float(n_cond / len(fv)),
                "discovery_effect": float(c.get("effect_size", 0.0)),
                "discovery_stability": float(c.get("stability", 0.0)),
                "discovery_importance": float(c.get("importance", 0.0)),
                "effect_val": float(wild["effect"]),
                "wild_p": float(wild["p_value"]),
                "shift_p": float(shift["p_value"]),
                "replication_pass": bool(repl),
                "replication_effects": effects,
                "original_final_p": original_p,
            }
        )

    # Reproduce the exact multiple-testing decision for these candidates.
    pvals = [r["original_final_p"] for r in rows]
    reject, holm_cutoff = holm_step_down(pvals, alpha=ALPHA_FINAL)
    for r, is_rejected in zip(rows, reject.tolist()):
        r["original_test_c_survives"] = bool(is_rejected)

    rows.sort(
        key=lambda x: (
            not x["original_test_c_survives"],
            x["original_final_p"],
            x["candidate_rank"],
        )
    )

    survivors = [r for r in rows if r["original_test_c_survives"]]

    return {
        "seed": seed,
        "n_candidates": len(candidates),
        "n_tested": len(rows),
        "n_original_survivors": len(survivors),
        "holm_cutoff": float(holm_cutoff),
        "original_survivors": survivors,
        "top_by_raw_wild_p": sorted(rows, key=lambda x: x["wild_p"])[:15],
    }


out = {"seeds": [run(seed) for seed in SEEDS]}
os.makedirs("results", exist_ok=True)
with open("results/c_diagnostic_3014_3019.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
