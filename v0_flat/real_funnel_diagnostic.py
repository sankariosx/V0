"""
Run-5 real XAUUSD discovery funnel diagnostic.

This is diagnostic-only. It does not modify the production V0 runner or launch
the 170-job synthetic validation. It reproduces the real-data discovery path
and records where hypotheses are lost before held-out validation.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from config_final import (
    HORIZON, VOL_WINDOW, PURGE_GAP_BARS, BLOCK_LENGTH,
    MIN_SAMPLES_FOR_EDGE, EFFECT_SIZE_THRESH, N_BOOTSTRAP
)
from features import build_features
from validation import split_discovery_validation
from ebm_discovery import SimpleEBM
from real_runner import load_data, apply_real_horizon_mask, DATA_PATH, DISCOVERY_FRAC

SEED = 20260926
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "real_xauusd_funnel_diagnostic.json"


def main():
    df = load_data(DATA_PATH)
    feat, target = build_features(df, horizon=HORIZON, vol_window=VOL_WINDOW)
    feat, target, n_valid = apply_real_horizon_mask(feat, target, df.index)

    feat_disc, target_disc, feat_val, target_val = split_discovery_validation(
        feat, target, frac=DISCOVERY_FRAC, purge=PURGE_GAP_BARS
    )

    model = SimpleEBM(seed=SEED)
    model.fit(feat_disc, target_disc)

    p = feat_disc.shape[1]
    top = list(getattr(model, "top_main_effect_pool_", []))
    pair_pool = list(getattr(model, "pair_pool_before_dedup_", []))
    dedup_pool = []
    for c in getattr(model, "dedup_clusters_", []):
        name = c["representative"]
        dedup_pool.append(model.feature_names_.index(name))

    interaction_importances = getattr(model, "interaction_importances_", {})
    purified = getattr(model, "interaction_scores_purified_", {})

    all_pairs_after_dedup = set()
    for a in range(len(dedup_pool)):
        for b in range(a + 1, len(dedup_pool)):
            i, j = sorted((int(dedup_pool[a]), int(dedup_pool[b])))
            all_pairs_after_dedup.add((i, j))

    # Reconstruct the exact alias-expanded pair queue used by _generate_twoway.
    family_by_feature = {}
    for cluster in getattr(model, "dedup_clusters_", []):
        members = [model.feature_names_.index(name) for name in cluster["members"]]
        for member in members:
            family_by_feature[member] = members

    ranked_pairs = sorted(purified.items(), key=lambda x: x[1], reverse=True)
    pair_queue = []
    seen_pairs = set()
    max_candidates = 50
    queue_family_cut = None

    for (fi, fj), score in ranked_pairs:
        base = tuple(sorted((int(fi), int(fj))))
        expansions = [base]
        fam_i = family_by_feature.get(base[0], [base[0]])
        fam_j = family_by_feature.get(base[1], [base[1]])
        alias_count = 0
        for ai in fam_i:
            for aj in fam_j:
                if ai == aj:
                    continue
                pair = tuple(sorted((int(ai), int(aj))))
                if pair not in expansions:
                    expansions.append(pair)
                alias_count += 1
                if alias_count >= 12:
                    break
            if alias_count >= 12:
                break
        for pair in expansions:
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                pair_queue.append((pair, float(score)))
        if len(pair_queue) >= max(100, max_candidates * 4):
            queue_family_cut = len(pair_queue)
            break

    X = feat_disc.values
    y = target_disc.values
    one_total = 0
    one_npass = 0
    one_effect = 0
    one_stable = 0
    one_stable_values = []
    one_effect_values = []

    for fname, imp, fi in model.get_top_features(k=model.top_features_for_pairs):
        col = X[:, fi]
        for q in [0.10, 0.15, 0.20, 0.80, 0.85]:
            one_total += 1
            thresh = np.quantile(col, q)
            cond = col < thresh if q < 0.5 else col > thresh
            n_cond = int(cond.sum())
            if n_cond < MIN_SAMPLES_FOR_EDGE or n_cond > len(col) * 0.5:
                continue
            one_npass += 1
            effect = np.mean(y[cond]) - np.mean(y)
            one_effect_values.append(float(effect))
            if abs(effect) < EFFECT_SIZE_THRESH:
                continue
            one_effect += 1
            stable = 0
            for bag in model.bags_:
                idx = bag["indices"]
                cb = col[idx]
                yb = y[idx]
                cond_b = cb < thresh if q < 0.5 else cb > thresh
                if cond_b.sum() >= 50 and np.sign(np.mean(yb[cond_b]) - np.mean(yb)) == np.sign(effect):
                    stable += 1
            stability = stable / len(model.bags_) if model.bags_ else 1.0
            one_stable_values.append(float(stability))
            if stability >= 0.6:
                one_stable += 1

    two_total_tests = 0
    two_npass_samples = 0
    two_effect = 0
    two_stable = 0
    two_stable_values = []
    two_effect_values = []
    pair_effect_pass = set()
    pair_stable_pass = set()

    for (fi, fj), score in pair_queue:
        col1, col2 = X[:, fi], X[:, fj]
        for q1, q2 in [(0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.2, 0.2)]:
            two_total_tests += 1
            t1, t2 = np.quantile(col1, q1), np.quantile(col2, q2)
            c1 = col1 < t1 if q1 < 0.5 else col1 > t1
            c2 = col2 < t2 if q2 < 0.5 else col2 > t2
            cond = c1 & c2
            n_cond = int(cond.sum())
            if n_cond < MIN_SAMPLES_FOR_EDGE:
                continue
            two_npass_samples += 1
            effect = np.mean(y[cond]) - np.mean(y)
            two_effect_values.append(float(effect))
            if abs(effect) < EFFECT_SIZE_THRESH:
                continue
            two_effect += 1
            pair_effect_pass.add((fi, fj))
            stable = 0
            for bag in model.bags_:
                idx = bag["indices"]
                c1b = col1[idx] < t1 if q1 < 0.5 else col1[idx] > t1
                c2b = col2[idx] < t2 if q2 < 0.5 else col2[idx] > t2
                cb = c1b & c2b
                if cb.sum() >= 30 and np.sign(np.mean(y[idx][cb]) - np.mean(y[idx])) == np.sign(effect):
                    stable += 1
            stability = stable / len(model.bags_) if model.bags_ else 1.0
            two_stable_values.append(float(stability))
            if stability >= 0.6:
                two_stable += 1
                pair_stable_pass.add((fi, fj))

    candidates = model.generate_candidate_hypotheses(
        feat_disc, target_disc,
        min_samples=MIN_SAMPLES_FOR_EDGE,
        effect_thresh=EFFECT_SIZE_THRESH,
        max_candidates=50,
    )

    one_final = [c for c in candidates if c["type"] == "1way"]
    two_final = [c for c in candidates if c["type"] == "2way"]

    report = {
        "diagnostic": "Run-5 real XAUUSD discovery funnel",
        "seed": SEED,
        "dataset": {
            "rows_raw": int(len(df)),
            "rows_usable": int(len(feat)),
            "rows_true_4h": int(n_valid),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "discovery_rows": int(len(feat_disc)),
            "validation_rows": int(len(feat_val)),
        },
        "config": {
            "feature_count": int(p),
            "top_features_for_pairs": int(model.top_features_for_pairs),
            "max_interactions": int(model.max_interactions),
            "min_samples": int(MIN_SAMPLES_FOR_EDGE),
            "effect_threshold": float(EFFECT_SIZE_THRESH),
            "bootstrap_bags": int(len(model.bags_)),
            "candidate_budget": 50,
        },
        "funnel": {
            "features_total": int(p),
            "top_main_effect_pool": int(len(top)),
            "pair_pool_before_dedup": int(len(pair_pool)),
            "dedup_representatives": int(len(dedup_pool)),
            "dedup_clusters": int(len(getattr(model, "dedup_clusters_", []))),
            "all_pairs_after_dedup": int(len(all_pairs_after_dedup)),
            "ebm_interaction_recall_terms": int(len(interaction_importances)),
            "purified_pair_scores": int(len(purified)),
            "alias_expanded_pair_queue": int(len(pair_queue)),
            "queue_early_cut_value": queue_family_cut,
        },
        "one_way_funnel": {
            "tests_generated": int(one_total),
            "pass_sample_size": int(one_npass),
            "pass_effect": int(one_effect),
            "pass_stability": int(one_stable),
            "final_budget": int(len(one_final)),
            "effect_abs_min": float(min([abs(x) for x in one_effect_values], default=0.0)),
            "stability_min_after_effect": float(min(one_stable_values, default=0.0)),
        },
        "two_way_funnel": {
            "threshold_tests": int(two_total_tests),
            "pass_sample_size": int(two_npass_samples),
            "pass_effect": int(two_effect),
            "pass_stability": int(two_stable),
            "unique_pairs_pass_effect": int(len(pair_effect_pass)),
            "unique_pairs_pass_stability": int(len(pair_stable_pass)),
            "final_budget": int(len(two_final)),
        },
        "final_discovery_output": {
            "n_candidates": int(len(candidates)),
            "n_one_way": int(len(one_final)),
            "n_two_way": int(len(two_final)),
            "top_candidates": [
                {
                    "type": c["type"],
                    "features": c["features"],
                    "effect_size": float(c["effect_size"]),
                    "stability": float(c["stability"]),
                    "importance": float(c["importance"]),
                    "n_samples": int(c["n_samples"]),
                }
                for c in candidates[:50]
            ],
        },
        "top_purified_pairs": [
            {
                "features": [model.feature_names_[i], model.feature_names_[j]],
                "score": float(score),
            }
            for (i, j), score in ranked_pairs[:30]
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
