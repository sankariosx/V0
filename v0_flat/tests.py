import numpy as np
import pandas as pd
import json
from pathlib import Path
from config_final import *
from generators import generate_null_market, generate_injected_market, generate_adversarial_null, verify_no_predictive_signal
from features import build_features
from ebm_discovery import SimpleEBM
from validation import evaluate_hypothesis, benjamini_hochberg, split_discovery_validation, recompute_condition
from ledger import clear_ledger, get_ledger_count, append_hypotheses, check_budget

DISCOVERY_FRAC = 0.6  # fraction of bars used for search; the rest is held out for testing
MIN_SAMPLES_VAL = 100  # min samples a recomputed condition needs in the held-out slice to be testable

def run_discovery_on_dataset(features, y, seed=None):
    model = SimpleEBM(outer_bags=OUTER_BAGS, bag_frac=BAG_SAMPLE_FRAC, boost_rounds=BOOST_ROUNDS, max_depth=TREE_MAX_DEPTH, top_features_for_pairs=TOP_FEATURES_FOR_PAIRS, max_interactions=MAX_INTERACTIONS, seed=seed)
    model.fit(features, y)
    candidates = model.generate_candidate_hypotheses(features, y, min_samples=MIN_SAMPLES_FOR_EDGE, effect_thresh=EFFECT_SIZE_THRESH, max_candidates=50)
    return model, candidates

def validate_candidates(feat_val, y_val, candidates, block_len=BLOCK_LENGTH, n_boot=N_BOOTSTRAP, seed=None):
    """Tests each candidate's fixed rule (feature/threshold/direction chosen
    on the DISCOVERY slice) against the HELD-OUT validation slice. A
    candidate that only looked good on the data it was mined from will not
    survive this: its recomputed effect on fresh data regresses toward zero,
    so the bootstrap p-value is no longer artificially small."""
    validated=[]
    for cand in candidates:
        try:
            cond = recompute_condition(cand, feat_val)
        except Exception:
            cond = np.zeros(len(feat_val), dtype=bool)
        n_cond_val = int(cond.sum())
        cand_validated = {k:v for k,v in cand.items() if k!="condition_mask"}
        if n_cond_val < MIN_SAMPLES_VAL:
            # Not enough held-out samples to test this rule -> can't confirm it, treat as non-significant
            cand_validated.update({"boot_p_value": 1.0, "boot_effect": 0.0, "boot_std": 1.0, "n_cond_val": n_cond_val})
        else:
            res = evaluate_hypothesis(y_val.values, cond, block_len=block_len, n_boot=n_boot, seed=seed)
            cand_validated.update({"boot_p_value": res["p_value"], "boot_effect": res["effect"], "boot_std": res["boot_std"], "n_cond_val": res["n_cond"]})
        validated.append(cand_validated)
    return validated

def test_A_null_markets(n_markets=TEST_A_N_MARKETS, seed_base=1000):
    print(f"\n=== TEST A: Null Markets ({n_markets}) FINAL N={N_BARS} ===")
    false_positives=0; results=[]
    for i in range(n_markets):
        seed=seed_base+i
        print(f"  Null market {i+1}/{n_markets} seed={seed}")
        df=generate_null_market(n_bars=N_BARS, seed=seed)
        feat,target=build_features(df, horizon=HORIZON, vol_window=VOL_WINDOW)
        checks=verify_no_predictive_signal(df, feat, horizon=HORIZON)
        print(f"    Verify overall_pass={checks['overall_pass']} max_corr={checks['max_abs_corr']:.4f}")
        feat_disc,target_disc,feat_val,target_val=split_discovery_validation(feat, target, frac=DISCOVERY_FRAC, purge=PURGE_GAP_BARS)
        model,candidates=run_discovery_on_dataset(feat_disc, target_disc, seed=seed)
        print(f"    Candidates: {len(candidates)}")
        n_surv=0; validated=[]
        if len(candidates)>0:
            validated=validate_candidates(feat_val, target_val, candidates, seed=seed)
            can_afford,_,_=check_budget(len(validated))
            if can_afford:
                append_hypotheses([{"market":"testA","seed":seed,"type":v["type"],"features":v["features"]} for v in validated])
            p_vals=[v["boot_p_value"] for v in validated]
            reject,_=benjamini_hochberg(p_vals, q=FDR_Q)
            n_surv=int(reject.sum())
            print(f"    After BH FDR {FDR_Q}: {n_surv} surviving")
            if n_surv>0: false_positives+=1
        results.append({"market_idx":i,"seed":seed,"verification":checks,"n_candidates":len(candidates),"n_surviving_BH":n_surv,"candidates":validated})
    fpr=false_positives/n_markets
    print(f"TEST A RESULT: FPR = {false_positives}/{n_markets} = {fpr:.3f} (threshold {MAX_FPR_TEST_A})")
    return {"fpr":fpr,"false_positives":false_positives,"n_markets":n_markets,"details":results}

def test_B_injected(effect_sizes=TEST_B_EFFECT_SIZES, n_per_size=TEST_B_N_MARKETS_PER_SIZE, seed_base=2000):
    print(f"\n=== TEST B: Injected Edges FINAL N={N_BARS} ===")
    all_results={}
    for effect_type in ["1way","2way","3way"]:
        print(f"\n-- {effect_type} --")
        type_results={}
        for eff_size in effect_sizes:
            print(f"  Effect size {eff_size}")
            detections=0; details=[]
            for j in range(n_per_size):
                seed=seed_base+abs(hash((effect_type, eff_size, j)))%100000
                injected=generate_injected_market(n_bars=N_BARS, effect_type=effect_type, effect_size=eff_size, seed=seed)
                feat=injected["features"]; y=injected["y_injected"]
                feat_disc,y_disc,feat_val,y_val=split_discovery_validation(feat, y, frac=DISCOVERY_FRAC, purge=PURGE_GAP_BARS)
                model,candidates=run_discovery_on_dataset(feat_disc, y_disc, seed=seed)
                surviving=[]
                if len(candidates)>0:
                    validated=validate_candidates(feat_val, y_val, candidates, seed=seed)
                    p_vals=[v["boot_p_value"] for v in validated]
                    reject,_=benjamini_hochberg(p_vals, q=FDR_Q)
                    surviving=[validated[i] for i in range(len(validated)) if reject[i]]
                detected=False
                if len(surviving)>0:
                    for surv in surviving:
                        feats=surv["features"]
                        if effect_type=="1way":
                            if any("ret_16" in f for f in feats): detected=True
                        elif effect_type=="2way":
                            has_vol=any("vol_expansion" in f or "vol_regime" in f for f in feats)
                            has_pct=any("pct_rank" in f for f in feats)
                            if has_vol and has_pct: detected=True
                        elif effect_type=="3way":
                            has_vol=any("vol_regime" in f for f in feats)
                            has_pct=any("pct_rank" in f for f in feats)
                            has_trend=any("trend_persist" in f for f in feats)
                            if has_vol and has_pct and has_trend: detected=True
                if detected: detections+=1
                details.append({"seed":seed,"n_candidates":len(candidates),"n_surviving":len(surviving),"detected":detected,"true_info":injected["info"],"surviving":surviving[:2]})
            power=detections/n_per_size
            print(f"    Power at {eff_size}: {detections}/{n_per_size} = {power:.2f}")
            type_results[eff_size]={"power":power,"detections":detections,"details":details}
        all_results[effect_type]=type_results
    return all_results

def test_C_adversarial(n_markets=TEST_C_N_MARKETS, seed_base=3000):
    print(f"\n=== TEST C: Adversarial Null ({n_markets} markets, {TEST_C_N_FEATURES} features) FINAL ===")
    false_positives=0; results=[]
    for i in range(n_markets):
        seed=seed_base+i
        print(f"  Adversarial {i+1}/{n_markets} seed={seed}")
        adv=generate_adversarial_null(n_bars=N_BARS, n_features_target=TEST_C_N_FEATURES, seed=seed)
        feat=adv["features"]; y=adv["y_base"]
        feat_disc,y_disc,feat_val,y_val=split_discovery_validation(feat, y, frac=DISCOVERY_FRAC, purge=PURGE_GAP_BARS)
        model,candidates=run_discovery_on_dataset(feat_disc, y_disc, seed=seed)
        print(f"    Candidates: {len(candidates)}")
        n_surv=0; validated=[]
        if len(candidates)>0:
            validated=validate_candidates(feat_val, y_val, candidates, seed=seed)
            p_vals=[v["boot_p_value"] for v in validated]
            reject,_=benjamini_hochberg(p_vals, q=FDR_Q)
            n_surv=int(reject.sum())
            print(f"    After BH: {n_surv} surviving")
            if n_surv>0: false_positives+=1
        results.append({"market_idx":i,"seed":seed,"n_candidates":len(candidates),"n_surviving":n_surv,"candidates":validated})
    fpr=false_positives/n_markets
    print(f"TEST C RESULT: FPR = {false_positives}/{n_markets} = {fpr:.3f} (threshold {MAX_FPR_TEST_C})")
    return {"fpr":fpr,"false_positives":false_positives,"n_markets":n_markets,"details":results}

def run_all_tests():
    clear_ledger()
    print(f"Ledger cleared. Budget {HYPOTHESIS_BUDGET}")
    print(f"Final config: N_BARS={N_BARS} TOP={TOP_FEATURES_FOR_PAIRS} MAX={MAX_INTERACTIONS} BAGS={OUTER_BAGS} ROUNDS={BOOST_ROUNDS}")
    print(f"Purified interaction scoring replacing FAST screening, EBM validation kept")
    resA=test_A_null_markets(n_markets=TEST_A_N_MARKETS)
    resB=test_B_injected(effect_sizes=TEST_B_EFFECT_SIZES, n_per_size=TEST_B_N_MARKETS_PER_SIZE)
    resC=test_C_adversarial(n_markets=TEST_C_N_MARKETS)
    used=get_ledger_count()
    print(f"\nLedger used: {used}/{HYPOTHESIS_BUDGET}")
    proceed=True; reasons=[]
    if resA["fpr"] > MAX_FPR_TEST_A:
        proceed=False; reasons.append(f"Test A FAIL: FPR {resA['fpr']:.3f} > {MAX_FPR_TEST_A}")
    else:
        reasons.append(f"Test A PASS: FPR {resA['fpr']:.3f} <= {MAX_FPR_TEST_A}")
    if resC["fpr"] > MAX_FPR_TEST_C:
        proceed=False; reasons.append(f"Test C FAIL: FPR {resC['fpr']:.3f} > {MAX_FPR_TEST_C}")
    else:
        reasons.append(f"Test C PASS: FPR {resC['fpr']:.3f} <= {MAX_FPR_TEST_C}")
    try:
        power_1way_03=resB["1way"][0.3]["power"]
        if power_1way_03 < MIN_POWER_1WAY_03:
            proceed=False; reasons.append(f"Test B 1-way FAIL: power {power_1way_03:.2f} < {MIN_POWER_1WAY_03}")
        else:
            reasons.append(f"Test B 1-way PASS: power {power_1way_03:.2f} >= {MIN_POWER_1WAY_03}")
    except Exception as e:
        reasons.append(f"Test B 1-way error: {e}"); proceed=False
    try:
        power_2way_03=resB["2way"][0.3]["power"]
        if power_2way_03 < MIN_POWER_2WAY_03:
            proceed=False; reasons.append(f"Test B 2-way FAIL: power {power_2way_03:.2f} < {MIN_POWER_2WAY_03}")
        else:
            reasons.append(f"Test B 2-way PASS: power {power_2way_03:.2f} >= {MIN_POWER_2WAY_03}")
    except Exception as e:
        reasons.append(f"Test B 2-way error: {e}"); proceed=False
    try:
        power_3way_03=resB["3way"][0.3]["power"]
        if power_3way_03 > MAX_POWER_3WAY_EXPECTED:
            reasons.append(f"Test B 3-way UNEXPECTED PASS: power {power_3way_03:.2f} > {MAX_POWER_3WAY_EXPECTED} (model can detect 3-way)")
        else:
            reasons.append(f"Test B 3-way EXPECTED FAIL: power {power_3way_03:.2f} <= {MAX_POWER_3WAY_EXPECTED} - correctly identifies model incapability for 3-way")
    except Exception as e:
        reasons.append(f"Test B 3-way: {e}")
    report={"proceed":proceed,"reasons":reasons,"test_A":resA,"test_B":resB,"test_C":resC,"ledger_used":used,"model":"SimpleEBM with purified interaction scoring (TOP30 MAX50)"}
    out_path=Path(__file__).parent / "results" / "v0_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        def convert(o):
            if isinstance(o, (np.integer, np.floating)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
        json.dump(report, f, indent=2, default=convert)
    print("\n" + "="*70)
    print("V0 FINAL REPORT - PROCEED / DO-NOT-PROCEED")
    print("="*70)
    for r in reasons:
        print(f" - {r}")
    print(f"\nFINAL DECISION: {'PROCEED' if proceed else 'DO-NOT-PROCEED'}")
    print(f"Full report: {out_path}")
    print("="*70)
    return report
