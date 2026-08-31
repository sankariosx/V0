"""
parallel_runner.py — Production runner for the COMPLETE V0 A+B+C validation.

WHAT THIS DOES
---------------
Runs exactly the same statistical procedure as tests.py::run_all_tests(),
using the exact same functions (run_discovery_on_dataset, validate_candidates,
benjamini_hochberg, generate_null_market, generate_injected_market,
generate_adversarial_null, verify_no_predictive_signal) imported unmodified
from this package. Nothing about N_BARS, OUTER_BAGS, BOOST_ROUNDS, market
counts, bootstrap counts, effect sizes, thresholds, or acceptance criteria
is changed. config_final.py is imported as-is and used as-is.

WHAT IS PARALLELIZED
---------------------
Only the outer loop across independent markets/effect-size/repeat
combinations (170 total jobs: 30 for Test A, 120 for Test B, 20 for Test C).
Each job is fully self-contained: it draws its own np.random.default_rng(seed)
inside generate_*_market(...) and inside SimpleEBM.fit(...) — nothing is
shared or mutated across jobs. Running job X on core 1 and job Y on core 2
produces bit-identical output to running X then Y sequentially on one core,
because neither job's RNG stream depends on execution order. This is NOT
true of the internal outer-bag loop inside SimpleEBM.fit() (bags share one
RNG stream drawn sequentially), so bags are NOT parallelized — only whole
market-jobs are, which keeps every result identical to sequential execution.

DETERMINISM CAVEAT (found in the provided code, not introduced by this
runner): Test B's seed is computed with Python's built-in hash() on a tuple
of (effect_type, eff_size, j). Since Python 3.3, str hashing is randomized
per-process via PYTHONHASHSEED unless that env var is fixed — so even the
ORIGINAL sequential tests.py would give different Test-B seeds (and thus
different synthetic markets) on every separate invocation unless
PYTHONHASHSEED is pinned. This runner pins PYTHONHASHSEED=0 (re-execing
itself once at startup if needed) purely so that (a) repeated runs of this
exact script are reproducible and (b) all worker subprocesses agree with
the coordinator on the seed values. This does not touch the seed FORMULA,
config, or any statistical/acceptance logic — only fixes an environment
variable so the formula gives a stable answer.

CHECKPOINTING / RESUME
-----------------------
Every completed job is written to results/checkpoints/<job_key>.json the
moment it finishes (atomic write: tmp file + os.replace). On startup this
script scans that directory; any job whose checkpoint has status=="ok" AND
whose stored config snapshot matches the current config_final.py is treated
as already done and is skipped. Anything else (never run, or errored, or
run under a different config) is (re)submitted. This means:
  - You can Ctrl-C, reboot, or lose power, and rerun the exact same command
    to resume from where it left off.
  - If you edit config_final.py, stale checkpoints are automatically
    detected (config snapshot mismatch) and recomputed — never silently
    reused.
  - Errored jobs are never counted as done; they are retried on next run
    and are called out explicitly in the final summary.

USAGE
-----
    PYTHONHASHSEED=0 python3 run_v0_parallel.py [--workers N]

(You do not strictly need to set PYTHONHASHSEED yourself — the script
re-execs itself with it pinned if it isn't already set — but setting it
explicitly is harmless and makes the determinism visible.)

Only after ALL 170 jobs have status=="ok" does this script assemble the
final consolidated report and apply the acceptance criteria (verbatim
from tests.py::run_all_tests). If any job is missing or errored, it prints
an explicit INCOMPLETE status and does NOT declare PROCEED or DO-NOT-PROCEED.
"""
import os
import sys

# ---------------------------------------------------------------------------
# Pin PYTHONHASHSEED before anything that might call hash() on a str/tuple.
# Must happen before importing v0_lab_final.tests (which computes Test B
# seeds using hash()) and before the multiprocessing workers spawn.
# ---------------------------------------------------------------------------
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)

import json
import time
import hashlib
import traceback
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config_final as CFG
from generators import (
    generate_null_market,
    generate_injected_market,
    generate_adversarial_null,
    verify_no_predictive_signal,
)
from features import build_features
from validation import benjamini_hochberg
from ledger import clear_ledger, get_ledger_count
# Import the exact, unmodified helper functions from tests.py so the
# statistical procedure is guaranteed identical (not re-transcribed).
from tests import run_discovery_on_dataset, validate_candidates, DISCOVERY_FRAC
from validation import split_discovery_validation

CHECKPOINT_DIR = CFG.RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Config snapshot — used only to detect "this checkpoint was produced under
# a different config" so stale results are never silently reused.
# ---------------------------------------------------------------------------
_CONFIG_KEYS = [
    "N_BARS", "INITIAL_PRICE", "GARCH_OMEGA", "GARCH_ALPHA", "GARCH_BETA",
    "T_DF", "GAP_PROB", "GAP_MULT", "HORIZON", "VOL_WINDOW", "PURGE_GAP_BARS",
    "BLOCK_LENGTH", "OUTER_BAGS", "BAG_SAMPLE_FRAC", "BOOST_ROUNDS",
    "TREE_MAX_DEPTH", "TOP_FEATURES_FOR_PAIRS", "MAX_INTERACTIONS",
    "MIN_SAMPLES_FOR_EDGE", "EFFECT_SIZE_THRESH", "N_BOOTSTRAP", "FDR_Q",
    "TEST_A_N_MARKETS", "TEST_B_EFFECT_SIZES", "TEST_B_N_MARKETS_PER_SIZE",
    "TEST_C_N_MARKETS", "TEST_C_N_FEATURES", "MAX_FPR_TEST_A",
    "MAX_FPR_TEST_C", "MIN_POWER_1WAY_03", "MIN_POWER_2WAY_03",
    "MAX_POWER_3WAY_EXPECTED",
]


def config_snapshot():
    d = {k: getattr(CFG, k) for k in _CONFIG_KEYS}
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


CONFIG_HASH = config_snapshot()


def convert(o):
    if isinstance(o, (np.integer, np.floating)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def atomic_write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=convert)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Job enumeration — seed formulas copied EXACTLY from tests.py
# ---------------------------------------------------------------------------
def build_job_list():
    jobs = []

    # Test A
    seed_base = 1000
    for i in range(CFG.TEST_A_N_MARKETS):
        seed = seed_base + i
        jobs.append({"test": "A", "key": f"testA_market_{i:03d}_seed{seed}",
                     "i": i, "seed": seed})

    # Test B — seed formula identical to tests.py::test_B_injected
    seed_base = 2000
    for effect_type in ["1way", "2way", "3way"]:
        for eff_size in CFG.TEST_B_EFFECT_SIZES:
            for j in range(CFG.TEST_B_N_MARKETS_PER_SIZE):
                seed = seed_base + abs(hash((effect_type, eff_size, j))) % 100000
                key = f"testB_{effect_type}_eff{eff_size}_j{j:02d}_seed{seed}"
                jobs.append({"test": "B", "key": key, "effect_type": effect_type,
                             "eff_size": eff_size, "j": j, "seed": seed})

    # Test C
    seed_base = 3000
    for i in range(CFG.TEST_C_N_MARKETS):
        seed = seed_base + i
        jobs.append({"test": "C", "key": f"testC_market_{i:03d}_seed{seed}",
                     "i": i, "seed": seed})

    return jobs


# ---------------------------------------------------------------------------
# Per-job workers — bodies mirror tests.py's per-market loop iterations
# exactly (same functions, same order of operations, same fields recorded).
# ---------------------------------------------------------------------------
def _worker_test_A(job):
    i, seed = job["i"], job["seed"]
    df = generate_null_market(n_bars=CFG.N_BARS, seed=seed)
    feat, target = build_features(df, horizon=CFG.HORIZON, vol_window=CFG.VOL_WINDOW)
    checks = verify_no_predictive_signal(df, feat, horizon=CFG.HORIZON)
    feat_disc, target_disc, feat_val, target_val = split_discovery_validation(
        feat, target, frac=DISCOVERY_FRAC, purge=CFG.PURGE_GAP_BARS)
    model, candidates = run_discovery_on_dataset(feat_disc, target_disc, seed=seed)
    n_surv = 0
    validated = []
    if len(candidates) > 0:
        validated = validate_candidates(feat_val, target_val, candidates, seed=seed)
        p_vals = [v["boot_p_value"] for v in validated]
        reject, _ = benjamini_hochberg(p_vals, q=CFG.FDR_Q)
        n_surv = int(reject.sum())
    return {
        "market_idx": i, "seed": seed, "verification": checks,
        "n_candidates": len(candidates), "n_surviving_BH": n_surv,
        "candidates": validated,
    }


def _worker_test_B(job):
    effect_type, eff_size, j, seed = job["effect_type"], job["eff_size"], job["j"], job["seed"]
    injected = generate_injected_market(n_bars=CFG.N_BARS, effect_type=effect_type,
                                         effect_size=eff_size, seed=seed)
    feat = injected["features"]
    y = injected["y_injected"]
    feat_disc, y_disc, feat_val, y_val = split_discovery_validation(
        feat, y, frac=DISCOVERY_FRAC, purge=CFG.PURGE_GAP_BARS)
    model, candidates = run_discovery_on_dataset(feat_disc, y_disc, seed=seed)
    surviving = []
    if len(candidates) > 0:
        validated = validate_candidates(feat_val, y_val, candidates, seed=seed)
        p_vals = [v["boot_p_value"] for v in validated]
        reject, _ = benjamini_hochberg(p_vals, q=CFG.FDR_Q)
        surviving = [validated[k] for k in range(len(validated)) if reject[k]]
    detected = False
    if len(surviving) > 0:
        for surv in surviving:
            feats = surv["features"]
            if effect_type == "1way":
                if any("ret_16" in f for f in feats):
                    detected = True
            elif effect_type == "2way":
                has_vol = any("vol_expansion" in f or "vol_regime" in f for f in feats)
                has_pct = any("pct_rank" in f for f in feats)
                if has_vol and has_pct:
                    detected = True
            elif effect_type == "3way":
                has_vol = any("vol_regime" in f for f in feats)
                has_pct = any("pct_rank" in f for f in feats)
                has_trend = any("trend_persist" in f for f in feats)
                if has_vol and has_pct and has_trend:
                    detected = True
    return {
        "seed": seed, "effect_type": effect_type, "eff_size": eff_size, "j": j,
        "n_candidates": len(candidates), "n_surviving": len(surviving),
        "detected": detected, "true_info": injected["info"], "surviving": surviving[:2],
    }


def _worker_test_C(job):
    i, seed = job["i"], job["seed"]
    adv = generate_adversarial_null(n_bars=CFG.N_BARS, n_features_target=CFG.TEST_C_N_FEATURES, seed=seed)
    feat = adv["features"]
    y = adv["y_base"]
    feat_disc, y_disc, feat_val, y_val = split_discovery_validation(
        feat, y, frac=DISCOVERY_FRAC, purge=CFG.PURGE_GAP_BARS)
    model, candidates = run_discovery_on_dataset(feat_disc, y_disc, seed=seed)
    n_surv = 0
    validated = []
    if len(candidates) > 0:
        validated = validate_candidates(feat_val, y_val, candidates, seed=seed)
        p_vals = [v["boot_p_value"] for v in validated]
        reject, _ = benjamini_hochberg(p_vals, q=CFG.FDR_Q)
        n_surv = int(reject.sum())
    return {
        "market_idx": i, "seed": seed, "n_candidates": len(candidates),
        "n_surviving": n_surv, "candidates": validated,
    }


def run_one_job(job):
    """Top-level function (must be module-level for multiprocessing pickling).
    Runs the job, writes its checkpoint immediately, returns a small summary
    for progress tracking (NOT the full result — the checkpoint on disk is
    the source of truth used for final aggregation)."""
    key = job["key"]
    ckpt_path = CHECKPOINT_DIR / f"{key}.json"
    t0 = time.time()
    try:
        if job["test"] == "A":
            result = _worker_test_A(job)
        elif job["test"] == "B":
            result = _worker_test_B(job)
        elif job["test"] == "C":
            result = _worker_test_C(job)
        else:
            raise ValueError(f"unknown test type {job['test']}")
        elapsed = time.time() - t0
        payload = {
            "status": "ok", "job": job, "config_hash": CONFIG_HASH,
            "elapsed_sec": elapsed, "result": result,
        }
        atomic_write_json(ckpt_path, payload)
        return {"key": key, "test": job["test"], "status": "ok", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        payload = {
            "status": "error", "job": job, "config_hash": CONFIG_HASH,
            "elapsed_sec": elapsed, "error": str(e), "traceback": traceback.format_exc(),
        }
        atomic_write_json(ckpt_path, payload)
        return {"key": key, "test": job["test"], "status": "error", "elapsed": elapsed,
                "error": str(e)}


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------
def load_done_keys():
    """Return set of job keys with an on-disk checkpoint that is status==ok
    AND matches the current config. Anything else must be (re)run."""
    done = set()
    for p in CHECKPOINT_DIR.glob("*.json"):
        if p.name.endswith(".tmp"):
            continue
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("status") == "ok" and data.get("config_hash") == CONFIG_HASH:
            done.add(p.stem)
    return done


def load_all_checkpoints():
    ckpts = {}
    for p in CHECKPOINT_DIR.glob("*.json"):
        if p.name.endswith(".tmp"):
            continue
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception:
            continue
        ckpts[p.stem] = data
    return ckpts


# ---------------------------------------------------------------------------
# Final aggregation — mirrors tests.py::run_all_tests() acceptance logic
# verbatim, applied to results loaded from checkpoints instead of freshly
# computed in-loop.
# ---------------------------------------------------------------------------
def assemble_final_report(all_jobs):
    ckpts = load_all_checkpoints()

    errored = [j["key"] for j in all_jobs if ckpts.get(j["key"], {}).get("status") != "ok"
               or ckpts.get(j["key"], {}).get("config_hash") != CONFIG_HASH]
    if errored:
        print("\n" + "=" * 70)
        print(f"INCOMPLETE: {len(errored)}/{len(all_jobs)} jobs missing or errored "
              f"(or stale vs current config). Cannot produce a verdict.")
        print("First 20 outstanding job keys:")
        for k in errored[:20]:
            status = ckpts.get(k, {}).get("status", "MISSING")
            err = ckpts.get(k, {}).get("error", "")
            print(f"  - {k}: {status} {('- ' + err) if err else ''}")
        print("Rerun this script (same command) to resume and retry these jobs.")
        print("=" * 70)
        return {"status": "incomplete", "n_total": len(all_jobs), "n_missing_or_errored": len(errored),
                "outstanding_keys": errored}

    # --- Test A aggregation (mirrors test_A_null_markets) ---
    a_details = sorted(
        (ckpts[j["key"]]["result"] for j in all_jobs if j["test"] == "A"),
        key=lambda r: r["market_idx"],
    )
    a_fp = sum(1 for r in a_details if r["n_surviving_BH"] > 0)
    a_n = len(a_details)
    resA = {"fpr": a_fp / a_n, "false_positives": a_fp, "n_markets": a_n, "details": a_details}

    # --- Test B aggregation (mirrors test_B_injected) ---
    resB = {}
    for effect_type in ["1way", "2way", "3way"]:
        resB[effect_type] = {}
        for eff_size in CFG.TEST_B_EFFECT_SIZES:
            group = sorted(
                (ckpts[j["key"]]["result"] for j in all_jobs
                 if j["test"] == "B" and j["effect_type"] == effect_type and j["eff_size"] == eff_size),
                key=lambda r: r["j"],
            )
            n = len(group)
            detections = sum(1 for r in group if r["detected"])
            resB[effect_type][eff_size] = {
                "power": detections / n if n else 0.0, "detections": detections, "details": group,
            }

    # --- Test C aggregation (mirrors test_C_adversarial) ---
    c_details = sorted(
        (ckpts[j["key"]]["result"] for j in all_jobs if j["test"] == "C"),
        key=lambda r: r["market_idx"],
    )
    c_fp = sum(1 for r in c_details if r["n_surviving"] > 0)
    c_n = len(c_details)
    resC = {"fpr": c_fp / c_n, "false_positives": c_fp, "n_markets": c_n, "details": c_details}

    used = get_ledger_count()

    # --- Acceptance criteria — copied verbatim from tests.py::run_all_tests ---
    proceed = True
    reasons = []
    if resA["fpr"] > CFG.MAX_FPR_TEST_A:
        proceed = False
        reasons.append(f"Test A FAIL: FPR {resA['fpr']:.3f} > {CFG.MAX_FPR_TEST_A}")
    else:
        reasons.append(f"Test A PASS: FPR {resA['fpr']:.3f} <= {CFG.MAX_FPR_TEST_A}")

    if resC["fpr"] > CFG.MAX_FPR_TEST_C:
        proceed = False
        reasons.append(f"Test C FAIL: FPR {resC['fpr']:.3f} > {CFG.MAX_FPR_TEST_C}")
    else:
        reasons.append(f"Test C PASS: FPR {resC['fpr']:.3f} <= {CFG.MAX_FPR_TEST_C}")

    try:
        power_1way_03 = resB["1way"][0.3]["power"]
        if power_1way_03 < CFG.MIN_POWER_1WAY_03:
            proceed = False
            reasons.append(f"Test B 1-way FAIL: power {power_1way_03:.2f} < {CFG.MIN_POWER_1WAY_03}")
        else:
            reasons.append(f"Test B 1-way PASS: power {power_1way_03:.2f} >= {CFG.MIN_POWER_1WAY_03}")
    except Exception as e:
        reasons.append(f"Test B 1-way error: {e}")
        proceed = False

    try:
        power_2way_03 = resB["2way"][0.3]["power"]
        if power_2way_03 < CFG.MIN_POWER_2WAY_03:
            proceed = False
            reasons.append(f"Test B 2-way FAIL: power {power_2way_03:.2f} < {CFG.MIN_POWER_2WAY_03}")
        else:
            reasons.append(f"Test B 2-way PASS: power {power_2way_03:.2f} >= {CFG.MIN_POWER_2WAY_03}")
    except Exception as e:
        reasons.append(f"Test B 2-way error: {e}")
        proceed = False

    try:
        power_3way_03 = resB["3way"][0.3]["power"]
        if power_3way_03 > CFG.MAX_POWER_3WAY_EXPECTED:
            reasons.append(f"Test B 3-way UNEXPECTED PASS: power {power_3way_03:.2f} > "
                            f"{CFG.MAX_POWER_3WAY_EXPECTED} (model can detect 3-way)")
        else:
            reasons.append(f"Test B 3-way EXPECTED FAIL: power {power_3way_03:.2f} <= "
                            f"{CFG.MAX_POWER_3WAY_EXPECTED} - correctly identifies model incapability for 3-way")
    except Exception as e:
        reasons.append(f"Test B 3-way: {e}")

    report = {
        "status": "complete", "proceed": proceed, "reasons": reasons,
        "test_A": resA, "test_B": resB, "test_C": resC, "ledger_used": used,
        "model": "SimpleEBM with purified interaction scoring (TOP30 MAX50)",
        "config_hash": CONFIG_HASH,
    }
    out_path = CFG.RESULTS_DIR / "v0_report_parallel.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=convert)

    print("\n" + "=" * 70)
    print("V0 FINAL REPORT - PROCEED / DO-NOT-PROCEED (parallel runner)")
    print("=" * 70)
    for r in reasons:
        print(f" - {r}")
    print(f"\nFINAL DECISION: {'PROCEED' if proceed else 'DO-NOT-PROCEED'}")
    print(f"Full report: {out_path}")
    print("=" * 70)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run the complete V0 A+B+C validation in parallel.")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1),
                         help="Number of worker processes (default: cpu_count()-1)")
    args = parser.parse_args()

    all_jobs = build_job_list()
    assert len(all_jobs) == CFG.TEST_A_N_MARKETS + 3 * len(CFG.TEST_B_EFFECT_SIZES) * CFG.TEST_B_N_MARKETS_PER_SIZE + CFG.TEST_C_N_MARKETS

    if not any(CHECKPOINT_DIR.glob("*.json")):
        clear_ledger()

    done_keys = load_done_keys()
    pending = [j for j in all_jobs if j["key"] not in done_keys]

    print(f"Config hash: {CONFIG_HASH}")
    print(f"Total jobs: {len(all_jobs)}  |  Already done: {len(done_keys)}  |  Pending: {len(pending)}")
    print(f"Workers: {args.workers}")

    if pending:
        n_total = len(all_jobs)
        n_done_at_start = len(done_keys)
        n_done = n_done_at_start
        t_start = time.time()
        # Test C has ~2.35x the features of A/B (120 vs ~51), so its
        # per-job cost is materially different -> track group averages
        # separately for a more honest ETA.
        times_by_group = {"AB": [], "C": []}
        total_ab_pending = sum(1 for j in pending if j["test"] != "C")
        total_c_pending = sum(1 for j in pending if j["test"] == "C")
        done_ab_this_run = 0
        done_c_this_run = 0

        with Pool(processes=args.workers) as pool:
            for res in pool.imap_unordered(run_one_job, pending):
                n_done += 1
                group = "C" if res["test"] == "C" else "AB"
                times_by_group[group].append(res["elapsed"])
                if group == "C":
                    done_c_this_run += 1
                else:
                    done_ab_this_run += 1

                elapsed_total = time.time() - t_start
                avg_ab = (sum(times_by_group["AB"]) / len(times_by_group["AB"])
                          if times_by_group["AB"] else None)
                avg_c = (sum(times_by_group["C"]) / len(times_by_group["C"])
                         if times_by_group["C"] else None)
                fallback_avg = avg_ab or avg_c or 1.0

                ab_remaining = max(0, total_ab_pending - done_ab_this_run)
                c_remaining = max(0, total_c_pending - done_c_this_run)
                est_work_secs = ab_remaining * (avg_ab or fallback_avg) + c_remaining * (avg_c or fallback_avg)
                est_remaining_wall_h = (est_work_secs / max(1, args.workers)) / 3600

                status_str = "OK" if res["status"] == "ok" else f"ERROR: {res.get('error')}"
                print(f"[{n_done}/{n_total}] {res['key']} ({res['test']}) {status_str} "
                      f"in {res['elapsed']:.0f}s | elapsed {elapsed_total/3600:.2f}h | "
                      f"est. remaining ~{est_remaining_wall_h:.2f}h (workers={args.workers})")
                sys.stdout.flush()
    else:
        print("Nothing pending — all jobs already checkpointed for this config.")

    assemble_final_report(all_jobs)


if __name__ == "__main__":
    main()
