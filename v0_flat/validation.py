import numpy as np

def circular_block_bootstrap(y, condition, block_len=64, n_boot=2000, seed=None):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); cond = np.asarray(condition, dtype=bool); n = len(y)
    if cond.sum() == 0:
        return {"effect": 0.0, "p_value": 1.0, "n_cond": 0, "boot_std": 1.0, "boot_effects": np.zeros(n_boot)}
    obs_effect = np.mean(y[cond]) - np.mean(y)
    boot_effects = np.zeros(n_boot)
    y_extended = np.concatenate([y, y[:block_len]])
    n_blocks = int(np.ceil(n / block_len))
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        y_boot = np.zeros(n); pos = 0
        for s in starts:
            block = y_extended[s:s+block_len]
            take = min(block_len, n - pos)
            y_boot[pos:pos+take] = block[:take]
            pos += take
            if pos >= n: break
        boot_effects[b] = np.mean(y_boot[cond]) - np.mean(y_boot)
    p_val = np.mean(np.abs(boot_effects) >= np.abs(obs_effect) + 1e-12)
    return {"effect": float(obs_effect), "p_value": float(p_val), "boot_std": float(np.std(boot_effects)), "n_cond": int(cond.sum()), "boot_effects": boot_effects}

def circular_shift_test(y, condition, n_perm=2000, seed=None, min_shift=None):
    """Held-out permutation test preserving the serial structure of y.

    The discovery rule is fixed before this test is run. Random circular shifts
    destroy feature/target alignment while preserving the target's marginal
    distribution and its autocorrelation, giving a direct null distribution for
    the observed conditional mean effect.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    cond = np.asarray(condition, dtype=bool)
    n = len(y)
    n_cond = int(cond.sum())
    if n_cond == 0 or n < 2:
        return {"effect": 0.0, "p_value": 1.0, "n_cond": n_cond, "perm_std": 1.0}
    obs_effect = float(np.mean(y[cond]) - np.mean(y))
    if min_shift is None:
        min_shift = 64
    if n <= 2 * min_shift:
        min_shift = max(1, n // 10)
    shifts = rng.integers(min_shift, n - min_shift + 1, size=n_perm)
    perm_effects = np.empty(n_perm, dtype=float)
    for k, shift in enumerate(shifts):
        y_shift = np.roll(y, int(shift))
        perm_effects[k] = np.mean(y_shift[cond]) - np.mean(y_shift)
    p_val = (1.0 + np.sum(np.abs(perm_effects) >= abs(obs_effect))) / (n_perm + 1.0)
    return {"effect": obs_effect, "p_value": float(p_val), "perm_std": float(np.std(perm_effects)), "n_cond": n_cond}

def benjamini_hochberg_fdr(p_values, q=0.10):
    """True Benjamini-Hochberg FDR procedure, retained for exploratory use."""
    p = np.asarray(p_values, dtype=float); n = len(p)
    if n == 0: return np.array([], dtype=bool), 0.0
    order = np.argsort(p); sorted_p = p[order]
    thresholds = (np.arange(1, n+1)/n) * q
    below = sorted_p <= thresholds
    k = np.max(np.where(below)[0]) + 1 if np.any(below) else 0
    reject = np.zeros(n, dtype=bool)
    if k > 0: reject[order[:k]] = True
    return reject, (thresholds[k-1] if k>0 else 0.0)

def holm_step_down(p_values, alpha=0.05):
    """Holm step-down multiple-testing procedure controlling FWER."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), alpha
    order = np.argsort(p)
    sorted_p = p[order]
    reject = np.zeros(n, dtype=bool)
    cutoff = alpha
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        cutoff = threshold
        if sorted_p[rank] <= threshold:
            reject[idx] = True
        else:
            break
    return reject, cutoff

def benjamini_hochberg(p_values, q=0.05):
    """Compatibility entry point used by the legacy parallel runner."""
    return holm_step_down(p_values, alpha=q)

def bonferroni(p_values, alpha=0.05):
    n = len(p_values); thresh = alpha / n if n>0 else alpha
    return np.asarray(p_values) <= thresh, thresh

def replication_check(y, condition, n_blocks=3, min_effect=0.05, min_pass_fraction=2/3):
    """Check that a fixed discovery rule reproduces a meaningful effect
    across multiple contiguous parts of held-out validation data."""
    y = np.asarray(y, dtype=float)
    condition = np.asarray(condition, dtype=bool)
    n = len(y)
    if n_blocks < 2 or n < n_blocks * 2 or condition.sum() == 0:
        return False, []
    effects = []
    edges = np.linspace(0, n, n_blocks + 1, dtype=int)
    for i in range(n_blocks):
        lo, hi = edges[i], edges[i+1]
        yy = y[lo:hi]
        cc = condition[lo:hi]
        if cc.sum() < 30:
            effects.append(0.0)
            continue
        effects.append(float(np.mean(yy[cc]) - np.mean(yy)))
    nonzero = [e for e in effects if abs(e) >= min_effect]
    if not nonzero:
        return False, effects
    dominant_sign = np.sign(np.sum(nonzero))
    passes = sum(np.sign(e) == dominant_sign and abs(e) >= min_effect for e in effects)
    required = int(np.ceil(n_blocks * min_pass_fraction))
    return bool(passes >= required), effects

def evaluate_hypothesis(y, condition, block_len=64, n_boot=2000, seed=None):
    """Evaluate a fixed held-out rule with a serially-valid null test.

    The circular-shift p-value is primary; replication remains a hard gate.
    """
    result = circular_shift_test(y, condition, n_perm=n_boot, seed=seed, min_shift=block_len)
    replicated, effects = replication_check(y, condition)
    result["replication_pass"] = bool(replicated)
    result["replication_effects"] = effects
    result["boot_std"] = result["perm_std"]
    if not replicated:
        result["p_value"] = 1.0
    return result

def split_discovery_validation(feat, y, frac=0.6, purge=16):
    """Time-ordered split: hypothesis GENERATION only ever sees the discovery
    slice; hypothesis TESTING only ever sees the validation slice. `purge`
    drops bars at the boundary so the forward-looking target cannot leak."""
    n = len(feat)
    disc_end = int(n * frac)
    val_start = disc_end + purge
    feat_disc = feat.iloc[:disc_end]
    y_disc = y.iloc[:disc_end]
    feat_val = feat.iloc[val_start:]
    y_val = y.iloc[val_start:]
    return feat_disc, y_disc, feat_val, y_val

def recompute_condition(cand, feat_df):
    """Re-apply a candidate's discovery-derived rule to held-out data."""
    ctype = cand["type"]
    if ctype == "1way":
        fname = cand["features"][0]
        thresh = cand["thresholds"][0]
        q = cand["quantile"]
        col = feat_df[fname].values
        cond = col < thresh if q < 0.5 else col > thresh
    elif ctype == "2way":
        fname1, fname2 = cand["features"]
        t1, t2 = cand["thresholds"]
        q1, q2 = cand["quantiles"]
        col1 = feat_df[fname1].values
        col2 = feat_df[fname2].values
        c1 = col1 < t1 if q1 < 0.5 else col1 > t1
        c2 = col2 < t2 if q2 < 0.5 else col2 > t2
        cond = c1 & c2
    else:
        raise ValueError(f"Unknown candidate type: {ctype}")
    return cond
