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

def benjamini_hochberg(p_values, q=0.10):
    """Benjamini-Hochberg FDR control. Kept for exploratory/FDR use.
    It controls expected false-discovery proportion, not the probability of
    at least one false rejection (FWER)."""
    p = np.asarray(p_values); n = len(p)
    if n == 0: return np.array([], dtype=bool), 0.0
    order = np.argsort(p); sorted_p = p[order]
    thresholds = (np.arange(1, n+1)/n) * q
    below = sorted_p <= thresholds
    k = np.max(np.where(below)[0]) + 1 if np.any(below) else 0
    reject = np.zeros(n, dtype=bool)
    if k > 0: reject[order[:k]] = True
    return reject, (thresholds[k-1] if k>0 else 0.0)

def holm_step_down(p_values, alpha=0.05):
    """Holm step-down multiple-testing procedure.

    Controls family-wise error rate (FWER): the probability of one or more
    false rejections, under arbitrary dependence of valid p-values. This is
    the appropriate correction for Test A/C, whose acceptance criterion is
    an empirical probability of at least one false discovery per market.
    """
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
            # Holm is step-down: once one ordered hypothesis fails, all later
            # hypotheses are not rejected.
            break
    return reject, cutoff

def bonferroni(p_values, alpha=0.05):
    n = len(p_values); thresh = alpha / n if n>0 else alpha
    return np.asarray(p_values) <= thresh, thresh

def evaluate_hypothesis(y, condition, block_len=64, n_boot=2000, seed=None):
    return circular_block_bootstrap(y, condition, block_len=block_len, n_boot=n_boot, seed=seed)

def split_discovery_validation(feat, y, frac=0.6, purge=16):
    """Time-ordered split: hypothesis GENERATION only ever sees the discovery
    slice; hypothesis TESTING only ever sees the validation slice. `purge`
    drops a few bars at the boundary so the forward-looking target (which
    peeks `horizon` bars ahead) can't leak discovery-period information into
    the validation target, or vice versa."""
    n = len(feat)
    disc_end = int(n * frac)
    val_start = disc_end + purge
    feat_disc = feat.iloc[:disc_end]
    y_disc = y.iloc[:disc_end]
    feat_val = feat.iloc[val_start:]
    y_val = y.iloc[val_start:]
    return feat_disc, y_disc, feat_val, y_val

def recompute_condition(cand, feat_df):
    """Re-apply a candidate's discovery-set-derived rule (feature name +
    fixed threshold + direction) to a *different* dataset (the held-out
    validation slice). This is the piece that was missing: candidates must
    be tested on data they were not selected to fit."""
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
