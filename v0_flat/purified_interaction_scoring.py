"""
Purified Interaction Scoring - Final Version
"""
import numpy as np

def purified_interaction_score(feat, y, i, j, qi=0.8, qj=0.2, min_samples=200):
    col_i = feat.iloc[:, i].values
    col_j = feat.iloc[:, j].values
    thresh_i = np.quantile(col_i, qi)
    thresh_j = np.quantile(col_j, qj)
    cond_i = col_i > thresh_i if qi>0.5 else col_i < thresh_i
    cond_j = col_j > thresh_j if qj>0.5 else col_j < thresh_j
    cond_ij = cond_i & cond_j
    n_ij = cond_ij.sum()
    if n_ij < min_samples:
        return None
    I_i = cond_i.astype(float)
    I_j = cond_j.astype(float)
    I_ij = cond_ij.astype(float)
    X = np.column_stack([np.ones(len(y)), I_i, I_j, I_ij])
    try:
        beta = np.linalg.lstsq(X, y.values, rcond=None)[0]
        beta_ij = beta[3]
        y_pred = X @ beta
        resid = y.values - y_pred
        rss = np.sum(resid**2)
        mse = rss / (len(y)-4) if len(y)>4 else 1
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            var_beta = mse * np.diag(XtX_inv)
            se = np.sqrt(var_beta[3]) if var_beta[3]>0 else 1
            t_stat = beta_ij / (se + 1e-12)
        except:
            t_stat = beta_ij
        effect_ij = y[cond_ij].mean() - y.mean()
        return {"score": abs(t_stat), "t_stat": t_stat, "beta_ij": beta_ij, "effect_ij": effect_ij, "n_ij": n_ij, "cond": cond_ij, "thresh_i": thresh_i, "thresh_j": thresh_j, "qi": qi, "qj": qj}
    except:
        return None

def generate_purified_candidates(feat, y, top_features_idx, min_samples=200, max_candidates=50):
    feat_names = list(feat.columns)
    all_cands=[]
    for a in range(len(top_features_idx)):
        for b in range(a+1, len(top_features_idx)):
            i=top_features_idx[a]; j=top_features_idx[b]
            for qi,qj in [(0.8,0.2),(0.2,0.8)]:
                res = purified_interaction_score(feat, y, i, j, qi=qi, qj=qj, min_samples=min_samples)
                if res is None: continue
                cond_str = f"{feat_names[i]} {'>' if qi>0.5 else '<'} {res['thresh_i']:.3f} AND {feat_names[j]} {'>' if qj>0.5 else '<'} {res['thresh_j']:.3f}"
                all_cands.append({"features": [feat_names[i], feat_names[j]], "feature_indices": [i,j], "condition_mask": res["cond"], "condition_str": cond_str, "effect_size": res["effect_ij"], "purified_score": res["score"], "beta_ij": res["beta_ij"], "n_samples": res["n_ij"], "t_stat": res["score"]})
    all_cands.sort(key=lambda x: abs(x["purified_score"]), reverse=True)
    return all_cands[:max_candidates]
