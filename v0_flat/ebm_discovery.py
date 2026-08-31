import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

class SimpleEBM:
    def __init__(self, outer_bags=25, bag_frac=0.8, boost_rounds=120, max_depth=3, top_features_for_pairs=30, max_interactions=50, seed=None):
        self.outer_bags = outer_bags
        self.bag_frac = bag_frac
        self.boost_rounds = boost_rounds
        self.max_depth = max_depth
        self.top_features_for_pairs = top_features_for_pairs
        self.max_interactions = max_interactions
        self.seed = seed
        self.feature_names_ = None
        self.bags_ = []
        self.feature_importances_ = None
        self.interaction_importances_ = None
        self.interaction_scores_purified_ = None
        
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_np = X.values
        else:
            self.feature_names_ = [f"f{i}" for i in range(X.shape[1])]
            X_np = np.asarray(X)
        y_np = np.asarray(y)
        n, p = X_np.shape
        self.bags_ = []
        feat_imp_accum = np.zeros(p)
        inter_accum = {}
        for bag in range(self.outer_bags):
            indices = rng.choice(n, size=int(n*self.bag_frac), replace=False)
            X_b = X_np[indices]
            y_b = y_np[indices]
            pred = np.zeros(len(X_b))
            residual = y_b - pred
            feature_trees = {i: [] for i in range(p)}
            feat_contrib = np.zeros(p)
            for _ in range(self.boost_rounds):
                order = rng.permutation(p)
                for fi in order:
                    Xi = X_b[:, fi].reshape(-1,1)
                    if np.std(Xi) < 1e-12:
                        continue
                    tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=50, random_state=rng.integers(0,1e6))
                    tree.fit(Xi, residual)
                    update = tree.predict(Xi) * 0.1
                    pred += update
                    residual = y_b - pred
                    feature_trees[fi].append(tree)
                    feat_contrib[fi] += np.mean(np.abs(update))
            top_idx = np.argsort(feat_contrib)[-self.top_features_for_pairs:]
            pair_scores = []
            for a in range(len(top_idx)):
                for b in range(a+1, len(top_idx)):
                    i = top_idx[a]
                    j = top_idx[b]
                    for qi,qj in [(0.8,0.2),(0.2,0.8)]:
                        col_i = X_b[:, i]
                        col_j = X_b[:, j]
                        thresh_i = np.quantile(col_i, qi)
                        thresh_j = np.quantile(col_j, qj)
                        cond_i = col_i > thresh_i if qi>0.5 else col_i < thresh_i
                        cond_j = col_j > thresh_j if qj>0.5 else col_j < thresh_j
                        cond_ij = cond_i & cond_j
                        n_ij = cond_ij.sum()
                        if n_ij < 100:
                            continue
                        I_i = cond_i.astype(float)
                        I_j = cond_j.astype(float)
                        I_ij = cond_ij.astype(float)
                        X_design = np.column_stack([np.ones(len(y_b)), I_i, I_j, I_ij])
                        try:
                            beta = np.linalg.lstsq(X_design, y_b, rcond=None)[0]
                            beta_ij = beta[3]
                            y_pred = X_design @ beta
                            resid = y_b - y_pred
                            rss = np.sum(resid**2)
                            mse = rss / (len(y_b)-4) if len(y_b)>4 else 1
                            try:
                                XtX_inv = np.linalg.inv(X_design.T @ X_design)
                                var_beta = mse * np.diag(XtX_inv)
                                se = np.sqrt(var_beta[3]) if var_beta[3]>0 else 1
                                t_stat = beta_ij / (se + 1e-12)
                            except:
                                t_stat = beta_ij
                            score = abs(t_stat)
                            pair_scores.append(((i,j,qi,qj,thresh_i,thresh_j), score, beta_ij, n_ij))
                        except:
                            continue
            pair_scores.sort(key=lambda x: x[1], reverse=True)
            inter_trees = {}
            inter_contrib = {}
            for (i,j,qi,qj,ti,tj), score, beta_ij, n_ij in pair_scores[:self.max_interactions]:
                Xij = X_b[:, [i,j]]
                tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=100, random_state=rng.integers(0,1e6))
                tree.fit(Xij, residual)
                update = tree.predict(Xij) * 0.1
                residual = residual - update
                inter_trees[(i,j)] = tree
                inter_contrib[(i,j)] = np.mean(np.abs(update))
                inter_accum[(i,j)] = inter_accum.get((i,j), 0) + inter_contrib[(i,j)]
            feat_imp_accum += feat_contrib
            self.bags_.append({
                "indices": indices, 
                "feature_trees": feature_trees, 
                "inter_trees": inter_trees, 
                "feat_contrib": feat_contrib, 
                "inter_contrib": inter_contrib, 
                "top_idx": top_idx,
                "pair_scores": pair_scores
            })
        self.feature_importances_ = feat_imp_accum / self.outer_bags
        self.interaction_importances_ = {k: v/self.outer_bags for k,v in inter_accum.items()}
        all_pair_scores = {}
        for bag in self.bags_:
            for (i,j,qi,qj,ti,tj), score, beta_ij, n_ij in bag["pair_scores"]:
                key = (i,j)
                if key not in all_pair_scores:
                    all_pair_scores[key] = []
                all_pair_scores[key].append(score)
        self.interaction_scores_purified_ = {k: np.mean(v) for k,v in all_pair_scores.items()}
        return self
    
    def get_top_features(self, k=12):
        idx = np.argsort(self.feature_importances_)[-k:][::-1]
        return [(self.feature_names_[i], self.feature_importances_[i], i) for i in idx]
    
    def get_top_interactions(self, k=12):
        if self.interaction_scores_purified_:
            sorted_purified = sorted(self.interaction_scores_purified_.items(), key=lambda x: x[1], reverse=True)[:k]
            result = []
            for (i,j), score in sorted_purified:
                result.append((self.feature_names_[i], self.feature_names_[j], score, (i,j)))
            return result
        else:
            sorted_inter = sorted(self.interaction_importances_.items(), key=lambda x: x[1], reverse=True)[:k]
            result = []
            for (i,j), imp in sorted_inter:
                result.append((self.feature_names_[i], self.feature_names_[j], imp, (i,j)))
            return result
    
    def generate_candidate_hypotheses(self, X, y, min_samples=500, effect_thresh=0.12, max_candidates=50):
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)
        y_np = np.asarray(y)
        candidates = []
        top_feats = self.get_top_features(k=self.top_features_for_pairs)
        for fname, imp, fi in top_feats:
            col = X_np[:, fi]
            for q in [0.10, 0.15, 0.20, 0.80, 0.85]:
                thresh = np.quantile(col, q)
                cond = col < thresh if q < 0.5 else col > thresh
                n_cond = cond.sum()
                if n_cond < min_samples or n_cond > len(col)*0.5:
                    continue
                effect = np.mean(y_np[cond]) - np.mean(y_np)
                if abs(effect) < effect_thresh:
                    continue
                stable = 0
                for bag in self.bags_:
                    idx = bag["indices"]
                    X_b = X_np[idx]
                    y_b = y_np[idx]
                    col_b = X_b[:, fi]
                    cond_b = col_b < thresh if q<0.5 else col_b > thresh
                    if cond_b.sum() < 50:
                        continue
                    if np.sign(np.mean(y_b[cond_b]) - np.mean(y_b)) == np.sign(effect):
                        stable += 1
                stability = stable / len(self.bags_)
                if stability < 0.6:
                    continue
                cond_str = f"{fname} < {thresh:.4f} ({int(q*100)}th)" if q<0.5 else f"{fname} > {thresh:.4f} ({int(q*100)}th)"
                candidates.append({
                    "type": "1way", "features": [fname], "feature_indices": [fi],
                    "thresholds": [thresh], "quantile": q, "condition_str": cond_str,
                    "n_samples": int(n_cond), "effect_size": float(effect),
                    "stability": float(stability), "importance": float(imp),
                    "condition_mask": cond
                })
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        if self.interaction_scores_purified_:
            sorted_pairs = sorted(self.interaction_scores_purified_.items(), key=lambda x: x[1], reverse=True)[:self.max_interactions*2]
            for (fi,fj), score in sorted_pairs:
                col1 = X_np[:, fi]
                col2 = X_np[:, fj]
                for q1,q2 in [(0.8,0.2),(0.2,0.8),(0.8,0.8),(0.2,0.2)]:
                    t1 = np.quantile(col1, q1)
                    t2 = np.quantile(col2, q2)
                    c1 = col1 < t1 if q1<0.5 else col1 > t1
                    c2 = col2 < t2 if q2<0.5 else col2 > t2
                    cond = c1 & c2
                    n_cond = cond.sum()
                    if n_cond < min_samples:
                        continue
                    effect = np.mean(y_np[cond]) - np.mean(y_np)
                    if abs(effect) < effect_thresh:
                        continue
                    stable = 0
                    for bag in self.bags_:
                        idx = bag["indices"]
                        X_b = X_np[idx]
                        y_b = y_np[idx]
                        col1_b = X_b[:, fi]
                        col2_b = X_b[:, fj]
                        c1_b = col1_b < t1 if q1<0.5 else col1_b > t1
                        c2_b = col2_b < t2 if q2<0.5 else col2_b > t2
                        cond_b = c1_b & c2_b
                        if cond_b.sum() < 30:
                            continue
                        if np.sign(np.mean(y_b[cond_b]) - np.mean(y_b)) == np.sign(effect):
                            stable += 1
                    stability = stable / len(self.bags_)
                    if stability < 0.6:
                        continue
                    fname1 = self.feature_names_[fi]
                    fname2 = self.feature_names_[fj]
                    cond_str = f"{fname1} {'<' if q1<0.5 else '>'} {t1:.3f} AND {fname2} {'<' if q2<0.5 else '>'} {t2:.3f}"
                    candidates.append({
                        "type": "2way", "features": [fname1, fname2], "feature_indices": [fi, fj],
                        "thresholds": [t1, t2], "quantiles": [q1,q2], "condition_str": cond_str,
                        "n_samples": int(n_cond), "effect_size": float(effect),
                        "stability": float(stability), "importance": float(score),
                        "condition_mask": cond
                    })
                    if len(candidates) >= max_candidates*2:
                        break
                if len(candidates) >= max_candidates*2:
                    break
        candidates.sort(key=lambda x: abs(x["effect_size"])*x["stability"], reverse=True)
        return candidates[:max_candidates]
