import numpy as np
import pandas as pd

# --- ATTEMPT #1 - REAL EBM ONLY - NO FALLBACK - FAILS LOUDLY ---
try:
    from interpret.glassbox import ExplainableBoostingRegressor
    EBM_AVAILABLE = True
    _import_error = None
except Exception as e:
    EBM_AVAILABLE = False
    _import_error = e
    ExplainableBoostingRegressor = None

class SimpleEBM:
    """
    Drop-in replacement for your old SimpleEBM but using REAL ExplainableBoostingRegressor.
    Keeps same public API: fit, get_top_features, get_top_interactions, generate_candidate_hypotheses
    """
    def __init__(self, outer_bags=25, bag_frac=0.8, boost_rounds=120, max_depth=3, top_features_for_pairs=50, max_interactions=75, seed=None):
        if not EBM_AVAILABLE:
            raise RuntimeError(
                f"REAL EBM REQUIRED BUT NOT AVAILABLE: {_import_error}. "
                "Install: Python 3.10 + pip install interpret==0.4.4 scikit-learn==1.3.2 pandas==2.1.4 numpy==1.26.2 scipy==1.11.4 . NO FALLBACK ALLOWED."
            )
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
        self._ebm_model = None

    def fit(self, X, y):
        if not EBM_AVAILABLE:
            raise RuntimeError("REAL EBM REQUIRED - aborting, no SimpleEBM fallback")

        rng = np.random.default_rng(self.seed)
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_np = X.values
            X_for_ebm = X
        else:
            self.feature_names_ = [f"f{i}" for i in range(X.shape[1])]
            X_np = np.asarray(X)
            X_for_ebm = X_np
        y_np = np.asarray(y)
        n, p = X_np.shape

        # Use the configured interaction budget. The previous code accidentally
        # used top_features_for_pairs (50) and ignored MAX_INTERACTIONS (75).
        interaction_search = min(int(self.max_interactions), max(0, p * (p - 1) // 2))
        self._ebm_model = ExplainableBoostingRegressor(
            outer_bags=self.outer_bags,
            learning_rate=0.01,
            max_leaves=self.max_depth,
            interactions=interaction_search,
            random_state=self.seed if self.seed is not None else 42
        )
        self._ebm_model.fit(X_for_ebm, y_np)

        if hasattr(self._ebm_model, 'term_importances_'):
            term_importances = self._ebm_model.term_importances_
        elif hasattr(self._ebm_model, 'term_importances'):
            term_importances = self._ebm_model.term_importances
        else:
            raise RuntimeError("EBM has no term_importances attribute - incompatible interpret version")

        term_names = getattr(self._ebm_model, 'term_names_', None)
        if term_names is None:
            term_names = getattr(self._ebm_model, 'feature_names_in_', self.feature_names_)

        feat_imp = np.zeros(p)
        inter_accum = {}
        for t_idx, t_name in enumerate(term_names):
            imp = float(term_importances[t_idx]) if t_idx < len(term_importances) else 0.0
            if isinstance(t_name, str) and " x " in t_name:
                parts = [s.strip() for s in t_name.split(" x ")]
                if len(parts) == 2:
                    try:
                        i = self.feature_names_.index(parts[0])
                        j = self.feature_names_.index(parts[1])
                        key = (i, j) if i < j else (j, i)
                        inter_accum[key] = inter_accum.get(key, 0.0) + imp
                    except ValueError:
                        continue
            else:
                try:
                    if isinstance(t_name, str):
                        idx = self.feature_names_.index(t_name)
                    else:
                        idx = int(t_name) if str(t_name).startswith("f") else self.feature_names_.index(str(t_name))
                    feat_imp[idx] += imp
                except Exception:
                    if isinstance(t_name, str) and t_name.startswith("f"):
                        try:
                            idx = int(t_name[1:])
                            if 0 <= idx < p:
                                feat_imp[idx] += imp
                        except Exception:
                            pass

        self.feature_importances_ = feat_imp
        self.interaction_importances_ = inter_accum
        self.interaction_scores_purified_ = {k: v for k, v in inter_accum.items()}

        self.bags_ = []
        for bag in range(self.outer_bags):
            indices = rng.choice(n, size=int(n*self.bag_frac), replace=False)
            self.bags_.append({"indices": indices})

        return self

    def get_top_features(self, k=12):
        idx = np.argsort(self.feature_importances_)[-k:][::-1]
        return [(self.feature_names_[i], self.feature_importances_[i], i) for i in idx]

    def get_top_interactions(self, k=12):
        if self.interaction_scores_purified_:
            sorted_purified = sorted(self.interaction_scores_purified_.items(), key=lambda x: x[1], reverse=True)[:k]
            return [(self.feature_names_[i], self.feature_names_[j], score, (i,j)) for (i,j), score in sorted_purified]
        sorted_inter = sorted(self.interaction_importances_.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.feature_names_[i], self.feature_names_[j], imp, (i,j)) for (i,j), imp in sorted_inter]

    def _generate_oneway(self, X_np, y_np, min_samples, effect_thresh, max_candidates):
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
                    col_b = X_np[idx, fi]
                    y_b = y_np[idx]
                    cond_b = col_b < thresh if q < 0.5 else col_b > thresh
                    if cond_b.sum() < 50:
                        continue
                    if np.sign(np.mean(y_b[cond_b]) - np.mean(y_b)) == np.sign(effect):
                        stable += 1
                stability = stable / len(self.bags_) if self.bags_ else 1.0
                if stability < 0.6:
                    continue
                candidates.append({"type":"1way", "features":[fname], "feature_indices":[fi], "thresholds":[thresh], "quantile":q, "condition_str":f"{fname} {'<' if q<0.5 else '>'} {thresh:.4f} ({int(q*100)}th)", "n_samples":int(n_cond), "effect_size":float(effect), "stability":float(stability), "importance":float(imp), "condition_mask":cond})
        candidates.sort(key=lambda x: abs(x["effect_size"])*x["stability"], reverse=True)
        return candidates[:max_candidates]

    def _generate_twoway(self, X_np, y_np, min_samples, effect_thresh, max_candidates):
        candidates = []
        if not self.interaction_scores_purified_:
            return candidates
        sorted_pairs = sorted(self.interaction_scores_purified_.items(), key=lambda x: x[1], reverse=True)
        for (fi,fj), score in sorted_pairs:
            col1 = X_np[:, fi]; col2 = X_np[:, fj]
            for q1,q2 in [(0.8,0.2),(0.2,0.8),(0.8,0.8),(0.2,0.2)]:
                t1 = np.quantile(col1, q1); t2 = np.quantile(col2, q2)
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
                    col1_b = X_np[idx, fi]; col2_b = X_np[idx, fj]; y_b = y_np[idx]
                    c1_b = col1_b < t1 if q1<0.5 else col1_b > t1
                    c2_b = col2_b < t2 if q2<0.5 else col2_b > t2
                    cond_b = c1_b & c2_b
                    if cond_b.sum() < 30:
                        continue
                    if np.sign(np.mean(y_b[cond_b]) - np.mean(y_b)) == np.sign(effect):
                        stable += 1
                stability = stable / len(self.bags_) if self.bags_ else 1.0
                if stability < 0.6:
                    continue
                fname1 = self.feature_names_[fi]; fname2 = self.feature_names_[fj]
                candidates.append({"type":"2way", "features":[fname1,fname2], "feature_indices":[fi,fj], "thresholds":[t1,t2], "quantiles":[q1,q2], "condition_str":f"{fname1} {'<' if q1<0.5 else '>'} {t1:.3f} AND {fname2} {'<' if q2<0.5 else '>'} {t2:.3f}", "n_samples":int(n_cond), "effect_size":float(effect), "stability":float(stability), "importance":float(score), "condition_mask":cond})
        # Evaluate all searched pairs before truncating. Returning early when the
        # first max_candidates are found can crowd out a genuine weak interaction.
        candidates.sort(key=lambda x: abs(x["effect_size"])*x["stability"], reverse=True)
        return candidates[:max_candidates]

    def generate_candidate_hypotheses(self, X, y, min_samples=500, effect_thresh=0.12, max_candidates=50):
        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_np = np.asarray(y)
        one_way_budget = max(1, max_candidates // 2)
        two_way_budget = max_candidates - one_way_budget
        one_way = self._generate_oneway(X_np, y_np, min_samples, effect_thresh, one_way_budget)
        two_way = self._generate_twoway(X_np, y_np, min_samples, effect_thresh, two_way_budget)
        candidates = one_way + two_way
        candidates.sort(key=lambda x: abs(x["effect_size"])*x["stability"], reverse=True)
        return candidates[:max_candidates]
