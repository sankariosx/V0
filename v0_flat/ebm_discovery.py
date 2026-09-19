import numpy as np
import pandas as pd

# Real EBM is mandatory. There is intentionally no fallback model.
try:
    from interpret.glassbox import ExplainableBoostingRegressor
    EBM_AVAILABLE = True
    _import_error = None
except Exception as e:
    EBM_AVAILABLE = False
    _import_error = e

from purified_interaction_scoring import purified_interaction_score


class SimpleEBM:
    """Real EBM feature discovery plus explicitly purified pair discovery.

    The EBM ranks individual features and also provides a direct interaction
    recall channel. Pair candidates are then purified on the discovery slice
    with a main-effect-adjusted interaction coefficient. Final significance is
    evaluated only on the untouched validation slice in tests.py.
    """

    def __init__(self, outer_bags=25, bag_frac=0.8, boost_rounds=120,
                 max_depth=3, top_features_for_pairs=50,
                 max_interactions=75, seed=None):
        if not EBM_AVAILABLE:
            raise RuntimeError(
                f"REAL EBM REQUIRED BUT NOT AVAILABLE: {_import_error}. "
                "Install: Python 3.10 + pip install interpret==0.4.4 "
                "scikit-learn==1.3.2 pandas==2.1.4 numpy==1.26.2 "
                "scipy==1.11.4 . NO FALLBACK ALLOWED."
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
        self.interaction_importances_ = {}
        self.interaction_scores_purified_ = {}
        self._ebm_model = None

    def _deduplicate_feature_pool(self, X_df, feature_indices, corr_threshold=0.90):
        """Cluster highly correlated discovery features and keep one representative.

        This is deliberately done on the DISCOVERY slice only, before any pair
        combinations are generated. Representatives are chosen by main-effect
        importance, with feature index as a deterministic tie-breaker.
        """
        indices = sorted(set(int(i) for i in feature_indices))
        if len(indices) < 2:
            return indices, {i: i for i in indices}, []

        data = X_df.iloc[:, indices].astype(float)
        corr = data.corr().abs().fillna(0.0)
        parent = {i: i for i in indices}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a_pos, i in enumerate(indices):
            for j in indices[a_pos + 1:]:
                if corr.loc[self.feature_names_[i], self.feature_names_[j]] > corr_threshold:
                    union(i, j)

        clusters = {}
        for i in indices:
            clusters.setdefault(find(i), []).append(i)

        representatives = []
        mapping = {}
        cluster_info = []
        for members in clusters.values():
            rep = max(members, key=lambda i: (float(self.feature_importances_[i]), -i))
            representatives.append(rep)
            for i in members:
                mapping[i] = rep
            cluster_info.append({
                "representative": self.feature_names_[rep],
                "members": [self.feature_names_[i] for i in members],
                "size": len(members)
            })

        representatives.sort(key=lambda i: (-float(self.feature_importances_[i]), i))
        return representatives, mapping, cluster_info

    def fit(self, X, y):
        if not EBM_AVAILABLE:
            raise RuntimeError("REAL EBM REQUIRED - aborting, no fallback")

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

        interaction_search = min(self.max_interactions, p * (p - 1) // 2)
        self._ebm_model = ExplainableBoostingRegressor(
            outer_bags=self.outer_bags,
            learning_rate=0.01,
            max_leaves=self.max_depth,
            interactions=interaction_search,
            random_state=self.seed if self.seed is not None else 42
        )
        self._ebm_model.fit(X_for_ebm, y_np)

        term_importances = getattr(self._ebm_model, "term_importances_", None)
        if term_importances is None:
            term_importances = getattr(self._ebm_model, "term_importances", None)
        if callable(term_importances):
            term_importances = term_importances()
        if term_importances is None:
            raise RuntimeError("EBM has no usable term_importances attribute")

        term_names = getattr(self._ebm_model, "term_names_", None)
        if callable(term_names):
            term_names = term_names()
        if term_names is None:
            term_names = getattr(self._ebm_model, "feature_names_in_", self.feature_names_)
            if callable(term_names):
                term_names = term_names()

        feat_imp = np.zeros(p)
        inter_accum = {}
        for t_idx, t_name in enumerate(term_names):
            imp = float(term_importances[t_idx]) if t_idx < len(term_importances) else 0.0
            if isinstance(t_name, str) and " x " in t_name:
                parts = [s.strip() for s in t_name.split(" x ")]
                if len(parts) == 2 and parts[0] in self.feature_names_ and parts[1] in self.feature_names_:
                    i = self.feature_names_.index(parts[0]); j = self.feature_names_.index(parts[1])
                    key = (i, j) if i < j else (j, i)
                    inter_accum[key] = inter_accum.get(key, 0.0) + imp
            else:
                try:
                    idx = self.feature_names_.index(str(t_name))
                    feat_imp[idx] += imp
                except ValueError:
                    if isinstance(t_name, str) and t_name.startswith("f"):
                        try:
                            idx = int(t_name[1:])
                            if 0 <= idx < p:
                                feat_imp[idx] += imp
                        except Exception:
                            pass

        self.feature_importances_ = feat_imp
        self.interaction_importances_ = inter_accum

        # Pair recall uses TWO independent discovery channels:
        #   1) all pairs among the top main-effect features
        #   2) the EBM's selected interaction terms
        #
        # IMPORTANT: deduplicate highly correlated features BEFORE generating
        # pair combinations. Otherwise near-copies (for example atr_100_norm
        # and an almost identical copy) create redundant hypotheses, inflate
        # the multiple-testing burden, and can make the same underlying signal
        # appear many times. Correlation is computed on discovery data only.
        top_idx = np.argsort(self.feature_importances_)[-self.top_features_for_pairs:][::-1]
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X_np, columns=self.feature_names_)
        pair_pool = list(dict.fromkeys([int(i) for i in top_idx] + [int(i) for key in sorted(inter_accum) for i in key]))
        self.pair_pool_before_dedup_ = list(pair_pool)
        dedup_pool, feature_map, dedup_clusters = self._deduplicate_feature_pool(X_df, pair_pool, corr_threshold=0.90)
        self.dedup_clusters_ = dedup_clusters
        self.dedup_feature_map_ = feature_map

        pair_keys = set()
        for a in range(len(dedup_pool)):
            for b in range(a + 1, len(dedup_pool)):
                i, j = int(dedup_pool[a]), int(dedup_pool[b])
                pair_keys.add((i, j) if i < j else (j, i))

        self.pair_keys_ = pair_keys

        # Preserve the EBM interaction-recall channel, but map correlated
        # members to their representatives and discard self-pairs.
        for key, _ in sorted(inter_accum.items(), key=lambda x: x[1], reverse=True)[:self.max_interactions]:
            i, j = feature_map.get(int(key[0]), int(key[0])), feature_map.get(int(key[1]), int(key[1]))
            if i != j:
                pair_keys.add((i, j) if i < j else (j, i))

        # pair_keys is finalized below after the EBM interaction-recall additions.
        purified = {}
        y_series = pd.Series(y_np, index=X_df.index)
        for i, j in sorted(pair_keys):
            res = purified_interaction_score(
                X_df, y_series, i, j,
                qi=0.8, qj=0.2, min_samples=200
            )
            res_rev = purified_interaction_score(
                X_df, y_series, i, j,
                qi=0.2, qj=0.8, min_samples=200
            )
            best = None
            for r in (res, res_rev):
                if r is not None and (best is None or abs(r["score"]) > abs(best["score"])):
                    best = r
            if best is not None:
                purified[(i, j)] = float(abs(best["score"]))
        self.interaction_scores_purified_ = purified
        self.pair_keys_ = set(purified.keys())
        print(f"  Correlation dedup: {len(pair_pool)} candidate features -> {len(dedup_pool)} representatives; {len(dedup_clusters)} clusters")

        # Bootstrap bags are used only for stability of candidate direction;
        # the validation slice remains completely untouched.
        self.bags_ = []
        for _ in range(self.outer_bags):
            indices = rng.choice(n, size=int(n * self.bag_frac), replace=False)
            self.bags_.append({"indices": indices})
        return self

    def diagnose_pair(self, feature_a, feature_b, candidates=None):
        """Trace a named pair through discovery without changing discovery behavior."""
        names = {name: i for i, name in enumerate(self.feature_names_)}
        ia, ib = names.get(feature_a), names.get(feature_b)
        out = {"feature_a": feature_a, "feature_b": feature_b}
        if ia is None or ib is None:
            out.update({"initial_top_pool": False, "pair_pool": False, "dedup_survives": False, "pair_generated": False})
            return out
        out["initial_top_pool"] = bool(ia in getattr(self, "pair_pool_before_dedup_", []))
        mapped_a = getattr(self, "dedup_feature_map_", {}).get(ia, ia)
        mapped_b = getattr(self, "dedup_feature_map_", {}).get(ib, ib)
        out["dedup_representatives"] = [self.feature_names_[mapped_a], self.feature_names_[mapped_b]]
        out["dedup_survives"] = bool(mapped_a != mapped_b or ia == ib)
        pair = tuple(sorted((int(mapped_a), int(mapped_b)))) if mapped_a != mapped_b else None
        pair_keys = getattr(self, "pair_keys_", set())
        out["pair_generated"] = bool(pair is not None and pair in pair_keys)
        scores = getattr(self, "interaction_scores_purified_", {})
        if pair is not None and pair in scores:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            out["purified_score"] = float(scores[pair])
            out["purified_rank"] = next((i + 1 for i, (k, _) in enumerate(ranked) if k == pair), None)
            out["purified_pairs_total"] = len(ranked)
        else:
            out["purified_score"] = None
            out["purified_rank"] = None
            out["purified_pairs_total"] = len(scores)
        if candidates is not None:
            pair_cands = [x for x in candidates if x.get("type") == "2way"]
            target = None
            for i, cand in enumerate(pair_cands, 1):
                fs = set(cand.get("features", []))
                if fs == {self.feature_names_[mapped_a], self.feature_names_[mapped_b]}:
                    target = (i, cand)
                    break
            out["final_two_way_candidate_count"] = len(pair_cands)
            out["final_candidate_rank"] = target[0] if target else None
            out["final_candidate"] = target[1] if target else None
        return out

    def get_top_features(self, k=12):
        idx = np.argsort(self.feature_importances_)[-k:][::-1]
        return [(self.feature_names_[i], self.feature_importances_[i], int(i)) for i in idx]

    def get_top_interactions(self, k=12):
        pairs = sorted(self.interaction_scores_purified_.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.feature_names_[i], self.feature_names_[j], score, (i, j))
                for (i, j), score in pairs]

    def _generate_oneway(self, X_np, y_np, min_samples, effect_thresh, max_candidates):
        candidates = []
        top_feats = self.get_top_features(k=self.top_features_for_pairs)
        for fname, imp, fi in top_feats:
            col = X_np[:, fi]
            for q in [0.10, 0.15, 0.20, 0.80, 0.85]:
                thresh = np.quantile(col, q)
                cond = col < thresh if q < 0.5 else col > thresh
                n_cond = int(cond.sum())
                if n_cond < min_samples or n_cond > len(col) * 0.5:
                    continue
                effect = np.mean(y_np[cond]) - np.mean(y_np)
                if abs(effect) < effect_thresh:
                    continue
                stable = 0
                for bag in self.bags_:
                    idx = bag["indices"]
                    cb = X_np[idx, fi]
                    yb = y_np[idx]
                    cond_b = cb < thresh if q < 0.5 else cb > thresh
                    if cond_b.sum() >= 50 and np.sign(np.mean(yb[cond_b]) - np.mean(yb)) == np.sign(effect):
                        stable += 1
                stability = stable / len(self.bags_) if self.bags_ else 1.0
                if stability >= 0.6:
                    candidates.append({
                        "type": "1way", "features": [fname], "feature_indices": [fi],
                        "thresholds": [thresh], "quantile": q,
                        "condition_str": f"{fname} {'<' if q < 0.5 else '>'} {thresh:.4f} ({int(q*100)}th)",
                        "n_samples": n_cond, "effect_size": float(effect),
                        "stability": float(stability), "importance": float(imp),
                        "condition_mask": cond
                    })
        candidates.sort(key=lambda x: abs(x["effect_size"]) * x["stability"], reverse=True)
        return candidates[:max_candidates]

    def _generate_twoway(self, X_np, y_np, min_samples, effect_thresh, max_candidates):
        candidates = []
        for (fi, fj), score in sorted(self.interaction_scores_purified_.items(), key=lambda x: x[1], reverse=True):
            col1, col2 = X_np[:, fi], X_np[:, fj]
            for q1, q2 in [(0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.2, 0.2)]:
                t1, t2 = np.quantile(col1, q1), np.quantile(col2, q2)
                c1 = col1 < t1 if q1 < 0.5 else col1 > t1
                c2 = col2 < t2 if q2 < 0.5 else col2 > t2
                cond = c1 & c2
                n_cond = int(cond.sum())
                if n_cond < min_samples:
                    continue
                effect = np.mean(y_np[cond]) - np.mean(y_np)
                if abs(effect) < effect_thresh:
                    continue
                stable = 0
                for bag in self.bags_:
                    idx = bag["indices"]
                    c1b = col1[idx] < t1 if q1 < 0.5 else col1[idx] > t1
                    c2b = col2[idx] < t2 if q2 < 0.5 else col2[idx] > t2
                    cb = c1b & c2b
                    if cb.sum() >= 30 and np.sign(np.mean(y_np[idx][cb]) - np.mean(y_np[idx])) == np.sign(effect):
                        stable += 1
                stability = stable / len(self.bags_) if self.bags_ else 1.0
                if stability >= 0.6:
                    n1, n2 = self.feature_names_[fi], self.feature_names_[fj]
                    candidates.append({
                        "type": "2way", "features": [n1, n2], "feature_indices": [fi, fj],
                        "thresholds": [t1, t2], "quantiles": [q1, q2],
                        "condition_str": f"{n1} {'<' if q1 < 0.5 else '>'} {t1:.3f} AND {n2} {'<' if q2 < 0.5 else '>'} {t2:.3f}",
                        "n_samples": n_cond, "effect_size": float(effect),
                        "stability": float(stability), "importance": float(score),
                        "condition_mask": cond
                    })
        candidates.sort(key=lambda x: abs(x["effect_size"]) * x["stability"], reverse=True)
        return candidates[:max_candidates]

    def generate_candidate_hypotheses(self, X, y, min_samples=500, effect_thresh=0.12, max_candidates=50):
        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_np = np.asarray(y)
        one_budget = max(1, max_candidates // 2)
        two_budget = max_candidates - one_budget
        one = self._generate_oneway(X_np, y_np, min_samples, effect_thresh, one_budget)
        two = self._generate_twoway(X_np, y_np, min_samples, effect_thresh, two_budget)
        out = one + two
        out.sort(key=lambda x: abs(x["effect_size"]) * x["stability"], reverse=True)
        return out[:max_candidates]
