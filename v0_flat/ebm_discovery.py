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
from config_final import ONE_WAY_CANDIDATE_FRACTION


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
        self.top_main_effect_pool_ = [int(i) for i in top_idx]
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

        # Keep the correlation-dedup map for diagnostics and for the EBM
        # interaction-recall channel, but DO NOT replace a top-ranked feature
        # with a correlated proxy when constructing the actual hypotheses.
        #
        # The injected B2 signal is deliberately attached to the original
        # feature identity. A correlated representative can be a useful proxy,
        # but substituting it here can erase a real edge before held-out
        # validation. Discovery selection is allowed to choose among these
        # hypotheses; validation remains the gatekeeper.
        for a_pos, i in enumerate(top_idx):
            for j in top_idx[a_pos + 1:]:
                i, j = int(i), int(j)
                pair_keys.add((i, j) if i < j else (j, i))

        self.pair_keys_generated_ = set(pair_keys)

        # Preserve the EBM interaction-recall channel. Map correlated members
        # to representatives only for EBM-recalled terms; top-pool hypotheses
        # retain their original feature identities.
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
        """Trace a named pair through each discovery stage without changing discovery."""
        names = {name: i for i, name in enumerate(self.feature_names_)}
        ia, ib = names.get(feature_a), names.get(feature_b)
        out = {"feature_a": feature_a, "feature_b": feature_b}
        if ia is None or ib is None:
            out.update({"feature_a_known": ia is not None, "feature_b_known": ib is not None})
            return out

        ia, ib = int(ia), int(ib)
        original_pair = tuple(sorted((ia, ib))) if ia != ib else None
        top_main = set(getattr(self, "top_main_effect_pool_", []))
        pair_pool = set(getattr(self, "pair_pool_before_dedup_", []))
        interaction_importances = getattr(self, "interaction_importances_", {})

        out["feature_a_in_top_main_effect_pool"] = ia in top_main
        out["feature_b_in_top_main_effect_pool"] = ib in top_main
        out["both_in_top_main_effect_pool"] = ia in top_main and ib in top_main
        out["pair_in_ebm_interaction_recall"] = bool(
            original_pair is not None and original_pair in interaction_importances
        )
        out["pair_in_pair_pool_before_dedup"] = bool(
            original_pair is not None and ia in pair_pool and ib in pair_pool
        )

        mapped_a = getattr(self, "dedup_feature_map_", {}).get(ia, ia)
        mapped_b = getattr(self, "dedup_feature_map_", {}).get(ib, ib)
        out["dedup_representatives"] = [
            self.feature_names_[mapped_a], self.feature_names_[mapped_b]
        ]
        out["dedup_collapsed_pair"] = bool(mapped_a == mapped_b and ia != ib)
        out["dedup_survives"] = not out["dedup_collapsed_pair"]

        pair = tuple(sorted((int(mapped_a), int(mapped_b)))) if mapped_a != mapped_b else None
        generated_pairs = getattr(self, "pair_keys_generated_", set())
        final_pair_keys = getattr(self, "pair_keys_", set())
        purified_scores = getattr(self, "interaction_scores_purified_", {})

        out["pair_in_all_pairs_after_dedup"] = bool(
            pair is not None and pair in generated_pairs
        )
        out["pair_in_final_pair_keys"] = bool(
            pair is not None and pair in final_pair_keys
        )
        out["pair_in_purified_scores"] = bool(
            pair is not None and pair in purified_scores
        )

        if pair is not None and pair in purified_scores:
            ranked = sorted(purified_scores.items(), key=lambda x: x[1], reverse=True)
            out["purified_score"] = float(purified_scores[pair])
            out["purified_rank"] = next(
                (i + 1 for i, (k, _) in enumerate(ranked) if k == pair), None
            )
            out["purified_pairs_total"] = len(ranked)
        else:
            out["purified_score"] = None
            out["purified_rank"] = None
            out["purified_pairs_total"] = len(purified_scores)

        if candidates is not None:
            generated_pair_cands = getattr(self, "last_two_way_candidates_", [])
            target_generated = None
            for i, cand in enumerate(generated_pair_cands, 1):
                fs = set(cand.get("features", []))
                if fs == {self.feature_names_[mapped_a], self.feature_names_[mapped_b]}:
                    target_generated = (i, cand)
                    break

            pair_cands = [x for x in candidates if x.get("type") == "2way"]
            target = None
            for i, cand in enumerate(pair_cands, 1):
                fs = set(cand.get("features", []))
                if fs == {self.feature_names_[mapped_a], self.feature_names_[mapped_b]}:
                    target = (i, cand)
                    break

            out["generated_two_way_candidate_count"] = len(generated_pair_cands)
            out["generated_two_way_rank"] = target_generated[0] if target_generated else None
            out["generated_two_way_candidate"] = target_generated[1] if target_generated else None
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

        # Correlation-aware alias rescue:
        # A highly correlated proxy can carry the same discovered interaction
        # signal as the injected/original feature, while the original pair may
        # have a weaker purified score because the proxy is numerically cleaner.
        # Deduplication therefore defines families, not identities. We keep the
        # original pair AND expand the highest-scoring pair families into a
        # bounded set of member aliases. The final candidate budget remains
        # unchanged, and held-out validation still controls false discoveries.
        pair_queue = []
        seen_pairs = set()
        clusters = getattr(self, "dedup_clusters_", [])
        family_by_feature = {}
        for cluster in clusters:
            members = [self.feature_names_.index(name) for name in cluster["members"]]
            for member in members:
                family_by_feature[member] = members

        ranked_pairs = sorted(
            self.interaction_scores_purified_.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for (fi, fj), score in ranked_pairs:
            base = tuple(sorted((int(fi), int(fj))))
            expansions = [(base[0], base[1])]
            fam_i = family_by_feature.get(base[0], [base[0]])
            fam_j = family_by_feature.get(base[1], [base[1]])
            # Bound alias expansion per pair family so correlated feature
            # clusters cannot consume the whole discovery budget.
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
            # Once enough families have been queued, later families are still
            # represented by their base pair; aliases are reserved for the
            # strongest discovery signals.
            if len(pair_queue) >= max(100, max_candidates * 4):
                break

        # Evaluate aliases using their own raw conditional effect/stability.
        # We intentionally do not copy the proxy's score into the candidate;
        # 'importance' remains the purified score that motivated the family.
        for (fi, fj), score in pair_queue:
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
        one_budget = max(1, int(round(max_candidates * ONE_WAY_CANDIDATE_FRACTION)))
        two_budget = max_candidates - one_budget
        one = self._generate_oneway(X_np, y_np, min_samples, effect_thresh, one_budget)
        two = self._generate_twoway(X_np, y_np, min_samples, effect_thresh, two_budget)
        self.last_one_way_candidates_ = list(one)
        self.last_two_way_candidates_ = list(two)
        out = one + two
        out.sort(key=lambda x: abs(x["effect_size"]) * x["stability"], reverse=True)
        return out[:max_candidates]
