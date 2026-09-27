"""Geographic hold-out audit of a COI reference library (one genus).

run() compares, for the same queries, identification with the query's block held
out (LOBO) and with its continent held out (LOCO), and returns:
  transitions   outcome changes U/A/W between the two scenarios
  components    net change split into conspecific loss and competitor removal
  influential   reference records whose removal resolves the most failures
  rule          cost of the 1% rule (P1) relative to nearest species (P0)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

from holdout import config as C
from holdout import geography as V
from holdout import refsets as S6
from holdout import scenarios as K

TOL = 1e-6
STATES = ("U", "A", "W")


@dataclass
class AuditResult:
    genus: str
    library: str
    summary: dict
    transitions: pd.DataFrame
    components: pd.DataFrame
    species: pd.DataFrame
    influential: pd.DataFrame
    removal_curve: pd.DataFrame
    rule: pd.DataFrame
    queries: pd.DataFrame = field(repr=False)

    def save(self, folder=None):
        out = C.RESULTS / "audit" if folder is None else folder
        out.mkdir(parents=True, exist_ok=True)
        for name in ("transitions", "components", "species", "influential", "removal_curve", "rule", "queries"):
            getattr(self, name).to_csv(out / f"{name}.tsv", sep="\t", index=False)
        pd.Series(self.summary).to_csv(out / "summary.tsv", sep="\t", header=False)
        return out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out


def _species_boot(values, species, rng, n=10000):
    S = pd.DataFrame({"v": values, "s": species}).groupby("s").v.mean().to_numpy()
    B = S[rng.integers(0, len(S), (n, len(S)))].mean(1)
    return S.mean(), np.quantile(B, 0.025), np.quantile(B, 0.975)


def _states(M, tcol):
    d1 = M.min(1)
    tie = (M <= d1[:, None] + TOL) & np.isfinite(M)
    r = np.arange(len(M)); has = tcol >= 0
    hit = np.zeros(len(M), bool); hit[has] = tie[r[has], tcol[has]]
    n = tie.sum(1)
    st = np.where(~np.isfinite(d1), "N", np.where(hit & (n == 1), "U", np.where(hit, "A", "W")))
    return st, tie


def _libraries(meta, dist, valid):
    """hapshared splits: region represented (random CV), block unseen, continent unseen."""
    thr = K.PARAMS["strict_identity_min_overlap"]
    lib = K.build_all(meta, dist, valid, thr)
    ca, cl = K.common_sets(meta, lib)
    rid = meta.record_id.to_numpy(); sp = meta.species.to_numpy(); Q = lib["Q"]
    U = meta.in_v3.to_numpy().copy()
    rng = np.random.default_rng(K.SEED + 80)
    folds = V.grouped_folds(Q, sp[Q], V.union_groups(meta, Q), K.PARAMS["random_cv_folds"], rng)
    R = [(f"fold{f}", K.blocker(meta, dist, valid, te, U.copy(), False, thr), te) for f, te in enumerate(folds)]
    plan = []
    for ctx, splits, keep in (("R", R, ca), ("B", lib["L"]["lobo_hapshared"], ca), ("C", lib["L"]["loco_hapshared"], cl)):
        for lab, ref, te in splits:
            te = np.array([q for q in te if rid[q] in keep])
            if len(te):
                plan.append((ctx, lab, np.asarray(ref), te))
    return lib, ca, cl, plan


def _split(dist, sp, ref, te):
    order = np.argsort(sp[ref], kind="stable"); ref = ref[order]
    spp, starts = np.unique(sp[ref], return_index=True)
    seg = np.diff(np.r_[starts, len(ref)])
    sub = dist[np.ix_(te, ref)]; sub = np.where(np.isnan(sub), np.inf, sub).astype(np.float32)
    pos = {x: i for i, x in enumerate(spp)}
    return ref, starts, seg, sub, np.array([pos.get(x, -1) for x in sp[te]])


def _influence(meta, dist, plan, top_k=(1, 5, 10, 20, 50)):
    sp = meta.species.to_numpy(); rid = meta.record_id.to_numpy(); N = len(meta)
    fix = np.zeros(N); brk = np.zeros(N); cache = []; base = []
    victims = {}
    for ctx, lab, ref0, te in plan:
        ref, starts, seg, sub, tcol = _split(dist, sp, ref0, te)
        M = np.minimum.reduceat(sub, starts, axis=1)
        st0, tie0 = _states(M, tcol)
        at_min = sub <= np.repeat(M, seg, axis=1) + TOL
        cnt = np.add.reduceat(at_min.astype(np.int32), starts, axis=1)
        m2 = np.minimum.reduceat(np.where(at_min, np.inf, sub), starts, axis=1)
        argm = np.stack([a + np.argmin(sub[:, a:a + n], axis=1) for a, n in zip(starts, seg)], axis=1)
        for i in range(len(te)):
            if st0[i] == "N":
                continue
            for j in np.where(tie0[i])[0]:
                if cnt[i, j] != 1:
                    continue
                Mi = M[i].copy(); Mi[j] = m2[i, j]
                s1 = _states(Mi[None, :], tcol[i:i + 1])[0][0]
                if s1 == st0[i]:
                    continue
                r = ref[argm[i, j]]
                if s1 == "U":
                    fix[r] += 1; victims.setdefault(r, {}).setdefault(sp[te[i]], 0); victims[r][sp[te[i]]] += 1
                elif st0[i] == "U":
                    brk[r] += 1
        base.append(st0); cache.append((ref, starts, sub, tcol))
    base = np.concatenate(base)
    rank = pd.DataFrame({"record_id": rid, "label": sp, "source": meta.source.to_numpy(), "country": meta.country.to_numpy(),
                         "bin": meta.bold_bin.to_numpy(), "resolved": fix, "worsened": brk})
    rank["net"] = rank.resolved - rank.worsened
    rank["affects"] = ["; ".join(f"{k.split()[-1]} {v}" for k, v in sorted(victims[i].items(), key=lambda x: -x[1])[:3])
                       if i in victims else "" for i in range(N)]
    rank = rank.sort_values(["net", "resolved"], ascending=False)
    rank = rank[rank.net > 0].reset_index(drop=True)
    failures = int(np.isin(base, ["A", "W"]).sum())
    curve = []
    for k in top_k:
        drop = np.isin(np.arange(N), [np.where(rid == r)[0][0] for r in rank.record_id[:k]])
        new = np.concatenate([_states(np.minimum.reduceat(np.where(~drop[ref][None, :], sub, np.inf), starts, axis=1), tcol)[0]
                              for ref, starts, sub, tcol in cache])
        curve.append({"removed": min(k, len(rank)), "resolved": int((np.isin(base, ["A", "W"]) & (new == "U")).sum()),
                      "worsened": int(((base == "U") & np.isin(new, ["A", "W"])).sum())})
    curve = pd.DataFrame(curve); curve["share_of_failures"] = curve.resolved / max(failures, 1)
    return rank, curve, failures, len(base)


def _rule(meta, dist, lib, ca, cl):
    sp = meta.species.to_numpy(); rid = meta.record_id.to_numpy(); rows = []
    for ctx, reg, keep in (("block unseen", "lobo_hapshared", ca), ("continent unseen", "loco_hapshared", cl)):
        for lab, ref, te in lib["L"][reg]:
            te = np.array([q for q in te if rid[q] in keep])
            if not len(te):
                continue
            sm = S6.SpeciesMin(dist, te, ref, sp); ev = sm.evaluate()
            M = np.sort(np.minimum.reduceat(sm.sub, sm.starts, axis=1), 1)
            d1 = M[:, 0]; d2 = M[:, 1] if M.shape[1] > 1 else np.full(len(te), np.inf)
            ok = np.isfinite(d1)
            p0 = (ev["nS"] == 1) & (d1 <= 0.03)
            p1 = p0 & (d1 <= 0.01) & (d2 > 0.01)
            rows.append(pd.DataFrame({"scenario": ctx, "species": sp[te], "correct": ev["state"] == "U",
                                      "p0": p0, "p1": p1})[ok])
    R = pd.concat(rows, ignore_index=True); out = []
    for ctx, g in R.groupby("scenario", sort=False):
        moved = g.p0 & ~g.p1
        w, c = int((moved & ~g.correct).sum()), int((moved & g.correct).sum())
        out.append({"scenario": ctx, "queries": len(g), "single_P0": g.p0.mean(), "single_P1": g.p1.mean(),
                    "wrong_among_single_P0": (g.p0 & ~g.correct).sum() / max(g.p0.sum(), 1),
                    "wrong_among_single_P1": (g.p1 & ~g.correct).sum() / max(g.p1.sum(), 1),
                    "wrong_withheld": w, "correct_withheld": c, "correct_per_wrong": c / w if w else np.nan})
    return pd.DataFrame(out)


def run(library="hapshared", influence=True, seed=None):
    """Run the audit for the genus set with holdout.use(); returns an AuditResult."""
    rng = np.random.default_rng(K.SEED + 1 if seed is None else seed)
    D, meta, dist, valid = K.load()
    lib, ca, cl, plan = _libraries(meta, dist, valid)
    rid = meta.record_id.to_numpy()
    T = K.mechanism(meta, dist, lib, np.where(np.isin(rid, list(cl)))[0], rng=rng, controls=0)
    T = T[T.variant == library].copy()
    T["change"] = T.credit_LOCO - T.credit_LOBO
    T["changed"] = (T.state_LOBO != T.state_LOCO).astype(float)

    tr = (T.groupby(["state_LOBO", "state_LOCO"]).size().rename("queries").reset_index()
          .rename(columns={"state_LOBO": "block_unseen", "state_LOCO": "continent_unseen"}))
    tr["share"] = tr.queries / len(T)

    comps = []
    for name, col in (("net change", "change"), ("conspecific loss", "phi_geo_con_credit"),
                      ("competitor removal", "phi_geo_het_credit"), ("blocking change", None)):
        v = (T.phi_X_removed_credit + T.phi_X_added_credit + T.phi_geo_added_credit) if col is None else T[col]
        est, lo, hi = _species_boot(v.to_numpy(), T.species.to_numpy(), rng)
        comps.append({"component": name, "estimate": est, "ci_low": lo, "ci_high": hi})
    comps = pd.DataFrame(comps)
    if library == "hapshared":
        comps = comps[comps.component != "blocking change"].reset_index(drop=True)

    turnover_sw = _species_boot(T.changed.to_numpy(), T.species.to_numpy(), rng)
    species = (T.groupby("species").agg(queries=("idx", "size"), recall_block=("credit_LOBO", "mean"),
                                         recall_continent=("credit_LOCO", "mean"), changed=("changed", "mean"))
               .reset_index())
    uw = T[(T.state_LOBO == "U") & (T.state_LOCO == "W")].groupby("species").size()
    au = T[(T.state_LOBO == "A") & (T.state_LOCO == "U")].groupby("species").size()
    species["U_to_W"] = species.species.map(uw).fillna(0).astype(int)
    species["A_to_U"] = species.species.map(au).fillna(0).astype(int)
    species = species.sort_values("changed", ascending=False).reset_index(drop=True)

    if influence:
        infl, curve, failures, pairs = _influence(meta, dist, plan)
    else:
        infl, curve, failures, pairs = pd.DataFrame(), pd.DataFrame(), np.nan, np.nan
    rule = _rule(meta, dist, lib, ca, cl)

    share = lambda a, b: float(((T.state_LOBO == a) & (T.state_LOCO == b)).mean())
    net = comps.set_index("component").loc["net change"]
    summary = {
        "genus": C.GENUS, "library": library, "queries": len(T), "species": T.species.nunique(),
        "net_change_pp": 100 * net.estimate, "net_change_ci_pp": (100 * net.ci_low, 100 * net.ci_high),
        "changed_outcome_species_weighted": turnover_sw[0], "changed_outcome_ci": turnover_sw[1:],
        "changed_outcome_pooled": float(T.changed.mean()),
        "U_to_W": share("U", "W"), "A_to_U": share("A", "U"),
        "failures": failures, "query_scenario_pairs": pairs,
    }
    return AuditResult(C.GENUS, library, summary, tr, comps, species, infl, curve, rule, T)
