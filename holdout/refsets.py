"""v6 helpers: LOBO/LOCO reference construction with optional record masks and
alternative distance matrices, and fast vectorised state/credit computation.

build_geo() reproduces the LOBO and LOCO splits of 18_v3/27_v4 exactly when
keep=None and dist is the standard core p-distance matrix (checked in
36_v6_transitions.py against results/v4/splits_check_v4.tsv and v4 credits).

States (v4 tie rule, S = species at minimum distance within TOL):
  U  S == {true}            A  true in S, |S| > 1
  W  S non-empty, true not in S      N  no comparable reference (all distances NaN)
Flag 'absent': no reference record of the true species in the library.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from holdout import config as C
from holdout import geography as V

TOL = 1e-6


def eligibility():
    el = pd.read_csv(C.RESULTS / "v3" / "eligibility_v3.tsv", sep="\t")
    elig = set(el[el.eligible_v3].species); elig_loco = set(el[el.eligible_loco].species)
    testable = {r.species: set(str(r.testable_blocks).split(";")) for r in el[el.eligible_v3].itertuples()}
    return elig, elig_loco, testable


def build_geo(meta, dist, keep=None):
    """Return dict with Q, Ql, and refs[(variant, level, key)] = (ref_idx, test_idx).
    variant in {strict, hapshared}; level in {lobo, loco}."""
    elig, elig_loco, testable = eligibility()
    sp = meta.species.to_numpy(); blk = meta.block.to_numpy(); cont = meta.continent.to_numpy()
    ctry = meta.country.fillna("").to_numpy()
    U = meta.in_v3.to_numpy().copy()
    if keep is not None:
        U &= keep
    res = V.resolved(blk)
    Q = np.where(U & res & np.array([s in elig and b in testable.get(s, ()) for s, b in zip(sp, blk)]))[0]
    bcountries = V.block_country_sets(meta[meta.in_v3.to_numpy()])   # as in v3 (country sets of full set)
    unres = (~res) & (blk != "Unknown")

    def geo_base(test, level):
        if level == "lobo":
            hold_c = set().union(*[bcountries.get(b, set()) for b in test])
            m = U & res & ~np.isin(blk, list(test))
            m |= U & unres & ~np.isin(ctry, list(hold_c))
        else:
            m = U & res & (cont != test)
            bad = {"Europe": ("Russia_unresolved",), "Asia": ("Russia_unresolved", "China_unresolved"),
                   "North_America": ("Canada_unresolved", "United States_unresolved")}.get(test, ())
            m |= U & unres & ~np.isin(blk, list(bad)) & (cont != test)
        return m
    refs, bases = {}, {}
    for b in sorted(set(blk[Q])):
        te = Q[blk[Q] == b]; base = geo_base({b}, "lobo"); bases[("lobo", b)] = base
        refs[("strict", "lobo", b)] = (V.blocked_reference(te, base, meta, dist), te)
        refs[("hapshared", "lobo", b)] = (V.blocked_reference(te, base, meta, dist, haplotype=False), te)
    Ql = Q[np.isin(sp[Q], list(elig_loco))]
    for k in sorted(set(cont[Ql])):
        te = Ql[cont[Ql] == k]; base = geo_base(k, "loco"); bases[("loco", k)] = base
        refs[("strict", "loco", k)] = (V.blocked_reference(te, base, meta, dist), te)
        refs[("hapshared", "loco", k)] = (V.blocked_reference(te, base, meta, dist, haplotype=False), te)
    return {"Q": Q, "Ql": Ql, "refs": refs, "bases": bases}


class SpeciesMin:
    """Per-query, per-species minimum distance over a (masked) reference set."""

    def __init__(self, dist, q_idx, ref_idx, sp):
        self.q_idx = np.asarray(q_idx); self.ref_idx = np.asarray(ref_idx)
        rsp = sp[self.ref_idx]
        order = np.argsort(rsp, kind="stable")
        self.ref_idx = self.ref_idx[order]; rsp = rsp[order]
        self.species, self.starts = np.unique(rsp, return_index=True)
        sub = dist[np.ix_(self.q_idx, self.ref_idx)]
        self.sub = np.where(np.isnan(sub), np.inf, sub).astype(np.float32)
        self.rsp = rsp
        tsp = sp[self.q_idx]
        pos = {s: i for i, s in enumerate(self.species)}
        self.tcol = np.array([pos.get(s, -1) for s in tsp])

    def evaluate(self, keep_mask=None):
        """keep_mask: bool over (sorted) ref columns; returns dict of arrays."""
        sub = self.sub if keep_mask is None else np.where(keep_mask[None, :], self.sub, np.inf)
        M = np.minimum.reduceat(sub, self.starts, axis=1)          # nq x nspecies
        d1 = M.min(1)
        fin = np.isfinite(d1)
        tie = (M <= d1[:, None] + TOL) & np.isfinite(M)
        nS = tie.sum(1)
        r = np.arange(len(d1)); has = self.tcol >= 0
        dcon = np.full(len(d1), np.inf); dcon[has] = M[r[has], self.tcol[has]]
        true_in = np.zeros(len(d1), bool); true_in[has] = tie[r[has], self.tcol[has]]
        Mh = M.copy(); Mh[r[has], self.tcol[has]] = np.inf
        dhet = Mh.min(1)
        credit = np.where(true_in, 1.0 / np.maximum(nS, 1), 0.0)
        state = np.where(~fin, "N", np.where(true_in & (nS == 1), "U", np.where(true_in, "A", "W")))
        if keep_mask is None:
            present = has.copy()
        else:
            cnt = np.add.reduceat(keep_mask.astype(int), self.starts)
            present = np.zeros(len(d1), bool); present[has] = cnt[self.tcol[has]] > 0
        return {"credit": credit, "state": state, "d1": d1, "nS": nS, "dcon": dcon, "dhet": dhet,
                "absent": ~present, "tie": tie}

    def tie_species(self, tie_row):
        return ";".join(self.species[tie_row])


def species_balanced(true, values):
    s = pd.Series(values).groupby(pd.Series(true)).mean()
    return float(s.mean())


def score_regimes(meta, dist, query_ids, keep=None, ref_exclude=None):
    """Build LOBO/LOCO (strict & hapshared) with optional record mask `keep`
    (applied to queries AND references, as a changed analysis set) and optional
    `ref_exclude` (bool mask removed from references only, after blocking), then
    score the LOCO queries whose record_id is in query_ids.
    Returns per-query DataFrame with credit/state/absent for the 4 regimes and a
    dict of library sizes."""
    from holdout import geography as V_
    sp = meta.species.to_numpy(); blk = meta.block.to_numpy(); rid = meta.record_id.to_numpy()
    G = build_geo(meta, dist, keep=keep)
    Ql = G["Ql"]; Qs = Ql[np.isin(rid[Ql], list(query_ids))]
    out = {"record_id": rid[Qs], "species": sp[Qs], "block": blk[Qs]}
    sizes = {}
    pos = {q: i for i, q in enumerate(Qs)}
    for v in ("strict", "hapshared"):
        for lev in ("lobo", "loco"):
            cred = np.zeros(len(Qs)); st = np.empty(len(Qs), dtype=object); ab = np.zeros(len(Qs), bool)
            n_ref = 0
            for (vv, ll, key), (ref, te) in G["refs"].items():
                if vv != v or ll != lev:
                    continue
                if ref_exclude is not None:
                    ref = ref[~ref_exclude[ref]]
                qk = te[np.isin(te, Qs)]
                if len(qk) == 0:
                    continue
                n_ref += len(ref)
                ev = SpeciesMin(dist, qk, ref, sp).evaluate()
                ii = [pos[q] for q in qk]
                cred[ii] = ev["credit"]; st[ii] = ev["state"]; ab[ii] = ev["absent"]
            out[f"credit_{v}_{lev}"] = cred; out[f"state_{v}_{lev}"] = st; out[f"absent_{v}_{lev}"] = ab
            sizes[f"{v}_{lev}_ref_records_summed_over_splits"] = n_ref
    return pd.DataFrame(out), sizes


def regime_summary(df, label):
    rows = []
    for v in ("strict", "hapshared"):
        r = {"analysis": label, "variant": v, "n_queries": len(df), "n_species": df.species.nunique()}
        for lev in ("lobo", "loco"):
            r[f"exp_{lev}"] = round(species_balanced(df.species, df[f"credit_{v}_{lev}"]), 4)
            r[f"uniq_{lev}"] = round(species_balanced(df.species, (df[f"state_{v}_{lev}"] == "U").astype(float)), 4)
            r[f"fracA_{lev}"] = round(float((df[f"state_{v}_{lev}"] == "A").mean()), 4)
        r["delta_exp_loco_minus_lobo"] = round(r["exp_loco"] - r["exp_lobo"], 4)
        r["delta_uniq_loco_minus_lobo"] = round(r["uniq_loco"] - r["uniq_lobo"], 4)
        rows.append(r)
    out = pd.DataFrame(rows)
    inter = {"analysis": label, "variant": "interaction(hapshared-strict)",
             "delta_exp_loco_minus_lobo": round(out.delta_exp_loco_minus_lobo.iloc[1] - out.delta_exp_loco_minus_lobo.iloc[0], 4),
             "delta_uniq_loco_minus_lobo": round(out.delta_uniq_loco_minus_lobo.iloc[1] - out.delta_uniq_loco_minus_lobo.iloc[0], 4)}
    return pd.concat([out, pd.DataFrame([inter])], ignore_index=True)
