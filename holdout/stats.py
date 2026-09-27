"""v7 helpers.

1. Corrected cluster bootstrap for species-balanced means.
   species_boot: resample species WITH their multiplicity (equivalent to
   resampling per-species means; each draw is its own cluster), fixing the v6
   bug where duplicated species were merged by name.
   block_boot: resample test blocks; species means are recomputed from
   multiplicity-weighted block sums/counts (duplicated blocks keep their weight).
2. Strict blocking with a configurable identity definition (min overlap for a
   zero-distance match), and LOBO/LOCO/R_G construction (copy of v6 logic; v6
   code is left unchanged).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from holdout import config as C
from holdout import geography as V
from holdout import refsets as S6

NB = 10000


def species_boot(df, cols, rng, nb=NB):
    """df: per-query rows with 'species' and value columns. Returns dict col -> (est, lo, hi, samples)."""
    x = df.groupby("species")[cols].mean()
    X = x.to_numpy(); n = len(X)
    idx = rng.integers(0, n, size=(nb, n))
    B = X[idx].mean(1)
    return {c: (float(X[:, i].mean()), float(np.quantile(B[:, i], 0.025)), float(np.quantile(B[:, i], 0.975)))
            for i, c in enumerate(cols)}, B


def block_boot(df, cols, rng, nb=NB):
    sp = df.species.to_numpy(); bl = df.block.to_numpy()
    su, si = np.unique(sp, return_inverse=True); bu, bi = np.unique(bl, return_inverse=True)
    out = {}
    cnt = np.zeros((len(su), len(bu))); np.add.at(cnt, (si, bi), 1)
    picks = rng.integers(0, len(bu), size=(nb, len(bu)))
    mult = np.stack([np.bincount(p, minlength=len(bu)) for p in picks])     # nb x nblocks
    den = mult @ cnt.T                                                      # nb x nspecies
    for c in cols:
        s = np.zeros((len(su), len(bu))); np.add.at(s, (si, bi), df[c].to_numpy())
        num = mult @ s.T
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(den > 0, num / den, np.nan)
        B = np.nanmean(m, axis=1)
        est = float(df.groupby("species")[c].mean().mean())
        out[c] = (est, float(np.quantile(B, 0.025)), float(np.quantile(B, 0.975)))
    return out


def blocked_reference_thr(test_idx, base_mask, meta, dist, valid, min_overlap_identity=300, haplotype=True):
    """As geography.blocked_reference, but a zero-distance match counts as identity
    only if the comparable overlap is >= min_overlap_identity (raw identical
    sequences and shared specimen keys are always blocked)."""
    m = base_mask.copy(); m[test_idx] = False
    hh = meta.seq_hash.to_numpy()
    if haplotype:
        m &= ~np.isin(hh, list(set(hh[test_idx])))
    sk = meta.specimen_key.to_numpy(); tk = set(sk[test_idx]) - {""}
    if tk:
        m &= ~np.isin(sk, list(tk))
    if haplotype:
        cand = np.where(m)[0]; zero = np.zeros(len(cand), bool)
        for k in range(0, len(test_idx), 512):
            t = test_idx[k:k + 512]
            zero |= np.any((dist[np.ix_(t, cand)] == 0) & (valid[np.ix_(t, cand)] >= min_overlap_identity), axis=0)
        m[cand[zero]] = False
    return np.where(m)[0]


def build_geo_v7(meta, dist, valid=None, identity_overlap=300, with_RG=True):
    """LOBO / LOCO libraries (strict & hapshared) plus the intermediate R_G(B):
    LOCO geographic base of B's continent, blocked by B's LOBO test queries.
    identity_overlap applies to strict blocking (300 reproduces v4)."""
    elig, elig_loco, testable = S6.eligibility()
    sp = meta.species.to_numpy(); blk = meta.block.to_numpy(); cont = meta.continent.to_numpy()
    ctry = meta.country.fillna("").to_numpy()
    U = meta.in_v3.to_numpy().copy(); res = V.resolved(blk)
    Q = np.where(U & res & np.array([s in elig and b in testable.get(s, ()) for s, b in zip(sp, blk)]))[0]
    bcountries = V.block_country_sets(meta[meta.in_v3.to_numpy()])
    unres = (~res) & (blk != "Unknown")

    def geo_base(test, level):
        if level == "lobo":
            hold_c = set().union(*[bcountries.get(b, set()) for b in test])
            m = U & res & ~np.isin(blk, list(test)); m |= U & unres & ~np.isin(ctry, list(hold_c))
        else:
            m = U & res & (cont != test)
            bad = {"Europe": ("Russia_unresolved",), "Asia": ("Russia_unresolved", "China_unresolved"),
                   "North_America": ("Canada_unresolved", "United States_unresolved")}.get(test, ())
            m |= U & unres & ~np.isin(blk, list(bad)) & (cont != test)
        return m

    def blocker(te, base, hap):
        if valid is None or identity_overlap == 300:
            return V.blocked_reference(te, base, meta, dist, haplotype=hap)
        return blocked_reference_thr(te, base, meta, dist, valid, identity_overlap, haplotype=hap)
    refs = {}
    Ql = Q[np.isin(sp[Q], list(elig_loco))]
    loco_bases = {k: geo_base(k, "loco") for k in sorted(set(cont[Ql]))}
    for k, base in loco_bases.items():
        te = Ql[cont[Ql] == k]
        refs[("strict", "loco", k)] = (blocker(te, base, True), te)
        refs[("hapshared", "loco", k)] = (blocker(te, base, False), te)
    for b in sorted(set(blk[Q])):
        te = Q[blk[Q] == b]; base = geo_base({b}, "lobo")
        refs[("strict", "lobo", b)] = (blocker(te, base, True), te)
        refs[("hapshared", "lobo", b)] = (blocker(te, base, False), te)
        K = V.continent_of(b)
        if with_RG and K in loco_bases:
            refs[("strict", "G", b)] = (blocker(te, loco_bases[K], True), te)
            refs[("hapshared", "G", b)] = (blocker(te, loco_bases[K], False), te)
    return {"Q": Q, "Ql": Ql, "refs": refs}
