"""v8 frozen pipeline core (parameters: scripts/v8_params.json; spec: docs_v8/FROZEN_SPEC_v8.md).

Taxon-agnostic given the upstream pipeline outputs of a project (clean data,
alignment matrices, record_aux, outlier flags, v3 geography/eligibility).
"""
from __future__ import annotations
import itertools, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
from holdout import config as C
from holdout import distances as A
from holdout import geography as V
from holdout import refsets as S6
from holdout import stats as S7

PARAMS = json.loads((Path(__file__).parent / "params.json").read_text())
SEED = PARAMS["seed"]


# ------------------------------------------------------------------ data
def load():
    D, meta, dist = V.load_v3()
    return D, meta, dist, D["valid"]


def kmer_distance(D, k):
    """Cosine distance between k-mer count profiles of the UNALIGNED main sequences."""
    seqs = {r.id: str(r.seq).upper().replace("-", "") for r in SeqIO.parse(str(C.CLEAN / "main_coi.fasta"), "fasta")}
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    X = np.zeros((len(D["ids"]), 4 ** k), np.float32)
    for i, rid in enumerate(D["ids"]):
        s = seqs[rid]
        code = 0; valid_run = 0; mask = 4 ** k - 1
        for ch in s:
            v = idx.get(ch)
            if v is None:
                valid_run = 0; code = 0; continue
            code = ((code << 2) | v) & mask; valid_run += 1
            if valid_run >= k:
                X[i, code] += 1
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    dk = (1.0 - X @ X.T).astype(np.float32)
    dk[dk < 0] = 0
    np.fill_diagonal(dk, np.nan)
    return dk


# ------------------------------------------------------------------ libraries
def blocker(meta, dist, valid, te, base, strict, thr):
    if strict:
        return S7.blocked_reference_thr(te, base, meta, dist, valid, thr, haplotype=True)
    return V.blocked_reference(te, base, meta, dist, haplotype=False)


def build_all(meta, dist, valid, thr, universe=None):
    """All regimes. universe: optional bool mask restricting BOTH queries and references."""
    elig, elig_loco, testable = S6.eligibility()
    sp = meta.species.to_numpy(); blk = meta.block.to_numpy(); cont = meta.continent.to_numpy()
    ctry = meta.country.fillna("").to_numpy()
    U = meta.in_v3.to_numpy().copy()
    if universe is not None:
        U &= universe
    res = V.resolved(blk)
    Q = np.where(U & res & np.array([s in elig and b in testable.get(s, ()) for s, b in zip(sp, blk)]))[0]
    bc = V.block_country_sets(meta[meta.in_v3.to_numpy()])
    unres = (~res) & (blk != "Unknown")

    def base_lobo(b):
        m = U & res & (blk != b); m |= U & unres & ~np.isin(ctry, list(bc.get(b, set()))); return m

    def base_loco(k):
        m = U & res & (cont != k)
        bad = {"Europe": ("Russia_unresolved",), "Asia": ("Russia_unresolved", "China_unresolved"),
               "North_America": ("Canada_unresolved", "United States_unresolved")}.get(k, ())
        m |= U & unres & ~np.isin(blk, list(bad)) & (cont != k); return m
    L = {}
    rng = np.random.default_rng(SEED + 80)
    grp = V.union_groups(meta, Q)
    for f, te in enumerate(V.grouped_folds(Q, sp[Q], grp, PARAMS["random_cv_folds"], rng)):
        L.setdefault("random_cv_strict", []).append((f"fold{f}", blocker(meta, dist, valid, te, U.copy(), True, thr), te))
    for b in sorted(set(blk[Q])):
        qb = Q[blk[Q] == b]
        if len(qb) < PARAMS["within_block_min_queries"]:
            continue
        g = V.union_groups(meta, qb)
        for f, te in enumerate(V.grouped_folds(qb, sp[qb], g, PARAMS["random_cv_folds"], rng)):
            if len(te):
                L.setdefault("within_block_strict", []).append(
                    (f"{b}_f{f}", blocker(meta, dist, valid, te, U & (blk == b), True, thr), te))
    Ql = Q[np.isin(sp[Q], list(elig_loco))]
    locobase = {k: base_loco(k) for k in sorted(set(cont[Ql]))}
    for v, strict in (("strict", True), ("hapshared", False)):
        for b in sorted(set(blk[Q])):
            te = Q[blk[Q] == b]
            L.setdefault(f"lobo_{v}", []).append((b, blocker(meta, dist, valid, te, base_lobo(b), strict, thr), te))
            K = V.continent_of(b)
            if K in locobase:
                L.setdefault(f"G_{v}", []).append((b, blocker(meta, dist, valid, te, locobase[K], strict, thr), te))
        for k, base in locobase.items():
            te = Ql[cont[Ql] == k]
            L.setdefault(f"loco_{v}", []).append((k, blocker(meta, dist, valid, te, base, strict, thr), te))
    return {"Q": Q, "Ql": Ql, "L": L}


def common_sets(meta, lib):
    rid = meta.record_id.to_numpy()
    ids = {r: set(rid[np.concatenate([te for _, _, te in v])]) for r, v in lib["L"].items() if not r.startswith("G_")}
    ca = set.intersection(*[ids[r] for r in ids if not r.startswith("loco")])
    cl = set.intersection(*ids.values())
    return ca, cl


# ------------------------------------------------------------------ scoring
def score_regimes(meta, dmat, lib, query_idx):
    """credit/state/absent for every regime for the given query indices."""
    sp = meta.species.to_numpy()
    qset = set(query_idx.tolist())
    out = []
    for reg, splits in lib["L"].items():
        for lab, ref, te in splits:
            qk = np.array([q for q in te if q in qset])
            if len(qk) == 0:
                continue
            ev = S6.SpeciesMin(dmat, qk, ref, sp).evaluate()
            out.append(pd.DataFrame({"idx": qk, "regime": reg, "split": lab, "credit": ev["credit"],
                                     "U": (ev["state"] == "U").astype(float), "state": ev["state"], "absent": ev["absent"]}))
    P = pd.concat(out, ignore_index=True)
    P["species"] = sp[P.idx]; P["block"] = meta.block.to_numpy()[P.idx]; P["record_id"] = meta.record_id.to_numpy()[P.idx]
    return P


def wide(P, regimes):
    w = P[P.regime.isin(regimes)].pivot_table(index=["idx", "record_id", "species", "block"], columns="regime",
                                              values=["credit", "U"], aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    return w.reset_index()


# ------------------------------------------------------------------ mechanism
def mechanism(meta, dmat, lib, query_idx, rng=None, controls=0):
    """LOBO -> G -> LOCO per test block, with 4(+1)-player Shapley and optional
    random controls for the G step (C1 size-matched, C2 species-count-matched)."""
    sp = meta.species.to_numpy(); blk = meta.block.to_numpy(); rid = meta.record_id.to_numpy()
    qset = set(query_idx.tolist())
    rows, reps, qm = [], [], []
    for v in ("strict", "hapshared"):
        lo_ = {lab: ref for lab, ref, _ in lib["L"][f"lobo_{v}"]}
        g_ = {lab: ref for lab, ref, _ in lib["L"][f"G_{v}"]}
        co_ = {lab: ref for lab, ref, _ in lib["L"][f"loco_{v}"]}
        prep = {}
        for lab, _, te in lib["L"][f"lobo_{v}"]:
            qB = np.array([q for q in te if q in qset])
            if len(qB) == 0:
                continue
            K = V.continent_of(lab)
            Rl, Rg, Rc = lo_[lab], g_[lab], co_[K]
            un = np.union1d(np.union1d(Rl, Rg), Rc)
            SM = S6.SpeciesMin(dmat, qB, un, sp)
            lo = np.isin(SM.ref_idx, Rl); g = np.isin(SM.ref_idx, Rg); co = np.isin(SM.ref_idx, Rc)
            comp = {"geo_rm": lo & ~g, "geo_add": g & ~lo, "X_removed": g & ~co, "X_added": co & ~g}
            prep[lab] = (SM, lo, g, co, comp)
            ev = {k: SM.evaluate(m) for k, m in (("LOBO", lo), ("G", g), ("LOCO", co))}
            players = ["geo_con", "geo_het", "X_removed", "X_added"] + (["geo_added"] if comp["geo_add"].any() else [])
            tsp = sp[qB]
            for s in sorted(set(tsp)):
                qi = np.where(tsp == s)[0]; con = SM.rsp == s
                parts = {"geo_con": comp["geo_rm"] & con, "geo_het": comp["geo_rm"] & ~con,
                         "X_removed": comp["X_removed"], "X_added": comp["X_added"], "geo_added": comp["geo_add"]}
                P_ = len(players); vals = {}
                for T in itertools.product((0, 1), repeat=P_):
                    on = dict(zip(players, T)); m = lo.copy()
                    for p in ("geo_con", "geo_het", "X_removed"):
                        if on[p]:
                            m &= ~parts[p]
                    if on.get("geo_added"):
                        m |= parts["geo_added"] & ~(parts["X_removed"] if on["X_removed"] else False)
                    if on["X_added"]:
                        m |= parts["X_added"]
                    e = SM.evaluate(m); vals[T] = (e["credit"][qi], (e["state"][qi] == "U").astype(float))
                assert np.allclose(vals[tuple([1] * P_)][0], ev["LOCO"]["credit"][qi])
                phi = {}
                for pi, pn in enumerate(players):
                    ac = np.zeros(len(qi)); au = np.zeros(len(qi))
                    for T in itertools.product((0, 1), repeat=P_):
                        if T[pi]:
                            continue
                        Tp = list(T); Tp[pi] = 1; Tp = tuple(Tp); kk = sum(T)
                        w = math.factorial(kk) * math.factorial(P_ - kk - 1) / math.factorial(P_)
                        ac += w * (vals[Tp][0] - vals[T][0]); au += w * (vals[Tp][1] - vals[T][1])
                    phi[pn] = (ac, au)
                for a, qq in enumerate(qi):
                    r = {"variant": v, "idx": qB[qq], "record_id": rid[qB[qq]], "species": s, "block": lab}
                    for k in ("LOBO", "G", "LOCO"):
                        r[f"credit_{k}"] = ev[k]["credit"][qq]; r[f"U_{k}"] = float(ev[k]["state"][qq] == "U")
                        r[f"state_{k}"] = ev[k]["state"][qq]; r[f"absent_{k}"] = bool(ev[k]["absent"][qq])
                    for pn in ("geo_con", "geo_het", "X_removed", "X_added", "geo_added"):
                        r[f"phi_{pn}_credit"] = phi[pn][0][a] if pn in phi else 0.0
                        r[f"phi_{pn}_U"] = phi[pn][1][a] if pn in phi else 0.0
                    rows.append(r)
        if controls:
            blocks = sorted(prep)
            acc = {}
            for kind in ("C1", "C2"):
                for rep in range(controls):
                    cr, un_ = [], []
                    for lab in blocks:
                        SM, lo, g, co, comp = prep[lab]
                        m = lo.copy(); cols = np.where(lo)[0]
                        if kind == "C1":
                            m[rng.choice(cols, int(comp["geo_rm"].sum()), replace=False)] = False
                        else:
                            for s_, k_ in pd.Series(SM.rsp[comp["geo_rm"]]).value_counts().items():
                                cs = cols[SM.rsp[cols] == s_]
                                m[rng.choice(cs, k_, replace=False)] = False
                        m |= comp["geo_add"]
                        e = SM.evaluate(m); cr.append(e["credit"]); un_.append((e["state"] == "U").astype(float))
                    cr = np.concatenate(cr); un_ = np.concatenate(un_)
                    a_ = acc.setdefault(kind, [np.zeros(len(cr)), np.zeros(len(cr))]); a_[0] += cr; a_[1] += un_
                    reps.append({"variant": v, "kind": kind, "rep": rep})
            order = np.concatenate([prep[lab][0].q_idx for lab in blocks])
            for kind, (c_, u_) in acc.items():
                qm.append(pd.DataFrame({"variant": v, "idx": order, f"credit_{kind}": c_ / controls, f"U_{kind}": u_ / controls}))
    T = pd.DataFrame(rows)
    if qm:
        Q = pd.concat(qm).groupby(["variant", "idx"]).first().reset_index()
        T = T.merge(Q, on=["variant", "idx"], how="left")
    return T


def mechanism_summary(T, rng, nb=None):
    """Per-variant step differences, Shapley means and control contrasts with species/block CIs,
    plus interactions (hapshared - strict)."""
    nb = nb or PARAMS["bootstrap_replicates"]
    out = []
    W = {}
    for v, g in T.groupby("variant"):
        g = g.copy()
        for m in ("credit", "U"):
            tag = "exp" if m == "credit" else "uniq"
            g[f"{tag}_LOBO"] = g[f"{m}_LOBO"]; g[f"{tag}_LOCO"] = g[f"{m}_LOCO"]
            g[f"{tag}_total"] = g[f"{m}_LOCO"] - g[f"{m}_LOBO"]
            g[f"{tag}_G"] = g[f"{m}_G"] - g[f"{m}_LOBO"]; g[f"{tag}_X"] = g[f"{m}_LOCO"] - g[f"{m}_G"]
            for pn in ("geo_con", "geo_het", "X_removed", "X_added", "geo_added"):
                g[f"{tag}_phi_{pn}"] = g[f"phi_{pn}_{m}"]
            if f"{m}_C1" in g:
                g[f"{tag}_G_minus_C1"] = g[f"{m}_G"] - g[f"{m}_C1"]; g[f"{tag}_G_minus_C2"] = g[f"{m}_G"] - g[f"{m}_C2"]
        W[v] = g
        cols = [c for c in g.columns if c.startswith(("exp_", "uniq_"))]
        sb, _ = S7.species_boot(g, cols, rng, nb); bb = S7.block_boot(g, cols, rng, nb)
        for c in cols:
            out.append({"variant": v, "quantity": c, "estimate": round(sb[c][0], 4),
                        "species_lo": round(sb[c][1], 4), "species_hi": round(sb[c][2], 4),
                        "block_lo": round(bb[c][1], 4), "block_hi": round(bb[c][2], 4), "n_queries": len(g),
                        "n_species": g.species.nunique(), "n_blocks": g.block.nunique()})
    if len(W) == 2:
        a = W["strict"].set_index("idx"); b = W["hapshared"].set_index("idx")
        I = pd.DataFrame({"species": a.species, "block": a.block})
        for tag in ("exp", "uniq"):
            for q in ("total", "G", "X"):
                I[f"{tag}_I_{q}"] = b[f"{tag}_{q}"] - a[f"{tag}_{q}"]
        I = I.reset_index()
        cols = [c for c in I.columns if c.startswith(("exp_", "uniq_"))]
        sb, _ = S7.species_boot(I, cols, rng, nb); bb = S7.block_boot(I, cols, rng, nb)
        for c in cols:
            out.append({"variant": "interaction", "quantity": c, "estimate": round(sb[c][0], 4),
                        "species_lo": round(sb[c][1], 4), "species_hi": round(sb[c][2], 4),
                        "block_lo": round(bb[c][1], 4), "block_hi": round(bb[c][2], 4), "n_queries": len(I),
                        "n_species": I.species.nunique(), "n_blocks": I.block.nunique()})
    return pd.DataFrame(out)


def transitions(T):
    rows = []
    for v, g in T.groupby("variant"):
        for (a, b), n in g.groupby(["state_LOBO", "state_LOCO"]).size().items():
            gg = g[(g.state_LOBO == a) & (g.state_LOCO == b)]
            rows.append({"variant": v, "from_LOBO": a, "to_LOCO": b, "n": int(n), "n_absent_LOCO": int(gg.absent_LOCO.sum())})
    return pd.DataFrame(rows)
