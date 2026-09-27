"""v8 frozen pipeline run.

  python run_scenarios.py [main|thr300|thr640|kmer|blast|temporal|coord|all]   (blast: v8.1 addendum)
Outputs: results/v8/<analysis>/*.tsv
"""
from __future__ import annotations
import itertools, math, sys, time
import numpy as np
import pandas as pd
import common
from holdout import config as C
from holdout import geography as V
from holdout import refsets as S6
from holdout import stats as S7
from holdout import scenarios as K

P8 = K.PARAMS
OUT = C.RESULTS / "v8"; OUT.mkdir(exist_ok=True)
NB = P8["bootstrap_replicates"]


def regime_metrics(meta, dmat, lib, ca, cl, rng, outdir):
    rid = meta.record_id.to_numpy()
    qidx = np.where(np.isin(rid, list(ca | cl)))[0]
    P = K.score_regimes(meta, dmat, lib, qidx)
    P.to_csv(outdir / "regime_query_scores.tsv", sep="\t", index=False)
    rows = []
    for qs_name, qs in (("common_all", ca), ("common_loco", cl)):
        regs = [r for r in P.regime.unique() if not r.startswith("G_") and (qs_name == "common_loco" or not r.startswith("loco"))]
        w = K.wide(P[P.record_id.isin(qs)], regs)
        for r in regs:
            for m in ("credit", "U"):
                rows.append({"query_set": qs_name, "regime": r, "metric": "expected" if m == "credit" else "unique",
                             "value": round(w.groupby("species")[f"{m}_{r}"].mean().mean(), 4),
                             "n_queries": len(w), "n_species": w.species.nunique()})
        cons = [("lobo_strict", "random_cv_strict"), ("within_block_strict", "random_cv_strict"),
                ("lobo_hapshared", "lobo_strict")]
        if qs_name == "common_loco":
            cons += [("loco_strict", "lobo_strict"), ("loco_hapshared", "lobo_hapshared"),
                     ("loco_strict", "random_cv_strict")]
        for m in ("credit", "U"):
            for a, b in cons:
                w[f"d_{a}_minus_{b}_{m}"] = w[f"{m}_{a}"] - w[f"{m}_{b}"]
        cols = [c for c in w.columns if c.startswith("d_")]
        sb, _ = S7.species_boot(w, cols, rng, NB); bb = S7.block_boot(w, cols, rng, NB)
        for c in cols:
            rows.append({"query_set": qs_name, "regime": c, "metric": "contrast", "value": round(sb[c][0], 4),
                         "species_lo": round(sb[c][1], 4), "species_hi": round(sb[c][2], 4),
                         "block_lo": round(bb[c][1], 4), "block_hi": round(bb[c][2], 4),
                         "n_queries": len(w), "n_species": w.species.nunique()})
    R = pd.DataFrame(rows); R.to_csv(outdir / "regime_metrics.tsv", sep="\t", index=False)
    return R


def run_mechanism(meta, dmat, lib, cl, rng, outdir, controls):
    rid = meta.record_id.to_numpy()
    qidx = np.where(np.isin(rid, list(cl)))[0]
    T = K.mechanism(meta, dmat, lib, qidx, rng=rng, controls=controls)
    T.to_csv(outdir / "mechanism_query_table.tsv", sep="\t", index=False)
    K.transitions(T).to_csv(outdir / "transitions.tsv", sep="\t", index=False)
    S = K.mechanism_summary(T, rng); S.to_csv(outdir / "mechanism_summary.tsv", sep="\t", index=False)
    return T, S


def analysis_main(meta, dist, valid, D, name="main", thr=None, dmat=None, controls=None):
    t0 = time.time()
    thr = thr or P8["strict_identity_min_overlap"]
    outdir = OUT / name; outdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(K.SEED + {"main": 1, "thr300": 2, "thr640": 3, "kmer": 4, "blast": 7}[name])
    lib = K.build_all(meta, dist, valid, thr)
    ca, cl = K.common_sets(meta, lib)
    pd.DataFrame([{"analysis": name, "identity_min_overlap": thr, "method": {"kmer": "kmer", "blast": "blast_bitscore"}.get(name, "pdist"),
                   "n_common_all": len(ca), "n_common_loco": len(cl)}]).to_csv(outdir / "design.tsv", sep="\t", index=False)
    dm = dist if dmat is None else dmat
    regime_metrics(meta, dm, lib, ca, cl, rng, outdir)
    T, S = run_mechanism(meta, dm, lib, cl, rng, outdir, controls if controls is not None else P8["controls_replicates"])
    print(name, f"done [{time.time()-t0:.0f}s]", flush=True)
    return S


def analysis_temporal(meta, dist, valid):
    t0 = time.time(); outdir = OUT / "temporal"; outdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(K.SEED + 5)
    thr = P8["strict_identity_min_overlap"]
    y = pd.to_numeric(meta.year, errors="coerce").to_numpy()
    yk = ~np.isnan(y)
    period = np.full(len(y), -1)
    for i, (a, b) in enumerate(P8["temporal_periods"]):
        period[yk & (y >= a) & (y <= b)] = i
    lib = K.build_all(meta, dist, valid, thr, universe=yk)
    ca, cl = K.common_sets(meta, lib)
    rid = meta.record_id.to_numpy(); sp = meta.species.to_numpy()
    U = meta.in_v3.to_numpy() & yk
    qidx = np.where(np.isin(rid, list(cl)))[0]
    # LOPO libraries
    L = {}
    for v, strict in (("strict", True), ("hapshared", False)):
        for p in sorted(set(period[qidx])):
            te = qidx[period[qidx] == p]
            base = U & (period != p)
            L.setdefault(f"lopo_{v}", []).append((f"period{p}", K.blocker(meta, dist, valid, te, base, strict, thr), te))
    lib2 = {"L": {**{k: v for k, v in lib["L"].items() if k in ("random_cv_strict", "lobo_strict", "lobo_hapshared")}, **L}}
    P = K.score_regimes(meta, dist, lib2, qidx)
    w = K.wide(P, list(lib2["L"].keys()))
    w = w.dropna()
    rows = []
    for r in lib2["L"]:
        for m in ("credit", "U"):
            rows.append({"regime": r, "metric": "expected" if m == "credit" else "unique",
                         "value": round(w.groupby("species")[f"{m}_{r}"].mean().mean(), 4), "n_queries": len(w),
                         "n_species": w.species.nunique()})
    for m in ("credit", "U"):
        for a, b in (("lopo_strict", "random_cv_strict"), ("lopo_strict", "lobo_strict"), ("lopo_hapshared", "lobo_hapshared")):
            w[f"d_{a}_minus_{b}_{m}"] = w[f"{m}_{a}"] - w[f"{m}_{b}"]
    cols = [c for c in w.columns if c.startswith("d_")]
    sb, _ = S7.species_boot(w, cols, rng, NB); bb = S7.block_boot(w, cols, rng, NB)
    for c in cols:
        rows.append({"regime": c, "metric": "contrast", "value": round(sb[c][0], 4), "species_lo": round(sb[c][1], 4),
                     "species_hi": round(sb[c][2], 4), "block_lo": round(bb[c][1], 4), "block_hi": round(bb[c][2], 4),
                     "n_queries": len(w), "n_species": w.species.nunique()})
    pd.DataFrame(rows).to_csv(outdir / "temporal_metrics.tsv", sep="\t", index=False)
    pd.Series(period[qidx]).value_counts().sort_index().rename("n_queries").to_csv(outdir / "period_query_counts.tsv", sep="\t")
    print("temporal done", f"[{time.time()-t0:.0f}s]", flush=True)


def kmeans_sphere(lat, lon, k, seed, iters=100):
    p = np.pi / 180
    X = np.c_[np.cos(lat * p) * np.cos(lon * p), np.cos(lat * p) * np.sin(lon * p), np.sin(lat * p)]
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmax(X @ cent.T, axis=1)
        new = np.array([X[lab == j].mean(0) if np.any(lab == j) else cent[j] for j in range(k)])
        new /= np.linalg.norm(new, axis=1, keepdims=True)
        if np.allclose(new, cent):
            break
        cent = new
    return np.argmax(X @ cent.T, axis=1)


def analysis_coord(meta, dist, valid):
    t0 = time.time(); outdir = OUT / "coord"; outdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(K.SEED + 6)
    thr = P8["strict_identity_min_overlap"]
    ck = meta.coord_valid.fillna(False).astype(bool).to_numpy() & meta.in_v3.to_numpy()
    lat = pd.to_numeric(meta.lat, errors="coerce").to_numpy(); lon = pd.to_numeric(meta.lon, errors="coerce").to_numpy()
    cl_lab = np.full(len(ck), -1)
    cl_lab[ck] = kmeans_sphere(lat[ck], lon[ck], P8["coord_kmeans_k"], K.SEED)
    lib = K.build_all(meta, dist, valid, thr, universe=ck)
    ca, cl = K.common_sets(meta, lib)
    rid = meta.record_id.to_numpy(); sp = meta.species.to_numpy()
    qidx = np.where(np.isin(rid, list(cl)))[0]
    U = meta.in_v3.to_numpy() & ck
    pd.DataFrame({"record_id": rid[ck], "cluster": cl_lab[ck], "block": meta.block.to_numpy()[ck]}).to_csv(
        outdir / "coord_clusters.tsv", sep="\t", index=False)
    Lc = {}
    for v, strict in (("strict", True), ("hapshared", False)):
        for c in sorted(set(cl_lab[qidx])):
            te = qidx[cl_lab[qidx] == c]
            Lc.setdefault(f"lobocoord_{v}", []).append((f"c{c}", K.blocker(meta, dist, valid, te, U & (cl_lab != c), strict, thr), te))
    lib2 = {"L": {**{k: v for k, v in lib["L"].items() if k.startswith(("lobo_", "loco_"))}, **Lc}}
    P = K.score_regimes(meta, dist, lib2, qidx)
    w = K.wide(P, list(lib2["L"].keys())).dropna()
    rows = []
    for r in lib2["L"]:
        for m in ("credit", "U"):
            rows.append({"regime": r, "metric": "expected" if m == "credit" else "unique",
                         "value": round(w.groupby("species")[f"{m}_{r}"].mean().mean(), 4), "n_queries": len(w),
                         "n_species": w.species.nunique()})
    for m in ("credit", "U"):
        for v in ("strict", "hapshared"):
            w[f"d_loco_minus_lobocoord_{v}_{m}"] = w[f"{m}_loco_{v}"] - w[f"{m}_lobocoord_{v}"]
            w[f"d_loco_minus_lobocountry_{v}_{m}"] = w[f"{m}_loco_{v}"] - w[f"{m}_lobo_{v}"]
        w[f"d_I_coord_{m}"] = w[f"d_loco_minus_lobocoord_hapshared_{m}"] - w[f"d_loco_minus_lobocoord_strict_{m}"]
        w[f"d_I_country_{m}"] = w[f"d_loco_minus_lobocountry_hapshared_{m}"] - w[f"d_loco_minus_lobocountry_strict_{m}"]
    cols = [c for c in w.columns if c.startswith("d_")]
    sb, _ = S7.species_boot(w, cols, rng, NB); bb = S7.block_boot(w, cols, rng, NB)
    for c in cols:
        rows.append({"regime": c, "metric": "contrast", "value": round(sb[c][0], 4), "species_lo": round(sb[c][1], 4),
                     "species_hi": round(sb[c][2], 4), "block_lo": round(bb[c][1], 4), "block_hi": round(bb[c][2], 4),
                     "n_queries": len(w), "n_species": w.species.nunique()})
    # 3-player Shapley for LOBO(coord) -> LOCO per variant (removed con / removed het / added)
    shp = []
    lo_c = {v: {lab: (ref, te) for lab, ref, te in lib2["L"][f"lobocoord_{v}"]} for v in ("strict", "hapshared")}
    co = {v: {lab: ref for lab, ref, _ in lib2["L"][f"loco_{v}"]} for v in ("strict", "hapshared")}
    qset = set(w.idx)
    cont = meta.continent.to_numpy()
    for v in ("strict", "hapshared"):
        for lab, (Rl, te) in lo_c[v].items():
            for Kc in sorted(set(cont[te])):
                qB = np.array([q for q in te if cont[q] == Kc and q in qset])
                if len(qB) == 0 or Kc not in co[v]:
                    continue
                Rc = co[v][Kc]; un = np.union1d(Rl, Rc)
                SM = S6.SpeciesMin(dist, qB, un, sp)
                lo = np.isin(SM.ref_idx, Rl); cc = np.isin(SM.ref_idx, Rc)
                rm = lo & ~cc; ad = cc & ~lo
                tsp = sp[qB]
                for s in sorted(set(tsp)):
                    qi = np.where(tsp == s)[0]; con = SM.rsp == s
                    parts = [rm & con, rm & ~con, ad]; vals = {}
                    for T in itertools.product((0, 1), repeat=3):
                        m = lo.copy()
                        if T[0]: m &= ~parts[0]
                        if T[1]: m &= ~parts[1]
                        if T[2]: m |= parts[2]
                        vals[T] = SM.evaluate(m)["credit"][qi]
                    for pi, pn in enumerate(("rm_con", "rm_het", "added")):
                        acc = np.zeros(len(qi))
                        for T in itertools.product((0, 1), repeat=3):
                            if T[pi]:
                                continue
                            Tp = list(T); Tp[pi] = 1; kk = sum(T)
                            acc += math.factorial(kk) * math.factorial(2 - kk) / 6 * (vals[tuple(Tp)] - vals[T])
                        for q, val in zip(qB[qi], acc):
                            shp.append({"variant": v, "idx": q, "species": s, "block": meta.block.to_numpy()[q],
                                        "player": pn, "phi": val})
    SH = pd.DataFrame(shp).pivot_table(index=["variant", "idx", "species", "block"], columns="player", values="phi").reset_index()
    for v in ("strict", "hapshared"):
        g = SH[SH.variant == v]
        cols2 = ["rm_con", "rm_het", "added"]
        sb, _ = S7.species_boot(g, cols2, rng, NB); bb = S7.block_boot(g, cols2, rng, NB)
        for c in cols2:
            rows.append({"regime": f"shapley_lobocoord_to_loco_{v}_{c}", "metric": "contrast", "value": round(sb[c][0], 4),
                         "species_lo": round(sb[c][1], 4), "species_hi": round(sb[c][2], 4),
                         "block_lo": round(bb[c][1], 4), "block_hi": round(bb[c][2], 4), "n_queries": len(g)})
    pd.DataFrame(rows).to_csv(outdir / "coord_metrics.tsv", sep="\t", index=False)
    print("coord done", f"[{time.time()-t0:.0f}s]", flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    D, meta, dist, valid = K.load()
    if which in ("main", "all"):
        analysis_main(meta, dist, valid, D, "main")
    if which in ("thr300", "all"):
        analysis_main(meta, dist, valid, D, "thr300", thr=300, controls=0)
    if which in ("thr640", "all"):
        analysis_main(meta, dist, valid, D, "thr640", thr=640, controls=0)
    if which in ("kmer", "all"):
        dk = K.kmer_distance(D, P8["second_method"]["k"])
        analysis_main(meta, dist, valid, D, "kmer", dmat=dk)
        del dk
    if which in ("blast", "all"):   # v8.1 addendum
        from holdout import blast as v8_blast
        q = pd.read_csv(OUT / "main" / "regime_query_scores.tsv", sep="\t", usecols=["idx"]).idx.unique()
        db = v8_blast.blast_matrix(D, q)
        analysis_main(meta, dist, valid, D, "blast", dmat=db)
        del db
    if which in ("temporal", "all"):
        analysis_temporal(meta, dist, valid)
    if which in ("coord", "all"):
        analysis_coord(meta, dist, valid)


if __name__ == "__main__":
    main()
