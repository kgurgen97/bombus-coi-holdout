"""v12: decision rules per specimen -- one species, candidate list, or reject.
No new models: v10 logistic and GB+isotonic (existing classes) are refit exactly as in v11 on the rebuilt
training rows. Thresholds are chosen on training data only; results are evaluated on held-out blocks (R, B)
and held-out continents (C). Outputs: results/v12/."""
import json, time
import numpy as np
import pandas as pd
import common
from holdout import config as C
from holdout import geography as V
from holdout import refsets as S6
from holdout import scenarios as K

import calibration_unseen_blocks as m66
m64, m65 = m66.m64, m66.m65
OUT = C.RESULTS / "v12"; OUT.mkdir(exist_ok=True)
REJECT_D1 = 0.03
DELTAS = (0.0, 0.01, 0.02)
ALPHAS = (0.01, 0.02)
T_GRID = [0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03]
G_GRID = [0, 0.0025, 0.005, 0.01, 0.02, 0.03]
NB = 10000
MARGIN = 0.01   # v12.1 non-inferiority margin for error among unique answers (absolute)


# ------------------------------------------------------------------ test-row extras (lists)
def list_extras(meta, dist, valid):
    """For every v10 test row: distance to the true species and list sizes for each delta."""
    thr = K.PARAMS["strict_identity_min_overlap"]; sp = meta.species.to_numpy()
    orig = m64.features

    def feat_extra(meta_, dist_, ref, te, ctx, variant, split):
        sm = S6.SpeciesMin(dist_, te, ref, sp)
        M = np.minimum.reduceat(sm.sub, sm.starts, axis=1); d1 = M.min(1)
        r = np.arange(len(te)); has = sm.tcol >= 0
        dtrue = np.full(len(te), np.inf); dtrue[has] = M[r[has], sm.tcol[has]]
        out = {"record_id": meta_.record_id.to_numpy()[te], "context": ctx, "variant": variant, "d_true": dtrue,
               "true_in_library": has.astype(int), "true_comparable": np.isfinite(dtrue).astype(int)}
        for dl in DELTAS:
            out[f"list_size_{dl}"] = (M <= d1[:, None] + dl + 1e-6).sum(1)
            out[f"list_has_true_{dl}"] = (dtrue <= d1 + dl + 1e-6).astype(int)
        return pd.DataFrame(out)[np.isfinite(d1)]
    m64.features = feat_extra
    try:
        X = m64.build_rows(meta, dist, valid)
    finally:
        m64.features = orig
    return X


# ------------------------------------------------------------------ rules
def decide(d, unique_mask, delta):
    """returns arrays: decision (U/L/R), unique_wrong, list_has_true, list_size"""
    rej = (d.d1 > REJECT_D1).to_numpy()
    uni = unique_mask & ~rej & (d.nS == 1).to_numpy()
    lst = ~uni & ~rej
    dec = np.where(uni, "one", np.where(lst, "list", "reject"))
    return {"decision": dec, "unique": uni.astype(float), "unique_wrong": (uni & (d.U.to_numpy() == 0)).astype(float),
            "list": lst.astype(float), "list_has_true": np.where(lst, d[f"list_has_true_{delta}"].to_numpy(), np.nan),
            "list_size": np.where(lst, d[f"list_size_{delta}"].to_numpy(), np.nan), "reject": rej.astype(float)}


def choose_threshold(cands, U, alpha):
    """cands: list of (label, bool mask over training rows); maximise coverage s.t. error among unique <= alpha.
    v12.1: candidates with zero coverage are not admissible; if no candidate reaches alpha -> None
    (the rule gives NO unique answers in this fold)."""
    best = None
    for lab, m in cands:
        n = m.sum()
        if n == 0:
            continue
        err = (m & (U == 0)).sum() / n
        cov = n / len(U)
        if err <= alpha and (best is None or cov > best[1]):
            best = (lab, cov, err)
    return best[0] if best else None


def p2_masks(d):
    base = ((d.nS == 1) & (d.d1 <= REJECT_D1)).to_numpy()
    return [((t, g), base & (d.d1 <= t).to_numpy() & (d.gap >= g).to_numpy()) for t in T_GRID for g in G_GRID]


def tau_grid(p):
    return np.unique(np.quantile(p, np.linspace(0, 1, 201)))


def model_inner_oob(tr, kind):
    """leave-one-block-out predictions of p_U inside the training rows (for threshold choice)"""
    p = np.full(len(tr), np.nan); y = tr.U.to_numpy()
    for b in tr.block.unique():
        v = (tr.block == b).to_numpy()
        if y[~v].min() == y[~v].max():
            continue
        m = fit_model(tr[~v].reset_index(drop=True), y[~v], kind)
        p[v] = m.predict(tr[v])
    return p


def fit_model(tr, y, kind):
    if kind == "logistic":
        return m65.LR(m65.X_lr).fit(tr, y)
    return m65.GBIso(False, K.SEED + 400).fit(tr, y)


# ------------------------------------------------------------------ outer loop
def run_fold(tr_all, te_all, ctx_list, fold_label, rows):
    for var in ("hapshared", "strict"):
        tr_v = tr_all[tr_all.variant == var].reset_index(drop=True)
        te_v = te_all[te_all.variant == var]
        if te_v.empty:
            continue
        models = {}
        for kind in ("logistic", "gb_isotonic"):
            models[kind] = (fit_model(tr_v, tr_v.U.to_numpy(), kind), model_inner_oob(tr_v, kind))
        for ctx in ctx_list:
            te = te_v[te_v.context == ctx].reset_index(drop=True)
            tr = tr_v[tr_v.context == ctx]
            if te.empty or tr.empty:
                continue
            trU = tr.U.to_numpy(); tr_idx = tr.index.to_numpy()
            out = te[["record_id", "species", "block", "context", "variant", "U", "y", "d1", "nS", "d_true"]].copy()
            out["fold"] = fold_label
            out["true_in_library"] = te.true_in_library; out["true_comparable"] = te.true_comparable
            for dl in DELTAS:
                out[f"list_size_{dl}"] = te[f"list_size_{dl}"]; out[f"list_has_true_{dl}"] = te[f"list_has_true_{dl}"]
            base = ((te.nS == 1) & (te.d1 <= REJECT_D1)).to_numpy()
            out["mask_P0"] = base
            out["mask_P1"] = base & (te.d1 <= 0.01).to_numpy() & (te.d2.fillna(1) > 0.01).to_numpy()
            for a in ALPHAS:
                tg = choose_threshold(p2_masks(tr), trU, a)
                if tg is None:
                    out[f"mask_P2_a{a}"] = np.zeros(len(te), bool); out[f"thr_P2_a{a}"] = "none"
                else:
                    t, g = tg
                    out[f"mask_P2_a{a}"] = base & (te.d1 <= t).to_numpy() & (te.gap >= g).to_numpy()
                    out[f"thr_P2_a{a}"] = json.dumps([t, g])
                for code, kind in (("P3", "logistic"), ("P4", "gb_isotonic")):
                    model, oob = models[kind]
                    po = oob[tr_idx]; ok = ~np.isnan(po)
                    trb = ((tr.nS == 1) & (tr.d1 <= REJECT_D1)).to_numpy()
                    cands = [(tau, trb & ok & (po >= tau)) for tau in tau_grid(po[ok])]
                    tau = choose_threshold(cands, trU, a)
                    if tau is None:
                        out[f"mask_{code}_a{a}"] = np.zeros(len(te), bool); out[f"thr_{code}_a{a}"] = "none"
                    else:
                        out[f"mask_{code}_a{a}"] = base & (model.predict(te) >= tau); out[f"thr_{code}_a{a}"] = str(tau)
            rows.append(out)


def evaluate(P, rng):
    rules = ["P0", "P1"] + [f"{r}_a{a}" for r in ("P2", "P3", "P4") for a in ALPHAS]
    res, per_query = [], {}
    for (var, ctx), d in P.groupby(["variant", "context"]):
        d = d.reset_index(drop=True)
        for rule in rules:
            for dl in DELTAS:
                dec = decide(d, d[f"mask_{rule}"].to_numpy(bool), dl)
                q = pd.DataFrame({"species": d.species, **{k: v for k, v in dec.items() if k != "decision"}})
                sm = q.groupby("species").mean(numeric_only=True)
                r = {"variant": var, "context": ctx, "rule": rule, "delta": dl, "n_queries": len(d), "n_species": d.species.nunique(),
                     "unique_coverage_query": q.unique.mean(), "unique_coverage_species": sm.unique.mean(),
                     "wrong_among_unique_query": q.unique_wrong.sum() / q.unique.sum() if q.unique.sum() else np.nan,
                     "wrong_among_unique_ratio_of_species_means": sm.unique_wrong.mean() / sm.unique.mean() if sm.unique.mean() else np.nan,
                     "wrong_among_unique_mean_of_species_rates": (sm.unique_wrong / sm.unique)[sm.unique > 0].mean() if (sm.unique > 0).any() else np.nan,
                     "n_species_with_unique": int((sm.unique > 0).sum()),
                     "folds_without_unique_answers": int((d.groupby("fold")[f"mask_{rule}"].sum() == 0).sum()) if rule not in ("P0", "P1") else 0,
                     "n_folds": d.fold.nunique(),
                     "wrong_unique_per100_query": 100 * q.unique_wrong.mean(), "wrong_unique_per100_species": 100 * sm.unique_wrong.mean(),
                     "list_share_query": q.list.mean(), "list_inclusion_query": np.nanmean(q.list_has_true) if q.list.sum() else np.nan,
                     "list_inclusion_species": q.dropna(subset=["list_has_true"]).groupby("species").list_has_true.mean().mean() if q.list.sum() else np.nan,
                     "list_size_mean": np.nanmean(q.list_size) if q.list.sum() else np.nan,
                     "list_size_median": np.nanmedian(q.list_size) if q.list.sum() else np.nan,
                     "reject_share_query": q.reject.mean(),
                     "reject_true_species_absent_share": (d.true_in_library[dec["reject"] == 1] == 0).mean() if dec["reject"].sum() else np.nan,
                     "reject_true_not_comparable_share": (d.true_comparable[dec["reject"] == 1] == 0).mean() if dec["reject"].sum() else np.nan}
                res.append(r)
                if dl == 0.01:
                    per_query[(var, ctx, rule)] = q[["species", "unique", "unique_wrong"]]
    R = pd.DataFrame(res)
    # paired comparisons (species bootstrap)
    comp = []
    for var in ("hapshared", "strict"):
        for ctx in ("R", "B", "C"):
            for a in ALPHAS:
                for x, y in ((f"P3_a{a}", f"P2_a{a}"), (f"P4_a{a}", f"P2_a{a}"), (f"P3_a{a}", "P1"), (f"P4_a{a}", "P1"),
                             (f"P2_a{a}", "P1")):
                    if (var, ctx, x) not in per_query:
                        continue
                    qx, qy = per_query[(var, ctx, x)], per_query[(var, ctx, y)]
                    Sx = qx.groupby("species")[["unique", "unique_wrong"]].mean(); Sy = qy.groupby("species")[["unique", "unique_wrong"]].mean().loc[Sx.index]
                    A = np.column_stack([Sx.unique, Sx.unique_wrong, Sy.unique, Sy.unique_wrong]); n = len(A)
                    idx = rng.integers(0, n, size=(NB, n)); B = A[idx].mean(1)
                    dcov = B[:, 0] - B[:, 2]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        derr = B[:, 1] / B[:, 0] - B[:, 3] / B[:, 2]      # ratio of species means; NaN if no unique answers
                    dwr = B[:, 1] - B[:, 3]
                    e = A.mean(0); fin = np.isfinite(derr)
                    comp.append({"variant": var, "context": ctx, "alpha": a, "rule_x": x, "rule_y": y, "n_species": n,
                                 "d_coverage": e[0] - e[2], "d_coverage_lo": np.quantile(dcov, .025), "d_coverage_hi": np.quantile(dcov, .975),
                                 "d_wrong_among_unique_ratio_of_species_means": (e[1] / e[0] - e[3] / e[2]) if e[0] > 0 and e[2] > 0 else np.nan,
                                 "d_wrong_lo": np.quantile(derr[fin], .025) if fin.mean() > 0.99 else np.nan,
                                 "d_wrong_hi": np.quantile(derr[fin], .975) if fin.mean() > 0.99 else np.nan,
                                 "share_boot_defined": fin.mean(),
                                 "d_wrong_per100": 100 * (e[1] - e[3]), "d_wrong_per100_lo": 100 * np.quantile(dwr, .025),
                                 "d_wrong_per100_hi": 100 * np.quantile(dwr, .975)})
    Cm = pd.DataFrame(comp)
    # v12.1 criterion: coverage gain (lower CI > 0) AND non-inferior error (upper CI of difference <= +1 pp)
    Cm["coverage_gain"] = Cm.d_coverage_lo > 0
    Cm["error_noninferior"] = Cm.d_wrong_hi <= MARGIN
    Cm["verdict"] = np.where(Cm.d_wrong_hi.isna(), "undefined (no unique answers in some resamples)",
                             np.where(Cm.coverage_gain & Cm.error_noninferior, "advantage",
                                      np.where(~Cm.error_noninferior, "difference uncertain (error bound exceeds margin)",
                                               "no coverage gain")))
    return R, Cm


def main():
    t0 = time.time()
    D, meta, dist, valid = K.load()
    test = pd.read_csv(C.RESULTS / "v10_1" / "query_features_plus.tsv.gz", sep="\t")
    xf = OUT / "test_list_extras.tsv.gz"
    if xf.exists():
        X = pd.read_csv(xf, sep="\t")
    else:
        X = list_extras(meta, dist, valid); X.to_csv(xf, sep="\t", index=False)
    test = test.merge(X, on=["record_id", "context", "variant"], how="left", validate="one_to_one")
    assert test.d_true.notna().all()
    test["continent"] = test.block.map(V.continent_of)
    print("test rows", len(test), f"[{time.time()-t0:.0f}s]", flush=True)
    key66 = (C.RESULTS / "v11" / "calibration" / "cache_key.txt").read_text().strip()
    c66 = C.RESULTS / "v11" / "calibration" / f"cache_{key66}"
    c68 = [p for p in (C.RESULTS / "v11" / "calibration_continent").glob("cache_*") if p.is_dir()]
    assert len(c68) == 1, c68
    rows = []
    for b in sorted(test.block.unique()):
        tr = pd.read_csv(c66 / f"train_rows_{b}.tsv.gz", sep="\t")
        run_fold(tr, test[test.block == b], ("R", "B"), f"block:{b}", rows)
        print("block", b, f"[{time.time()-t0:.0f}s]", flush=True)
    for k in sorted(test[test.context == "C"].continent.unique()):
        tr = pd.read_csv(c68[0] / f"train_rows_{k}.tsv.gz", sep="\t")
        run_fold(tr, test[(test.continent == k)], ("C",), f"continent:{k}", rows)
        print("continent", k, f"[{time.time()-t0:.0f}s]", flush=True)
    P = pd.concat(rows, ignore_index=True)
    P.to_csv(OUT / "decisions_per_query.tsv.gz", sep="\t", index=False)
    R, Cmp = evaluate(P, np.random.default_rng(K.SEED + 1300))
    R.to_csv(OUT / "rule_performance.tsv", sep="\t", index=False); Cmp.to_csv(OUT / "paired_comparisons.tsv", sep="\t", index=False)
    pd.set_option("display.width", 260)
    k = ["variant", "context", "rule", "unique_coverage_query", "unique_coverage_species", "wrong_among_unique_query",
         "wrong_among_unique_ratio_of_species_means", "wrong_among_unique_mean_of_species_rates", "list_inclusion_query", "list_size_mean", "reject_share_query"]
    print(R[(R.delta == 0.01)][k].round(4).to_string(index=False))
    print(Cmp.round(4).to_string(index=False))
    st = []
    for (var, x, y), g in Cmp[Cmp.context.isin(["B", "C"])].assign(
            rx=Cmp.rule_x.str.replace(r"_a0\.0[12]", "", regex=True), ry=Cmp.rule_y.str.replace(r"_a0\.0[12]", "", regex=True)
    ).groupby(["variant", "rx", "ry"]):
        st.append({"variant": var, "rule_x": x, "rule_y": y, "n_checks": len(g),
                   "stable_advantage": bool(len(g) == 4 and (g.verdict == "advantage").all()),
                   "verdicts": "; ".join(f"{r.context}/a{r.alpha}: {r.verdict}" for r in g.itertuples())})
    pd.DataFrame(st).to_csv(OUT / "stable_advantage.tsv", sep="\t", index=False)
    print(pd.DataFrame(st).to_string(index=False))
    print(f"done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
