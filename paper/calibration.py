"""v10: hidden errors and calibration of COI identification reliability in Bombus.
Outputs: results/v10/, results/v10/."""
import json, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
from holdout import config as C
from holdout import geography as V
from holdout import refsets as S6
from holdout import stats as S7
from holdout import scenarios as K

P8 = K.PARAMS
OUT = C.RESULTS / "v10"; OUT.mkdir(exist_ok=True)
F = C.RESULTS / "v10"; F.mkdir(exist_ok=True)
CTX = {"R": "region represented (random CV)", "B": "block removed (LOBO)", "C": "continent removed (LOCO)"}
FEATS = ["d1", "d1sq", "lgap", "tie", "cov_block", "cov_continent", "lnref"]


def features(meta, dist, ref, te, ctx, variant, split):
    sp = meta.species.to_numpy(); blk = meta.block.to_numpy(); cont = meta.continent.to_numpy()
    sm = S6.SpeciesMin(dist, te, ref, sp)
    ev = sm.evaluate()
    M = np.minimum.reduceat(sm.sub, sm.starts, axis=1)
    Ms = np.sort(M, axis=1)
    d1 = Ms[:, 0]; d2 = Ms[:, 1] if M.shape[1] > 1 else np.full(len(d1), np.inf)
    gap = np.where(ev["nS"] >= 2, 0.0, np.minimum(np.where(np.isfinite(d2), d2, d1 + 0.1) - d1, 0.1))
    # coverage / reference counts of the best species (max over tie species)
    rblk, rcont = blk[sm.ref_idx], cont[sm.ref_idx]
    rows = []
    for i, q in enumerate(te):
        ts = np.where(ev["tie"][i])[0]
        cov, nref = 0, 0
        for s in ts:
            a, b = sm.starts[s], (sm.starts[s + 1] if s + 1 < len(sm.starts) else len(sm.ref_idx))
            nref = max(nref, b - a)
            if (rblk[a:b] == blk[q]).any():
                cov = max(cov, 2)
            elif (rcont[a:b] == cont[q]).any():
                cov = max(cov, 1)
        top = ";".join(sm.species[ts])
        rows.append((cov, nref, top))
    cov = np.array([r[0] for r in rows]); nref = np.array([r[1] for r in rows])
    fin = np.isfinite(d1)
    df = pd.DataFrame({"idx": te, "record_id": meta.record_id.to_numpy()[te], "species": sp[te], "block": blk[te],
                       "context": ctx, "variant": variant, "split": split, "d1": np.where(fin, d1, np.nan),
                       "d2": np.where(np.isfinite(d2), d2, np.nan), "gap": gap, "nS": ev["nS"],
                       "tie": (ev["nS"] >= 2).astype(int), "cov": np.array(["none", "continent", "block"])[cov],
                       "nref": nref, "top": [r[2] for r in rows],
                       "y": (ev["credit"] > 0).astype(int), "U": (ev["state"] == "U").astype(int)})
    return df[fin]


def build_rows(meta, dist, valid):
    thr = P8["strict_identity_min_overlap"]
    lib = K.build_all(meta, dist, valid, thr)
    ca, cl = K.common_sets(meta, lib)
    rid = meta.record_id.to_numpy()
    # context R, hapshared: identical folds to v8 random_cv_strict (same rng, same first call), hapshared blocking
    U = meta.in_v3.to_numpy().copy(); sp = meta.species.to_numpy(); Q = lib["Q"]
    rng = np.random.default_rng(K.SEED + 80)
    grp = V.union_groups(meta, Q)
    Rh = [(f"fold{f}", K.blocker(meta, dist, valid, te, U.copy(), False, thr), te)
          for f, te in enumerate(V.grouped_folds(Q, sp[Q], grp, P8["random_cv_folds"], rng))]
    # sanity: same folds as the frozen strict random CV
    assert all(np.array_equal(a[2], b[2]) for a, b in zip(Rh, lib["L"]["random_cv_strict"]))
    plan = [("R", "hapshared", Rh, ca), ("R", "strict", lib["L"]["random_cv_strict"], ca),
            ("B", "hapshared", lib["L"]["lobo_hapshared"], ca), ("B", "strict", lib["L"]["lobo_strict"], ca),
            ("C", "hapshared", lib["L"]["loco_hapshared"], cl), ("C", "strict", lib["L"]["loco_strict"], cl)]
    out = []
    for ctx, v, splits, keep in plan:
        for lab, ref, te in splits:
            te = np.array([q for q in te if rid[q] in keep])
            if len(te):
                out.append(features(meta, dist, ref, te, ctx, v, lab))
        print(ctx, v, "done", flush=True)
    return pd.concat(out, ignore_index=True)


def design(df):
    X = pd.DataFrame({"d1": df.d1.clip(upper=0.1), "lgap": np.log(df.gap + 1e-3), "tie": df.tie,
                      "cov_block": (df["cov"] == "block").astype(int), "cov_continent": (df["cov"] == "continent").astype(int),
                      "lnref": np.log1p(df.nref)})
    X["d1sq"] = X.d1 ** 2
    return X[FEATS].to_numpy(float)


def ece(p, y, nb=10):
    o = np.argsort(p); bins = np.array_split(o, nb)
    return float(sum(len(b) * abs(p[b].mean() - y[b].mean()) for b in bins) / len(p))


def nested(df, target="y"):
    """leave-one-query-block-out across all contexts; returns out-of-block predictions"""
    X, y = design(df), df[target].to_numpy()
    p_lr, p_gb = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for b in sorted(df.block.unique()):
        te = (df.block == b).to_numpy(); tr = ~te
        if y[tr].min() == y[tr].max():
            continue
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr], y[tr])
        p_lr[te] = lr.predict_proba(X[te])[:, 1]
        # GB + isotonic, recalibrated on an inner split of the training blocks (half of the blocks)
        tb = np.array(sorted(df.block[tr].unique())); r = np.random.default_rng(K.SEED + 100 + len(b))
        cal_b = set(r.choice(tb, len(tb) // 2, replace=False))
        cal = tr & df.block.isin(cal_b).to_numpy(); fit = tr & ~cal
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=K.SEED).fit(X[fit], y[fit])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(gb.predict_proba(X[cal])[:, 1], y[cal])
        p_gb[te] = iso.predict(gb.predict_proba(X[te])[:, 1])
    return p_lr, p_gb


def metrics(df, p, y, label):
    rows = []
    for ctx in ["all", "R", "B", "C"]:
        m = np.ones(len(df), bool) if ctx == "all" else (df.context == ctx).to_numpy()
        m = m & ~np.isnan(p)
        pp, yy = p[m], y[m]
        w = 1.0 / df[m].groupby("species").species.transform("size").to_numpy()
        rows.append({"model": label, "context": ctx, "n": int(m.sum()), "brier": float(np.mean((pp - yy) ** 2)),
                     "brier_species": float(np.sum(w * (pp - yy) ** 2) / w.sum()), "ece": ece(pp, yy),
                     "auc": roc_auc_score(yy, pp) if 0 < yy.mean() < 1 else np.nan,
                     "mean_p": float(pp.mean()), "mean_y": float(yy.mean())})
    return rows


def hidden(df, rng):
    rows = []
    d = df.copy()
    d["conf"] = ((d.nS == 1) & (d.d1 <= 0.01) & (d.d2.fillna(1) > 0.01)).astype(float)
    d["vconf"] = (d.conf.astype(bool) & (d.gap >= 0.02)).astype(float)
    d["err"] = 1.0 - d.U
    for var in ("hapshared", "strict"):
        for key, grp in [("context", "R"), ("context", "B"), ("context", "C"),
                         ("cov", "block"), ("cov", "continent"), ("cov", "none")]:
            g = d[(d.variant == var) & (d[key] == grp)]
            for rule in ("conf", "vconf"):
                c = g[g[rule] == 1].copy(); c["wrong"] = 1.0 - c.U
                e = g[g.err == 1].copy(); e["hidden"] = e[rule]
                est = {"share_confident": S7.species_boot(g.assign(v=g[rule]), ["v"], rng)[0]["v"],
                       "error_rate_among_confident": S7.species_boot(c, ["wrong"], rng)[0]["wrong"] if len(c) else (np.nan,) * 3,
                       "hidden_share_of_errors": S7.species_boot(e, ["hidden"], rng)[0]["hidden"] if len(e) else (np.nan,) * 3}
                for q, (m, lo, hi) in est.items():
                    rows.append({"variant": var, "by": key, "group": grp, "rule": rule, "quantity": q,
                                 "species_mean": m, "lo": lo, "hi": hi,
                                 "query_level": {"share_confident": g[rule].mean(),
                                                 "error_rate_among_confident": c.wrong.mean() if len(c) else np.nan,
                                                 "hidden_share_of_errors": e.hidden.mean() if len(e) else np.nan}[q],
                                 "n_queries": len(g), "n_confident": len(c), "n_errors": len(e), "n_species": g.species.nunique()})
    return pd.DataFrame(rows)


def reliability_fig(df, preds):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, ctx in zip(axes, ("R", "B", "C")):
        m = (df.context == ctx).to_numpy()
        for lab, p, col in preds:
            mm = m & ~np.isnan(p); pp, yy = p[mm], df.y.to_numpy()[mm]
            o = np.argsort(pp); bins = np.array_split(o, 10)
            ax.plot([pp[b].mean() for b in bins], [yy[b].mean() for b in bins], "o-", color=col, ms=4, label=lab)
        ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--")
        ax.set_title(CTX[ctx], loc="left", fontsize=9); ax.set_xlabel("predicted probability")
    axes[0].set_ylabel("observed share correct"); axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Bombus, hapshared library: reliability on held-out blocks (nested leave-one-block-out)", x=0.01, ha="left", fontsize=9.5)
    for e in ("png", "pdf"):
        fig.savefig(F / f"reliability_v10.{e}", bbox_inches="tight")
    plt.close(fig)


def tail_table(d):
    """lowest decile of each score within each context: predicted vs observed (hapshared)"""
    rows = []
    for ctx in ("R", "B", "C"):
        x = d[d.context == ctx]
        for col in ("similarity", "p_logistic", "p_gb"):
            v = (1 - x.d1).to_numpy() if col == "similarity" else x[col].to_numpy()
            b = np.argsort(v)[: len(v) // 10]
            rows.append({"context": ctx, "score": col, "n": len(b), "n_species": x.species.to_numpy()[b].size and
                         len(set(x.species.to_numpy()[b])), "predicted": v[b].mean(), "observed": x.y.to_numpy()[b].mean()})
    lo = d[(d.context == "C") & (d.p_gb < 0.5)]
    pairs = lo[lo.y == 0].groupby(["species", "top"]).size().sort_values(ascending=False).rename("n").reset_index()
    return pd.DataFrame(rows), pairs


def main():
    t0 = time.time()
    D, meta, dist, valid = K.load()
    cache = OUT / "query_features.tsv.gz"
    if cache.exists():
        df = pd.read_csv(cache, sep="\t")
    else:
        df = build_rows(meta, dist, valid); df.to_csv(cache, sep="\t", index=False)
    print("rows", len(df), f"[{time.time()-t0:.0f}s]")
    rng = np.random.default_rng(K.SEED + 10)
    H = hidden(df, rng); H.to_csv(OUT / "hidden_errors.tsv", sep="\t", index=False)
    allm, preds_h = [], None
    for var in ("hapshared", "strict"):
        d = df[df.variant == var].reset_index(drop=True)
        p_lr, p_gb = nested(d)
        y = d.y.to_numpy()
        base_sim = 1.0 - d.d1.to_numpy()
        conf = ((d.nS == 1) & (d.d1 <= 0.01) & (d.d2.fillna(1) > 0.01)).to_numpy().astype(float)
        const = np.full(len(d), np.nan)
        for b in d.block.unique():
            te = (d.block == b).to_numpy(); const[te] = y[~te].mean()
        for lab, p in (("logistic", p_lr), ("gb_isotonic", p_gb), ("similarity_1_minus_d1", base_sim),
                       ("bold_confident_rule", conf), ("constant", const)):
            for r in metrics(d, p, y, lab):
                r["variant"] = var; allm.append(r)
        d.assign(p_logistic=p_lr, p_gb=p_gb).to_csv(OUT / f"oob_predictions_{var}.tsv.gz", sep="\t", index=False)
        if var == "hapshared":
            T, pr = tail_table(d.assign(p_logistic=p_lr, p_gb=p_gb))
            T.to_csv(OUT / "lowest_decile_hapshared.tsv", sep="\t", index=False)
            pr.to_csv(OUT / "loco_low_confidence_errors_by_pair.tsv", sep="\t", index=False)
            preds_h = (d, [("logistic (ours)", p_lr, "#2a78d6"), ("GB + isotonic", p_gb, "#1baf7a"),
                           ("similarity = 1 - d1", base_sim, "#eb6834")])
    Mt = pd.DataFrame(allm); Mt.to_csv(OUT / "calibration_metrics.tsv", sep="\t", index=False)
    reliability_fig(*preds_h)
    # final model on all hapshared rows (for the tool); coefficients reported
    d = df[df.variant == "hapshared"]
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(design(d), d.y)
    sc, m = lr.named_steps["standardscaler"], lr.named_steps["logisticregression"]
    json.dump({"features": FEATS, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
               "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
               "note": "Bombus only; hapshared library"},
              open(OUT / "final_logistic_model.json", "w"), indent=2)
    crit = Mt[(Mt.variant == "hapshared") & (Mt.model == "logistic") & (Mt.context.isin(["R", "B", "C"]))]
    base = Mt[(Mt.variant == "hapshared") & (Mt.model == "similarity_1_minus_d1") & (Mt.context.isin(["R", "B", "C"]))]
    ok = bool((crit.ece <= 0.05).all() and (crit.brier.to_numpy() < base.brier.to_numpy()).all())
    print(Mt.round(4).to_string(index=False))
    print("usefulness criterion met:", ok)
    print(H[(H.variant == "hapshared")].round(3).to_string(index=False))
    print(f"done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
