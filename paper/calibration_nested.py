"""v10.1: calibration with model selection nested inside each outer fold.
Reuses v10 code (calibration.py) unchanged; adds feature n_sp_2pct. Outputs: results/v10_1/, results/v10_1/."""
import json, time
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
from holdout import config as C
from holdout import refsets as S6
from holdout import scenarios as K

import calibration as m64
OUT = C.RESULTS / "v10_1"; OUT.mkdir(exist_ok=True)
F = C.RESULTS / "v10_1"; F.mkdir(exist_ok=True)
_orig = m64.features


def features_plus(meta, dist, ref, te, ctx, variant, split):
    df = _orig(meta, dist, ref, te, ctx, variant, split)
    sm = S6.SpeciesMin(dist, te, ref, meta.species.to_numpy())
    M = np.minimum.reduceat(sm.sub, sm.starts, axis=1)
    n2 = pd.Series((M <= 0.02).sum(1), index=te)
    df["n_sp_2pct"] = n2.loc[df.idx].to_numpy()
    return df


def base_frame(d):
    X = pd.DataFrame({"d1": d.d1.clip(upper=0.1), "lgap": np.log(d.gap + 1e-3), "tie": d.tie,
                      "cov_block": (d["cov"] == "block").astype(int), "cov_continent": (d["cov"] == "continent").astype(int),
                      "lnref": np.log1p(d.nref), "n2": np.log1p(d.n_sp_2pct)})
    X["d1sq"] = X.d1 ** 2
    return X


def X_lr(d):
    return base_frame(d)[["d1", "d1sq", "lgap", "tie", "cov_block", "cov_continent", "lnref", "n2"]].to_numpy(float)


def X_lri(d):
    X = base_frame(d)
    X["d1_x_block"] = X.d1 * X.cov_block; X["d1_x_cont"] = X.d1 * X.cov_continent
    X["d1_x_tie"] = X.d1 * X.tie; X["le1pct"] = (d.d1 <= 0.01).astype(int).to_numpy()
    return X.to_numpy(float)


GB_COLS = ["d1", "lgap", "tie", "cov_block", "cov_continent", "lnref", "n2"]
MONO = [-1, 1, -1, 0, 0, 0, 0]   # P(correct) non-increasing in d1, non-decreasing in gap, non-increasing with tie


def X_gb(d):
    return base_frame(d)[GB_COLS].to_numpy(float)


class LR:
    def __init__(self, xf): self.xf = xf
    def fit(self, d, y):
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(self.xf(d), y); return self
    def predict(self, d): return self.m.predict_proba(self.xf(d))[:, 1]


class GBIso:
    """GB fitted on half of the training blocks, isotonic recalibration on the other half (as in v10)."""
    def __init__(self, mono, seed): self.mono, self.seed = mono, seed
    def fit(self, d, y):
        tb = np.array(sorted(d.block.unique())); r = np.random.default_rng(self.seed)
        cal = d.block.isin(set(r.choice(tb, len(tb) // 2, replace=False))).to_numpy()
        kw = {"monotonic_cst": MONO} if self.mono else {}
        self.gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=K.SEED, **kw)
        self.gb.fit(X_gb(d[~cal]), y[~cal])
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(self.gb.predict_proba(X_gb(d[cal]))[:, 1], y[cal])
        return self
    def predict(self, d): return self.iso.predict(self.gb.predict_proba(X_gb(d))[:, 1])


def candidates(seed):
    return {"logistic_v10": LR(X_lr), "logistic_interactions": LR(X_lri),
            "gb_isotonic": GBIso(False, seed), "gb_monotone_isotonic": GBIso(True, seed)}


def block_folds(blocks, k, rng):
    b = np.array(sorted(blocks)); rng.shuffle(b)
    return [set(x) for x in np.array_split(b, k)]


def nested_select(d):
    y = d.y.to_numpy(); p = np.full(len(d), np.nan); chosen = {}; inner_rows = []
    for ob in sorted(d.block.unique()):
        te = (d.block == ob).to_numpy(); tr_d = d[~te].reset_index(drop=True); tr_y = y[~te]
        rng = np.random.default_rng(K.SEED + 200)
        folds = block_folds(tr_d.block.unique(), 5, rng)
        score = {}
        for name in candidates(0):
            se = []
            for fi, fb in enumerate(folds):
                v = tr_d.block.isin(fb).to_numpy()
                if tr_y[~v].min() == tr_y[~v].max():
                    continue
                m = candidates(K.SEED + 300 + fi)[name].fit(tr_d[~v].reset_index(drop=True), tr_y[~v])
                se.append(np.mean((m.predict(tr_d[v]) - tr_y[v]) ** 2) * v.sum())
            score[name] = sum(se) / len(tr_d)
            inner_rows.append({"outer_block": ob, "candidate": name, "inner_brier": score[name]})
        best = min(score, key=score.get); chosen[ob] = best
        m = candidates(K.SEED + 400)[best].fit(tr_d, tr_y)
        p[te] = m.predict(d[te])
    return p, chosen, pd.DataFrame(inner_rows)


def main():
    t0 = time.time()
    cache = OUT / "query_features_plus.tsv.gz"
    if cache.exists():
        df = pd.read_csv(cache, sep="\t")
    else:
        D, meta, dist, valid = K.load()
        m64.features = features_plus
        df = m64.build_rows(meta, dist, valid); df.to_csv(cache, sep="\t", index=False)
    # consistency with v10 rows
    v10 = pd.read_csv(C.RESULTS / "v10" / "query_features.tsv.gz", sep="\t")
    assert len(v10) == len(df) and (v10.y.to_numpy() == df.y.to_numpy()).all()
    print("rows", len(df), f"[{time.time()-t0:.0f}s]", flush=True)
    rows, sel = [], []
    for var in ("hapshared", "strict"):
        d = df[df.variant == var].reset_index(drop=True)
        p, chosen, inner = nested_select(d)
        inner.assign(variant=var).to_csv(OUT / f"inner_scores_{var}.tsv", sep="\t", index=False)
        for ob, c in chosen.items():
            sel.append({"variant": var, "outer_block": ob, "chosen": c})
        y = d.y.to_numpy()
        for lab, pp in (("nested_selected", p), ("similarity_1_minus_d1", 1 - d.d1.to_numpy())):
            for r in m64.metrics(d, pp, y, lab):
                r["variant"] = var; rows.append(r)
        d.assign(p_nested=p).to_csv(OUT / f"oob_predictions_{var}.tsv.gz", sep="\t", index=False)
        if var == "hapshared":
            T, pr = m64.tail_table(d.assign(p_logistic=p, p_gb=p))
            T[T.score != "p_gb"].replace({"p_logistic": "p_nested"}).to_csv(OUT / "lowest_decile_hapshared.tsv", sep="\t", index=False)
            dh = d.assign(pn=p)
        print(var, "selection:", dict(Counter(chosen.values())), f"[{time.time()-t0:.0f}s]", flush=True)
    Mt = pd.DataFrame(rows); Mt.to_csv(OUT / "calibration_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(sel).to_csv(OUT / "selected_models.tsv", sep="\t", index=False)
    h = Mt[Mt.variant == "hapshared"].set_index(["model", "context"])
    ok = all(h.loc[("nested_selected", c), "ece"] <= 0.05 and
             h.loc[("nested_selected", c), "brier"] < h.loc[("similarity_1_minus_d1", c), "brier"] for c in "RBC")
    print(Mt.round(4).to_string(index=False)); print("usefulness criterion met (hapshared):", ok)
    # reliability figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, ctx in zip(axes, "RBC"):
        m = (dh.context == ctx).to_numpy()
        for lab, pp, col in (("nested-selected model", dh.pn.to_numpy(), "#2a78d6"),
                             ("similarity = 1 - d1", 1 - dh.d1.to_numpy(), "#eb6834")):
            q, yy = pp[m], dh.y.to_numpy()[m]; bins = np.array_split(np.argsort(q), 10)
            ax.plot([q[b].mean() for b in bins], [yy[b].mean() for b in bins], "o-", color=col, ms=4, label=lab)
        ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--"); ax.set_title(m64.CTX[ctx], loc="left", fontsize=9)
        ax.set_xlabel("predicted probability")
    axes[0].set_ylabel("observed share correct"); axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Bombus, hapshared: reliability on held-out blocks, model chosen inside each outer fold (v10.1)",
                 x=0.01, ha="left", fontsize=9.5)
    for e in ("png", "pdf"):
        fig.savefig(F / f"reliability_v10_1.{e}", bbox_inches="tight")
    print(f"done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
