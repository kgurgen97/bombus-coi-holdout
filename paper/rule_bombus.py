"""v14 A: paired P0 vs P1 on held-out regions from v12 decisions.
Also the shared helpers used by rule_features.py. Outputs: results/v14/A/."""
import numpy as np
import pandas as pd
import common
from holdout import config as C

OUT = C.RESULTS / "v14" / "A"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260924


def rule_masks(d):
    """P0/P1 single-species masks from d1, d2, nS (reject d1 > 0.03)."""
    rej = d.d1 > 0.03
    p0 = (d.nS == 1) & ~rej
    p1 = p0 & (d.d1 <= 0.01) & (d.d2.fillna(np.inf) > 0.01)
    return p0.to_numpy(), p1.to_numpy()


def paired(sp, U, m0, m1, rng, nb=10000):
    """species bootstrap of P1 - P0: single share, wrong per 100, wrong among single (ratio of species means)."""
    S = pd.DataFrame({"species": sp, "o0": m0.astype(float), "o1": m1.astype(float),
                      "w0": (m0 & (U == 0)).astype(float), "w1": (m1 & (U == 0)).astype(float)})
    A = S.groupby("species")[["o0", "w0", "o1", "w1"]].mean().to_numpy(); n = len(A)
    B = A[rng.integers(0, n, (nb, n))].mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r0, r1 = B[:, 1] / B[:, 0], B[:, 3] / B[:, 2]
    e = A.mean(0); q = lambda x: (np.nanquantile(x, .025), np.nanquantile(x, .975))
    out = {"n_queries": len(S), "n_species": n,
           "P0_single_query": m0.mean(), "P1_single_query": m1.mean(),
           "P0_wrong_among_single_query": (m0 & (U == 0)).sum() / max(m0.sum(), 1),
           "P1_wrong_among_single_query": (m1 & (U == 0)).sum() / max(m1.sum(), 1),
           "P0_wrong_n": int((m0 & (U == 0)).sum()), "P0_single_n": int(m0.sum()),
           "P1_wrong_n": int((m1 & (U == 0)).sum()), "P1_single_n": int(m1.sum()),
           "d_single_species": e[2] - e[0], "d_single_lo": q(B[:, 2] - B[:, 0])[0], "d_single_hi": q(B[:, 2] - B[:, 0])[1],
           "d_wrong_per100_species": 100 * (e[3] - e[1]), "d_wrong_per100_lo": 100 * q(B[:, 3] - B[:, 1])[0],
           "d_wrong_per100_hi": 100 * q(B[:, 3] - B[:, 1])[1],
           "d_wrong_among_single_ratio": e[3] / e[2] - e[1] / e[0], "d_ratio_lo": q(r1 - r0)[0], "d_ratio_hi": q(r1 - r0)[1]}
    return out


def by_species(sp, U, m, rule):
    S = pd.DataFrame({"species": sp, "one": m, "w": m & (U == 0)}).groupby("species").agg(n_queries=("one", "size"),
                                                                                      n_single=("one", "sum"), n_wrong=("w", "sum"))
    S["rate"] = np.where(S.n_single > 0, S.n_wrong / S.n_single.clip(lower=1), np.nan); S["rule"] = rule
    return S.reset_index()


def species_summary(S):
    x = S[S.n_single > 0]
    bins = pd.cut(x.n_single, [0, 9, 49, np.inf], labels=["1-9", "10-49", ">=50"])
    r = {"species_with_single": len(x), "mean_species_rate": x.rate.mean(), "median_species_rate": x.rate.median(),
         "share_species_rate_gt_5pct": (x.rate > 0.05).mean(), "species_with_any_wrong": int((x.n_wrong > 0).sum())}
    for b, g in x.groupby(bins, observed=False):
        r[f"n_species_{b}"] = len(g); r[f"mean_rate_{b}"] = g.rate.mean() if len(g) else np.nan
        r[f"pooled_rate_{b}"] = g.n_wrong.sum() / g.n_single.sum() if len(g) else np.nan
    return r


def main():
    rng = np.random.default_rng(SEED + 1500)
    D = pd.read_csv(C.RESULTS / "v12" / "decisions_per_query.tsv.gz", sep="\t")
    rows, sprows, spsum, folds = [], [], [], []
    for (var, ctx), d in D[D.context.isin(["B", "C"])].groupby(["variant", "context"]):
        m0, m1 = d.mask_P0.to_numpy(bool), d.mask_P1.to_numpy(bool); U = d.U.to_numpy(); sp = d.species.to_numpy()
        rows.append({"variant": var, "context": ctx, **paired(sp, U, m0, m1, rng)})
        for m, rule in ((m0, "P0"), (m1, "P1")):
            S = by_species(sp, U, m, rule); S["variant"], S["context"] = var, ctx; sprows.append(S)
            spsum.append({"variant": var, "context": ctx, "rule": rule, **species_summary(S)})
            for f, g in d.groupby("fold"):
                mm = g[f"mask_{rule}"].to_numpy(bool)
                folds.append({"variant": var, "context": ctx, "rule": rule, "fold": f, "n_queries": len(g),
                              "n_single": int(mm.sum()), "n_wrong": int((mm & (g.U.to_numpy() == 0)).sum())})
    P = pd.DataFrame(rows); P.to_csv(OUT / "paired_P1_minus_P0.tsv", sep="\t", index=False)
    pd.concat(sprows).to_csv(OUT / "per_species.tsv", sep="\t", index=False)
    SS = pd.DataFrame(spsum); SS.to_csv(OUT / "species_distribution_summary.tsv", sep="\t", index=False)
    F = pd.DataFrame(folds)
    F["share_of_rule_errors"] = F.n_wrong / F.groupby(["variant", "context", "rule"]).n_wrong.transform("sum")
    F.to_csv(OUT / "per_fold.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(P.round(4).T.to_string())
    print(SS.round(4).to_string(index=False))
    print(F[(F.variant == "hapshared") & (F.context == "C")].to_string(index=False))
    print(F[(F.variant == "hapshared") & (F.context == "B")].sort_values("n_wrong", ascending=False).head(6).to_string(index=False))


if __name__ == "__main__":
    main()
