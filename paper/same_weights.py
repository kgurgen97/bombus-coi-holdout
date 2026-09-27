"""v15: main contrast (net change vs outcome turnover) under the SAME weighting, both pooled over queries and
species-weighted, with species-bootstrap intervals. Source: v8 per-query mechanism tables. Output: results/v15/same_weights.tsv"""
import numpy as np
import pandas as pd
import common
from holdout import config as C
from holdout import scenarios as K

OUT = C.RESULTS / "v15"; OUT.mkdir(exist_ok=True)


def main():
    rng = np.random.default_rng(K.SEED + 1700); rows = []
    for tx in common.GENERA:
        for a in ("main", "blast"):
            t = pd.read_csv(common.workdir(tx) / "results" / "v8" / a / "mechanism_query_table.tsv", sep="\t")
            for v, g in t.groupby("variant"):
                g = g.assign(net=g.credit_LOCO - g.credit_LOBO, changed=(g.state_LOBO != g.state_LOCO).astype(float),
                             UW=((g.state_LOBO == "U") & (g.state_LOCO == "W")).astype(float),
                             AU=((g.state_LOBO == "A") & (g.state_LOCO == "U")).astype(float))
                cols = ["net", "changed", "UW", "AU"]
                S = g.groupby("species")[cols].mean(); A = S.to_numpy(); n = len(A)
                B = A[rng.integers(0, n, (10000, n))].mean(1)
                r = {"taxon": tx, "analysis": a, "variant": v, "n_queries": len(g), "n_species": n}
                for i, c in enumerate(cols):
                    r[f"{c}_pooled"] = g[c].mean()
                    r[f"{c}_species"] = A[:, i].mean(); r[f"{c}_species_lo"] = np.quantile(B[:, i], .025); r[f"{c}_species_hi"] = np.quantile(B[:, i], .975)
                rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(OUT / "same_weights.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(R[(R.analysis == "main") & (R.variant == "hapshared")][["taxon", "n_queries", "net_pooled", "net_species", "changed_pooled",
          "changed_species", "changed_species_lo", "changed_species_hi", "UW_species", "AU_species"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
