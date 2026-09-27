"""v14 B/C summary: paired P1 - P0 per genus and for the conflict-kept Bombus variant (fixed and full query sets)."""
import numpy as np
import pandas as pd
import common
from holdout import config as C

import rule_bombus as m75
IN = C.RESULTS / "v14" / "rules"; OUT = C.RESULTS / "v14"
CORE = ["bombus_original", "andrena", "lasioglossum", "megachile", "eupithecia", "xestia", "aedes"]; EXTRA = ["hylaeus", "pardosa"]


def summarise(d, rng, label):
    rows = []
    for ctx, g in d.groupby("context"):
        m0, m1 = m75.rule_masks(g)
        r = m75.paired(g.species.to_numpy(), g.U.to_numpy(), m0, m1, rng)
        lst = ~m1 & ~(g.d1 > 0.03).to_numpy()
        r.update({"P1_list_inclusion_1pct": g.list_has_true_0_01.to_numpy()[lst].mean() if lst.any() else np.nan,
                  "P1_list_inclusion_2pct": g.list_has_true_0_02.to_numpy()[lst].mean() if lst.any() else np.nan})
        S1 = m75.by_species(g.species.to_numpy(), g.U.to_numpy(), m1, "P1")
        r["P1_mean_species_rate"] = S1.rate.mean(); r["P1_species_rate_gt_5pct"] = (S1.rate > 0.05).mean()
        rows.append({"set": label, "context": ctx, **r})
    return rows


def main():
    rng = np.random.default_rng(m75.SEED + 1600)
    rows = []
    for t in CORE + EXTRA:
        d = pd.read_csv(IN / f"{t}.tsv.gz", sep="\t").rename(columns=lambda c: c.replace("0.0", "0_0"))
        for r in summarise(d, rng, "core" if t in CORE else "extra"):
            rows.append({"genus": t.replace("_original", ""), **r})
    G = pd.DataFrame(rows); G.to_csv(OUT / "C_generality_P0_P1_by_genus.tsv", sep="\t", index=False)
    o = pd.read_csv(IN / "bombus_original.tsv.gz", sep="\t").rename(columns=lambda c: c.replace("0.0", "0_0"))
    k = pd.read_csv(IN / "bombus_conflict_kept.tsv.gz", sep="\t").rename(columns=lambda c: c.replace("0.0", "0_0"))
    key = ["record_id", "context"]
    shared = o[key].merge(k[key])
    rows = []
    for view, (a, b) in (("fixed_common_set", (o.merge(shared), k.merge(shared))), ("full_set", (o, k))):
        for lab, d in (("original_qc", a), ("conflict_kept", b)):
            for r in summarise(d, rng, view):
                rows.append({"qc": lab, **r})
    B = pd.DataFrame(rows); B.to_csv(OUT / "B_conflict_kept_P0_P1.tsv", sep="\t", index=False)
    pd.set_option("display.width", 260)
    k1 = ["genus", "set", "context", "n_queries", "n_species", "P0_single_query", "P1_single_query", "P0_wrong_among_single_query",
          "P1_wrong_among_single_query", "d_wrong_per100_species", "d_wrong_per100_lo", "d_wrong_per100_hi", "d_single_species",
          "P1_mean_species_rate", "P1_list_inclusion_1pct"]
    print(G[k1].round(4).to_string(index=False))
    k2 = ["set", "qc", "context", "n_queries", "P0_wrong_n", "P0_single_n", "P1_wrong_n", "P1_single_n", "P0_wrong_among_single_query",
          "P1_wrong_among_single_query", "d_wrong_per100_species", "d_wrong_per100_lo", "d_wrong_per100_hi", "P1_mean_species_rate"]
    print(B[k2].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
