"""v14.1: P1 as a filter of P0 -- how many wrong and how many correct single answers are moved to lists.
By construction P1 is a subset of P0 with the same species assignment, so it cannot add wrong single answers."""
import numpy as np
import pandas as pd
import common
from holdout import config as C

import rule_bombus as m75
IN = C.RESULTS / "v14" / "rules"; OUT = C.RESULTS / "v14"


def tradeoff(d, label, ctx):
    m0, m1 = m75.rule_masks(d); U = d.U.to_numpy()
    assert not (m1 & ~m0).any()          # P1 subset of P0
    moved = m0 & ~m1
    return {"set": label, "context": ctx, "n_queries": len(d),
            "P0_single_share_pooled": m0.mean(), "P1_single_share_pooled": m1.mean(),
            "d_single_share_pooled_pp": 100 * (m1.mean() - m0.mean()),
            "wrong_moved_to_list": int((moved & (U == 0)).sum()), "correct_moved_to_list": int((moved & (U == 1)).sum()),
            "wrong_kept_single": int((m1 & (U == 0)).sum()), "correct_kept_single": int((m1 & (U == 1)).sum()),
            "correct_lost_per_wrong_removed": (moved & (U == 1)).sum() / max((moved & (U == 0)).sum(), 1),
            "share_of_P0_wrong_removed": (moved & (U == 0)).sum() / max((m0 & (U == 0)).sum(), 1)}


def main():
    rows = []
    for t in ["bombus_original", "andrena", "lasioglossum", "megachile", "eupithecia", "xestia", "aedes", "hylaeus", "pardosa", "bombus_conflict_kept"]:
        d = pd.read_csv(IN / f"{t}.tsv.gz", sep="\t")
        for ctx, g in d.groupby("context"):
            rows.append(tradeoff(g, t.replace("_original", ""), {"B": "block unseen", "C": "continent unseen"}[ctx]))
    for cut in ("2021", "2019"):
        d = pd.read_csv(C.RESULTS / "v13" / "prospective" / f"queries_cutoff_{cut}.tsv.gz", sep="\t")
        rows.append(tradeoff(d, "bombus_forward", f"cutoff {cut}"))
    T = pd.DataFrame(rows); T.to_csv(OUT / "P1_as_filter_tradeoff.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
