"""v11 B: original QC vs conflicting labels kept.
(i) fixed common query set: record_ids present in the LOCO mechanism tables of both variants -> recompute
    species-mean phi_con, phi_het, net and U/A/W transitions from per-query tables (species bootstrap);
(ii) full set of each variant: the saved mechanism summaries.
Outputs: results/v11/conflict_kept/."""
import numpy as np
import pandas as pd
import common
from holdout import config as C
from holdout import stats as S7
from holdout import scenarios as K

OUT = C.RESULTS / "v11" / "conflict_kept"; OUT.mkdir(parents=True, exist_ok=True)
CORE = {"Bombus", "Andrena", "Lasioglossum", "Megachile", "Eupithecia", "Xestia", "Aedes"}
COLS = ["phi_geo_con_credit", "phi_geo_het_credit", "net", "credit_LOBO", "credit_LOCO"]


def qtable(root, a, v):
    t = pd.read_csv(root / "results" / "v8" / a / "mechanism_query_table.tsv", sep="\t")
    t = t[t.variant == v].copy(); t["net"] = t.credit_LOCO - t.credit_LOBO
    return t.set_index("record_id")


def summarise(t, rng):
    est, _ = S7.species_boot(t.reset_index(), COLS, rng)
    r = {}
    for c, (m, lo, hi) in est.items():
        r.update({c: m, f"{c}_lo": lo, f"{c}_hi": hi})
    n = len(t)
    for a in "UAW":
        for b in "UAW":
            if a != b:
                r[f"share_{a}_to_{b}"] = ((t.state_LOBO == a) & (t.state_LOCO == b)).sum() / n
    return r


def main():
    rng = np.random.default_rng(K.SEED + 1200)
    rows, full, size = [], [], []
    for tx in common.GENERA:
        o, n = common.workdir(tx), common.workdir(tx, conflicts=True)
        if not (n / "results" / "v8" / "blast" / "mechanism_query_table.tsv").exists():
            print("not ready:", tx); continue
        for root, lab in ((o, "original_qc"), (n, "conflict_kept")):
            m = pd.read_csv(root / "clean" / "clean_metadata.tsv", sep="\t", low_memory=False)
            mm = m[m.dataset == "main"]
            size.append({"taxon": tx, "qc": lab, "n_main": len(mm), "n_species_main": mm.species.nunique(),
                         "n_conflict_label_main": int(mm.get("conflict_label", pd.Series(False, index=mm.index)).astype(str).eq("True").sum())})
        for a in ("main", "blast"):
            for v in ("hapshared", "strict"):
                to, tn = qtable(o, a, v), qtable(n, a, v)
                shared = to.index.intersection(tn.index)
                for view, (x, y) in (("fixed_common_set", (to.loc[shared], tn.loc[shared])), ("full_set", (to, tn))):
                    for lab, t in (("original_qc", x), ("conflict_kept", y)):
                        rows.append({"taxon": tx, "set": "core" if tx in CORE else "extra", "analysis": a, "variant": v,
                                     "view": view, "qc": lab, "n_queries": len(t), "n_species": t.species.nunique(),
                                     "n_common": len(shared), **summarise(t, rng)})
                for lab, root in (("original_qc", o), ("conflict_kept", n)):
                    s = pd.read_csv(root / "results" / "v8" / a / "mechanism_summary.tsv", sep="\t")
                    s = s[s.variant == v].set_index("quantity")
                    full.append({"taxon": tx, "analysis": a, "variant": v, "qc": lab,
                                 **{q: s.loc[q, "estimate"] for q in ("exp_LOBO", "exp_LOCO", "exp_total",
                                                                      "exp_phi_geo_con", "exp_phi_geo_het", "exp_G_minus_C2")}})
        print(tx, "done", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "compare_query_level.tsv", sep="\t", index=False)
    pd.DataFrame(full).to_csv(OUT / "compare_saved_summaries.tsv", sep="\t", index=False)
    pd.DataFrame(size).to_csv(OUT / "dataset_sizes.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(size).to_string(index=False))
    k = ["taxon", "analysis", "variant", "view", "qc", "n_queries", "phi_geo_con_credit", "phi_geo_het_credit", "net", "net_lo", "net_hi",
         "share_U_to_W", "share_A_to_U"]
    print(R[k].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
