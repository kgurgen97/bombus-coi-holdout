"""v11 A: corrected reporting. No new models.
1) outcome table U/A/W by context; 2) hidden errors with explicit denominators; 3) paired R vs C comparison
at similarity >= 99%; 4) compensation reported as magnitudes + U/A/W transitions for all genera.
Outputs: results/v11/reporting/."""
import numpy as np
import pandas as pd
import common
from holdout import config as C
from holdout import stats as S7
from holdout import scenarios as K

OUT = C.RESULTS / "v11" / "reporting"; OUT.mkdir(parents=True, exist_ok=True)
CORE = ["Bombus", "Andrena", "Lasioglossum", "Megachile", "Eupithecia", "Xestia", "Aedes"]
EXTRA = ["Hylaeus", "Pardosa"]
ORDER = {"Bombus": "Hymenoptera", "Andrena": "Hymenoptera", "Lasioglossum": "Hymenoptera", "Megachile": "Hymenoptera",
         "Hylaeus": "Hymenoptera", "Eupithecia": "Lepidoptera", "Xestia": "Lepidoptera", "Aedes": "Diptera", "Pardosa": "Araneae"}


def sboot(df, col, rng):
    (m, lo, hi), = S7.species_boot(df, [col], rng)[0].values()
    return m, lo, hi


def outcomes(d, rng):
    rows = []
    d = d.assign(W=((d.y == 0)).astype(float), A=((d.y == 1) & (d.U == 0)).astype(float), Uf=d.U.astype(float),
                 conf=((d.nS == 1) & (d.d1 <= 0.01) & (d.d2.fillna(1) > 0.01)).astype(float))
    for (var, ctx), g in d.groupby(["variant", "context"]):
        r = {"variant": var, "context": ctx, "n_queries": len(g), "n_species": g.species.nunique()}
        for col, lab in (("Uf", "U_unique_correct"), ("A", "A_ambiguous_with_true"), ("W", "W_wrong_set"), ("y", "y_true_in_set")):
            m, lo, hi = sboot(g.assign(v=g[col].astype(float)), "v", rng)
            r.update({f"{lab}_query": g[col].mean(), f"{lab}_species": m, f"{lab}_lo": lo, f"{lab}_hi": hi})
        rows.append(r)
    T = pd.DataFrame(rows)
    H = []
    for (var, ctx), g in d.groupby(["variant", "context"]):
        W, AW, cf = g[g.W == 1], g[(g.W == 1) | (g.A == 1)], g[g.conf == 1]
        rec = {"variant": var, "context": ctx, "n_W": len(W), "n_A": int(g.A.sum()), "n_confident": len(cf),
               "n_hidden_confident_W": int((W.conf == 1).sum())}
        for lab, sub, col in (("hidden_among_W", W, "conf"), ("hidden_among_A_plus_W", AW, "conf"),
                              ("wrong_among_confident", cf, "W")):
            if len(sub):
                m, lo, hi = sboot(sub.assign(v=sub[col]), "v", rng)
                rec.update({f"{lab}_query": sub[col].mean(), f"{lab}_species": m, f"{lab}_lo": lo, f"{lab}_hi": hi,
                            f"{lab}_n_species": sub.species.nunique()})
        H.append(rec)
    return T, pd.DataFrame(H)


def paired_R_C(d, rng):
    rows = []
    for var in ("hapshared", "strict"):
        x = d[d.variant == var]
        R = x[x.context == "R"].set_index("record_id"); Cc = x[x.context == "C"].set_index("record_id")
        ids = R.index.intersection(Cc.index)
        ids = [i for i in ids if R.loc[i, "d1"] <= 0.01 and Cc.loc[i, "d1"] <= 0.01]
        P = pd.DataFrame({"species": R.loc[ids, "species"].to_numpy(),
                          "yR": R.loc[ids, "y"].to_numpy(float), "yC": Cc.loc[ids, "y"].to_numpy(float),
                          "UR": R.loc[ids, "U"].to_numpy(float), "UC": Cc.loc[ids, "U"].to_numpy(float)})
        P["dy"] = P.yC - P.yR; P["dU"] = P.UC - P.UR
        est, _ = S7.species_boot(P, ["yR", "yC", "dy", "UR", "UC", "dU"], rng)
        r = {"variant": var, "n_records": len(P), "n_species": P.species.nunique(),
             "species_with_any_y_loss": int((P.groupby("species").dy.mean() < 0).sum())}
        for k, (m, lo, hi) in est.items():
            r.update({f"{k}": m, f"{k}_lo": lo, f"{k}_hi": hi})
        rows.append(r)
    return pd.DataFrame(rows)


def compensation():
    rows, trans = [], []
    for tx in common.GENERA:
        r8 = common.workdir(tx) / "results" / "v8"
        for a in ("main", "blast"):
            s = pd.read_csv(r8 / a / "mechanism_summary.tsv", sep="\t").set_index(["variant", "quantity"])
            for v in ("hapshared", "strict"):
                g = lambda q: s.loc[(v, q)]
                con, het, tot, c2 = g("exp_phi_geo_con"), g("exp_phi_geo_het"), g("exp_total"), g("exp_G_minus_C2")
                rows.append({"taxon": tx, "set": "core" if tx in CORE else "extra", "order": ORDER[tx], "analysis": a,
                             "variant": v, "LOBO": g("exp_LOBO").estimate, "LOCO": g("exp_LOCO").estimate,
                             "phi_con": con.estimate, "phi_con_lo": con.species_lo, "phi_con_hi": con.species_hi,
                             "phi_het": het.estimate, "phi_het_lo": het.species_lo, "phi_het_hi": het.species_hi,
                             "net": tot.estimate, "net_lo": tot.species_lo, "net_hi": tot.species_hi,
                             "gross_reshuffle": abs(con.estimate) + het.estimate,
                             "net_over_gross": abs(tot.estimate) / (abs(con.estimate) + het.estimate),
                             "G_minus_C2": c2.estimate, "G_minus_C2_lo": c2.species_lo, "G_minus_C2_hi": c2.species_hi})
        t = pd.read_csv(r8 / "main" / "transitions.tsv", sep="\t")
        for v, g in t.groupby("variant"):
            n = g.n.sum(); f = lambda a, b: int(g[(g.from_LOBO == a) & (g.to_LOCO == b)].n.sum())
            trans.append({"taxon": tx, "set": "core" if tx in CORE else "extra", "variant": v, "n_queries": int(n),
                          "U_to_W": f("U", "W"), "U_to_A": f("U", "A"), "A_to_U": f("A", "U"), "A_to_W": f("A", "W"),
                          "W_to_U": f("W", "U"), "W_to_A": f("W", "A"),
                          **{f"share_{a}_to_{b}": f(a, b) / n for a in "UAW" for b in "UAW" if a != b},
                          "sum_to_W_share(U->W+A->W)": (f("U", "W") + f("A", "W")) / n,
                          "sum_to_U_share(A->U+W->U)": (f("A", "U") + f("W", "U")) / n,
                          "changed_state_share": g[g.from_LOBO != g.to_LOCO].n.sum() / n})
    return pd.DataFrame(rows), pd.DataFrame(trans)


def main():
    rng = np.random.default_rng(K.SEED + 1100)
    d = pd.read_csv(C.RESULTS / "v10" / "query_features.tsv.gz", sep="\t")
    T, H = outcomes(d, rng)
    T.to_csv(OUT / "outcomes_UAW_by_context.tsv", sep="\t", index=False)
    H.to_csv(OUT / "hidden_errors_explicit_denominators.tsv", sep="\t", index=False)
    P = paired_R_C(d, rng); P.to_csv(OUT / "paired_R_vs_C_sim99.tsv", sep="\t", index=False)
    Cm, Tr = compensation()
    Cm.to_csv(OUT / "compensation_magnitudes.tsv", sep="\t", index=False)
    Tr.to_csv(OUT / "transitions_UAW_all_genera.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(T[["variant", "context", "U_unique_correct_query", "A_ambiguous_with_true_query", "W_wrong_set_query",
             "U_unique_correct_species", "y_true_in_set_species"]].round(3).to_string(index=False))
    print(H.round(3).to_string(index=False))
    print(P.round(4).T.to_string())
    print(Cm[(Cm.analysis == "main")][["taxon", "set", "variant", "phi_con", "phi_het", "net", "net_lo", "net_hi",
                                       "gross_reshuffle", "net_over_gross"]].round(3).to_string(index=False))
    print(Tr.round(4).to_string(index=False))
    # net LOBO -> LOCO per method (never transferred between methods; CI containing 0 is not equivalence)
    N = Cm.pivot_table(index=["taxon", "set", "variant"], columns="analysis", values=["net", "net_lo", "net_hi"]).round(4)
    N.columns = [f"{a}_{b}" for a, b in N.columns]; N = N.reset_index()
    N.to_csv(OUT / "net_by_method.tsv", sep="\t", index=False)
    print(N.to_string(index=False))


if __name__ == "__main__":
    main()
