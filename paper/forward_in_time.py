"""v13 B: forward-in-time test. Bombus, hapshared rule.
Reference library = analysis records deposited before the cutoff; queries = block-resolved records deposited
on/after it. Rules P0 (nearest species) and P1 (1% rule), lists delta 1%/2%, reject d1 > 3%. Outputs: results/v13/prospective/."""
import re, time
import numpy as np
import pandas as pd
import common
from holdout import config as C
from holdout import geography as V
from holdout import refsets as S6
from holdout import stats as S7
from holdout import scenarios as K

OUT = C.RESULTS / "v13" / "prospective"; OUT.mkdir(parents=True, exist_ok=True)
CUTOFFS = ("2021-01-01", "2019-01-01")
TOL = 1e-6


def deposit_dates(meta):
    b = pd.read_csv(C.RAW / "bold_bombus_v5.tsv", sep="\t", dtype=str,
                    usecols=["processid", "processid_minted_date", "sequence_upload_date", "insdc_acs"])
    b["d"] = pd.concat([pd.to_datetime(b.processid_minted_date, errors="coerce"),
                        pd.to_datetime(b.sequence_upload_date, errors="coerce")], axis=1).min(1)
    bold = b.groupby("processid").d.min()
    sub, acc = {}, None
    with open(C.RAW / "genbank_bombus_coi.gb") as fh:
        for line in fh:
            if line.startswith("ACCESSION"):
                acc = line.split()[1]
            m = re.search(r"Submitted \((\d{2}-[A-Z]{3}-\d{4})\)", line)
            if m and acc:
                d = pd.to_datetime(m.group(1), format="%d-%b-%Y"); sub[acc] = min(sub.get(acc, d), d)
    out = []
    for r, a in zip(meta.record_id, meta.accession.fillna("")):
        ds = [bold.get(r, pd.NaT)]
        for x in (a, str(r)):
            x = str(x).split(".")[0]
            if x in sub:
                ds.append(sub[x])
        ds = [d for d in ds if pd.notna(d)]
        out.append(min(ds) if ds else pd.NaT)
    return pd.Series(out, index=meta.index)


def evaluate(meta, dist, lib_mask, q):
    sp = meta.species.to_numpy(); blk = meta.block.to_numpy(); cont = meta.continent.to_numpy()
    ref = V.blocked_reference(q, lib_mask, meta, dist, haplotype=False)
    sm = S6.SpeciesMin(dist, q, ref, sp)
    ev = sm.evaluate()
    M = np.minimum.reduceat(sm.sub, sm.starts, axis=1)
    Ms = np.sort(M, 1); d1 = Ms[:, 0]; d2 = Ms[:, 1] if M.shape[1] > 1 else np.full(len(q), np.inf)
    r = np.arange(len(q)); has = sm.tcol >= 0
    dtrue = np.full(len(q), np.inf); dtrue[has] = M[r[has], sm.tcol[has]]
    rb, rc, rs = blk[sm.ref_idx], cont[sm.ref_idx], sp[sm.ref_idx]
    cov = []
    for i, qi in enumerate(q):
        m = rs == sp[qi]
        cov.append("absent" if not m.any() else "block" if (rb[m] == blk[qi]).any() else "continent" if (rc[m] == cont[qi]).any() else "none")
    D = pd.DataFrame({"record_id": meta.record_id.to_numpy()[q], "species": sp[q], "block": blk[q], "study_key": meta.study_key.to_numpy()[q],
                      "d1": d1, "d2": d2, "nS": ev["nS"], "U": (ev["state"] == "U").astype(int), "y": (ev["credit"] > 0).astype(int),
                      "cov": cov})
    for dl in (0.01, 0.02):
        D[f"list_size_{dl}"] = (M <= d1[:, None] + dl + TOL).sum(1)
        D[f"list_has_true_{dl}"] = (dtrue <= d1 + dl + TOL).astype(int)
    rej = D.d1 > 0.03
    D["one_P0"] = (D.nS == 1) & ~rej
    D["one_P1"] = (D.nS == 1) & (D.d1 <= 0.01) & (D.d2.fillna(np.inf) > 0.01) & ~rej
    D["reject"] = rej
    return D, len(ref)


def summarise(D, rule, dl):
    one = D[f"one_{rule}"]; lst = ~one & ~D.reject
    w = one & (D.U == 0)
    sm = pd.DataFrame({"species": D.species, "one": one, "w": w}).groupby("species").mean()
    return {"rule": rule, "delta": dl, "n_queries": len(D), "n_species": D.species.nunique(),
            "single_share_query": one.mean(), "single_share_species": sm.one.mean(),
            "wrong_among_single_query": w.sum() / one.sum() if one.sum() else np.nan,
            "wrong_among_single_ratio_of_species_means": sm.w.mean() / sm.one.mean() if sm.one.mean() else np.nan,
            "wrong_single_per100_query": 100 * w.mean(),
            "list_share": lst.mean(), "list_inclusion": D.loc[lst, f"list_has_true_{dl}"].mean() if lst.any() else np.nan,
            "list_size_mean": D.loc[lst, f"list_size_{dl}"].mean() if lst.any() else np.nan,
            "reject_share": D.reject.mean()}


def paired(D, rng):
    S = D.assign(o1=D.one_P1.astype(float), o0=D.one_P0.astype(float),
                 w1=(D.one_P1 & (D.U == 0)).astype(float), w0=(D.one_P0 & (D.U == 0)).astype(float))
    A = S.groupby("species")[["o1", "w1", "o0", "w0"]].mean().to_numpy(); n = len(A)
    B = A[rng.integers(0, n, (10000, n))].mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        derr = B[:, 1] / B[:, 0] - B[:, 3] / B[:, 2]
    e = A.mean(0)
    return {"d_single_share": e[0] - e[2], "d_single_lo": np.quantile(B[:, 0] - B[:, 2], .025), "d_single_hi": np.quantile(B[:, 0] - B[:, 2], .975),
            "d_wrong_among_single": e[1] / e[0] - e[3] / e[2], "d_wrong_lo": np.nanquantile(derr, .025), "d_wrong_hi": np.nanquantile(derr, .975),
            "d_wrong_per100": 100 * (e[1] - e[3]), "d_wrong_per100_lo": 100 * np.quantile(B[:, 1] - B[:, 3], .025),
            "d_wrong_per100_hi": 100 * np.quantile(B[:, 1] - B[:, 3], .975)}


def main():
    t0 = time.time()
    D0, meta, dist, valid = K.load()
    dep = deposit_dates(meta)
    meta = meta.assign(deposit=dep.to_numpy())
    inv = meta.in_v3.to_numpy(); res = V.resolved(meta.block.to_numpy()); known = meta.deposit.notna().to_numpy()
    print("analysis records", inv.sum(), "with deposit date", (inv & known).sum(), f"[{time.time()-t0:.0f}s]", flush=True)
    rng = np.random.default_rng(K.SEED + 1400)
    rows, comp, strata = [], [], []
    for cut in CUTOFFS:
        c = pd.Timestamp(cut)
        lib = inv & known & (meta.deposit < c).to_numpy()
        q = np.where(inv & known & res & (meta.deposit >= c).to_numpy())[0]
        D, nref = evaluate(meta, dist, lib, q)
        lib_studies = set(meta.study_key.to_numpy()[lib]) - {""}
        variants = {"all_queries": D, "species_in_library": D[D["cov"] != "absent"],
                    "new_studies_only": D[~D.study_key.isin(lib_studies)]}
        for vname, X in variants.items():
            for rule in ("P0", "P1"):
                for dl in (0.01, 0.02):
                    rows.append({"cutoff": cut, "subset": vname, "n_library": int(lib.sum()), "n_ref_after_blocking": nref,
                                 **summarise(X, rule, dl)})
            comp.append({"cutoff": cut, "subset": vname, **paired(X, rng)})
            if vname == "all_queries":
                for cv, g in X.groupby("cov"):
                    for rule in ("P0", "P1"):
                        strata.append({"cutoff": cut, "coverage_of_true_species": cv, **summarise(g, rule, 0.01)})
        D.to_csv(OUT / f"queries_cutoff_{cut[:4]}.tsv.gz", sep="\t", index=False)
        print(cut, "library", int(lib.sum()), "queries", len(q), f"[{time.time()-t0:.0f}s]", flush=True)
    Rr = pd.DataFrame(rows); Cc = pd.DataFrame(comp); Ss = pd.DataFrame(strata)
    Rr.to_csv(OUT / "rule_performance.tsv", sep="\t", index=False); Cc.to_csv(OUT / "P1_vs_P0.tsv", sep="\t", index=False)
    Ss.to_csv(OUT / "by_coverage.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    k = ["cutoff", "subset", "rule", "delta", "n_queries", "n_species", "single_share_query", "wrong_among_single_query",
         "wrong_among_single_ratio_of_species_means", "wrong_single_per100_query", "list_share", "list_inclusion", "list_size_mean", "reject_share"]
    print(Rr[k].round(4).to_string(index=False)); print(Cc.round(4).to_string(index=False))
    print(Ss[["cutoff", "coverage_of_true_species", "rule", "n_queries", "single_share_query", "wrong_among_single_query", "list_inclusion"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
