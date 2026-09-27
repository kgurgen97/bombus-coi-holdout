"""v11.1: calibration on a fully unseen CONTINENT.
Outer loop = leave-one-continent-out. The training universe excludes the whole continent (and unresolved
records from its countries); libraries/features of training queries are rebuilt on it (calibration_unseen_blocks.py).
Test rows = context-C rows (LOCO, v10.1 features) of queries from that continent.
Models: v10 logistic (no inner selection -> fully isolated) and the v10.1 inner-selected model (partially isolated).
Outputs: results/v11/calibration_continent/."""
import time
import numpy as np
import pandas as pd
import common
from holdout import config as C
from holdout import geography as V
from holdout import scenarios as K

import calibration_unseen_blocks as m66
OUT = C.RESULTS / "v11" / "calibration_continent"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    D, meta, dist, valid = K.load()
    blk = meta.block.to_numpy(); cont = meta.continent.to_numpy(); ctry = meta.country.fillna("").to_numpy()
    unres = (~V.resolved(blk)) & (blk != "Unknown")
    bc = V.block_country_sets(meta[meta.in_v3.to_numpy()])
    test_all = pd.read_csv(C.RESULTS / "v10_1" / "query_features_plus.tsv.gz", sep="\t")
    test_all["continent"] = test_all.block.map(V.continent_of)
    key = m66.sha_files([C.CLEAN / "clean_metadata.tsv", C.ALIGNMENT / "matches.npy", C.ALIGNMENT / "valid.npy",
                         *common.code_files("calibration.py", "calibration_nested.py", "calibration_unseen_blocks.py", "calibration_unseen_continents.py")])
    cdir = OUT / f"cache_{key}"; cdir.mkdir(exist_ok=True)
    preds, sel = [], []
    for k in sorted(test_all[test_all.context == "C"].continent.unique()):
        tk = time.time()
        kc = set().union(*[bc.get(b, set()) for b in set(blk[cont == k])])
        bad = {"Europe": ("Russia_unresolved",), "Asia": ("Russia_unresolved", "China_unresolved"),
               "North_America": ("Canada_unresolved", "United States_unresolved")}.get(k, ())
        universe = (cont != k) & ~(unres & (np.isin(ctry, list(kc)) | np.isin(blk, list(bad))))
        f = cdir / f"train_rows_{k}.tsv.gz"
        if f.exists():
            tr_all = pd.read_csv(f, sep="\t")
        else:
            tr_all = m66.build_rows_universe(meta, dist, valid, universe); tr_all.to_csv(f, sep="\t", index=False)
        assert not np.isin(tr_all.record_id, meta.record_id.to_numpy()[~universe]).any()
        for var in ("hapshared", "strict"):
            tr = tr_all[tr_all.variant == var].reset_index(drop=True)
            te = test_all[(test_all.variant == var) & (test_all.context == "C") & (test_all.continent == k)].reset_index(drop=True)
            out = te[["record_id", "species", "block", "continent", "context", "variant", "y", "U", "d1"]].copy()
            for target in ("y", "U"):
                name, model, score = m66.select_and_fit(tr, target)
                out[f"p_{target}_selected"] = model.predict(te)
                out[f"p_{target}_logistic"] = m66.m65.LR(m66.m65.X_lr).fit(tr, tr[target].to_numpy()).predict(te)
                sel.append({"outer_continent": k, "variant": var, "target": target, "chosen": name,
                            "n_train": len(tr), "n_test": len(te)})
            preds.append(out)
        print(f"{k}: train rows {len(tr_all)} [{time.time()-tk:.0f}s]", flush=True)
    P = pd.concat(preds, ignore_index=True); P.to_csv(OUT / "oob_predictions.tsv.gz", sep="\t", index=False)
    pd.DataFrame(sel).to_csv(OUT / "selected_models.tsv", sep="\t", index=False)
    rows = []
    for var in ("hapshared", "strict"):
        d = P[P.variant == var].reset_index(drop=True)
        for target in ("y", "U"):
            yv = d[target].to_numpy()
            for lab, p in (("logistic_fully_isolated", d[f"p_{target}_logistic"].to_numpy()),
                           ("selected_partially_isolated", d[f"p_{target}_selected"].to_numpy()),
                           ("similarity_1_minus_d1", 1 - d.d1.to_numpy())):
                r = [x for x in m66.m64.metrics(d, p, yv, lab) if x["context"] == "C"][0]
                rows.append({**r, "variant": var, "target": target})
                # lowest decile of each score
                b = np.argsort(p)[: len(p) // 10]
                rows[-1].update({"lowest_decile_predicted": p[b].mean(), "lowest_decile_observed": yv[b].mean(),
                                 "lowest_decile_n_species": d.species.to_numpy()[b].size and len(set(d.species.to_numpy()[b]))})
    Mt = pd.DataFrame(rows); Mt.to_csv(OUT / "calibration_metrics.tsv", sep="\t", index=False)
    print(Mt.round(4).to_string(index=False)); print(f"done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
