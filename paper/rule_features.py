"""v14 B/C: P0 vs P1 (no tuning) on LOBO/LOCO hapshared for one genus.
Usage: rule_features.py <Genus> [--conflicts]  -> work/bombus/results/v14/rules/<genus>[_conflict_kept].tsv.gz"""
import os, sys
GENUS = sys.argv[1]; CONFLICTS = "--conflicts" in sys.argv
import common
os.environ["HOLDOUT_GENUS"] = GENUS
os.environ["HOLDOUT_WORKDIR"] = str(common.workdir(GENUS, CONFLICTS))
import numpy as np
import pandas as pd
from holdout import config as C
from holdout import refsets as S6
from holdout import scenarios as K

TAG = GENUS.lower() + ("_conflict_kept" if CONFLICTS else "_original" if GENUS == "Bombus" else "")
OUT = common.workdir("Bombus") / "results" / "v14" / "rules"; OUT.mkdir(parents=True, exist_ok=True)
TOL = 1e-6


def main():
    D, meta, dist, valid = K.load()
    lib = K.build_all(meta, dist, valid, K.PARAMS["strict_identity_min_overlap"])
    ca, cl = K.common_sets(meta, lib)
    sp = meta.species.to_numpy(); rid = meta.record_id.to_numpy(); cont = meta.continent.to_numpy()
    rows = []
    for ctx, reg, keep in (("B", "lobo_hapshared", ca), ("C", "loco_hapshared", cl)):
        for lab, ref, te in lib["L"][reg]:
            te = np.array([q for q in te if rid[q] in keep])
            if not len(te):
                continue
            sm = S6.SpeciesMin(dist, te, ref, sp); ev = sm.evaluate()
            M = np.minimum.reduceat(sm.sub, sm.starts, axis=1); Ms = np.sort(M, 1)
            d2 = Ms[:, 1] if M.shape[1] > 1 else np.full(len(te), np.inf)
            r = np.arange(len(te)); has = sm.tcol >= 0
            dtrue = np.full(len(te), np.inf); dtrue[has] = M[r[has], sm.tcol[has]]
            d1 = Ms[:, 0]
            X = pd.DataFrame({"record_id": rid[te], "species": sp[te], "continent": cont[te], "context": ctx, "fold": lab,
                              "d1": d1, "d2": np.where(np.isfinite(d2), d2, np.nan), "nS": ev["nS"],
                              "U": (ev["state"] == "U").astype(int), "y": (ev["credit"] > 0).astype(int)})
            for dl in (0.01, 0.02):
                X[f"list_has_true_{dl}"] = (dtrue <= d1 + dl + TOL).astype(int)
                X[f"list_size_{dl}"] = (M <= d1[:, None] + dl + TOL).sum(1)
            rows.append(X[np.isfinite(d1)])
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(OUT / f"{TAG}.tsv.gz", sep="\t", index=False)
    print(TAG, len(R), R.groupby("context").species.nunique().to_dict())


if __name__ == "__main__":
    main()
