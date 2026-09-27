"""v11 C: calibration validated on fully unseen blocks.
For every outer block b: rebuild all libraries/features of TRAINING queries on a universe without block b
(and without unresolved records from b's countries); select the model on those rows only (v10.1 procedure);
predict the v10 test rows of block b. Targets: y (primary) and U (secondary). Outputs: results/v11/calibration/."""
import hashlib, json, sys, time
import numpy as np
import pandas as pd
import common
from holdout import config as C
from holdout import geography as V
from holdout import scenarios as K

import calibration as m64
import calibration_nested as m65
P8 = K.PARAMS
OUT = C.RESULTS / "v11" / "calibration"; OUT.mkdir(parents=True, exist_ok=True)


def sha_files(paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as fh:
            for ch in iter(lambda: fh.read(1 << 22), b""):
                h.update(ch)
    return h.hexdigest()[:20]


def build_rows_universe(meta, dist, valid, universe):
    """= m64.build_rows restricted to a universe (queries AND references), with n_sp_2pct."""
    thr = P8["strict_identity_min_overlap"]
    lib = K.build_all(meta, dist, valid, thr, universe=universe)
    ca, cl = K.common_sets(meta, lib)
    rid = meta.record_id.to_numpy(); sp = meta.species.to_numpy(); Q = lib["Q"]
    U = meta.in_v3.to_numpy() & universe
    rng = np.random.default_rng(K.SEED + 80)
    grp = V.union_groups(meta, Q)
    Rh = [(f"fold{f}", K.blocker(meta, dist, valid, te, U.copy(), False, thr), te)
          for f, te in enumerate(V.grouped_folds(Q, sp[Q], grp, P8["random_cv_folds"], rng))]
    plan = [("R", "hapshared", Rh, ca), ("R", "strict", lib["L"]["random_cv_strict"], ca),
            ("B", "hapshared", lib["L"]["lobo_hapshared"], ca), ("B", "strict", lib["L"]["lobo_strict"], ca),
            ("C", "hapshared", lib["L"]["loco_hapshared"], cl), ("C", "strict", lib["L"]["loco_strict"], cl)]
    out = []
    for ctx, v, splits, keep in plan:
        for lab, ref, te in splits:
            assert universe[ref].all()
            te = np.array([q for q in te if rid[q] in keep])
            if len(te):
                out.append(m65.features_plus(meta, dist, ref, te, ctx, v, lab))
    return pd.concat(out, ignore_index=True)


def select_and_fit(tr, target):
    """v10.1 inner selection on training rows only; returns (name, fitted model, inner scores)."""
    y = tr[target].to_numpy(); rng = np.random.default_rng(K.SEED + 200)
    folds = m65.block_folds(tr.block.unique(), 5, rng); score = {}
    for name in m65.candidates(0):
        se = []
        for fi, fb in enumerate(folds):
            v = tr.block.isin(fb).to_numpy()
            if y[~v].min() == y[~v].max():
                continue
            m = m65.candidates(K.SEED + 300 + fi)[name].fit(tr[~v].reset_index(drop=True), y[~v])
            se.append(np.mean((m.predict(tr[v]) - y[v]) ** 2) * v.sum())
        score[name] = sum(se) / len(tr)
    best = min(score, key=score.get)
    return best, m65.candidates(K.SEED + 400)[best].fit(tr.reset_index(drop=True), y), score


def main():
    t0 = time.time()
    D, meta, dist, valid = K.load()
    blk = meta.block.to_numpy(); ctry = meta.country.fillna("").to_numpy()
    unres = (~V.resolved(blk)) & (blk != "Unknown")
    bc = V.block_country_sets(meta[meta.in_v3.to_numpy()])
    test_all = pd.read_csv(C.RESULTS / "v10_1" / "query_features_plus.tsv.gz", sep="\t")
    key = sha_files([C.CLEAN / "clean_metadata.tsv", C.ALIGNMENT / "matches.npy", C.ALIGNMENT / "valid.npy",
                     *common.code_files("calibration.py", "calibration_nested.py", "calibration_unseen_blocks.py")])
    cdir = OUT / f"cache_{key}"; cdir.mkdir(exist_ok=True)
    (OUT / "cache_key.txt").write_text(key + "\n")
    blocks = sorted(test_all.block.unique())
    only = sys.argv[1:]  # optional subset of blocks (for testing)
    preds, sel = [], []
    for b in (only or blocks):
        tb = time.time()
        universe = (blk != b) & ~(unres & np.isin(ctry, list(bc.get(b, set()))))
        f = cdir / f"train_rows_{b}.tsv.gz"
        if f.exists():
            tr_all = pd.read_csv(f, sep="\t")
        else:
            tr_all = build_rows_universe(meta, dist, valid, universe); tr_all.to_csv(f, sep="\t", index=False)
        assert not (tr_all.block == b).any()
        assert not np.isin(tr_all.record_id, meta.record_id.to_numpy()[~universe]).any()
        for var in ("hapshared", "strict"):
            tr = tr_all[tr_all.variant == var].reset_index(drop=True)
            te = test_all[(test_all.variant == var) & (test_all.block == b)].reset_index(drop=True)
            if te.empty:
                continue
            out = te[["record_id", "species", "block", "context", "variant", "y", "U", "d1"]].copy()
            for target in ("y", "U"):
                name, model, score = select_and_fit(tr, target)
                out[f"p_{target}_selected"] = model.predict(te)
                lr = m65.LR(m65.X_lr).fit(tr, tr[target].to_numpy())
                out[f"p_{target}_logistic"] = lr.predict(te)
                sel.append({"outer_block": b, "variant": var, "target": target, "chosen": name,
                            **{f"inner_{k}": v for k, v in score.items()}, "n_train": len(tr), "n_test": len(te)})
            preds.append(out)
        print(f"{b}: train rows {len(tr_all)} [{time.time()-tb:.0f}s, total {time.time()-t0:.0f}s]", flush=True)
    P = pd.concat(preds, ignore_index=True)
    tag = "_subset" if only else ""
    P.to_csv(OUT / f"oob_predictions{tag}.tsv.gz", sep="\t", index=False)
    pd.DataFrame(sel).to_csv(OUT / f"selected_models{tag}.tsv", sep="\t", index=False)
    if only:
        return
    rows = []
    for var in ("hapshared", "strict"):
        d = P[P.variant == var].reset_index(drop=True)
        for target in ("y", "U"):
            yv = d[target].to_numpy()
            for lab in ("selected", "logistic"):
                for r in m64.metrics(d, d[f"p_{target}_{lab}"].to_numpy(), yv, f"{lab}"):
                    rows.append({**r, "variant": var, "target": target})
            for r in m64.metrics(d, 1 - d.d1.to_numpy(), yv, "similarity_1_minus_d1"):
                rows.append({**r, "variant": var, "target": target})
    Mt = pd.DataFrame(rows); Mt.to_csv(OUT / "calibration_metrics.tsv", sep="\t", index=False)
    crit = []
    for var in ("hapshared", "strict"):
        for target in ("y", "U"):
            for lab in ("selected", "logistic"):
                h = Mt[(Mt.variant == var) & (Mt.target == target)].set_index(["model", "context"])
                ok = all(h.loc[(lab, c), "ece"] <= 0.05 and h.loc[(lab, c), "brier"] < h.loc[("similarity_1_minus_d1", c), "brier"]
                         for c in "RBC")
                crit.append({"variant": var, "target": target, "model": lab, "criterion_met": ok})
    pd.DataFrame(crit).to_csv(OUT / "criterion.tsv", sep="\t", index=False)
    print(Mt.round(4).to_string(index=False)); print(pd.DataFrame(crit).to_string(index=False))
    print(f"done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
