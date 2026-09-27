"""v13 A: influential reference records. Bombus, hapshared.
Exact single-record counterfactuals via per-species minima, minimum multiplicity and second minimum;
joint removal of the top-k records by full recomputation. Outputs: results/v13/influence/."""
import time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import common
from holdout import config as C
from holdout import geography as V
from holdout import scenarios as K

import calibration as m64
OUT = C.RESULTS / "v13" / "influence"; OUT.mkdir(parents=True, exist_ok=True)
TOL = 1e-6
KS = (1, 5, 10, 20, 50, 100)


def plan(meta, dist, valid):
    """hapshared splits of v10 (R: random CV folds, B: LOBO, C: LOCO) with query sets restricted as in v10."""
    thr = K.PARAMS["strict_identity_min_overlap"]
    lib = K.build_all(meta, dist, valid, thr); ca, cl = K.common_sets(meta, lib)
    rid = meta.record_id.to_numpy(); sp = meta.species.to_numpy(); Q = lib["Q"]
    U = meta.in_v3.to_numpy().copy()
    rng = np.random.default_rng(K.SEED + 80); grp = V.union_groups(meta, Q)
    Rh = [(f"fold{f}", K.blocker(meta, dist, valid, te, U.copy(), False, thr), te)
          for f, te in enumerate(V.grouped_folds(Q, sp[Q], grp, K.PARAMS["random_cv_folds"], rng))]
    out = []
    for ctx, splits, keep in (("R", Rh, ca), ("B", lib["L"]["lobo_hapshared"], ca), ("C", lib["L"]["loco_hapshared"], cl)):
        for lab, ref, te in splits:
            te = np.array([q for q in te if rid[q] in keep])
            if len(te):
                out.append((ctx, lab, np.asarray(ref), te))
    return out


def states(M, tcol):
    d1 = M.min(1); fin = np.isfinite(d1)
    tie = (M <= d1[:, None] + TOL) & np.isfinite(M)
    r = np.arange(len(M)); has = tcol >= 0
    tin = np.zeros(len(M), bool); tin[has] = tie[r[has], tcol[has]]
    n = tie.sum(1)
    return np.where(~fin, "N", np.where(tin & (n == 1), "U", np.where(tin, "A", "W"))), tie


def split_arrays(dist, sp, ref, te):
    order = np.argsort(sp[ref], kind="stable"); ref = ref[order]
    rsp = sp[ref]; spp, starts = np.unique(rsp, return_index=True)
    seg = np.diff(np.r_[starts, len(ref)])
    sub = dist[np.ix_(te, ref)]; sub = np.where(np.isnan(sub), np.inf, sub).astype(np.float32)
    pos = {x: i for i, x in enumerate(spp)}
    tcol = np.array([pos.get(x, -1) for x in sp[te]])
    return ref, spp, starts, seg, sub, tcol


def main():
    t0 = time.time()
    D, meta, dist, valid = K.load()
    sp = meta.species.to_numpy(); rid = meta.record_id.to_numpy()
    P = plan(meta, dist, valid)
    print("splits", len(P), f"[{time.time()-t0:.0f}s]", flush=True)
    N = len(meta)
    fix = {c: np.zeros(N) for c in "RBC"}; brk = {c: np.zeros(N) for c in "RBC"}; comp = {c: np.zeros(N) for c in "RBC"}
    base_rows = []; victims = defaultdict(Counter); uniq_fixed = defaultdict(set)
    cache = []
    for ctx, lab, ref0, te in P:
        ref, spp, starts, seg, sub, tcol = split_arrays(dist, sp, ref0, te)
        M = np.minimum.reduceat(sub, starts, axis=1)
        st0, tie0 = states(M, tcol)
        Mexp = np.repeat(M, seg, axis=1)
        at_min = sub <= Mexp + TOL
        cnt = np.add.reduceat(at_min.astype(np.int32), starts, axis=1)
        m2 = np.minimum.reduceat(np.where(at_min, np.inf, sub), starts, axis=1)
        # column of the (unique) minimum record per species where cnt == 1
        argm = np.full(M.shape, -1)
        for j, (a, n) in enumerate(zip(starts, seg)):
            argm[:, j] = a + np.argmin(sub[:, a:a + n], axis=1)
        d1 = M.min(1)
        for i in range(len(te)):
            if st0[i] == "N":
                continue
            tset = np.where(tie0[i])[0]
            # competitors at d1: every heterospecific record at the best distance
            for j in tset:
                if j != tcol[i]:
                    a, n = starts[j], seg[j]
                    cols = a + np.where(sub[i, a:a + n] <= d1[i] + TOL)[0]
                    np.add.at(comp[ctx], ref[cols], 1)
                if cnt[i, j] != 1:
                    continue
                Mi = M[i].copy(); Mi[j] = m2[i, j]
                s1, _ = states(Mi[None, :], tcol[i:i + 1])
                if s1[0] == st0[i]:
                    continue
                r = ref[argm[i, j]]
                if s1[0] == "U" and st0[i] in "AW":
                    fix[ctx][r] += 1; victims[r][sp[te[i]]] += 1; uniq_fixed[r].add(te[i])
                elif st0[i] == "U":
                    brk[ctx][r] += 1
        base_rows.append(pd.DataFrame({"idx": te, "species": sp[te], "context": ctx, "split": lab, "state": st0}))
        cache.append((ctx, lab, ref, spp, starts, sub, tcol))
        print(ctx, lab, f"[{time.time()-t0:.0f}s]", flush=True)
    B = pd.concat(base_rows, ignore_index=True)
    R = pd.DataFrame({"record_id": rid, "species": sp, "source": meta.source.to_numpy(), "country": meta.country.to_numpy(),
                      "block": meta.block.to_numpy(), "bold_bin": meta.bold_bin.to_numpy()})
    for c in "RBC":
        R[f"fix_{c}"] = fix[c]; R[f"break_{c}"] = brk[c]; R[f"compete_{c}"] = comp[c]
    R["fix"] = R[[f"fix_{c}" for c in "RBC"]].sum(1); R["break"] = R[[f"break_{c}" for c in "RBC"]].sum(1)
    R["compete"] = R[[f"compete_{c}" for c in "RBC"]].sum(1); R["net"] = R.fix - R["break"]
    # descriptive evidence for records
    maj = meta.groupby("species").bold_bin.agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan)
    R["bin_is_own_species_major"] = [b == maj.get(s) if isinstance(b, str) else np.nan for b, s in zip(R.bold_bin, R.species)]
    R["isolated_flag"] = meta.flag_isolated.to_numpy()
    R["fixed_unique_queries"] = [len(uniq_fixed.get(i, ())) for i in range(N)]
    R["fixed_query_species"] = ["; ".join(f"{k.replace('Bombus ', '')}:{v}" for k, v in victims[i].most_common(3)) if i in victims else ""
                                for i in range(N)]
    R = R.sort_values(["net", "fix"], ascending=False)
    # species whose queries each record most often competes with
    R.to_csv(OUT / "record_influence.tsv.gz", sep="\t", index=False)
    # joint removal of top-k by net
    top = R[R.net > 0].record_id.tolist()
    pos = {r: i for i, r in enumerate(rid)}
    rows = []
    n_harm0 = int((B.state.isin(["A", "W"])).sum())
    for k in (0,) + KS:
        rem = set(pos[r] for r in top[:k])
        st_all = []
        for ctx, lab, ref, spp, starts, sub, tcol in cache:
            keep = ~np.isin(ref, list(rem)) if rem else np.ones(len(ref), bool)
            sk = np.where(keep[None, :], sub, np.inf)
            M = np.minimum.reduceat(sk, starts, axis=1)
            st, _ = states(M, tcol); st_all.append(st)
        st = np.concatenate(st_all)
        d = B.assign(new=st)
        r = {"k": k, "n_pairs": len(d), "n_harm_before": n_harm0,
             "harm_to_U": int((d.state.isin(["A", "W"]) & (d.new == "U")).sum()),
             "U_to_harm": int(((d.state == "U") & d.new.isin(["A", "W"])).sum()),
             "U_query": float((d.new == "U").mean()),
             "U_species": float(d.assign(u=(d.new == "U")).groupby(["context", "species"]).u.mean().groupby("context").mean().mean())}
        for c in "RBC":
            x = d[d.context == c]
            r[f"U_{c}"] = float((x.new == "U").mean()); r[f"harm_to_U_{c}"] = int((x.state.isin(["A", "W"]) & (x.new == "U")).sum())
        rows.append(r)
    J = pd.DataFrame(rows); J["share_of_harm_fixed"] = J.harm_to_U / n_harm0
    J.to_csv(OUT / "topk_joint_removal.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(J.round(4).to_string(index=False))
    print(R.head(15)[["record_id", "species", "source", "country", "fix", "break", "fixed_unique_queries", "fixed_query_species",
                      "bin_is_own_species_major"]].to_string(index=False))
    print("records with net>0:", int((R.net > 0).sum()), " with fix>0:", int((R.fix > 0).sum()), f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
