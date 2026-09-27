"""v9 toy model: when do conspecific loss and competitor removal compensate? (exploratory, simulated data only)

Genus of S species in sister pairs, each on two continents (A, B) with 3 blocks each.
Sequence model (4 states, L sites, mutations = random substitutions):
  species lineage     : sisters differ by delta_sp; pairs differ by more
  continental lineage : each species' continent lineage differs from its species seq by gamma  (phylogeographic structure)
  block, individual   : small extra mutation (identical haplotypes remain common)
  mito capture        : a fraction iota of each species' individuals on a continent carry the SISTER's haplotype
                        of that continent (local introgression / shared haplotypes between species)
Queries: all individuals carrying their own species' haplotype. Identification: NN1 p-distance, ties
credited 1/|S| (as in the empirical pipeline). LOBO removes the query's block, LOCO its whole continent.
Shapley split of LOBO -> LOCO into conspecific removal and heterospecific removal (2 players, exact).
Outputs: results/v9/toy_model_grid.tsv, results/v9/toy_model_v9.{png,pdf}
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
from holdout import config as C

L, S, NB, NI = 600, 16, 3, 8          # sites, species (8 sister pairs), blocks per continent, individuals per block
DELTA_SP, DELTA_PAIR = 0.015, 0.08
BLOCK_MU, IND_MU = 0.002, 0.0015
GAMMAS = [0.0, 0.004, 0.008, 0.016, 0.03]
IOTAS = [0.0, 0.05, 0.1, 0.2, 0.35]
REPS = 10
OUT = C.RESULTS / "v9"; F = C.RESULTS / "v9"
OUT.mkdir(exist_ok=True); F.mkdir(exist_ok=True)


def mutate(seq, rate, rng):
    s = seq.copy(); k = rng.binomial(L, rate)
    if k:
        pos = rng.choice(L, k, replace=False)
        s[pos] = (s[pos] + rng.integers(1, 4, k)) % 4
    return s


def simulate(gamma, iota, rng, asym=False):
    root = rng.integers(0, 4, L)
    sp_seq = []
    for p in range(S // 2):
        anc = mutate(root, DELTA_PAIR / 2, rng)
        sp_seq += [mutate(anc, DELTA_SP / 2, rng), mutate(anc, DELTA_SP / 2, rng)]
    cont = {(s, c): mutate(sp_seq[s], gamma, rng) for s in range(S) for c in range(2)}
    seqs, spp, conts, blocks, own = [], [], [], [], []
    for s in range(S):
        sis = s ^ 1
        for c in range(2):
            for b in range(NB):
                bl = mutate(cont[(s, c)], BLOCK_MU, rng)
                bl_sis = mutate(cont[(sis, c)], BLOCK_MU, rng)
                for i in range(NI):
                    cap = rng.random() < (iota if (not asym or c == 0) else 0.0)
                    seqs.append(mutate(bl_sis if cap else bl, IND_MU, rng))
                    spp.append(s); conts.append(c); blocks.append(c * NB + b); own.append(not cap)
    X = np.array(seqs); d = (X[:, None, :] != X[None, :, :]).mean(2)
    np.fill_diagonal(d, np.inf)
    return d, np.array(spp), np.array(conts), np.array(blocks), np.array(own)


def credit(drow, mask, sp, s):
    if not mask.any():
        return 0.0
    v = np.where(mask, drow, np.inf); m = v.min()
    winners = set(sp[v <= m + 1e-9])
    return (s in winners) / len(winners)


def run_cell(gamma, iota, rng, asym=False):
    d, sp, co, bl, own = simulate(gamma, iota, rng, asym)
    n = len(sp); res = []
    for q in np.where(own)[0]:
        base = np.ones(n, bool); base[q] = False
        lobo = base & (bl != bl[q])
        rm = lobo & (co == co[q])                      # removed by LOCO beyond LOBO
        rc, rh = rm & (sp == sp[q]), rm & (sp != sp[q])
        f0 = credit(d[q], lobo, sp, sp[q]); fc = credit(d[q], lobo & ~rc, sp, sp[q])
        fh = credit(d[q], lobo & ~rh, sp, sp[q]); f1 = credit(d[q], lobo & ~rm, sp, sp[q])
        res.append((sp[q], f0, f1, 0.5 * ((fc - f0) + (f1 - fh)), 0.5 * ((fh - f0) + (f1 - fc))))
    r = pd.DataFrame(res, columns=["sp", "lobo", "loco", "phi_con", "phi_het"]).groupby("sp").mean().mean()
    return r


def main():
    rng = np.random.default_rng(C.RANDOM_SEED)
    rows = []
    for sc in ("symmetric", "one_continent"):
        for g in GAMMAS:
            for i in IOTAS:
                for rep in range(REPS):
                    r = run_cell(g, i, rng, sc == "one_continent")
                    rows.append({"scenario": sc, "gamma": g, "iota": i, "rep": rep, **r.to_dict()})
            print(sc, "gamma", g, "done", flush=True)
    T = pd.DataFrame(rows); T["net"] = T.loco - T.lobo
    T.to_csv(OUT / "toy_model_reps.tsv", sep="\t", index=False)
    G = T.groupby(["scenario", "gamma", "iota"])[["lobo", "loco", "phi_con", "phi_het", "net"]].agg(["mean", "std"])
    G.columns = ["_".join(c) for c in G.columns]; G = G.reset_index()
    G.to_csv(OUT / "toy_model_grid.tsv", sep="\t", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    lim = np.nanmax(np.abs(G[["phi_con_mean", "phi_het_mean", "net_mean"]].to_numpy()))
    for row, sc in enumerate(("symmetric", "one_continent")):
        for ax, q, t in zip(axes[row], ("phi_con_mean", "phi_het_mean", "net_mean"),
                            ("conspecific loss", "competitor removal", "net LOBO -> LOCO")):
            M = G[G.scenario == sc].pivot(index="gamma", columns="iota", values=q).to_numpy()
            im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
            for a in range(M.shape[0]):
                for b in range(M.shape[1]):
                    ax.text(b, a, f"{M[a, b]:+.2f}", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(IOTAS)), IOTAS); ax.set_yticks(range(len(GAMMAS)), GAMMAS); ax.grid(False)
            ax.set_title(t, loc="left", fontsize=9)
            if row == 1: ax.set_xlabel("iota (share with sister's haplotype)")
        axes[row, 0].set_ylabel(("A. shared haplotypes on both continents" if sc == "symmetric" else
                                 "B. shared haplotypes on one continent") + "\n\ngamma (between-continent divergence)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="change in expected recall (species mean)")
    fig.suptitle("Toy model (simulated data): the two components scale with shared haplotypes; the net depends on how they are distributed",
                 x=0.01, y=0.99, ha="left", fontsize=9)
    for e in ("png", "pdf"):
        fig.savefig(F / f"toy_model_v9.{e}", bbox_inches="tight")
    print(G.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
