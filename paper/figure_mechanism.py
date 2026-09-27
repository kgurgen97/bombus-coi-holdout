"""Fig 5 (mechanism in sequence space, PCoA of core p-distances) and Fig S3 (t-SNE overview).
Uses existing data and the frozen libraries only; no new analysis. -> figures/."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import common
from holdout import config as C
from holdout import refsets as S6
from holdout import scenarios as K

F = common.FIGURES; F.mkdir(parents=True, exist_ok=True)
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.bbox": "tight", "figure.dpi": 150})


def pcoa(Dm):
    Dm = np.where(np.isfinite(Dm), Dm, np.nanmax(Dm[np.isfinite(Dm)]))
    np.fill_diagonal(Dm, 0)
    n = len(Dm); J = np.eye(n) - 1 / n
    B = -0.5 * J @ (Dm ** 2) @ J
    w, v = np.linalg.eigh(B); o = np.argsort(w)[::-1]
    w, v = w[o], v[:, o]
    X = v[:, :2] * np.sqrt(np.maximum(w[:2], 0))
    return X, w[:2] / w[w > 0].sum()


def row_legend(axs):
    H = {}
    for a in axs:
        for h, l in zip(*a.get_legend_handles_labels()):
            H.setdefault(l, h)
    axs[1].legend(H.values(), H.keys(), fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=1.2,
                  handletextpad=0.2)


def limits(X, q=(2, 98), pad=0.25):
    lo, hi = np.percentile(X, q[0], 0), np.percentile(X, q[1], 0); r = hi - lo
    return lo[0] - pad * r[0], hi[0] + pad * r[0], lo[1] - pad * r[1], hi[1] + pad * r[1]


def outcome(dist, sp, q, ref):
    ev = S6.SpeciesMin(dist, q, ref, sp).evaluate()
    return pd.Series(ev["state"]).value_counts().to_dict(), ev


MK = {"Europe": "o", "Asia": "^", "North_America": "s"}


def panel(ax, X, idx, meta, sp, colors, removed, query, special, title, note, lim, var=None):
    pos = {r: i for i, r in enumerate(idx)}
    cont = meta.continent.to_numpy()
    m = np.array([removed[i] and i not in query for i in idx])
    ax.scatter(X[m, 0], X[m, 1], s=10, color="#e3e1da", lw=0, label="removed from library", zorder=1)
    mq = np.array([i in query for i in idx])
    ax.scatter(X[mq, 0], X[mq, 1], s=42, marker="*", color="#6f6f6f", ec="white", lw=0.3, label="queries", zorder=2)
    for s, c in colors.items():
        for k, mk in MK.items():
            m = np.array([sp[i] == s and not removed[i] and i not in query and cont[i] == k for i in idx])
            if m.any():
                ax.scatter(X[m, 0], X[m, 1], s=16, marker=mk, color=c, alpha=0.9, ec="white", lw=0.3, zorder=3,
                           label=f"{s.replace('Bombus ', 'B. ')} ({k.replace('_', ' ')})")
    for i, lab in special:
        ax.scatter(*X[pos[i]], s=160, facecolor="none", ec="#d6455d", lw=1.6, zorder=5)
        ax.annotate(lab, X[pos[i]], xytext=(30, -75), textcoords="offset points", fontsize=6.5, color="#d6455d",
                    arrowprops=dict(arrowstyle="-", color="#d6455d", lw=0.8))
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    ax.set_title(title, loc="left", fontsize=8)
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7, va="bottom", bbox=dict(fc="white", ec="#cccccc", lw=0.5))
    ax.set_xticks([]); ax.set_yticks([])
    if var is not None:
        ax.set_xlabel(f"PCo1 ({100*var[0]:.0f}%)", fontsize=7); ax.set_ylabel(f"PCo2 ({100*var[1]:.0f}%)", fontsize=7)


def main():
    D, meta, dist, valid = K.load()
    lib = K.build_all(meta, dist, valid, K.PARAMS["strict_identity_min_overlap"])
    sp = meta.species.to_numpy(); blk = meta.block.to_numpy(); cont = meta.continent.to_numpy(); rid = meta.record_id.to_numpy()
    rng = np.random.default_rng(K.SEED)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.8), gridspec_kw={"hspace": 0.32, "wspace": 0.08})
    # ---- example A: competitor removal (B. lapidarius, EU_Central; MN652870.1)
    qA = np.where((sp == "Bombus lapidarius") & (blk == "EU_Central"))[0]
    refA_lobo = dict((l, r) for l, r, t in lib["L"]["lobo_hapshared"])["EU_Central"]
    refA_loco = dict((l, r) for l, r, t in lib["L"]["loco_hapshared"])["Europe"]
    mn = int(np.where(rid == "MN652870.1")[0][0])
    ter = np.where(sp == "Bombus terrestris")[0]
    dmin = np.nanmin(dist[np.ix_(np.where(sp == "Bombus lapidarius")[0], ter)], axis=0)
    far_ter = float(np.nanmin(np.where(ter == mn, np.inf, dmin)))   # nearest OTHER B. terrestris record
    ter = np.array([mn])
    idxA = np.r_[np.where(sp == "Bombus lapidarius")[0], ter]
    XA, vA = pcoa(dist[np.ix_(idxA, idxA)].astype(float))
    stA_lobo, _ = outcome(dist, sp, qA, refA_lobo); stA_loco, _ = outcome(dist, sp, qA, refA_loco)
    colA = {"Bombus lapidarius": "#2a78d6", "Bombus terrestris": "#eda100"}
    for ax, ref, st, t in ((axes[0, 0], refA_lobo, stA_lobo, "a  B. lapidarius queries (Central Europe)\n    block removed (LOBO)"),
                           (axes[0, 1], refA_loco, stA_loco, "b  same queries\n    continent removed (LOCO)")):
        inref = np.zeros(len(sp), bool); inref[ref] = True
        panel(ax, XA, idxA, meta, sp, colA, ~inref, set(qA), [(mn, "MN652870.1\nlabelled B. terrestris\n(Spain), identical to\nB. lapidarius")],
              t, f"queries: U {st.get('U', 0)}, A {st.get('A', 0)}, W {st.get('W', 0)} (n = {len(qA)})", limits(XA, (1, 99), 0.35), vA)
    row_legend(axes[0])
    # ---- example B: conspecific loss (B. monticola, EU_North -> B. johanseni)
    qB = np.where((sp == "Bombus monticola") & (blk == "EU_North"))[0]
    refB_lobo = dict((l, r) for l, r, t in lib["L"]["lobo_hapshared"])["EU_North"]
    refB_loco = dict((l, r) for l, r, t in lib["L"]["loco_hapshared"])["Europe"]
    idxB = np.where(np.isin(sp, ["Bombus monticola", "Bombus johanseni", "Bombus lapponicus"]))[0]
    XB, vB = pcoa(dist[np.ix_(idxB, idxB)].astype(float))
    stB_lobo, _ = outcome(dist, sp, qB, refB_lobo); stB_loco, _ = outcome(dist, sp, qB, refB_loco)
    colB = {"Bombus monticola": "#2a78d6", "Bombus johanseni": "#d6455d", "Bombus lapponicus": "#1baf7a"}
    for ax, ref, st, t in ((axes[1, 0], refB_lobo, stB_lobo, "c  B. monticola queries (Northern Europe)\n    block removed (LOBO)"),
                           (axes[1, 1], refB_loco, stB_loco, "d  same queries\n    continent removed (LOCO)")):
        inref = np.zeros(len(sp), bool); inref[ref] = True
        panel(ax, XB, idxB, meta, sp, colB, ~inref, set(qB), [], t,
              f"queries: U {st.get('U', 0)}, A {st.get('A', 0)}, W {st.get('W', 0)} (n = {len(qB)})", limits(XB), vB)
    row_legend(axes[1])
    for a in (axes[0, 1], axes[1, 1]):
        a.set_ylabel("")
    cap = (f"PCoA of core COI p-distances (axes 1–2 explain {100*vA.sum():.0f}% (a, b) and {100*vB.sum():.0f}% (c, d) "
             "of positive variation; view zoomed to exclude a few distant records). hapshared library. In (a, b) only B. lapidarius and the single "
             f"B. terrestris-labelled record MN652870.1 are shown; every other B. terrestris record is ≥ {100*far_ter:.1f}% from any B. lapidarius. Competitor removal (a→b): a single identical, differently labelled "
             "record ties with the queries until\nthe continent is removed. Conspecific loss (c→d): without European "
             "B. monticola all 61 queries are misidentified, 52 of them as the North American B. johanseni.")
    import textwrap
    fig.text(0.01, -0.01, "\n".join(textwrap.wrap(cap.replace("\n", " "), 150)), fontsize=6.4, color=INK2, va="top")
    for e in ("png", "pdf"):
        fig.savefig(F / f"Fig5_mechanism_pcoa.{e}", dpi=300 if e == "png" else None)
    plt.close(fig)
    print("Fig5", stA_lobo, stA_loco, stB_lobo, stB_loco)
    # ---- Fig S3: t-SNE overview of all analysed records
    U = np.where(meta.in_v3.to_numpy())[0]
    Dm = dist[np.ix_(U, U)].astype(np.float32)
    Dm = np.where(np.isfinite(Dm), Dm, np.nanmax(Dm)); np.fill_diagonal(Dm, 0)
    Y = TSNE(n_components=2, metric="precomputed", init="random", perplexity=40, random_state=K.SEED).fit_transform(Dm)
    ccol = {"Europe": "#2a78d6", "Asia": "#eb6834", "North_America": "#1baf7a", "South_America": "#8a5cd6",
            "Africa": "#9a7b4f", "Oceania": "#e87ba4"}
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    cu = cont[U]
    other = ~np.isin(cu, list(ccol))
    ax.scatter(Y[other, 0], Y[other, 1], s=2, color="#d6d4cc", lw=0, label="block unresolved", rasterized=True)
    for k, c in ccol.items():
        m = cu == k
        if m.any():
            ax.scatter(Y[m, 0], Y[m, 1], s=2, color=c, lw=0, alpha=0.8, label=k.replace("_", " "), rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=6.5, markerscale=4, loc="lower left")
    ax.set_title(f"t-SNE of core COI p-distances, {len(U):,} Bombus records (overview only; t-SNE distorts distances)",
                 loc="left", fontsize=8)
    for e in ("png", "pdf"):
        fig.savefig(F / f"FigS3_tsne_overview.{e}", dpi=300 if e == "png" else None)
    plt.close(fig)
    print("FigS3 done")


if __name__ == "__main__":
    main()
