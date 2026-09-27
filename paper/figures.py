"""Manuscript figures (main + supplementary) from existing results only -> figures/.
Fig 1 data & design (Bombus); Fig 2 outcome transitions LOBO->LOCO (Bombus); Fig 3 components across genera;
Fig 4 decision rules on held-out blocks/continents. Supplementary: S1 conflict-kept comparison, S2 hidden errors."""
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import common
from holdout import config as C
from holdout import geography as V
from holdout import scenarios as K

F = common.FIGURES; F.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
STATE = {"U": "#2a78d6", "A": "#eda100", "W": "#d6455d", "N": "#9a9a9a"}
ORDC = {"Hymenoptera": "#2a78d6", "Lepidoptera": "#1baf7a", "Diptera": "#eb6834", "Araneae": "#8a5cd6"}
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": False, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
                     "font.family": "DejaVu Sans"})
R = C.RESULTS


def save(fig, name):
    for e in ("png", "pdf"):
        fig.savefig(F / f"{name}.{e}", dpi=300 if e == "png" else None)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 1
def fig1():
    D, meta, dist, valid = K.load()
    m = meta[meta.in_v3.to_numpy()].copy()
    lat = pd.to_numeric(m.lat, errors="coerce"); lon = pd.to_numeric(m.lon, errors="coerce")
    ok = m.coord_valid.fillna(False).astype(bool) & lat.notna() & lon.notna()
    res = V.resolved(m.block.to_numpy())
    fig = plt.figure(figsize=(7.2, 4.9))
    ax = fig.add_axes([0.0, 0.33, 1.0, 0.67])
    ax.scatter(lon[ok], lat[ok], s=1.2, color="#c9c7c0", lw=0, rasterized=True)
    g = m[res & ok.to_numpy()].groupby("block").agg(lon=("lon", lambda s: pd.to_numeric(s).median()),
                                                     lat=("lat", lambda s: pd.to_numeric(s).median()))
    n = m[res].groupby("block").size(); nsp = m[res].groupby("block").species.nunique()
    g = g.join(n.rename("n")).join(nsp.rename("sp")); g["n"] = g.n.astype(int)
    cont = {b: V.continent_of(b) for b in g.index}
    ccol = {"Europe": "#2a78d6", "Asia": "#eb6834", "North_America": "#1baf7a", "South_America": "#8a5cd6",
            "Africa": "#9a7b4f", "Oceania": "#e87ba4"}
    for b, r in g.iterrows():
        ax.scatter(r.lon, r.lat, s=12 + r.n / 6, color=ccol[cont[b]], alpha=0.55, ec="white", lw=0.6, zorder=3)
        if r.n >= 150:
            off = {"EU_West": (-14, 6), "EU_Central": (12, 8), "EU_South": (-12, -8)}.get(b, (0, 9 + (r.n ** 0.5) / 5))
            ax.annotate(f"{int(r.n):,}", (r.lon, r.lat), xytext=off, textcoords="offset points", fontsize=6.3,
                        ha="center", va="center", color=INK, zorder=4)
    ax.set_xlim(-170, 180); ax.set_ylim(-45, 80); ax.set_aspect("equal"); ax.axis("off")
    for k, c in ccol.items():
        if k in set(cont.values()):
            ax.scatter([], [], s=30, color=c, alpha=0.6, label=k.replace("_", " "))
    ax.legend(loc="lower left", fontsize=7, ncol=3, handletextpad=0.2, columnspacing=0.8)
    ax.set_title(f"a  Bombus COI records by geographic block ({int(res.sum()):,} block-resolved records, "
                 f"{m.species.nunique()} species; circle area = records)", loc="left", fontsize=8.5)
    # design panel
    bx = fig.add_axes([0.02, 0.0, 0.96, 0.27]); bx.axis("off"); bx.set_xlim(0, 3); bx.set_ylim(0, 1)
    bx.set_title("b  Reference library available to a query (grey = removed)", loc="left", fontsize=8.5)
    specs = [("R  region represented", "random CV: other records of the query's block stay"),
             ("B  block unseen (LOBO)", "all records of the query's block removed"),
             ("C  continent unseen (LOCO)", "all records of the query's continent removed")]
    for i, (t, s) in enumerate(specs):
        x0 = i + 0.05
        for j in range(3):
            for k in range(3):
                removed = (i == 1 and j == 0 and k == 0) or (i == 2 and j == 0)
                col = "#e3e1da" if removed else "#9cc3ee"
                bx.add_patch(Rectangle((x0 + 0.07 + k * 0.17, 0.55 - j * 0.17), 0.15, 0.15, color=col, lw=0))
        bx.add_patch(Rectangle((x0 + 0.07, 0.55), 0.15, 0.15, fill=False, ec=INK, lw=1.2))
        bx.text(x0 + 0.145, 0.625, "q", ha="center", va="center", fontsize=8, weight="bold")
        bx.text(x0 + 0.62, 0.64, t.split("  ")[0], fontsize=11, weight="bold", va="center")
        bx.text(x0 + 0.62, 0.40, t.split("  ")[1], fontsize=7, va="center")
        bx.text(x0 + 0.07, 0.05, s, fontsize=6.3, color=INK2)
    bx.text(3.0, 0.92, "rows = continents, columns = blocks, q = query block", fontsize=6, color=INK2, ha="right")
    save(fig, "Fig1_data_design")


# ------------------------------------------------------------------ Fig 2
def ribbons(ax, T, x0, x1, title):
    order = ["U", "A", "W"]
    n = T.n.sum()
    left = {s: T[T.from_LOBO == s].n.sum() / n for s in order}
    right = {s: T[T.to_LOCO == s].n.sum() / n for s in order}
    ly = {}; y = 1.0
    for s in order:
        ly[s] = y; ax.add_patch(Rectangle((x0 - 0.06, y - left[s]), 0.06, left[s], color=STATE[s], lw=0)); y -= left[s] + 0.0
    ry = {}; y = 1.0
    for s in order:
        ry[s] = y; ax.add_patch(Rectangle((x1, y - right[s]), 0.06, right[s], color=STATE[s], lw=0)); y -= right[s]
    lcur, rcur = dict(ly), dict(ry)
    for a in order:
        for b in order:
            f = T[(T.from_LOBO == a) & (T.to_LOCO == b)].n.sum() / n
            if f <= 0:
                continue
            ya0, ya1 = lcur[a], lcur[a] - f; yb0, yb1 = rcur[b], rcur[b] - f
            xs = np.linspace(x0, x1, 30); w = (1 - np.cos(np.pi * (xs - x0) / (x1 - x0))) / 2
            top = ya0 + (yb0 - ya0) * w; bot = ya1 + (yb1 - ya1) * w
            ax.add_patch(Polygon(np.r_[np.c_[xs, top], np.c_[xs[::-1], bot[::-1]]], color=STATE[a if a == b else b],
                                 alpha=0.25 if a == b else 0.6, lw=0))
            lcur[a] -= f; rcur[b] -= f
    for s in order:
        if left[s] > 0.03:
            ax.text(x0 - 0.08, ly[s] - left[s] / 2, f"{s} {100*left[s]:.0f}%", ha="right", va="center", fontsize=7)
        if right[s] > 0.03:
            ax.text(x1 + 0.08, ry[s] - right[s] / 2, f"{s} {100*right[s]:.0f}%", ha="left", va="center", fontsize=7)
    ax.text(x0 - 0.03, 1.03, "LOBO", ha="center", fontsize=7.5); ax.text(x1 + 0.03, 1.03, "LOCO", ha="center", fontsize=7.5)
    ax.set_title(title, loc="left", fontsize=8)
    flows = [(a, b, T[(T.from_LOBO == a) & (T.to_LOCO == b)].n.sum() / n) for a in order for b in order if a != b]
    txt = "\n".join(f"{a}→{b}  {100*f:.1f}% of queries" for a, b, f in sorted(flows, key=lambda z: -z[2])[:3])
    ax.text((x0 + x1) / 2, -0.05, txt, ha="center", va="top", fontsize=6.8)
    ax.set_xlim(x0 - 0.45, x1 + 0.45); ax.set_ylim(-0.02, 1.08); ax.axis("off")


def fig2():
    T = pd.read_csv(R / "v8" / "main" / "transitions.tsv", sep="\t")
    ms = pd.read_csv(R / "v8" / "main" / "mechanism_summary.tsv", sep="\t").set_index(["variant", "quantity"])
    bl = pd.read_csv(R / "v8" / "blast" / "mechanism_summary.tsv", sep="\t").set_index(["variant", "quantity"])
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.05], "wspace": 0.35})
    for ax, v, lab in zip(axes[:2], ("hapshared", "strict"), ("a", "b")):
        net = ms.loc[(v, "exp_total")]
        ribbons(ax, T[T.variant == v], 0.0, 1.0,
                f"{lab}  {v} library\nnet {100*net.estimate:+.1f} pp [{100*net.species_lo:+.1f}, {100*net.species_hi:+.1f}]")
    ax = axes[2]
    rows = []
    for meth, s in (("NN p-distance", ms), ("BLAST top hit", bl)):
        for v in ("hapshared", "strict"):
            for q, lab in (("exp_phi_geo_con", "conspecific loss"), ("exp_phi_geo_het", "competitor removal"), ("exp_total", "net")):
                r = s.loc[(v, q)]; rows.append((meth, v, lab, r.estimate, r.species_lo, r.species_hi))
    y = 0; ticks = []
    for meth in ("NN p-distance", "BLAST top hit"):
        for v in ("hapshared", "strict"):
            for lab, mk, col in (("conspecific loss", "o", "#d6455d"), ("competitor removal", "s", "#2a78d6"), ("net", "D", INK)):
                r = [x for x in rows if x[0] == meth and x[1] == v and x[2] == lab][0]
                ax.hlines(y, 100 * r[4], 100 * r[5], color=col, lw=1.6, alpha=0.5)
                ax.scatter(100 * r[3], y, marker=mk, color=col, s=18, zorder=3, ec="white", lw=0.4)
                y -= 0.28
            ticks.append((y + 0.56, f"{meth.split()[0]}\n{v}")); y -= 0.35
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=6.5)
    ax.set_xlabel("change in expected recall (pp), species 95% CI", fontsize=7)
    ax.set_title("c  components\n● conspecific loss ■ competitor removal ◆ net", loc="left", fontsize=7.5)
    fig.suptitle("Bombus: the same 3,388 queries (28 species) with the query block held out (LOBO) and the query continent held out (LOCO)",
                 x=0.01, ha="left", fontsize=8.5, y=1.06)
    save(fig, "Fig2_transitions_bombus")


# ------------------------------------------------------------------ Fig 3
def fig3():
    Cm = pd.read_csv(R / "v11" / "reporting" / "compensation_magnitudes.tsv", sep="\t")
    Tr = pd.read_csv(R / "v11" / "reporting" / "transitions_UAW_all_genera.tsv", sep="\t")
    order = ["Bombus", "Andrena", "Lasioglossum", "Megachile", "Eupithecia", "Xestia", "Aedes", "Hylaeus", "Pardosa"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"width_ratios": [1.35, 1]}, sharey=True)
    ax = axes[0]
    for i, tx in enumerate(order):
        y = -i - (0.6 if i >= 7 else 0)
        for a, off, alpha in (("main", 0.13, 0.95), ("blast", -0.13, 0.5)):
            r = Cm[(Cm.taxon == tx) & (Cm.analysis == a) & (Cm.variant == "hapshared")].iloc[0]
            col = ORDC[r.order]
            ax.hlines(y + off, 100 * r.phi_con_lo, 100 * r.phi_con_hi, color="#d6455d", lw=1.4, alpha=0.4 * alpha)
            ax.scatter(100 * r.phi_con, y + off, marker="o", color="#d6455d", s=14, alpha=alpha, zorder=3)
            ax.hlines(y + off, 100 * r.phi_het_lo, 100 * r.phi_het_hi, color="#2a78d6", lw=1.4, alpha=0.4 * alpha)
            ax.scatter(100 * r.phi_het, y + off, marker="s", color="#2a78d6", s=14, alpha=alpha, zorder=3)
            ax.scatter(100 * r.net, y + off, marker="D", color=INK, s=14, alpha=alpha, zorder=4)
    ax.axvline(0, color=INK2, lw=0.7)
    ys = [-i - (0.6 if i >= 7 else 0) for i in range(len(order))]
    orders = Cm.drop_duplicates("taxon").set_index("taxon").order
    ax.set_yticks(ys, [f"{t} ({orders[t][:4]}.)" + (" *" if t in ("Hylaeus", "Pardosa") else "") for t in order], fontsize=7)
    ax.set_xlabel("Shapley component, pp (hapshared; upper = NN, lower = BLAST)", fontsize=7)
    ax.set_title("a  loss (●), removal (■), net (◆)", loc="left", fontsize=8.5)
    ax = axes[1]
    for i, tx in enumerate(order):
        y = -i - (0.6 if i >= 7 else 0)
        r = Tr[(Tr.taxon == tx) & (Tr.variant == "hapshared")].iloc[0]
        uw, au = 100 * r.share_U_to_W, 100 * r.share_A_to_U
        ax.barh(y + 0.15, uw, height=0.28, color=STATE["W"]); ax.barh(y - 0.15, au, height=0.28, color=STATE["U"])
        ax.text(uw + 0.5, y + 0.15, f"{uw:.1f}", fontsize=6, va="center"); ax.text(au + 0.5, y - 0.15, f"{au:.1f}", fontsize=6, va="center")
    ax.set_xlabel("% of queries", fontsize=7)
    ax.set_title("b  U→W (red) and A→U (blue)", loc="left", fontsize=8.5)
    fig.text(0.01, -0.03, "* extra set (pre-registered stop rule). Core set: 7 genera, 3 orders. Signs of the two components follow "
             "from the nearest-neighbour rule; magnitudes, net and transitions are the result.", fontsize=6.5, color=INK2)
    save(fig, "Fig3_components_genera")


# ------------------------------------------------------------------ Fig 4
def fig4():
    Rp = pd.read_csv(R / "v12" / "rule_performance.tsv", sep="\t")
    lab = {"P0": "nearest species", "P1": "BOLD 1% rule", "P2_a0.01": "tuned thresholds α=1%", "P2_a0.02": "tuned thresholds α=2%",
           "P3_a0.01": "logistic α=1%", "P3_a0.02": "logistic α=2%", "P4_a0.01": "GB α=1%", "P4_a0.02": "GB α=2%"}
    mk = {"P0": ("X", INK), "P1": ("o", "#2a78d6"), "P2_a0.01": ("^", "#1baf7a"), "P2_a0.02": ("v", "#1baf7a"),
          "P3_a0.01": ("s", "#eb6834"), "P3_a0.02": ("s", "#f2a47a"), "P4_a0.01": ("D", "#8a5cd6"), "P4_a0.02": ("D", "#bea2e8")}
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1, 1, 0.8], "wspace": 0.38})
    ymax = 100 * Rp[(Rp.variant == "hapshared") & (Rp.context.isin(["B", "C"]))].wrong_among_unique_query.max() + 1.0
    for ax, ctx, t in zip(axes[:2], ("B", "C"), ("a  block unseen\n(18 held-out blocks)", "b  continent unseen\n(4 held-out continents)")):
        d = Rp[(Rp.variant == "hapshared") & (Rp.context == ctx) & (Rp.delta == 0.01)]
        for r in d.itertuples():
            m, c = mk[r.rule]
            ax.scatter(100 * r.unique_coverage_query, 100 * r.wrong_among_unique_query, marker=m, color=c, s=34, ec="white", lw=0.5,
                       label=lab[r.rule], zorder=3)
        ax.axhline(1, color=INK2, lw=0.6, ls=":"); ax.axhline(2, color=INK2, lw=0.6, ls=":")
        ax.set_xlabel("% queries with a single-species answer", fontsize=7); ax.set_ylabel("% of those that are wrong", fontsize=7)
        ax.set_title(t, loc="left", fontsize=8); ax.set_xlim(0, 102); ax.set_ylim(0, ymax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.4, -0.13), handletextpad=0.2, columnspacing=1.0)
    ax = axes[2]
    d = Rp[(Rp.variant == "hapshared") & (Rp.rule == "P1")]
    for i, ctx in enumerate(("B", "C")):
        for j, dl in enumerate((0.0, 0.01, 0.02)):
            r = d[(d.context == ctx) & (d.delta == dl)].iloc[0]
            x = i * 3.6 + j
            ax.bar(x, 100 * r.list_inclusion_query, color="#9cc3ee" if ctx == "B" else "#f2a47a", width=0.8)
            ax.text(x, 100 * r.list_inclusion_query + 1, f"{r.list_size_mean:.1f}", ha="center", fontsize=6)
    ax.set_xticks([0, 1, 2, 3.6, 4.6, 5.6], ["0", "1", "2", "0", "1", "2"], fontsize=6.5)
    ax.set_xlabel("list width δ (%)   B | C", fontsize=7); ax.set_ylabel("% lists containing true species", fontsize=7)
    ax.set_ylim(50, 105); ax.set_title("c  lists, BOLD 1% rule\n(labels = mean list size)", loc="left", fontsize=8)
    fig.suptitle("Bombus, hapshared library: thresholds chosen on training data only; results on held-out regions",
                 x=0.01, ha="left", fontsize=8.5, y=1.07)
    save(fig, "Fig4_decision_rules")


# ------------------------------------------------------------------ Supplementary
def figS1():
    X = pd.read_csv(R / "v11" / "conflict_kept" / "compare_query_level.tsv", sep="\t")
    X = X[(X.analysis == "main") & (X.view == "fixed_common_set")]
    order = ["Bombus", "Andrena", "Lasioglossum", "Megachile", "Eupithecia", "Xestia", "Aedes", "Hylaeus", "Pardosa"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, v in zip(axes, ("hapshared", "strict")):
        for i, tx in enumerate(order):
            for qc, off, col in (("original_qc", 0.12, INK2), ("conflict_kept", -0.12, "#eb6834")):
                r = X[(X.taxon == tx) & (X.variant == v) & (X.qc == qc)].iloc[0]
                ax.hlines(-i + off, 100 * r.net_lo, 100 * r.net_hi, color=col, lw=1.4, alpha=0.5)
                ax.scatter(100 * r.net, -i + off, color=col, s=16, zorder=3, label=qc.replace("_", " ") if i == 0 else None)
        ax.axvline(0, color=INK2, lw=0.7); ax.set_title(f"{v}: net LOBO→LOCO on the fixed common query set", loc="left", fontsize=8)
        ax.set_xlabel("pp (species 95% CI)", fontsize=7)
    axes[0].set_yticks(-np.arange(len(order)), order, fontsize=7); axes[0].legend(fontsize=6.5)
    save(fig, "FigS1_conflict_kept")


def figS2():
    H = pd.read_csv(R / "v11" / "reporting" / "hidden_errors_explicit_denominators.tsv", sep="\t")
    H = H[H.variant == "hapshared"].set_index("context").loc[["R", "B", "C"]]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.arange(3)
    ax.bar(x, 100 * H.hidden_among_W_query, color=STATE["W"], width=0.6)
    for i, (q, n) in enumerate(zip(H.hidden_among_W_query, H.n_W)):
        ax.text(i, 100 * q + 0.8, f"{100*q:.1f}%\n(n W = {n})", ha="center", fontsize=6.5)
    ax.set_xticks(x, ["R region\nrepresented", "B block\nunseen", "C continent\nunseen"], fontsize=7)
    ax.set_ylabel("% of wrong answers that look confident", fontsize=7); ax.set_ylim(0, 45)
    ax.set_title("Bombus, hapshared: confident (single species\nwithin 1%) among wrong sets (W)", loc="left", fontsize=8)
    save(fig, "FigS2_hidden_errors")


# ------------------------------------------------------------------ main figure (v14): turnover vs net
def fig_main():
    T = pd.read_csv(R / "v8" / "main" / "transitions.tsv", sep="\t")
    Tr = pd.read_csv(R / "v11" / "reporting" / "transitions_UAW_all_genera.tsv", sep="\t")
    Cm = pd.read_csv(R / "v11" / "reporting" / "compensation_magnitudes.tsv", sep="\t")
    order = ["Bombus", "Andrena", "Lasioglossum", "Megachile", "Eupithecia", "Xestia", "Aedes", "Hylaeus", "Pardosa"]
    fig = plt.figure(figsize=(7.6, 4.4))
    ax0 = fig.add_axes([0.0, 0.08, 0.30, 0.78])
    ribbons(ax0, T[T.variant == "hapshared"], 0.0, 1.0, "a  Bombus: the same 3,388 queries\n    block held out → continent held out")
    ax1 = fig.add_axes([0.40, 0.12, 0.30, 0.74]); ax2 = fig.add_axes([0.74, 0.12, 0.24, 0.74], sharey=ax1)
    ys = [-i - (0.6 if i >= 7 else 0) for i in range(len(order))]
    for y, tx in zip(ys, order):
        r = Tr[(Tr.taxon == tx) & (Tr.variant == "hapshared")].iloc[0]
        uw, au = 100 * r.share_U_to_W, 100 * r.share_A_to_U
        other = 100 * r.changed_state_share - uw - au
        ax1.barh(y, uw, color=STATE["W"], height=0.6)
        ax1.barh(y, au, left=uw, color=STATE["U"], height=0.6)
        ax1.barh(y, other, left=uw + au, color="#bdbbb4", height=0.6)
        ax1.text(uw + au + other + 0.8, y, f"{uw + au + other:.0f}%", va="center", fontsize=6.5)
        for a, mk, col in (("main", "D", INK), ("blast", "o", "#8a8a8a")):
            c = Cm[(Cm.taxon == tx) & (Cm.analysis == a) & (Cm.variant == "hapshared")].iloc[0]
            off = 0.14 if a == "main" else -0.14
            ax2.hlines(y + off, 100 * c.net_lo, 100 * c.net_hi, color=col, lw=1.5)
            ax2.scatter(100 * c.net, y + off, marker=mk, color=col, s=16, zorder=3)
    ax1.set_yticks(ys, [t + (" *" if t in ("Hylaeus", "Pardosa") else "") for t in order], fontsize=7)
    ax1.set_xlabel("% of queries that change outcome", fontsize=7)
    ax1.set_title("b  turnover: U→W (red), A→U (blue),\n    other changes (grey)", loc="left", fontsize=8)
    ax2.axvline(0, color=INK2, lw=0.7)
    ax2.set_xlabel("net change in expected recall, pp\n(species 95% CI; ◆ NN, ● BLAST)", fontsize=7)
    ax2.set_title("c  net change", loc="left", fontsize=8)
    plt.setp(ax2.get_yticklabels(), visible=False)
    fig.text(0.40, -0.02, "LOBO → LOCO, hapshared library. * extra set. Net changes are small or uncertain (intervals often include 0), "
             "while 4–46% of queries change outcome.", fontsize=6.5, color=INK2)
    save(fig, "Fig_main_turnover")


# ------------------------------------------------------------------ Fig S7: calibration on unseen blocks / continents (v11)
def figS_calibration():
    B = pd.read_csv(R / "v11" / "calibration" / "oob_predictions.tsv.gz", sep="\t")
    Cn = pd.read_csv(R / "v11" / "calibration_continent" / "oob_predictions.tsv.gz", sep="\t")
    panels = [("a  block unseen (outer = block)", B[(B.variant == "hapshared") & (B.context == "B")]),
              ("b  continent unseen (outer = continent)", Cn[Cn.variant == "hapshared"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
    for ax, (t, d) in zip(axes, panels):
        for lab, p, col in (("logistic, fully isolated", d.p_y_logistic.to_numpy(), "#2a78d6"),
                            ("similarity = 1 − d1", 1 - d.d1.to_numpy(), "#eb6834")):
            y = d.y.to_numpy(); bins = np.array_split(np.argsort(p), 10)
            ax.plot([p[b].mean() for b in bins], [y[b].mean() for b in bins], "o-", color=col, ms=4, label=lab)
        ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--"); ax.set_title(t, loc="left", fontsize=8.5)
        ax.set_xlabel("predicted probability (deciles)", fontsize=7)
    axes[0].set_ylabel("observed share with true species listed (y)", fontsize=7); axes[0].legend(fontsize=6.5)
    fig.suptitle("Bombus, hapshared: calibration with training libraries rebuilt without the held-out region (v11)",
                 x=0.01, ha="left", fontsize=8.5, y=1.02)
    save(fig, "FigS7_calibration")


# ------------------------------------------------------------------ Fig 4 (v15): cost of the 1% rule, P1 as a filter of P0
def fig4_tradeoff():
    T = pd.read_csv(R / "v14" / "P1_as_filter_tradeoff.tsv", sep="\t")
    order = {"bombus": ("Bombus", "Hymenoptera"), "andrena": ("Andrena", "Hymenoptera"), "lasioglossum": ("Lasioglossum", "Hymenoptera"),
             "megachile": ("Megachile", "Hymenoptera"), "eupithecia": ("Eupithecia", "Lepidoptera"), "xestia": ("Xestia", "Lepidoptera"),
             "aedes": ("Aedes", "Diptera")}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.35})
    ax = axes[0]
    for r in T.itertuples():
        if r.set not in order:
            continue
        name, o = order[r.set]
        x = 100 * r.wrong_moved_to_list / r.n_queries; y = 100 * r.correct_moved_to_list / r.n_queries
        mk = "o" if r.context == "continent unseen" else "s"
        ax.scatter(max(x, 0.02), y, marker=mk, s=40 if r.set == "bombus" else 26, color=ORDC[o],
                   ec=INK if r.set == "bombus" else "white", lw=0.8 if r.set == "bombus" else 0.4, zorder=3)
        lab = (name + (" (0 removed)" if x == 0 else "")) if r.context == "continent unseen" else ""
        ax.annotate(lab, (max(x, 0.02), y), xytext=(4, 2), textcoords="offset points", fontsize=6.3)
    xs = np.logspace(-2, 1.5, 50)
    for k in (1, 10, 100):
        ax.plot(xs, k * xs, color="#bdbbb4", lw=0.7, ls="--"); ax.text(xs[-1] * 0.55 if k == 1 else 25 / k, min(k * xs[-1] * 0.55, 55) if k == 1 else 25, f"{k}:1", fontsize=6, color=INK2)
    ax.set_xscale("log"); ax.set_xlim(0.015, 40); ax.set_ylim(0, 60)
    ax.set_xlabel("wrong single answers removed per 100 queries (log)", fontsize=7)
    ax.set_ylabel("correct single answers moved to lists per 100 queries", fontsize=7)
    ax.set_title("a  cost of the 1% rule (P1 vs P0), core genera\n    ● continent unseen, ■ block unseen; dashed = correct:wrong", loc="left", fontsize=7.8)
    ax.scatter([], [], marker="o", color="#2a78d6", label="Hymenoptera"); ax.scatter([], [], marker="o", color="#1baf7a", label="Lepidoptera")
    ax.scatter([], [], marker="o", color="#eb6834", label="Diptera"); ax.legend(fontsize=6, loc="upper left")
    ax = axes[1]
    rows = [("block unseen", "bombus", "Block unseen"), ("continent unseen", "bombus", "Continent unseen"), ("cutoff 2021", "bombus_forward", "Forward in time")]
    F15 = pd.read_csv(R / "v15" / "prospective" / "tradeoff_forward.tsv", sep="\t", dtype={"cutoff": str}).set_index("cutoff")
    for i, (ctx, s, lab) in enumerate(rows):
        if s == "bombus_forward":   # v15: availability dates corrected
            f = F15.loc["2021"]
            r = pd.Series({"wrong_moved_to_list": f.wrong_moved, "correct_moved_to_list": f.correct_moved, "correct_lost_per_wrong_removed": f.ratio})
        else:
            r = T[(T.set == s) & (T.context == ctx)].iloc[0]
        ax.barh(i + 0.18, r.wrong_moved_to_list, height=0.34, color=STATE["W"]); ax.barh(i - 0.18, r.correct_moved_to_list, height=0.34, color=STATE["U"])
        ax.text(r.wrong_moved_to_list + 30, i + 0.18, f"{int(r.wrong_moved_to_list)}", va="center", fontsize=6.5)
        ax.text(r.correct_moved_to_list + 30, i - 0.18, f"{int(r.correct_moved_to_list):,}  ({r.correct_lost_per_wrong_removed:.1f}:1)", va="center", fontsize=6.5)
    ax.set_yticks(range(len(rows)), [x[2] for x in rows], fontsize=7); ax.set_xlim(0, 3300)
    ax.set_xlabel("single answers moved to candidate lists", fontsize=7)
    ax.set_title("b  Bombus: wrong (red) and correct (blue)\n    answers withheld by P1", loc="left", fontsize=7.8)
    save(fig, "Fig4_rule_tradeoff")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); figS1(); figS2()
    fig_main(); figS_calibration(); fig4_tradeoff()
    for ext in ("png", "pdf"):
        shutil.copy(F / f"Fig4_decision_rules.{ext}", F / f"FigS8_rules_and_models.{ext}")
    print("done")
