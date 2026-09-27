"""Figures for an audit result."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

COL = {"U": "#2a78d6", "A": "#eda100", "W": "#d6455d", "N": "#9a9a9a"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def turnover(res, ax=None):
    """Flows of the same queries between U, A and W, block unseen -> continent unseen."""
    ax = ax or plt.subplots(figsize=(4.2, 4.6))[1]
    tr = res.transitions; n = tr.queries.sum(); order = [s for s in "UAWN" if s in set(tr.block_unseen) | set(tr.continent_unseen)]
    left = {s: tr[tr.block_unseen == s].queries.sum() / n for s in order}
    right = {s: tr[tr.continent_unseen == s].queries.sum() / n for s in order}
    ly, ry, y1, y2 = {}, {}, 1.0, 1.0
    for s in order:
        ly[s] = y1; ax.add_patch(Rectangle((-0.06, y1 - left[s]), 0.06, left[s], color=COL[s])); y1 -= left[s]
        ry[s] = y2; ax.add_patch(Rectangle((1.0, y2 - right[s]), 0.06, right[s], color=COL[s])); y2 -= right[s]
    lc, rc = dict(ly), dict(ry); xs = np.linspace(0, 1, 30); w = (1 - np.cos(np.pi * xs)) / 2
    for a in order:
        for b in order:
            f = tr[(tr.block_unseen == a) & (tr.continent_unseen == b)].queries.sum() / n
            if f <= 0:
                continue
            top = lc[a] + (rc[b] - lc[a]) * w; bot = top - f
            ax.add_patch(Polygon(np.r_[np.c_[xs, top], np.c_[xs[::-1], bot[::-1]]], color=COL[b], alpha=0.2 if a == b else 0.6, lw=0))
            lc[a] -= f; rc[b] -= f
    for s in order:
        if left[s] > 0.03:
            ax.text(-0.09, ly[s] - left[s] / 2, f"{s} {100 * left[s]:.0f}%", ha="right", va="center")
        if right[s] > 0.03:
            ax.text(1.09, ry[s] - right[s] / 2, f"{s} {100 * right[s]:.0f}%", ha="left", va="center")
    s = res.summary
    ax.set_title(f"{s['genus']}: {s['queries']:,} queries\nnet change {s['net_change_pp']:+.1f} pp "
                 f"[{s['net_change_ci_pp'][0]:+.1f}, {s['net_change_ci_pp'][1]:+.1f}], "
                 f"{100 * s['changed_outcome_species_weighted']:.0f}% changed outcome (species-weighted)", loc="left", fontsize=9)
    ax.text(0, 1.03, "block unseen", ha="center"); ax.text(1.03, 1.03, "continent unseen", ha="center")
    ax.set_xlim(-0.45, 1.45); ax.set_ylim(-0.02, 1.08); ax.axis("off")
    return ax


def components(res, ax=None):
    ax = ax or plt.subplots(figsize=(4.6, 2.2))[1]
    c = res.components.iloc[::-1].reset_index(drop=True)
    for i, r in c.iterrows():
        ax.hlines(i, 100 * r.ci_low, 100 * r.ci_high, color="#888", lw=2)
        ax.plot(100 * r.estimate, i, "o", color="black" if r.component == "net change" else "#2a78d6")
    ax.axvline(0, color="#888", lw=0.8); ax.set_yticks(range(len(c)), c.component)
    ax.set_xlabel("change in expected recall, pp (species-weighted, 95% CI)")
    return ax


def influential(res, n=10, ax=None):
    ax = ax or plt.subplots(figsize=(5.5, 0.32 * n + 0.8))[1]
    d = res.influential.head(n).iloc[::-1]
    ax.barh(range(len(d)), d.resolved, color="#2a78d6")
    ax.set_yticks(range(len(d)), [f"{r.record_id}  ({r.label.split()[-1]})" for r in d.itertuples()], fontsize=8)
    ax.set_xlabel("failed identifications resolved if this record were removed")
    return ax


def rule_cost(res, ax=None):
    ax = ax or plt.subplots(figsize=(4.8, 2.2))[1]
    r = res.rule.reset_index(drop=True)
    for i, x in r.iterrows():
        ax.barh(i + 0.18, x.wrong_withheld, height=0.34, color=COL["W"])
        ax.barh(i - 0.18, x.correct_withheld, height=0.34, color=COL["U"])
        ax.text(x.correct_withheld, i - 0.18, f"  {x.correct_per_wrong:.1f} correct per wrong", va="center", fontsize=8)
    ax.set_yticks(range(len(r)), r.scenario); ax.set_xlabel("single-species answers withheld by the 1% rule")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=COL["W"]), Rectangle((0, 0), 1, 1, color=COL["U"])],
              labels=["wrong", "correct"], frameon=False, fontsize=8, loc="lower right")
    return ax
