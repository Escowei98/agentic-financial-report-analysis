"""
Figures 3-7 of Kapitel 5, drawn from the aggregate outputs of
`aggregate_runs.py` and `analyze_final_eval.py`. Static PNG + SVG for Word.

    uv run python scripts/plot_chapter5_figures.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGG = PROJECT_ROOT / "data" / "results" / "final_eval" / "aggregate"
OUT = PROJECT_ROOT / "docs" / "thesis" / "figures"

SYSTEMS = ["S1", "S2", "S3", "S4"]
LABELS = {"S1": "S1 Monolith-RAG", "S2": "S2 Single-Agent-RAG",
          "S3": "S3 Single-Agent-LC", "S4": "S4 Multi-Agent-LC"}
# validated categorical slots 1, 2, 3, 7 (dataviz palette, light surface)
COLORS = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a", "S4": "#4a3aa7"}
# sequential blue ramp, steps 250 -> 700
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, INK2, GRID = "#1a1a19", "#5f5e58", "#e6e5e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _rows(name: str) -> list[dict]:
    with open(AGG / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x) -> float:
    return float(x) if x not in (None, "", "None") else math.nan


def _save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.svg")
    plt.close(fig)


# Abbildung 3 -- exact_match heatmap, stratum x system -----------------------

def fig_heatmap() -> None:
    rows = [r for r in _rows("stratum_breakdown.csv") if r["grouping"] == "fa_type"]
    strata = ["FA-1", "FA-2", "FA-3", "FA-4"]
    names = {"FA-1": "FA-1 Single-Fact", "FA-2": "FA-2 Cross-Section",
             "FA-3": "FA-3 Multi-Year", "FA-4": "FA-4 Multi-Company"}
    val = {(r["group"], r["system"]): _f(r["exact_match_mean"]) for r in rows}
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for i, st in enumerate(strata):
        for j, s in enumerate(SYSTEMS):
            v = val[(st, s)]
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94,
                                       facecolor=cmap(v), edgecolor="none"))
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.55 else INK, fontsize=9)
    ax.set_xlim(0, 4)
    ax.set_ylim(4, 0)
    ax.set_xticks([j + 0.5 for j in range(4)])
    ax.set_xticklabels(SYSTEMS)
    ax.set_yticks([i + 0.5 for i in range(4)])
    ax.set_yticklabels([names[s] for s in strata])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("exact_match je Stratum (Mittel über drei Durchläufe, n = 30 je Zelle)",
                 loc="left", fontsize=9, color=INK)
    _save(fig, "abb3_heatmap_exact_match")


def fig_heatmap_with_windows() -> None:
    """Variant of Abbildung 3 with the two FA-3 window sub-rows (Tabelle 18)."""
    fa = [r for r in _rows("stratum_breakdown.csv") if r["grouping"] == "fa_type"]
    win = [r for r in _rows("stratum_breakdown.csv") if r["grouping"] == "window_class"]
    rows = [("FA-1", "FA-1 Single-Fact", fa, 30), ("FA-2", "FA-2 Cross-Section", fa, 30),
            ("FA-3", "FA-3 Multi-Year", fa, 30),
            ("within_window", "   davon within_window", win, 15),
            ("cross_window", "   davon cross_window", win, 15),
            ("FA-4", "FA-4 Multi-Company", fa, 30)]
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for i, (key, name, src, n) in enumerate(rows):
        sub = key.endswith("_window")
        for j, s in enumerate(SYSTEMS):
            v = next(_f(r["exact_match_mean"]) for r in src
                     if r["group"] == key and r["system"] == s)
            inset = 0.14 if sub else 0.03
            ax.add_patch(plt.Rectangle((j + 0.03, i + inset), 0.94, 1 - 2 * inset,
                                       facecolor=cmap(v), edgecolor="none"))
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.55 else INK, fontsize=8 if sub else 9)
    ax.set_xlim(0, 4)
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([j + 0.5 for j in range(4)])
    ax.set_xticklabels(SYSTEMS)
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels([f"{name}  (n = {n})" for _, name, _, n in rows],
                       fontsize=8.5)
    for lab, (key, *_rest) in zip(ax.get_yticklabels(), rows):
        if key.endswith("_window"):
            lab.set_color(INK2)
            lab.set_fontsize(7.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("exact_match je Stratum (Mittel über drei Durchläufe)",
                 loc="left", fontsize=9, color=INK)
    _save(fig, "abb3_neu_heatmap_exact_match_windows")


# Abbildung 4 -- reasoning dimensions per system ------------------------------

def fig_reasoning() -> None:
    rows = {r["system"]: r for r in _rows("aggregate_summary.csv")}
    dims = [("groundedness", "Belegtheit"), ("validity", "Validität\n(deskriptiv)"),
            ("completeness", "Vollständigkeit")]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    w = 0.19
    for k, s in enumerate(SYSTEMS):
        xs = [i + (k - 1.5) * w for i in range(len(dims))]
        ys = [_f(rows[s][f"{d}_mean"]) for d, _ in dims]
        sds = [_f(rows[s][f"{d}_sd"]) for d, _ in dims]
        ax.bar(xs, ys, width=w - 0.02, color=COLORS[s], label=LABELS[s],
               yerr=sds, error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 0})
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([n for _, n in dims])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Reasoning-Qualität auf den beantwortbaren Anfragen (Mittel ± SD über drei Durchläufe)",
                 loc="left", fontsize=9, color=INK)
    _save(fig, "abb4_reasoning_dimensions")


# Abbildung 5 -- quality vs cost ---------------------------------------------

def fig_tradeoff() -> None:
    rows = {r["system"]: r for r in _rows("aggregate_summary.csv")}
    eff = {r["system"]: r for r in _rows("efficiency.csv")}
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for s in SYSTEMS:
        x = _f(eff[s]["tokens_mean"])
        y = _f(rows[s]["exact_match_mean"])
        ax.scatter([x], [y], s=70, color=COLORS[s], edgecolor="white", linewidth=1.5, zorder=3)
        below = s in ("S2", "S3", "S4")
        ax.annotate(f"{LABELS[s]}\n{y:.2f} · {x/1000:,.0f}k Tokens",
                    (x, y), xytext=(0, -30 if below else 14), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1.5e3, 4e6)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Tokens je Anfrage (Mittel, log-Skala)")
    ax.set_ylabel("exact_match (n = 120)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Antwortgenauigkeit gegen Tokenverbrauch", loc="left", fontsize=9, color=INK)
    _save(fig, "abb5_quality_vs_cost")


# Abbildung 6 -- forest plot of the seven confirmatory endpoints --------------

def fig_forest() -> None:
    rows = _rows("hypothesis_tests.csv")
    labels, effs, los, his, sig, cols = [], [], [], [], [], []
    for r in rows:
        labels.append(f"{r['hypothesis']}  {r['baseline']}→{r['predicted']}  {r['metric']}")
        effs.append(_f(r["effect"]))
        los.append(_f(r["ci_low"]))
        his.append(_f(r["ci_high"]))
        sig.append(r["significant"] == "True")
        cols.append(COLORS[r["predicted"]])
    n = len(rows)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ys = list(range(n))[::-1]
    ax.axvline(0, color=INK2, linewidth=0.8)
    for y, e, lo, hi, sg, c in zip(ys, effs, los, his, sig, cols):
        ax.plot([lo, hi], [y, y], color=c, linewidth=2, solid_capstyle="round")
        ax.scatter([e], [y], s=60 if sg else 40, color=c if sg else "white",
                   edgecolor=c, linewidth=1.5, zorder=3)
        ax.text(hi + 0.02, y, f"{e:+.3f}  [{lo:+.2f}; {hi:+.2f}]", va="center", fontsize=7, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(-0.45, 1.05)
    ax.set_xlabel("Δ (B − A) mit 95-%-Bootstrap-KI · gefüllt = signifikant nach Holm")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Konfirmatorische Endpunkte (Kap. 5.3)", loc="left", fontsize=9, color=INK)
    _save(fig, "abb6_forest_hypotheses")


# Abbildung 7 -- forest plot of all six pairs on the 120 answerable queries --

EXPL_PAIRS = ["S1 vs S2", "S2 vs S3", "S3 vs S4", "S1 vs S3", "S2 vs S4", "S1 vs S4"]
EXPL_METRICS = [("exact_match", "exact_match"), ("answer_recall", "answer_recall"),
                ("groundedness", "Belegtheit"), ("completeness", "Vollständigkeit")]


def fig_forest_exploratory() -> None:
    """Tabelle 24 as a figure: Δ with bootstrap CI per pair, one panel per metric."""
    rows = [r for r in _rows("exploratory_pairs.csv") if r["population"] == "answerable"]
    by_key = {(r["pair"], r["metric"]): r for r in rows}
    ys = list(range(len(EXPL_PAIRS)))[::-1]
    fig, axes = plt.subplots(1, len(EXPL_METRICS), figsize=(6.3, 2.7), sharey=True)
    for ax, (metric, title) in zip(axes, EXPL_METRICS):
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.axhline(2.5, color=GRID, linewidth=0.8, linestyle=(0, (3, 2)))
        for y, pair in zip(ys, EXPL_PAIRS):
            r = by_key[(pair, metric)]
            e, lo, hi = _f(r["effect"]), _f(r["ci_low"]), _f(r["ci_high"])
            sg = _f(r["p"]) < 0.05
            c = COLORS[r["predicted"]]
            ax.plot([lo, hi], [y, y], color=c, linewidth=2, solid_capstyle="round")
            ax.scatter([e], [y], s=42 if sg else 30, color=c if sg else "white",
                       edgecolor=c, linewidth=1.4, zorder=3)
            ax.text(1.0, y, f"{e:+.2f}", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=6.5, color=INK2)
        ax.set_xlim(-0.42, 1.12)
        ax.set_xticks([-0.25, 0, 0.25, 0.5, 0.75])
        ax.set_xticklabels(["", "0", "", "0,5", ""])
        ax.tick_params(axis="x", labelsize=7)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([p.replace(" vs ", " → ") for p in EXPL_PAIRS], fontsize=7.5)
    axes[0].tick_params(axis="y", length=0)
    fig.supxlabel("Δ (B − A) mit 95-%-Bootstrap-KI · gefüllt = p < 0,05 (unadjustiert) "
                  "· oberhalb der Linie: geplante Paare", fontsize=7.5, color=INK, y=-0.04)
    fig.subplots_adjust(wspace=0.12)
    _save(fig, "abb7_forest_exploratory")


if __name__ == "__main__":
    fig_heatmap()
    fig_heatmap_with_windows()
    fig_reasoning()
    fig_tradeoff()
    fig_forest()
    fig_forest_exploratory()
    print("wrote", sorted(p.name for p in OUT.glob("abb*.png")))
