"""
Survival Analysis — Visualization
The Unnamed Pipeline, Episode 2

Kaplan-Meier curves, group comparisons, and Cox hazard ratio plots.
Dark theme matching Episode 01 style.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter

BG = "#0d1117"
AMBER = "#e8963e"
RED = "#e05252"
BLUE = "#3a7bd5"
GREEN = "#2ecc71"
MUTED = "#8b949e"
TITLE = "#f0f6fc"
SPINE = "#1a2332"
PALETTE = [AMBER, BLUE, RED, GREEN]


def _dark_axes(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_color(SPINE)


def _save_or_show(fig, save_path):
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=BG, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_survival_curve(
    kmf: KaplanMeierFitter,
    title: str = "Kaplan-Meier Survival Curve",
    save_path: str = None,
    figsize: tuple = (12, 6),
) -> None:
    """
    Plot a single KM survival curve with 95% confidence interval band
    and a dashed median survival line.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _dark_axes(ax, fig)

    sf = kmf.survival_function_
    ci = kmf.confidence_interval_
    t = sf.index
    s = sf.iloc[:, 0].values
    lo = ci.iloc[:, 0].values
    hi = ci.iloc[:, 1].values

    ax.step(t, s, where="post", color=AMBER, linewidth=2.5, label=kmf._label)
    ax.fill_between(t, lo, hi, step="post",
                    color=AMBER, alpha=0.12, label="95% CI")

    med = kmf.median_survival_time_
    if not np.isinf(med):
        ax.axvline(x=med, color=RED, linewidth=1.2,
                   linestyle="--", alpha=0.8)
        ax.axhline(y=0.5, color=RED, linewidth=1.2,
                   linestyle="--", alpha=0.8)
        ax.text(med + (t.max() * 0.01), 0.52,
                f"median = {med:.0f} days",
                color=RED, fontsize=10, fontfamily="monospace")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Days", fontsize=12)
    ax.set_ylabel("Survival probability S(t)", fontsize=12)
    ax.set_title(title, color=TITLE, fontsize=15, pad=15)
    ax.legend(facecolor=BG, edgecolor=SPINE,
              labelcolor=MUTED, fontsize=10)

    _save_or_show(fig, save_path)


def plot_group_comparison(
    fitters: dict,
    title: str = "Survival by Group",
    p_value: float = None,
    save_path: str = None,
    figsize: tuple = (12, 6),
) -> None:
    """
    Overlay multiple KM survival curves on one plot.
    Annotates p-value from log-rank test if provided.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _dark_axes(ax, fig)

    for i, (group, kmf) in enumerate(fitters.items()):
        color = PALETTE[i % len(PALETTE)]
        sf = kmf.survival_function_
        t = sf.index
        s = sf.iloc[:, 0].values
        ax.step(t, s, where="post", color=color,
                linewidth=2.2, label=str(group))

    if p_value is not None:
        sig = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.4f}"
        ax.text(0.97, 0.97, f"Log-rank test: {sig}",
                transform=ax.transAxes,
                ha="right", va="top",
                color=MUTED, fontsize=10,
                fontfamily="monospace")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Days", fontsize=12)
    ax.set_ylabel("Survival probability S(t)", fontsize=12)
    ax.set_title(title, color=TITLE, fontsize=15, pad=15)
    ax.legend(facecolor=BG, edgecolor=SPINE,
              labelcolor=MUTED, fontsize=10)

    _save_or_show(fig, save_path)


def plot_hazard_ratios(
    cph: CoxPHFitter,
    title: str = "Cox Model — Hazard Ratios",
    save_path: str = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    Horizontal bar chart of hazard ratios with 95% CI error bars.
    Red = increases risk (HR > 1), green = protective (HR < 1).
    Reference line at HR = 1.0.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _dark_axes(ax, fig)

    hr = cph.hazard_ratios_
    ci = cph.confidence_intervals_
    features = list(hr.index)
    hrs = [hr[f] for f in features]
    lo_err = [hr[f] - np.exp(ci.loc[f, "95% lower-bound"]) for f in features]
    hi_err = [np.exp(ci.loc[f, "95% upper-bound"]) - hr[f] for f in features]
    colors = [RED if h > 1 else GREEN for h in hrs]

    y = range(len(features))
    ax.barh(list(y), hrs, xerr=[lo_err, hi_err],
            color=colors, edgecolor=BG, linewidth=0.5,
            error_kw={"ecolor": MUTED, "capsize": 4, "linewidth": 1.2},
            height=0.55)

    ax.axvline(x=1.0, color=MUTED, linewidth=1,
               linestyle="--", alpha=0.6)

    ax.set_yticks(list(y))
    ax.set_yticklabels(features, color=MUTED, fontsize=11)
    ax.set_xlabel("Hazard ratio (HR)", fontsize=12)
    ax.set_title(title, color=TITLE, fontsize=14, pad=15)

    risk_patch = mpatches.Patch(color=RED, label="Increases risk (HR > 1)")
    prot_patch = mpatches.Patch(color=GREEN, label="Protective (HR < 1)")
    ax.legend(handles=[risk_patch, prot_patch],
              facecolor=BG, edgecolor=SPINE,
              labelcolor=MUTED, fontsize=10)

    _save_or_show(fig, save_path)
