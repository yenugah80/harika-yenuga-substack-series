"""
Benford's Law — Visualization
The Unnamed Pipeline, Episode 1

Generates bar chart comparison of observed vs expected
digit frequencies.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path


def plot_benfords(
    results: pd.DataFrame,
    title: str = "Benford's Law — First-Digit Analysis",
    save_path: str = None,
    figsize: tuple = (12, 6),
) -> None:
    """
    Plot observed vs expected digit frequencies.

    Parameters
    ----------
    results : pd.DataFrame
        Output from benfords_test().
    title : str
        Chart title.
    save_path : str, optional
        File path to save the figure. If None, displays interactively.
    figsize : tuple
        Figure dimensions in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    x = np.arange(1, 10)
    width = 0.35

    bars_observed = ax.bar(
        x - width / 2,
        results["observed_pct"],
        width,
        label="Observed",
        color="#e8963e",
        edgecolor="#0d1117",
        linewidth=0.5,
    )
    bars_expected = ax.bar(
        x + width / 2,
        results["expected_pct"],
        width,
        label="Expected (Benford)",
        color="#1a2332",
        edgecolor="#3a424b",
        linewidth=0.5,
    )

    ax.set_xlabel("Leading digit", color="#8b949e", fontsize=12)
    ax.set_ylabel("Frequency (%)", color="#8b949e", fontsize=12)
    ax.set_title(title, color="#f0f6fc", fontsize=16, pad=20)
    ax.set_xticks(range(1, 10))
    ax.tick_params(colors="#484f58")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    for spine in ax.spines.values():
        spine.set_color("#1a2332")

    ax.legend(
        facecolor="#0d1117",
        edgecolor="#1a2332",
        labelcolor="#8b949e",
        fontsize=10,
    )

    # Add percentage labels on observed bars
    for bar, pct in zip(bars_observed, results["observed_pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            color="#e8963e",
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, facecolor="#0d1117", bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_deviation(
    results: pd.DataFrame,
    title: str = "Deviation from Benford's Expected Distribution",
    save_path: str = None,
    figsize: tuple = (12, 5),
) -> None:
    """
    Plot deviation of observed from expected frequencies.
    Positive bars = overrepresented. Negative = underrepresented.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    colors = [
        "#e05252" if d > 3.0 else "#e8963e" if d > 0 else "#3a7bd5"
        for d in results["deviation_pct"]
    ]

    ax.bar(
        range(1, 10),
        results["deviation_pct"],
        color=colors,
        edgecolor="#0d1117",
        linewidth=0.5,
    )

    ax.axhline(y=0, color="#3a424b", linewidth=0.5)
    ax.axhline(y=3.0, color="#e05252", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(y=-3.0, color="#e05252", linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_xlabel("Leading digit", color="#8b949e", fontsize=12)
    ax.set_ylabel("Deviation (pp)", color="#8b949e", fontsize=12)
    ax.set_title(title, color="#f0f6fc", fontsize=14, pad=15)
    ax.set_xticks(range(1, 10))
    ax.tick_params(colors="#484f58")

    for spine in ax.spines.values():
        spine.set_color("#1a2332")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, facecolor="#0d1117", bbox_inches="tight")
        print(f"Deviation chart saved to {save_path}")
    else:
        plt.show()

    plt.close()
