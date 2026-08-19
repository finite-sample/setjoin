"""Visualization functions for match diagnostics (requires matplotlib)."""

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

from setjoin.diagnostics import MatchReport
from setjoin.types import MatchResult, ScoreMatrix


def _get_plt() -> Any:
    """Import matplotlib, raising helpful error if not installed."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install setjoin[viz]"
        ) from None


def _figure_and_axes(ax: "Axes | None", **kwargs: Any) -> tuple[Any, Any]:
    """Return the (figure, axes) pair to draw on, creating one if ax is None."""
    if ax is None:
        return _get_plt().subplots(**kwargs)
    return ax.figure, ax


def plot_score_heatmap(
    scores: ScoreMatrix,
    matches: list[tuple[int, int]] | None = None,
    ax: "Axes | None" = None,
    cmap: str = "viridis",
    mark_matches: bool = True,
) -> "Figure":
    """Plot score matrix as a heatmap with optional match overlay.

    Args:
        scores: Score matrix (n_source x n_target)
        matches: Optional list of matches to highlight
        ax: Optional matplotlib Axes to plot on
        cmap: Colormap name
        mark_matches: Whether to mark matched pairs

    Returns:
        Figure: Matplotlib figure with heatmap
    """
    fig, axes = _figure_and_axes(ax, figsize=(10, 8))

    im = axes.imshow(scores, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=axes, label="Score")

    if matches and mark_matches:
        src_idx, tgt_idx = zip(*matches, strict=True)
        axes.scatter(tgt_idx, src_idx, c="red", marker="x", s=50, linewidths=2)

    axes.set_xlabel("Target index")
    axes.set_ylabel("Source index")
    axes.set_title("Score Matrix")

    return fig


def plot_match_comparison(
    report1: MatchReport,
    report2: MatchReport,
    label1: str = "Method 1",
    label2: str = "Method 2",
    ax: "Axes | None" = None,
) -> "Figure":
    """Compare scores of matches between two methods.

    Args:
        report1: First method's report
        report2: Second method's report
        label1: Label for first method
        label2: Label for second method
        ax: Optional matplotlib Axes

    Returns:
        Figure: Matplotlib figure with comparison
    """
    fig, axes = _figure_and_axes(ax, figsize=(8, 6))

    conf1 = report1.match_confidence()
    conf2 = report2.match_confidence()

    axes.hist(conf1["score"], bins=30, alpha=0.5, label=label1)
    axes.hist(conf2["score"], bins=30, alpha=0.5, label=label2)

    mean1 = conf1["score"].mean()
    mean2 = conf2["score"].mean()
    axes.axvline(mean1, color="blue", linestyle="--", label=f"{label1} mean")
    axes.axvline(mean2, color="orange", linestyle="--", label=f"{label2} mean")

    axes.set_xlabel("Match Score")
    axes.set_ylabel("Count")
    axes.set_title("Match Score Distribution Comparison")
    axes.legend()

    return fig


def plot_method_comparison_bar(
    results: dict[str, MatchResult],
    metric: str = "total_score",
    ax: "Axes | None" = None,
) -> "Figure":
    """Bar chart comparing a metric across methods.

    Args:
        results: Dictionary of method name to MatchResult
        metric: Which metric to plot ("total_score" or "n_matches")
        ax: Optional matplotlib Axes

    Returns:
        Figure: Matplotlib figure with bar chart

    Raises:
        ValueError: If metric is unknown
    """
    plt = _get_plt()
    fig, axes = _figure_and_axes(ax, figsize=(8, 5))

    methods = list(results.keys())
    if metric == "total_score":
        values = [r.total_score for r in results.values()]
        ylabel = "Total Score"
    elif metric == "n_matches":
        values = [len(r) for r in results.values()]
        ylabel = "Number of Matches"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    axes.bar(methods, values)
    axes.set_ylabel(ylabel)
    axes.set_title(f"{ylabel} by Method")
    plt.xticks(rotation=15)

    return fig


def plot_confidence_distribution(
    report: MatchReport,
    ax: "Axes | None" = None,
) -> "Figure":
    """Plot distribution of match confidence (margin to second-best).

    Args:
        report: MatchReport to analyze
        ax: Optional matplotlib Axes

    Returns:
        Figure: Matplotlib figure with distribution
    """
    fig, axes = _figure_and_axes(ax, figsize=(8, 5))

    conf = report.match_confidence()
    margins = conf["margin"].replace([np.inf, -np.inf], np.nan).dropna()

    axes.hist(margins, bins=30, edgecolor="black")
    median_val = margins.median()
    axes.axvline(
        median_val, color="red", linestyle="--", label=f"Median: {median_val:.2f}"
    )

    axes.set_xlabel("Margin (score - second best)")
    axes.set_ylabel("Count")
    axes.set_title("Match Confidence Distribution")
    axes.legend()

    return fig


def plot_accuracy_by_ambiguity(
    data: "NDArray[np.floating[Any]] | list[dict[str, float]]",
    methods: list[str],
    ambiguity_values: list[float],
    metric: str = "record_accuracy",
    ax: "Axes | None" = None,
) -> "Figure":
    """Plot accuracy metric vs ambiguity level for multiple methods.

    Args:
        data: DataFrame or list of dicts with columns [method, ambiguity, <metric>]
        methods: List of method names to plot
        ambiguity_values: List of ambiguity values
        metric: Which metric to plot
        ax: Optional matplotlib Axes

    Returns:
        Figure: Matplotlib figure with accuracy plot
    """
    _ = ambiguity_values
    fig, axes = _figure_and_axes(ax, figsize=(8, 5))

    df = pd.DataFrame(data)

    for method in methods:
        method_data = df[df["method"] == method].sort_values(by="ambiguity")
        axes.plot(
            method_data["ambiguity"],
            method_data[metric],
            marker="o",
            label=method,
        )

    axes.set_xlabel("Ambiguity Level")
    axes.set_ylabel(metric.replace("_", " ").title())
    axes.set_title(f"{metric.replace('_', ' ').title()} vs Ambiguity")
    axes.legend()

    return fig
