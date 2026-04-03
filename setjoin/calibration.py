"""Calibration and raking for matched records.

This module provides tools to ensure that linked tables match known marginal
distributions (e.g., age distribution, treatment shares, geographic distribution).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from setjoin.hierarchy import HierarchySpec
from setjoin.matchers import hungarian_match, structure_aware_match
from setjoin.types import ScoreMatrix


@dataclass
class CalibrationSpec:
    """
    Specification of target marginal distributions for calibration.

    Defines the desired distribution of matched records across categories.
    For example, if you know the population should be 30% young, 40% middle-aged,
    and 30% old, you can enforce this in the matched output.
    """

    margins: dict[str, dict[object, float]]
    """Target marginal proportions for each variable.

    Maps variable name -> {category: target_proportion}.
    Proportions should sum to 1 for each variable.
    Example: {"age_bucket": {0: 0.3, 1: 0.4, 2: 0.3}}
    """

    tolerance: float = 0.01
    """Maximum allowed deviation from target proportions."""

    max_iterations: int = 100
    """Maximum iterations for raking algorithm."""

    def __post_init__(self) -> None:
        for var, dist in self.margins.items():
            total = sum(dist.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Margins for '{var}' sum to {total}, expected ~1.0"
                )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        columns: list[str],
        tolerance: float = 0.01,
        max_iterations: int = 100,
    ) -> "CalibrationSpec":
        """
        Create CalibrationSpec from actual distribution in a DataFrame.

        Args:
            df: DataFrame to extract marginal distributions from
            columns: Columns to compute marginal distributions for
            tolerance: Maximum allowed deviation from targets
            max_iterations: Maximum raking iterations

        Returns:
            CalibrationSpec with marginal distributions from df
        """
        margins: dict[str, dict[object, float]] = {}
        for col in columns:
            counts = df[col].value_counts(normalize=True)
            margins[col] = {k: float(v) for k, v in counts.items()}
        return cls(
            margins=margins,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )


@dataclass
class CalibratedMatchResult:
    """Result of calibrated matching including weights for matched records."""

    matches: list[tuple[int, int]]
    """List of (source_idx, target_idx) pairs."""

    weights: NDArray[np.float64]
    """Calibration weights for each matched pair."""

    total_score: float
    """Sum of scores for all matched pairs (unweighted)."""

    method: str
    """Name of the matching method used."""

    group_assignments: dict[int, int] | None = None
    """Mapping from source group ID to target group ID (structure-aware)."""

    calibration_achieved: dict[str, dict[object, float]] = field(default_factory=dict)
    """Actual weighted proportions achieved after calibration."""

    iterations_used: int = 0
    """Number of raking iterations used."""

    def __len__(self) -> int:
        return len(self.matches)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert matches to a DataFrame with weights."""
        df = pd.DataFrame(self.matches)
        df.columns = pd.Index(["source_idx", "target_idx"])
        df["weight"] = self.weights
        return df


def rake_weights(
    matches: list[tuple[int, int]],
    source_df: pd.DataFrame,
    calibration: CalibrationSpec,
) -> tuple[NDArray[np.float64], dict[str, dict[object, float]], int]:
    """
    Compute calibration weights for matched records using iterative raking.

    Raking (iterative proportional fitting) adjusts weights so that weighted
    marginal distributions match targets.

    Args:
        matches: List of (source_idx, target_idx) matched pairs
        source_df: DataFrame containing source records with calibration variables
        calibration: Target marginal distributions

    Returns:
        Tuple of:
        - Weights array (one per match)
        - Dictionary of achieved proportions per variable
        - Number of iterations used
    """
    if not matches:
        return np.array([], dtype=np.float64), {}, 0

    n_matches = len(matches)
    weights = np.ones(n_matches, dtype=np.float64)

    source_indices = [m[0] for m in matches]
    matched_df = source_df.iloc[source_indices].copy()

    iterations_completed = 0
    for iteration in range(calibration.max_iterations):
        iterations_completed = iteration + 1
        max_deviation = 0.0

        for var, target_dist in calibration.margins.items():
            if var not in matched_df.columns:
                raise ValueError(f"Variable '{var}' not found in source DataFrame")

            for category, target_prop in target_dist.items():
                if target_prop <= 0:
                    continue

                mask_arr = np.asarray(matched_df[var] == category)
                if not mask_arr.any():
                    continue

                current_prop = weights[mask_arr].sum() / weights.sum()
                if current_prop > 0:
                    adjustment = target_prop / current_prop
                    weights[mask_arr] *= adjustment

                    deviation = abs(current_prop - target_prop)
                    max_deviation = max(max_deviation, deviation)

        weights = weights / weights.sum() * n_matches

        if max_deviation <= calibration.tolerance:
            break

    achieved: dict[str, dict[object, float]] = {}
    total_weight = weights.sum()
    for var in calibration.margins:
        achieved[var] = {}
        for category in calibration.margins[var]:
            mask_arr = np.asarray(matched_df[var] == category)
            achieved[var][category] = float(weights[mask_arr].sum() / total_weight)

    return weights, achieved, iterations_completed


def calibrated_match(
    scores: ScoreMatrix,
    source_df: pd.DataFrame,
    calibration: CalibrationSpec,
    hierarchy: HierarchySpec | None = None,
    show_progress: bool = False,
) -> CalibratedMatchResult:
    """
    Match records and calibrate weights to hit target marginal distributions.

    This performs standard matching first, then applies iterative raking to
    produce weights that ensure the matched records represent the target
    population proportions.

    Args:
        scores: Pairwise score matrix (n_source x n_target)
        source_df: DataFrame containing source records with calibration variables
        calibration: Target marginal distributions
        hierarchy: Optional hierarchy for structure-aware matching
        show_progress: Whether to show progress bar

    Returns:
        CalibratedMatchResult with matches and calibration weights
    """
    if hierarchy is not None:
        base_result = structure_aware_match(
            scores, hierarchy, show_progress=show_progress
        )
    else:
        base_result = hungarian_match(scores)

    weights, achieved, iterations = rake_weights(
        base_result.matches, source_df, calibration
    )

    return CalibratedMatchResult(
        matches=base_result.matches,
        weights=weights,
        total_score=base_result.total_score,
        method=f"calibrated_{base_result.method}",
        group_assignments=base_result.group_assignments,
        calibration_achieved=achieved,
        iterations_used=iterations,
    )


def compute_calibration_penalty(
    source_indices: list[int],
    source_df: pd.DataFrame,
    calibration: CalibrationSpec,
) -> float:
    """
    Compute calibration penalty for a set of matched source indices.

    This measures how far the current selection deviates from target proportions.
    Lower values indicate better calibration.

    Args:
        source_indices: List of source record indices
        source_df: DataFrame containing source records
        calibration: Target marginal distributions

    Returns:
        Sum of squared deviations from target proportions
    """
    if not source_indices:
        return float("inf")

    matched_df = source_df.iloc[source_indices]
    penalty = 0.0

    for var, target_dist in calibration.margins.items():
        if var not in matched_df.columns:
            continue

        counts = matched_df[var].value_counts(normalize=True)

        for category, target_prop in target_dist.items():
            actual_prop = counts.get(category, 0.0)
            penalty += (actual_prop - target_prop) ** 2

    return penalty
