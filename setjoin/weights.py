"""Soft matching with probabilistic weights using optimal transport.

This module provides soft (probabilistic) assignments instead of hard 1-to-1 matching,
enabling uncertainty propagation and Lahiri-Larsen style variance estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from setjoin.hierarchy import HierarchySpec
from setjoin.types import ScoreMatrix

if TYPE_CHECKING:
    from setjoin.calibration import CalibrationSpec


@dataclass
class MatchWeights:
    """
    Soft assignment matrix from entropy-regularized optimal transport.

    Instead of hard 1-to-1 assignments, this holds a weight matrix where
    entry (i, j) represents the probability/weight that source i matches target j.
    """

    matrix: NDArray[np.float64]
    """Weight matrix (n_source x n_target) with row/col sums <= 1."""

    regularization: float
    """Entropy regularization parameter used."""

    iterations_used: int
    """Number of Sinkhorn iterations to converge."""

    converged: bool
    """Whether algorithm converged within max_iterations."""

    @property
    def n_source(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_target(self) -> int:
        return int(self.matrix.shape[1])

    def row_sums(self) -> NDArray[np.float64]:
        """Sum of weights for each source record (should be <= 1)."""
        result: NDArray[np.float64] = self.matrix.sum(axis=1)
        return result

    def col_sums(self) -> NDArray[np.float64]:
        """Sum of weights for each target record (should be <= 1)."""
        result: NDArray[np.float64] = self.matrix.sum(axis=0)
        return result

    def nonzero_for(
        self, source_idx: int, threshold: float = 1e-6
    ) -> list[tuple[int, float]]:
        """
        Get non-zero target indices and weights for a source record.

        Args:
            source_idx: Index of source record
            threshold: Minimum weight to include

        Returns:
            list[tuple[int, float]]: (target_idx, weight) tuples
        """
        row = self.matrix[source_idx]
        mask = row > threshold
        indices = np.where(mask)[0]
        return [(int(idx), float(row[idx])) for idx in indices]

    def to_hard(self, threshold: float = 0.5) -> list[tuple[int, int]]:
        """
        Convert soft weights to hard 1-to-1 assignment by thresholding.

        Args:
            threshold: Minimum weight to count as a match

        Returns:
            list[tuple[int, int]]: (source_idx, target_idx) pairs
        """
        matches = []
        used_targets: set[int] = set()

        for i in range(self.n_source):
            best_j = int(np.argmax(self.matrix[i]))
            if self.matrix[i, best_j] >= threshold and best_j not in used_targets:
                matches.append((i, best_j))
                used_targets.add(best_j)

        return matches

    def to_hard_hungarian(self) -> list[tuple[int, int]]:
        """
        Convert soft weights to hard 1-to-1 assignment using Hungarian algorithm.

        This uses the soft weights as scores for optimal assignment.

        Returns:
            list[tuple[int, int]]: (source_idx, target_idx) pairs
        """
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-self.matrix)
        return list(zip(rows.tolist(), cols.tolist(), strict=True))

    def expected_score(self, scores: ScoreMatrix) -> float:
        """
        Compute expected total score under soft assignment.

        Args:
            scores: Original pairwise score matrix

        Returns:
            float: Sum of weight * score over all pairs
        """
        return float((self.matrix * scores).sum())


def soft_match(
    scores: ScoreMatrix,
    regularization: float = 0.1,
    max_iterations: int = 1000,
    convergence_threshold: float = 1e-8,
    hierarchy: HierarchySpec | None = None,
) -> MatchWeights:
    """
    Compute soft matching weights using Sinkhorn algorithm.

    Uses entropy-regularized optimal transport to produce a doubly-stochastic
    weight matrix. Higher regularization produces softer (more uniform) weights;
    lower regularization approaches the hard Hungarian solution.

    Args:
        scores: Pairwise score matrix (n_source x n_target)
        regularization: Entropy regularization parameter (epsilon).
            Higher values = softer assignments.
            As regularization -> 0, approaches Hungarian solution.
            As regularization -> inf, approaches uniform weights.
        max_iterations: Maximum Sinkhorn iterations
        convergence_threshold: Stop when marginal error below this
        hierarchy: Optional hierarchy for structure-aware soft matching.
            When provided, applies Sinkhorn at group level first.

    Returns:
        MatchWeights: Soft assignment matrix
    """
    if hierarchy is not None:
        return _soft_match_hierarchical(
            scores, regularization, max_iterations, convergence_threshold, hierarchy
        )

    return _sinkhorn(scores, regularization, max_iterations, convergence_threshold)


def _logsumexp(a: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Numerically stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    result: NDArray[np.float64] = np.squeeze(a_max, axis=axis) + np.log(
        np.sum(np.exp(a - a_max), axis=axis)
    )
    return result


def _sinkhorn(
    scores: ScoreMatrix,
    regularization: float,
    max_iterations: int,
    convergence_threshold: float,
) -> MatchWeights:
    """
    Log-domain Sinkhorn algorithm for entropy-regularized optimal transport.

    Solves: max sum_{ij} w_ij * s_ij - reg * sum_{ij} w_ij * log(w_ij)
    subject to: row sums = 1, col sums = 1, w_ij >= 0

    Uses log-domain operations for numerical stability.
    """
    n, m = scores.shape

    if regularization <= 0:
        raise ValueError("regularization must be positive")

    C = scores / regularization

    f = np.zeros(n, dtype=np.float64)
    g = np.zeros(m, dtype=np.float64)

    converged = False
    iterations = 0

    for iteration in range(max_iterations):
        iterations = iteration + 1

        f_prev = f.copy()

        g = -_logsumexp(C + f[:, np.newaxis], axis=0)
        f = -_logsumexp(C + g[np.newaxis, :], axis=1)

        if np.max(np.abs(f - f_prev)) < convergence_threshold:
            converged = True
            break

    log_transport = C + f[:, np.newaxis] + g[np.newaxis, :]
    transport = np.exp(log_transport)

    transport = np.clip(transport, 0, None)
    transport_sum = transport.sum()
    if transport_sum > 1e-10:
        transport = transport / transport_sum * min(n, m)

    return MatchWeights(
        matrix=transport,
        regularization=regularization,
        iterations_used=iterations,
        converged=converged,
    )


def _soft_match_hierarchical(
    scores: ScoreMatrix,
    regularization: float,
    max_iterations: int,
    convergence_threshold: float,
    hierarchy: HierarchySpec,
) -> MatchWeights:
    """
    Structure-aware soft matching.

    Applies Sinkhorn within each group pair, then aggregates.
    """
    n, m = scores.shape
    transport = np.zeros((n, m), dtype=np.float64)

    source_ids = hierarchy.source_group_ids
    target_ids = hierarchy.target_group_ids

    group_weights = np.zeros((len(source_ids), len(target_ids)), dtype=np.float64)
    for si, src_id in enumerate(source_ids):
        src_indices = list(hierarchy.source_groups[src_id])
        for ti, tgt_id in enumerate(target_ids):
            tgt_indices = list(hierarchy.target_groups[tgt_id])
            sub_matrix = scores[np.ix_(src_indices, tgt_indices)]
            group_weights[si, ti] = sub_matrix.sum()

    group_result = _sinkhorn(
        group_weights, regularization, max_iterations, convergence_threshold
    )

    total_iterations = group_result.iterations_used
    all_converged = group_result.converged

    for si, src_id in enumerate(source_ids):
        src_indices = list(hierarchy.source_groups[src_id])
        for ti, tgt_id in enumerate(target_ids):
            tgt_indices = list(hierarchy.target_groups[tgt_id])

            group_weight = group_result.matrix[si, ti]
            if group_weight < 1e-10:
                continue

            sub_matrix = scores[np.ix_(src_indices, tgt_indices)]
            within_result = _sinkhorn(
                sub_matrix, regularization, max_iterations, convergence_threshold
            )

            total_iterations = max(total_iterations, within_result.iterations_used)
            all_converged = all_converged and within_result.converged

            for i_local, i_global in enumerate(src_indices):
                for j_local, j_global in enumerate(tgt_indices):
                    transport[i_global, j_global] = (
                        group_weight * within_result.matrix[i_local, j_local]
                    )

    return MatchWeights(
        matrix=transport,
        regularization=regularization,
        iterations_used=total_iterations,
        converged=all_converged,
    )


def soft_match_with_calibration(
    scores: ScoreMatrix,
    source_df: pd.DataFrame,
    calibration: CalibrationSpec,
    regularization: float = 0.1,
    calibration_weight: float = 1.0,
    max_iterations: int = 1000,
    convergence_threshold: float = 1e-8,
    hierarchy: HierarchySpec | None = None,
) -> MatchWeights:
    """
    Soft matching with calibration penalty incorporated into objective.

    This modifies the score matrix to penalize selections that deviate
    from target calibration proportions.

    Args:
        scores: Pairwise score matrix (n_source x n_target)
        source_df: DataFrame with calibration variables
        calibration: Target marginal distributions
        regularization: Entropy regularization parameter
        calibration_weight: Weight for calibration penalty vs matching score
        max_iterations: Maximum Sinkhorn iterations
        convergence_threshold: Convergence threshold
        hierarchy: Optional hierarchy for structure-aware matching

    Returns:
        MatchWeights: Calibration-aware soft assignment
    """
    n_source = scores.shape[0]
    calibration_bonus = np.zeros(n_source, dtype=np.float64)

    for var, target_dist in calibration.margins.items():
        if var not in source_df.columns:
            continue

        current_dist = source_df[var].value_counts(normalize=True)

        for category, target_prop in target_dist.items():
            val = current_dist.get(category)
            actual_prop = float(val) if val is not None else 0.0
            ratio = target_prop / max(actual_prop, 1e-10)

            mask_arr = np.asarray(source_df[var] == category)
            calibration_bonus[mask_arr] += np.log(max(ratio, 1e-10))

    adjusted_scores = scores + calibration_weight * calibration_bonus[:, np.newaxis]

    return soft_match(
        adjusted_scores,
        regularization=regularization,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        hierarchy=hierarchy,
    )
