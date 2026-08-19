"""setjoin: Set-aware record linkage with structure-preserving joins.

This package provides algorithms for matching records across datasets while
preserving hierarchical structure (e.g., matching persons while keeping
household membership coherent).
"""

from importlib.metadata import version

from setjoin.calibration import (
    CalibratedMatchResult,
    CalibrationSpec,
    calibrated_match,
    compute_calibration_penalty,
    rake_weights,
)
from setjoin.diagnostics import MatchReport, evaluate_matches
from setjoin.hierarchy import HierarchySpec, compute_group_score_matrix
from setjoin.matchers import (
    compare_methods,
    greedy_match,
    hungarian_match,
    match,
    structure_aware_match,
)
from setjoin.scorers import Scorer, score_matrix
from setjoin.types import FieldConfig, GroupSpec, MatchResult, ScoreMatrix
from setjoin.weights import MatchWeights, soft_match, soft_match_with_calibration

__version__ = version("setjoin")

__all__ = [
    "CalibratedMatchResult",
    "CalibrationSpec",
    "FieldConfig",
    "GroupSpec",
    "HierarchySpec",
    "MatchReport",
    "MatchResult",
    "MatchWeights",
    "ScoreMatrix",
    "Scorer",
    "__version__",
    "calibrated_match",
    "compare_methods",
    "compute_calibration_penalty",
    "compute_group_score_matrix",
    "evaluate_matches",
    "greedy_match",
    "hungarian_match",
    "match",
    "rake_weights",
    "score_matrix",
    "soft_match",
    "soft_match_with_calibration",
    "structure_aware_match",
]
