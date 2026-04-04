"""Configurable scoring functions for record comparison."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from setjoin.types import FieldConfig, ScoreMatrix


def exact_match(a: NDArray[np.object_], b: NDArray[np.object_]) -> NDArray[np.float64]:
    """Return 1.0 where values match exactly, 0.0 otherwise."""
    result: NDArray[np.float64] = (a == b).astype(np.float64)
    return result


def abs_diff(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return negative absolute difference (larger = more similar)."""
    return -np.abs(a - b)


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Compute Levenshtein similarity ratio between two strings."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)

    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev

    distance = prev[len2]
    return 1.0 - distance / max(len1, len2)


def levenshtein(a: NDArray[np.object_], b: NDArray[np.object_]) -> NDArray[np.float64]:
    """Compute Levenshtein similarity ratio for each pair."""
    result = np.zeros((len(a), len(b)), dtype=np.float64)
    for i, s1 in enumerate(a):
        for j, s2 in enumerate(b):
            result[i, j] = _levenshtein_ratio(str(s1), str(s2))
    return result


def _jaro_similarity(s1: str, s2: str) -> float:
    """Compute Jaro similarity between two strings."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        matches / len1 + matches / len2 + (matches - transpositions / 2) / matches
    ) / 3
    return jaro


def _jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """Compute Jaro-Winkler similarity between two strings."""
    jaro = _jaro_similarity(s1, s2)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    return jaro + prefix_len * prefix_weight * (1 - jaro)


def jaro_winkler(a: NDArray[np.object_], b: NDArray[np.object_]) -> NDArray[np.float64]:
    """Compute Jaro-Winkler similarity for each pair."""
    result = np.zeros((len(a), len(b)), dtype=np.float64)
    for i, s1 in enumerate(a):
        for j, s2 in enumerate(b):
            result[i, j] = _jaro_winkler_similarity(str(s1), str(s2))
    return result


ComparatorType = Callable[
    [NDArray[np.object_], NDArray[np.object_]], NDArray[np.float64]
]

COMPARATORS: dict[str, ComparatorType] = {
    "exact": exact_match,
    "abs_diff": abs_diff,  # type: ignore[dict-item]
    "levenshtein": levenshtein,
    "jaro_winkler": jaro_winkler,
}


class Scorer:
    """
    Configurable scorer for computing pairwise similarity between records.

    Args:
        config: Mapping from field name to configuration.
            Each value can be a FieldConfig or a dict with keys:
            - weight: float (default 1.0)
            - comparator: str (default "exact")
            - missing_value: float (default 0.0)
    """

    def __init__(self, config: dict[str, dict[str, float | str] | FieldConfig]) -> None:
        self.fields: dict[str, FieldConfig] = {}
        for name, cfg in config.items():
            if isinstance(cfg, FieldConfig):
                self.fields[name] = cfg
            else:
                self.fields[name] = FieldConfig(
                    weight=float(cfg.get("weight", 1.0)),
                    comparator=str(cfg.get("comparator", "exact")),
                    missing_value=float(cfg.get("missing_value", 0.0)),
                )

    def score(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        source_suffix: str = "",
        target_suffix: str = "",
    ) -> ScoreMatrix:
        """
        Compute pairwise score matrix between source and target records.

        Args:
            source: DataFrame with source records
            target: DataFrame with target records
            source_suffix: Suffix to append to field names for source columns
            target_suffix: Suffix to append to field names for target columns

        Returns:
            ScoreMatrix: Score matrix of shape (len(source), len(target))

        Raises:
            ValueError: If field not found or unknown comparator
        """
        n_source, n_target = len(source), len(target)
        total_score = np.zeros((n_source, n_target), dtype=np.float64)

        for field_name, cfg in self.fields.items():
            src_col = f"{field_name}{source_suffix}"
            tgt_col = f"{field_name}{target_suffix}"

            if src_col not in source.columns or tgt_col not in target.columns:
                raise ValueError(f"Field '{field_name}' not found in DataFrames")

            a = source[src_col].to_numpy()
            b = target[tgt_col].to_numpy()

            comparator_fn = COMPARATORS.get(cfg.comparator)
            if comparator_fn is None:
                raise ValueError(f"Unknown comparator: {cfg.comparator}")

            if cfg.comparator == "abs_diff":
                a_f = a.astype(np.float64)[:, None]
                b_f = b.astype(np.float64)[None, :]
                field_score = comparator_fn(a_f, b_f)  # type: ignore[arg-type]
            else:
                a_obj = a.astype(object)
                b_obj = b.astype(object)
                if a_obj.ndim == 1:
                    field_scores = []
                    for val_a in a_obj:
                        row = []
                        for val_b in b_obj:
                            cmp_result = comparator_fn(
                                np.array([val_a]), np.array([val_b])
                            )
                            row.append(cmp_result[0, 0])
                        field_scores.append(row)
                    field_score = np.array(field_scores, dtype=np.float64)
                else:
                    field_score = comparator_fn(a_obj, b_obj)

            total_score += cfg.weight * field_score

        return total_score


def score_matrix(
    source: pd.DataFrame,
    target: pd.DataFrame,
    weights: dict[str, float],
    comparators: dict[str, str] | None = None,
) -> ScoreMatrix:
    """
    Convenience function to compute a score matrix with simple configuration.

    Args:
        source: Source DataFrame
        target: Target DataFrame
        weights: Mapping from field name to weight
        comparators: Optional mapping from field name to comparator name

    Returns:
        ScoreMatrix: Score matrix of shape (len(source), len(target))
    """
    comparators = comparators or {}
    config: dict[str, dict[str, float | str] | FieldConfig] = {
        name: {"weight": weight, "comparator": comparators.get(name, "abs_diff")}
        for name, weight in weights.items()
    }
    return Scorer(config).score(source, target)
