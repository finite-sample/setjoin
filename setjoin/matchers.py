"""Matching algorithms: greedy, Hungarian, and structure-aware."""

from scipy.optimize import linear_sum_assignment

from setjoin._logging import track
from setjoin.hierarchy import HierarchySpec, compute_group_score_matrix
from setjoin.types import MatchResult, ScoreMatrix


def greedy_match(
    scores: ScoreMatrix,
    show_progress: bool = False,
) -> MatchResult:
    """
    Match records greedily by selecting highest-scoring pairs first.

    Args:
        scores: Pairwise score matrix (n_source x n_target)
        show_progress: Whether to show progress bar

    Returns:
        MatchResult: Greedy matches
    """
    n, m = scores.shape
    triples = [(i, j, scores[i, j]) for i in range(n) for j in range(m)]
    triples.sort(key=lambda x: x[2], reverse=True)

    used_source: set[int] = set()
    used_target: set[int] = set()
    matches: list[tuple[int, int]] = []
    total_score = 0.0

    iterable = track(triples, "Greedy matching", disable=not show_progress)
    for i, j, score in iterable:
        if i not in used_source and j not in used_target:
            matches.append((i, j))
            total_score += score
            used_source.add(i)
            used_target.add(j)
            if len(used_source) == min(n, m):
                break

    return MatchResult(
        matches=matches,
        total_score=total_score,
        method="greedy",
    )


def hungarian_match(
    scores: ScoreMatrix,
) -> MatchResult:
    """
    Match records using Hungarian algorithm to maximize total score.

    Args:
        scores: Pairwise score matrix (n_source x n_target)

    Returns:
        MatchResult: Optimal global assignment
    """
    rows, cols = linear_sum_assignment(-scores)
    matches = list(zip(rows.tolist(), cols.tolist(), strict=True))
    total_score = float(scores[rows, cols].sum())

    return MatchResult(
        matches=matches,
        total_score=total_score,
        method="hungarian",
    )


def structure_aware_match(
    scores: ScoreMatrix,
    hierarchy: HierarchySpec,
    show_progress: bool = False,
) -> MatchResult:
    """
    Match records while preserving group structure.

    Two-level assignment: first assign groups optimally, then assign records
    within matched groups. This ensures all members of a source group
    map to the same target group.

    Args:
        scores: Pairwise record score matrix (n_source x n_target)
        hierarchy: HierarchySpec defining group structure
        show_progress: Whether to show progress bar

    Returns:
        MatchResult: Structure-preserving matches
    """
    group_scores, within_matches = compute_group_score_matrix(hierarchy, scores)

    source_ids = hierarchy.source_group_ids
    target_ids = hierarchy.target_group_ids

    group_rows, group_cols = linear_sum_assignment(-group_scores)

    matches: list[tuple[int, int]] = []
    group_assignments: dict[int, int] = {}

    paired = list(zip(group_rows.tolist(), group_cols.tolist(), strict=True))
    if show_progress:
        paired = list(track(paired, "Assembling matches", disable=not show_progress))

    for gi, gj in paired:
        src_id = source_ids[gi]
        tgt_id = target_ids[gj]
        group_assignments[src_id] = tgt_id
        matches.extend(within_matches[(src_id, tgt_id)])

    if matches:
        idx_tuple = tuple(zip(*matches, strict=True))
        total_score = float(scores[idx_tuple].sum())
    else:
        total_score = 0.0

    return MatchResult(
        matches=matches,
        total_score=total_score,
        method="structure_aware",
        group_assignments=group_assignments,
    )


def match(
    scores: ScoreMatrix,
    method: str = "hungarian",
    hierarchy: HierarchySpec | None = None,
    show_progress: bool = False,
) -> MatchResult:
    """
    Match records using the specified method.

    Args:
        scores: Pairwise score matrix (n_source x n_target)
        method: Matching method - "greedy", "hungarian", or "structure_aware"
        hierarchy: Required for structure_aware method
        show_progress: Whether to show progress bar

    Returns:
        MatchResult: Matches using specified method

    Raises:
        ValueError: If method is unknown or hierarchy missing for structure_aware
    """
    if method == "greedy":
        return greedy_match(scores, show_progress=show_progress)
    elif method == "hungarian":
        return hungarian_match(scores)
    elif method == "structure_aware":
        if hierarchy is None:
            raise ValueError("hierarchy is required for structure_aware matching")
        return structure_aware_match(scores, hierarchy, show_progress=show_progress)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'greedy', 'hungarian', or 'structure_aware'"
        )


def compare_methods(
    scores: ScoreMatrix,
    hierarchy: HierarchySpec | None = None,
    methods: list[str] | None = None,
) -> dict[str, MatchResult]:
    """
    Run multiple matching methods and return results for comparison.

    Args:
        scores: Pairwise score matrix (n_source x n_target)
        hierarchy: Required if "structure_aware" is in methods
        methods: List of methods to run (default: all available)

    Returns:
        dict[str, MatchResult]: Method name to result mapping
    """
    if methods is None:
        methods = ["greedy", "hungarian"]
        if hierarchy is not None:
            methods.append("structure_aware")

    results: dict[str, MatchResult] = {}
    for method in methods:
        results[method] = match(scores, method=method, hierarchy=hierarchy)

    return results
