"""Tests for matching algorithms."""

import numpy as np
import pytest

from setjoin.hierarchy import HierarchySpec
from setjoin.matchers import (
    compare_methods,
    greedy_match,
    hungarian_match,
    match,
    structure_aware_match,
)


class TestGreedyMatch:
    def test_perfect_diagonal(self) -> None:
        scores = np.array(
            [
                [10.0, 1.0, 1.0],
                [1.0, 10.0, 1.0],
                [1.0, 1.0, 10.0],
            ]
        )
        result = greedy_match(scores)
        assert set(result.matches) == {(0, 0), (1, 1), (2, 2)}
        assert result.total_score == 30.0
        assert result.method == "greedy"

    def test_ambiguous_scores(self) -> None:
        scores = np.array(
            [
                [10.0, 9.0],
                [9.0, 10.0],
            ]
        )
        result = greedy_match(scores)
        assert len(result.matches) == 2
        assert result.total_score >= 18.0

    def test_rectangular_matrix(self) -> None:
        scores = np.array(
            [
                [10.0, 5.0, 1.0],
                [5.0, 10.0, 1.0],
            ]
        )
        result = greedy_match(scores)
        assert len(result.matches) == 2


class TestHungarianMatch:
    def test_perfect_diagonal(self) -> None:
        scores = np.array(
            [
                [10.0, 1.0, 1.0],
                [1.0, 10.0, 1.0],
                [1.0, 1.0, 10.0],
            ]
        )
        result = hungarian_match(scores)
        assert set(result.matches) == {(0, 0), (1, 1), (2, 2)}
        assert result.total_score == 30.0
        assert result.method == "hungarian"

    def test_optimal_assignment(self) -> None:
        scores = np.array(
            [
                [10.0, 8.0],
                [8.0, 1.0],
            ]
        )
        result = hungarian_match(scores)
        assert set(result.matches) == {(0, 1), (1, 0)}
        assert result.total_score == 16.0


class TestStructureAwareMatch:
    def test_preserves_group_structure(self) -> None:
        scores = np.array(
            [
                [10.0, 9.0, 1.0, 1.0],
                [9.0, 10.0, 1.0, 1.0],
                [1.0, 1.0, 10.0, 9.0],
                [1.0, 1.0, 9.0, 10.0],
            ]
        )
        hierarchy = HierarchySpec(
            source_groups={0: [0, 1], 1: [2, 3]},
            target_groups={0: [0, 1], 1: [2, 3]},
        )
        result = structure_aware_match(scores, hierarchy)

        match_dict = dict(result.matches)
        assert (match_dict[0] in [0, 1]) == (match_dict[1] in [0, 1])
        assert (match_dict[2] in [2, 3]) == (match_dict[3] in [2, 3])
        assert result.group_assignments is not None
        assert len(result.group_assignments) == 2

    def test_group_assignments_populated(self) -> None:
        scores = np.array(
            [
                [10.0, 1.0],
                [1.0, 10.0],
            ]
        )
        hierarchy = HierarchySpec(
            source_groups={0: [0], 1: [1]},
            target_groups={0: [0], 1: [1]},
        )
        result = structure_aware_match(scores, hierarchy)

        assert result.group_assignments is not None
        assert 0 in result.group_assignments
        assert 1 in result.group_assignments

    def test_variable_group_sizes(self) -> None:
        scores = np.array(
            [
                [10.0, 1.0, 1.0],
                [9.0, 1.0, 1.0],
                [1.0, 10.0, 9.0],
            ]
        )
        hierarchy = HierarchySpec(
            source_groups={0: [0, 1], 1: [2]},
            target_groups={0: [0], 1: [1, 2]},
        )
        result = structure_aware_match(scores, hierarchy)

        assert len(result.matches) == 2


class TestMatchFunction:
    def test_greedy_method(self) -> None:
        scores = np.eye(3) * 10
        result = match(scores, method="greedy")
        assert result.method == "greedy"

    def test_hungarian_method(self) -> None:
        scores = np.eye(3) * 10
        result = match(scores, method="hungarian")
        assert result.method == "hungarian"

    def test_structure_aware_requires_hierarchy(self) -> None:
        scores = np.eye(3) * 10
        with pytest.raises(ValueError, match="hierarchy is required"):
            match(scores, method="structure_aware")

    def test_unknown_method_raises(self) -> None:
        scores = np.eye(3) * 10
        with pytest.raises(ValueError, match="Unknown method"):
            match(scores, method="unknown")


class TestCompareMethods:
    def test_runs_multiple_methods(self) -> None:
        scores = np.eye(3) * 10
        results = compare_methods(scores)

        assert "greedy" in results
        assert "hungarian" in results
        assert all(r.method in ["greedy", "hungarian"] for r in results.values())

    def test_includes_structure_aware_with_hierarchy(self) -> None:
        scores = np.eye(3) * 10
        hierarchy = HierarchySpec(
            source_groups={0: [0], 1: [1], 2: [2]},
            target_groups={0: [0], 1: [1], 2: [2]},
        )
        results = compare_methods(scores, hierarchy=hierarchy)

        assert "structure_aware" in results
