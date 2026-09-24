"""Tests for hierarchy specification and decomposition."""

import numpy as np
import pandas as pd
import pytest

from setjoin.hierarchy import (
    HierarchySpec,
    compute_group_score_matrix,
    decompose_by_size,
)
from setjoin.matchers import structure_aware_match


class TestHierarchySpec:
    def test_from_dataframe(self) -> None:
        source = pd.DataFrame({"group": [0, 0, 1, 1], "value": [1, 2, 3, 4]})
        target = pd.DataFrame({"group": [0, 0, 1, 1], "value": [5, 6, 7, 8]})

        hierarchy = HierarchySpec.from_dataframe(
            source, target, source_group_col="group", target_group_col="group"
        )

        assert hierarchy.n_source_groups == 2
        assert hierarchy.n_target_groups == 2
        assert set(hierarchy.source_group_ids) == {0, 1}
        assert set(hierarchy.target_group_ids) == {0, 1}

    def test_dataframe_uses_positions_and_preserves_string_ids(self) -> None:
        source = pd.DataFrame({"group": ["alpha", "alpha", "beta"]}, index=[10, 11, 12])
        target = pd.DataFrame({"group": ["x", "x", "y"]}, index=[20, 21, 22])
        hierarchy = HierarchySpec.from_dataframe(source, target, "group", "group")

        assert hierarchy.source_groups == {"alpha": [0, 1], "beta": [2]}
        assert hierarchy.target_groups == {"x": [0, 1], "y": [2]}
        result = structure_aware_match(np.eye(3), hierarchy)
        assert result.matches == [(0, 0), (1, 1), (2, 2)]

    def test_groupby_uses_positions(self) -> None:
        source = pd.DataFrame({"group": ["a", "a"]}, index=[10, 11])
        target = pd.DataFrame({"group": ["b", "b"]}, index=[20, 21])
        hierarchy = HierarchySpec.from_groupby(
            source.groupby("group"), target.groupby("group")
        )
        assert hierarchy.source_groups == {"a": [0, 1]}
        assert hierarchy.target_groups == {"b": [0, 1]}

    def test_missing_or_duplicate_group_positions_raise(self) -> None:
        scores = np.eye(2)
        for source_groups in ({"a": [0]}, {"a": [0, 0]}):
            hierarchy = HierarchySpec(
                source_groups=source_groups,
                target_groups={"x": [0, 1]},
            )
            with pytest.raises(ValueError, match="partition"):
                structure_aware_match(scores, hierarchy)

    def test_group_sizes(self) -> None:
        hierarchy = HierarchySpec(
            source_groups={0: [0, 1], 1: [2, 3, 4]},
            target_groups={0: [0, 1, 2], 1: [3]},
        )

        src_sizes = hierarchy.source_group_sizes()
        tgt_sizes = hierarchy.target_group_sizes()

        assert src_sizes == {0: 2, 1: 3}
        assert tgt_sizes == {0: 3, 1: 1}

    def test_get_group(self) -> None:
        hierarchy = HierarchySpec(
            source_groups={0: [0, 1], 1: [2, 3]},
            target_groups={0: [0, 1], 1: [2, 3]},
        )

        src_group = hierarchy.get_source_group(0)
        assert src_group.group_id == 0
        assert list(src_group.indices) == [0, 1]


class TestComputeGroupScoreMatrix:
    def test_optimal_within_group_assignment(self) -> None:
        scores = np.array(
            [
                [10.0, 5.0, 1.0, 1.0],
                [5.0, 10.0, 1.0, 1.0],
                [1.0, 1.0, 10.0, 5.0],
                [1.0, 1.0, 5.0, 10.0],
            ]
        )
        hierarchy = HierarchySpec(
            source_groups={0: [0, 1], 1: [2, 3]},
            target_groups={0: [0, 1], 1: [2, 3]},
        )

        group_scores, within_matches = compute_group_score_matrix(hierarchy, scores)

        assert group_scores.shape == (2, 2)
        assert group_scores[0, 0] == 20.0
        assert group_scores[1, 1] == 20.0

        assert (0, 0) in within_matches
        assert (1, 1) in within_matches

    def test_variable_group_sizes(self) -> None:
        scores = np.array(
            [
                [10.0, 1.0],
                [9.0, 1.0],
                [8.0, 1.0],
            ]
        )
        hierarchy = HierarchySpec(
            source_groups={0: [0, 1, 2]},
            target_groups={0: [0], 1: [1]},
        )

        group_scores, _within_matches = compute_group_score_matrix(hierarchy, scores)

        assert group_scores.shape == (1, 2)
        assert group_scores[0, 0] == 10.0


class TestDecomposeBySize:
    def test_groups_by_size(self) -> None:
        hierarchy = HierarchySpec(
            source_groups={0: [0, 1], 1: [2, 3], 2: [4, 5, 6]},
            target_groups={0: [0, 1], 1: [2, 3, 4]},
        )

        decomposition = decompose_by_size(hierarchy)

        assert (2, 2) in decomposition
        assert decomposition[(2, 2)] == ([0, 1], [0])
        assert (3, 3) in decomposition
