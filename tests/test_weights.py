"""Tests for soft matching and weighting functionality."""

import numpy as np
import pandas as pd
import pytest

from setjoin.calibration import CalibrationSpec
from setjoin.hierarchy import HierarchySpec
from setjoin.weights import MatchWeights, soft_match, soft_match_with_calibration


class TestMatchWeights:
    def test_properties(self) -> None:
        matrix = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
        weights = MatchWeights(
            matrix=matrix,
            regularization=0.1,
            iterations_used=50,
            converged=True,
        )

        assert weights.n_source == 3
        assert weights.n_target == 2
        assert weights.converged is True

    def test_row_col_sums(self) -> None:
        matrix = np.array([[0.5, 0.3], [0.3, 0.5]])
        weights = MatchWeights(
            matrix=matrix,
            regularization=0.1,
            iterations_used=10,
            converged=True,
        )

        row_sums = weights.row_sums()
        col_sums = weights.col_sums()

        assert len(row_sums) == 2
        assert len(col_sums) == 2

    def test_nonzero_for(self) -> None:
        matrix = np.array([[0.9, 0.1, 0.0], [0.0, 0.8, 0.2]])
        weights = MatchWeights(
            matrix=matrix,
            regularization=0.1,
            iterations_used=10,
            converged=True,
        )

        nonzero_0 = weights.nonzero_for(0, threshold=0.05)
        assert len(nonzero_0) == 2
        assert nonzero_0[0][0] == 0
        assert nonzero_0[0][1] == pytest.approx(0.9)

        nonzero_1 = weights.nonzero_for(1, threshold=0.05)
        assert len(nonzero_1) == 2

    def test_to_hard(self) -> None:
        matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        weights = MatchWeights(
            matrix=matrix,
            regularization=0.1,
            iterations_used=10,
            converged=True,
        )

        hard = weights.to_hard(threshold=0.5)
        assert len(hard) == 2
        assert (0, 0) in hard
        assert (1, 1) in hard

    def test_to_hard_hungarian(self) -> None:
        matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        weights = MatchWeights(
            matrix=matrix,
            regularization=0.1,
            iterations_used=10,
            converged=True,
        )

        hard = weights.to_hard_hungarian()
        assert len(hard) == 2
        assert set(hard) == {(0, 0), (1, 1)}

    def test_expected_score(self) -> None:
        matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        weights = MatchWeights(
            matrix=matrix,
            regularization=0.1,
            iterations_used=10,
            converged=True,
        )

        scores = np.array([[10.0, 2.0], [2.0, 10.0]])
        expected = weights.expected_score(scores)

        assert expected == pytest.approx(12.0)


class TestSoftMatch:
    def test_returns_match_weights(self) -> None:
        scores = np.array([[10.0, 1.0], [1.0, 10.0]])
        result = soft_match(scores, regularization=0.1)

        assert isinstance(result, MatchWeights)
        assert result.matrix.shape == (2, 2)

    def test_row_col_sums_bounded(self) -> None:
        scores = np.array([[10.0, 5.0, 1.0], [5.0, 10.0, 1.0], [1.0, 1.0, 10.0]])
        result = soft_match(scores, regularization=0.5)

        row_sums = result.row_sums()
        col_sums = result.col_sums()

        assert all(s <= 1.1 for s in row_sums)
        assert all(s <= 1.1 for s in col_sums)

    def test_low_regularization_approaches_hungarian(self) -> None:
        scores = np.array([[10.0, 1.0], [1.0, 10.0]])

        hard_result = soft_match(scores, regularization=0.1)

        assert hard_result.matrix[0, 0] > hard_result.matrix[0, 1]
        assert hard_result.matrix[1, 1] > hard_result.matrix[1, 0]

    def test_high_regularization_more_uniform(self) -> None:
        scores = np.array([[10.0, 1.0], [1.0, 10.0]])

        low_reg = soft_match(scores, regularization=0.5)
        high_reg = soft_match(scores, regularization=10.0)

        low_variance = np.var(low_reg.matrix)
        high_variance = np.var(high_reg.matrix)

        assert high_variance < low_variance

    def test_converges(self) -> None:
        np.random.seed(42)
        scores = np.random.rand(5, 5) * 10
        result = soft_match(scores, regularization=1.0, max_iterations=1000)

        assert result.converged

    def test_positive_regularization_required(self) -> None:
        scores = np.eye(3)
        with pytest.raises(ValueError, match="positive"):
            soft_match(scores, regularization=0.0)


class TestSoftMatchHierarchical:
    def test_with_hierarchy(self) -> None:
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

        result = soft_match(scores, regularization=0.1, hierarchy=hierarchy)

        assert result.matrix.shape == (4, 4)

        within_group_0 = result.matrix[0:2, 0:2].sum()
        cross_group = result.matrix[0:2, 2:4].sum()

        assert within_group_0 > cross_group

    def test_preserves_group_structure(self) -> None:
        scores = np.ones((4, 4)) * 5
        scores[0, 0] = 10
        scores[1, 1] = 10
        scores[2, 2] = 10
        scores[3, 3] = 10

        hierarchy = HierarchySpec(
            source_groups={0: [0, 1], 1: [2, 3]},
            target_groups={0: [0, 1], 1: [2, 3]},
        )

        result = soft_match(scores, regularization=0.1, hierarchy=hierarchy)

        block_01_01 = result.matrix[0:2, 0:2]
        block_01_23 = result.matrix[0:2, 2:4]

        assert block_01_01.sum() > block_01_23.sum() * 0.5


class TestSoftMatchWithCalibration:
    def test_basic_calibration(self) -> None:
        scores = np.ones((4, 4)) * 5
        np.fill_diagonal(scores, 10)

        source_df = pd.DataFrame(
            {"age": [0, 0, 1, 1]},
            index=range(4),
        )

        spec = CalibrationSpec(margins={"age": {0: 0.5, 1: 0.5}})

        result = soft_match_with_calibration(
            scores,
            source_df,
            spec,
            regularization=0.1,
        )

        assert result.matrix.shape == (4, 4)
        assert result.matrix.sum() > 0

    def test_calibration_influences_weights(self) -> None:
        scores = np.ones((6, 6)) * 5

        source_df = pd.DataFrame(
            {"age": [0, 0, 0, 0, 1, 1]},
            index=range(6),
        )

        spec_skewed_age0 = CalibrationSpec(margins={"age": {0: 0.9, 1: 0.1}})
        spec_skewed_age1 = CalibrationSpec(margins={"age": {0: 0.1, 1: 0.9}})

        result_age0 = soft_match_with_calibration(
            scores,
            source_df,
            spec_skewed_age0,
            regularization=0.5,
            calibration_weight=5.0,
        )
        result_age1 = soft_match_with_calibration(
            scores,
            source_df,
            spec_skewed_age1,
            regularization=0.5,
            calibration_weight=5.0,
        )

        age0_records_weight_0 = result_age0.matrix[0:4, :].sum()
        age0_records_weight_1 = result_age1.matrix[0:4, :].sum()

        assert age0_records_weight_0 > age0_records_weight_1

    def test_with_hierarchy(self) -> None:
        scores = np.array(
            [
                [10.0, 9.0, 1.0, 1.0],
                [9.0, 10.0, 1.0, 1.0],
                [1.0, 1.0, 10.0, 9.0],
                [1.0, 1.0, 9.0, 10.0],
            ]
        )
        source_df = pd.DataFrame(
            {"age": [0, 1, 0, 1]},
            index=range(4),
        )
        hierarchy = HierarchySpec(
            source_groups={0: [0, 1], 1: [2, 3]},
            target_groups={0: [0, 1], 1: [2, 3]},
        )

        spec = CalibrationSpec(margins={"age": {0: 0.5, 1: 0.5}})

        result = soft_match_with_calibration(
            scores,
            source_df,
            spec,
            regularization=0.1,
            hierarchy=hierarchy,
        )

        assert result.matrix.shape == (4, 4)
