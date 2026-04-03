"""Tests for calibration and raking functionality."""

import numpy as np
import pandas as pd
import pytest

from setjoin.calibration import (
    CalibratedMatchResult,
    CalibrationSpec,
    calibrated_match,
    compute_calibration_penalty,
    rake_weights,
)
from setjoin.hierarchy import HierarchySpec


class TestCalibrationSpec:
    def test_basic_construction(self) -> None:
        spec = CalibrationSpec(
            margins={"age": {0: 0.3, 1: 0.4, 2: 0.3}},
            tolerance=0.05,
        )
        assert spec.margins["age"][0] == 0.3
        assert spec.tolerance == 0.05

    def test_margins_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to"):
            CalibrationSpec(margins={"age": {0: 0.5, 1: 0.2}})

    def test_multiple_variables(self) -> None:
        spec = CalibrationSpec(
            margins={
                "age": {0: 0.5, 1: 0.5},
                "region": {"north": 0.4, "south": 0.6},
            }
        )
        assert len(spec.margins) == 2
        assert spec.margins["region"]["south"] == 0.6

    def test_from_dataframe(self) -> None:
        df = pd.DataFrame(
            {
                "age": [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
                "region": ["N", "N", "N", "N", "N", "S", "S", "S", "S", "S"],
            }
        )
        spec = CalibrationSpec.from_dataframe(df, columns=["age", "region"])

        assert abs(spec.margins["age"][0] - 0.3) < 0.01
        assert abs(spec.margins["age"][1] - 0.4) < 0.01
        assert abs(spec.margins["region"]["N"] - 0.5) < 0.01


class TestRakeWeights:
    def test_empty_matches(self) -> None:
        df = pd.DataFrame({"age": [0, 1, 2]})
        spec = CalibrationSpec(margins={"age": {0: 0.33, 1: 0.34, 2: 0.33}})

        weights, achieved, iterations = rake_weights([], df, spec)
        assert len(weights) == 0
        assert iterations == 0

    def test_hits_target_margins(self) -> None:
        df = pd.DataFrame(
            {"age": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1], "idx": range(10)}
        )
        matches = [(i, i) for i in range(10)]

        spec = CalibrationSpec(
            margins={"age": {0: 0.5, 1: 0.5}},
            tolerance=0.01,
            max_iterations=100,
        )

        weights, achieved, iterations = rake_weights(matches, df, spec)

        assert len(weights) == 10
        assert abs(achieved["age"][0] - 0.5) < 0.02
        assert abs(achieved["age"][1] - 0.5) < 0.02

    def test_multiple_variables(self) -> None:
        df = pd.DataFrame(
            {
                "age": [0, 0, 1, 1, 0, 0, 1, 1],
                "region": ["N", "N", "N", "N", "S", "S", "S", "S"],
            }
        )
        matches = [(i, i) for i in range(8)]

        spec = CalibrationSpec(
            margins={
                "age": {0: 0.6, 1: 0.4},
                "region": {"N": 0.5, "S": 0.5},
            },
            tolerance=0.05,
        )

        weights, achieved, _ = rake_weights(matches, df, spec)

        assert abs(achieved["age"][0] - 0.6) < 0.1
        assert abs(achieved["region"]["N"] - 0.5) < 0.1

    def test_missing_variable_raises(self) -> None:
        df = pd.DataFrame({"age": [0, 1, 2]})
        matches = [(0, 0), (1, 1)]

        spec = CalibrationSpec(margins={"missing_var": {0: 0.5, 1: 0.5}})

        with pytest.raises(ValueError, match="not found"):
            rake_weights(matches, df, spec)


class TestCalibratedMatch:
    def test_basic_calibration(self) -> None:
        scores = np.array(
            [
                [10.0, 1.0, 1.0, 1.0],
                [1.0, 10.0, 1.0, 1.0],
                [1.0, 1.0, 10.0, 1.0],
                [1.0, 1.0, 1.0, 10.0],
            ]
        )
        source_df = pd.DataFrame(
            {"age": [0, 0, 1, 1], "idx": range(4)},
            index=range(4),
        )

        spec = CalibrationSpec(
            margins={"age": {0: 0.5, 1: 0.5}},
            tolerance=0.01,
        )

        result = calibrated_match(scores, source_df, spec)

        assert isinstance(result, CalibratedMatchResult)
        assert len(result.matches) == 4
        assert len(result.weights) == 4
        assert "calibrated" in result.method
        assert result.iterations_used > 0

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
            {"group": [0, 0, 1, 1], "age": [0, 1, 0, 1]},
            index=range(4),
        )
        hierarchy = HierarchySpec(
            source_groups={0: [0, 1], 1: [2, 3]},
            target_groups={0: [0, 1], 1: [2, 3]},
        )

        spec = CalibrationSpec(
            margins={"age": {0: 0.5, 1: 0.5}},
            tolerance=0.05,
        )

        result = calibrated_match(
            scores, source_df, spec, hierarchy=hierarchy
        )

        assert result.group_assignments is not None
        assert len(result.group_assignments) == 2
        assert "structure_aware" in result.method

    def test_to_dataframe(self) -> None:
        scores = np.eye(3) * 10
        source_df = pd.DataFrame({"age": [0, 1, 2]}, index=range(3))
        spec = CalibrationSpec(margins={"age": {0: 0.33, 1: 0.34, 2: 0.33}})

        result = calibrated_match(scores, source_df, spec)
        df = result.to_dataframe()

        assert "source_idx" in df.columns
        assert "target_idx" in df.columns
        assert "weight" in df.columns
        assert len(df) == 3


class TestCalibrationPenalty:
    def test_perfect_calibration(self) -> None:
        df = pd.DataFrame({"age": [0, 0, 1, 1]}, index=range(4))
        spec = CalibrationSpec(margins={"age": {0: 0.5, 1: 0.5}})

        penalty = compute_calibration_penalty([0, 1, 2, 3], df, spec)
        assert penalty < 0.01

    def test_bad_calibration(self) -> None:
        df = pd.DataFrame({"age": [0, 0, 0, 1]}, index=range(4))
        spec = CalibrationSpec(margins={"age": {0: 0.5, 1: 0.5}})

        penalty_all = compute_calibration_penalty([0, 1, 2, 3], df, spec)
        penalty_subset = compute_calibration_penalty([0, 1, 2], df, spec)

        assert penalty_subset > penalty_all

    def test_empty_indices(self) -> None:
        df = pd.DataFrame({"age": [0, 1]})
        spec = CalibrationSpec(margins={"age": {0: 0.5, 1: 0.5}})

        penalty = compute_calibration_penalty([], df, spec)
        assert penalty == float("inf")
