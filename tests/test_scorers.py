"""Tests for scoring functions."""

import numpy as np
import pandas as pd
import pytest

from setjoin.scorers import (
    Scorer,
    _jaro_winkler_similarity,
    _levenshtein_ratio,
    abs_diff,
    exact_match,
    jaro_winkler,
    levenshtein,
    score_matrix,
)


class TestComparators:
    def test_exact_match_identical(self) -> None:
        a = np.array(["foo", "bar", "baz"])
        b = np.array(["foo", "bar", "baz"])
        result = exact_match(a[:, None], b[None, :])
        expected = np.eye(3)
        np.testing.assert_array_equal(result, expected)

    def test_exact_match_different(self) -> None:
        a = np.array(["a", "b"])
        b = np.array(["c", "d"])
        result = exact_match(a[:, None], b[None, :])
        expected = np.zeros((2, 2))
        np.testing.assert_array_equal(result, expected)

    def test_abs_diff(self) -> None:
        a = np.array([1.0, 2.0, 3.0])[:, None]
        b = np.array([1.0, 3.0])[None, :]
        result = abs_diff(a, b)
        expected = np.array([[0.0, -2.0], [-1.0, -1.0], [-2.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_levenshtein_identical(self) -> None:
        assert _levenshtein_ratio("hello", "hello") == 1.0

    def test_levenshtein_different(self) -> None:
        assert 0 < _levenshtein_ratio("hello", "hallo") < 1.0

    def test_levenshtein_empty(self) -> None:
        assert _levenshtein_ratio("", "") == 1.0
        assert _levenshtein_ratio("hello", "") == 0.0

    def test_jaro_winkler_identical(self) -> None:
        assert _jaro_winkler_similarity("hello", "hello") == 1.0

    def test_jaro_winkler_similar_prefix(self) -> None:
        sim1 = _jaro_winkler_similarity("hello", "hallo")
        sim2 = _jaro_winkler_similarity("hello", "xello")
        assert sim1 > sim2

    def test_levenshtein_matrix(self) -> None:
        a = np.array(["foo", "bar"])
        b = np.array(["foo", "baz"])
        result = levenshtein(a, b)
        assert result[0, 0] == 1.0
        assert result[1, 1] < 1.0
        assert result.shape == (2, 2)

    def test_jaro_winkler_matrix(self) -> None:
        a = np.array(["foo", "bar"])
        b = np.array(["foo", "baz"])
        result = jaro_winkler(a, b)
        assert result[0, 0] == 1.0
        assert result[1, 1] < 1.0
        assert result.shape == (2, 2)


class TestScorer:
    def test_basic_scoring(self) -> None:
        source = pd.DataFrame({"age": [25, 30, 35]})
        target = pd.DataFrame({"age": [25, 31, 40]})

        scorer = Scorer({"age": {"weight": 1.0, "comparator": "abs_diff"}})
        scores = scorer.score(source, target)

        assert scores.shape == (3, 3)
        assert scores[0, 0] == 0.0
        assert scores[0, 1] == -6.0

    def test_multiple_fields(self) -> None:
        source = pd.DataFrame({"age": [25, 30], "score": [100, 200]})
        target = pd.DataFrame({"age": [25, 35], "score": [100, 150]})

        scorer = Scorer(
            {
                "age": {"weight": 1.0, "comparator": "abs_diff"},
                "score": {"weight": 0.5, "comparator": "abs_diff"},
            }
        )
        scores = scorer.score(source, target)

        expected_00 = -1.0 * 0 - 0.5 * 0
        expected_01 = -1.0 * 10 - 0.5 * 50
        assert scores[0, 0] == expected_00
        assert scores[0, 1] == expected_01

    def test_missing_field_raises(self) -> None:
        source = pd.DataFrame({"age": [25]})
        target = pd.DataFrame({"name": ["Alice"]})

        scorer = Scorer({"age": {"weight": 1.0, "comparator": "abs_diff"}})
        with pytest.raises(ValueError, match="not found"):
            scorer.score(source, target)

    def test_unknown_comparator_raises(self) -> None:
        source = pd.DataFrame({"age": [25]})
        target = pd.DataFrame({"age": [30]})

        scorer = Scorer({"age": {"weight": 1.0, "comparator": "unknown"}})
        with pytest.raises(ValueError, match="Unknown comparator"):
            scorer.score(source, target)


class TestScoreMatrix:
    def test_convenience_function(self) -> None:
        source = pd.DataFrame({"x": [1.0, 2.0]})
        target = pd.DataFrame({"x": [1.0, 3.0]})

        scores = score_matrix(source, target, weights={"x": 1.5})
        assert scores.shape == (2, 2)
        assert scores[0, 0] == 0.0
        assert scores[0, 1] == -3.0
