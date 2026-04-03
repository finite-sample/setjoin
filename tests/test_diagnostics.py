"""Tests for diagnostic metrics and reporting."""

import numpy as np
import pandas as pd

from setjoin.diagnostics import MatchReport, evaluate_matches
from setjoin.types import MatchResult


class TestMatchReport:
    def test_record_accuracy_perfect(self) -> None:
        scores = np.eye(3) * 10
        result = MatchResult(
            matches=[(0, 0), (1, 1), (2, 2)],
            total_score=30.0,
            method="test",
        )
        ground_truth = [(0, 0), (1, 1), (2, 2)]

        report = MatchReport(result=result, scores=scores, ground_truth=ground_truth)
        assert report.record_accuracy == 1.0

    def test_record_accuracy_partial(self) -> None:
        scores = np.eye(3) * 10
        result = MatchResult(
            matches=[(0, 0), (1, 2), (2, 1)],
            total_score=10.0,
            method="test",
        )
        ground_truth = [(0, 0), (1, 1), (2, 2)]

        report = MatchReport(result=result, scores=scores, ground_truth=ground_truth)
        assert report.record_accuracy == 1 / 3

    def test_record_accuracy_no_ground_truth(self) -> None:
        scores = np.eye(3) * 10
        result = MatchResult(
            matches=[(0, 0), (1, 1), (2, 2)],
            total_score=30.0,
            method="test",
        )
        report = MatchReport(result=result, scores=scores)
        assert report.record_accuracy is None

    def test_group_exact_match_rate(self) -> None:
        scores = np.eye(4) * 10
        result = MatchResult(
            matches=[(0, 0), (1, 1), (2, 2), (3, 3)],
            total_score=40.0,
            method="test",
            group_assignments={0: 0, 1: 1},
        )
        source_groups = {0: [0, 1], 1: [2, 3]}
        target_groups = {0: [0, 1], 1: [2, 3]}

        report = MatchReport(
            result=result,
            scores=scores,
            source_groups=source_groups,
            target_groups=target_groups,
        )
        assert report.group_exact_match_rate == 1.0

    def test_group_exact_match_rate_partial(self) -> None:
        scores = np.eye(4) * 10
        result = MatchResult(
            matches=[(0, 0), (1, 2), (2, 1), (3, 3)],
            total_score=20.0,
            method="test",
        )
        source_groups = {0: [0, 1], 1: [2, 3]}
        target_groups = {0: [0, 1], 1: [2, 3]}

        report = MatchReport(
            result=result,
            scores=scores,
            source_groups=source_groups,
            target_groups=target_groups,
        )
        assert report.group_exact_match_rate == 0.0

    def test_match_confidence(self) -> None:
        scores = np.array(
            [
                [10.0, 5.0, 1.0],
                [1.0, 10.0, 5.0],
            ]
        )
        result = MatchResult(
            matches=[(0, 0), (1, 1)],
            total_score=20.0,
            method="test",
        )

        report = MatchReport(result=result, scores=scores)
        conf = report.match_confidence()

        assert len(conf) == 2
        assert "score" in conf.columns
        assert "rank" in conf.columns
        assert "margin" in conf.columns
        assert conf.iloc[0]["score"] == 10.0
        assert conf.iloc[0]["margin"] == 5.0

    def test_method_comparison(self) -> None:
        scores = np.eye(3) * 10
        result1 = MatchResult(
            matches=[(0, 0), (1, 1), (2, 2)],
            total_score=30.0,
            method="method1",
        )
        result2 = MatchResult(
            matches=[(0, 0), (1, 2), (2, 1)],
            total_score=10.0,
            method="method2",
        )

        report1 = MatchReport(result=result1, scores=scores)
        report2 = MatchReport(result=result2, scores=scores)

        comparison = report1.method_comparison(report2)

        assert len(comparison) == 5
        both = comparison[(comparison["in_this"]) & (comparison["in_other"])]
        assert len(both) == 1

    def test_summary(self) -> None:
        scores = np.eye(3) * 10
        result = MatchResult(
            matches=[(0, 0), (1, 1), (2, 2)],
            total_score=30.0,
            method="test",
        )

        report = MatchReport(result=result, scores=scores)
        summary = report.summary()

        assert "n_matches" in summary
        assert "total_score" in summary
        assert summary["n_matches"] == 3
        assert summary["total_score"] == 30.0


class TestEvaluateMatches:
    def test_perfect_accuracy(self) -> None:
        source = pd.DataFrame(
            {
                "latent_person_id": [0, 1, 2],
                "group": [0, 0, 1],
            }
        )
        target = pd.DataFrame(
            {
                "latent_person_id": [0, 1, 2],
                "group": [0, 0, 1],
            }
        )
        matches = [(0, 0), (1, 1), (2, 2)]

        metrics = evaluate_matches(source, target, matches)
        assert metrics["record_accuracy"] == 1.0

    def test_partial_accuracy(self) -> None:
        source = pd.DataFrame(
            {
                "latent_person_id": [0, 1, 2],
            }
        )
        target = pd.DataFrame(
            {
                "latent_person_id": [0, 2, 1],
            }
        )
        matches = [(0, 0), (1, 1), (2, 2)]

        metrics = evaluate_matches(source, target, matches)
        assert metrics["record_accuracy"] == 1 / 3

    def test_group_coherence(self) -> None:
        source = pd.DataFrame(
            {
                "latent_person_id": [0, 1, 2, 3],
                "group": [0, 0, 1, 1],
            }
        )
        target = pd.DataFrame(
            {
                "latent_person_id": [0, 1, 2, 3],
                "group": [0, 0, 1, 1],
            }
        )
        matches = [(0, 0), (1, 1), (2, 2), (3, 3)]

        metrics = evaluate_matches(
            source, target, matches, source_group_col="group", target_group_col="group"
        )
        assert metrics["group_exact_match_rate"] == 1.0
