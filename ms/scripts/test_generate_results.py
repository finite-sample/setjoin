#!/usr/bin/env python3
"""Unit tests for generate_results.py."""
import numpy as np
import pandas as pd
import pytest

from setjoin import greedy_match, hungarian_match, structure_aware_match, HierarchySpec

from generate_results import (
    evaluate,
    make_files,
    pair_score_matrix,
    simulate_population,
)


class TestSeedNonCollision:
    """Verify that seeds across all runs are unique."""

    def test_no_seed_collision_across_ambiguity_levels(self) -> None:
        ambiguity_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        n_runs = 200
        all_seeds: set[int] = set()
        for ambiguity in ambiguity_values:
            for r in range(n_runs):
                amb_bucket = int(ambiguity * 10)
                pop_seed = amb_bucket * 100_000 + r
                file_seed = pop_seed + 1000
                if pop_seed in all_seeds or file_seed in all_seeds:
                    pytest.fail(f"Seed collision at ambiguity={ambiguity}, r={r}")
                all_seeds.add(pop_seed)
                all_seeds.add(file_seed)


class TestInputValidation:
    """Test input validation."""

    def test_n_groups_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_groups must be positive"):
            simulate_population(n_groups=0)
        with pytest.raises(ValueError, match="n_groups must be positive"):
            simulate_population(n_groups=-5)

    def test_ambiguity_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="ambiguity must be positive"):
            simulate_population(ambiguity=0)
        with pytest.raises(ValueError, match="ambiguity must be positive"):
            simulate_population(ambiguity=-1.0)


class TestHungarianMaximizesTotalScore:
    """Verify Hungarian algorithm finds the global optimum."""

    def test_hungarian_beats_greedy_total_score(self) -> None:
        np.random.seed(42)
        score = np.random.randn(20, 20)
        hungarian_result = hungarian_match(score)
        greedy_result = greedy_match(score)
        assert hungarian_result.total_score >= greedy_result.total_score - 1e-9

    def test_hungarian_is_optimal_known_case(self) -> None:
        score = np.array([[10, 1, 1], [1, 10, 1], [1, 1, 10]], dtype=float)
        result = hungarian_match(score)
        assert result.total_score == 30.0


class TestSetAwarePreservesGroupCoherence:
    """Verify set-aware method preserves group structure."""

    def test_perfect_coherence_with_unambiguous_data(self) -> None:
        pop = simulate_population(n_groups=20, ambiguity=0.1, seed=123)
        A, B = make_files(pop, ambiguity=0.1, seed=456)
        score = pair_score_matrix(A, B)
        hierarchy = HierarchySpec.from_dataframe(A, B, "a_group_id", "b_group_id")
        result = structure_aware_match(score, hierarchy)
        metrics = evaluate(A, B, result.matches)
        assert metrics["group_exact_match_rate"] == 1.0

    def test_set_aware_better_group_coherence_than_hungarian(self) -> None:
        pop = simulate_population(n_groups=70, ambiguity=1.5, seed=999)
        A, B = make_files(pop, ambiguity=1.5, seed=1999)
        score = pair_score_matrix(A, B)

        hierarchy = HierarchySpec.from_dataframe(A, B, "a_group_id", "b_group_id")
        hungarian_result = hungarian_match(score)
        set_aware_result = structure_aware_match(score, hierarchy)

        hungarian_metrics = evaluate(A, B, hungarian_result.matches)
        set_aware_metrics = evaluate(A, B, set_aware_result.matches)

        assert (
            set_aware_metrics["group_exact_match_rate"]
            >= hungarian_metrics["group_exact_match_rate"]
        )


class TestMetrics:
    """Test metric calculations on known data."""

    def test_perfect_match_metrics(self) -> None:
        A = pd.DataFrame(
            {
                "latent_person_id": [0, 1, 2, 3],
                "latent_group_id": [0, 0, 1, 1],
                "a_group_id": [0, 0, 1, 1],
                "z_true": [0, 0, 1, 1],
                "y_true": [10.0, 12.0, 20.0, 22.0],
            }
        )
        B = pd.DataFrame(
            {
                "latent_person_id": [0, 1, 2, 3],
                "latent_group_id": [0, 0, 1, 1],
                "b_group_id": [0, 0, 1, 1],
                "z_true": [0, 0, 1, 1],
                "y_true": [10.0, 12.0, 20.0, 22.0],
            }
        )
        matches = [(0, 0), (1, 1), (2, 2), (3, 3)]
        metrics = evaluate(A, B, matches)
        assert metrics["person_accuracy"] == 1.0
        assert metrics["group_exact_match_rate"] == 1.0
        assert metrics["abs_bias_treatment_gap"] == 0.0

    def test_completely_wrong_match_metrics(self) -> None:
        A = pd.DataFrame(
            {
                "latent_person_id": [0, 1, 2, 3],
                "latent_group_id": [0, 0, 1, 1],
                "a_group_id": [0, 0, 1, 1],
                "z_true": [0, 0, 1, 1],
                "y_true": [10.0, 12.0, 20.0, 22.0],
            }
        )
        B = pd.DataFrame(
            {
                "latent_person_id": [2, 3, 0, 1],
                "latent_group_id": [1, 1, 0, 0],
                "b_group_id": [1, 1, 0, 0],
                "z_true": [1, 1, 0, 0],
                "y_true": [20.0, 22.0, 10.0, 12.0],
            }
        )
        matches = [(0, 0), (1, 1), (2, 2), (3, 3)]
        metrics = evaluate(A, B, matches)
        assert metrics["person_accuracy"] == 0.0
        assert metrics["group_exact_match_rate"] == 0.0


class TestGreedyMatch:
    """Test greedy matching algorithm."""

    def test_greedy_selects_highest_first(self) -> None:
        score = np.array([[1, 100], [2, 3]], dtype=float)
        result = greedy_match(score)
        matched_pairs = set(result.matches)
        assert (0, 1) in matched_pairs


class TestSimulatePopulation:
    """Test population simulation."""

    def test_correct_number_of_records(self) -> None:
        pop = simulate_population(n_groups=50, seed=42)
        assert len(pop) == 100

    def test_two_members_per_group(self) -> None:
        pop = simulate_population(n_groups=30, seed=42)
        counts = pop.groupby("latent_group_id").size()
        assert (counts == 2).all()


class TestMakeFiles:
    """Test file generation."""

    def test_file_b_is_shuffled(self) -> None:
        pop = simulate_population(n_groups=20, seed=42)
        A, B = make_files(pop, seed=123)
        assert not (A["latent_person_id"] == B["latent_person_id"]).all()

    def test_files_have_same_length(self) -> None:
        pop = simulate_population(n_groups=20, seed=42)
        A, B = make_files(pop, seed=123)
        assert len(A) == len(B) == len(pop)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
