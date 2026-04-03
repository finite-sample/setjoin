#!/usr/bin/env python3
"""
Extended simulation experiments for the setjoin package.

These simulations demonstrate package capabilities and can expand the paper's
empirical section with additional experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from setjoin import (
    HierarchySpec,
    MatchReport,
    compare_methods,
    structure_aware_match,
)


def simulate_variable_sizes(
    n_groups: int = 50,
    size_distribution: list[int] | None = None,
    ambiguity: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generate population with variable group sizes.

    Args:
        n_groups: Number of groups
        size_distribution: List of possible group sizes (sampled uniformly)
        ambiguity: Ambiguity parameter
        seed: Random seed
    """
    if size_distribution is None:
        size_distribution = [1, 2, 2, 3, 4]

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    person_id = 0

    for g in range(n_groups):
        group_size = rng.choice(size_distribution)
        cluster = g // 2
        z = int(rng.binomial(1, 0.5))

        cluster_base_age = 35 + 6 * rng.normal() + 2 * (cluster % 4)
        within_cluster_age_sd = max(0.3, 1.2 / ambiguity)
        base_age = cluster_base_age + rng.normal(0, within_cluster_age_sd)

        cluster_name_range = max(5, int(40 / ambiguity))
        cluster_name = int(rng.integers(0, cluster_name_range))
        surname_code = int(cluster_name + rng.integers(-1, 2))

        first_name_mod = max(12, int(50 / ambiguity))

        for role in range(group_size):
            first_name_code = int((surname_code * 3 + role * 5 + rng.integers(-1, 2)) % first_name_mod)
            age_true = float(base_age + (role - group_size / 2) * 3 + rng.normal(0, 1.0))
            y_true = float(20 + 6 * z + 0.5 * age_true + rng.normal(0, 2.0))
            rows.append(
                {
                    "latent_person_id": person_id,
                    "latent_group_id": g,
                    "cluster": cluster,
                    "role": role,
                    "z_true": z,
                    "age_true": age_true,
                    "surname_code_true": surname_code,
                    "first_name_code_true": first_name_code,
                    "y_true": y_true,
                }
            )
            person_id += 1

    return pd.DataFrame(rows)


def make_noisy_files(
    pop: pd.DataFrame, ambiguity: float = 1.0, seed: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create noisy versions for linkage."""
    rng = np.random.default_rng(seed)
    A = pop.copy()
    B = pop.copy()

    A["surname_obs"] = A["surname_code_true"] + rng.integers(-1, 2, len(A))
    A["fname_obs"] = A["first_name_code_true"] + rng.integers(-2, 3, len(A))
    A["age_obs"] = A["age_true"] + rng.normal(0, 1.0, len(A))
    A["a_group_id"] = A["latent_group_id"]

    noisy_member = ((B["cluster"] % 2 == 0) & (B["role"] == 0)).to_numpy()
    B["surname_obs"] = B["surname_code_true"] + rng.integers(-int(1 + ambiguity), int(2 + ambiguity), len(B))
    extra = rng.integers(-int(2 * ambiguity), int(2 * ambiguity) + 1, len(B))
    B["fname_obs"] = B["first_name_code_true"] + rng.integers(-3, 4, len(B)) + noisy_member * extra
    B["age_obs"] = B["age_true"] + rng.normal(0, 1.3 + 0.6 * ambiguity, len(B)) + noisy_member * rng.normal(
        0, 0.8 * ambiguity, len(B)
    )
    B["b_group_id"] = B["latent_group_id"]

    perm = rng.permutation(len(B))
    B = B.iloc[perm].reset_index(drop=True)
    return A.reset_index(drop=True), B.reset_index(drop=True)


def pair_score_matrix(A: pd.DataFrame, B: pd.DataFrame) -> np.ndarray:
    """Compute score matrix."""
    a_s = A["surname_obs"].to_numpy()[:, None]
    b_s = B["surname_obs"].to_numpy()[None, :]
    a_f = A["fname_obs"].to_numpy()[:, None]
    b_f = B["fname_obs"].to_numpy()[None, :]
    a_a = A["age_obs"].to_numpy()[:, None]
    b_a = B["age_obs"].to_numpy()[None, :]
    return -1.5 * np.abs(a_s - b_s) - 1.2 * np.abs(a_f - b_f) - 0.35 * np.abs(a_a - b_a)


def evaluate(
    A: pd.DataFrame, B: pd.DataFrame, matches: list[tuple[int, int]]
) -> dict[str, float]:
    """Evaluate match quality."""
    correct_person = 0
    correct_group = 0

    for i, j in matches:
        if A.iloc[i]["latent_person_id"] == B.iloc[j]["latent_person_id"]:
            correct_person += 1
        if A.iloc[i]["latent_group_id"] == B.iloc[j]["latent_group_id"]:
            correct_group += 1

    return {
        "person_accuracy": correct_person / len(matches) if matches else 0,
        "group_accuracy": correct_group / len(matches) if matches else 0,
    }


def run_variable_size_experiment(
    n_groups: int = 50,
    n_runs: int = 30,
    ambiguity: float = 1.5,
) -> pd.DataFrame:
    """Run experiment with variable group sizes."""
    results = []

    for run in range(n_runs):
        pop = simulate_variable_sizes(
            n_groups=n_groups,
            size_distribution=[1, 2, 2, 3, 4],
            ambiguity=ambiguity,
            seed=run,
        )
        A, B = make_noisy_files(pop, ambiguity=ambiguity, seed=run + 10000)
        scores = pair_score_matrix(A, B)
        hierarchy = HierarchySpec.from_dataframe(A, B, "a_group_id", "b_group_id")

        method_results = compare_methods(scores, hierarchy=hierarchy)

        for method, result in method_results.items():
            ev = evaluate(A, B, result.matches)
            results.append(
                {
                    "run": run,
                    "method": method,
                    "person_accuracy": ev["person_accuracy"],
                    "group_accuracy": ev["group_accuracy"],
                    "total_score": result.total_score,
                }
            )

    return pd.DataFrame(results)


def run_weight_sensitivity(
    n_groups: int = 50,
    n_runs: int = 20,
    ambiguity: float = 1.5,
) -> pd.DataFrame:
    """Sweep scoring weights to measure robustness."""
    weight_configs = [
        {"surname": 1.0, "fname": 1.0, "age": 0.5},
        {"surname": 1.5, "fname": 1.2, "age": 0.35},
        {"surname": 2.0, "fname": 1.0, "age": 0.2},
        {"surname": 1.0, "fname": 1.5, "age": 0.3},
        {"surname": 1.2, "fname": 1.2, "age": 0.6},
    ]

    results = []

    for run in range(n_runs):
        pop = simulate_variable_sizes(
            n_groups=n_groups,
            size_distribution=[2],
            ambiguity=ambiguity,
            seed=run,
        )
        A, B = make_noisy_files(pop, ambiguity=ambiguity, seed=run + 10000)

        for config_idx, weights in enumerate(weight_configs):
            a_s = A["surname_obs"].to_numpy()[:, None]
            b_s = B["surname_obs"].to_numpy()[None, :]
            a_f = A["fname_obs"].to_numpy()[:, None]
            b_f = B["fname_obs"].to_numpy()[None, :]
            a_a = A["age_obs"].to_numpy()[:, None]
            b_a = B["age_obs"].to_numpy()[None, :]

            scores = (
                -weights["surname"] * np.abs(a_s - b_s)
                - weights["fname"] * np.abs(a_f - b_f)
                - weights["age"] * np.abs(a_a - b_a)
            )

            hierarchy = HierarchySpec.from_dataframe(A, B, "a_group_id", "b_group_id")
            result = structure_aware_match(scores, hierarchy)
            ev = evaluate(A, B, result.matches)

            results.append(
                {
                    "run": run,
                    "config_idx": config_idx,
                    "surname_weight": weights["surname"],
                    "fname_weight": weights["fname"],
                    "age_weight": weights["age"],
                    "person_accuracy": ev["person_accuracy"],
                    "group_accuracy": ev["group_accuracy"],
                }
            )

    return pd.DataFrame(results)


def run_scale_experiment(
    group_counts: list[int] | None = None,
    n_runs: int = 10,
    ambiguity: float = 1.5,
) -> pd.DataFrame:
    """Test performance at different scales."""
    if group_counts is None:
        group_counts = [20, 50, 100, 200]

    results = []

    for n_groups in group_counts:
        for run in range(n_runs):
            pop = simulate_variable_sizes(
                n_groups=n_groups,
                size_distribution=[2],
                ambiguity=ambiguity,
                seed=run + n_groups * 1000,
            )
            A, B = make_noisy_files(pop, ambiguity=ambiguity, seed=run + n_groups * 1000 + 10000)
            scores = pair_score_matrix(A, B)
            hierarchy = HierarchySpec.from_dataframe(A, B, "a_group_id", "b_group_id")

            result = structure_aware_match(scores, hierarchy)
            ev = evaluate(A, B, result.matches)

            results.append(
                {
                    "run": run,
                    "n_groups": n_groups,
                    "n_records": len(A),
                    "person_accuracy": ev["person_accuracy"],
                    "group_accuracy": ev["group_accuracy"],
                }
            )

    return pd.DataFrame(results)


def run_fine_ambiguity_sweep(
    n_groups: int = 50,
    n_runs: int = 30,
    ambiguity_values: list[float] | None = None,
) -> pd.DataFrame:
    """Fine-grained ambiguity sweep."""
    if ambiguity_values is None:
        ambiguity_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

    results = []

    for ambiguity in ambiguity_values:
        for run in range(n_runs):
            seed = int(ambiguity * 100) * 1000 + run
            pop = simulate_variable_sizes(
                n_groups=n_groups,
                size_distribution=[2],
                ambiguity=ambiguity,
                seed=seed,
            )
            A, B = make_noisy_files(pop, ambiguity=ambiguity, seed=seed + 10000)
            scores = pair_score_matrix(A, B)
            hierarchy = HierarchySpec.from_dataframe(A, B, "a_group_id", "b_group_id")

            method_results = compare_methods(scores, hierarchy=hierarchy)

            for method, result in method_results.items():
                ev = evaluate(A, B, result.matches)
                results.append(
                    {
                        "run": run,
                        "ambiguity": ambiguity,
                        "method": method,
                        "person_accuracy": ev["person_accuracy"],
                        "group_accuracy": ev["group_accuracy"],
                    }
                )

    return pd.DataFrame(results)


def generate_diagnostic_showcase(output_dir: Path) -> None:
    """Generate all diagnostic artifacts for paper appendix."""
    pop = simulate_variable_sizes(n_groups=30, size_distribution=[2, 3], ambiguity=1.5, seed=42)
    A, B = make_noisy_files(pop, ambiguity=1.5, seed=1042)
    scores = pair_score_matrix(A, B)
    hierarchy = HierarchySpec.from_dataframe(A, B, "a_group_id", "b_group_id")

    result = structure_aware_match(scores, hierarchy)

    ground_truth = []
    for idx, row in A.iterrows():
        pid = row["latent_person_id"]
        for b_idx, b_row in B.iterrows():
            if b_row["latent_person_id"] == pid:
                ground_truth.append((int(idx), int(b_idx)))  # type: ignore[arg-type]
                break

    report = MatchReport(
        result=result,
        scores=scores,
        ground_truth=ground_truth,
        source_groups={int(k): [int(x) for x in v] for k, v in hierarchy.source_groups.items()},
        target_groups={int(k): [int(x) for x in v] for k, v in hierarchy.target_groups.items()},
    )

    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    report.to_csv(diag_dir)

    summary = report.summary()
    pd.DataFrame([summary]).to_csv(diag_dir / "summary.csv", index=False)

    print(f"Diagnostic artifacts written to {diag_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/extended"))
    parser.add_argument(
        "--experiment",
        choices=["variable_sizes", "weight_sensitivity", "scale", "fine_sweep", "diagnostics", "all"],
        default="all",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment in ("variable_sizes", "all"):
        print("Running variable size experiment...")
        df = run_variable_size_experiment()
        df.to_csv(args.output_dir / "variable_sizes.csv", index=False)
        print(df.groupby("method")[["person_accuracy", "group_accuracy"]].mean())

    if args.experiment in ("weight_sensitivity", "all"):
        print("\nRunning weight sensitivity experiment...")
        df = run_weight_sensitivity()
        df.to_csv(args.output_dir / "weight_sensitivity.csv", index=False)
        print(df.groupby("config_idx")[["person_accuracy", "group_accuracy"]].mean())

    if args.experiment in ("scale", "all"):
        print("\nRunning scale experiment...")
        df = run_scale_experiment()
        df.to_csv(args.output_dir / "scale.csv", index=False)
        print(df.groupby("n_groups")[["person_accuracy", "group_accuracy"]].mean())

    if args.experiment in ("fine_sweep", "all"):
        print("\nRunning fine ambiguity sweep...")
        df = run_fine_ambiguity_sweep()
        df.to_csv(args.output_dir / "fine_ambiguity_sweep.csv", index=False)
        print(df.groupby(["ambiguity", "method"])["person_accuracy"].mean().unstack())

    if args.experiment in ("diagnostics", "all"):
        print("\nGenerating diagnostic showcase...")
        generate_diagnostic_showcase(args.output_dir)

    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
