#!/usr/bin/env python3
"""
Generate simulation results, tables, and figures for the FSM-style join paper.

This script reproduces all tables/figures included in the paper.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from setjoin import HierarchySpec, evaluate_matches, greedy_match, hungarian_match, structure_aware_match


def simulate_population(n_groups: int = 70, ambiguity: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic population with group structure and treatment assignment."""
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    if ambiguity <= 0:
        raise ValueError("ambiguity must be positive")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for g in range(n_groups):
        cluster = g // 2
        z = int(rng.binomial(1, 0.5))

        cluster_base_age = 35 + 6 * rng.normal() + 2 * (cluster % 4)
        within_cluster_age_sd = max(0.3, 1.2 / ambiguity if ambiguity > 0 else 1.2)
        base_age = cluster_base_age + rng.normal(0, within_cluster_age_sd)

        cluster_name_range = max(5, int(40 / ambiguity))
        cluster_name = int(rng.integers(0, cluster_name_range))
        surname_code = int(cluster_name + rng.integers(-1, 2))

        first_name_mod = max(12, int(50 / ambiguity))

        for role in [0, 1]:
            first_name_code = int((surname_code * 3 + role * 5 + rng.integers(-1, 2)) % first_name_mod)
            age_true = float(base_age + (-2 if role == 0 else 2) + rng.normal(0, 1.0))
            y_true = float(20 + 6 * z + 0.5 * age_true + rng.normal(0, 2.0))
            rows.append(
                {
                    "latent_person_id": len(rows),
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
    return pd.DataFrame(rows)


def make_files(pop: pd.DataFrame, ambiguity: float = 1.0, seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create two noisy files (A and B) from a population for linkage evaluation."""
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
    """
    Compute pairwise similarity scores between records in A and B.

    Scoring weights: surname (-1.5), first name (-1.2), age (-0.35).
    """
    a_s = A["surname_obs"].to_numpy()[:, None]
    b_s = B["surname_obs"].to_numpy()[None, :]
    a_f = A["fname_obs"].to_numpy()[:, None]
    b_f = B["fname_obs"].to_numpy()[None, :]
    a_a = A["age_obs"].to_numpy()[:, None]
    b_a = B["age_obs"].to_numpy()[None, :]
    return -1.5 * np.abs(a_s - b_s) - 1.2 * np.abs(a_f - b_f) - 0.35 * np.abs(a_a - b_a)


def evaluate(A: pd.DataFrame, B: pd.DataFrame, matches: list[tuple[int, int]]) -> dict[str, float]:
    """Compute linkage quality metrics: person accuracy, group coherence, and bias."""
    rows: list[dict[str, float]] = []
    for i, j in matches:
        a = A.iloc[i]
        b = B.iloc[j]
        rows.append(
            {
                "true_person_match": float(a["latent_person_id"] == b["latent_person_id"]),
                "true_group_match": float(a["latent_group_id"] == b["latent_group_id"]),
                "z_a_true": float(a["z_true"]),
                "z_b_linked": float(b["z_true"]),
                "y_true": float(a["y_true"]),
                "a_group_id": float(a["a_group_id"]),
            }
        )
    df = pd.DataFrame(rows)

    group_exact = df.groupby("a_group_id")["true_group_match"].mean().eq(1.0).mean()

    true_gap = A.loc[A["z_true"] == 1, "y_true"].mean() - A.loc[A["z_true"] == 0, "y_true"].mean()
    linked_gap = df.loc[df["z_b_linked"] == 1, "y_true"].mean() - df.loc[df["z_b_linked"] == 0, "y_true"].mean()

    return {
        "person_accuracy": float(df["true_person_match"].mean()),
        "group_accuracy_personlevel": float(df["true_group_match"].mean()),
        "group_exact_match_rate": float(group_exact),
        "abs_bias_treatment_gap": float(abs(linked_gap - true_gap)),
    }


def one_run(ambiguity: float, seed: int, n_groups: int = 70) -> pd.DataFrame:
    pop = simulate_population(n_groups=n_groups, ambiguity=ambiguity, seed=seed)
    A, B = make_files(pop, ambiguity=ambiguity, seed=seed + 1000)
    score = pair_score_matrix(A, B)

    hierarchy = HierarchySpec.from_dataframe(A, B, source_group_col="a_group_id", target_group_col="b_group_id")

    methods = {
        "Greedy person-level": greedy_match(score).matches,
        "Hungarian person-level": hungarian_match(score).matches,
        "Set-aware group-first": structure_aware_match(score, hierarchy).matches,
    }

    rows: list[dict[str, float]] = []
    for method, matches in methods.items():
        ev = evaluate(A, B, matches)
        ev["method"] = method  # type: ignore[assignment]
        ev["ambiguity"] = ambiguity
        ev["seed"] = seed
        rows.append(ev)
    return pd.DataFrame(rows)


def run_many(ambiguity_values: list[float], n_runs: int) -> pd.DataFrame:
    all_rows: list[pd.DataFrame] = []
    for ambiguity in ambiguity_values:
        for r in range(n_runs):
            amb_bucket = int(ambiguity * 10)
            seed = amb_bucket * 100_000 + r
            all_rows.append(one_run(ambiguity=ambiguity, seed=seed))
    return pd.concat(all_rows, ignore_index=True)


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    grouped = df.groupby(by)
    out = grouped.agg(
        person_accuracy_mean=("person_accuracy", "mean"),
        person_accuracy_se=("person_accuracy", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        group_exact_mean=("group_exact_match_rate", "mean"),
        group_exact_se=("group_exact_match_rate", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        abs_bias_mean=("abs_bias_treatment_gap", "mean"),
        abs_bias_se=("abs_bias_treatment_gap", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        group_person_mean=("group_accuracy_personlevel", "mean"),
        group_person_se=("group_accuracy_personlevel", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    ).reset_index()
    return out


def format_table(baseline_summary: pd.DataFrame, table_path: Path) -> None:
    order = ["Greedy person-level", "Hungarian person-level", "Set-aware group-first"]
    baseline_summary = baseline_summary.copy()
    baseline_summary["method"] = pd.Categorical(baseline_summary["method"], categories=order, ordered=True)
    baseline_summary = baseline_summary.sort_values("method")

    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Person accuracy & Group exact match rate & Absolute bias \\")
    lines.append(r"\midrule")
    for _, row in baseline_summary.iterrows():
        lines.append(
            f'{row["method"]} & '
            f'{row["person_accuracy_mean"]:.3f} ({row["person_accuracy_se"]:.3f}) & '
            f'{row["group_exact_mean"]:.3f} ({row["group_exact_se"]:.3f}) & '
            f'{row["abs_bias_mean"]:.3f} ({row["abs_bias_se"]:.3f}) \\\\'
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    table_path.write_text("\n".join(lines), encoding="utf-8")


def write_macros(baseline_summary: pd.DataFrame, macro_path: Path) -> None:
    row_h = baseline_summary.loc[baseline_summary["method"] == "Hungarian person-level"].iloc[0]
    row_s = baseline_summary.loc[baseline_summary["method"] == "Set-aware group-first"].iloc[0]
    row_g = baseline_summary.loc[baseline_summary["method"] == "Greedy person-level"].iloc[0]

    def cmd(name: str, value: float) -> str:
        return rf"\newcommand{{\{name}}}{{{value:.3f}}}"

    lines = [
        cmd("GreedyPersonAcc", row_g["person_accuracy_mean"]),
        cmd("HungarianPersonAcc", row_h["person_accuracy_mean"]),
        cmd("SetAwarePersonAcc", row_s["person_accuracy_mean"]),
        cmd("GreedyGroupExact", row_g["group_exact_mean"]),
        cmd("HungarianGroupExact", row_h["group_exact_mean"]),
        cmd("SetAwareGroupExact", row_s["group_exact_mean"]),
        cmd("HungarianHHExact", row_h["group_exact_mean"]),
        cmd("SetAwareHHExact", row_s["group_exact_mean"]),
        cmd("GreedyBias", row_g["abs_bias_mean"]),
        cmd("HungarianBias", row_h["abs_bias_mean"]),
        cmd("SetAwareBias", row_s["abs_bias_mean"]),
        cmd("PersonGainVsHungarian", row_s["person_accuracy_mean"] - row_h["person_accuracy_mean"]),
        cmd("GroupGainVsHungarian", row_s["group_exact_mean"] - row_h["group_exact_mean"]),
        cmd("HHGainVsHungarian", row_s["group_exact_mean"] - row_h["group_exact_mean"]),
        cmd("BiasReductionVsHungarian", row_h["abs_bias_mean"] - row_s["abs_bias_mean"]),
    ]
    macro_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_baseline(summary: pd.DataFrame, out_dir: Path) -> None:
    order = ["Greedy person-level", "Hungarian person-level", "Set-aware group-first"]
    s = summary.copy()
    s["method"] = pd.Categorical(s["method"], categories=order, ordered=True)
    s = s.sort_values("method")

    plt.figure(figsize=(6.4, 4.2))
    plt.bar(s["method"], s["group_exact_mean"])
    plt.ylabel("Mean exact group match rate")
    plt.title("Whole-group coherence at baseline ambiguity")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_baseline_group_exact.pdf")
    plt.close()

    plt.figure(figsize=(6.4, 4.2))
    plt.bar(s["method"], s["abs_bias_mean"])
    plt.ylabel("Mean absolute bias")
    plt.title("Bias in linked treatment-effect estimate at baseline ambiguity")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_baseline_bias.pdf")
    plt.close()


def plot_sweep(sweep_summary: pd.DataFrame, out_dir: Path) -> None:
    order = ["Greedy person-level", "Hungarian person-level", "Set-aware group-first"]
    sweep_summary = sweep_summary.copy()
    sweep_summary["method"] = pd.Categorical(sweep_summary["method"], categories=order, ordered=True)
    sweep_summary = sweep_summary.sort_values(["method", "ambiguity"])

    plt.figure(figsize=(6.4, 4.2))
    for method in order:
        sub = sweep_summary.loc[sweep_summary["method"] == method]
        plt.plot(sub["ambiguity"], sub["group_exact_mean"], marker="o", label=method)
    plt.xlabel("Ambiguity level")
    plt.ylabel("Mean exact group match rate")
    plt.title("Set-aware linkage preserves group coherence as ambiguity rises")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_sweep_group_exact.pdf")
    plt.close()

    plt.figure(figsize=(6.4, 4.2))
    for method in order:
        sub = sweep_summary.loc[sweep_summary["method"] == method]
        plt.plot(sub["ambiguity"], sub["abs_bias_mean"], marker="o", label=method)
    plt.xlabel("Ambiguity level")
    plt.ylabel("Mean absolute bias")
    plt.title("Set-aware linkage lowers downstream bias across ambiguity levels")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_sweep_bias.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--baseline-ambiguity", type=float, default=1.5)
    parser.add_argument("--baseline-runs", type=int, default=150)
    parser.add_argument("--sweep-runs", type=int, default=60)
    parser.add_argument("--sweep-values", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 2.5])
    args = parser.parse_args()

    project_root = args.project_root
    if not project_root.exists():
        raise ValueError(f"project_root does not exist: {project_root}")
    results_dir = project_root / "results"
    tables_dir = project_root / "tables"
    figures_dir = project_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_runs = run_many([args.baseline_ambiguity], n_runs=args.baseline_runs)
    baseline_summary = summarize(baseline_runs, ["ambiguity", "method"])
    baseline_runs.to_csv(results_dir / "baseline_runs.csv", index=False)
    baseline_summary.to_csv(results_dir / "baseline_summary.csv", index=False)

    sweep_runs = run_many(args.sweep_values, n_runs=args.sweep_runs)
    sweep_summary = summarize(sweep_runs, ["ambiguity", "method"])
    sweep_runs.to_csv(results_dir / "sweep_runs.csv", index=False)
    sweep_summary.to_csv(results_dir / "sweep_summary.csv", index=False)

    baseline_table = baseline_summary.loc[baseline_summary["ambiguity"] == args.baseline_ambiguity].copy()
    format_table(baseline_table, tables_dir / "tab_baseline_metrics.tex")
    write_macros(baseline_table, tables_dir / "baseline_macros.tex")

    plot_baseline(baseline_table, figures_dir)
    plot_sweep(sweep_summary, figures_dir)

    print("Wrote baseline and sweep results to:", results_dir)
    print("Wrote LaTeX tables to:", tables_dir)
    print("Wrote figures to:", figures_dir)


if __name__ == "__main__":
    main()
