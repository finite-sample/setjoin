"""Diagnostic metrics and report generation for match evaluation."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from setjoin.types import MatchResult, ScoreMatrix


@dataclass
class MatchReport:
    """Comprehensive diagnostics for evaluating match quality."""

    result: MatchResult
    """The match result being evaluated."""

    scores: ScoreMatrix
    """The score matrix used for matching."""

    ground_truth: list[tuple[int, int]] | None = None
    """Optional ground truth matches for accuracy computation."""

    source_groups: dict[int, list[int]] | None = None
    """Optional source group structure for coherence metrics."""

    target_groups: dict[int, list[int]] | None = None
    """Optional target group structure for coherence metrics."""

    source_df: pd.DataFrame | None = None
    """Optional source DataFrame for detailed analysis."""

    target_df: pd.DataFrame | None = None
    """Optional target DataFrame for detailed analysis."""

    _cache: dict[str, object] = field(default_factory=dict, repr=False)

    @property
    def n_matches(self) -> int:
        """Number of matches."""
        return len(self.result.matches)

    @property
    def record_accuracy(self) -> float | None:
        """
        Fraction of matches that are correct (if ground truth provided).
        """
        if self.ground_truth is None:
            return None

        truth_set = set(self.ground_truth)
        matched_set = set(self.result.matches)
        correct = len(truth_set & matched_set)
        return correct / len(truth_set) if truth_set else 0.0

    @property
    def group_accuracy(self) -> float | None:
        """
        Fraction of records matched to the correct group (if groups and truth provided).
        """
        if (
            self.ground_truth is None
            or self.source_groups is None
            or self.target_groups is None
        ):
            return None

        src_to_true_tgt_group = self._source_to_true_target_group()
        if src_to_true_tgt_group is None:
            return None

        tgt_idx_to_group = self._target_idx_to_group()

        correct = 0
        for src_idx, tgt_idx in self.result.matches:
            true_tgt_group = src_to_true_tgt_group.get(src_idx)
            if true_tgt_group is None:
                continue
            matched_tgt_group = tgt_idx_to_group.get(tgt_idx)
            if matched_tgt_group == true_tgt_group:
                correct += 1

        return correct / len(self.result.matches) if self.result.matches else 0.0

    @property
    def group_exact_match_rate(self) -> float | None:
        """
        Fraction of source groups where ALL members matched to the same target group.
        """
        if self.source_groups is None or self.target_groups is None:
            return None

        tgt_idx_to_group = self._target_idx_to_group()
        src_idx_to_tgt_idx = dict(self.result.matches)

        exact_matches = 0
        for _src_group_id, src_indices in self.source_groups.items():
            tgt_groups_matched: set[int | None] = set()
            for src_idx in src_indices:
                tgt_idx = src_idx_to_tgt_idx.get(src_idx)
                if tgt_idx is not None:
                    tgt_groups_matched.add(tgt_idx_to_group.get(tgt_idx))
            if len(tgt_groups_matched) == 1 and None not in tgt_groups_matched:
                exact_matches += 1

        return exact_matches / len(self.source_groups) if self.source_groups else 0.0

    @property
    def score_sacrifice(self) -> float | None:
        """
        Total score lost compared to unconstrained Hungarian matching.

        Positive value means structure-aware matching sacrificed score for coherence.
        """
        from setjoin.matchers import hungarian_match

        hungarian_result = hungarian_match(self.scores)
        return hungarian_result.total_score - self.result.total_score

    def match_confidence(self) -> pd.DataFrame:
        """
        Compute per-match confidence metrics.

        Returns DataFrame with columns:
        - source_idx, target_idx: the match
        - score: score of this match
        - rank: rank of this match among alternatives for source
        - margin: difference between this score and second-best alternative
        """
        rows = []
        for src_idx, tgt_idx in self.result.matches:
            score = self.scores[src_idx, tgt_idx]
            src_scores = self.scores[src_idx, :]
            sorted_scores = np.sort(src_scores)[::-1]
            rank = int(np.searchsorted(-sorted_scores, -score)) + 1
            if len(sorted_scores) > 1:
                margin = float(score - sorted_scores[1])
            else:
                margin = float("inf")
            rows.append(
                {
                    "source_idx": src_idx,
                    "target_idx": tgt_idx,
                    "score": float(score),
                    "rank": rank,
                    "margin": margin,
                }
            )
        return pd.DataFrame(rows)

    def group_margins(self) -> pd.DataFrame | None:
        """
        Compute per-group score margins.

        Returns DataFrame with columns:
        - source_group: source group ID
        - target_group: matched target group ID
        - within_score: total score of within-group matches
        - best_alternative_score: best score with a different target group
        - margin: difference between within_score and best_alternative
        """
        if self.source_groups is None or self.target_groups is None:
            return None
        if self.result.group_assignments is None:
            return None

        rows = []
        for src_gid, tgt_gid in self.result.group_assignments.items():
            src_indices = self.source_groups[src_gid]
            tgt_indices = self.target_groups[tgt_gid]

            within_score = float(self.scores[np.ix_(src_indices, tgt_indices)].sum())

            best_alt_score = float("-inf")
            for alt_tgt_gid, alt_tgt_indices in self.target_groups.items():
                if alt_tgt_gid == tgt_gid:
                    continue
                alt_score = float(
                    self.scores[np.ix_(src_indices, alt_tgt_indices)].sum()
                )
                best_alt_score = max(best_alt_score, alt_score)

            if best_alt_score != float("-inf"):
                margin = within_score - best_alt_score
                alt_score_val: float | None = best_alt_score
            else:
                margin = float("inf")
                alt_score_val = None

            rows.append(
                {
                    "source_group": src_gid,
                    "target_group": tgt_gid,
                    "within_score": within_score,
                    "best_alternative_score": alt_score_val,
                    "margin": margin,
                }
            )
        return pd.DataFrame(rows)

    def error_anatomy(self) -> pd.DataFrame | None:
        """
        Analyze errors by comparing to ground truth.

        Returns DataFrame with columns:
        - source_idx: source record
        - matched_target_idx: where we matched it
        - true_target_idx: where it should have matched
        - score_diff: how much score we lost
        - error_type: 'swap' or 'mismatch'
        """
        if self.ground_truth is None:
            return None

        truth_dict = dict(self.ground_truth)
        match_dict = dict(self.result.matches)

        rows = []
        for src_idx, matched_tgt in match_dict.items():
            true_tgt = truth_dict.get(src_idx)
            if true_tgt is None:
                continue
            if matched_tgt == true_tgt:
                continue

            matched_score = self.scores[src_idx, matched_tgt]
            true_score = self.scores[src_idx, true_tgt]
            score_diff = float(true_score - matched_score)

            true_match_for_matched = truth_dict.get(matched_tgt)
            error_type = "swap" if true_match_for_matched == src_idx else "mismatch"

            rows.append(
                {
                    "source_idx": src_idx,
                    "matched_target_idx": matched_tgt,
                    "true_target_idx": true_tgt,
                    "score_diff": score_diff,
                    "error_type": error_type,
                }
            )
        return pd.DataFrame(rows)

    def method_comparison(self, other: "MatchReport") -> pd.DataFrame:
        """
        Compare this result with another method's result.

        Returns DataFrame showing where the methods differ.
        """
        this_matches = set(self.result.matches)
        other_matches = set(other.result.matches)

        this_only = this_matches - other_matches
        other_only = other_matches - this_matches
        both = this_matches & other_matches

        rows = []
        for src, tgt in both:
            rows.append(
                {
                    "source_idx": src,
                    "target_idx": tgt,
                    "in_this": True,
                    "in_other": True,
                    "score": float(self.scores[src, tgt]),
                }
            )
        for src, tgt in this_only:
            rows.append(
                {
                    "source_idx": src,
                    "target_idx": tgt,
                    "in_this": True,
                    "in_other": False,
                    "score": float(self.scores[src, tgt]),
                }
            )
        for src, tgt in other_only:
            rows.append(
                {
                    "source_idx": src,
                    "target_idx": tgt,
                    "in_this": False,
                    "in_other": True,
                    "score": float(self.scores[src, tgt]),
                }
            )
        return pd.DataFrame(rows)

    def to_csv(self, output_dir: str | Path) -> None:
        """
        Write all diagnostic artifacts to CSV files.

        Creates:
        - match_confidence.csv
        - group_margins.csv (if groups provided)
        - error_anatomy.csv (if ground truth provided)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.match_confidence().to_csv(output_dir / "match_confidence.csv", index=False)

        margins = self.group_margins()
        if margins is not None:
            margins.to_csv(output_dir / "group_margins.csv", index=False)

        errors = self.error_anatomy()
        if errors is not None:
            errors.to_csv(output_dir / "error_anatomy.csv", index=False)

    def summary(self) -> dict[str, float | int | None]:
        """Return a dictionary of summary metrics."""
        return {
            "n_matches": self.n_matches,
            "total_score": self.result.total_score,
            "record_accuracy": self.record_accuracy,
            "group_accuracy": self.group_accuracy,
            "group_exact_match_rate": self.group_exact_match_rate,
            "score_sacrifice": self.score_sacrifice,
        }

    def _source_to_true_target_group(self) -> dict[int, int] | None:
        """Map source index to true target group."""
        if self.ground_truth is None or self.target_groups is None:
            return None

        tgt_idx_to_group = self._target_idx_to_group()
        truth_dict = dict(self.ground_truth)

        result: dict[int, int] = {}
        for src_idx, tgt_idx in truth_dict.items():
            tgt_group = tgt_idx_to_group.get(tgt_idx)
            if tgt_group is not None:
                result[src_idx] = tgt_group
        return result

    def _target_idx_to_group(self) -> dict[int, int]:
        """Map target index to target group."""
        if "_target_idx_to_group" in self._cache:
            return self._cache["_target_idx_to_group"]  # type: ignore[return-value]

        result: dict[int, int] = {}
        if self.target_groups:
            for gid, indices in self.target_groups.items():
                for idx in indices:
                    result[idx] = gid
        self._cache["_target_idx_to_group"] = result
        return result


def evaluate_matches(
    source: pd.DataFrame,
    target: pd.DataFrame,
    matches: list[tuple[int, int]],
    source_id_col: str = "latent_person_id",
    target_id_col: str = "latent_person_id",
    source_group_col: str | None = None,
    target_group_col: str | None = None,
) -> dict[str, float]:
    """
    Evaluate linkage quality with standard metrics.

    Args:
        source: Source DataFrame
        target: Target DataFrame
        matches: List of (source_idx, target_idx) pairs
        source_id_col: Column in source containing true record ID
        target_id_col: Column in target containing true record ID
        source_group_col: Optional column for group coherence metrics
        target_group_col: Optional column for group coherence metrics

    Returns:
        Dictionary of metrics
    """
    correct = 0
    for src_idx, tgt_idx in matches:
        if source.iloc[src_idx][source_id_col] == target.iloc[tgt_idx][target_id_col]:
            correct += 1

    record_accuracy = correct / len(matches) if matches else 0.0

    result: dict[str, float] = {"record_accuracy": record_accuracy}

    if source_group_col and target_group_col:
        group_correct = 0
        src_idx_to_tgt_idx = dict(matches)

        source_groups: dict[int, list[int]] = {}
        for i in range(len(source)):
            gid = int(source.iloc[i][source_group_col])
            source_groups.setdefault(gid, []).append(i)

        target_groups: dict[int, list[int]] = {}
        for i in range(len(target)):
            gid = int(target.iloc[i][target_group_col])
            target_groups.setdefault(gid, []).append(i)

        tgt_idx_to_group = {
            idx: gid for gid, indices in target_groups.items() for idx in indices
        }

        for _src_gid, src_indices in source_groups.items():
            matched_groups: set[int | None] = set()
            for src_idx in src_indices:
                tgt_idx_result = src_idx_to_tgt_idx.get(src_idx)
                if tgt_idx_result is not None:
                    matched_groups.add(tgt_idx_to_group.get(tgt_idx_result))
            if len(matched_groups) == 1 and None not in matched_groups:
                group_correct += 1

        if source_groups:
            result["group_exact_match_rate"] = group_correct / len(source_groups)
        else:
            result["group_exact_match_rate"] = 0.0

    return result
