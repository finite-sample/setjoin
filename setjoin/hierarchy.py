"""Hierarchy specification and decomposition for structure-aware matching."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from setjoin.types import GroupSpec


@dataclass
class HierarchySpec:
    """
    Specification of hierarchical structure for record groups.

    This defines how records are grouped (e.g., persons within households,
    students within schools, items within orders) for structure-preserving matching.
    """

    source_groups: dict[int, Sequence[int]]
    """Mapping from source group ID to record indices."""

    target_groups: dict[int, Sequence[int]]
    """Mapping from target group ID to record indices."""

    @classmethod
    def from_dataframe(
        cls,
        source: pd.DataFrame,
        target: pd.DataFrame,
        source_group_col: str,
        target_group_col: str,
    ) -> "HierarchySpec":
        """
        Create HierarchySpec from DataFrames with group columns.

        Args:
            source: Source DataFrame
            target: Target DataFrame
            source_group_col: Column name containing group IDs in source
            target_group_col: Column name containing group IDs in target

        Returns:
            HierarchySpec: Group mappings
        """
        source_groups: dict[int, Sequence[int]] = {}
        for group_id, group_df in source.groupby(source_group_col):
            source_groups[int(group_id)] = list(group_df.index)  # type: ignore[arg-type]

        target_groups: dict[int, Sequence[int]] = {}
        for group_id, group_df in target.groupby(target_group_col):
            target_groups[int(group_id)] = list(group_df.index)  # type: ignore[arg-type]

        return cls(source_groups=source_groups, target_groups=target_groups)

    @classmethod
    def from_groupby(
        cls,
        source_groupby: Any,
        target_groupby: Any,
    ) -> "HierarchySpec":
        """
        Create HierarchySpec from pandas GroupBy objects.

        Args:
            source_groupby: GroupBy object for source records
            target_groupby: GroupBy object for target records

        Returns:
            HierarchySpec: Group mappings
        """
        source_groups: dict[int, Sequence[int]] = {}
        for group_id, group_df in source_groupby:
            source_groups[int(group_id)] = list(group_df.index)

        target_groups: dict[int, Sequence[int]] = {}
        for group_id, group_df in target_groupby:
            target_groups[int(group_id)] = list(group_df.index)

        return cls(source_groups=source_groups, target_groups=target_groups)

    @property
    def n_source_groups(self) -> int:
        """Number of groups in source."""
        return len(self.source_groups)

    @property
    def n_target_groups(self) -> int:
        """Number of groups in target."""
        return len(self.target_groups)

    @property
    def source_group_ids(self) -> list[int]:
        """List of source group IDs."""
        return list(self.source_groups.keys())

    @property
    def target_group_ids(self) -> list[int]:
        """List of target group IDs."""
        return list(self.target_groups.keys())

    def get_source_group(self, group_id: int) -> GroupSpec:
        """Get specification for a source group."""
        return GroupSpec(group_id=group_id, indices=self.source_groups[group_id])

    def get_target_group(self, group_id: int) -> GroupSpec:
        """Get specification for a target group."""
        return GroupSpec(group_id=group_id, indices=self.target_groups[group_id])

    def source_group_sizes(self) -> dict[int, int]:
        """Get sizes of all source groups."""
        return {gid: len(indices) for gid, indices in self.source_groups.items()}

    def target_group_sizes(self) -> dict[int, int]:
        """Get sizes of all target groups."""
        return {gid: len(indices) for gid, indices in self.target_groups.items()}


def compute_group_score_matrix(
    hierarchy: HierarchySpec,
    record_scores: NDArray[np.float64],
) -> tuple[NDArray[np.float64], dict[tuple[int, int], list[tuple[int, int]]]]:
    """
    Compute score matrix at the group level with optimal within-group assignments.

    For each pair of (source_group, target_group), computes the optimal assignment
    of records within those groups and returns the total score.

    Args:
        hierarchy: HierarchySpec defining group structure
        record_scores: Pairwise record score matrix
            (shape: n_source_records x n_target_records)

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], list[tuple[int, int]]]]:
            Group score matrix and within-group match assignments
    """
    from scipy.optimize import linear_sum_assignment

    source_ids = hierarchy.source_group_ids
    target_ids = hierarchy.target_group_ids

    group_scores = np.zeros((len(source_ids), len(target_ids)), dtype=np.float64)
    within_matches: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for si, src_id in enumerate(source_ids):
        src_indices = list(hierarchy.source_groups[src_id])
        for ti, tgt_id in enumerate(target_ids):
            tgt_indices = list(hierarchy.target_groups[tgt_id])

            sub_matrix = record_scores[np.ix_(src_indices, tgt_indices)]

            n_src, n_tgt = len(src_indices), len(tgt_indices)
            n_match = min(n_src, n_tgt)

            if n_match == 0:
                group_scores[si, ti] = 0.0
                within_matches[(src_id, tgt_id)] = []
                continue

            rows, cols = linear_sum_assignment(-sub_matrix)
            total_score = float(sub_matrix[rows, cols].sum())
            group_scores[si, ti] = total_score

            matches = [
                (src_indices[int(r)], tgt_indices[int(c)])
                for r, c in zip(rows, cols, strict=True)
            ]
            within_matches[(src_id, tgt_id)] = matches

    return group_scores, within_matches


def decompose_by_size(
    hierarchy: HierarchySpec,
) -> dict[tuple[int, int], tuple[list[int], list[int]]]:
    """
    Decompose groups by size for efficient matching.

    Groups source and target groups by their sizes to enable size-aware matching
    strategies.

    Args:
        hierarchy: HierarchySpec defining group structure

    Returns:
        dict[tuple[int, int], tuple[list[int], list[int]]]:
            Size pairs to group ID lists
    """
    source_sizes = hierarchy.source_group_sizes()
    target_sizes = hierarchy.target_group_sizes()

    source_by_size: dict[int, list[int]] = {}
    for gid, size in source_sizes.items():
        source_by_size.setdefault(size, []).append(gid)

    target_by_size: dict[int, list[int]] = {}
    for gid, size in target_sizes.items():
        target_by_size.setdefault(size, []).append(gid)

    result: dict[tuple[int, int], tuple[list[int], list[int]]] = {}
    for src_size, src_ids in source_by_size.items():
        for tgt_size, tgt_ids in target_by_size.items():
            result[(src_size, tgt_size)] = (src_ids, tgt_ids)

    return result
