"""Type definitions and protocols for setjoin."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@runtime_checkable
class Comparator(Protocol):
    """Protocol for string/value comparison functions."""

    def __call__(
        self, a: NDArray[np.object_], b: NDArray[np.object_]
    ) -> NDArray[np.float64]:
        """Compare arrays element-wise, returning similarity scores in [0, 1]."""
        ...


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Result of a matching operation."""

    matches: list[tuple[int, int]]
    """List of (source_idx, target_idx) pairs."""

    total_score: float
    """Sum of scores for all matched pairs."""

    method: str
    """Name of the matching method used."""

    group_assignments: dict[int, int] | None = None
    """Mapping from source group ID to target group ID (structure-aware)."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Additional metadata about the matching process."""

    def __len__(self) -> int:
        return len(self.matches)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert matches to a DataFrame."""
        df = pd.DataFrame(self.matches)
        df.columns = pd.Index(["source_idx", "target_idx"])
        return df


@dataclass(frozen=True, slots=True)
class FieldConfig:
    """Configuration for a single field in scoring."""

    weight: float = 1.0
    """Weight applied to this field's similarity score."""

    comparator: str = "exact"
    """Name of comparison function: exact, abs_diff, levenshtein, jaro_winkler."""

    missing_value: float = 0.0
    """Score to use when either value is missing."""


@dataclass(frozen=True, slots=True)
class GroupSpec:
    """Specification of a group of records."""

    group_id: int
    """Identifier for this group."""

    indices: Sequence[int]
    """Record indices belonging to this group."""

    def __len__(self) -> int:
        return len(self.indices)


ScoreMatrix = NDArray[np.float64]
"""Type alias for score matrices (n_source x n_target)."""
