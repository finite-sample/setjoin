# setjoin

[![PyPI version](https://img.shields.io/pypi/v/setjoin.svg)](https://pypi.org/project/setjoin/)
[![Python versions](https://img.shields.io/pypi/pyversions/setjoin.svg)](https://pypi.org/project/setjoin/)
[![License](https://img.shields.io/pypi/l/setjoin.svg)](LICENSE)
[![CI](https://github.com/finite-sample/setjoin/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/setjoin/actions/workflows/ci.yml)

Record linkage that keeps groups together. Match persons while preserving household membership, students while respecting school assignments, or any hierarchical data where group integrity matters.

## The Problem

Standard record linkage matches individuals optimally but ignores group structure. When household members should stay together, Hungarian matching might send them to different target households because it maximizes individual scores.

**Example**: Two source households, each with 2 members:
- Household 1: Alice (age 35), Bob (age 10)
- Household 2: Carol (age 45), Dan (age 42)

Target records (shuffled, unknown household IDs):
- Record A: age 36, Record B: age 11, Record C: age 44, Record D: age 43

Suppose A and C are in target household X, while B and D are in target household Y.

**Person-level Hungarian** might match Alice→A, Bob→B, Carol→C, Dan→D. This maximizes individual match quality but splits both source families across different target households.

**Structure-aware matching** first assigns household→household, then matches within. Result: all source household members map to the same target household.

## Why It Matters

In simulations with realistic ambiguity:

| Method | Group Coherence | Person Accuracy | Downstream Bias |
|--------|-----------------|-----------------|-----------------|
| Hungarian (person-level) | 12% | 32% | 3.7 |
| **Structure-aware** | **52%** | **48%** | **2.3** |

Structure-aware matching achieves **4x better group coherence** while also improving person-level accuracy. When groups are preserved, downstream analyses (treatment effects, household income, etc.) have less bias.

## When to Use

Use setjoin when your records have **known group structure** that must be preserved:

- **Household surveys → Admin records**: All family members must link to the same admin household
- **Student rosters → Test scores**: Students in class 3A must all map to the same target class
- **Employee lists → Payroll**: Workers at Plant X must link to the same payroll entity

Don't use setjoin if:
- Records are truly independent (no group structure)
- You want to discover groups (use clustering instead)

## How It Works

Structure-aware matching uses **two-level assignment**:

1. **Compute group scores**: For each (source_group, target_group) pair, solve the optimal within-group assignment using Hungarian algorithm. The group score is the sum of matched record scores.

2. **Assign groups**: Apply Hungarian algorithm to the group score matrix to find the best overall group-to-group pairing.

3. **Extract matches**: From matched groups, use the within-group assignments.

This guarantees all records in a source group map to a single target group while maximizing total match quality.

## Installation

```bash
pip install setjoin
```

## Quick Start

```python
import numpy as np
from setjoin import match, HierarchySpec

# Score matrix: how well does each source record match each target?
scores = np.array(
    [
        [10.0, 2.0, 1.0, 1.0],  # Person A scores high with targets 0,1
        [9.0, 10.0, 1.0, 1.0],  # Person B scores high with targets 0,1
        [1.0, 1.0, 10.0, 2.0],  # Person C scores high with targets 2,3
        [1.0, 1.0, 9.0, 10.0],  # Person D scores high with targets 2,3
    ]
)

# Define household structure: persons 0,1 are in household 0; persons 2,3 in household 1
hierarchy = HierarchySpec(
    source_groups={0: [0, 1], 1: [2, 3]},
    target_groups={0: [0, 1], 1: [2, 3]},
)

# Match while keeping households together
result = match(scores, method="structure_aware", hierarchy=hierarchy)
print(result.matches)  # [(0, 0), (1, 1), (2, 2), (3, 3)]
print(result.group_assignments)  # {0: 0, 1: 1} - household mappings
```

## Examples

### Basic Matching (No Hierarchy)

```python
import numpy as np
from setjoin import hungarian_match, greedy_match

scores = np.array(
    [
        [10.0, 1.0, 1.0],
        [1.0, 10.0, 1.0],
        [1.0, 1.0, 10.0],
    ]
)

# Optimal global assignment
result = hungarian_match(scores)
print(result.matches)  # [(0, 0), (1, 1), (2, 2)]
print(result.total_score)  # 30.0

# Fast greedy alternative
result = greedy_match(scores)
```

### Building Scores from DataFrames

```python
import pandas as pd
from setjoin import Scorer, FieldConfig

source = pd.DataFrame({"age": [25, 30, 35], "income": [50000, 60000, 70000]})
target = pd.DataFrame({"age": [26, 31, 34], "income": [51000, 59000, 72000]})

scorer = Scorer(
    {
        "age": FieldConfig(weight=1.0, comparator="abs_diff"),
        "income": FieldConfig(weight=0.001, comparator="abs_diff"),
    }
)
scores = scorer.score(source, target)
```

### Structure-Aware Matching (Groups)

```python
import pandas as pd
from setjoin import match, HierarchySpec, Scorer, FieldConfig

# Survey data with household IDs
survey = pd.DataFrame(
    {
        "household_id": [1, 1, 2, 2],
        "age": [35, 10, 45, 42],
        "income": [50000, 0, 60000, 58000],
    }
)

# Admin records with household IDs
admin = pd.DataFrame(
    {
        "household_id": [101, 101, 102, 102],
        "age": [36, 11, 44, 43],
        "income": [51000, 0, 59000, 57000],
    }
)

# Build score matrix (higher = better match, abs_diff returns negative distances)
scorer = Scorer(
    {
        "age": FieldConfig(weight=1.0, comparator="abs_diff"),
        "income": FieldConfig(weight=0.0001, comparator="abs_diff"),
    }
)
scores = scorer.score(survey, admin)

# Define hierarchy from dataframes
hierarchy = HierarchySpec.from_dataframe(
    survey,
    admin,
    source_group_col="household_id",
    target_group_col="household_id",
)

# Match: all members of survey household 1 -> same admin household
result = match(scores, method="structure_aware", hierarchy=hierarchy)
```

### Soft Matching (Uncertainty)

```python
import numpy as np
from setjoin import soft_match

scores = np.array(
    [
        [10.0, 9.0],
        [9.0, 10.0],
    ]
)

# Get probabilistic weights instead of hard assignments
weights = soft_match(scores, regularization=0.5)
print(weights.matrix)  # Soft assignment probabilities
print(weights.to_hard())  # Convert to hard matches when needed
```

### Calibration to Known Marginals

```python
import numpy as np
import pandas as pd
from setjoin import calibrated_match, CalibrationSpec

scores = np.eye(100) * 10  # 100 records
source_df = pd.DataFrame({"region": ["north"] * 60 + ["south"] * 40})

# Target: 50/50 split, not the 60/40 in source
calibration = CalibrationSpec(margins={"region": {"north": 0.5, "south": 0.5}})

result = calibrated_match(scores, source_df, calibration)
print(result.weights)  # Calibration weights for each match
print(result.calibration_achieved)  # Achieved proportions
```

## API Overview

| Function | Purpose |
|----------|---------|
| `match()` | Main entry point - routes to greedy, hungarian, or structure_aware |
| `hungarian_match()` | Optimal 1-to-1 assignment maximizing total score |
| `greedy_match()` | Fast heuristic picking highest scores first |
| `structure_aware_match()` | Optimal assignment preserving group structure |
| `soft_match()` | Probabilistic weights via entropy-regularized transport |
| `calibrated_match()` | Match + rake weights to hit target marginals |
| `Scorer` | Build score matrices from DataFrames with configurable comparators |
| `HierarchySpec` | Define group structure for structure-aware matching |
| `CalibrationSpec` | Define target marginal distributions |

## License

MIT
