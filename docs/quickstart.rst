Quickstart
==========

This guide walks through the basic workflow for structure-aware record linkage
with setjoin.

Problem Setup
-------------

Consider matching persons between two datasets where persons are grouped
(e.g., in households). Standard matching algorithms optimize individual
matches without considering group structure, potentially assigning members
of the same source group to different target groups.

setjoin's structure-aware matching ensures all members of a source group
map to the same target group.

Basic Workflow
--------------

1. **Compute Scores**

   Create a score matrix measuring similarity between all source-target pairs:

   .. code-block:: python

      from setjoin import Scorer

      scorer = Scorer({
          "surname": {"weight": 1.5, "comparator": "abs_diff"},
          "first_name": {"weight": 1.2, "comparator": "abs_diff"},
          "age": {"weight": 0.35, "comparator": "abs_diff"},
      })
      scores = scorer.score(source_df, target_df)

   Available comparators:

   - ``exact``: 1.0 if equal, 0.0 otherwise
   - ``abs_diff``: Negative absolute difference (for numeric fields)
   - ``levenshtein``: Levenshtein similarity ratio (for strings)
   - ``jaro_winkler``: Jaro-Winkler similarity (for strings, rewards common prefixes)

2. **Define Hierarchy**

   Specify how records are grouped:

   .. code-block:: python

      from setjoin import HierarchySpec

      hierarchy = HierarchySpec.from_dataframe(
          source_df, target_df,
          source_group_col="group_id",
          target_group_col="group_id",
      )

3. **Match**

   Run the matching algorithm:

   .. code-block:: python

      from setjoin import match

      # Structure-aware matching (preserves groups)
      result = match(scores, method="structure_aware", hierarchy=hierarchy)

      # Access matches
      for src_idx, tgt_idx in result.matches:
          print(f"Source {src_idx} -> Target {tgt_idx}")

      # Group assignments
      for src_group, tgt_group in result.group_assignments.items():
          print(f"Source group {src_group} -> Target group {tgt_group}")

4. **Evaluate**

   Analyze match quality:

   .. code-block:: python

      from setjoin import MatchReport

      report = MatchReport(
          result,
          scores,
          ground_truth=true_matches,  # Optional
          source_groups=hierarchy.source_groups,
          target_groups=hierarchy.target_groups,
      )

      print(f"Record accuracy: {report.record_accuracy}")
      print(f"Group exact match rate: {report.group_exact_match_rate}")

      # Per-match confidence
      confidence = report.match_confidence()
      print(confidence.head())

Comparing Methods
-----------------

Compare structure-aware matching against baselines:

.. code-block:: python

   from setjoin import compare_methods

   results = compare_methods(scores, hierarchy=hierarchy)

   for method, result in results.items():
       report = MatchReport(result, scores, source_groups=..., target_groups=...)
       print(f"{method}: {report.group_exact_match_rate:.3f} group coherence")

Variable Group Sizes
--------------------

setjoin handles groups of any size:

.. code-block:: python

   hierarchy = HierarchySpec(
       source_groups={
           0: [0, 1],        # 2-member group
           1: [2, 3, 4, 5],  # 4-member group
           2: [6],           # 1-member group
       },
       target_groups={
           0: [0, 1, 2],     # 3-member group
           1: [3, 4],        # 2-member group
           2: [5, 6, 7],     # 3-member group
       },
   )

The algorithm matches groups optimally and then matches members within
matched groups.
