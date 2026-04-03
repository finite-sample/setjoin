setjoin
=======

**Set-aware record linkage with structure-preserving joins**

setjoin provides algorithms for matching records across datasets while
preserving hierarchical structure. When linking records that belong to groups
(households, schools, orders, etc.), standard matching algorithms can split
group members across different target groups. setjoin's structure-aware
matching ensures coherent group assignments.

Key Features
------------

- **Structure-aware matching**: Preserves group membership during linkage
- **Multiple algorithms**: Greedy, Hungarian (optimal), and structure-aware methods
- **Configurable scoring**: Pluggable comparators with customizable weights
- **Rich diagnostics**: Match confidence, group margins, error analysis
- **Variable group sizes**: Not limited to fixed-size groups

Quick Example
-------------

.. code-block:: python

   from setjoin import HierarchySpec, Scorer, match, MatchReport

   # Configure scoring
   scorer = Scorer({
       "surname": {"weight": 1.5, "comparator": "levenshtein"},
       "first_name": {"weight": 1.2, "comparator": "jaro_winkler"},
       "age": {"weight": 0.35, "comparator": "abs_diff"},
   })
   scores = scorer.score(source_df, target_df)

   # Define hierarchy
   hierarchy = HierarchySpec.from_dataframe(
       source_df, target_df,
       source_group_col="household_id",
       target_group_col="household_id",
   )

   # Match with structure preservation
   result = match(scores, method="structure_aware", hierarchy=hierarchy)

   # Analyze results
   report = MatchReport(result, scores)
   print(report.summary())

Installation
------------

.. code-block:: bash

   pip install setjoin

   # With visualization support
   pip install setjoin[viz]

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   api
   theory

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
