Theory
======

This section provides theoretical background on structure-aware record linkage.

The Problem
-----------

Standard record linkage finds a matching between source records and target
records that maximizes total similarity score. Given a score matrix
:math:`S \in \mathbb{R}^{n \times m}` where :math:`S_{ij}` is the similarity
between source record :math:`i` and target record :math:`j`, the Hungarian
algorithm finds:

.. math::

   \max_{\pi} \sum_{i=1}^{n} S_{i,\pi(i)}

where :math:`\pi` is a permutation (or partial permutation if :math:`n \neq m`).

When records have hierarchical structure, this formulation ignores an important
constraint: members of the same group should map to the same target group.

Two-Level Assignment
--------------------

Structure-aware matching decomposes the problem into two levels:

1. **Group-level assignment**: Match source groups to target groups
2. **Record-level assignment**: Match records within matched groups

Given:

- Source groups :math:`\mathcal{G}_S = \{g_1^S, g_2^S, \ldots\}`
- Target groups :math:`\mathcal{G}_T = \{g_1^T, g_2^T, \ldots\}`

For each pair of groups :math:`(g^S, g^T)`, compute the optimal within-group
assignment score:

.. math::

   W(g^S, g^T) = \max_{\pi_{g^S \to g^T}} \sum_{i \in g^S} S_{i, \pi(i)}

This gives a group-level score matrix :math:`W`. Apply Hungarian algorithm
at the group level to find optimal group matching :math:`\Phi`:

.. math::

   \max_{\Phi} \sum_{g^S \in \mathcal{G}_S} W(g^S, \Phi(g^S))

The final record-level matching is the union of within-group assignments
for matched groups.

Properties
----------

**Coherence guarantee**: All members of a source group map to the same
target group by construction.

**Optimality within constraint**: Given the coherence constraint, the
matching maximizes total score. It is globally optimal among all
structure-preserving matchings.

**Score sacrifice**: The total score may be lower than unconstrained
Hungarian matching. This "sacrifice" measures the cost of structure
preservation:

.. math::

   \text{sacrifice} = \text{score}_{\text{Hungarian}} - \text{score}_{\text{structure-aware}}

When to Use Structure-Aware Matching
------------------------------------

Structure-aware matching is appropriate when:

1. Records have meaningful group structure that should be preserved
2. Downstream analysis requires coherent group assignments
3. Domain knowledge suggests members of source groups belong together

It may sacrifice individual match quality for group coherence. Use standard
Hungarian matching when group structure doesn't matter or when maximizing
individual match accuracy is the primary goal.

Computational Complexity
------------------------

Let :math:`n` be the number of records, :math:`k` the number of groups,
and :math:`s` the maximum group size.

- Computing group scores: :math:`O(k^2 \cdot s^3)` (Hungarian per pair)
- Group-level assignment: :math:`O(k^3)` (Hungarian on :math:`k \times k` matrix)
- Total: :math:`O(k^2 \cdot s^3 + k^3)`

For typical hierarchical data where :math:`k \approx n/s`, this is
:math:`O(n^2 s)`, competitive with standard Hungarian's :math:`O(n^3)`.
