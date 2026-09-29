# Tree interpolation engine seam

## Status

Proposal for review. Milestone M1 of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md).
It adds public API to `tensor4all-treetn` and `tensor4all-treetci` and must be
approved before implementation. It also amends the scope statement of
[partitioned-treetn.md](./partitioned-treetn.md) (see the last section).

## Goal

A patch driver in `tensor4all-partitionedtreetn` (milestone M2) must run a
tree interpolation engine on one patch at a time without depending on any
engine crate. This record defines the contract between the driver and an
engine, and the TreeTCI implementation of it. Adding another engine later
(TreeACI, RSI) means implementing the contract in that engine's crate only.

## What the driver needs from one patch

For a patch whose projected sites are fixed, the driver hands the engine the
remaining (active) sites and needs back:

1. a network over the active sites, using the caller's site identities;
2. a verdict that separates "converged within tolerance" from "stopped at the
   bond cap" and "stopped at the iteration limit", because only the first is
   accepted and the others split the patch;
3. the error estimate and the maximum sampled magnitude, so the driver can
   apply its own normalization rule (milestone M3);
4. full-domain pivots of the result, so children can be seeded with them
   (pivot recycling).

## Findings that shape the design

- `tensor4all_treetci::crossinterpolate2` returns
  `(TreeTN<IdxTensor, usize>, ranks_per_iter, errors_per_iter)` only.
- `tensor4all_treetci::optimize_with_proposer` stops either when the errors
  of the last sweeps are below tolerance with no new global pivots and a
  stable rank, or when the rank reached `max_bond_dim`. Both return the same
  `Ok((ranks, errors))`; a caller cannot tell them apart. Exhausting
  `max_iter` is a third, also indistinguishable, outcome.
- `TreeTCI2` keeps `max_sample_value` and per-subtree pivot sets `ijset`
  (`SubtreeKey -> [n_subtree_sites, n_pivots]`), from which full-domain pivots
  can be assembled edge by edge.
- `tensor4all_treetci::to_treetn` creates fresh site indices
  (`DynIndex::new_dyn`), one per vertex. A `TreeTciGraph` vertex has exactly
  one local dimension.
- `TreeTN::replace_site_index_with_indices` splits one index into several by
  an exact local reshape, which maps a fused vertex back to a node's site
  indices.

## Contract in `tensor4all-treetn`

A new module `tensor4all_treetn::interpolation` (names are proposals):

```rust
/// Column-major batch of points over the problem's active sites:
/// shape (n_sites, n_points), coordinates zero-based.
pub struct SiteBatch<'a> { /* data, n_sites, n_points */ }

/// One interpolation problem: a named tree with the active site indices
/// of each node, in a fixed site order used by batches and pivots.
pub struct InterpolationProblem<V> {
    /// Node names and edges of the tree; every patch of one partition
    /// uses the same topology.
    pub topology: NodeNameNetwork<V>,
    /// Active site indices per node; a node may have none (all of its
    /// sites fixed), in which case it becomes a site-free node.
    pub node_sites: BTreeMap<V, Vec<DynIndex>>,
    /// Order of all active sites in `SiteBatch` rows and pivots.
    pub site_order: Vec<DynIndex>,
    /// Candidate pivots in `site_order` coordinates.
    pub initial_pivots: Vec<Vec<usize>>,
    /// Requested tolerance and hard bond cap.
    pub tolerance: f64,
    pub max_bond_dim: Option<usize>,
    /// Seed for engine-internal randomness.
    pub seed: u64,
}

#[non_exhaustive]
pub enum InterpolationTermination { Converged, BondCapReached, IterationLimit }

pub struct InterpolationOutcome<T, V> {
    /// Network over the active sites with the problem's site identities
    /// and topology; site-free nodes carry no site index.
    pub network: TreeTN<IdxTensor, V>,
    pub termination: InterpolationTermination,
    /// Final error estimate as defined by the engine, not normalized.
    pub error_estimate: f64,
    /// Largest sampled magnitude.
    pub max_sample_magnitude: f64,
    /// Full-domain pivots of the result in `site_order` coordinates.
    pub pivots: Vec<Vec<usize>>,
}

pub trait TreeInterpolator<T> {
    fn interpolate<V, F>(
        &self,
        problem: &InterpolationProblem<V>,
        evaluate: F,
    ) -> Result<InterpolationOutcome<T, V>, TreeTNOperationError>
    where
        V: /* same bounds as TreeTN node names */,
        F: Fn(SiteBatch<'_>) -> anyhow::Result<Vec<T>>;
}
```

Design points:

- **Universal knobs only.** Tolerance, bond cap, and seed are part of the
  problem because the driver sets them per patch. Engine-specific settings
  (sweep counts, proposers, global pivot search) are fields of the engine
  value implementing the trait, so a new engine adds no field here.
- **Caller identities.** The engine returns the network in the problem's site
  identities and node names, so the driver never handles engine-created
  indices. This moves index mapping into each engine implementation, where
  the engine's own conventions are known.
- **Raw error.** The engine reports its error estimate and the maximum sampled
  magnitude without normalizing, so the driver applies one normalization rule
  across all patches (M3 pins the reference scale globally).
- **Batch type.** `SiteBatch` has the same column-major layout as
  `tensor4all_treetci::GlobalIndexBatch`, so the TreeTCI implementation wraps
  it without copying. `tensor4all-treetn` cannot use the treetci type.
- **Scalar bound.** `T` is bounded by the scalar traits `IdxTensor` and TreeTCI
  already require for `f64` and `Complex64`; the exact bound set is fixed in
  implementation to the minimum both sides need.

## TreeTCI implementation in `tensor4all-treetci`

1. **Termination reason.** `optimize_with_proposer` returns a report with
   `ranks`, `errors`, and a termination reason
   (`Converged`, `MaxBondDimension`, `MaxIterations`) instead of the bare
   tuple. `crossinterpolate2` keeps its current return for existing callers
   or is updated with its callers in the same PR (early development, no
   compatibility shim).
2. **Vertices.** Each node of the problem becomes one TreeTCI vertex; its
   local dimension is the product of its active site dimensions (fused in the
   node's `node_sites` order, column-major), or one for a site-free node.
   Batches are translated from vertex coordinates to `site_order` rows.
3. **Pivots.** Initial pivots are converted to vertex coordinates and passed
   through `add_global_pivots`. After optimization, full-domain pivots are
   assembled from `ijset` by joining the two sides of each edge pivot by
   pivot, then converted back to `site_order` coordinates and deduplicated.
4. **Network.** The materialized network's fresh vertex indices are replaced
   by the node's site identities with `replace_site_index_with_indices`
   (splitting fused vertices); a site-free node's dimension-one index is
   removed.

## Tests

- Chain and branched trees (at least one node of degree three or more),
  compared with dense references on small cases.
- A node with several active sites (fused vertex) and a site-free node.
- Each termination reason is produced and reported: converged, bond cap
  reached (a function of known rank above the cap), iteration limit.
- Returned pivots are valid full-domain points and seed a second run.
- Site identities and node names of the result equal the problem's.
- A test-only mock engine implements the trait and runs through the same
  generic test helper, demonstrating that a second engine needs no change to
  the trait.
- Rustdoc: runnable, asserted examples for every new public item, with
  `# Errors` naming the failure conditions.

## Amendment to partitioned-treetn.md

The migration record states that adaptive interpolation is not part of
`tensor4all-partitionedtreetn`. Its reason was to keep that crate free of TCI
dependencies. With this seam the M2 driver lives in `partitionedtreetn` and
depends only on the trait in `tensor4all-treetn`, so the reason still holds.
The implementation PR for M2 updates the scope statement accordingly.

## Open questions for review

1. Names: `TreeInterpolator`, `InterpolationProblem`, `SiteBatch`,
   `InterpolationOutcome`.
2. Whether `InterpolationProblem` should borrow the topology instead of owning
   it, since the driver solves many patches on one topology.
3. Whether the trait should take the evaluator by reference (`&F`) so the
   driver can reuse one closure across patches without cloning.
