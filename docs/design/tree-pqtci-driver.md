# Sequential tree pQTCI driver

## Status

Proposal for review. Milestone M2 of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md).
It adds public API to `tensor4all-partitionedtreetn` and must be approved
before implementation. It builds on the M1 contract in
[tree-interpolation-engine-seam.md](./tree-interpolation-engine-seam.md).

## Goal

Adaptive patched interpolation of a function on an arbitrary tree: run an
interpolation engine on the whole domain, and wherever the engine does not
converge below the bond cap, fix the next site in a given order and retry on
each child region. The result is a `PartitionedTreeTN` whose patches are
disjoint. This is the producer of every patch that later milestones measure
or consume, so it favours correctness and determinism over speed; parallel
execution is M7 and split-site selection beyond a fixed order is M5.

## Findings that shape the design

Verified against the M1 branch state.

- The M1 contract (`tensor4all_treetn::interpolation`) takes an
  `InterpolationProblem` (topology, active sites per node, initial pivots,
  absolute tolerance, bond cap, seed) and returns an `InterpolationOutcome`
  whose network carries only the active site indices, a termination verdict,
  the raw error estimate, the maximum sampled magnitude, and optional pivots.
  `Converged` is the only accepted verdict. All-zero initial samples return
  `AllSamplesZero`; non-finite initial samples return `Evaluator`.
- `SubDomainTreeTN::new(tree, projector)` requires every projector index to be
  a site index of `tree` (otherwise `ProjectorIndexNotFound`) and masks the
  projected indices eagerly. An outcome network therefore cannot be wrapped
  directly; the fixed sites must be re-attached first.
- `tensor4all_core::outer_product` forms an explicit tensor product and
  `IdxTensor::onehot` builds a one-hot tensor. Attaching the one-hot vector of
  a fixed coordinate to a node reproduces exactly the eagerly masked form.
  The one-hot factor must match the node tensor's scalar type, as
  `mask_index` does, so that partitions stay homogeneous in dtype.
- `tensor4all_core::CachedFunction` caches evaluations keyed by full
  multi-indices, supports batch evaluation, counts hits, and is shareable
  across threads (`RwLock`).
- A `PartitionedTreeTN` treats an absent patch as zero, and its projectors
  must be disjoint; they need not cover the whole domain.
- Design lineage: the chain driver `tensor4all-partitionedtt::adaptiveinterpolate`
  (itself following TCIAlgorithms.jl) uses a FIFO queue of patches, a fixed
  patch order, per-patch seeds derived from the patch path, compatible and
  recycled pivots replenished with seeded random candidates, and a sampled-zero
  shortcut. It is design lineage only, not a verification baseline.

## Public surface

A new module `tensor4all_partitionedtreetn::interpolation` (names are
proposals):

```rust
pub struct AdaptiveInterpolationOptions {
    /// Relative tolerance; the absolute tolerance passed to the engine is
    /// `relative_tolerance * reference_scale`.
    pub relative_tolerance: f64,
    /// Scale used for every patch. `None` pins it once to the largest
    /// magnitude of the root patch's initial samples.
    pub reference_scale: Option<f64>,
    /// Bond cap per patch; patching needs a cap.
    pub max_bond_dim: NonZeroUsize,
    /// Order in which sites are fixed when a patch splits: an exact
    /// permutation of all site indices, or empty for the derived site order.
    pub patch_order: Vec<DynIndex>,
    /// Target number of distinct initial pivots per patch (default 5).
    pub n_initial_pivots: usize,
    /// Seed children with the full-domain pivots of their rejected parent.
    pub recycle_pivots: bool,
    /// Root seed; each patch derives its own seed from it and its path.
    pub seed: u64,
    /// Resource limit on processed patches; `None` means unlimited.
    pub max_patches: Option<usize>,
}

pub struct AdaptiveInterpolationReport {
    pub accepted_patches: usize,
    pub zero_patches: usize,
    pub splits: usize,
    pub reference_scale: f64,
    pub function_evaluations: usize,
    pub cache_hits: usize,
}

pub struct AdaptiveInterpolationResult<V> {
    pub partition: PartitionedTreeTN<V>,
    pub report: AdaptiveInterpolationReport,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AdaptiveInterpolationError {
    InvalidInput { message: String },
    Interpolation { source: InterpolationError },
    Partition { source: PartitionedTreeTNError },
    NoSplitIndexLeft { projector: Projector },
    PatchLimitExceeded { limit: usize },
}

pub fn adaptive_interpolate<T, V, E, F>(
    engine: &E,
    topology: NodeNameNetwork<V>,
    node_sites: BTreeMap<V, Vec<DynIndex>>,
    initial_pivots: ColMajorArray<usize>,
    evaluate: F,
    options: &AdaptiveInterpolationOptions,
) -> Result<AdaptiveInterpolationResult<V>, AdaptiveInterpolationError>
where
    E: TreeInterpolator<T>,
    F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>>;
```

- `evaluate` receives full-domain points in the site order derived from
  `node_sites` (`InterpolationProblem::derive_site_order`), shape
  `[n_sites, n_points]`, column-major. `initial_pivots` uses the same order.
- The driver depends only on the M1 trait; `partitionedtreetn` gains no
  dependency on any engine crate.

## Algorithm

1. **Validate** inputs (exact permutation for `patch_order`, finite
   nonnegative tolerance, positive `n_initial_pivots`, pivot shape and
   bounds, finite positive `reference_scale` if given) and build one
   `CachedFunction` over the full domain around `evaluate`.
2. **Queue.** Start with the root patch (empty projector). Process patches in
   FIFO order. Count processed patches against `max_patches`.
3. **Candidates.** For a patch, keep the user pivots and (if enabled) the
   recycled parent pivots that are compatible with its projector, deduplicate
   them, and replenish with seeded random points inside the patch up to
   `n_initial_pivots` (bounded by the number of points in the patch, with
   checked arithmetic). The patch seed is derived from the root seed and the
   patch path, so results do not depend on queue order.
4. **Reference scale.** If not given, it is pinned once from the root
   patch's candidate samples (the largest magnitude) and reused for every
   patch. A zero or non-finite pinned scale is an `InvalidInput` error with a
   remedy (pass `reference_scale`).
5. **Zero screening.** If every candidate sample of a patch is exactly zero,
   the patch is recorded as a zero patch and omitted from the partition. This
   is a finite-sampling policy, as in the chain lineage; callers with sparse
   functions should supply pivots in the support.
6. **Fully fixed patch.** If every site is fixed, the patch is a single value
   evaluated directly.
7. **Interpolate.** Otherwise build an `InterpolationProblem` with the active
   sites, the candidate pivots in active coordinates,
   `absolute_tolerance = relative_tolerance * reference_scale`, the bond cap,
   and the patch seed. The engine's evaluator inserts the fixed coordinates
   and calls the cache.
8. **Accept or split.** `Converged` is accepted. Any other verdict splits the
   patch at the next site in `patch_order` that is not fixed yet, one child
   per coordinate; with no site left the driver returns `NoSplitIndexLeft`.
   With `recycle_pivots`, the outcome's pivots are converted to full-domain
   points and passed to the children.
9. **Re-embed.** An accepted outcome network gets every fixed site
   re-attached to its original node by an outer product with a one-hot vector
   of the node tensor's scalar type, then is wrapped with
   `SubDomainTreeTN::new(tree, projector)`. This helper is shared by all
   engines and lives in the driver.
10. **Assemble** the accepted patches into a `PartitionedTreeTN` and return it
    with the report.

The partition stays in the eager (masked) form required by the current
`partitionedtreetn` invariant; M4 decides whether that changes.

## Determinism

For a fixed seed, a deterministic evaluator, and a deterministic engine, the
partition and report are identical across runs: patch seeds depend only on
the root seed and the patch path, the reference scale is pinned once, and the
queue order is fixed.

## Provenance

The driver follows the published algorithm and the queue design of the chain
lineage; it is an independent implementation, not a translation of
TCIAlgorithms.jl code. The PR adds a row to
`docs/PROVENANCE_AND_CITATION_POLICY.md` (relationship: Inspired, with the
algorithm reference arXiv:2602.22372). If any code is translated closely from
`tensor4all-partitionedtt`, the PR instead follows the derivation-notice and
`LICENSE-TCIALGORITHMS-MIT` obligations stated in
[partitioned-treetn.md](./partitioned-treetn.md).

## Tests

- Chain and branched trees (a node of degree three or more) and a node with
  several sites, with the TreeTCI engine and the M1 test-only mock engine.
- A function with localized features on a small quantics grid whose monolithic
  rank exceeds the cap, so patching must split; the assembled partition is
  compared with a dense reference (materialize once, subtract, `maxabs`) within
  a tolerance consistent with `relative_tolerance`.
- A function with a vanishing region: zero patches are omitted, the remaining
  projectors are disjoint, and together with the zero patches they cover the
  domain.
- Pivot recycling on and off both reach the tolerance.
- Determinism across two runs; the report's evaluation counts show no
  duplicate evaluation of a point.
- `NoSplitIndexLeft`, `PatchLimitExceeded`, a zero pinned reference scale, and
  each `InvalidInput` branch.
- Index identity: sites sharing an ID but differing in prime level or tags.
- Rustdoc: runnable, asserted examples for every public item and `# Errors`
  naming the variants.

## Non-goals

- Parallel execution (M7), error norms and verified bounds (M3), the patch
  representation decision (M4), and split strategies other than a fixed order
  (M5).
- Retiring or changing `tensor4all-partitionedtt`.

## Open questions for review

1. Module and function names.
2. Whether a global cache (one `CachedFunction` for all patches) is
   preferable to per-patch caches handed to children, as in the chain lineage.
   This proposal uses the global cache: patches are disjoint, keys are full
   multi-indices, and no transfer step is needed; its memory grows with all
   evaluations.
