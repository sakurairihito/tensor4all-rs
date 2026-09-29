# Sequential tree pQTCI driver

## Status

Proposal for review, revised after a first design review. Milestone M2 of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md).
It adds public API to `tensor4all-partitionedtreetn` and must be approved
before implementation. It builds on the M1 contract in
[tree-interpolation-engine-seam.md](./tree-interpolation-engine-seam.md).

## Goal

Adaptive patched interpolation of a function on an arbitrary tree: run an
interpolation engine on the whole domain, and wherever the engine does not
converge below the bond cap, fix the next site in a given order and retry on
each child region. The result is a `PartitionedTreeTN` with disjoint patches.
This is the producer of every patch that later milestones measure or consume,
so it favours correctness and determinism over speed; parallel execution is
M7 and split-site selection beyond a fixed order is M5.

## Findings that shape the design

Verified against the M1 branch state.

- The M1 contract (`tensor4all_treetn::interpolation`) returns an outcome whose
  network carries only the active site indices. `Converged` means the error
  criterion holds with the final rank strictly below the cap, so a patch with
  a nonzero value on several nodes never converges with a cap of one.
  All-zero initial samples return `AllSamplesZero`; non-finite initial
  samples return `Evaluator`. The contract requires at least one active site.
  M1 pivots cover the active sites only.
- `SubDomainTreeTN::new(tree, projector)` requires every projector index to be
  a site index of `tree` and masks projected indices eagerly; the
  crate-internal `from_masked_data` accepts data that is already masked.
- `tensor4all_core::outer_product` promotes mixed dtypes, and
  `IdxTensor::onehot` always builds `f64`. A one-hot factor must therefore be
  built from the patch scalar type `T`, not with `onehot`, or a patch of a
  different dtype becomes inhomogeneous and fails with `DTypeMismatch`.
- `tensor4all_core::CachedFunction` requires an infallible, `Send + Sync`
  point function and a `'static` batch function over `&[Vec<I>]`, and its keys
  stop at 1024 bits. It cannot wrap a fallible column-major evaluator, and its
  entries can only be cleared all at once.
- `PartitionedTreeTN` treats an absent patch as zero and requires disjoint
  projectors, not coverage; `from_subdomains(vec![])` is valid and later
  operations on an empty partition return `Empty`. `from_subdomains` checks all
  patch pairs for overlap.
- `PatchingOptions::patch_order` in the same crate accepts a partial order.
- `tensor4all-partitionedtreetn` has no random-number dependency. The workspace
  provides `rand` and `rand_chacha`; REPOSITORY_RULES.md requires named RNG
  algorithms in seed-based production code and a caller-owned `&mut R` API for
  randomized algorithms.
- Design lineage and provenance: the chain driver
  `tensor4all-partitionedtt::adaptiveinterpolate` states that its queue, split
  flow, and pivot recycling derive from TCIAlgorithms.jl. The approved records
  ([partitioned-treetn.md](./partitioned-treetn.md), the M1 seam record) state
  that the M2 driver derives its queue from that lineage and carries the
  derivation notice and `LICENSE-TCIALGORITHMS-MIT`. The chain crate is not a
  verification baseline.

## Public surface

A new module `tensor4all_partitionedtreetn::adaptive_interpolation`:

```rust
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PatchedInterpolationOptions {
    pub rtol: f64,
    pub reference_scale: Option<f64>,
    pub max_bond_dim: usize,
    pub patch_order: Vec<DynIndex>,
    pub n_initial_pivots: usize,
    pub recycle_pivots: bool,
    pub seed: u64,
    pub max_patches: Option<usize>,
}
// Default plus `with_*` builders; fields documented as below.

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PatchRecord {
    pub projector: Projector,
    pub termination: InterpolationTermination,
    pub error_estimate: f64,
    pub max_sample_magnitude: f64,
    pub max_bond_dim: usize,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PatchedInterpolationReport {
    pub reference_scale: f64,
    pub accepted: Vec<PatchRecord>,
    pub zero_projectors: Vec<Projector>,
    pub splits: usize,
    pub function_evaluations: usize,
    pub cache_hits: usize,
}

#[derive(Debug)]
pub struct PatchedInterpolationResult<V> {
    pub partition: PartitionedTreeTN<V>,
    pub report: PatchedInterpolationReport,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PatchedInterpolationError {
    InvalidInput { message: String },
    Interpolation { projector: Projector, source: InterpolationError },
    Partition { source: PartitionedTreeTNError },
    NoSplitIndexLeft { projector: Projector },
    ResourceLimit { resource: &'static str, limit: usize },
}

pub fn patched_interpolate<T, V, E, F>(
    engine: &E,
    topology: NodeNameNetwork<V>,
    node_sites: BTreeMap<V, Vec<DynIndex>>,
    initial_pivots: ColMajorArray<usize>,
    evaluate: F,
    options: &PatchedInterpolationOptions,
) -> Result<PatchedInterpolationResult<V>, PatchedInterpolationError>
where
    T: /* scalar bounds of IdxTensor data and magnitudes */,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T> + Sync,
    F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>> + Send + Sync;
```

The exact scalar bound set is the minimum needed for magnitudes, zero tests,
one-hot factors, and dense construction, fixed in implementation. `E: Sync`
and `F: Send + Sync` are required now so that M7 adds no bound.

### Options

| Field | Meaning | Default and guidance |
|---|---|---|
| `rtol` | Relative tolerance; the engine's absolute tolerance is `rtol * reference_scale`. `0` is allowed and splits until every patch is exact | `1e-8` |
| `reference_scale` | Scale for every patch. `None` pins it to the largest magnitude of the root patch's candidate samples, a sampled lower bound on `max |f|`; for localized functions this makes the tolerance tighter than intended, so passing a known scale is recommended | `None` |
| `max_bond_dim` | Bond cap per patch, at least 2 (a cap of one could only accept patches nonzero on a single node) | required; a smaller cap means more, smaller patches |
| `patch_order` | Sites fixed when a patch splits, in order; partial orders are allowed. Empty means the derived site order | empty |
| `n_initial_pivots` | Target number of distinct initial pivots per patch, at least 1 | `5` |
| `recycle_pivots` | Seed children with the parent's pivots | `false` |
| `seed` | Root seed | `0` |
| `max_patches` | Limit on processed patches (accepted, zero, fully fixed, and split alike); `Some(0)` is invalid | `None` |

`rtol` and `max_bond_dim` trade off: a tighter tolerance or a smaller cap
produces more patches.

### Error criterion

Acceptance uses the engine's sampled pivot-error criterion against
`rtol * reference_scale`. It is not a verified bound and makes no L2 claim;
rustdoc states this. M3 adds a user-selectable error norm with verified L2 as
the default (Decision 3); the options and report types are
`#[non_exhaustive]` so that M3 extends them without silently changing the
meaning of `rtol` and `reference_scale`.

## Algorithm

1. **Validate** the inputs: `patch_order` entries are distinct site indices of
   the problem (full identity); `rtol` is finite and nonnegative;
   `reference_scale`, if given, is finite and positive; `max_bond_dim >= 2`;
   `n_initial_pivots >= 1`; `max_patches` is not `Some(0)`; `initial_pivots`
   has one row per site and coordinates in range (zero columns are allowed).
   The site order is `InterpolationProblem::derive_site_order(&node_sites)`.
2. **Queue.** Start with the root patch (empty projector) and process patches
   in FIFO order, counting every processed patch against `max_patches`
   (`ResourceLimit` when exceeded).
3. **Cache.** Each patch owns an evaluation cache keyed by its active
   coordinates, encoded as a mixed-radix integer of the narrowest sufficient
   width; a domain too large for the widest supported key is `InvalidInput`.
   When a patch splits, its cache is partitioned among its children in one
   pass; when a patch is accepted or found to be zero, its cache is dropped.
   An evaluator error is returned immediately; nothing is cached for it.
4. **Candidates.** Keep, in order and without duplicates, the user pivots and
   (if enabled) the recycled parent pivots that are compatible with the
   patch's projector; then add random points inside the patch up to
   `n_initial_pivots`. The number of points in a patch is computed with
   saturating arithmetic (it only needs to be compared with the target).
   Random candidates use bounded rejection attempts and then fall back to the
   first unused points in column-major order.
5. **Sample checks.** Candidate samples are checked for finiteness in every
   patch before anything else; a non-finite sample is reported as
   `Interpolation { projector, source: InterpolationError::Evaluator }`.
6. **Reference scale.** If not given, it is pinned from the root patch's
   candidate samples and reused for every patch. A root with all candidate
   samples exactly zero cannot pin a scale and returns `InvalidInput` with the
   remedy to pass `reference_scale` or pivots in the support.
7. **Zero screening.** If every candidate sample of a patch is exactly zero,
   the patch is a zero patch: its projector is recorded in
   `zero_projectors` and it is omitted from the partition. Exact zero matches
   M1's `AllSamplesZero` rule (the chain lineage used a `1e-30` threshold).
   This is a finite-sampling policy; sparse functions need pivots in their
   support. An engine that still returns `AllSamplesZero` after screening is
   propagated as an `Interpolation` error.
8. **Exact small patches.** A patch with no active site is one value; a patch
   with exactly one active site has `d` values. Both are evaluated directly and
   built as a network with dimension-one links: the values on the node that
   carries the active site (or on the smallest node name if none), and a
   one-hot factor for every fixed site on its own node. No engine call.
9. **Interpolate.** Otherwise build an `InterpolationProblem` with the active
   sites, the candidates in active coordinates,
   `absolute_tolerance = rtol * reference_scale`, the bond cap, and the engine
   seed of the patch. The problem's evaluator inserts the fixed coordinates and
   goes through the patch cache.
10. **Accept or split.** `Converged` is accepted. Any other verdict splits the
    patch at the next site of `patch_order` that is not fixed yet, one child per
    coordinate; if no site of `patch_order` is left, the driver returns
    `NoSplitIndexLeft`. With `recycle_pivots`, the outcome's pivots (active
    sites only) are completed with the patch's fixed coordinates and passed to
    the children, which keep those compatible with their split coordinate.
11. **Re-embed.** An accepted outcome network gets every fixed site
    re-attached to its original node by an outer product with a one-hot vector
    built from `T`, then is wrapped with the crate-internal `from_masked_data`
    (the data is already masked). This helper is shared by all engines.
12. **Assemble** the accepted patches with a crate-internal constructor that
    skips the pairwise overlap check, because the queue produces disjoint
    projectors by construction; a debug assertion keeps the check in tests.
    An all-zero run returns an empty partition.

The partition stays in the eager (masked) form required by the current
`partitionedtreetn` invariant; M4 decides whether that changes.

## Randomness and determinism

The driver uses `ChaCha8Rng` (workspace `rand` and `rand_chacha`, added as
dependencies of `tensor4all-partitionedtreetn`). Each patch derives two
sub-seeds, one for candidate sampling and one for the engine, by a SplitMix64
mix of the root seed and the patch path, encoded as pairs of (position in the
derived site order, coordinate); the path never uses `DynIndex` IDs, so the
encoding survives the adaptive split sites of M5.

Exception to the caller-owned `&mut R` rule: a single caller stream would make
each patch's randomness depend on the processing order, so parallel execution
(M7) could not reproduce sequential results. The driver therefore offers only
the seed API and documents this exception in rustdoc.

For a fixed seed, a deterministic evaluator, and a deterministic engine, the
partition and report are identical across runs.

## Provenance

Following the approved records, the driver is derived from the TCIAlgorithms.jl
lineage through `tensor4all-partitionedtt`. The M2 PR:

- adds the derivation notice to the module header and copies
  `LICENSE-TCIALGORITHMS-MIT` into `crates/tensor4all-partitionedtreetn/`;
- adds a "Derived (MIT)" row for `tensor4all-partitionedtreetn` adaptive
  interpolation to `docs/PROVENANCE_AND_CITATION_POLICY.md` and rewrites the
  statement there that the crate does not contain adaptive interpolation.

## Public-surface updates in the M2 PR

- `crates/tensor4all-partitionedtreetn/README.md` and the crate docs in
  `src/lib.rs`, which say the crate provides no TCI or sampled-zero inference.
- `docs/book/src/guides/partitioned-treetn.md` (same statement; add the entry
  point) and `docs/book/src/architecture.md`.
- `skills/use-tensor4all-rs/SKILL.md` and `references/crates.md`, and
  `llms.txt`.

## Tests

- A driver-local test engine in `tensor4all-partitionedtreetn`: it builds the
  exact network from dense samples (supporting nodes without active sites) and
  reports `BondCapReached` above a configured rank, so splitting is tested
  without a real engine.
- TreeTCI end to end through a path-only dev-dependency on
  `tensor4all-treetci` (as the crate already does for
  `tensor4all-quanticstransform`).
- Topologies: a chain, a branched tree with a node of degree three or more,
  a node with several sites, a node without sites in the caller's topology,
  and a single-node topology. Splits at a leaf, an internal node, the junction,
  and a multi-site node, including a node whose sites all become fixed.
- A function with localized features on a small quantics grid whose monolithic
  rank exceeds the cap; the assembled partition is compared with a dense
  reference (materialize once, subtract, `maxabs`) with the bound
  `10 * rtol * reference_scale`, recorded as a test constant.
- A vanishing region: zero projectors are reported, the accepted and zero
  projectors are disjoint, and together they cover the domain.
- The exact small-patch paths, a complex scalar type with homogeneous
  re-embedded dtypes, recycling on and off, and determinism across two runs
  compared by projector key.
- Evaluation counts show no duplicate evaluation of a point.
- Errors: `NoSplitIndexLeft` (partial order), `ResourceLimit`, an unpinnable
  scale, a driver-side evaluator failure and a non-finite sample, a domain too
  large for the cache key, and each `InvalidInput` branch.
- Index identity: sites sharing an ID but differing in prime level or tags.
- Rustdoc: runnable, asserted examples for every public item and `# Errors`
  naming the variants.

## Non-goals

- Parallel execution (M7), error norms and verified bounds (M3), the patch
  representation decision (M4), and split strategies other than a fixed order
  (M5).
- Retiring or changing `tensor4all-partitionedtt`.

## Open question for review

1. Names: `adaptive_interpolation`, `patched_interpolate`, and the
   `Patched*` types, chosen to avoid `AdaptiveInterpolationResult` in the
   chain crate.
