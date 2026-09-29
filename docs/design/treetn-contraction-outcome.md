# TreeTN contraction outcome and early abort

## Status

Proposal for review. Item P1 of
[tree-patching-m1-prerequisites.md](./tree-patching-m1-prerequisites.md).
It adds public API to `tensor4all-treetn` and must be approved before
implementation.

## Problem

`tensor4all_treetn::contraction::contract` returns only a `TreeTN`. Callers
cannot learn the realized rank of each output edge, and cannot stop a
contraction whose output is already known to be too large.

`tensor4all-partitionedtreetn::contract_adaptive` needs both. It contracts
every compatible pair of a group, compares the maximum bond dimension with the
patch cap afterwards, and, if the group saturates, discards the results and
recomputes them on projected children. The discarded work is paid in full
even when the first processed edge already reached the cap.

## Goals

- Report the realized rank of every output edge of a contraction.
- Optionally stop a contraction as soon as one edge reaches a caller-given
  rank threshold, and say which edge did.
- Keep `contract` unchanged for existing callers.

## Non-goals

- Exact detection of whether a rank cap discarded nonzero weight. That needs
  core `factorize` to expose the discarded tail and is deferred to a core
  issue (P1 item 4 of the M1 plan).
- Early abort for Fit. The open draft PR #656 restructures `fit.rs`; Fit gets
  the post-hoc behavior described below until that PR is resolved.
- Outcome variants of `partial_contract` and `hadamard`. Patched element-wise
  products are an M6 item; they will reuse the entry point below.

## Public surface

```rust
/// Realized rank of one output edge.
pub struct EdgeRank<V> {
    pub source: V,
    pub destination: V,
    pub rank: usize,
}

/// Ranks observed during a contraction, in processing order.
pub struct ContractionReport<V> {
    pub edge_ranks: Vec<EdgeRank<V>>,
}

pub enum ContractionOutcome<T, V> {
    /// The contraction finished; `report` lists every output edge.
    Completed { network: TreeTN<T, V>, report: ContractionReport<V> },
    /// An edge reached `abort_at_rank`; `report` lists the edges processed
    /// up to and including `edge`.
    Saturated { edge: EdgeRank<V>, report: ContractionReport<V> },
}

pub fn contract_with_outcome<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    options: ContractionOptions,
    abort_at_rank: Option<usize>,
) -> Result<ContractionOutcome<T, V>, TreeTNOperationError>;
```

- `abort_at_rank` is separate from `ContractionOptions::max_bond_dim`. The
  truncation cap limits what the output may keep; the abort threshold is the
  caller's probe budget. `contract_adaptive` sets the threshold to the patch
  cap while its contraction options may carry a different (or no) cap.
- Saturation means realized rank `>= abort_at_rank`. Equality counts, matching
  `partitionedtreetn`: a capped probe cannot distinguish an exact rank equal
  to the cap from a truncated one.
- `abort_at_rank = Some(0)` is rejected as invalid options. `None` never
  aborts, and the outcome is always `Completed`.
- `ContractionOptions` is unchanged, so no existing struct literal breaks.
  `contract` delegates to `contract_with_outcome(..., None)` and returns the
  network.
- The new types derive `Debug` and `Clone`. `tensor4all-treetn` has no
  documented `#[non_exhaustive]` policy (existing `ContractionMethod` is
  exhaustive), so marking is an open question below.

Alternative considered: an `abort_at_rank` field on `ContractionOptions`.
Rejected because `contract` would then need a rule for what to return when it
aborts, and every existing struct literal of the options would break.

## Method coverage

| Method | Report | Early abort |
|---|---|---|
| Zipup, chain path | Rank of each factorization in sweep order | Yes: stop after the first factorization whose rank reaches the threshold |
| Zipup, tree path | Rank of each factorization, leaves toward center | Yes: same rule in the post-order edge loop |
| Src | Realized bond dimensions of the finished network | Post-hoc: evaluated after completion |
| Naive | Realized bond dimensions of the finished network | Post-hoc |
| Fit | Realized bond dimensions of the finished network | Post-hoc (deferred, see non-goals) |

Post-hoc methods return `Saturated` with the same meaning but without saving
work, so callers see one contract regardless of method. Zip-up records ranks
from the `rank` field of each core `FactorizeResult`; no extra factorization
or norm is computed.

## Adoption in `partitionedtreetn`

- `SubDomainTreeTN` gains a crate-internal contraction that returns the
  outcome and keeps projector bookkeeping unchanged.
- In `contract_group_project_first`, with `PatchSplitStrategy::Sequential`,
  contributions are computed with `abort_at_rank = patching cap`. The first
  `Saturated` outcome stops contraction of the remaining pairs and the group
  splits immediately, extending the #788 shortcut to skip the remaining
  contractions as well.
- With `ExactParameterGain` the split decision needs the full group sum, so
  contributions are contracted without abort, as today.

## Tests

- `contract_with_outcome(..., None)` equals `contract` on chain and branched
  trees (dense comparison) and reports one rank per output edge that matches
  the returned network's bond dimensions.
- Zip-up with a threshold reached on an early edge returns `Saturated`, names
  that edge, and processes fewer edges than a full run (asserted through the
  report length).
- Equality at the threshold counts as saturation; a threshold above every
  realized rank yields `Completed`.
- Src and Naive report post-hoc saturation consistently.
- `Some(0)` is rejected with a typed error.
- `partitionedtreetn`: a Sequential group stops contracting after the first
  saturated contribution (counter), with unchanged dense results; the
  `ExactParameterGain` path is unchanged.

## Open questions for review

1. Name: `abort_at_rank` versus `rank_budget` or `saturation_threshold`.
2. Whether `Saturated` should also return partially built tensors. This
   proposal returns none, because every current caller discards them.
3. Whether `ContractionOutcome` should be `#[non_exhaustive]` so that later
   variants (for example a Fit-specific outcome) are not breaking changes.
