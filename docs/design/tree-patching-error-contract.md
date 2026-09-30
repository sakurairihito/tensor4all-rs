# Tree patching error contract

## Status

Proposal for milestone M3 of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md),
revised after one independent review. It changes the public API and the
acceptance semantics of the M2 driver
([tree-pqtci-driver.md](./tree-pqtci-driver.md)) and must be approved before
implementation. It covers the interpolation side of M3 in full. The
patched-algebra side (the optional global-budget mode for addition and
contraction) is scoped here and gets its own design record before
implementation ([Patched algebra](#patched-algebra-m3b)).

Implementation has two prerequisites, both stated below: frozen M2 golden
digests committed before the refactor ([Tests](#tests)), and an
evaluation-order audit of the TreeTN evaluator that the measurement uses
([Determinism](#determinism)).

The decisions that depend on the user are collected under
[Open questions](#open-questions-for-the-user). No measurement was run for
this record; the defaults that need data name the measurement and the
milestone that takes it.

## Goal

Give `patched_interpolate` one accuracy requirement with a measured, reported
error, following roadmap Decision 3: the L2 error is the default, the
sampled max-norm criterion of M2 stays available as an explicit choice, and a
norm without an implementation is a typed placeholder that fails before any
evaluation.

Concretely:

1. define what "verified L2" can and cannot mean for a black-box function;
2. split one global L2 allowance into per-patch allowances that stay valid
   when patches split;
3. add a user-selectable error norm to the options and report types without
   silently changing the meaning of an existing field;
4. add a verification step to the driver's acceptance flow that keeps the M2
   determinism guarantees and the per-patch evaluation cache;
5. keep the M1 engine contract and every engine unchanged.

## Findings that shape the design

Verified on `feat/tree-adaptive-patching` at `caf164f0`. API names were
checked against the `cargo run -p xtask --release -- api-dump` inventory
(`target/api-dump/*.md`) and the listed source.

- **M2 acceptance.** The driver accepts a patch only when the engine returns
  `InterpolationTermination::Converged` with a bond dimension strictly below
  the cap; any other verdict splits it
  (`crates/tensor4all-partitionedtreetn/src/adaptive_interpolation.rs`,
  lines 1029-1097). The engine's absolute tolerance is
  `rtol * reference_scale` (lines 1014-1021). The module documentation states
  that this is not a verified bound and makes no L2 claim (lines 43-47).
  `NoSplitIndexLeft` means "did not converge and no site of `patch_order` is
  left" (lines 595-604).
- **M2 reference scale.** Without `reference_scale` the driver pins the scale
  once, from the largest magnitude among the root patch's candidate samples
  (user pivots, recycled pivots, and random points), or from the exact values
  of an exact root (lines 991-1005, 1120-1122). It is a sampled lower bound on
  `max |f|`, in max-norm units. An all-zero exact root returns an empty
  partition with `reference_scale = 0`.
- **What the engine estimate is.** TreeTCI reports the maximum over edges of
  the last LU pivot error of each edge update
  (`crates/tensor4all-treetci/src/state.rs:193`, `update.rs:107-108`); the
  M1 engine runs with `normalize_error = false`, so the raw value is compared
  with the absolute tolerance. Its global pivot search looks for large
  `|f(x) - tt(x)|` from random starts with local coordinate optimization
  (`globalpivot.rs`, module header). Both are max-type quantities on points
  that the engine chose; neither estimates an L2 norm, and the points are not
  independent of the approximation.
- **Initial pivots in TreeTCI.** `TreeTCI2::add_global_pivots` adds every
  pivot's projection to the pivot sets of both subregions of every edge
  (`treetci/src/state.rs:117-160`). The M1 contract says nothing about how the
  number of initial pivots relates to the bond cap.
- **M1 contract.** `InterpolationOutcome::error_estimate` is the engine's raw
  error estimate, the quantity compared with the absolute tolerance
  (`crates/tensor4all-treetn/src/interpolation.rs`, lines 630-651). The
  outcome network carries only the active sites. The contract has no norm
  parameter, and the M1 record states that the driver computes the absolute
  tolerance from a reference scale "pinned for all patches in M3".
- **Exact small patches.** A patch with at most one active site is evaluated
  on all its points and built from those values with dimension-one links and
  one-hot factors; its record has `error_estimate = 0`
  (`adaptive_interpolation.rs`, lines 1102-1136).
- **Evaluation cache.** Every function value reaches the driver through the
  per-patch cache, which checks count and finiteness and is partitioned among
  the children in one pass on a split (`adaptive_interpolation/cache.rs`,
  `PatchSampler::sample` at line 185, `PatchCache::split` at line 127).
- **Randomness in the driver.** The driver implements SplitMix64 with
  Lemire's unbiased bounded draw. Each patch absorbs its path into a state and
  derives its candidate and engine sub-seeds with the stream selectors
  `CANDIDATE_STREAM` and `ENGINE_STREAM` (`adaptive_interpolation/sampling.rs`,
  lines 17-27 and `patch_seeds` at line 79). The driver offers only the seed
  API and documents the exception to the caller-owned `&mut R` rule.
- **Randomness outside the driver: index IDs.** `DynIndex` IDs come from
  `generate_id()`, a per-thread, unseeded `rand::rng()`
  (`tensor4all-core/src/defaults/index.rs:413-427`), and
  `sort_indices_deterministic` breaks ties of equal dimension and prime level
  by `id()` (`tensor4all-core/src/index_like.rs:333-344`). Earlier
  run-to-run differences in SRC contraction came from exactly this
  (comments at `tensor4all-treetn/src/treetn/contraction/src_probe.rs:576-578,
  786` and `src_tree.rs:622-625`). Every engine outcome carries fresh bond
  IDs.
- **Norms of networks.** `SubDomainTreeTN::norm_squared` clones and
  canonicalizes the patch. `PartitionedTreeTN` stores patches in a
  `HashMap` (`partitioned_tree_tn.rs:45`) and `norm_squared` sums in its
  iteration order (lines 333-337). Neither is bitwise reproducible across
  runs (#791 and the hash order), which matters for any value that feeds a
  decision.
- **Point evaluation of a network.** `TreeTNEvaluator::evaluate_batched`
  passes nodes without requested sites through unchanged
  (`treetn/evaluator.rs`, lines 294-345), but evaluates every point through a
  temporary network built with `TreeTN::from_tensors` and
  `contract_to_tensor`, the path of
  [#791](https://github.com/tensor4all/tensor4all-rs/issues/791).
  `TreeTNCachedEvaluator::evaluate_batched_typed::<T>`
  (`treetn/cached_evaluator.rs:2052`):
  - iterates nodes in sorted name order (lines 1452-1453) and treats a node
    without requested sites as having no entries (lines 764-768, 2329-2333);
    no test covers a site-free node;
  - uses its raw kernels only when every node has exactly one requested site
    (`can_use_raw_messages`, lines 2513-2557). A tree with a multi-site node
    or a site-free node, such as the M2 `quantics_tree` test tree, takes the
    generic `IdxTensor` path for every message;
  - on the generic path mints a fresh random-ID index per message
    computation (lines 4388, 4759, 5592) and contracts through core
    `IdxTensor` contraction;
  - in the chain kernel chooses BLAS or a scalar loop from the composition of
    the batch (lines 2752-2775), and keeps message caches across calls, so a
    value can depend at rounding level on the batch contents, the call
    history, and the hint.

  A complete search of `cached_evaluator.rs` finds no direct use of `id()`
  or `sort_indices_deterministic`; whether the core contraction it calls
  orders legs independently of IDs has not been verified.
- **Reconstruction precedent.** Reconstruction supports only the unweighted
  discrete L2 norm and uses the allowance `max(atol, rtol * reference_scale)`
  with `ReconstructionTolerance { rtol, atol }` (default `rtol = 1e-6`), where
  `reference_scale` is an L2 norm (`reconstruction/mod.rs`, lines 37-68 and
  187). It measures a residual as the norm of an explicit difference network,
  `source.axpby(1, candidate, -1)` followed by `norm`
  (`reconstruction/engine.rs:366-367`), and combines disjoint regions with
  `hypot`. It derives its reference from the target and has no public
  override.
- **Algebra precedent.** `PatchingOptions::cutoff` allocates a local
  discarded-weight threshold in proportion to patch volume,
  `cutoff * ||F||^2 * volume_p / total_volume` (`patching.rs:61-68`), and is
  documented as best effort with no whole-network bound (lines 69-115); that
  was a recorded maintainer decision in
  [partitioned-treetn.md](./partitioned-treetn.md) (review of #655).
  `PatchingOptions` is not `#[non_exhaustive]`.
- **Extensibility.** `PatchedInterpolationOptions`, `PatchRecord`,
  `PatchedInterpolationReport`, and `PatchedInterpolationError` are
  `#[non_exhaustive]`; `PatchedInterpolationResult` is not. `Projector`
  implements `Debug`, `Clone`, `PartialEq`, and `Eq`.
- **Dense references.** `PartitionedTreeTN::to_treetn`,
  `TreeTN::contract_to_tensor` / `to_dense`, and `IdxTensor::{from_dense, sub,
  maxabs, norm}` exist for small dense test comparisons. REPOSITORY_RULES.md
  allows dense or exhaustive work in production only behind an explicit,
  caller-visible size limit.

## What "verified L2" means

### The quantity

The domain `X` is the product of all site dimensions and `|X|` its number of
points. For a function `g` on a subset `P` of `X`,

```text
||g||_P^2 = sum over x in P of |g(x)|^2          (unweighted discrete L2)
ms_P(g)   = ||g||_P^2 / |P|,  rms_P(g) = sqrt(ms_P(g))
```

This is the norm of `TreeTN::norm`, `PartitionedTreeTN::norm`, and
reconstruction. For a uniform quantics grid, `||g||_X^2 * V / |X|` is a
Riemann sum of the continuum `||g||^2` over the grid domain of volume `V`, so
the discrete norm approximates the continuum norm times `sqrt(|X| / V)`, with
the discretization error of that sum.

The target quantity is the **absolute L2 error of the whole partition against
`f` over the whole domain**,

```text
E^2 = ||f - f~||_X^2 = sum over accepted patches P of ||f - f~_P||_P^2
                     + sum over zero patches Z   of ||f||_Z^2 ,
```

where `f~` is the returned partition. The equality is exact because accepted
and zero patches are disjoint and together cover `X` (an M2 report invariant).
Zero patches are part of the error: omitting a patch is an approximation by
zero, and it is charged like any other.

A relative statement needs `||f||`, which is not computable. It is bounded
from the computable side instead: `||f~||` is exact up to rounding from the
patch networks, and by the triangle inequality `||f|| >= ||f~|| - E`, so

```text
E / ||f|| <= E / (||f~|| - E)        whenever ||f~|| > E.
```

This a-posteriori relative bound holds whenever `E` is bounded, independently
of any reference norm, and the report states it (see
[Records and report](#records-and-report)).

### What can be computed without the dense function

| Quantity | How | Guarantee | Cost |
|---|---|---|---|
| `||f~_P||_P`, `||f~||_X` | `SubDomainTreeTN::norm_squared`, summed by the driver in canonical path order | exact up to rounding | one canonicalization per patch, no evaluations of `f` |
| `f~(x)` at chosen points | TreeTN batch evaluators | exact up to rounding | one network evaluation per point |
| `||f - f~_P||_P` for a small patch | evaluate `f` and `f~_P` at every point of `P` | exact up to rounding (a certificate) | `|P|` evaluations of `f` (fewer with cache hits), `|P|` network evaluations |
| `||f - f~_P||_P` for a large patch | Monte Carlo on `n` fresh uniform points of `P` | a statistical estimate, see below; no bound | `n` evaluations of `f` (fewer with cache hits), `n` network evaluations |
| `||f||_X` | exactly only by evaluating all of `X`; otherwise a Monte Carlo estimate | a heavy-tailed estimate, see [Reference norm](#reference-norm) | `n` evaluations |

A held-out sample fixed in advance is the Monte Carlo row with a fixed point
set: its statement is exact on that set and statistical elsewhere.

For a black-box `f`, finitely many samples cannot bound `||f - f~_P||_P`: a
residual supported on a fraction `rho` of `P` is missed by all `n` uniform
samples with probability `(1 - rho)^n`, about `exp(-n rho)`, and its size is
unconstrained. Any claim of an L2 bound for a large patch is statistical,
never worst case.

### What a sampled measurement guarantees, before selection

Let `x_1, ..., x_n` be drawn uniformly and independently from `P` (with
replacement), independently of `f~_P` **and of any decision that uses them**,
and let `r = f - f~_P`.

- `m = (1/n) sum |r(x_i)|^2` is an unbiased estimate of `ms_P(r)`. Its
  standard error is `s / sqrt(n)` with `s^2` the sample variance of
  `|r(x_i)|^2` (so `n >= 2`). A confidence interval from it is asymptotic
  (central limit theorem) and can be badly optimistic when the residual is
  concentrated.
- Distribution-free: by exchangeability, a further uniform point exceeds
  `max_i |r(x_i)|` with probability at most `1 / (n + 1)`, so the maximum
  sampled residual is exceeded, in expectation, on at most a fraction
  `1 / (n + 1)` of the patch. This is a quantile statement, not an L2 bound.
- Exact on the sampled set: the residual at every sampled point is known.
- A standard error of zero carries no information. It arises whenever all
  sampled `|r|^2` are equal, in particular all zero, which is also what a
  residual concentrated on an unsampled set produces.

Independence of the choice of points from the approximation is essential: a
pivot of the engine has a residual near zero by construction. Values served
from the evaluation cache are the same values of `f`, so cache hits do not
break independence; only the choice of points matters.

### What it does not guarantee after selection

The driver accepts a patch when its measurement is small, using the same
sample. For an accepted patch the measurement is conditioned on acceptance,
and none of the statements above hold for it: `m` is not unbiased, and the
`1 / (n + 1)` statement does not hold for the reported maximum.

Counterexample: let `|r| = M` on a fraction `rho` of `P` and `r = 0`
elsewhere. The patch is accepted whenever no sample hits the support, with
probability `(1 - rho)^n`, and is then reported with `m = 0`, standard error
`0`, and maximum `0`, while the true error `sqrt(rho |P|) M` is arbitrarily
large. With `rho = 2/n` the acceptance probability is about `exp(-2)`, 0.13
for `n = 64`, and the true fraction above the reported maximum is `2/n`, about
twice the pre-selection bound.

- The bias depends on how concentrated the residual is, not on the distance
  to the allowance: it is small only for a residual spread over the patch and
  largest for a concentrated one, however large its norm.
- **Multiple testing.** A rerun after a failed verification may return a
  nearly unchanged approximation and draws a fresh sample, and each split
  gives the children new chances. `k` independent samples of an unchanged
  approximation accept the patch above with probability
  `1 - (1 - (1 - rho)^n)^k`.
- An upper confidence bound `m + z * SE` does not help against this: in the
  counterexample it is zero.

Valid statements about accepted patches need an **independent audit sample**,
drawn after the acceptance decision from a stream that no decision uses. The
audit estimate of an accepted patch is unbiased and carries the
`1 / (n + 1)` statement; it is still a statistical estimate, not a bound. The
proposal draws one by default for every sampled contribution
([open question 5](#open-questions-for-the-user)).

### Definition

A patch error is **verified** when it has been measured from values of `f` at
points chosen independently of the approximation, by one of three methods,
which the report names per patch:

- **Exact**: the patch was built from all its values (the M2 exact small-patch
  path); its error is zero.
- **Exhaustive**: the residual was evaluated at every point of the patch; the
  measured error is exact up to floating-point rounding. This is a
  certificate, unaffected by selection.
- **Sampled**: the residual was evaluated at `n` fresh uniform points. The
  acceptance measurement is only a decision statistic. With an audit, the
  audit measurement is an unbiased estimate with a standard error; without
  one, the report carries no estimate for the patch.

The run's global error is **certified** when every contribution is exact or
exhaustive. `certified` is a statement about the absolute error, `E <= delta`
up to a named rounding margin, where `delta` is the allowance actually used.
When the reference norm was estimated, `delta` is itself random (see
[Reference norm](#reference-norm)), and only the a-posteriori relative bound
`E / (||f~|| - E)` is free of it. Otherwise the report says whether the global
number is an audited estimate, with its standard error and the certified
fraction of the domain, or only a sum of acceptance statistics.

Whether this definition is what the user means by "verified" is
[open question 1](#open-questions-for-the-user).

## Budget

### Global allowance

The allowance is kept in root-mean-square units, so that no quantity scales
with `|X|` (which can reach the `f64` range, while `|X| * value^2` would
overflow much earlier):

```text
S_rms = S / sqrt(|X|)                          RMS value of the reference
tau   = max(atol / sqrt(|X|), rtol * S_rms)    RMS allowance
delta = sqrt(|X|) * tau = max(atol, rtol * S)  the same allowance as an L2 norm
```

Here `S` is a reference L2 norm of `f` and `S_rms` its RMS value; `delta` has
the form of the reconstruction allowance. `tau` is pinned once, before any
patch is accepted, and is the same for every patch, so acceptance does not
depend on processing order, as M7 requires. Values in L2 units (`delta`, `S`,
`E`, `||f~||`) are derived from RMS values for the report and are `None` when
the product overflows.

`rtol = atol = 0` is allowed and means `tau = 0`: a patch is accepted only if
every measured residual is exactly zero, so in practice the driver splits
down to exact patches (bounded by `max_patches`), as `rtol = 0` does in M2.

### Per-patch allowance

The allowance is split in proportion to patch volume, in squared norm, the
same rule as the volume-proportional local `cutoff` of `PatchingOptions`
(`patching.rs:61-68`):

```text
||f - f~_P||_P^2 <= delta^2 * |P| / |X|     equivalently     rms_P(f - f~_P) <= tau
```

Summing over the disjoint accepted and zero patches, whose volumes add up to
`|X|`, gives `E <= delta`. The comparison is a root-mean-square test against
one constant.

- **Splits.** A patch's allowance depends on its volume only. The children of
  a split have volumes that sum to the parent's, so their squared allowances
  sum to the parent's: splitting never reallocates or borrows budget. A
  rejected parent approximation is discarded, so its error is never charged
  (as in reconstruction, the children replace it).
- **Zero patches** are charged `||f||_Z^2` against `delta^2 |Z| / |X|`.
- **Unused budget** of accurate patches is not redistributed; the guarantee
  holds without it, and redistribution would make acceptance depend on order.
- **Rounding.** `|P|` and `|X|` are inexact in `f64` above `2^53`, and the sum
  over patches rounds. Global inequalities hold up to a named relative margin
  `GLOBAL_ROUNDING_MARGIN`, fixed at implementation and used by the report and
  the tests.

Rejected: an equal split by patch count. The final count is unknown while the
queue runs, so it needs reallocation and makes acceptance order-dependent.
Considered: a split proportional to each patch's own norm
([open question 2](#open-questions-for-the-user)).

### Engine tolerance

The engine still receives one absolute tolerance, now `tau`. If the engine's
pointwise criterion held everywhere on a patch, `|r| <= tau` would imply the
patch's allowance; verification checks what the engine did not. Whether a
smaller engine tolerance (a factor below one) lowers the total cost by avoiding
failed verifications is a measurement for M9 on M2 patches, not a knob in M3.

### Rounding floor

`tau` is tied to `rms(f)`, not `max |f|`. For a function localized on a
fraction `rho` of the domain, `rms(f)` is about `sqrt(rho) max |f|`: with the
default `rtol = 1e-8` and `rho = 2^-40`, `tau` is about `1e-14 max |f|`, which
is the rounding level of evaluating `f` and of the network contraction near
the peak. Verification of a peak patch can then never pass, and the patch
would split down to exact patches until `max_patches` is hit. A Monte Carlo
reference that underestimates `S` makes this worse.

The driver detects it instead of splitting blindly. With `eps` the machine
epsilon of the scalar type, `A_P` the largest magnitude among the patch's
measured values of `f` and the engine's `max_sample_magnitude`, and a named
constant `ROUNDING_FLOOR_FACTOR` fixed at implementation (checked in M9):

- at the root, after pinning `tau`: if `0 < tau < ROUNDING_FLOOR_FACTOR * eps *
  A_root`, return `ToleranceBelowRounding` before interpolating;
- when a verification fails: if `tau < ROUNDING_FLOOR_FACTOR * eps * A_P`,
  return `ToleranceBelowRounding` for that patch instead of retrying or
  splitting.

The remedy is to raise `rtol`, set `atol`, or give a reference norm
(`L2Reference::Given`) if the estimate was too small. The check is skipped
for `tau = 0`, which requests exact patches explicitly.

### Reference norm

The two norms need references with different units:

| Norm | Reference | Units | Engine tolerance |
|---|---|---|---|
| L2 | `S`, an L2 norm of `f` | `sqrt(|X|)` times a function value | `tau`, about `rtol * rms(f)` |
| sampled max (M2) | `max_reference`, a sampled lower bound on `max |f|` | a function value | `max(atol, rtol * max_reference)` |

Because `rms(f) <= max |f|`, the L2 engine tolerance is never looser than the
max-norm one for the same `rtol` and exact references, and for a localized
function it is tighter by about `sqrt(rho)`. A max-norm reference passed as an
L2 reference would be wrong by `sqrt(|X|)`. The options and the report keep
the two references in separate, typed places (`ErrorNorm` variants and
`NormReport` variants) with different names, and the SampledMax field is
renamed `max_reference` so that it does not collide with reconstruction's
`reference_scale`, which is an L2 norm.

Where `S` comes from:

1. `L2Reference::Given(s)`: the caller's L2 norm (for a uniform quantics grid,
   approximately the continuum norm times `sqrt(|X| / V)`);
2. not needed when `rtol = 0` (then `tau = atol / sqrt(|X|)`);
3. the exact values of the root when the root is an exact small patch; an
   all-zero exact root returns an empty partition, as in M2, whatever the
   tolerance;
4. `L2Reference::MonteCarlo`, an explicit opt-in: `S_rms^2 = mean |f(x_i)|^2`
   over `verification.samples` uniform root points of a dedicated stream,
   pinned once and reported with its standard error;
5. otherwise `L2Reference::Required`, the default, fails before any
   evaluation with the remedy to give a reference norm, set `rtol = 0` with
   `atol`, or opt into the estimate.

The estimate is not a safe default, because it can loosen the tolerance
relative to `||f||`, not only tighten it. For a function of height `H` on a
fraction `rho = 1e-4` and `n = 64`: with probability about 0.994 no sample
hits the support and `S = 0` (a failure with `atol = 0`, or `delta = atol`);
with probability about 0.006 exactly one sample hits, and
`S_rms = H / 8` while the true value is `H / 100`, so `delta` is about 12
times too large and the run meets a tolerance 12 times looser than requested
without any sign of it except the a-posteriori relative bound. The default is
[open question 3](#open-questions-for-the-user); the previous revision of this
record proposed the estimate as the default.

## Public surface

All new items live in `tensor4all-partitionedtreetn`. Names are proposals.

```rust
/// The norm in which an accuracy requirement is stated and measured.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ErrorNorm {
    /// Unweighted discrete L2 norm over the whole domain (default).
    #[non_exhaustive]
    L2 { reference: L2Reference },
    /// The engine's own sampled criterion against a max-norm reference: the
    /// M2 behavior, with no measurement by the driver.
    #[non_exhaustive]
    SampledMax { max_reference: Option<f64> },
    /// Placeholder: a verified maximum norm over the whole domain.
    MaxAbs,
    /// Placeholder: an L2 norm with caller-supplied weights.
    WeightedL2,
}
// Constructors: ErrorNorm::l2(L2Reference), ErrorNorm::sampled_max(),
// ErrorNorm::sampled_max_with_reference(scale).
// Default: ErrorNorm::L2 { reference: L2Reference::Required }.

/// Where the L2 reference norm comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum L2Reference {
    /// The caller's L2 norm of the function, finite and positive.
    Given(f64),
    /// Estimate it from uniform root samples (opt-in; see "Reference norm").
    MonteCarlo,
    /// No reference: allowed when rtol = 0 or the root is exact, otherwise
    /// an error before any evaluation (default).
    Required,
}

/// Accuracy requirement in the units of the selected norm:
/// allowance = max(atol, rtol * reference).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorTolerance { pub rtol: f64, pub atol: f64 }
// Default: rtol = 1e-8 (the M2 default value), atol = 0.

/// How the driver measures accepted and zero patches under ErrorNorm::L2.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct VerificationOptions {
    /// Fresh uniform points per sampled measurement, at least 2. Default 64.
    pub samples: usize,
    /// A patch with at most max(max_exhaustive_points, samples) points is
    /// measured exhaustively. Bounds the exhaustive work and cache growth per
    /// patch. Default 1024; 0 means only patches with at most `samples`
    /// points.
    pub max_exhaustive_points: usize,
    /// Engine reruns of one patch after a failed verification, before the
    /// patch splits. Default 1.
    pub retries: usize,
    /// Draw an independent audit sample for every sampled contribution after
    /// its acceptance decision. Default true.
    pub audit: bool,
}
// Default, VerificationOptions::new() (same values), and with_samples,
// with_max_exhaustive_points, with_retries, with_audit builders.
```

`ErrorNorm` and `ErrorTolerance` are crate-level types because the
patched-algebra mode of M3b uses the same pair. The variants with data are
`#[non_exhaustive]` so that fields can be added; they are built through the
constructors. Placeholders carry no data; a placeholder that gains an
implementation may gain fields, a deliberate breaking change at that time.

`ErrorTolerance` is deliberately a plain struct without `#[non_exhaustive]`:
its two fields are the whole contract (`max(atol, rtol * reference)`), callers
build it with a struct literal as they build `ReconstructionTolerance`, and
the planned merge with that type in M3b keeps exactly these fields. A future
field would be a deliberate breaking change.

An enum, not a trait: the driver must know how a norm combines over disjoint
patches (Euclidean for L2, maximum for a max-norm), how its allowance splits,
and how it is measured. A trait would have to expose all three before a second
implemented norm exists to shape it. A trait can replace the enum when one
does.

### Options

| M2 field | M3 | Meaning |
|---|---|---|
| `rtol` | moved to `tolerance: ErrorTolerance` | relative to the reference of the selected norm |
| `reference_scale` | removed; `ErrorNorm::SampledMax { max_reference }` | unchanged inside that variant, renamed |
| (new) | `error_norm: ErrorNorm` | default `L2 { reference: Required }` |
| (new) | `tolerance.atol` | absolute floor of the allowance, default `0` |
| (new) | `verification: VerificationOptions` | used only by `L2`; validated under every norm |
| `max_bond_dim`, `patch_order`, `n_initial_pivots`, `recycle_pivots`, `seed`, `max_patches` | unchanged | unchanged |

Builders: `with_error_norm`, `with_tolerance`, and `with_verification` are
added; `with_rtol` and `with_reference_scale` are removed, so every caller
that set either one fails to compile and must choose a norm explicitly.

`verification` is validated under every norm because its checks do not depend
on the domain and an M2-equivalent call leaves it at its valid defaults, so no
input that M2 accepted is rejected; the domain-size check is L2 only (see
[Errors](#errors)).

### Records and report

Measurements are stored in RMS units, which cannot overflow for finite values;
L2-unit values are accessors returning `Option<f64>` (`None` on overflow).

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MeasurementMethod { Exact, Exhaustive, Sampled }

/// One L2 measurement of a patch residual (under ErrorNorm::L2 only).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct L2Measurement {
    pub method: MeasurementMethod,
    /// Points measured: |P| for Exact and Exhaustive, the drawn sample count
    /// (duplicates included) for Sampled.
    pub points: usize,
    /// |P| as f64 (inexact above 2^53).
    pub patch_points: f64,
    /// rms_P(f - f~_P) over the measured points; exact for Exact/Exhaustive.
    pub rms: f64,
    /// Standard error of the mean square divided by the mean square; 0 unless
    /// Sampled, and 0 when the mean square is 0 (no information).
    pub mean_square_rel_std_error: f64,
    /// Largest |f - f~_P| over the measured points.
    pub max_residual: f64,
}
// error_norm() -> Option<f64> = sqrt(patch_points) * rms.

#[derive(Debug, Clone)]
pub struct PatchRecord {                  // #[non_exhaustive], as in M2
    pub projector: Projector,
    pub termination: InterpolationTermination,
    pub engine_error_estimate: f64,       // renamed from error_estimate
    pub max_sample_magnitude: f64,
    pub max_bond_dim: usize,
    pub retries_used: usize,              // new: engine reruns, 0 if none
    pub acceptance: Option<L2Measurement>,// new: the measurement that decided
    pub audit: Option<L2Measurement>,     // new: independent, Sampled only
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ZeroPatchRecord {
    pub projector: Projector,
    pub acceptance: Option<L2Measurement>, // Some under L2
    pub audit: Option<L2Measurement>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum L2ReferenceSource {
    Given,
    ExactRoot,
    /// Only when rtol = 0.
    NotNeeded,
    MonteCarlo { samples: usize, mean_square_rel_std_error: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MaxReferenceSource { Given, ExactRoot, MaxOfRootCandidates }

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct L2ErrorReport {
    /// |X| as f64.
    pub domain_points: f64,
    /// Global E / sqrt(|X|): from Exact and Exhaustive measurements and, for
    /// Sampled contributions, the audit if every one was audited, otherwise
    /// the acceptance measurements.
    pub rms_error: f64,
    /// Relative standard error of rms_error^2 from the audit samples; None
    /// unless audited.
    pub mean_square_rel_std_error: Option<f64>,
    /// Every contribution is Exact or Exhaustive.
    pub certified: bool,
    /// Every Sampled contribution has an audit measurement.
    pub audited: bool,
    /// Fraction of |X| whose contribution is Exact or Exhaustive.
    pub certified_fraction: f64,
    /// ||f~|| / sqrt(|X|), exact up to rounding, summed in path order.
    pub approximation_rms: f64,
    /// rms_error / (approximation_rms - rms_error) when positive: the
    /// a-posteriori bound on E / ||f|| (certified runs) or its estimate.
    pub relative_error: Option<f64>,
}
// error_norm(), approximation_norm(), delta() -> Option<f64>.

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NormReport {
    #[non_exhaustive]
    L2 {
        reference_rms: Option<f64>,       // None only for NotNeeded
        source: L2ReferenceSource,
        tau: f64,                         // RMS allowance and engine tolerance
        error: L2ErrorReport,
    },
    #[non_exhaustive]
    SampledMax {
        max_reference: f64,               // 0 for an all-zero exact root
        source: MaxReferenceSource,
        engine_tolerance: f64,
    },
}

#[derive(Debug, Clone)]
pub struct PatchedInterpolationReport {   // #[non_exhaustive], as in M2
    pub tolerance: ErrorTolerance,        // new
    pub norm: NormReport,                 // new; replaces reference_scale
    pub accepted: Vec<PatchRecord>,
    pub zero_patches: Vec<ZeroPatchRecord>, // renamed from zero_projectors
    pub splits: usize,
    pub function_evaluations: usize,
    pub cache_hits: usize,
    pub measurement_evaluations: usize,   // new: part of function_evaluations
    pub audit_evaluations: usize,         // new: part of the above
    pub verification_failures: usize,     // new
    pub engine_retries: usize,            // new
}
```

Every acceptance measurement satisfies `rms <= tau`, so the sum of acceptance
statistics never exceeds `tau` in RMS. The audited global estimate can exceed
`tau`; it is reported as measured, because that is its purpose.

### Errors

`PatchedInterpolationError` gains:

- `UnsupportedNorm { norm: ErrorNorm }`: validation, before any evaluation and
  before every other `InvalidInput` check, for `MaxAbs` and `WeightedL2`, with
  the remedy "use `ErrorNorm::L2` (the default) or `ErrorNorm::SampledMax`".
  It never falls back to another norm.
- `VerificationFailed { projector, measurement: L2Measurement }`: a patch
  converged below the cap, its measured L2 error still exceeded its allowance
  after the retries, and no site of `patch_order` is left to split it. Remedy:
  "list more sites in `patch_order`, raise `rtol` or `atol`, or check the
  reference norm". `NoSplitIndexLeft` keeps its M2 meaning (the patch did not
  converge).
- `ToleranceBelowRounding { projector, tau, floor }`: see
  [Rounding floor](#rounding-floor). Remedy: "raise `rtol`, set `atol`, or
  give a reference norm with `L2Reference::Given`".

and new `InvalidInput` branches, all before any evaluation: `rtol` or `atol`
negative or not finite; a given reference not finite and positive;
`verification.samples < 2`; under `L2` only, a domain whose point count is not
finite in `f64`, and `L2Reference::Required` when `rtol > 0` and the root is
not exact. A Monte Carlo reference that comes out zero with `atol = 0` is
`InvalidInput` after the root sample, as in M2. A non-finite value of a patch
network at a measured point is `Interpolation { source: Engine }`.
`SampledMax` accepts every input that M2 accepted.

### Explicit breaking changes

Early development allows them; each is deliberate:

1. `PatchedInterpolationOptions::new(cap)` now selects the verified L2 norm
   and requires a reference norm unless `rtol = 0` or the root is exact.
   Callers that relied on the M2 criterion pass `ErrorNorm::sampled_max()`
   and get the M2 behavior bit for bit (checked against frozen digests, see
   [Tests](#tests)). Under L2 a run costs more evaluations.
2. `rtol` moves into `tolerance`, `reference_scale` becomes
   `ErrorNorm::SampledMax { max_reference }`, and their builders are removed.
3. `PatchRecord::error_estimate` is renamed `engine_error_estimate`, so it
   cannot be read as the L2 error.
4. `PatchedInterpolationReport::reference_scale` is replaced by
   `norm: NormReport`, and `zero_projectors` becomes
   `zero_patches: Vec<ZeroPatchRecord>`.

No field or variant keeps its name with a different meaning.

## Semantics

- **L2** (default): acceptance requires the M2 conditions (`Converged`,
  strictly below the cap, layout checks) and an acceptance measurement with
  `rms <= tau`. Zero patches require the same of the zero approximation. A
  certified run has `E <= delta` up to `GLOBAL_ROUNDING_MARGIN`, and
  `E / ||f|| <= relative_error`. Otherwise the audited global value is an
  estimate with the reported standard error, and without an audit there is no
  estimate.
- **SampledMax**: exactly the M2 driver, with the engine tolerance
  `max(atol, rtol * max_reference)`; `atol = 0` reproduces M2. No measurement,
  and the rustdoc keeps the M2 statement that this is not a verified bound.
- **Placeholders**: `UnsupportedNorm` before any evaluation.
- **Engines** keep the M1 contract: one absolute tolerance, their native
  criterion, and `error_estimate` in that criterion's units. They never see
  the norm.

## Algorithm

Changes to the M2 steps
([tree-pqtci-driver.md](./tree-pqtci-driver.md#algorithm)) under
`ErrorNorm::L2`. Everything not mentioned is unchanged.

1. **Validate** before any evaluation: the norm first (placeholders), then
   the tolerance, the reference, the verification options, and `|X|`.
2. **Pin** the reference at the root (given, not needed, exact root, or the
   opt-in estimate), then `tau`; apply the root rounding-floor check.
3. **Exact small patches** (at most one active site): unchanged. Their
   acceptance measurement is `Exact` with `rms = 0`: the network is built from
   the evaluated values and the one-hot factors multiply by exactly one, so no
   verification runs and no evaluation is added. An exact all-zero patch is a
   zero patch with an `Exact` measurement.
4. **Zero screening.** If every candidate sample is exactly zero, the zero
   approximation is measured on the zero-screen stream (exhaustively if the
   patch is small enough). If it passes, the patch is a zero patch, audited if
   the measurement was sampled. If it fails, the measured points with
   `f != 0`, largest `|f|` first and at most `max_bond_dim - 1` of them, are
   appended to the candidates (they are cached, so no evaluation is added),
   and the patch proceeds to the engine. An exhaustive zero screen leaves every
   value of the patch in the cache; later exhaustive measurements of it
   evaluate nothing. The zero screen does not consume a retry.
5. **Interpolate** (engine run `a = 0`) with the absolute tolerance `tau` and
   the M2 engine seed.
6. **Not converged** (`BondCapReached`, `IterationLimit`, or any future
   variant): split as in M2. No measurement runs on a patch that is not
   accepted anyway ([open question 4](#open-questions-for-the-user)).
7. **Verify** the `Converged` outcome of run `a` after the M2 layout and cap
   checks, on the re-embedded patch that would be stored, so the measured
   network is the returned one:
   - if `|P| <= max(max_exhaustive_points, samples)`, evaluate every point of
     the patch in column-major order (first active site fastest), in chunks of
     a fixed size `MEASUREMENT_CHUNK`: `Exhaustive`;
   - otherwise draw `samples` points uniformly with replacement from the
     verification stream `a`: `Sampled`.

   Values of `f` come through the patch cache; values of the network come
   from one fresh evaluator per measurement, as specified under
   [Determinism](#determinism). Sums of squared magnitudes use scaled
   accumulation, so finite residuals cannot overflow.
8. **Accept.** `rms <= tau` accepts the patch with this acceptance
   measurement. For a sampled acceptance with `audit` on, draw the audit
   sample from the audit stream, measure it the same way, and record it; audit
   points never become pivots.
9. **Retry.** Otherwise apply the rounding-floor check, then, while
   `a < retries`, rerun the engine as run `a + 1`. Its initial pivots are the
   patch's candidates, then the measured points with `|r| > tau` in
   descending order of `|r|`, then the outcome's pivots (if any), without
   duplicates, with the points added to the candidates capped at
   `max_bond_dim - 1` so that the added set alone cannot reach the cap. A
   rerun that does not converge splits the patch (step 6).
10. **Split** when the retries are exhausted, as for a non-converged patch
    (M2 step 10), with one addition: the measured points with `|r| > tau`
    (largest first, at most `max_bond_dim - 1`) are passed to the children as
    candidates, like recycled pivots but independent of `recycle_pivots`,
    because they locate what the approximation missed and are already cached.
    A patch that cannot split returns `VerificationFailed`.
11. **Report.** Records stay in canonical path order. The global report sums
    the contributions and the approximation norms (from
    `SubDomainTreeTN::norm_squared` per accepted patch) in that order, so it
    does not depend on processing order or on the hash order of the partition.
    The approximation norm feeds the report only, never a decision.

Failure handling in one line: a failed verification first reruns the engine
with the worst points as pivots (a missed feature that fits under the cap),
then splits (a feature that needs more rank); a residual at the rounding level
is `ToleranceBelowRounding`, and an exhausted `patch_order` or `max_patches`
is `VerificationFailed` or `ResourceLimit`.

Whether the retry pivots help is a measurement: `add_global_pivots` puts every
added point into the pivot sets of every edge, so they raise the starting rank
of the rerun and can make it reach the cap sooner. The cap on added points
limits that; its effect on retry success, rank, and evaluations is measured in
M9.

### Attempts, seeds, and streams

New stream selectors next to `CANDIDATE_STREAM` and `ENGINE_STREAM` in
`adaptive_interpolation/sampling.rs`; `s` is the patch path state of
`patch_seeds`, `R` is `verification.retries`.

| Stage | Engine seed | Measurement stream | Retries left afterwards |
|---|---|---|---|
| zero screen (only if every candidate is zero) | none | `mix(s ^ ZERO_SCREEN_STREAM)` | `R` |
| engine run 0 | the M2 engine seed | `mix(mix(s ^ VERIFY_STREAM) ^ 0)` | `R` |
| engine run `a`, `1 <= a <= R` | `mix(M2 engine seed ^ a)` | `mix(mix(s ^ VERIFY_STREAM) ^ a)` | `R - a` |
| audit of an accepted or zero patch | none | `mix(s ^ AUDIT_STREAM)` | none |
| reference estimate (root, opt-in) | none | `mix(s_root ^ SCALE_STREAM)` | none |

The engine run and its verification stream share the index `a`, so they stay
aligned whatever happens at the zero screen. Exhaustive measurements use no
stream. Each coordinate of a sampled point is drawn in active-site order with
the existing Lemire draw. The streams depend only on the root seed, the patch
path, and the stage, so a patch's measurements are independent of processing
order and of the engine's randomness. Unit tests pin the new streams against
an independent implementation, as for M2.

### Cache

- Measurement points go through the patch cache: cached values are reused,
  new values are cached, counted in `function_evaluations` and
  `measurement_evaluations` (audits also in `audit_evaluations`), and checked
  for finiteness like every other value.
- On a split the cache, including measured values, is partitioned among the
  children in the existing single pass; the children reuse those values
  without re-evaluation, and their own streams are independent of them.
- On acceptance or a zero verdict the cache is dropped after the audit, as in
  M2.
- Exhaustive measurement adds at most `max(max_exhaustive_points, samples)`
  entries to one patch's cache; that option is the explicit size limit the
  repository rules require for exhaustive work.

### Determinism

The M2 guarantee (identical report and bitwise identical stored node tensors
for a fixed seed, a deterministic evaluator, and a deterministic engine)
extends to L2 runs only if every measured network value is bitwise
reproducible, because an acceptance decision near `tau` could otherwise flip
between runs. Two sources of run-to-run variation stand in the way:

1. **#791**: networks built with `TreeTN::from_tensors` and materialized with
   `contract_to_tensor` are not reproducible at rounding level. This excludes
   `TreeTNEvaluator` for the measurement.
2. **Random index IDs**: IDs come from an unseeded per-thread generator and
   break ties in `sort_indices_deterministic`; every engine outcome has fresh
   bond IDs, and the generic path of `TreeTNCachedEvaluator` mints fresh IDs
   per message. Fixing #791 does not remove ID-dependent ordering.

The gate is a code-level argument backed by a test: **the floating-point
operation order of a measurement is a function of the sorted node names, the
positional legs of the stored node tensors, and the batch contents only.**
The driver's part of the argument is fixed by this design:

- one fresh `TreeTNCachedEvaluator` per measurement, so no message cache
  carries values from another measurement;
- a fixed center (`CachedEvaluatorOptions::center` set to the smallest node
  name, so no greedy search runs) and `EvaluationHint::default()` for every
  batch;
- points in a deterministic order (the stream order, or column-major for
  exhaustive), evaluated in chunks of the fixed size `MEASUREMENT_CHUNK`, so
  batch composition and call history are functions of the patch path and the
  seed. The BLAS or scalar choice of the chain kernel then depends only on the
  batch contents, which are fixed.

The evaluator's part cannot be completed from the code read for this record:
`cached_evaluator.rs` does not use `id()` or `sort_indices_deterministic`
directly, but the generic `IdxTensor` path, which the test trees take, mints
random-ID indices per message and contracts through core `IdxTensor`
contraction, whose leg order has not been audited for ID independence.
**Prerequisite**: an audit of the generic message path of
`TreeTNCachedEvaluator` and the core contraction it calls, in
`tensor4all-treetn` and `tensor4all-core`, that establishes ID-independent
operation order, or makes it so (for example by ordering legs positionally, as
the SRC fixes did). The M3 implementation starts after that audit; the driver
does not work around it.

The test (a unit test of the measurement): build a patch on a branched tree
with a multi-site node and a site-free junction, then, in each of several
repetitions, rebuild it from its raw node data with fresh IDs for every bond
index and a fresh `TreeTN::from_tensors`, create a fresh evaluator, and
measure the same points with the same chunking and hint; all repetitions must
give bitwise identical residuals. "The same patch" is never the same object.
Fresh IDs and fresh hash maps in each repetition cover the per-process sources
of variation within one process.

## Placement

Following roadmap Decision 1 (option E):

- **`tensor4all-treetn`: no change to the M1 contract.** The trait, problem,
  outcome, and termination stay as they are; engines still receive one
  absolute tolerance. The measurement uses the existing public batch
  evaluator. The determinism audit above may change evaluator internals, not
  its API. Rejected: a `TreeInterpolator` method that estimates the error (the
  measurement must be independent of the engine's sampling, and every engine
  would have to implement it, against Decision 2); a public residual
  estimator in `treetn` (its only consumer would be the driver; it can be
  promoted when a second consumer, for example single-network interpolation,
  needs it).
- **`tensor4all-treetci` and other engines: no change.**
- **`tensor4all-partitionedtreetn`:** `ErrorNorm`, `L2Reference`, and
  `ErrorTolerance` at the crate root (shared with M3b); the verification
  options, measurement, record, and report types and the driver changes in
  `adaptive_interpolation`; the measurement itself in a private
  `adaptive_interpolation/verify.rs`; the new streams in `sampling.rs`.
- **`tensor4all-core`: no API change.** `CachedFunction` still does not fit
  (M2 finding), and magnitudes use `CommonScalar::abs_val`.

No code reaches into another crate's internals; the driver uses only public
TreeTN and partition APIs.

## Patched algebra (M3b)

The roadmap's M3 scope also includes an optional global-budget mode for
`add_with_patching`, `truncate_adaptive`, and `contract_adaptive`. This record
fixes only its contract; the algorithm needs its own design record, because it
adds an alternative to the maintainer decision that `cutoff` is best effort
with no whole-network bound.

- The mode is opt-in; the local discarded-weight `cutoff` stays the default
  and keeps its meaning.
- It accepts `ErrorNorm::L2` with an `ErrorTolerance`; any other norm returns
  a typed unsupported error before any work. The reference is the operation's
  exact input norm, computed from the networks, with no caller override, as
  in reconstruction. `PartitionedTreeTN::norm_squared` is not bitwise
  reproducible (hash-order summation at `partitioned_tree_tn.rs:333-337` and
  canonicalization under #791), so a pinned reference must be summed in a
  canonical order and computed by a reproducible path.
- Every truncation is measured, not assumed: the residual is the norm of the
  explicit difference network against the untruncated source
  (`TreeTN::axpby` followed by `norm`, as in `reconstruction/engine.rs`).
  Stages within one output patch add by the triangle inequality; disjoint
  output patches combine in squares, with the volume-proportional split
  above. The result is an a-posteriori bound up to floating-point rounding,
  stronger than the interpolation side because no sampling is involved.
- For contraction the untruncated source is the exact product network, whose
  bond dimensions multiply; its cost, and the reuse of the M6 contraction
  outcome API, are the main questions of the M3b record.

Whether M3b is designed now or after M6 is
[open question 6](#open-questions-for-the-user).

## Tests

**Prerequisite, before the refactor.** A separate commit on the unmodified
M2 code adds a test that records digests of the M2 driver outputs for the M2
test scenarios: projector keys, per-node raw column-major data in the ID-free
leg order of the M2 determinism test (sites by position, bonds by neighbor and
dimension), and every report value. The digests are committed as constants.
After the refactor the same scenarios under `ErrorNorm::sampled_max()` must
reproduce them.

All tests live in `tensor4all-partitionedtreetn`, with the existing
driver-local dense test engine and TreeTCI through the path-only
dev-dependency. Topologies with a claim about trees use a node of degree three
or more, checked in the test.

1. **L2 guarantee against a dense reference, certified.** The M2
   `quantics_tree` (a site-free junction of degree three, two branches of three
   binary quantics sites, and a binary flag: `2^7 = 128` points), extended by
   one leaf node with two sites of dimensions 2 and 3, for `768` points in
   total; `max_exhaustive_points = 1024`, so every patch, the root included,
   is measured exhaustively. A localized function, TreeTCI, a given reference
   norm, and a sufficient cap so that some patches are accepted from the
   engine with rank at least two. The report must be `certified`.
   Materialize the partition once (`to_treetn()?.contract_to_tensor()?`),
   subtract the dense reference, and assert
   `diff.norm() <= delta * (1 + GLOBAL_ROUNDING_MARGIN)`, the agreement of
   `diff.norm()` with `error_norm()` within that margin, and
   `diff.norm() / reference.norm() <= relative_error`.
2. **L2 against a dense reference, sampled.** The same problem with
   `samples = 16`, `max_exhaustive_points = 0`, and a fixed seed, so a patch
   is exhaustive only with at most 16 points. Assert that some accepted patch
   has a `Sampled` acceptance measurement and an audit, that the true
   `diff.norm()` is at most a recorded constant times `delta`, and, when the
   audited relative standard error is positive, that the audited estimate lies
   within a recorded number of standard errors of the true error; when it is
   zero, only the first bound is asserted. Both constants are fixed-seed
   regression constants, not probabilistic claims.
3. **A missed feature is caught.** A dense test engine configured to drop a
   narrow feature returns `Converged`; exhaustive verification rejects it, the
   rerun receives the worst points (at most `max_bond_dim - 1`), and the patch
   is accepted or split; `verification_failures` and `engine_retries` match.
   With `retries = 0` the patch splits immediately.
4. **Sampled retry on a fresh stream.** A sampled verification that fails
   leads to a rerun whose measurement uses verification stream 1, not 0.
5. **Retries exhausted.** The patch splits and the children receive the worst
   points as candidates, with `recycle_pivots` off.
6. **Failure at the end of the order.** `VerificationFailed` when no split
   site is left after a verification failure (and `NoSplitIndexLeft`
   unchanged for a non-converged patch); `ResourceLimit` when `max_patches`
   is reached after a verification failure.
7. **Rounding floor.** A tolerance below the floor at the root, and a patch
   whose failure sits at the rounding level, return `ToleranceBelowRounding`;
   `rtol = atol = 0` splits to exact patches without that error.
8. **Zero patches.** A region where every candidate is zero but the function
   is nonzero on half of the region fails the zero screen (fixed seed) and
   goes to the engine with the nonzero points as candidates, without
   consuming a retry; a truly zero region is a zero patch; accepted and zero
   patches still cover the domain.
9. **Exhaustive threshold.** A patch with exactly
   `max(max_exhaustive_points, samples)` points is exhaustive and one with one
   point more is sampled; an exhaustive measurement larger than
   `MEASUREMENT_CHUNK` is evaluated in several chunks with the same result as
   a single chunk would give on a patch built for the purpose.
10. **Exact small patches** carry `Exact` measurements with zero error and
    add no measurement evaluations.
11. **Budget arithmetic.** Every acceptance measurement has `rms <= tau`,
    and the global quantities combine as specified, within
    `GLOBAL_ROUNDING_MARGIN`.
12. **Reference.** Given, not needed (`rtol = 0`), exact root, and Monte
    Carlo references; `Required` with `rtol > 0` and a non-exact root fails
    before any evaluation; a zero Monte Carlo estimate with `atol = 0` fails
    with its remedy; an all-zero exact root returns an empty partition under
    every tolerance.
13. **SampledMax.** The frozen M2 digests are reproduced; `atol > 0` raises
    the engine tolerance to `atol` when it exceeds `rtol * max_reference`; a
    domain too large for `f64` is accepted under `SampledMax` as in M2.
14. **Determinism.** Two L2 runs with the same seed are bitwise identical
    (patches and reports) on the branched tree, with the dense engine and with
    TreeTCI; plus the fresh-ID measurement test of
    [Determinism](#determinism).
15. **Cache.** Measurement points reuse cached values, a point is never
    evaluated twice, measured values reach the children, and audit points
    never appear among pivots.
16. **Streams.** The zero-screen, verification, audit, and reference streams
    follow the documented SplitMix64 and Lemire mapping.
17. **Errors.** `UnsupportedNorm` for each placeholder, ordered before every
    `InvalidInput` check (a placeholder together with an invalid `rtol`
    reports `UnsupportedNorm`), with an evaluator that fails the test if
    called; every new `InvalidInput` branch; a domain too large for `f64`
    under L2; a non-finite network value.
18. **Complex scalars** in the measurement (`Complex64`), with residual
    magnitudes by `abs_val`.
19. **Rustdoc** examples for every new public item, runnable and asserted,
    with `# Errors` naming the variants.

Inaccuracy from an insufficient bond cap is not a failure in these tests. In
the L2 mode an insufficient cap shows up as more splits, `VerificationFailed`,
`NoSplitIndexLeft`, or `ResourceLimit`; accuracy assertions use a sufficient
cap.

## Documentation surface

The M3 PR updates the rustdoc of the driver module and every changed type,
`crates/tensor4all-partitionedtreetn/README.md` and `src/lib.rs`, the guides
`docs/book/src/guides/partitioned-treetn.md` and
`docs/book/src/guides/tree-tn.md`, and
[tree-pqtci-driver.md](./tree-pqtci-driver.md) ("Error criterion"). Rustdoc
states the definition of verified, that a sampled measurement is an estimate
only when audited and never a bound, that `certified` is absolute with
respect to the allowance used, and the units of each reference.

## Measurements needed later

None blocks the M3 implementation; the defaults below are provisional.

| Question | Measurement | When |
|---|---|---|
| Defaults of `samples`, `max_exhaustive_points`, `retries`, `audit` | measurement evaluations as a share of all evaluations, failure and retry rates | M9, on M2 patches of a real workload (TreeTCI, `rtol` near `1e-4`), chain and branched tree |
| Retry pivots and their cap | retry success, starting and final rank, evaluations per retry | M9, same workloads |
| Engine tolerance below `tau` | total evaluations and patch count against the factor | M9, same workloads, at matched measured accuracy |
| `ROUNDING_FLOOR_FACTOR` | residuals of exact representations at the rounding level | M3 implementation (unit scale), confirmed in M9 |
| Volume versus norm-proportional allocation | patches and evaluations on localized functions at matched measured error | M9, only if open question 2 selects both |

## Non-goals

- A worst-case L2 bound for a black-box function from finitely many samples;
  it does not exist.
- Changing the M1 contract or any engine's criterion.
- Implementing `MaxAbs` or `WeightedL2`.
- Weighted or non-L2 norms in reconstruction (roadmap non-goal).
- Truncating low-amplitude regions to zero patches on their measured error
  (zero patches still require exactly zero candidates first).
- The patched-algebra algorithm (M3b), parallel execution (M7), and
  split-site selection (M5).
- Merging `ReconstructionTolerance` into `ErrorTolerance`; it is proposed
  for M3b, where the algebra adopts the shared type (the defaults differ:
  `1e-6` there, `1e-8` here).

## Open questions for the user

1. **Definition of "verified".** Exact and exhaustive measurements are
   certificates of the absolute error against the allowance used; sampled
   ones are decision statistics, and unbiased estimates only through an
   independent audit; `certified` does not cover an estimated reference. Is
   this acceptable, or should the public API avoid the word "verified" for
   sampled measurements (for example "measured")?
2. **Budget allocation.** Proposed: volume-proportional with one pinned
   `tau` (matches the roadmap's pinned reference, reconstruction's allowance,
   the local `cutoff` allocation, and the M1 record). It needs a reference
   norm, which is either given or an estimate that can loosen the tolerance
   (see question 3), and it over-refines localized functions by about
   `sqrt(rho)` and can push their peak patches below the rounding floor.
   Alternative: proportional to each patch's own norm,
   `ms_P(r) <= rtol^2 ms_P(f~_P) + atol^2 / |X|`, giving
   `E^2 <= rtol^2 ||f~||^2 + atol^2`. It needs no reference norm and no
   estimate, is order-independent, and avoids the over-refinement; its
   guarantee is relative to `||f~||` (relative to `||f||` up to
   `1 / (1 - rtol)`), it requires `rtol < 1`, with `atol = 0` it demands
   relative accuracy in low-amplitude regions, and it puts `||f~_P||` into
   the acceptance decision, which then needs a bitwise reproducible patch norm
   (the canonicalizing `norm_squared` falls under #791). The a-posteriori
   relative bound is available under either. Keep the proposal, switch, or
   offer both?
3. **Default reference norm.** Proposed now: none (`L2Reference::Required`),
   with the Monte Carlo estimate as an explicit opt-in, because the estimate
   is heavy-tailed for localized functions and can silently loosen `delta`
   (about 12 times with probability about 0.6% in the example above) as well
   as fail. The previous revision proposed the estimate as the default.
   Alternatives: the estimate as default, or the norm of the root's (possibly
   capped) engine network.
4. **Accept capped outcomes on their measured error?** A `BondCapReached`
   patch whose measured error fits its allowance could be accepted, saving
   splits. This changes the M1/M2 rule "only `Converged` is accepted", and
   with a sampled measurement it would widen the selection effects described
   above. Proposed: not in M3.
5. **Acceptance statistic and audit.** Proposed: accept on the point estimate
   of the acceptance sample, and draw an independent audit sample for every
   sampled contribution by default (`audit = true`), at the cost of `samples`
   more evaluations per sampled accepted or zero patch; without the audit the
   report has no error estimate for sampled patches. An upper confidence bound
   `m + z * SE` rejects more borderline patches but does not help against a
   concentrated residual (it is zero in the counterexample). Should the audit
   be on by default, and should a confidence-bound acceptance be offered?
6. **Scope of M3.** Proposed: this record (interpolation) is implemented as
   M3; the patched-algebra global-budget mode gets its own record (M3b),
   designed after M6 so that it can use the contraction outcome API. Or
   design M3b now?
7. **Placeholders.** Proposed: `MaxAbs` and `WeightedL2`. Are these the norms
   you want reserved, or others (for example a Sobolev or a pointwise
   relative norm)?
