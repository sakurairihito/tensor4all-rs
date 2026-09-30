# Tree patching error contract

## Status

Proposal for milestone M3 of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md). It
changes the public API and the acceptance semantics of the M2 driver
([tree-pqtci-driver.md](./tree-pqtci-driver.md)) and must be approved before
implementation. It covers the interpolation side of M3 in full. The
patched-algebra side (the optional global-budget mode for addition and
contraction) is scoped here and gets its own design record before
implementation ([Patched algebra](#patched-algebra-m3b)).

The decisions below that depend on the user are collected under
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

1. define what "verified L2" can mean for a black-box function;
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
- **M2 reference scale.** Without `reference_scale` the driver pins the scale
  once, from the largest magnitude among the root patch's candidate samples
  (user pivots, recycled pivots, and random points), or from the exact values
  of an exact root (lines 991-1005, 1120-1122). It is a sampled lower bound on
  `max |f|`, in max-norm units.
- **What the engine estimate is.** TreeTCI reports the maximum over edges of
  the last LU pivot error of each edge update
  (`crates/tensor4all-treetci/src/state.rs:193`, `update.rs:107-108`); the
  M1 engine runs with `normalize_error = false`, so the raw value is compared
  with the absolute tolerance. Its global pivot search looks for large
  `|f(x) - tt(x)|` from random starts with local coordinate optimization
  (`globalpivot.rs`, module header). Both are max-type quantities on points
  that the engine chose; neither is an estimate of an L2 norm, and the points
  are not independent of the approximation.
- **M1 contract.** `InterpolationOutcome::error_estimate` is documented as the
  engine's raw error estimate, the quantity compared with the absolute
  tolerance (`crates/tensor4all-treetn/src/interpolation.rs`, lines 630-651).
  The outcome network carries only the active sites. The contract has no
  norm parameter, and the M1 record states that the driver computes the
  absolute tolerance from a reference scale "pinned for all patches in M3".
- **Exact small patches.** A patch with at most one active site is evaluated
  on all its points and built from those values with dimension-one links and
  one-hot factors; its record has `error_estimate = 0`
  (`adaptive_interpolation.rs`, lines 1102-1136).
- **Evaluation cache.** Every function value reaches the driver through the
  per-patch cache, which checks count and finiteness and is partitioned among
  the children in one pass on a split (`adaptive_interpolation/cache.rs`,
  `PatchSampler::sample` at line 185, `PatchCache::split` at line 127).
- **Randomness.** The driver implements SplitMix64 with Lemire's unbiased
  bounded draw. Each patch absorbs its path into a state and derives its
  candidate and engine sub-seeds with the stream selectors `CANDIDATE_STREAM`
  and `ENGINE_STREAM` (`adaptive_interpolation/sampling.rs`, lines 17-27 and
  `patch_seeds` at line 79). The driver offers only the seed API and
  documents the exception to the caller-owned `&mut R` rule.
- **Norms of networks.** `SubDomainTreeTN::norm_squared` clones and
  canonicalizes the patch; `PartitionedTreeTN::norm_squared` sums the patch
  norms because patches are disjoint
  (`partitioned_tree_tn.rs:333`). No dense tensor is formed.
- **Point evaluation of a network.** `TreeTN::evaluator` /
  `TreeTNEvaluator::evaluate_batched` and
  `TreeTNCachedEvaluator::evaluate_batched_typed::<T>` evaluate a network at a
  column-major batch of points without materializing it.
  `TreeTNEvaluator` passes nodes without requested sites through unchanged
  (`treetn/evaluator.rs`, lines 294-345), but evaluates every point through a
  temporary network built with `TreeTN::from_tensors` and
  `contract_to_tensor`, the path that
  [#791](https://github.com/tensor4all/tensor4all-rs/issues/791) makes
  non-reproducible at rounding level. `TreeTNCachedEvaluator` iterates nodes
  in sorted name order and treats a node without requested sites as having no
  entries (`treetn/cached_evaluator.rs`, lines 764-768, 1452-1453,
  2329-2333); no test covers a site-free node, and its cross-run bitwise
  reproducibility has not been verified.
- **Reconstruction precedent.** Reconstruction supports only the unweighted
  discrete L2 norm and uses the allowance `max(atol, rtol * reference_scale)`
  with `ReconstructionTolerance { rtol, atol }` (default `rtol = 1e-6`)
  (`reconstruction/mod.rs`, lines 37-68). It measures a residual as the norm
  of an explicit difference network, `source.axpby(1, candidate, -1)` followed
  by `norm` (`reconstruction/engine.rs:366-367`), and combines disjoint
  regions with `hypot`. It rejects a user override of its reference scale.
- **Algebra precedent.** `PatchingOptions::cutoff` is a local discarded-weight
  threshold, documented as best effort with no whole-network bound
  (`patching.rs`, lines 61-115); that was a recorded maintainer decision in
  [partitioned-treetn.md](./partitioned-treetn.md) (review of #655).
  `PatchingOptions` is not `#[non_exhaustive]`.
- **Extensibility.** `PatchedInterpolationOptions`, `PatchRecord`,
  `PatchedInterpolationReport`, and `PatchedInterpolationError` are
  `#[non_exhaustive]`; `PatchedInterpolationResult` is not.
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
ms_P(g)   = ||g||_P^2 / |P|                     (mean square on P)
```

This is the norm of `TreeTN::norm`, `PartitionedTreeTN::norm`, and
reconstruction. For a uniform quantics grid it is the continuum L2 norm
times the constant `sqrt(|X| / V)`, with `V` the physical volume of the grid
domain, so relative errors agree.

The guaranteed quantity is the **absolute L2 error of the whole partition
against `f` over the whole domain**,

```text
E^2 = ||f - f~||_X^2 = sum over accepted patches P of ||f - f~_P||_P^2
                     + sum over zero patches Z   of ||f||_Z^2 ,
```

where `f~` is the returned partition. The equality is exact because accepted
and zero patches are disjoint and together cover `X` (an M2 report invariant).
Zero patches are therefore part of the error: omitting a patch is an
approximation by zero, and it is charged like any other. Per-patch errors are
the means to this end; the relative error `E / ||f||` follows from `E` and the
reference norm (see [Budget](#budget)).

### What can be computed without the dense function

| Quantity | How | Guarantee | Cost |
|---|---|---|---|
| `||f~_P||_P`, `||f~||_X` | `SubDomainTreeTN::norm_squared`, sum over patches | exact up to rounding | one canonicalization per patch, no evaluations of `f` |
| `f~(x)` at chosen points | TreeTN batch evaluators | exact up to rounding | one network evaluation per point |
| `||f - f~_P||_P` for a small patch | evaluate `f` and `f~_P` at every point of `P` | exact up to rounding (a certificate) | `|P|` evaluations of `f` (fewer with cache hits), `|P|` network evaluations |
| `||f - f~_P||_P` for a large patch | Monte Carlo on `n` fresh uniform points of `P` | unbiased estimate with a standard error; no bound | `n` evaluations of `f` (fewer with cache hits), `n` network evaluations |
| `||f||_X` | exactly only by evaluating all of `X`; otherwise a Monte Carlo estimate | estimate only | `n` evaluations |

A held-out sample fixed in advance is the Monte Carlo row with a fixed point
set: its statement is exact on that set and statistical elsewhere.

The last row is the crux. For a black-box `f`, finitely many samples cannot
bound `||f - f~_P||_P`: a residual supported on a fraction `rho` of `P` is
missed by all `n` uniform samples with probability `(1 - rho)^n`, about
`exp(-n rho)`, and its size is unconstrained. Any claim of an L2 bound for a
large patch is statistical, never worst case.

### What a sampled measurement does guarantee

Let `x_1, ..., x_n` be drawn uniformly and independently from `P` (with
replacement), independently of `f~_P`, and let `r = f - f~_P`.

- `m = (1/n) sum |r(x_i)|^2` is an unbiased estimate of `ms_P(r)`, and
  `|P| * m` of `||r||_P^2`. Its standard error is `s / sqrt(n)` with `s^2` the
  sample variance of `|r(x_i)|^2` (so `n >= 2`). A confidence interval from it
  is asymptotic (central limit theorem) and can be badly optimistic when the
  residual is concentrated; it is reported, not presented as a bound.
- Distribution-free: by exchangeability, a further uniform point exceeds
  `max_i |r(x_i)|` with probability at most `1 / (n + 1)`. The reported
  maximum sampled residual is therefore exceeded, in expectation, on at most a
  fraction `1 / (n + 1)` of the patch. This is a quantile statement, not an L2
  bound.
- Exact on the sampled set: the residual at every sampled point is known.

Selection bias: the driver accepts a patch when its estimate is small, so a
patch whose true error is just above its allowance is accepted whenever its
sample happens to be low. Over the accepted patches the reported estimates are
therefore biased low for borderline patches; far from the allowance the effect
vanishes. An independent audit sample drawn after acceptance would remove the
bias at extra cost ([open question 5](#open-questions-for-the-user)).

Independence is essential. A point that the engine used as a pivot has a
residual near zero by construction, so a sample chosen from the engine's
points would be biased. What must be independent of `f~_P` is the choice of
the points, not the source of the values: a value served from the evaluation
cache is the same value of `f`, so cache hits keep the estimate unbiased.

### Definition

A patch error is **verified** when it has been measured from values of `f` at
points chosen independently of the approximation, by one of three methods,
which the report names per patch:

- **Exact**: the patch was built from all its values (the M2 exact small-patch
  path); its error is zero.
- **Exhaustive**: the residual was evaluated at every point of the patch; the
  measured error is exact up to floating-point rounding. This is a
  certificate.
- **Sampled**: the residual was evaluated at `n` fresh uniform points; the
  measured error is an unbiased estimate reported with its standard error and
  the maximum sampled residual. This is a statistical statement only.

The run's global error combines the per-patch measurements (exact where
exhaustive, estimated where sampled). It is **certified** when every
contribution is exact or exhaustive; otherwise the report says it is an
estimate and gives its standard error and the fraction of the domain that is
certified.

Whether this definition is what the user means by "verified" is
[open question 1](#open-questions-for-the-user).

## Budget

### Global allowance

With a relative tolerance `rtol`, an absolute tolerance `atol`, and a
reference norm `S` (an L2 norm of `f`, see below):

```text
delta = max(atol, rtol * S)          global L2 allowance, as in reconstruction
tau   = delta / sqrt(|X|)            the same allowance as a root mean square
```

`delta` and `tau` are pinned once, before any patch is accepted, and are the
same for every patch: acceptance does not depend on processing order, as M7
requires. `|X|` is computed in `f64` and must be finite (checked before any
evaluation); `tau` avoids forming `|X|`-sized products anywhere else.

### Per-patch allowance

The allowance is split in proportion to patch volume, in squared norm:

```text
||f - f~_P||_P^2 <= delta^2 * |P| / |X|     equivalently     ms_P(f - f~_P) <= tau^2
```

Summing over the disjoint accepted and zero patches, whose volumes add up to
`|X|`, gives `E^2 <= delta^2`. The per-patch condition is a root-mean-square
test against one constant, so no patch needs its volume in the comparison, and
the measured value `m` is compared with `tau^2` directly.

- **Splits.** A patch's allowance depends on its volume only. The children of
  a split have volumes that sum to the parent's, so their squared allowances
  sum to the parent's: splitting never reallocates or borrows budget. A
  rejected parent approximation is discarded, so its error is never charged
  (as in reconstruction, the children replace it).
- **Zero patches** are charged `||f||_Z^2` against `delta^2 |Z| / |X|`.
- **Unused budget** of accurate patches is not redistributed; the guarantee
  holds without it, and redistribution would make acceptance depend on order.

Rejected: an equal split by patch count. The final count is unknown while the
queue runs, so it needs reallocation and makes acceptance order-dependent.
Considered but not proposed: a split proportional to each patch's own norm
(see [open question 2](#open-questions-for-the-user)).

### Engine tolerance

The engine still receives one absolute tolerance, now `tau`. If the engine's
pointwise criterion held everywhere on a patch, `|r| <= tau` would imply the
patch's allowance; verification then checks what the engine did not. Whether a
smaller engine tolerance (a factor below one) lowers the total cost by avoiding
failed verifications is a measurement for M9 on M2 patches, not a knob in M3.

### Reference scale versus `||f||_2` and `max |f|`

The two norms need references with different units:

| Norm | Reference | Units | Engine tolerance |
|---|---|---|---|
| L2 | `S`, an estimate of `||f||_2` | `sqrt(|X|)` times a function value | `max(atol, rtol * S) / sqrt(|X|)`, about `rtol * rms(f)` |
| sampled max (M2) | `reference_scale`, a sampled lower bound on `max |f|` | a function value | `max(atol, rtol * reference_scale)` |

Because `rms(f) = ||f||_2 / sqrt(|X|) <= max |f|`, the L2 engine tolerance is
never looser than the max-norm one for the same `rtol` and exact references.
For a function localized on a fraction `rho` of the domain it is tighter by
about `sqrt(rho)`, and so is the relative accuracy that volume-proportional
allocation requires inside the localized patches. A reference in max-norm
units passed as an L2 reference would be wrong by `sqrt(|X|)`; the API keeps
the two references in different, typed places so that this cannot happen
silently.

`S` comes from, in order:

1. `reference_norm`, if the caller gives it (recommended when `||f||_2` or a
   good estimate is known; for a uniform quantics grid, the continuum norm
   times `sqrt(|X| / V)`);
2. the exact values of the root when the root is an exact small patch;
3. otherwise a Monte Carlo estimate `S^2 = |X| * mean |f(x_i)|^2` from
   `verification.samples` uniform root points of a dedicated stream, pinned
   once and reported with its standard error. It is skipped when `rtol = 0`.

A localized function can make the estimate zero or far off. With `S = 0` and
`atol = 0` the driver fails before interpolating, with the remedy to pass
`reference_norm` or `atol` (the M2 rule for an unpinnable scale). The default
source is [open question 3](#open-questions-for-the-user).

## Public surface

All new items live in `tensor4all-partitionedtreetn`. Names are proposals.

```rust
/// The norm in which an accuracy requirement is stated and measured.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ErrorNorm {
    /// Unweighted discrete L2 norm over the whole domain (default).
    /// `reference_norm` is an L2 norm of the function; `None` pins it as
    /// described under "Budget".
    #[non_exhaustive]
    L2 { reference_norm: Option<f64> },
    /// The engine's own sampled criterion against a max-norm scale: the M2
    /// behavior, with no measurement by the driver.
    #[non_exhaustive]
    SampledMax { reference_scale: Option<f64> },
    /// Placeholder: a verified maximum norm over the whole domain.
    MaxAbs,
    /// Placeholder: an L2 norm with caller-supplied weights.
    WeightedL2,
}
// Constructors: ErrorNorm::l2(), ErrorNorm::l2_with_reference(norm),
// ErrorNorm::sampled_max(), ErrorNorm::sampled_max_with_reference(scale).
// Default: ErrorNorm::L2 { reference_norm: None }.

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
}
```

`ErrorNorm` and `ErrorTolerance` are crate-level types because the
patched-algebra mode of M3b uses the same pair. The variants with data are
`#[non_exhaustive]` so that fields can be added; they are built through the
constructors. Placeholders carry no data; a placeholder that gains an
implementation may gain fields, which is a deliberate breaking change at that
time.

An enum, not a trait: the driver must know how a norm combines over disjoint
patches (Euclidean for L2, maximum for a max-norm), how its allowance splits,
and how it is measured. A trait would have to expose all three before a second
implemented norm exists to shape it. A trait can replace the enum when one
does.

### Options

| M2 field | M3 | Meaning |
|---|---|---|
| `rtol` | moved to `tolerance: ErrorTolerance` | relative to the reference of the selected norm |
| `reference_scale` | removed; `ErrorNorm::SampledMax { reference_scale }` | unchanged inside that variant |
| (new) | `error_norm: ErrorNorm` | default `L2 { reference_norm: None }` |
| (new) | `tolerance.atol` | absolute floor of the allowance, default `0` |
| (new) | `verification: VerificationOptions` | used only by `L2`, validated always |
| `max_bond_dim`, `patch_order`, `n_initial_pivots`, `recycle_pivots`, `seed`, `max_patches` | unchanged | unchanged |

Builders: `with_error_norm`, `with_tolerance`, and `with_verification` are
added; `with_rtol` and `with_reference_scale` are removed, so every caller
that set either one fails to compile and must choose a norm explicitly.

### Records and report

```rust
#[non_exhaustive]
pub enum MeasurementMethod { Exact, Exhaustive, Sampled }

/// One patch's measured L2 error (under ErrorNorm::L2 only).
#[non_exhaustive]
pub struct L2Measurement {
    pub method: MeasurementMethod,
    /// Points measured: |P| for Exact and Exhaustive, the drawn sample count
    /// (duplicates included) for Sampled.
    pub points: usize,
    /// |P| as f64.
    pub patch_points: f64,
    /// ms_P(f - f~_P): exact for Exact/Exhaustive, the estimate for Sampled.
    pub mean_square: f64,
    /// Standard error of mean_square; 0 unless Sampled.
    pub mean_square_std_error: f64,
    /// Largest |f - f~_P| over the measured points.
    pub max_residual: f64,
    /// Verification attempts consumed by this patch (0 for Exact).
    pub attempts: usize,
}
// error_squared() = patch_points * mean_square.

pub struct PatchRecord {                  // #[non_exhaustive], as in M2
    pub projector: Projector,
    pub termination: InterpolationTermination,
    pub engine_error_estimate: f64,       // renamed from error_estimate
    pub max_sample_magnitude: f64,
    pub max_bond_dim: usize,
    pub engine_runs: usize,               // new: 0 for Exact, 1 + retries
    pub l2: Option<L2Measurement>,        // new: Some under L2
}

#[non_exhaustive]
pub struct ZeroPatchRecord {
    pub projector: Projector,
    pub l2: Option<L2Measurement>,        // Some under L2
}

#[non_exhaustive]
pub enum ReferenceSource {
    Given,
    ExactRoot,
    RootSamples,
    Estimated { samples: usize, std_error: f64 },
}
// RootSamples: the M2 max-norm pinning. Estimated: the L2 Monte Carlo
// estimate; std_error is that of the estimate of S^2.

#[non_exhaustive]
pub struct L2ErrorReport {
    /// Sum of the per-patch error_squared(), accepted and zero patches.
    pub error_squared: f64,
    /// sqrt(sum of (patch_points * mean_square_std_error)^2); 0 if certified.
    pub error_squared_std_error: f64,
    /// Every contribution is Exact or Exhaustive.
    pub certified: bool,
    /// Fraction of |X| whose contribution is Exact or Exhaustive.
    pub certified_fraction: f64,
}

pub struct PatchedInterpolationReport {   // #[non_exhaustive], as in M2
    pub error_norm: ErrorNorm,            // new: the requested norm
    pub tolerance: ErrorTolerance,        // new
    pub reference: f64,                   // renamed from reference_scale
    pub reference_source: ReferenceSource, // new
    pub absolute_tolerance: f64,          // new: delta (L2) or engine tolerance
    pub engine_tolerance: f64,            // new: tau (L2) or the same value
    pub l2: Option<L2ErrorReport>,        // new: Some under L2
    pub accepted: Vec<PatchRecord>,
    pub zero_patches: Vec<ZeroPatchRecord>, // renamed from zero_projectors
    pub splits: usize,
    pub function_evaluations: usize,
    pub cache_hits: usize,
    pub verification_evaluations: usize,  // new: part of function_evaluations,
                                          // reference estimate included
    pub verification_failures: usize,     // new
    pub engine_retries: usize,            // new
}
```

A successful L2 run always has `sqrt(l2.error_squared) <= absolute_tolerance`,
because every contribution met its allowance; the statistical content lies in
`error_squared_std_error` and `certified`.

### Errors

`PatchedInterpolationError` gains:

- `UnsupportedNorm { norm: ErrorNorm }`, returned by validation before any
  evaluation for `MaxAbs` and `WeightedL2`, with the remedy "use
  `ErrorNorm::L2` (the default) or `ErrorNorm::SampledMax`". It never falls
  back to another norm.

and new `InvalidInput` branches, all before any evaluation: `rtol` or `atol`
negative or not finite; a given reference not finite and positive;
`verification.samples < 2`; a domain whose point count is not finite in
`f64`. A reference norm that cannot be pinned (estimate zero with `atol = 0`)
is `InvalidInput` after the root sample, as in M2. A non-finite value of a
patch network at a measured point is `Interpolation { source: Engine }`.

### Explicit breaking changes

Early development allows them; each is deliberate:

1. `PatchedInterpolationOptions::new(cap)` now selects the verified L2 norm.
   Callers that relied on the M2 criterion pass
   `ErrorNorm::sampled_max()`; they then get the M2 behavior bit for bit
   (see [Tests](#tests)). Under the new default a run costs more evaluations,
   and without `reference_norm` it can fail where M2 succeeded, because M2
   pinned its scale from candidates that include the user's pivots while the
   L2 estimate uses uniform points only.
2. `rtol` moves into `tolerance`, `reference_scale` into the `SampledMax`
   variant, and their builders are removed.
3. `PatchRecord::error_estimate` is renamed `engine_error_estimate`, so it
   cannot be read as the L2 error.
4. `PatchedInterpolationReport::reference_scale` is renamed `reference`, and
   `zero_projectors` becomes `zero_patches: Vec<ZeroPatchRecord>`.

No field keeps its name with a different meaning.

## Semantics

- **L2** (default): acceptance requires the M2 conditions (`Converged`,
  strictly below the cap, layout checks) and a verified measurement with
  `mean_square <= tau^2`. Zero patches require the same of the zero
  approximation. The global statement is `E <= delta` (up to rounding) for a
  certified run and "the estimate of `E` is at most `delta`, with the
  reported standard error" otherwise.
- **SampledMax**: exactly the M2 driver, with the engine tolerance
  `max(atol, rtol * reference_scale)`; `atol = 0` reproduces M2. No
  measurement, `l2: None`, and the rustdoc keeps the M2 statement that this is
  not a verified bound.
- **Placeholders**: `UnsupportedNorm` before any evaluation.
- **Engines** keep the M1 contract: one absolute tolerance, their native
  criterion, and `error_estimate` in that criterion's units. They never see
  the norm.

## Algorithm

Changes to the M2 steps
([tree-pqtci-driver.md](./tree-pqtci-driver.md#algorithm)) under
`ErrorNorm::L2`. Everything not mentioned is unchanged.

1. **Validate** before any evaluation: the norm (placeholders fail first),
   the tolerance, the reference, the verification options, and `|X|`.
2. **Pin** `S` at the root (given, exact root, or the Monte Carlo estimate
   from the root's scale stream), then `delta` and `tau`.
3. **Exact small patches** (at most one active site): unchanged. Their
   measurement is `Exact` with `mean_square = 0`: the network is built from
   the evaluated values and the one-hot factors multiply by exactly one, so no
   verification runs and no evaluation is added. An exact all-zero patch is a
   zero patch with an `Exact` measurement.
4. **Zero screening.** If every candidate sample is exactly zero, the zero
   approximation is verified (step 7 with `r = f`). If it passes, the patch
   is a zero patch. If it fails, the verification points with `f != 0` are
   appended to the candidates (they are in the cache, so no evaluation is
   added), a sampled measurement counts as spent, and the patch proceeds to
   the engine.
5. **Interpolate** with the absolute tolerance `tau` and the engine seed of
   attempt 0, which equals the M2 engine seed.
6. **Not converged** (`BondCapReached`, `IterationLimit`, or any future
   variant): split as in M2. No measurement runs on a patch that is not
   accepted anyway. Whether a capped outcome may be accepted on its measured
   error is [open question 4](#open-questions-for-the-user).
7. **Verify** a `Converged` outcome after the M2 layout and cap checks, on
   the re-embedded patch that would be stored, so the measured network is the
   returned one:
   - if `|P| <= max(max_exhaustive_points, samples)`, evaluate every point of
     the patch in column-major order (first active site fastest), in bounded
     chunks: `Exhaustive`;
   - otherwise draw `samples` points uniformly with replacement from the
     patch's verification stream of the current attempt: `Sampled`.

   Values of `f` come through the patch cache; values of the patch network
   come from a TreeTN batch evaluator (`TreeTNCachedEvaluator::
   evaluate_batched_typed::<T>`; see [Determinism](#determinism) for the
   precondition). Sums of squared magnitudes use scaled accumulation so that
   finite residuals cannot overflow.
8. **Accept or retry.** `mean_square <= tau^2` accepts the patch with its
   measurement. Otherwise, while fewer than `retries` reruns were made, rerun
   the engine on the same patch with the candidates, followed by the
   outcome's pivots (if any), followed by the measured points with
   `|r| > tau` in descending order of `|r|` (at most `samples` of them),
   without duplicates. The rerun uses the next attempt's engine seed and, for
   a sampled measurement, the next attempt's verification stream, because the
   failed sample now shaped the approximation. An exhaustive measurement
   needs no fresh points: it is not statistical. A rerun that does not
   converge splits the patch (step 6).
9. **Split** when the retries are exhausted, exactly as for a non-converged
   patch (M2 step 10), with one addition: the measured points with
   `|r| > tau` (the largest, at most `samples`) are passed to the children as
   candidates, like recycled pivots but independent of `recycle_pivots`,
   because they locate what the approximation missed and are already cached.
   A patch that cannot split returns `NoSplitIndexLeft` as in M2.
10. **Report.** Records stay in canonical path order. The global
    `L2ErrorReport` sums the contributions in that order, so it does not
    depend on processing order.

Failure handling in one line: a failed verification first reruns the engine
with the worst points as pivots (a missed feature that fits under the cap),
then splits (a feature that needs more rank), and only an exhausted
`patch_order` or `max_patches` escalates to an error.

### Seeds and streams

New stream selectors next to `CANDIDATE_STREAM` and `ENGINE_STREAM` in
`adaptive_interpolation/sampling.rs`, with `s` the patch path state of
`patch_seeds` and `a` the attempt number:

```text
verification seed (attempt a) = mix(mix(s ^ VERIFY_STREAM) ^ a)
engine seed (attempt 0)       = the M2 engine seed
engine seed (attempt a >= 1)  = mix(M2 engine seed ^ a)
reference estimate            = mix(s_root ^ SCALE_STREAM)
```

Each coordinate of a sampled point is drawn in active-site order with the
existing Lemire draw. The streams depend only on the root seed, the patch
path, and the attempt, so a patch's measurement is independent of processing
order and of the engine's randomness. Unit tests pin the new streams against
an independent implementation, as for M2.

### Cache

- Verification points are requested through the patch cache: cached values
  are reused, new values are cached, counted in `function_evaluations` and
  `verification_evaluations`, and checked for finiteness like every other
  value.
- On a split the cache, including the verification values, is partitioned
  among the children in the existing single pass; the children reuse those
  values without re-evaluation, and their own verification streams are
  independent of them.
- On acceptance or a zero verdict the cache is dropped, as in M2.
- Exhaustive measurement adds at most `max(max_exhaustive_points, samples)`
  entries to one patch's cache; that option is the explicit size limit the
  repository rules require for exhaustive work.

### Determinism

The M2 guarantee (identical report and bitwise identical stored node tensors
for a fixed seed, a deterministic evaluator, and a deterministic engine)
extends to L2 runs only if the network evaluation used in step 7 is bitwise
reproducible, because an acceptance decision near `tau^2` could otherwise flip
between runs. `TreeTNEvaluator` builds a temporary network through
`TreeTN::from_tensors` and `contract_to_tensor`, the path affected by
[#791](https://github.com/tensor4all/tensor4all-rs/issues/791).
`TreeTNCachedEvaluator` iterates in sorted node order, but its cross-run
reproducibility is not verified. The implementation therefore starts with a
cross-process test of repeated measurements (same patch, same points, bitwise
identical residuals) on a branched tree with a site-free junction. If it fails,
M3 waits for #791 or an evaluator fix in `tensor4all-treetn`; the driver does
not work around it.

## Placement

Following roadmap Decision 1 (option E):

- **`tensor4all-treetn`: no change.** The M1 trait, problem, outcome, and
  termination stay as they are; engines still receive one absolute tolerance.
  The measurement uses the existing public batch evaluators and norms.
  Rejected: a `TreeInterpolator` method that estimates the error (the
  measurement must be independent of the engine's sampling, and every engine
  would have to implement it, against Decision 2); a public residual
  estimator in `treetn` (its only consumer would be the driver, and it is
  arithmetic on top of the existing evaluators; it can be promoted when a
  second consumer, for example single-network interpolation, needs it).
- **`tensor4all-treetci` and other engines: no change.**
- **`tensor4all-partitionedtreetn`:** `ErrorNorm` and `ErrorTolerance` at the
  crate root (shared with M3b); `VerificationOptions`, the measurement and
  report types, and the driver changes in `adaptive_interpolation`; the
  measurement itself in a private `adaptive_interpolation/verify.rs`; the new
  streams in `sampling.rs`.
- **`tensor4all-core`: no change.** `CachedFunction` still does not fit (M2
  finding), and the magnitudes use `CommonScalar::abs_val`.

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
  exact input norm, computed from the networks; a caller override is rejected,
  as in reconstruction.
- Every truncation is measured, not assumed: the residual is the norm of the
  explicit difference network against the untruncated source
  (`TreeTN::axpby` followed by `norm`, as in `reconstruction/engine.rs`).
  Stages within one output patch add by the triangle inequality; disjoint
  output patches combine in squares, with the volume-proportional split
  above. The result is an a posteriori bound up to floating-point rounding,
  stronger than the interpolation side because no sampling is involved.
- For contraction the untruncated source is the exact product network, whose
  bond dimensions multiply; its cost, and the reuse of the M6 contraction
  outcome API, are the main questions of the M3b record.

Whether M3b is designed now or after M6 is
[open question 6](#open-questions-for-the-user).

## Tests

All in `tensor4all-partitionedtreetn`, with the existing driver-local dense
test engine and TreeTCI through the path-only dev-dependency. Topologies with
a claim about trees use a node of degree three or more, checked in the test.

1. **L2 guarantee against a dense reference, certified.** A branched tree
   (the `quantics_tree` of the M2 engine tests: a site-free junction of degree
   three with two three-bit quantics branches and a binary flag, extended by
   one node with two sites), a localized function, TreeTCI, and a sufficient
   cap so that some patches are accepted from the engine with rank at least
   two. `max_exhaustive_points` covers every patch, so the report must be
   `certified`. Materialize the partition once
   (`to_treetn()?.contract_to_tensor()?`), subtract the dense reference, and
   assert `diff.norm() <= absolute_tolerance * (1 + eps_margin)` and
   `|diff.norm()^2 - l2.error_squared| <= rounding_margin`, with both margins
   recorded as named test constants.
2. **L2 against a dense reference, sampled.** The same problem with
   `max_exhaustive_points = 0` and a fixed seed: the true `diff.norm()` is at
   most a recorded constant times `absolute_tolerance`, and the reported
   estimate lies within a recorded number of standard errors of the true
   error. Both are fixed-seed regression constants, not probabilistic claims.
3. **A missed feature is caught.** A dense test engine configured to drop a
   narrow feature returns `Converged`; exhaustive verification rejects it, the
   rerun receives the worst points, and the patch is accepted or split;
   `verification_failures` and `engine_retries` match. With `retries = 0` the
   patch splits immediately.
4. **Zero patches.** A region where every candidate is zero but the function
   is nonzero on half of the region is not reported as zero (fixed seed); a
   truly zero region is a zero patch with a measurement of zero; accepted and
   zero patches still cover the domain.
5. **Exact small patches** carry `Exact` measurements with zero error and add
   no verification evaluations.
6. **Budget arithmetic.** Every measurement has `mean_square <= tau^2`, the
   global `error_squared <= absolute_tolerance^2`, and the standard error
   combines as specified.
7. **Reference.** Given, exact root, and estimated references; an estimated
   zero reference with `atol = 0` fails with `InvalidInput` and its remedy;
   `rtol = 0` with `atol > 0` needs no estimate.
8. **SampledMax reproduces M2.** The M2 driver tests run with
   `ErrorNorm::sampled_max()` and give the same partitions as before,
   compared bitwise by projector as in the M2 determinism test, and the same
   values in the report fields that M2 had (under their new names).
9. **Determinism.** Two L2 runs with the same seed are bitwise identical
   (patches and reports) on the branched tree, with the dense engine and with
   TreeTCI; plus the cross-process measurement test of
   [Determinism](#determinism), including a site-free node.
10. **Cache.** Verification points reuse cached values, a point is never
    evaluated twice, and verification values reach the children.
11. **Streams.** The verification and reference streams follow the
    documented SplitMix64 and Lemire mapping.
12. **Errors.** `UnsupportedNorm` for each placeholder with an evaluator that
    fails the test if called; every new `InvalidInput` branch; a domain too
    large for `f64`; a non-finite network value.
13. **Complex scalars** in the measurement (`Complex64`), with residual
    magnitudes by `abs_val`.
14. **Rustdoc** examples for every new public item, runnable and asserted,
    with `# Errors` naming the variants.

Inaccuracy from an insufficient bond cap is not a failure in these tests. In
the L2 mode an insufficient cap shows up as more splits, `NoSplitIndexLeft`,
or `ResourceLimit`, not as an unreported error; accuracy assertions use a
sufficient cap.

## Documentation surface

The M3 PR updates the rustdoc of the driver module and every changed type,
`crates/tensor4all-partitionedtreetn/README.md` and `src/lib.rs`, the guides
`docs/book/src/guides/partitioned-treetn.md` and
`docs/book/src/guides/tree-tn.md`, and
[tree-pqtci-driver.md](./tree-pqtci-driver.md) ("Error criterion"). Rustdoc
states the definition of verified, that a sampled measurement is an estimate
and not a bound, and the units of each reference.

## Measurements needed later

None blocks the M3 implementation; the defaults below are provisional.

| Question | Measurement | When |
|---|---|---|
| Defaults of `samples`, `max_exhaustive_points`, `retries` | verification evaluations as a share of all evaluations, failure and retry rates | M9, on M2 patches of a real workload (TreeTCI, `rtol` near `1e-4`), chain and branched tree |
| Engine tolerance below `tau` | total evaluations and patch count against the factor | M9, same workloads, at matched measured accuracy |
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

1. **Definition of "verified".** Is the definition above acceptable: exact or
   exhaustive measurements are certificates, sampled ones are unbiased
   estimates with a standard error, and the report says which? Or should the
   public API avoid the word "verified" for sampled measurements (for
   example "measured")?
2. **Budget allocation.** Proposed: volume-proportional with one pinned
   `delta` (matches the roadmap's pinned reference scale, reconstruction, and
   the M1 record). Alternative: proportional to each patch's own norm,
   `ms_P(r) <= rtol^2 ms_P(f~_P) + atol^2 / |X|`, which gives
   `E^2 <= rtol^2 ||f~||^2 + atol^2`, needs no pinned reference at all, is
   still order-independent, and avoids the `sqrt(rho)` over-refinement of
   localized functions; its guarantee is relative to `||f~||` (so relative to
   `||f||` only up to `1 / (1 - rtol)`), it requires `rtol < 1`, and with
   `atol = 0` it demands relative accuracy in low-amplitude regions. Keep the
   proposal, switch, or offer both?
3. **Default reference norm.** Proposed: a Monte Carlo estimate from the
   root, reported with its standard error. Alternatives: require
   `reference_norm` or `atol` whenever the root is not exact, or use the norm
   of the root's (possibly capped) engine network.
4. **Accept capped outcomes on their measured error?** With verification, a
   `BondCapReached` patch whose measured error fits its allowance could be
   accepted, saving splits. This changes the M1/M2 rule "only `Converged` is
   accepted". Proposed: not in M3.
5. **Acceptance statistic for sampled patches.** Proposed: the point
   estimate, with the selection bias documented. Alternatives: an upper
   confidence bound `m + z * SE` with a user-set `z`, which rejects more
   borderline patches but still is not a bound for concentrated residuals;
   and, independently, an optional audit sample per accepted patch after
   acceptance, which makes the reported global estimate unbiased at the cost
   of `samples` more evaluations per patch.
6. **Scope of M3.** Proposed: this record (interpolation) is implemented as
   M3; the patched-algebra global-budget mode gets its own record (M3b),
   designed after M6 so that it can use the contraction outcome API. Or
   design M3b now?
7. **Placeholders.** Proposed: `MaxAbs` and `WeightedL2`. Are these the norms
   you want reserved, or others (for example a Sobolev or a pointwise
   relative norm)?
