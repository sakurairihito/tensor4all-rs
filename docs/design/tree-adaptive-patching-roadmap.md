# Tree adaptive patching roadmap

## Status

Planning record. This document decomposes the long-term goal into medium-sized
milestones. It is not an implementation contract: each milestone that changes
public API or algorithm semantics needs its own design record (or an update of
an existing one) and a maintainer review before implementation.

## Goal

Provide the complete adaptive-patching method of Grosso et al. on arbitrary
tree tensor networks, with shared-memory and distributed parallel execution:

- adaptive patched interpolation (the pQTCI algorithm) that produces a
  `PartitionedTreeTN` directly;
- patched and adaptive patched contraction, including element-wise products;
- patch-order selection and overpatching control;
- a coherent error contract across interpolation, algebra, and reconstruction;
- patch-level parallelism through Hataori (Rayon first, MPI later).

The chain tensor train is treated as one tree topology, not as a separate code
path.

## References

- G. Grosso, M. K. Ritter, S. Rohshap, S. Badr, A. Kauch, M. Wallerberger,
  J. von Delft, H. Shinaoka, *Adaptive Patching for Tensor Train
  Computations*, [arXiv:2602.22372](https://arxiv.org/abs/2602.22372).
- G. Grosso, *Efficient Tensor Compression through Adaptive Patched Quantics
  Tensor Cross Interpolation*, M.Sc. thesis, LMU/TUM (2025).
- Julia lineage: TCIAlgorithms.jl (adaptive interpolation) and
  PartitionedMPSs.jl (patch algebra).

Record any new port, derivation, or algorithm origin in
`docs/PROVENANCE_AND_CITATION_POLICY.md` in the PR that introduces it.

## Current state on `main`

| Capability | Chain (`tensor4all-partitionedtt`, deprecated) | Tree (`tensor4all-partitionedtreetn`) |
|---|---|---|
| Projector, subdomain, disjoint partition | yes | yes (eagerly masked, arbitrary named trees, multiple sites per node) |
| `add_with_patching`, `truncate_adaptive` | yes (`rtol`) | yes (local discarded-weight `cutoff`) |
| `contract_adaptive` (project-first recursion) | yes | yes |
| Split strategies | `Sequential`, `ExactParameterGain` | same |
| Adaptive patched interpolation | yes (`adaptiveinterpolate`) | **no**; excluded from the migration scope of [partitioned-treetn.md](./partitioned-treetn.md) |
| Global L2 reconstruction and merge | no | yes ([orthogonal-target-reconstruction.md](./orthogonal-target-reconstruction.md)) |
| Parallel patch execution | Hataori Rayon/MPI for interpolation ([adaptive-tci-parallel-execution.md](./adaptive-tci-parallel-execution.md)) | none |

Related prior work:

- Branch `feat/treetci-adaptive-patching` contains a sequential TreeTCI
  prototype (`adaptive_crossinterpolate2`). It returns fixed-value records
  rather than a `PartitionedTreeTN`, keeps fixed sites as dimension-one sites
  on the full graph, does not recycle pivots, and accepts on error alone
  without requiring a converged termination. It is reference material for
  milestone M3, not a merge candidate as is.
- TreeTN already provides an element-wise product,
  `tensor4all_treetn::hadamard`, built on structured copy tensors. No
  contraction API reports bond-cap saturation. Details are in
  [tree-patching-m1-prerequisites.md](./tree-patching-m1-prerequisites.md).

## Milestones

Each milestone lists its outcome, main scope, and exit criteria. Sizes are
relative: S (one focused PR), M (a few PRs), L (a design record plus several
PRs).

### M0. Architecture decisions (S)

Outcome: the decisions below are settled and recorded, so later
milestones do not reopen them.

Scope:

- trait, adapter, and driver placement (Decision 1);
- interpolation-engine abstraction (Decision 2);
- primary error contract (Decision 3);
- chain crate retirement (Decision 4: out of scope).

Exit: all four decisions are recorded below (done). The next gate is the
approved design record for M2/M3.

### M1. TreeTN prerequisites (M)

Outcome: the lower layers provide the primitives that patching needs, so the
partitioned layer does not reach through or reimplement them.

Scope (detailed plan:
[tree-patching-m1-prerequisites.md](./tree-patching-m1-prerequisites.md)):

- a contraction outcome API that reports final per-edge bond dimensions and
  threshold saturation, with early abort on the tree zip-up path and post-hoc
  detection elsewhere (design:
  [treetn-contraction-outcome.md](./treetn-contraction-outcome.md));
- the existing `hadamard` element-wise product needs no new primitive; its
  outcome variant is deferred to M6;
- fixed sites stay dimension-one TreeTCI vertices during interpolation, and
  nodes are never removed, so every patch keeps the original topology;
- a compact patch representation (projected indices removed instead of
  masked) is evaluated behind a measurement gate and a robustness gate before
  any change to the #648 invariant;
- a result-preserving reorder in `contract_adaptive` skips the group sum when
  a single contribution is already saturated and the split strategy is
  `Sequential` ([#788](https://github.com/tensor4all/tensor4all-rs/issues/788));
- sparse and block-sparse storage are evaluated and not adopted for this
  roadmap.

Exit: the outcome API and zip-up early abort are merged with tests,
`contract_adaptive` uses them, and the compact-representation gates have a
recorded result.

### M2. Interpolation engine seam (M)

Outcome: one patch driver can run any tree interpolation engine.

Scope:

- a trait for "interpolate one patch": inputs are a batch evaluator restricted
  to active sites, the tree topology, candidate pivots, a bond cap, and a
  tolerance; outputs are a `TreeTN`, a convergence verdict that distinguishes
  "converged" from "reached cap or iteration limit", an error estimate, the
  maximum sampled magnitude, and recyclable full-domain pivots;
- the trait is defined in `tensor4all-treetn` (Decision 1);
- a TreeTCI implementation of the trait inside `tensor4all-treetci`. TreeACI
  and RSI follow later as implementations in their own crates; adding one
  must not modify the trait, the driver, or the TreeTCI implementation
  (Decision 2);
- `partitionedtreetn` keeps no dependency on any interpolation crate;
- the design record for this milestone amends the scope statement of
  [partitioned-treetn.md](./partitioned-treetn.md).

Exit: the trait and the TreeTCI adapter are merged with tests on chain and
branched topologies. A test-only mock engine exercises the trait through the
driver, demonstrating that a second engine can be added without changing
existing code.

### M3. Sequential tree pQTCI (L)

Outcome: adaptive patched interpolation on arbitrary trees with the behavior of
`partitionedtt::adaptiveinterpolate`, returning a `PartitionedTreeTN`.

Scope:

- the driver is generic over the M2 trait and lives in
  `tensor4all-partitionedtreetn`;

- FIFO/BFS patch queue keyed by `Projector` over full `DynIndex` identities;
- acceptance only for a converged patch within tolerance and strictly below
  the bond cap;
- opt-in pivot recycling, per-patch deterministic seeds, sampled-zero policy,
  and a driver-level evaluation cache with one-pass transfer to children
  (TreeTCI itself has no evaluation cache);
- fixed-site handling through the M1 primitive.

Exit: accepted patches reproduce the source function within the requested
tolerance against dense or independently converged references, on chain and
branched topologies; tests cover splits at leaf, internal, and multi-site
nodes; the driver is deterministic for fixed seeds. The deprecated chain crate
is design lineage only, not a verification baseline.

### M4. Unified error contract (M)

Outcome: a user can state one accuracy requirement and get a reported,
measured bound for interpolation and patched algebra. Verified L2 is the
primary contract (Decision 3).

Scope:

- a user-selectable error-norm option shared by interpolation and patched
  algebra, with L2 as the default; unimplemented norms are typed
  placeholders that return an explicit error;
- interpolation: acceptance and reporting in the selected norm, with the
  reference scale pinned once for all patches instead of per-patch maximum
  samples (the sampled max-norm is one selectable option);
- patched contraction and addition: an optional global-budget mode (per-patch
  tolerance on the order of `tau / N_p`) verified with the difference-network
  norms already used by reconstruction;
- reports expose measured bounds, not only requested tolerances.

Exit: the contract is documented in rustdoc and in the relevant design records;
tests check reported bounds against dense references on small problems.

### M5. Patch-order selection and overpatching control (M)

Outcome: the patch tree adapts its split sites and does not proliferate
redundant patches.

Scope:

- a pivot-based split heuristic generalized from the chain algorithm to tree
  edge bipartitions: fix a candidate site in the pivot multi-indices of the
  highest-rank edge, estimate the resulting ranks, and pick the cheapest site;
- `ExactParameterGain` remains the algebra-side reference strategy, but its
  cost (about `L * d` truncations per split decision when `patch_order` is
  empty) motivates a cheaper default for large patch counts;
- a minimum patch size option and an in-loop merge of sibling patches, reusing
  the reconstruction merge logic.

Exit: benchmarks show the heuristic's parameter count against `Sequential` and
`ExactParameterGain`, and overpatching examples (smooth functions, too-small
cap) no longer exceed the unpatched parameter count by more than a documented
margin.

### M6. Complete adaptive patched contraction (M)

Outcome: the contraction side of the method is feature-complete on trees.

Scope:

- early abort at the bond cap through M1;
- patched element-wise products that pair only overlapping patches;
- refine only the input patches that contributed to unconverged output
  patches, recompute only those outputs, then merge converged neighbors;
- a prefix-tree index over projectors, replacing pairwise disjointness
  validation and `N_A * N_B` pair enumeration, and per-operation norm
  caching.

Exit: tests cover the worst, best, and general patch layouts; a benchmark
reproduces the qualitative ordering of those layouts.

### M7. Shared-memory parallel execution (L)

Outcome: interpolation and contraction run patch-parallel on one node with
reproducible results.

Scope:

- Hataori Rayon domains supplied explicitly by the caller, following
  [adaptive-tci-parallel-execution.md](./adaptive-tci-parallel-execution.md);
  Hataori becomes an optional dependency of `tensor4all-partitionedtreetn`,
  shared by the interpolation driver and patched contraction;
- dynamic scheduling of patches with very different costs instead of strict
  level-synchronous waves;
- parallel contraction over independent output-projector groups and their
  recursive children;
- a pinned global normalization (M4) so acceptance does not depend on
  execution order;
- a documented outer/inner parallelism policy to avoid oversubscription with
  batched callbacks and backend threads.

Exit: parallel and sequential runs produce identical partitions for fixed
seeds and deterministic callbacks; scaling is measured on the M9 workloads.

### M8. Distributed execution (L)

Outcome: MPI execution for large workloads.

Scope:

- a versioned wire format for `TreeTN`/`IdxTensor` patches;
- patch ownership by projector prefix so compatible contraction pairs are
  mostly rank-local;
- collective entry points that match the Hataori MPI conventions.

Exit: an MPI smoke test and a multi-rank benchmark; the feature stays opt-in.

### M9. Validation and benchmarks (M, continuous)

Outcome: evidence that the tree implementation is correct and useful.

Scope:

- correctness against dense or independently converged references (M3);
- the paper's workloads: 2D Green's function compression, bubble diagram via
  element-wise product, and Bethe-Salpeter vertex contraction;
- every runtime comparison is made at matched measured accuracy against an
  exact or independently converged reference, not only at equal requested
  tolerance.

Exit: benchmark records under `benchmarks/` for each milestone that claims a
performance gain.

### M10. Bindings (S-M, deferred)

Outcome: the tree implementation is reachable from other languages.

Scope:

- C API and Tensor4all.jl exposure after the Rust API stabilizes;
- `tensor4all-partitionedtt` is not retired or modified by this roadmap
  (Decision 4).

Exit: binding work is tracked in its own issues.

## Dependencies

```text
M0 ──► M2 ──► M3 ──► M4 ──► M5
 │            ▲       │
 └──► M1 ─────┴──► M6 ◄┘
               M3, M6 ──► M7 ──► M8
               M3 ... M8 ──► M9 (continuous)
               M3, M7 ──► M10 (deferred)
```

M1 and M2 can proceed in parallel. M4 must precede M7 because parallel
acceptance depends on a pinned global normalization.

## Decisions

1. **Placement.** Decided: no new crate and no dependency change.
   - The interpolation-engine trait lives in `tensor4all-treetn`, the layer
     that every engine and `tensor4all-partitionedtreetn` already depend on.
   - Each engine implements the trait in its own crate (TreeTCI in
     `tensor4all-treetci`; later engines likewise). Engines adapt to the
     contract; the driver never adapts to an engine.
   - The patch driver lives in `tensor4all-partitionedtreetn` and sees only
     the trait, so that crate still depends on no interpolation engine. The
     statement in [partitioned-treetn.md](./partitioned-treetn.md) that
     adaptive interpolation is outside that crate must be amended by the M2
     design record; its rationale (no TCI dependency) remains satisfied.
   - Rejected: a new orchestration crate that owns engine adapters (the driver
     side would change for every new engine and adds a public crate); putting
     engine dependencies into `partitionedtreetn` (violates the migration
     record); making engines depend on `partitionedtreetn` (inverts the
     layering).
2. **Engine scope.** Decided: implement TreeTCI first. The engine trait and
   driver must be designed so that adding TreeACI or RSI later means adding a
   new adapter only, without modifying the driver, the trait, or the TreeTCI
   adapter.
3. **Error norm.** Decided: verified L2 is the primary guarantee of the public
   API. Other norms (for example the sampled max-norm used by TCI) are
   user-selectable options. A selectable norm whose implementation does not
   exist yet is a placeholder that returns an explicit typed "not implemented"
   error; it must never silently fall back to another norm.
4. **Chain crate retirement.** Decided: out of scope. `tensor4all-partitionedtt`
   is left untouched; its fate is deferred to a future repository-wide
   restructuring.

## Non-goals

- Changing the TreeTCI or TreeACI algorithms themselves.
- Topology changes between patches beyond the compact fixed-site handling of
  M1; all patches share one named tree topology.
- Weighted or non-L2 norms in reconstruction.
