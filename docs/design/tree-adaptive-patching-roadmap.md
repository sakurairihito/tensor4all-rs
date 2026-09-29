# Tree adaptive patching roadmap

## Status

Planning record. This document orders the long-term goal into medium-sized
milestones by their data dependencies. It is not an implementation contract:
each milestone that changes public API or algorithm semantics needs its own
design record (or an update of an existing one) and a review before
implementation.

Verified facts about the current code that later milestones rely on are kept
in [tree-patching-findings.md](./tree-patching-findings.md).

## Goal

Provide the complete adaptive-patching method of Grosso et al. on arbitrary
tree tensor networks, with shared-memory and distributed parallel execution:

- adaptive patched interpolation (the pQTCI algorithm) that produces a
  `PartitionedTreeTN` directly;
- patched and adaptive patched contraction, including element-wise products;
- patch-order selection and overpatching control;
- a coherent error contract across interpolation, algebra, and reconstruction;
- patch-level parallelism through Hataori (Rayon first, MPI later).

The chain tensor train is one tree topology, not a separate code path. A
topology counts as a branched tree only if some node has degree three or
more; any claim about trees must be checked on such a topology.

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

The deprecated chain crate is design lineage only, not a verification
baseline. Branch `feat/treetci-adaptive-patching` holds a sequential TreeTCI
patching prototype; it is reference material for M2, not a merge candidate.

## Decisions

1. **Placement.** No new crate and no dependency change.
   - The interpolation-engine trait lives in `tensor4all-treetn`, the layer
     that every engine and `tensor4all-partitionedtreetn` already depend on.
   - Each engine implements the trait in its own crate (TreeTCI in
     `tensor4all-treetci`; later engines likewise). Engines adapt to the
     contract; the driver never adapts to an engine.
   - The patch driver lives in `tensor4all-partitionedtreetn` and sees only
     the trait, so that crate still depends on no interpolation engine. The
     statement in [partitioned-treetn.md](./partitioned-treetn.md) that
     adaptive interpolation is outside that crate must be amended by the M1
     design record; its rationale (no TCI dependency) remains satisfied.
   - Rejected: a new orchestration crate that owns engine adapters (the driver
     side would change for every new engine and adds a public crate); engine
     dependencies inside `partitionedtreetn` (violates the migration record);
     engines depending on `partitionedtreetn` (inverts the layering).
2. **Engine scope.** Implement TreeTCI first. Adding TreeACI or RSI later
   means adding an implementation in that engine's crate only, without
   modifying the trait, the driver, or the TreeTCI implementation.
3. **Error norm.** Verified L2 is the primary guarantee of the public API.
   Other norms (for example the sampled max-norm used by TCI) are
   user-selectable. A selectable norm without an implementation is a
   placeholder returning an explicit typed "not implemented" error; it never
   falls back silently to another norm.
4. **Chain crate retirement.** Out of scope. `tensor4all-partitionedtt` is
   left untouched until a future repository-wide restructuring.

## Ordering principle

Milestones follow the direction of data flow: a milestone that consumes
patches comes after the milestone that produces them, and every benchmark or
decision gate runs only on data from the real producer. Stand-in data (for
example patches cut from a dense decomposition) must not be used to decide
anything.

## Milestones

Sizes are relative: S (one focused PR), M (a few PRs), L (a design record plus
several PRs).

### M0. Architecture decisions (S, done)

The four decisions above are recorded. The next gate is the approved M1
design record.

### M1. Interpolation engine seam (M)

Outcome: one patch driver can run any tree interpolation engine.

Scope:

- a trait in `tensor4all-treetn` for "interpolate one patch": inputs are a
  batch evaluator restricted to the active sites, the tree topology,
  candidate pivots, a bond cap, and a tolerance; outputs are a `TreeTN`, a
  convergence verdict that distinguishes "converged" from "reached cap or
  iteration limit", an error estimate, the maximum sampled magnitude, and
  recyclable full-domain pivots;
- the TreeTCI implementation of the trait in `tensor4all-treetci`;
- the design record amends the scope statement of
  [partitioned-treetn.md](./partitioned-treetn.md);
- `partitionedtreetn` keeps no dependency on any interpolation crate.

Exit: the trait and the TreeTCI implementation are merged with tests on chain
and branched trees, and a test-only mock engine exercises the trait,
demonstrating that a second engine needs no change to existing code.

### M2. Sequential tree pQTCI (L)

Outcome: adaptive patched interpolation on arbitrary trees that returns a
`PartitionedTreeTN`. This is the producer of every patch that later
milestones measure or consume.

Scope:

- a driver in `tensor4all-partitionedtreetn`, generic over the M1 trait;
- FIFO patch queue keyed by `Projector` over full `DynIndex` identities;
- acceptance only for a converged patch within tolerance and strictly below
  the bond cap;
- fixed sites handled as described in the findings (dimension-one engine
  vertices, index mapping back to the caller's identities, fused-coordinate
  mapping for multi-index nodes); nodes are never removed;
- opt-in pivot recycling, per-patch deterministic seeds, sampled-zero policy,
  and a driver-level evaluation cache with one-pass transfer to children
  (TreeTCI itself has no evaluation cache).

Exit: accepted patches reproduce the source function within the requested
tolerance against dense references on small cases, on chain and branched
topologies; tests cover splits at leaf, internal, junction, and multi-site
nodes; the driver is deterministic for fixed seeds.

### M3. Error contract (M)

Outcome: one accuracy requirement with a reported, measured bound for
interpolation and patched algebra; verified L2 is the default (Decision 3).

Scope:

- a user-selectable error-norm option shared by interpolation and patched
  algebra, with typed placeholders for unimplemented norms;
- interpolation acceptance in the selected norm, with the reference scale
  pinned once for all patches instead of per-patch maximum samples;
- an optional global-budget mode for patched contraction and addition,
  verified with the difference-network norms already used by reconstruction;
- reports expose measured bounds, not only requested tolerances.

Exit: the contract is documented in rustdoc and design records; tests check
reported bounds against dense references on small problems.

### M4. Patch representation decision (M)

Outcome: a measured decision between the current eager (masked) patches and a
compact representation that removes projected site indices.

Scope:

- measure storage and runtime of both representations on patches produced by
  the M2 driver, with parameters matched to real downstream use (TreeTCI,
  tolerance on the order of `1e-4`), on chains and on branched trees;
- the robustness requirements listed in the findings (site-free nodes,
  including leaves) are tested in `tensor4all-treetn` before any adoption;
- adoption requires amending the full-site-index invariant of
  [partitioned-treetn.md](./partitioned-treetn.md).

Exit: a recorded decision with the measurements.

### M5. Split selection and overpatching control (M)

Outcome: the patch tree adapts its split sites and does not proliferate
redundant patches.

Scope:

- a pivot-based split heuristic generalized from the chain algorithm to tree
  edge bipartitions;
- `ExactParameterGain` remains the algebra-side reference strategy; its cost
  (about `L * d` truncations per split decision when `patch_order` is empty)
  motivates a cheaper default for large patch counts;
- a minimum patch size option and an in-loop merge of sibling patches,
  reusing the reconstruction merge logic.

Exit: measurements on M2 patches compare the heuristic with `Sequential` and
`ExactParameterGain`, and overpatching cases do not exceed the unpatched
parameter count by more than a documented margin.

### M6. Adaptive patched contraction (L)

Outcome: the contraction side of the method is complete on trees. This line
works on existing `PartitionedTreeTN` values and does not depend on M1 or M2;
it is ordered after them to keep the interpolation line first.

Scope:

- the contraction outcome API with early abort
  ([treetn-contraction-outcome.md](./treetn-contraction-outcome.md)) and its
  adoption in `contract_adaptive`;
- patched element-wise products on top of the existing `hadamard`, including
  the projector rule for paired distinct indices;
- refine only the input patches that contributed to unconverged outputs,
  recompute only those outputs, then merge converged neighbors;
- the contraction-side overhead items listed in the findings (prefix-tree
  projector index, per-operation norm caching); the `Sequential` group-sum
  shortcut is [#788](https://github.com/tensor4all/tensor4all-rs/issues/788).

Exit: tests cover the worst, best, and general patch layouts on branched
trees; a benchmark reproduces their qualitative ordering.

### M7. Shared-memory parallel execution (L)

Outcome: interpolation and contraction run patch-parallel on one node with
reproducible results.

Scope:

- Hataori Rayon domains supplied explicitly by the caller, following
  [adaptive-tci-parallel-execution.md](./adaptive-tci-parallel-execution.md);
  Hataori becomes an optional dependency of `tensor4all-partitionedtreetn`;
- dynamic scheduling of patches with very different costs;
- parallel contraction over independent output-projector groups;
- the pinned reference scale of M3, so acceptance does not depend on
  execution order;
- a documented outer/inner parallelism policy.

Exit: parallel and sequential runs produce identical partitions for fixed
seeds and deterministic callbacks; scaling is measured on M9 workloads.

### M8. Distributed execution (L)

Outcome: opt-in MPI execution for large workloads: a versioned wire format for
`TreeTN`/`IdxTensor` patches, patch ownership by projector prefix, and
collective entry points matching the Hataori MPI conventions.

Exit: an MPI smoke test and a multi-rank benchmark.

### M9. Validation and benchmarks (continuous)

- Correctness against dense or independently converged references.
- Workloads matched to real downstream use; every runtime comparison is made
  at matched measured accuracy.
- Each benchmark starts only after the component producing its inputs exists
  on the branch.
- Known risk to track: downstream TreeTCI runs on topologies with junction
  nodes have been observed to be much slower than on chains at low
  temperature; this affects tree pQTCI and must be profiled once M2 exists.

### M10. Bindings (deferred)

C API and Tensor4all.jl exposure after the Rust API stabilizes, tracked in
their own issues.

## Dependencies

```text
M0 ──► M1 ──► M2 ──► M3 ──► M4 ──► M5
                      │
                      └──────────► M7 ──► M8
M6 (independent of M1/M2; scheduled after M2) ──► M7
M9: continuous, each item gated on its producer
M10: deferred
```

## Non-goals

- Changing the TreeTCI or TreeACI algorithms themselves.
- Topology changes between patches; all patches share one named tree
  topology.
- Weighted or non-L2 norms in reconstruction.
