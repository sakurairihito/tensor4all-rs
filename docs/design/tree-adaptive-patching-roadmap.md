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
- A name search on `main` found no TreeTN element-wise (Hadamard) product
  entry point and no contraction API that reports bond-cap saturation. Both
  must be confirmed in M1.

## Milestones

Each milestone lists its outcome, main scope, and exit criteria. Sizes are
relative: S (one focused PR), M (a few PRs), L (a design record plus several
PRs).

### M0. Architecture decisions (S)

Outcome: the open decisions below are settled and recorded, so later
milestones do not reopen them.

Scope:

- crate placement of the interpolation driver (see Decision 1);
- interpolation-engine abstraction (see Decision 2);
- primary error contract (see Decision 3).

Exit: this document is updated with the chosen options, and a design record for
M2/M3 is approved.

### M1. TreeTN prerequisites (M)

Outcome: the lower layers provide the primitives that patching needs, so the
partitioned layer does not reach through or reimplement them.

Scope:

- contraction (zip-up and fit) can report that a requested bond cap was
  reached and optionally stop early, instead of completing a probe that is
  then discarded;
- a TreeTN element-wise (Hadamard) product on shared site indices, or a
  documented diagonal-operator path with the same cost;
- a compact representation for fixed sites: absorb nodes whose sites are all
  fixed into a neighbor, or a structured copy-selector embedding analogous to
  the chain implementation, so deep patches do not sweep dimension-one sites.

Exit: each primitive is available through the TreeTN public API with tests and
rustdoc; missing pieces are filed as issues against the owning crate first.

### M2. Interpolation engine seam (M)

Outcome: one patch driver can run any tree interpolation engine.

Scope:

- a trait for "interpolate one patch": inputs are a batch evaluator restricted
  to active sites, the tree topology, candidate pivots, a bond cap, and a
  tolerance; outputs are a `TreeTN`, a convergence verdict that distinguishes
  "converged" from "reached cap or iteration limit", an error estimate, the
  maximum sampled magnitude, and recyclable full-domain pivots;
- a TreeTCI implementation of the trait; TreeACI (and later RSI) are optional
  implementations after the first one is proven;
- `partitionedtreetn` keeps no dependency on any interpolation crate.

Exit: the trait and the TreeTCI adapter are merged with tests on chain and
branched topologies.

### M3. Sequential tree pQTCI with chain parity (L)

Outcome: adaptive patched interpolation on arbitrary trees with the behavior of
`partitionedtt::adaptiveinterpolate`, returning a `PartitionedTreeTN`.

Scope:

- FIFO/BFS patch queue keyed by `Projector` over full `DynIndex` identities;
- acceptance only for a converged patch within tolerance and strictly below
  the bond cap;
- opt-in pivot recycling, per-patch deterministic seeds, sampled-zero policy,
  and one-pass sample-cache transfer to children;
- fixed-site handling through the M1 primitive.

Exit: on chain topologies the result matches `partitionedtt` (patch set, ranks,
and sampled error) for fixed seeds; branched-tree tests cover splits at leaf,
internal, and multi-site nodes; the driver is deterministic.

### M4. Unified error contract (M)

Outcome: a user can state one accuracy requirement and get a reported,
measured bound for interpolation and patched algebra.

Scope:

- interpolation: optional global normalization by a pinned `||F||_inf`
  shared by all patches, instead of per-patch maximum samples;
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
- `ExactParameterGain` remains the algebra-side reference strategy;
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
  patches, recompute only those outputs, then merge converged neighbors.

Exit: tests cover the worst, best, and general patch layouts; a benchmark
reproduces the qualitative ordering of those layouts.

### M7. Shared-memory parallel execution (L)

Outcome: interpolation and contraction run patch-parallel on one node with
reproducible results.

Scope:

- Hataori Rayon domains supplied explicitly by the caller, following
  [adaptive-tci-parallel-execution.md](./adaptive-tci-parallel-execution.md);
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

- chain parity against `partitionedtt` (M3);
- the paper's workloads: 2D Green's function compression, bubble diagram via
  element-wise product, and Bethe-Salpeter vertex contraction;
- every runtime comparison is made at matched measured accuracy against an
  exact or independently converged reference, not only at equal requested
  tolerance.

Exit: benchmark records under `benchmarks/` for each milestone that claims a
performance gain.

### M10. Migration and bindings (S-M)

Outcome: one supported implementation.

Scope:

- propose retiring `partitionedtt::adaptiveinterpolate` once M3 and M7 reach
  parity (removal requires a separate maintainer decision);
- C API and Tensor4all.jl exposure after the Rust API stabilizes.

Exit: a maintainer decision is recorded; binding work is tracked separately.

## Dependencies

```text
M0 ──► M2 ──► M3 ──► M4 ──► M5
 │            ▲       │
 └──► M1 ─────┴──► M6 ◄┘
               M3, M6 ──► M7 ──► M8
               M3 ... M8 ──► M9 (continuous)
               M3, M7 ──► M10
```

M1 and M2 can proceed in parallel. M4 must precede M7 because parallel
acceptance depends on a pinned global normalization.

## Open decisions

1. **Driver crate.** A new orchestration crate that depends on both
   `tensor4all-partitionedtreetn` and the interpolation engines, or an optional
   feature of an existing crate. The migration record requires
   `partitionedtreetn` itself to stay free of interpolation dependencies.
2. **Engine scope.** TreeTCI only, or a trait designed from the start for
   TreeTCI, TreeACI, and RSI.
3. **Primary error norm.** Sampled max-norm (TCI convention) or verified L2
   (reconstruction convention) as the guarantee advertised by the public API.
4. **Chain crate retirement.** Whether and when `tensor4all-partitionedtt` is
   removed after parity.

## Non-goals

- Changing the TreeTCI or TreeACI algorithms themselves.
- Topology changes between patches beyond the compact fixed-site handling of
  M1; all patches share one named tree topology.
- Weighted or non-L2 norms in reconstruction.
