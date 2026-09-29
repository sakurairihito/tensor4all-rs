# Tree patching M1: TreeTN prerequisites

## Status

Detailed plan for milestone M1 of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md).
Findings below were verified against `main` at `9316c500` unless marked as a
model estimate. Items that change public API still need their own review
before implementation.

## Summary

| Item | Finding on `main` | Remaining work |
|---|---|---|
| P1. Cap-saturation report and early abort | Missing | New treetn contraction outcome API (main work of M1) |
| P2. Element-wise (Hadamard) product | Exists: `tensor4all_treetn::hadamard` | Verification only; no new primitive |
| P3. Fixed sites in interpolation output | Dimension-one TreeTCI vertices are valid on every tree | Driver-internal index mapping; no node removal |
| P4. Compact patch representation | Current patches keep projected indices at full dimension | Candidate; adopt only after the gates below |
| P5. Forced computational overhead | Several avoidable costs in patch algebra | One small fix in M1; the rest assigned to later milestones |
| P6. Sparse and block-sparse storage | Not needed for patching | Not adopted; possible separate proposal |

## P1. Contraction outcome report and early abort

### Findings

- `tensor4all_treetn::contraction::contract` returns only a `TreeTN`. No
  method (Zipup, Fit, Naive, Src) reports realized ranks or whether
  `max_bond_dim` was binding.
- Zip-up applies `max_bond_dim` through core `factorize` at each edge.
  `tensor4all_core::FactorizeResult` exposes the realized `rank` and, for SVD,
  the retained singular values; it does not expose the discarded weight or
  whether the rank cap, rather than the tolerance, determined the rank.
- `partitionedtreetn::contract_adaptive` therefore contracts a whole probe,
  compares the resulting maximum bond dimension with the cap afterwards, and
  treats equality as saturation. The probe cost is paid in full even when the
  first edge already saturates.

### Plan

Design record: [treetn-contraction-outcome.md](./treetn-contraction-outcome.md).

1. **Outcome API in treetn.** Add a contraction entry point that returns an
   outcome instead of a bare `TreeTN`: either a completed network with a
   report of final per-edge bond dimensions, or a saturated result naming the
   first edge at or above a caller-given threshold. The existing `contract`
   keeps its signature and delegates to the new entry point.
2. **Early abort.** On the tree zip-up path, where each factorization rank is
   the final link dimension, the contraction stops at the first edge that
   reaches the threshold. This is the case `contract_adaptive` needs: a
   saturated probe is discarded anyway.
3. **Method coverage.** The chain zip-up path, Src, Naive, and Fit evaluate
   saturation after completion in the first version: the chain path runs a
   final truncation sweep, so its per-edge ranks are only upper bounds, and
   Fit waits for the open draft PR #656, which restructures `fit.rs`.
4. **Exact binding detection (optional, core).** Rank equal to the cap is a
   conservative saturation signal and can cause unnecessary splits when the
   true rank equals the cap. Exact detection needs core `factorize` to report
   whether the cap discarded nonzero weight. This is a core feature request,
   to be filed as an issue and implemented only if the conservative signal
   proves costly in M9 measurements.
5. **Adopt in `contract_adaptive`.** Group pairs by output projector before
   contracting, and for `Sequential` groups with a split candidate contract
   with the patch cap as threshold, stopping at the first saturated pair. The
   partition layout and values stay unchanged. Depends on #788 (PR #789).

Exit: the outcome API and tree-path early abort are merged with tests on
chain and branched trees, including a test that an aborted probe performs
fewer factorizations than a completed one; `contract_adaptive` uses it.

## P2. Element-wise (Hadamard) product

### Findings

- `tensor4all_treetn::hadamard(left, right, index_pairs, center, options)`
  multiplies two TreeTNs element-wise along paired external indices. It wraps
  `partial_contract` with `PartialContractionSpec::diagonal_pairs`.
- Each diagonal pair attaches a structured copy tensor
  (`IdxTensor::copy_tensor`, diagonal storage) to the left node and then runs
  the ordinary contraction selected by `ContractionOptions`. No dense
  materialization occurs.
- An earlier roadmap statement that no TreeTN Hadamard entry point exists was
  wrong: the name search output had been truncated.

### Plan

- No new primitive. `hadamard` reaches `contraction::contract` through
  `partial_contract`; an outcome variant for it is deferred to M6, when
  patched element-wise products need it.
- Patch compatibility for Hadamard pairs distinct index identities
  (`left_index`, `right_index`); the projector-matching rule for such pairs is
  an M6 design item, not an M1 item.
- Cost characterization against the expected chain scaling belongs to M9.

## P3. Fixed sites in interpolation output

### Findings

- A `TreeTciGraph` vertex is one TreeTCI site with one local dimension. The
  prototype on `feat/treetci-adaptive-patching` represents a fixed site as a
  vertex of local dimension one on the unchanged graph.
- Removing a fixed vertex changes the tree. It is exact and cheap only for a
  leaf or a degree-two vertex. A fixed vertex of degree three or more is a
  junction: after removal its neighbors must be reconnected, and re-inserting
  it into the original topology fuses bonds into a product bond. Every patch
  must also share one named topology for patch algebra.
- `tensor4all_treetci::to_treetn` creates fresh site indices
  (`DynIndex::new_dyn`). The driver must map them back to the caller's
  `DynIndex` identities; active sites use the existing
  `TreeTN::replace_site_index_with_indices`.
- A TreeTN node may carry several site indices while a TreeTCI vertex has one
  fused local dimension. Splitting on one index of such a node shrinks the
  fused dimension instead of producing a dimension-one vertex.

### Plan

- Keep dimension-one vertices for fixed sites during interpolation. Never
  remove nodes: all patches keep the original named topology.
- How a fixed site appears in the returned patch follows the P4 decision:
  either an eagerly masked full-dimension index (current `partitionedtreetn`
  invariant) or the dimension-one index removed with `select_indices`. The
  earlier idea of a one-hot embedding is the eager form and is not adopted
  unless P4 is rejected.
- Specify the fused-coordinate mapping for multi-index nodes in the M2/M3
  design record.

## P4. Compact patch representation (candidate)

### Current representations

No current implementation rebuilds fixed parts with dense Kronecker products.

- `partitionedtreetn` masks each projected index with `IdxTensor::mask_index`
  and rebuilds the TreeTN. No node is inserted, but the projected index keeps
  its full dimension and stores zeros: a fixed node costs about `chi^k * d`
  with only a `1/d` fraction nonzero (`k` is the node degree).
- The deprecated chain crate embeds fixed middle sites with a compact
  copy-selector (`from_copy_selector`, payload `chi * d`) and boundary fixed
  sites with unit bonds. Factorization accepts dense storage only, so any
  later truncation or canonicalization through such a node is expected to
  densify it (inferred from the factorization contract, not measured). The
  chain crate is design lineage only, not a baseline.
- The TreeTCI prototype branch performs no tensor embedding; its
  `expand_local_batch` only inserts fixed values into evaluation batches.
- The only genuinely expensive path is assembling one network from all
  patches (`PartitionedTreeTN::to_treetn`): direct-sum bonds grow to
  `sum_p chi_p`. No library code calls it; reconstruction explicitly avoids a
  global direct sum.

### Proposal

A patch is a `Projector` plus a TreeTN whose projected site indices are
removed with `select_indices` instead of masked.

- Nodes and edges are unchanged, so every patch keeps the original named
  topology; a node whose sites are all fixed becomes a site-free node. No
  node is ever removed, which is why re-inserting a node with a bond
  Kronecker product never occurs.
- Operations:

| Operation | Compact form |
|---|---|
| Evaluation | Check projector compatibility, evaluate the compact network |
| Norm, inner product, truncation, canonicalization | On the compact network; fixed parts contribute a factor one |
| Contraction A·B | An index fixed in A and active in B is sliced out of B with `select_indices`; fixed external indices go to the output projector |
| Hadamard | Slice the other operand in the same way |
| Addition with equal projectors | Identical site sets; strict addition applies |
| Un-fixing (sibling merge, reconstruction merge, QFT, assembling one network) | Reattach the site leg on its original node with a one-hot vector; only that node grows by `d`, topology unchanged |
| Operators acting on fixed indices | Needs design: slice the operator or un-fix first |

### Verified TreeTN preconditions

- TreeTN supports site-free nodes (covered by `operator/linear_operator`
  tests; restructure treats them as internal connectors).
- Automatic absorption of site-free subtrees happens only in explicit
  topology changes (`fuse_to`, `split_to`, restructure). Partition algebra
  does not call them; reconstruction calls restructure only on operators.
- `TreeTN::same_topology`, checked before zip-up, compares nodes and edges
  only, not site indices, so patches with different site sets can contract.

### Expected savings (model estimate)

Per fixed node of degree `k` with bond dimension `chi`:

| Representation | Degree-two pass-through | Junction of degree `k` |
|---|---|---|
| Eager mask (current) | `chi^2 * d`, mostly zeros | `chi^k * d` |
| P4 | `chi^2` | `chi^k` |
| P4 with pass-through gauged to identity in diagonal storage | `chi` | `chi^k` (intrinsic) |

Relative to the eager form, P4 saves storage by a factor
`1 - f * (1 - 1/d)` overall (`f` is the fraction of fixed nodes), about 0.83
for `d = 2, f = 1/3`. Arithmetic on fixed nodes drops by `d` for contraction
and by `d` to `d^2` for factorization. The gauged variant is fragile because
any factorization through the node densifies it again.

### Gates before adoption

1. **Measurement.** On representative workloads (for example a tree QTT 2D
   Green's function), produce eager patches with the existing crate, record
   their bond dimensions, and compute storage for eager, P4, and gauged P4
   directly. Also measure runtime of truncation and contraction on masked
   versus sliced patches. This needs no P4 implementation.
2. **Robustness.** A treetn test matrix with site-free nodes and with
   operands whose per-node site sets differ, covering zip-up, fit,
   canonicalization, truncation, addition, and Hadamard.

Only if both gates pass, propose amending the full-site-index invariant of
[partitioned-treetn.md](./partitioned-treetn.md) (#648). Otherwise keep eager
masking and focus on never assembling one network in pipelines.

## P5. Forced computational overhead in patch algebra

| # | Overhead | Evidence | Remedy | Milestone |
|---|---|---|---|---|
| 1 | `contract_group_project_first` exact-adds all contributions (bond `sum chi`) and truncates before checking whether a single contribution already reached the cap, which forces a split anyway. After saturation the probe still feeds budgets, `choose_split_index`, and the no-split fallback | `partitionedtreetn/src/patching.rs` | Check individual contributions first; with `Sequential` and an available split index, skip the group sum (result-preserving). With `ExactParameterGain` the probe drives the split decision, so skipping would change results | M1 small fix ([#788](https://github.com/tensor4all/tensor4all-rs/issues/788)); `ExactParameterGain` case in M5 |
| 2 | Saturated probes are discarded and recomputed from projected originals at every recursion level, with no early abort | same | P1 early abort | M1 |
| 3 | Default `ExactParameterGain` projects and truncates every candidate index's `d` children for each split decision, about `L * d` truncations when `patch_order` is empty | `split_child_parameter_count` | Cheaper split heuristic, candidate limit, or `Sequential` default for QTT | M5 |
| 4 | Eager masking makes contraction and factorization iterate over zero coordinates; each projection needs an extra `64 * eps` compression sweep | `mask_index`, `project_if_present` | P4 gates (include runtime) | M1 |
| 5 | Pairwise disjointness validation on every partition construction and `N_A * N_B` pair enumeration in contraction | `partitioned_tree_tn.rs` pairwise loop | Prefix-tree index over projectors | M6, before M7 |
| 6 | `norm` and `norm_squared` clone and canonicalize on every call, used for budgets and probes | `SubDomainTreeTN::norm` | Cache norms within one operation | M6 |
| 7 | TreeTCI has no evaluation cache; dimension-one vertices are still swept; patches share no samples | no cache in `tensor4all-treetci` | Driver-level cache with transfer to children; measure the dimension-one overhead | M3, measurement in M9 |

## P6. Sparse and block-sparse storage

### Findings

- `tensor4all-tensorbackend` has `Dense`, `Diagonal`, and `Structured`
  (repeated axis classes) storage. Factorization accepts dense storage only,
  and the CUDA path requires dense axis classes.
- tenferro keeps its core dense by design and provides an extension mechanism
  for domain-specific representations with traced execution and AD. Its
  `ext/sparse` crate is a tutorial (COO sparse-sparse matmul, fixed
  structure, not published); it has no general einsum or SVD. tenferro's
  historical layering plan places block-sparse and graded arrays in a
  separate storage layer that delegates dense blocks to tenferro. No such
  crate exists in the tensor4all organization.
- `Structured` storage is a working precedent in tensor4all itself: its
  contraction becomes an einsum over compact payloads executed by tenferro,
  and AD follows automatically. A block-sparse layer could follow the same
  pattern in `tensor4all-tensorbackend` and core without tenferro changes;
  tenferro work would only be needed for fused traced operations, batched
  small-block kernels, or CUDA. How the tensor4all graph execution mode
  handles structured storage is not yet verified.

### Expected gains (model estimate)

Notation: `N_p` patches of bond dimension `chi_p`, local dimension `d`,
length `L`, `k` contributions per contraction group.

| Representation | Storage | Relative to current partition |
|---|---|---|
| Current partition (eager) | `N_p * L * d * chi_p^2` | 1 |
| P4 | `N_p * chi_p^2 * L * (d(1-f) + f)` | `1 - f(1 - 1/d)` |
| Block-sparse per patch | about P4 plus block metadata | about P4 |
| Block-sparse single network | `L * d * sum_p chi_p^2` | about 1 (or P4) |
| Dense single network (`to_treetn`) | `L * d * (sum_p chi_p)^2` | `N_p` |

For example `L = 30, d = 2, chi_p = 100, N_p = 200` gives about 1 GB for the
partition and about 190 GB for a dense single network.

- Compute: relative to the partition, block-sparse gains the same factor `d`
  to `d^2` on fixed nodes as P4. Relative to a dense single network it saves
  `N_p` in storage and `N_p^2` in SVD cost, but only while operations
  preserve the block structure. QFT or operators on coarse (fixed) bits,
  cross-patch merges, and the first QR of a direct sum mix blocks and
  densify. Many small blocks also run inefficient small GEMMs.
- Intermediate storage: the exact sum of `k` contributions in
  `contract_adaptive` is block diagonal before truncation (`k` times smaller
  in block form) but densifies at the first QR. `tensor4all_treetn::fit_sum`
  already fits a sum without forming it, and P5 item 1 removes the sum when a
  contribution is saturated. Output merges already use region-wise schedules
  (`schedule_merge_refine`).

### Decision

Sparse storage is not adopted for the patching roadmap. Relative to the
current partition it adds no order-of-magnitude gain beyond P4, and the main
intermediate peak is addressed by `fit_sum` and P5 item 1, at a fraction of
the cost of a cross-layer storage kind. A block-sparse proposal remains
worthwhile on its own merits if quantum-number symmetries are wanted, or if a
workflow must keep one network over many patches under block-preserving
operations. It would start in `tensor4all-tensorbackend` following the
`Structured` precedent.

## Relation to other work

- PR #656 (draft, adaptive fit initialization) touches `fit.rs`; P1 avoids
  fit until that PR is resolved.
- The deprecated chain crate is design lineage only and is not used as a
  verification baseline.
