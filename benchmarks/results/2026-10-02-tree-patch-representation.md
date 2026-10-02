# Adaptive TreeTN patch representation (M4)

## Pre-registered protocol

Registered before collecting timing samples on 2026-10-02.

- **Question:** Does removing projected physical site axes from real adaptive
  TreeTN patches reduce stored tensor payload and improve repeated norm and
  truncation work enough to justify changing the full-site-index invariant?
- **Representations:** eager keeps the projected site axes; compact applies
  `select_indices` to the projected coordinate at each owning node. Both retain
  the same named topology and must represent the same active dense values.
- **Patch source:** `patched_interpolate` with `TreeTciInterpolator`, seeded
  deterministic function with a separate product profile for each of the four
  switch-coordinate states, `rtol = 1e-4`, interpolation bond cap 2, and four
  accepted nonzero switch-coordinate patches. Four initial pivots cover all
  switch-coordinate states.
- **Cases:** an eight-site binary chain and a seven-site binary branched tree
  with a site-free leaf. Each patch fixes two binary switch sites. The chain
  uses `q04` as truncation center; the branched tree uses `r`.
- **Operations:** `TreeTN::norm` and `TreeTN::truncate` with bond cap 1. Each
  timed sample clones every patch before operating on it; the reported time is
  the mean nanoseconds per patch over 20 passes. One untimed warm-up precedes
  each operation/representation/case.
- **Noise check:** 10 eager/eager paired samples for every case and operation,
  alternating measurement order. The median relative pair gap must be at most
  15%; otherwise the whole comparison is inconclusive.
- **Comparison:** 10 eager/compact paired samples per case and operation,
  alternating order. Report all raw pairs and paired median ratios.
- **Correctness:** check eager and compact active dense values before timing,
  and compare active dense values after truncation. The relative residual must
  be at most `1e-12`.
- **Adoption gate:** consider compact representation only if payload falls by
  at least 20% for both cases, compact truncation median is at least 10% faster
  in both cases, compact norm median regresses by no more than 10% in either
  case, and all active-slice residuals pass. This benchmark covers norm and
  truncation; any broader operation requirement remains part of the M4 review.
- **Execution:** release build; pin to CPU 2; set Rayon, OMP, OpenBLAS, MKL, and
  BLAS thread counts to 1. Do not run concurrent CPU-heavy jobs during timing.
- **Host observed before timing:** x86_64, AMD Ryzen 9 6900HX, 16 online logical
  CPUs (8 cores, 2 threads/core); CPU 2 is in the process affinity mask.
- **Revision:** patching branch `feat/tree-adaptive-patching`, HEAD
  `ace06932ceba31e25a930f0852b480198b8e03ce`; benchmark and site-free leaf fix
  are local uncommitted changes. The final source and diff hashes are recorded
  below.
- **Raw outputs:** preserve complete stdout as JSONL in this directory. The
  final-source run is `2026-10-02-tree-patch-representation-run2.jsonl`.

### Protocol amendment before timing

The first fixture preflight used a rank-two conditioned function and cap 3. It
returned only two accepted chain patches, so it did not reach the noise study or
collect timing samples. Before timing, the fixture was changed to use four
switch-state-specific separable profiles and cap 2, so fixing the first switch
site leaves a rank-two conditional function that must split at the second site;
after both are fixed, each patch is rank one. The case matrix, measured
operations, noise rule, correctness threshold, and adoption gate are unchanged.

## Results

Keep the raw JSONL unchanged. The first fixture preflight produced no timing
samples; the revised fixture passed before the measured run.

### Environment immediately before timing

- Rust: `rustc 1.98.1 (48a229cea 2026-09-01)`, LLVM 22.1.8; Cargo 1.98.1.
- OS: Linux 6.18.33.2 Microsoft WSL2; release profile built successfully with
  `T4A_BENCH_GIT_COMMIT=ace06932ceba31e25a930f0852b480198b8e03ce`.
- Load at 10:08 UTC: 0.78, 0.63, 0.98 over 1/5/15 minutes. The earlier 10:04
  memory check showed 18 GiB available; the 16-CPU host affinity includes CPU 2.
- Runner SHA-256:
  `5e66780b0f149fc4e2cd5f98f4b8a9bfba95176f0940faad75b90928c5bd243a`.
- Thin example SHA-256:
  `ba648702bc28f2f64ba48fa6460a8dfbc8b30597465a69367faa25852f8a51e0`.
- Tracked local diff SHA-256:
  `03b853887b5b6fa1cc495c67f35dfe53a09180a96fa80be0e564479189b7edb5`.

### Final-source rerun environment

- Rust/Cargo, OS, release profile, host, and CPU affinity match the preceding
  environment record.
- Load before run 2 at 10:12 UTC: 0.93, 0.53, 0.83 over 1/5/15 minutes; no
  CPU-heavy process was active in the process list.
- Final runner SHA-256:
  `f25da5179c317a166b21ee1d317426593cdcdda5f6f34d0125c892780552c690`.
- Thin example SHA-256:
  `ba648702bc28f2f64ba48fa6460a8dfbc8b30597465a69367faa25852f8a51e0`.
- Tracked diff SHA-256 before run 2:
  `63435ba42867223f82b394c024065bd3578ce3fbe348de6d99410479971afb38`.

### Fixture preflight

The revised fixture passed in `--prepare-only` mode without running timed
operations. Both cases produced four accepted nonzero patches, with patch keys
`[0,0]`, `[0,1]`, `[1,0]`, `[1,1]`. Chain: 8 sites, 256 points, 3 splits, 256
function evaluations, 512 eager bytes, 448 compact bytes (12.5% reduction),
1,067,710 ns conversion. Branched: 7 sites, 128 points, 3 splits, 128
evaluations, 512 eager bytes, 448 compact bytes (12.5% reduction), 1,088,890 ns
conversion. The payload gate therefore cannot pass; timing still proceeds to
record whether compact axes help the two selected operations.

After the first timed run, clippy prompted a private helper signature cleanup
and an eager `ok_or` in the benchmark runner. Neither changes benchmark
semantics, but the final source is being measured again. The first complete raw
run is retained as `2026-10-02-tree-patch-representation.jsonl`; the final-source
rerun is in `2026-10-02-tree-patch-representation-run2.jsonl`. The case matrix,
pair protocol, noise rule, thresholds, and decision gate are unchanged. Run 2
is the primary result because it matches the final lint-clean source.

### Final-source measured comparison (run 2)

The noise study passed in all four strata. Median eager/eager relative gaps
were 1.03% (chain norm), 1.24% (chain truncate), 2.09% (branched norm), and
0.81% (branched truncate). Timed-run compact payload conversion took 920,432 ns
for chain patches and 1,062,450 ns for branched patches; conversion is reported
separately from operation timings.

| Topology | Operation | Eager median (ns/patch) | Compact median (ns/patch) | Paired compact/eager | Payload reduction |
|---|---:|---:|---:|---:|---:|
| Chain | Norm | 313,732 | 323,032 | 1.0280 | 12.5% |
| Chain | Truncate | 3,054,679 | 2,967,261 | 0.9873 | 12.5% |
| Branched | Norm | 365,668 | 341,785 | 0.9450 | 12.5% |
| Branched | Truncate | 3,150,114 | 2,804,103 | 0.8894 | 12.5% |

Maximum active-slice relative residual after truncation was `1.1018e-15` on
chain and `1.1923e-15` on the branched tree. Compact failed the adoption gate
because payload reduction was below 20% in both cases and chain truncation
improved by only 1.27%; branched truncation improved by 11.06%. Decision: retain
eager masked patches and leave the full-site-index invariant unchanged. This
is a decision for the measured norm/truncate cases, not a claim that every
patch algebra operation has been benchmarked.

Run 2 raw output SHA-256:
`754910f509d6f6d7f233078e8d76d48a4dbb53dd3da250294b30025f8c4e74db`.

Run 1, captured before the private helper signature and eager `ok_or` cleanup,
also passed its noise check. Its paired ratios were 1.0159 / 0.9890 for chain
norm / truncate and 0.9672 / 0.9005 for branched norm / truncate. It reached the
same no-adoption decision and is preserved unchanged. Its raw SHA-256 is
`eb26abec57c5b46b70d90efc58aed4c22cbc3c627ada40536e396872ba5d9563`.

Final-source runner SHA-256:
`f25da5179c317a166b21ee1d317426593cdcdda5f6f34d0125c892780552c690`.
Thin example SHA-256:
`ba648702bc28f2f64ba48fa6460a8dfbc8b30597465a69367faa25852f8a51e0`.
Tracked diff SHA-256 recorded before run 2:
`63435ba42867223f82b394c024065bd3578ce3fbe348de6d99410479971afb38`.

Final-source validation before cleanup: `cargo clippy --all-targets -p
tensor4all-treetn -p tensor4all-partitionedtreetn` completed without warnings;
`cargo fmt --all --check` and `git diff --check` passed. The complete
`cargo test -p tensor4all-treetn` suite passed before the lint-only helper
signature cleanup; the final source was exercised by the release benchmark.
