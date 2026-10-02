// M4 eager-versus-compact patch representation comparison.
// The body is included by
// `crates/tensor4all-partitionedtreetn/examples/benchmark_tree_patch_representation.rs`.
use std::collections::BTreeMap;
use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use serde_json::json;
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    patched_interpolate, PatchedInterpolationOptions,
};
use tensor4all_partitionedtreetn::{
    ErrorNorm, ErrorTolerance, Projector, SubDomainTreeTN,
};
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::{interpolation::InterpolationProblem, NodeNameNetwork, TreeTN};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Name = String;

const INTERPOLATION_CAP: usize = 2;
const INTERPOLATION_RTOL: f64 = 1.0e-4;
const TRUNCATION_CAP: usize = 1;
const PAIR_COUNT: usize = 10;
const NOISE_LIMIT: f64 = 0.15;
const TIMING_PASSES: usize = 20;
const SWITCH_AMPLITUDES: [f64; 4] = [1.0, 0.8, 1.25, 0.65];

struct Workload {
    name: &'static str,
    topology: NodeNameNetwork<Name>,
    node_sites: BTreeMap<Name, Vec<DynIndex>>,
    sites: Vec<DynIndex>,
    switch_sites: Vec<DynIndex>,
    switch_positions: Vec<usize>,
    initial_pivots: ColMajorArray<usize>,
    center: Name,
}

struct PatchPair {
    key: Vec<usize>,
    projector: Projector,
    eager: TreeTN<IdxTensor, Name>,
    compact: TreeTN<IdxTensor, Name>,
}

struct PreparedCase {
    workload: Workload,
    patches: Vec<PatchPair>,
    eager_bytes: u128,
    compact_bytes: u128,
    compact_conversion_ns: u128,
}

#[derive(Clone, Copy)]
enum Operation {
    Norm,
    Truncate,
}

fn main() -> Result<()> {
    println!(
        "{}",
        json!({
            "kind": "build",
            "benchmark": "tree_patch_representation",
            "build_commit": option_env!("T4A_BENCH_GIT_COMMIT").unwrap_or("unrecorded"),
            "interpolation_engine": "TreeTCI",
            "interpolation_max_bond_dim": INTERPOLATION_CAP,
            "interpolation_rtol": INTERPOLATION_RTOL,
            "truncation_max_bond_dim": TRUNCATION_CAP,
            "pairs": PAIR_COUNT,
            "passes_per_measurement": TIMING_PASSES,
            "noise_limit": NOISE_LIMIT,
            "thread_policy": "pin one CPU; Rayon, OMP and BLAS thread counts are one",
        })
    );

    let mut prepared = Vec::new();
    for workload in [chain_workload()?, branched_workload()?] {
        prepared.push(prepare_case(workload)?);
    }
    if std::env::args().any(|argument| argument == "--prepare-only") {
        println!("{}", json!({"kind": "preflight", "status": "PASS", "timings": false}));
        return Ok(());
    }

    if !noise_study(&prepared)? {
        println!("{}", json!({"kind": "decision", "status": "INCONCLUSIVE", "reason": "eager-versus-eager paired drift exceeded the predeclared 15% limit"}));
        return Ok(());
    }

    for case in &prepared {
        for operation in [Operation::Norm, Operation::Truncate] {
            let eager = representation(case, false);
            let compact = representation(case, true);
            let _ = measure(&eager, &case.workload.center, operation)?;
            let _ = measure(&compact, &case.workload.center, operation)?;
        }
    }

    for case in &prepared {
        for operation in [Operation::Norm, Operation::Truncate] {
            compare_representations(case, operation)?;
        }
    }

    println!(
        "{}",
        json!({
            "kind": "decision",
            "status": "MEASURED",
            "adoption_gate": "adopt compact only if payload drops by at least 20%, truncate median improves by at least 10% in both workloads, norm median regresses by no more than 10%, and all active-slice residuals pass",
            "note": "The gate is recorded before timing; final adoption is reviewed against the complete M4 operation requirements.",
        })
    );
    Ok(())
}

fn chain_workload() -> Result<Workload> {
    let names: Vec<Name> = (0..8).map(|site| format!("q{site:02}")).collect();
    let mut node_sites = BTreeMap::new();
    for name in &names {
        node_sites.insert(name.clone(), vec![DynIndex::new_dyn(2)]);
    }
    let edges: Vec<_> = names
        .windows(2)
        .map(|pair| (pair[0].as_str(), pair[1].as_str()))
        .collect();
    workload(
        "chain",
        node_sites,
        &edges,
        vec![names[0].clone(), names[7].clone()],
        names[4].clone(),
    )
}

fn branched_workload() -> Result<Workload> {
    let node_sites = [
        ("r", vec![]),
        ("x0", vec![DynIndex::new_dyn(2)]),
        ("x1", vec![DynIndex::new_dyn(2)]),
        ("x2", vec![DynIndex::new_dyn(2)]),
        ("y0", vec![DynIndex::new_dyn(2)]),
        ("y1", vec![DynIndex::new_dyn(2)]),
        ("y2", vec![DynIndex::new_dyn(2)]),
        ("z", vec![DynIndex::new_dyn(2)]),
        ("e", vec![]),
    ]
    .into_iter()
    .map(|(name, sites)| (name.to_string(), sites))
    .collect();
    let edges = [
        ("r", "x0"),
        ("x0", "x1"),
        ("x1", "x2"),
        ("x2", "e"),
        ("r", "y0"),
        ("y0", "y1"),
        ("y1", "y2"),
        ("r", "z"),
    ];
    workload(
        "branched",
        node_sites,
        &edges,
        vec!["x0".to_string(), "x2".to_string()],
        "r".to_string(),
    )
}

fn workload(
    name: &'static str,
    node_sites: BTreeMap<Name, Vec<DynIndex>>,
    edges: &[(&str, &str)],
    switch_nodes: Vec<Name>,
    center: Name,
) -> Result<Workload> {
    let mut topology = NodeNameNetwork::new();
    for node in node_sites.keys() {
        topology.add_node(node.clone())?;
    }
    for (left, right) in edges {
        topology.add_edge(&left.to_string(), &right.to_string())?;
    }

    let sites = InterpolationProblem::derive_site_order(&node_sites);
    let switch_sites: Vec<DynIndex> = switch_nodes
        .iter()
        .map(|node| {
            node_sites
                .get(node)
                .and_then(|indices| indices.first())
                .cloned()
                .ok_or_else(|| format!("switch node {node:?} has no site index"))
        })
        .collect::<std::result::Result<_, _>>()?;
    let switch_positions: Vec<usize> = switch_sites
        .iter()
        .map(|site| {
            sites
                .iter()
                .position(|candidate| candidate == site)
                .ok_or_else(|| "switch site is absent from derived site order".to_string())
        })
        .collect::<std::result::Result<_, _>>()?;

    let mut initial_data = Vec::with_capacity(sites.len() * 4);
    for switch_state in 0..4 {
        for (position, _) in sites.iter().enumerate() {
            let coordinate = switch_positions
                .iter()
                .position(|&switch_position| switch_position == position)
                .map(|switch| (switch_state >> (switch_sites.len() - 1 - switch)) & 1)
                .unwrap_or((position + switch_state) % 2);
            initial_data.push(coordinate);
        }
    }
    let initial_pivots = ColMajorArray::new(initial_data, vec![sites.len(), 4])?;
    Ok(Workload {
        name,
        topology,
        node_sites,
        sites,
        switch_sites,
        switch_positions,
        initial_pivots,
        center,
    })
}

fn conditioned_product(point: &[usize], switch_positions: &[usize]) -> f64 {
    let state = switch_positions
        .iter()
        .fold(0usize, |state, &position| (state << 1) | point[position]);
    let mut product = SWITCH_AMPLITUDES[state];
    for (position, &coordinate) in point.iter().enumerate() {
        if switch_positions.contains(&position) {
            continue;
        }
        let slope = (0.025 * (state + 1) as f64 * (position + 1) as f64).min(0.35);
        product *= 1.0 + slope * coordinate as f64;
    }
    product
}

fn max_reference(workload: &Workload) -> f64 {
    let size = workload.sites.iter().map(IndexLike::dim).product::<usize>();
    let mut maximum = 0.0f64;
    for linear in 0..size {
        let mut remainder = linear;
        let point: Vec<usize> = workload
            .sites
            .iter()
            .map(|site| {
                let value = remainder % site.dim();
                remainder /= site.dim();
                value
            })
            .collect();
        maximum = maximum.max(conditioned_product(&point, &workload.switch_positions).abs());
    }
    maximum
}

fn prepare_case(workload: Workload) -> Result<PreparedCase> {
    let reference_max = max_reference(&workload);
    let options = PatchedInterpolationOptions::new(INTERPOLATION_CAP)
        .with_tolerance(ErrorTolerance {
            rtol: INTERPOLATION_RTOL,
            atol: 0.0,
        })
        .with_error_norm(ErrorNorm::sampled_max_with_reference(reference_max))
        .with_patch_order(workload.switch_sites.clone())
        .with_seed(5);
    let n_sites = workload.sites.len();
    let switch_positions = workload.switch_positions.clone();
    let evaluate = move |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
        Ok(batch
            .data()
            .chunks(n_sites)
            .map(|point| conditioned_product(point, &switch_positions))
            .collect())
    };
    let result = patched_interpolate(
        &TreeTciInterpolator::default(),
        workload.topology.clone(),
        workload.node_sites.clone(),
        workload.initial_pivots.clone(),
        evaluate,
        &options,
    )?;

    if result.partition.len() != 4 || !result.report.zero_patches.is_empty() {
        return Err(format!(
            "{} workload produced {} patches and {} zero patches; expected four accepted patches",
            workload.name,
            result.partition.len(),
            result.report.zero_patches.len()
        )
        .into());
    }

    let mut ordered: Vec<(Vec<usize>, Projector, &SubDomainTreeTN<Name>)> = result
        .partition
        .iter()
        .map(|(projector, patch)| {
            let key = workload
                .switch_sites
                .iter()
                .map(|site| projector.get(site).unwrap_or(usize::MAX))
                .collect();
            (key, projector.clone(), patch)
        })
        .collect();
    ordered.sort_by(|left, right| left.0.cmp(&right.0));

    let mut patches = Vec::with_capacity(ordered.len());
    let mut eager_bytes = 0u128;
    let mut compact_bytes = 0u128;
    let conversion_start = Instant::now();
    for (key, projector, patch) in ordered {
        if workload
            .switch_sites
            .iter()
            .any(|site| !projector.is_projected_at(site))
        {
            return Err(format!("{} patch {key:?} did not fix every switch site", workload.name).into());
        }
        let eager = patch.data().clone();
        eager_bytes += payload_bytes(&eager)?;
        let compact = remove_projected_sites(&eager, &projector, &workload.node_sites)?;
        compact_bytes += payload_bytes(&compact)?;
        verify_same_active_values(&eager, &compact, &projector, &workload.node_sites)?;
        patches.push(PatchPair {
            key,
            projector,
            eager,
            compact,
        });
    }
    let compact_conversion_ns = conversion_start.elapsed().as_nanos();

    let case = PreparedCase {
        workload,
        patches,
        eager_bytes,
        compact_bytes,
        compact_conversion_ns,
    };
    print_case(&case, result.report.splits, result.report.function_evaluations);
    Ok(case)
}

fn remove_projected_sites(
    tree: &TreeTN<IdxTensor, Name>,
    projector: &Projector,
    node_sites: &BTreeMap<Name, Vec<DynIndex>>,
) -> Result<TreeTN<IdxTensor, Name>> {
    let mut node_names = tree.node_names();
    node_names.sort();
    let mut tensors = Vec::with_capacity(node_names.len());
    for node_name in &node_names {
        let node = tree
            .node_index(node_name)
            .ok_or_else(|| format!("missing node {node_name:?}"))?;
        let mut tensor = tree
            .tensor(node)
            .ok_or_else(|| format!("missing tensor {node_name:?}"))?
            .clone();
        for site in node_sites.get(node_name).into_iter().flatten() {
            if let Some(position) = projector.get(site) {
                tensor = tensor.select_indices(std::slice::from_ref(site), &[position])?;
            }
        }
        tensors.push(tensor);
    }
    let compact = TreeTN::from_tensors(tensors, node_names.clone())?;
    if !tree.same_topology(&compact) {
        return Err("projected-index removal changed the named tree topology".into());
    }
    Ok(compact)
}

fn verify_same_active_values(
    eager: &TreeTN<IdxTensor, Name>,
    compact: &TreeTN<IdxTensor, Name>,
    projector: &Projector,
    node_sites: &BTreeMap<Name, Vec<DynIndex>>,
) -> Result<f64> {
    let eager_active = remove_projected_sites(eager, projector, node_sites)?;
    let eager_dense = eager_active.to_dense()?;
    let compact_dense = compact.to_dense()?;
    let eager_values = eager_dense.to_vec::<f64>()?;
    let compact_values = compact_dense.to_vec::<f64>()?;
    if eager_values.len() != compact_values.len() {
        return Err(format!(
            "active dense lengths differ: eager {}, compact {}",
            eager_values.len(),
            compact_values.len()
        )
        .into());
    }
    let mut residual_squared = 0.0;
    let mut scale_squared = 0.0;
    for (&eager_value, &compact_value) in eager_values.iter().zip(&compact_values) {
        if !eager_value.is_finite() || !compact_value.is_finite() {
            return Err("active dense values contain a non-finite value".into());
        }
        residual_squared += (eager_value - compact_value).powi(2);
        scale_squared += eager_value.powi(2);
    }
    let residual = residual_squared.sqrt();
    let scale = scale_squared.sqrt().max(1.0);
    let relative = residual / scale;
    if relative > 1.0e-12 {
        return Err(format!("active-slice relative residual {relative:e} exceeds 1e-12").into());
    }
    Ok(relative)
}

fn payload_bytes(tree: &TreeTN<IdxTensor, Name>) -> Result<u128> {
    let mut bytes = 0u128;
    for node_name in tree.node_names() {
        let node = tree
            .node_index(&node_name)
            .ok_or_else(|| format!("missing node {node_name:?}"))?;
        let tensor = tree
            .tensor(node)
            .ok_or_else(|| format!("missing tensor {node_name:?}"))?;
        if !tensor.is_f64() {
            return Err(format!("{} is not f64", node_name).into());
        }
        let elements = tensor.indices().iter().try_fold(1u128, |size, index| {
            size.checked_mul(index.dim() as u128)
                .ok_or("tensor payload size overflow")
        })?;
        bytes = bytes
            .checked_add(elements.checked_mul(8).ok_or("tensor byte size overflow")?)
            .ok_or("TreeTN payload byte sum overflow")?;
    }
    Ok(bytes)
}

fn print_case(case: &PreparedCase, splits: usize, evaluations: usize) {
    let projected_sites = case
        .patches
        .iter()
        .map(|patch| patch.projector.iter().count())
        .sum::<usize>();
    println!(
        "{}",
        json!({
            "kind": "case",
            "name": case.workload.name,
            "sites": case.workload.sites.len(),
            "domain_points": case.workload.sites.iter().map(IndexLike::dim).product::<usize>(),
            "accepted_patches": case.patches.len(),
            "splits": splits,
            "function_evaluations": evaluations,
            "projected_site_axes": projected_sites,
            "eager_payload_bytes": case.eager_bytes,
            "compact_payload_bytes": case.compact_bytes,
            "payload_reduction_percent": 100.0 * (case.eager_bytes - case.compact_bytes) as f64 / case.eager_bytes as f64,
            "compact_conversion_ns": case.compact_conversion_ns,
            "patch_keys": case.patches.iter().map(|patch| patch.key.clone()).collect::<Vec<_>>(),
        })
    );
}

fn representation(case: &PreparedCase, compact: bool) -> Vec<&TreeTN<IdxTensor, Name>> {
    case.patches
        .iter()
        .map(|patch| if compact { &patch.compact } else { &patch.eager })
        .collect()
}

fn measure(
    trees: &[&TreeTN<IdxTensor, Name>],
    center: &str,
    operation: Operation,
) -> Result<(u128, f64)> {
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..TIMING_PASSES {
        for tree in trees {
            match operation {
                Operation::Norm => {
                    let mut copy = black_box((*tree).clone());
                    checksum += black_box(copy.norm()?);
                }
                Operation::Truncate => {
                    let truncated = black_box((*tree).clone()).truncate(
                        [center.to_string()],
                        tensor4all_treetn::TruncationOptions::default()
                            .with_max_bond_dim(TRUNCATION_CAP),
                    )?;
                    checksum += sample_tensor_values(&truncated)?;
                }
            }
        }
    }
    let elapsed_ns = start.elapsed().as_nanos();
    let denominator = (TIMING_PASSES * trees.len()) as u128;
    black_box(checksum);
    Ok((elapsed_ns / denominator, checksum))
}

fn sample_tensor_values(tree: &TreeTN<IdxTensor, Name>) -> Result<f64> {
    let mut names = tree.node_names();
    names.sort();
    let Some(name) = names.first() else {
        return Ok(0.0);
    };
    let node = tree
        .node_index(name)
        .ok_or_else(|| format!("missing node {name:?}"))?;
    let tensor = tree
        .tensor(node)
        .ok_or_else(|| format!("missing tensor {name:?}"))?;
    let data = tensor.to_vec::<f64>()?;
    Ok(data
        .iter()
        .take(16)
        .enumerate()
        .map(|(index, value)| (index + 1) as f64 * value)
        .sum())
}

fn noise_study(cases: &[PreparedCase]) -> Result<bool> {
    let mut valid = true;
    for case in cases {
        let eager = representation(case, false);
        for operation in [Operation::Norm, Operation::Truncate] {
            let mut relative_gaps = Vec::with_capacity(PAIR_COUNT);
            for pair in 0..PAIR_COUNT {
                let (first, second) = if pair % 2 == 0 {
                    (
                        measure(&eager, &case.workload.center, operation)?.0,
                        measure(&eager, &case.workload.center, operation)?.0,
                    )
                } else {
                    let second = measure(&eager, &case.workload.center, operation)?.0;
                    let first = measure(&eager, &case.workload.center, operation)?.0;
                    (first, second)
                };
                let denominator = ((first + second) as f64 * 0.5).max(1.0);
                relative_gaps.push((first.abs_diff(second) as f64) / denominator);
            }
            let median_gap = median_f64(&mut relative_gaps);
            let passes = median_gap <= NOISE_LIMIT;
            valid &= passes;
            println!(
                "{}",
                json!({
                    "kind": "noise",
                    "case": case.workload.name,
                    "operation": operation_name(operation),
                    "median_pair_relative_gap": median_gap,
                    "limit": NOISE_LIMIT,
                    "valid": passes,
                })
            );
        }
    }
    Ok(valid)
}

fn compare_representations(case: &PreparedCase, operation: Operation) -> Result<()> {
    let eager = representation(case, false);
    let compact = representation(case, true);
    let mut eager_samples = Vec::with_capacity(PAIR_COUNT);
    let mut compact_samples = Vec::with_capacity(PAIR_COUNT);
    let mut paired_ratios = Vec::with_capacity(PAIR_COUNT);
    for pair in 0..PAIR_COUNT {
        let (eager_ns, compact_ns) = if pair % 2 == 0 {
            let eager_ns = measure(&eager, &case.workload.center, operation)?.0;
            let compact_ns = measure(&compact, &case.workload.center, operation)?.0;
            (eager_ns, compact_ns)
        } else {
            let compact_ns = measure(&compact, &case.workload.center, operation)?.0;
            let eager_ns = measure(&eager, &case.workload.center, operation)?.0;
            (eager_ns, compact_ns)
        };
        eager_samples.push(eager_ns);
        compact_samples.push(compact_ns);
        paired_ratios.push(compact_ns as f64 / eager_ns.max(1) as f64);
        println!(
            "{}",
            json!({
                "kind": "pair",
                "case": case.workload.name,
                "operation": operation_name(operation),
                "pair": pair,
                "eager_ns_per_patch": eager_ns,
                "compact_ns_per_patch": compact_ns,
                "compact_over_eager": compact_ns as f64 / eager_ns.max(1) as f64,
            })
        );
    }
    let eager_median = median(&mut eager_samples);
    let compact_median = median(&mut compact_samples);
    let paired_median = median_f64(&mut paired_ratios);
    println!(
        "{}",
        json!({
            "kind": "summary",
            "case": case.workload.name,
            "operation": operation_name(operation),
            "eager_median_ns_per_patch": eager_median,
            "compact_median_ns_per_patch": compact_median,
            "median_paired_compact_over_eager": paired_median,
            "payload_reduction_percent": 100.0 * (case.eager_bytes - case.compact_bytes) as f64 / case.eager_bytes as f64,
        })
    );
    if matches!(operation, Operation::Truncate) {
        let mut max_relative_residual = 0.0f64;
        for patch in &case.patches {
            let eager_truncated = patch.eager.clone().truncate(
                [case.workload.center.clone()],
                tensor4all_treetn::TruncationOptions::default()
                    .with_max_bond_dim(TRUNCATION_CAP),
            )?;
            let compact_truncated = patch.compact.clone().truncate(
                [case.workload.center.clone()],
                tensor4all_treetn::TruncationOptions::default()
                    .with_max_bond_dim(TRUNCATION_CAP),
            )?;
            let eager_active = remove_projected_sites(
                &eager_truncated,
                &patch.projector,
                &case.workload.node_sites,
            )?;
            max_relative_residual = max_relative_residual.max(verify_same_active_values(
                &eager_active,
                &compact_truncated,
                &Projector::new(),
                &case.workload.node_sites,
            )?);
        }
        println!(
            "{}",
            json!({
                "kind": "correctness",
                "case": case.workload.name,
                "operation": operation_name(operation),
                "max_active_slice_relative_residual": max_relative_residual,
            })
        );
    }
    Ok(())
}

fn operation_name(operation: Operation) -> &'static str {
    match operation {
        Operation::Norm => "norm",
        Operation::Truncate => "truncate",
    }
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2
    } else {
        values[middle]
    }
}

fn median_f64(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}
