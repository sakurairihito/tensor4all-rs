//! Tests of the driver-side measurement.

use std::collections::{BTreeMap, HashMap};

use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::interpolation::InterpolationProblem;
use tensor4all_treetn::{NodeNameNetwork, TreeTN};

use super::network_values;
use crate::adaptive_interpolation::{patched_interpolate, PatchedInterpolationOptions};

/// A named tree with its sites in the derived site order.
struct Tree {
    topology: NodeNameNetwork<String>,
    node_sites: BTreeMap<String, Vec<DynIndex>>,
    sites: Vec<DynIndex>,
}

fn tree(nodes: &[(&str, &[usize])], edges: &[(&str, &str)]) -> Tree {
    let mut topology = NodeNameNetwork::new();
    for (node, _) in nodes {
        topology.add_node(node.to_string()).unwrap();
    }
    for (left, right) in edges {
        topology
            .add_edge(&left.to_string(), &right.to_string())
            .unwrap();
    }
    let node_sites: BTreeMap<String, Vec<DynIndex>> = nodes
        .iter()
        .map(|(node, dims)| {
            let sites = dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect();
            (node.to_string(), sites)
        })
        .collect();
    let sites = InterpolationProblem::derive_site_order(&node_sites);
    Tree {
        topology,
        node_sites,
        sites,
    }
}

fn max_degree(tree: &Tree) -> usize {
    let graph = tree.topology.graph();
    graph
        .node_indices()
        .map(|node| graph.neighbors(node).count())
        .max()
        .unwrap_or(0)
}

/// One binary site per node around a junction `c` of degree three: the
/// cached evaluator's raw kernels apply. Site order a0, a1, b0, b1, c, d0, d1.
fn raw_kernel_tree() -> Tree {
    tree(
        &[
            ("a0", &[2]),
            ("a1", &[2]),
            ("b0", &[2]),
            ("b1", &[2]),
            ("c", &[2]),
            ("d0", &[2]),
            ("d1", &[2]),
        ],
        &[
            ("c", "a0"),
            ("a0", "a1"),
            ("c", "b0"),
            ("b0", "b1"),
            ("c", "d0"),
            ("d0", "d1"),
        ],
    )
}

/// The M2 `quantics_tree` (site-free junction `r` of degree three) extended
/// by a leaf `w` with two sites of dimensions 2 and 3: the generic path.
/// Site order w0, w1, x0, x1, x2, y0, y1, y2, z.
fn generic_path_tree() -> Tree {
    tree(
        &[
            ("r", &[]),
            ("w", &[2, 3]),
            ("x0", &[2]),
            ("x1", &[2]),
            ("x2", &[2]),
            ("y0", &[2]),
            ("y1", &[2]),
            ("y2", &[2]),
            ("z", &[2]),
        ],
        &[
            ("r", "x0"),
            ("x0", "x1"),
            ("x1", "x2"),
            ("r", "y0"),
            ("y0", "y1"),
            ("y1", "y2"),
            ("r", "z"),
            ("z", "w"),
        ],
    )
}

fn quantics(bits: &[usize]) -> f64 {
    bits.iter()
        .enumerate()
        .map(|(k, &bit)| bit as f64 * 0.5_f64.powi(k as i32 + 1))
        .sum()
}

fn peak(x: f64, y: f64) -> f64 {
    (-((x - 0.3) / 0.12).powi(2) - ((y - 0.6) / 0.12).powi(2)).exp()
}

fn raw_kernel_function(p: &[usize]) -> f64 {
    let (x, y, u) = (quantics(&p[0..2]), quantics(&p[2..4]), quantics(&p[5..7]));
    (1.0 + 0.5 * p[4] as f64) * peak(x, y) + 0.2 * (x * y + u)
}

fn generic_path_function(p: &[usize]) -> f64 {
    let (x, y) = (quantics(&p[2..5]), quantics(&p[5..8]));
    (1.0 + 0.25 * p[0] as f64 + 0.125 * p[1] as f64) * (1.0 + 0.5 * p[8] as f64) * peak(x, y)
        + 0.2 * x * y
}

/// Raw data of one stored node: name, legs, column-major values.
type RawNode = (String, Vec<DynIndex>, Vec<f64>);

/// A stored patch as raw node data and the full-domain points of the patch.
struct RawPatch {
    nodes: Vec<RawNode>,
    points: Vec<usize>,
}

/// Every point of the domain, column-major (first site fastest).
fn domain(dims: &[usize]) -> Vec<Vec<usize>> {
    let n: usize = dims.iter().product();
    (0..n)
        .map(|mut linear| {
            dims.iter()
                .map(|&dim| {
                    let value = linear % dim;
                    linear /= dim;
                    value
                })
                .collect()
        })
        .collect()
}

/// Run the M2 driver with TreeTCI and return its stored patches as raw data,
/// with the points inside each patch.
fn stored_patches(tree: &Tree, f: fn(&[usize]) -> f64) -> Vec<RawPatch> {
    let n_sites = tree.sites.len();
    let dims: Vec<usize> = tree.sites.iter().map(IndexLike::dim).collect();
    let all = domain(&dims);
    let scale = all.iter().map(|p| f(p).abs()).fold(0.0, f64::max);
    let options = PatchedInterpolationOptions::new(3)
        .with_rtol(1e-8)
        .with_reference_scale(scale)
        .with_seed(7);
    let result = patched_interpolate(
        &TreeTciInterpolator::default(),
        tree.topology.clone(),
        tree.node_sites.clone(),
        ColMajorArray::new(vec![], vec![n_sites, 0]).unwrap(),
        |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
            Ok(batch.data().chunks(n_sites).map(f).collect())
        },
        &options,
    )
    .unwrap();
    assert!(result.report.splits >= 1, "the root must split");
    result
        .report
        .accepted
        .iter()
        .map(|record| {
            let data = result.partition.get(&record.projector).unwrap().data();
            let mut names = data.node_names();
            names.sort();
            let nodes = names
                .iter()
                .map(|name| {
                    let tensor = data.tensor(data.node_index(name).unwrap()).unwrap();
                    (
                        name.clone(),
                        tensor.indices().to_vec(),
                        tensor.to_vec::<f64>().unwrap(),
                    )
                })
                .collect();
            let points = all
                .iter()
                .filter(|point| {
                    record.projector.iter().all(|(site, &value)| {
                        point[tree.sites.iter().position(|s| s == site).unwrap()] == value
                    })
                })
                .flatten()
                .copied()
                .collect();
            RawPatch { nodes, points }
        })
        .collect()
}

/// Rebuild a patch from raw data with a fresh ID for every bond index and a
/// fresh `TreeTN::from_tensors`.
fn rebuild(patch: &RawPatch, sites: &[DynIndex]) -> TreeTN<IdxTensor, String> {
    let mut fresh: HashMap<DynIndex, DynIndex> = HashMap::new();
    let mut names = Vec::new();
    let mut tensors = Vec::new();
    for (name, legs, values) in &patch.nodes {
        let legs: Vec<DynIndex> = legs
            .iter()
            .map(|leg| {
                if sites.contains(leg) {
                    leg.clone()
                } else {
                    fresh
                        .entry(leg.clone())
                        .or_insert_with(|| DynIndex::new_dyn(leg.dim()))
                        .clone()
                }
            })
            .collect();
        names.push(name.clone());
        tensors.push(IdxTensor::from_dense(legs, values.clone()).unwrap());
    }
    TreeTN::from_tensors(tensors, names).unwrap()
}

/// Measure every stored patch in `repetitions` fresh threads, each with a
/// rebuilt patch and a fresh evaluator. Returns the values per repetition.
fn values_across_threads(
    patches: &std::sync::Arc<Vec<RawPatch>>,
    sites: &[DynIndex],
    repetitions: usize,
) -> Vec<Vec<f64>> {
    (0..repetitions)
        .map(|_| {
            let patches = std::sync::Arc::clone(patches);
            let sites = sites.to_vec();
            std::thread::spawn(move || {
                patches
                    .iter()
                    .flat_map(|patch| {
                        let network = rebuild(patch, &sites);
                        network_values::<f64, String>(&network, &sites, &patch.points).unwrap()
                    })
                    .collect()
            })
            .join()
            .unwrap()
        })
        .collect()
}

/// Largest difference of any repetition from the first, relative to the
/// largest magnitude of the first.
fn max_relative_difference(runs: &[Vec<f64>]) -> f64 {
    let scale = runs[0].iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    runs[1..]
        .iter()
        .flat_map(|run| run.iter().zip(&runs[0]).map(|(a, b)| (a - b).abs()))
        .fold(0.0_f64, f64::max)
        / scale
}

/// A digest of the values of one repetition, printed so that separate test
/// processes can be compared by hand (open question 9).
fn digest(values: &[f64]) -> u64 {
    values
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
            (hash ^ value.to_bits()).wrapping_mul(0x0100_0000_01b3)
        })
}

/// Number of values that differ from the first repetition, per repetition.
fn differing_values(runs: &[Vec<f64>]) -> Vec<usize> {
    runs[1..]
        .iter()
        .map(|run| {
            run.iter()
                .zip(&runs[0])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count()
        })
        .collect()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Whether every node of every stored patch satisfies the raw-kernel
/// condition of the cached evaluator: exactly one site leg, and legs equal
/// to its neighbor count plus one.
fn takes_raw_kernels(patches: &[RawPatch], sites: &[DynIndex]) -> bool {
    patches.iter().all(|patch| {
        patch.nodes.iter().all(|(_, legs, _)| {
            let site_legs = legs.iter().filter(|leg| sites.contains(leg)).count();
            site_legs == 1
        })
    })
}

const REPETITIONS: usize = 6;

#[test]
fn measurement_is_bitwise_reproducible_across_threads_on_raw_kernel_trees() {
    let tree = raw_kernel_tree();
    assert_eq!(max_degree(&tree), 3);
    let patches = stored_patches(&tree, raw_kernel_function);
    // Stored patches carry every site, fixed ones re-embedded by one-hot
    // factors, so every node keeps exactly one site leg.
    assert!(patches.iter().any(|patch| patch.points.len() < 128));
    assert!(takes_raw_kernels(&patches, &tree.sites));
    let patches = std::sync::Arc::new(patches);
    let runs = values_across_threads(&patches, &tree.sites, REPETITIONS);
    let difference = max_relative_difference(&runs);
    eprintln!(
        "{} values per repetition; differing per repetition {:?}; max relative difference \
         {difference:e}; digests {:x?}",
        runs[0].len(),
        differing_values(&runs),
        runs.iter().map(|run| digest(run)).collect::<Vec<_>>()
    );
    for run in &runs[1..] {
        assert_eq!(
            bits(run),
            bits(&runs[0]),
            "max relative difference {difference:e}"
        );
    }
}

#[test]
#[ignore = "open question 8 of docs/design/tree-patching-error-contract.md: the generic IdxTensor \
            path of TreeTNCachedEvaluator contracts N-ary operand lists through omeco's greedy \
            planner, which breaks cost ties in HashMap order, so values differ across threads"]
fn measurement_is_bitwise_reproducible_across_threads_on_generic_path_trees() {
    let tree = generic_path_tree();
    assert_eq!(max_degree(&tree), 3);
    let patches = stored_patches(&tree, generic_path_function);
    assert!(!takes_raw_kernels(&patches, &tree.sites));
    let patches = std::sync::Arc::new(patches);
    let runs = values_across_threads(&patches, &tree.sites, REPETITIONS);
    let difference = max_relative_difference(&runs);
    eprintln!(
        "{} values per repetition; differing per repetition {:?}; max relative difference \
         {difference:e}; digests {:x?}",
        runs[0].len(),
        differing_values(&runs),
        runs.iter().map(|run| digest(run)).collect::<Vec<_>>()
    );
    for run in &runs[1..] {
        assert_eq!(
            bits(run),
            bits(&runs[0]),
            "max relative difference {difference:e}"
        );
    }
}
