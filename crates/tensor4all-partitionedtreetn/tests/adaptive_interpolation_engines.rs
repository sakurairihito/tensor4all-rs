//! `patched_interpolate` end to end with the TreeTCI engine, and with a
//! fiber test engine on a domain wider than 128 bits.

mod adaptive_common;

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use adaptive_common::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{outer_product, ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::interpolation::{
    InterpolationError, InterpolationOutcome, InterpolationProblem, InterpolationTermination,
    TreeInterpolator,
};
use tensor4all_treetn::TreeTN;

// ---------------------------------------------------------------------------
// TreeTCI end to end
// ---------------------------------------------------------------------------

/// Quantics coordinate in [0, 1) of `bits` (most significant first).
fn quantics(bits: &[usize]) -> f64 {
    bits.iter()
        .enumerate()
        .map(|(k, &bit)| bit as f64 * 0.5_f64.powi(k as i32 + 1))
        .sum()
}

fn gaussian(x: f64, center: f64, width: f64) -> f64 {
    (-((x - center) / width).powi(2)).exp()
}

/// Largest magnitude of `f` over the whole (small) domain.
fn max_abs(problem: &Problem, f: &dyn Fn(&[usize]) -> f64) -> f64 {
    full_domain(&problem.dims())
        .iter()
        .map(|p| f(p).abs())
        .fold(0.0, f64::max)
}

#[test]
fn treetci_patches_a_localized_function_on_a_quantics_chain() {
    // 128 points; two narrow peaks give the whole domain a rank above the cap.
    let names: Vec<String> = (0..7).map(|k| format!("q{k}")).collect();
    let nodes: Vec<(&str, &[usize])> = names.iter().map(|n| (n.as_str(), &[2usize][..])).collect();
    let edges: Vec<(&str, &str)> = names
        .windows(2)
        .map(|w| (w[0].as_str(), w[1].as_str()))
        .collect();
    let problem = Problem::new(&nodes, &edges);
    let f = |p: &[usize]| {
        let x = quantics(p);
        gaussian(x, 0.3, 0.02) + 0.5 * gaussian(x, 0.71, 0.05)
    };
    let rtol = 1e-8;
    let options = PatchedInterpolationOptions::new(4)
        .with_rtol(rtol)
        .with_reference_scale(max_abs(&problem, &f));
    let result = run(
        &TreeTciInterpolator::default(),
        &problem,
        &f,
        &[vec![0, 1, 0, 0, 1, 1, 0]],
        &options,
    )
    .unwrap();
    assert!(result.report.splits >= 1, "the root must not converge");
    assert!(result
        .report
        .accepted
        .iter()
        .any(|record| record.max_bond_dim >= 2));
    assert_accurate(&result, &problem, &f, rtol);
}

/// Two quantics variables on branches of a site-free junction `r` of degree
/// three; the third branch `z` is a binary flag.
fn quantics_tree() -> Problem {
    Problem::new(
        &[
            ("r", &[]),
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
        ],
    )
}

/// A peak at (0.3, 0.6) whose height depends on the flag; site order
/// x0, x1, x2, y0, y1, y2, z.
fn tree_peak(p: &[usize]) -> f64 {
    let (x, y) = (quantics(&p[0..3]), quantics(&p[3..6]));
    (1.0 + 0.5 * p[6] as f64) * gaussian(x, 0.3, 0.12) * gaussian(y, 0.6, 0.12)
        + 0.2 * (x * y + p[6] as f64 * x)
}

fn quantics_tree_options(problem: &Problem, recycle: bool) -> PatchedInterpolationOptions {
    let order = ["x0", "y0", "x1", "y1", "x2", "y2", "z"]
        .iter()
        .map(|node| problem.site(node, 0))
        .collect();
    PatchedInterpolationOptions::new(4)
        .with_rtol(1e-8)
        .with_reference_scale(max_abs(problem, &tree_peak))
        .with_patch_order(order)
        .with_recycle_pivots(recycle)
        .with_seed(5)
}

#[test]
fn treetci_patches_a_function_on_a_branched_tree_deterministically() {
    let problem = quantics_tree();
    let index = problem.topology.node_index(&"r".to_string()).unwrap();
    assert_eq!(problem.topology.graph().neighbors(index).count(), 3);
    let engine = TreeTciInterpolator::default();
    let pivots = [vec![1, 0, 0, 1, 0, 0, 1]];
    let mut runs = Vec::new();
    for recycle in [false, true, true] {
        let options = quantics_tree_options(&problem, recycle);
        let result = run(&engine, &problem, &tree_peak, &pivots, &options).unwrap();
        assert!(result.report.splits >= 1, "the root must not converge");
        assert_accurate(&result, &problem, &tree_peak, options.rtol);
        runs.push(result);
    }
    assert_same_run(&runs[1], &runs[2]);
}

/// A test engine for product functions: it samples the fibers through the
/// first initial pivot `p` and returns the exact rank-one network
/// `f(p) * prod_i f(p with site i = x_i) / f(p)`, without dense evaluation.
struct FiberEngine;

impl TreeInterpolator<f64> for FiberEngine {
    fn interpolate<V, F>(
        &self,
        problem: &InterpolationProblem<V>,
        evaluate: F,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
        F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<f64>>,
    {
        let evaluator = |source: anyhow::Error| InterpolationError::Evaluator { source };
        let pivot = problem.initial_pivots().column(0).unwrap().to_vec();
        let n = pivot.len();
        let dims: Vec<usize> = problem.site_order().iter().map(IndexLike::dim).collect();
        let mut fibers = Vec::new();
        for (i, &dim) in dims.iter().enumerate() {
            for x in 0..dim {
                let mut point = pivot.clone();
                point[i] = x;
                fibers.extend(point);
            }
        }
        let shape = [n, fibers.len() / n];
        let values =
            evaluate(ColMajorArrayRef::new(&fibers, &shape).unwrap()).map_err(evaluator)?;
        let center = values[pivot[0]];
        let mut factors = Vec::new();
        let mut offset = 0;
        for &dim in &dims {
            factors.push(
                values[offset..offset + dim]
                    .iter()
                    .map(|v| v / center)
                    .collect::<Vec<_>>(),
            );
            offset += dim;
        }
        factors[0].iter_mut().for_each(|v| *v *= center);

        let topology = problem.topology();
        let graph = topology.graph();
        let mut links: HashMap<V, Vec<DynIndex>> = HashMap::new();
        for edge in graph.edge_indices() {
            let (a, b) = graph.edge_endpoints(edge).unwrap();
            let link = DynIndex::new_dyn(1);
            for node in [a, b] {
                let name = topology.node_name(node).unwrap().clone();
                links.entry(name).or_default().push(link.clone());
            }
        }
        let mut position = 0;
        let mut names = Vec::new();
        let mut tensors = Vec::new();
        for (node, sites) in problem.node_sites() {
            let mut tensor =
                IdxTensor::from_dense(links.remove(node).unwrap_or_default(), vec![1.0]).unwrap();
            for site in sites {
                let factor =
                    IdxTensor::from_dense(vec![site.clone()], factors[position].clone()).unwrap();
                tensor = outer_product(&tensor, &factor).unwrap();
                position += 1;
            }
            names.push(node.clone());
            tensors.push(tensor);
        }
        Ok(InterpolationOutcome {
            network: TreeTN::from_tensors(tensors, names).unwrap(),
            termination: InterpolationTermination::Converged,
            error_estimate: 0.0,
            max_sample_magnitude: values.iter().fold(0.0, |m, v| m.max(v.abs())),
            pivots: None,
        })
    }
}

#[test]
fn the_cache_supports_domains_wider_than_128_bits() {
    // Three variables of 43 bits each on a chain of 129 binary sites.
    let names: Vec<String> = (0..129).map(|k| format!("s{k:03}")).collect();
    let nodes: Vec<(&str, &[usize])> = names.iter().map(|n| (n.as_str(), &[2usize][..])).collect();
    let edges: Vec<(&str, &str)> = names
        .windows(2)
        .map(|w| (w[0].as_str(), w[1].as_str()))
        .collect();
    let problem = Problem::new(&nodes, &edges);
    // exp(x + y + z) is a product over the bits.
    let f = |p: &[usize]| p.chunks(43).map(quantics).sum::<f64>().exp();
    let options = PatchedInterpolationOptions::new(2).with_reference_scale(20.0);
    let result = run(
        &FiberEngine,
        &problem,
        &f,
        &[vec![0; 129], vec![1; 129]],
        &options,
    )
    .unwrap();
    assert_eq!(result.report.splits, 0);
    assert_eq!(result.report.accepted.len(), 1);
    // Five candidates (two user pivots, three random points), then the 129
    // new fiber points; the fibers repeat the cached pivot 129 times.
    assert_eq!(result.report.function_evaluations, 5 + 129);
    assert_eq!(result.report.cache_hits, 129);

    // Sampled comparison: materializing 2^129 values is impossible.
    let network = result.partition.to_treetn().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(129);
    for _ in 0..16 {
        let point: Vec<usize> = (0..129).map(|_| rng.random_range(0..2)).collect();
        let value = network
            .evaluate_point(&problem.sites, &point)
            .unwrap()
            .real();
        let expected = f(&point);
        assert!(
            (value - expected).abs() <= 1e-12 * expected,
            "{value} vs {expected}"
        );
    }
}
