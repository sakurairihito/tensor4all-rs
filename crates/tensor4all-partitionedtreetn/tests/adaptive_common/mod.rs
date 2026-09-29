//! Shared helpers of the adaptive interpolation integration tests.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashSet};
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use tensor4all_core::{
    ColMajorArray, ColMajorArrayRef, CommonScalar, DynIndex, IdxTensor, IndexLike, TensorElement,
};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    patched_interpolate, PatchedInterpolationError, PatchedInterpolationOptions,
    PatchedInterpolationResult,
};
use tensor4all_partitionedtreetn::Projector;
use tensor4all_treetn::interpolation::{
    InterpolationError, InterpolationProblem, TreeInterpolator,
};
use tensor4all_treetn::NodeNameNetwork;

/// Accuracy bound of accepted patches against a dense reference, in units of
/// `rtol * reference_scale`. The acceptance criterion is the engine's sampled
/// error estimate, not a verified bound, so the tests allow this margin.
pub(crate) const ACCURACY_FACTOR: f64 = 10.0;

pub(crate) type Name = String;

// ---------------------------------------------------------------------------
// Problems
// ---------------------------------------------------------------------------

/// A tree topology with the sites of every node and the derived site order.
pub(crate) struct Problem {
    pub(crate) topology: NodeNameNetwork<Name>,
    pub(crate) node_sites: BTreeMap<Name, Vec<DynIndex>>,
    pub(crate) sites: Vec<DynIndex>,
}

impl Problem {
    /// Nodes with their site dimensions, and edges.
    pub(crate) fn new(nodes: &[(&str, &[usize])], edges: &[(&str, &str)]) -> Self {
        let node_sites = nodes
            .iter()
            .map(|(node, dims)| {
                let sites = dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect();
                (node.to_string(), sites)
            })
            .collect();
        Self::with_sites(node_sites, edges)
    }

    pub(crate) fn with_sites(
        node_sites: BTreeMap<Name, Vec<DynIndex>>,
        edges: &[(&str, &str)],
    ) -> Self {
        let mut topology = NodeNameNetwork::new();
        for node in node_sites.keys() {
            topology.add_node(node.clone()).unwrap();
        }
        for (left, right) in edges {
            topology
                .add_edge(&left.to_string(), &right.to_string())
                .unwrap();
        }
        let sites = InterpolationProblem::derive_site_order(&node_sites);
        Self {
            topology,
            node_sites,
            sites,
        }
    }

    pub(crate) fn dims(&self) -> Vec<usize> {
        self.sites.iter().map(IndexLike::dim).collect()
    }

    pub(crate) fn site(&self, node: &str, k: usize) -> DynIndex {
        self.node_sites[node][k].clone()
    }

    pub(crate) fn position(&self, site: &DynIndex) -> usize {
        self.sites.iter().position(|s| s == site).unwrap()
    }

    pub(crate) fn pivots(&self, points: &[Vec<usize>]) -> ColMajorArray<usize> {
        ColMajorArray::new(points.concat(), vec![self.sites.len(), points.len()]).unwrap()
    }
}

/// Every point of a domain, column-major (first site fastest).
pub(crate) fn full_domain(dims: &[usize]) -> Vec<Vec<usize>> {
    let n_points: usize = dims.iter().product();
    (0..n_points)
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

/// Branched tree with a junction of degree three:
///
/// ```text
///   a - j - b - c - e
///       |
///       d
/// ```
///
/// `j` (junction, site j0: 2), `a` (leaf, a0: 3), `b` (internal, b0: 2),
/// `c` (internal, c0: 2), `d` (leaf, two sites d0, d1: 2), `e` (leaf, no
/// site). Site order: a0, b0, c0, d0, d1, j0.
pub(crate) fn branched() -> Problem {
    Problem::new(
        &[
            ("a", &[3]),
            ("b", &[2]),
            ("c", &[2]),
            ("d", &[2, 2]),
            ("e", &[]),
            ("j", &[2]),
        ],
        &[("j", "a"), ("j", "b"), ("j", "d"), ("b", "c"), ("c", "e")],
    )
}

/// Chain n0 - n1 - n2 with binary sites.
pub(crate) fn chain3() -> Problem {
    Problem::new(
        &[("n0", &[2]), ("n1", &[2]), ("n2", &[2])],
        &[("n0", "n1"), ("n1", "n2")],
    )
}

pub(crate) fn single_node(dims: &[usize]) -> Problem {
    Problem::new(&[("only", dims)], &[])
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/// A product function whose factors depend on `variant`.
pub(crate) fn product(point: &[usize], variant: usize) -> f64 {
    point
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            1.0 + 0.1 * ((variant + 1) * (i + 2)) as f64 * x as f64 + 0.05 * (x * x) as f64
        })
        .product()
}

/// `f = g_{x_s}(x)`: a different product function for every value of the
/// site at `position`. Rank two or more across every edge; rank one once the
/// site is fixed.
pub(crate) fn switch_on(position: usize) -> impl Fn(&[usize]) -> f64 + Sync {
    move |point| product(point, point[position])
}

// ---------------------------------------------------------------------------
// Evaluation bookkeeping and the generic run helper
// ---------------------------------------------------------------------------

/// Records every evaluated point and fails on a repeated one.
#[derive(Default)]
pub(crate) struct Recorder {
    pub(crate) seen: Mutex<HashSet<Vec<usize>>>,
    pub(crate) calls: AtomicUsize,
}

impl Recorder {
    pub(crate) fn points(&self) -> usize {
        self.seen.lock().unwrap().len()
    }
}

pub(crate) fn recording_evaluator<'a, T>(
    f: &'a (dyn Fn(&[usize]) -> T + Sync),
    n_sites: usize,
    recorder: &'a Recorder,
) -> impl Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>> + Send + Sync + 'a {
    move |batch| {
        assert_eq!(batch.shape()[0], n_sites, "batch rows");
        recorder.calls.fetch_add(1, Ordering::SeqCst);
        let mut seen = recorder.seen.lock().unwrap();
        Ok(batch
            .data()
            .chunks(n_sites)
            .map(|point| {
                assert!(
                    seen.insert(point.to_vec()),
                    "point {point:?} evaluated twice"
                );
                f(point)
            })
            .collect())
    }
}

/// The path of a projector in split order: (position, coordinate) pairs.
pub(crate) fn path_of(
    projector: &Projector,
    split_order: &[DynIndex],
    problem: &Problem,
) -> Vec<(usize, usize)> {
    split_order
        .iter()
        .filter_map(|site| {
            projector
                .get(site)
                .map(|value| (problem.position(site), value))
        })
        .collect()
}

/// Check the partition and report against the problem: accepted and zero
/// projectors are disjoint and cover the domain, both lists are in canonical
/// path order, and the partition holds exactly the accepted projectors.
pub(crate) fn check_invariants(
    result: &PatchedInterpolationResult<Name>,
    problem: &Problem,
    options: &PatchedInterpolationOptions,
) {
    let split_order = if options.patch_order.is_empty() {
        problem.sites.clone()
    } else {
        options.patch_order.clone()
    };
    let report = &result.report;
    let accepted: Vec<&Projector> = report.accepted.iter().map(|r| &r.projector).collect();
    for list in [accepted.clone(), report.zero_projectors.iter().collect()] {
        let paths: Vec<_> = list
            .iter()
            .map(|projector| path_of(projector, &split_order, problem))
            .collect();
        assert!(
            paths.windows(2).all(|w| w[0] < w[1]),
            "not canonical: {paths:?}"
        );
    }
    let all: Vec<&Projector> = accepted
        .iter()
        .copied()
        .chain(report.zero_projectors.iter())
        .collect();
    for (i, left) in all.iter().enumerate() {
        for right in &all[i + 1..] {
            assert!(
                !left.is_compatible_with(right),
                "{left:?} overlaps {right:?}"
            );
        }
    }
    // Volumes as f64: exact for these power-of-two and small domains, and
    // free of overflow for the 129-site domain.
    let volume = |sites: &mut dyn Iterator<Item = &DynIndex>| {
        sites.map(|site| site.dim() as f64).product::<f64>()
    };
    let covered: f64 = all
        .iter()
        .map(|projector| {
            volume(
                &mut problem
                    .sites
                    .iter()
                    .filter(|site| !projector.is_projected_at(site)),
            )
        })
        .sum();
    assert_eq!(covered, volume(&mut problem.sites.iter()));
    assert_eq!(result.partition.len(), accepted.len());
    for projector in accepted {
        assert!(result.partition.contains(projector));
    }
}

/// Run the driver with a recording evaluator. On success, check the
/// invariants and that the reported evaluation count matches the evaluator.
pub(crate) fn run<T, E>(
    engine: &E,
    problem: &Problem,
    f: &(dyn Fn(&[usize]) -> T + Sync),
    pivots: &[Vec<usize>],
    options: &PatchedInterpolationOptions,
) -> Result<PatchedInterpolationResult<Name>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement,
    E: TreeInterpolator<T> + Sync,
{
    let recorder = Recorder::default();
    let initial = if pivots.is_empty() {
        ColMajorArray::new(vec![], vec![problem.sites.len(), 0]).unwrap()
    } else {
        problem.pivots(pivots)
    };
    let result = patched_interpolate(
        engine,
        problem.topology.clone(),
        problem.node_sites.clone(),
        initial,
        recording_evaluator(f, problem.sites.len(), &recorder),
        options,
    );
    if let Ok(result) = &result {
        assert_eq!(result.report.function_evaluations, recorder.points());
        check_invariants(result, problem, options);
    }
    result
}

/// Materialize the partition once and return `maxabs(partition - f)` and
/// `max |f|` over the whole domain.
pub(crate) fn dense_residual<T>(
    result: &PatchedInterpolationResult<Name>,
    problem: &Problem,
    f: &dyn Fn(&[usize]) -> T,
) -> (f64, f64)
where
    T: CommonScalar + TensorElement,
{
    let values: Vec<T> = full_domain(&problem.dims()).iter().map(|p| f(p)).collect();
    let reference = IdxTensor::from_dense(problem.sites.clone(), values).unwrap();
    let max_abs = reference.maxabs().unwrap();
    if result.partition.is_empty() {
        return (max_abs, max_abs);
    }
    let dense = result
        .partition
        .to_treetn()
        .unwrap()
        .contract_to_tensor()
        .unwrap();
    (dense.sub(&reference).unwrap().maxabs().unwrap(), max_abs)
}

pub(crate) fn assert_accurate<T>(
    result: &PatchedInterpolationResult<Name>,
    problem: &Problem,
    f: &dyn Fn(&[usize]) -> T,
    rtol: f64,
) where
    T: CommonScalar + TensorElement,
{
    let (residual, _) = dense_residual(result, problem, f);
    let bound = ACCURACY_FACTOR * rtol * result.report.reference_scale;
    assert!(residual <= bound, "residual {residual} exceeds {bound}");
}

pub(crate) fn expect_invalid<T: Debug>(
    result: Result<T, PatchedInterpolationError>,
    needle: &str,
) -> String {
    match result {
        Err(PatchedInterpolationError::InvalidInput { message }) => {
            assert!(message.contains(needle), "{message:?} lacks {needle:?}");
            message
        }
        other => panic!("expected InvalidInput mentioning {needle:?}, got {other:?}"),
    }
}

pub(crate) fn expect_interpolation<T: Debug>(
    result: Result<T, PatchedInterpolationError>,
) -> (Projector, InterpolationError) {
    match result {
        Err(PatchedInterpolationError::Interpolation { projector, source }) => (projector, source),
        other => panic!("expected an Interpolation error, got {other:?}"),
    }
}

/// Two runs are identical: the same records in the same order, the same
/// counts, and bit-identical patch tensors.
pub(crate) fn assert_same_run(
    first: &PatchedInterpolationResult<Name>,
    second: &PatchedInterpolationResult<Name>,
) {
    let (a, b) = (&first.report, &second.report);
    assert_eq!(a.reference_scale, b.reference_scale);
    assert_eq!(a.splits, b.splits);
    assert_eq!(a.function_evaluations, b.function_evaluations);
    assert_eq!(a.cache_hits, b.cache_hits);
    assert_eq!(a.zero_projectors, b.zero_projectors);
    assert_eq!(a.accepted.len(), b.accepted.len());
    for (x, y) in a.accepted.iter().zip(&b.accepted) {
        assert_eq!(x.projector, y.projector);
        assert_eq!(x.termination, y.termination);
        assert_eq!(x.error_estimate, y.error_estimate);
        assert_eq!(x.max_sample_magnitude, y.max_sample_magnitude);
        assert_eq!(x.max_bond_dim, y.max_bond_dim);
    }
    for (projector, left) in first.partition.iter() {
        let right = second.partition.get(projector).unwrap();
        let (left, right) = (left.data(), right.data());
        for name in left.node_names() {
            let x = left.tensor(left.node_index(&name).unwrap()).unwrap();
            let y = right.tensor(right.node_index(&name).unwrap()).unwrap();
            assert_eq!(x.dims(), y.dims(), "node {name}");
            assert_eq!(
                x.to_vec::<f64>().unwrap(),
                y.to_vec::<f64>().unwrap(),
                "node {name}"
            );
        }
    }
}
