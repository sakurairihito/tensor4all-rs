use super::*;
use std::collections::BTreeMap;
use tensor4all_treetn::NodeNameNetwork;

#[test]
fn map_termination_applies_the_cap_precedence() {
    let cap = NonZeroUsize::new(4);
    let cases = [
        // The criterion met strictly below the cap is accepted.
        (
            TreeTciTermination::Converged,
            3,
            cap,
            InterpolationTermination::Converged,
        ),
        (
            TreeTciTermination::Converged,
            9,
            None,
            InterpolationTermination::Converged,
        ),
        // The criterion met at a rank equal to the cap is not.
        (
            TreeTciTermination::Converged,
            4,
            cap,
            InterpolationTermination::BondCapReached,
        ),
        (
            TreeTciTermination::MaxBondDimension,
            4,
            cap,
            InterpolationTermination::BondCapReached,
        ),
        (
            TreeTciTermination::MaxIterations,
            2,
            cap,
            InterpolationTermination::IterationLimit,
        ),
        (
            TreeTciTermination::MaxIterations,
            2,
            None,
            InterpolationTermination::IterationLimit,
        ),
    ];
    for (reason, rank, cap, expected) in cases {
        assert_eq!(
            map_termination(reason, rank, cap),
            expected,
            "{reason:?} at rank {rank} with cap {cap:?}"
        );
    }
}

#[test]
fn classify_separates_evaluator_failures_from_engine_failures() {
    // Marker directly under TreeTciError::Operation: the original error is kept.
    let wrapped = TreeTciError::from(anyhow::Error::new(EvaluatorFailure {
        source: anyhow::anyhow!("user failure"),
    }));
    match classify(wrapped) {
        InterpolationError::Evaluator { source } => assert_eq!(source.to_string(), "user failure"),
        other => panic!("expected Evaluator, got {other:?}"),
    }

    // Marker under extra context: still an evaluator failure.
    let with_context = TreeTciError::from(
        anyhow::Error::new(EvaluatorFailure {
            source: anyhow::anyhow!("deep failure"),
        })
        .context("while filling a site tensor"),
    );
    match classify(with_context) {
        InterpolationError::Evaluator { source } => {
            assert!(format!("{source:#}").contains("deep failure"))
        }
        other => panic!("expected Evaluator, got {other:?}"),
    }

    // No marker: an engine failure.
    assert!(matches!(
        classify(TreeTciError::from(anyhow::anyhow!("singular pivot matrix"))),
        InterpolationError::Engine { .. }
    ));
    assert!(matches!(
        classify(TreeTciError::InvalidGraph {
            message: "cycle".to_string()
        }),
        InterpolationError::Engine { .. }
    ));
    assert!(matches!(
        classify_anyhow(anyhow::anyhow!("shape mismatch")),
        InterpolationError::Engine { .. }
    ));
}

/// Chain "a" - "b" - "c" (max degree 2): "a" has sites of dimensions 2 and 3
/// (fused, local dimension 6), "b" none, "c" one site of dimension 2.
fn fused_chain_problem() -> InterpolationProblem<String> {
    let mut topology = NodeNameNetwork::new();
    for node in ["a", "b", "c"] {
        topology.add_node(node.to_string()).unwrap();
    }
    topology
        .add_edge(&"a".to_string(), &"b".to_string())
        .unwrap();
    topology
        .add_edge(&"b".to_string(), &"c".to_string())
        .unwrap();
    let node_sites = BTreeMap::from([
        (
            "a".to_string(),
            vec![DynIndex::new_dyn(2), DynIndex::new_dyn(3)],
        ),
        ("b".to_string(), vec![]),
        ("c".to_string(), vec![DynIndex::new_dyn(2)]),
    ]);
    let pivots = ColMajorArray::new(vec![1, 2, 1], vec![3, 1]).unwrap();
    InterpolationProblem::new(topology, node_sites, pivots, 0.0, None, 0).unwrap()
}

#[test]
fn layout_fuses_column_major_and_round_trips() {
    let problem = fused_chain_problem();
    let layout = VertexLayout::new(&problem).unwrap();
    assert_eq!(layout.local_dims, vec![6, 1, 2]);
    assert_eq!(layout.site_offsets, vec![0, 2, 2, 3]);

    // (a0, a1) = (1, 2) fuses to 1 + 2 * 2 = 5; "b" has coordinate 0.
    let vertices = layout
        .site_columns_to_vertices(problem.initial_pivots())
        .unwrap();
    assert_eq!(vertices, vec![vec![5, 0, 1]]);

    // Every vertex point splits back to the site point it came from.
    let mut sites = vec![usize::MAX; 3];
    for fused in 0..6 {
        for c in 0..2 {
            layout.split_point(&[fused, 0, c], &mut sites).unwrap();
            assert_eq!(sites, vec![fused % 2, fused / 2, c]);
            let column = ColMajorArray::new(sites.clone(), vec![3, 1]).unwrap();
            assert_eq!(
                layout.site_columns_to_vertices(&column).unwrap(),
                vec![vec![fused, 0, c]]
            );
        }
    }

    // A batch of two vertex points becomes a [3, 2] site batch.
    let data = [5, 0, 1, 0, 0, 0];
    let batch = GlobalIndexBatch::new(&data, 3, 2).unwrap();
    assert_eq!(
        layout.vertex_batch_to_sites(batch).unwrap(),
        vec![1, 2, 1, 0, 0, 0]
    );
}

#[test]
fn layout_rejects_malformed_vertex_points() {
    let problem = fused_chain_problem();
    let layout = VertexLayout::new(&problem).unwrap();
    let mut sites = vec![0usize; 3];
    let error = layout.split_point(&[6, 0, 0], &mut sites).unwrap_err();
    assert!(error.to_string().contains("out of range"));
    let error = layout.split_point(&[0, 0], &mut sites).unwrap_err();
    assert!(error.to_string().contains("expected 3 and 3"));

    let data = [0, 0];
    let batch = GlobalIndexBatch::new(&data, 2, 1).unwrap();
    let error = layout.vertex_batch_to_sites(batch).unwrap_err();
    assert!(error.to_string().contains("expected 3"));
}

#[test]
fn call_evaluator_marks_errors_and_wrong_lengths() {
    let data = [0usize, 1, 1, 0];
    let short = |_: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> { Ok(vec![1.0]) };
    let error = call_evaluator(&short, &data, 2).unwrap_err();
    assert!(has_evaluator_failure(error.as_ref()));
    assert!(format!("{error:#}").contains("returned 1 values for 2 points"));

    let failing =
        |_: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> { anyhow::bail!("nope") };
    let error = call_evaluator(&failing, &data, 2).unwrap_err();
    assert!(has_evaluator_failure(error.as_ref()));

    let echo = |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
        assert_eq!(batch.shape(), &[2, 2]);
        Ok(batch
            .data()
            .chunks(2)
            .map(|p| (p[0] * 10 + p[1]) as f64)
            .collect())
    };
    assert_eq!(call_evaluator(&echo, &data, 2).unwrap(), vec![1.0, 10.0]);
}
