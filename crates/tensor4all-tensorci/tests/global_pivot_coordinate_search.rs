//! Regression for #806, with explicit starts and literal residual maxima.
use rand::{Rng, RngCore};
use tensor4all_simplett::SimpleTensorTrain;
use tensor4all_tensorci::{
    DefaultGlobalPivotFinder, GlobalPivotFinder, GlobalPivotSearchInput, TCIError,
};

// Supplies the all-zero starting point through the RNG-only search interface.
struct ZeroStream;
impl RngCore for ZeroStream {
    fn next_u32(&mut self) -> u32 {
        0
    }
    fn next_u64(&mut self) -> u64 {
        0
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        dest.fill(0);
    }
}

fn input(n: usize) -> GlobalPivotSearchInput<f64> {
    GlobalPivotSearchInput {
        local_dims: vec![n, n],
        current_tt: SimpleTensorTrain::constant(&[n, n], 0.0),
        max_sample_value: 1000.0,
        i_set: vec![vec![vec![]], vec![vec![0]]],
        j_set: vec![vec![vec![0]], vec![vec![]]],
    }
}

#[test]
fn retains_coordinate_moves_to_find_the_above_threshold_corner() {
    let mut rng = ZeroStream;
    // A wrong start adapter must not make a weak finder pass accidentally.
    assert_eq!([rng.random_range(0..4), rng.random_range(0..4)], [0, 0]);
    let pivots = DefaultGlobalPivotFinder::new(1, 1, 10.0)
        .find_global_pivots(&input(4), &|p| (p[0] + p[1]) as f64, 0.5, &mut rng)
        .unwrap();
    assert_eq!(pivots, vec![vec![3, 3]]);
}

#[test]
fn repeats_coordinate_sweeps_until_the_coupled_maximum_is_found() {
    // First sweep from (0,0) reaches (2,2), residual 996. A second reaches
    // the unique maximum (3,3), residual 1000, above the threshold 999.5.
    let f = |p: &Vec<usize>| {
        let (x, y) = (p[0] as f64, p[1] as f64);
        1000.0 - 4.0 * (x - 3.0).powi(2) - (y - x).powi(2)
    };
    let pivots = DefaultGlobalPivotFinder::new(1, 1, 10.0)
        .find_global_pivots(&input(8), &f, 99.95, &mut ZeroStream)
        .unwrap();
    assert_eq!(pivots, vec![vec![3, 3]]);
}

#[test]
fn early_stop_is_checked_after_a_complete_sweep() {
    let f = |p: &Vec<usize>| {
        let (x, y) = (p[0] as f64, p[1] as f64);
        1000.0 - 4.0 * (x - 3.0).powi(2) - (y - x).powi(2)
    };
    // The start already exceeds the early-stop bound (10 * 10 * 1),
    // but Julia still performs a complete sweep before checking it.
    let pivots = DefaultGlobalPivotFinder::new(1, 1, 10.0)
        .find_global_pivots(&input(8), &f, 1.0, &mut ZeroStream)
        .unwrap();
    assert_eq!(pivots, vec![vec![2, 2]]);
}

#[test]
fn searches_residual_magnitude_including_complex_values() {
    use num_complex::Complex64;
    // |f - tt| = |i + j - 6|: the largest function value is the *smallest*
    // residual. Use an imaginary TT to exercise the complex magnitude too.
    let input = GlobalPivotSearchInput {
        local_dims: vec![4, 4],
        current_tt: SimpleTensorTrain::constant(&[4, 4], Complex64::new(0.0, 6.0)),
        max_sample_value: 6.0,
        i_set: vec![],
        j_set: vec![],
    };
    let pivots = DefaultGlobalPivotFinder::new(1, 1, 10.0)
        .find_global_pivots(
            &input,
            &|p| Complex64::new(0.0, (p[0] + p[1]) as f64),
            0.5,
            &mut ZeroStream,
        )
        .unwrap();
    assert_eq!(pivots, vec![vec![0, 0]]);
}

#[test]
fn preserves_strict_threshold_and_pivot_cap() {
    let f = |p: &Vec<usize>| (p[0] + p[1]) as f64;
    let finder = DefaultGlobalPivotFinder::new(3, 2, 10.0);
    assert_eq!(
        finder
            .find_global_pivots(&input(4), &f, 0.5, &mut ZeroStream)
            .unwrap(),
        vec![vec![3, 3]; 2]
    );
    assert!(finder
        .find_global_pivots(&input(4), &f, 0.6, &mut ZeroStream)
        .unwrap()
        .is_empty());
    assert!(finder
        .find_global_pivots(&input(4), &|_| 0.0, 0.0, &mut ZeroStream)
        .unwrap()
        .is_empty());
    // A finite acceptance threshold may have an infinite early-stop bound.
    assert!(finder
        .find_global_pivots(&input(4), &f, f64::MAX / 16.0, &mut ZeroStream)
        .unwrap()
        .is_empty());
}

#[test]
fn disabled_search_validates_inputs_without_evaluating_the_oracle() {
    let oracle =
        |_: &Vec<usize>| -> f64 { panic!("disabled or invalid search must not evaluate f") };
    for (nsearch, cap) in [(0, 1), (1, 0)] {
        let finder = DefaultGlobalPivotFinder::new(nsearch, cap, 10.0);
        assert!(finder
            .find_global_pivots(&input(4), &oracle, 0.5, &mut ZeroStream)
            .unwrap()
            .is_empty());
        for abs_tol in [-1.0, f64::NAN, f64::INFINITY, f64::MAX] {
            assert!(matches!(
                finder.find_global_pivots(&input(4), &oracle, abs_tol, &mut ZeroStream),
                Err(TCIError::InvalidConfiguration { .. })
            ));
        }
        for margin in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                DefaultGlobalPivotFinder::new(nsearch, cap, margin).find_global_pivots(
                    &input(4),
                    &oracle,
                    0.5,
                    &mut ZeroStream
                ),
                Err(TCIError::InvalidConfiguration { .. })
            ));
        }
        for dims in [vec![], vec![0, 4], vec![4], vec![4, 3]] {
            let mut bad = input(4);
            bad.local_dims = dims.clone();
            let result = finder.find_global_pivots(&bad, &oracle, 0.5, &mut ZeroStream);
            if dims.is_empty() || dims.contains(&0) {
                assert!(matches!(result, Err(TCIError::InvalidConfiguration { .. })));
            } else {
                assert!(matches!(result, Err(TCIError::DimensionMismatch { .. })));
            }
        }
    }
}

#[test]
fn nonfinite_residuals_are_errors_instead_of_missing_pivots() {
    for value in [f64::NAN, f64::INFINITY] {
        for bad_point in [vec![0, 0], vec![1, 0]] {
            let result = DefaultGlobalPivotFinder::new(1, 1, 10.0).find_global_pivots(
                &input(4),
                &|p| if *p == bad_point { value } else { 0.0 },
                0.5,
                &mut ZeroStream,
            );
            assert!(matches!(result, Err(TCIError::InvalidOperation { .. })));
        }
    }
}

#[test]
fn optimizer_propagates_custom_finder_errors() {
    use tensor4all_core::{MultiIndex, Scalar};
    use tensor4all_simplett::{EinsumScalar, TTScalar};
    use tensor4all_tensorci::{optimize_with_finder, TCI2Options, TensorCI2};
    struct FailingFinder;
    impl GlobalPivotFinder for FailingFinder {
        fn find_global_pivots<T, F>(
            &self,
            _: &GlobalPivotSearchInput<T>,
            _: &F,
            _: f64,
            _: &mut impl Rng,
        ) -> tensor4all_tensorci::Result<Vec<MultiIndex>>
        where
            T: Scalar + TTScalar + EinsumScalar,
            F: Fn(&MultiIndex) -> T,
        {
            Err(TCIError::InvalidOperation {
                message: "finder evaluation failed".into(),
            })
        }
    }
    let mut tci = TensorCI2::<f64>::new(vec![4, 4]).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();
    let result = optimize_with_finder::<f64, _, fn(&[MultiIndex]) -> Vec<f64>, _>(
        tci,
        |p| (p[0] + p[1] + 1) as f64,
        None,
        TCI2Options {
            max_iter: 1,
            seed: Some(1234),
            ..Default::default()
        },
        FailingFinder,
    );
    assert!(
        matches!(result, Err(TCIError::InvalidOperation { message }) if message == "finder evaluation failed")
    );
}
