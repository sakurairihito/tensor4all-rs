//! Driver-side measurement of patch networks.
//!
//! The floating-point operation order of a measurement must be a function of
//! the sorted node names, the positional legs of the stored node tensors, and
//! the batch contents only (see "Determinism" in
//! `docs/design/tree-patching-error-contract.md`). The driver's part of that
//! argument lives here: every measurement uses one fresh
//! [`TreeTNCachedEvaluator`], the smallest node name as a fixed center (no
//! greedy center search), [`EvaluationHint::default`] for every batch, and
//! points in a deterministic order evaluated in chunks of the fixed size
//! [`MEASUREMENT_CHUNK`].

use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, TensorElement};
use tensor4all_treetn::{
    CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator, TreeTNOperationError,
};

/// Number of points per evaluator batch of a measurement. Fixed, so that the
/// batch composition and call history of a measurement depend only on its
/// point list.
pub(super) const MEASUREMENT_CHUNK: usize = 256;

/// Values of `network` at the column-major `[sites.len(), n_points]` points
/// `points`, with a fresh evaluator centered at the smallest node name, the
/// default hint, and chunks of [`MEASUREMENT_CHUNK`] points.
pub(super) fn network_values<T, V>(
    network: &TreeTN<IdxTensor, V>,
    sites: &[DynIndex],
    points: &[usize],
) -> Result<Vec<T>, TreeTNOperationError>
where
    T: TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let n_sites = sites.len();
    if n_sites == 0 || points.is_empty() {
        return Ok(Vec::new());
    }
    let center = network.node_names().into_iter().min();
    let options = CachedEvaluatorOptions {
        center,
        ..CachedEvaluatorOptions::default()
    };
    let mut evaluator = TreeTNCachedEvaluator::new(network, sites, options)?;
    let mut values = Vec::with_capacity(points.len() / n_sites);
    for chunk in points.chunks(MEASUREMENT_CHUNK * n_sites) {
        let shape = [n_sites, chunk.len() / n_sites];
        let batch = ColMajorArrayRef::new(chunk, &shape)
            .map_err(|error| TreeTNOperationError::from(anyhow::Error::new(error)))?;
        values.extend(evaluator.evaluate_batched_typed::<T>(batch, EvaluationHint::default())?);
    }
    Ok(values)
}

#[cfg(test)]
mod tests;
