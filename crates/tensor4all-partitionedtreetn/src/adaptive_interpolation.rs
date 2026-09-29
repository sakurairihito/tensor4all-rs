//! Adaptive patched interpolation of a function on a tree.
//!
//! [`patched_interpolate`] runs a tree interpolation engine (any
//! [`TreeInterpolator`]) on the whole domain of a function. Wherever the
//! engine does not converge strictly below the bond cap, it fixes the next
//! site of a given order and retries on every child region. The result is a
//! [`PartitionedTreeTN`] with disjoint, eagerly masked patches, together with
//! a [`PatchedInterpolationReport`].
//!
//! # Derivation notice
//!
//! The patch queue, the accept-or-split flow, and pivot recycling are derived
//! from `adaptiveinterpolate`, `createpatch`, and `_globalpivots` in
//! [TCIAlgorithms.jl](https://github.com/tensor4all/TCIAlgorithms.jl) at
//! commit e501032278c9dd41b46c5851d8238169c8d178c5 (MIT license; Copyright
//! 2023 Ritter.Marc and contributors), through the chain driver of the
//! deprecated `tensor4all-partitionedtt` crate. See
//! `LICENSE-TCIALGORITHMS-MIT` in this crate. The tree generalization, the
//! evaluation cache, the sampled-zero policy, and the re-embedding of fixed
//! sites are original to this crate.
//!
//! # Algorithm
//!
//! 1. The inputs are validated before any evaluation; the topology and site
//!    checks are those of [`validate_layout`](tensor4all_treetn::interpolation::validate_layout).
//! 2. Patches are processed in FIFO order, starting from the whole domain.
//! 3. Each patch owns an evaluation cache of its points; a split hands every
//!    cached value to the child that contains it, so no point is evaluated
//!    twice.
//! 4. A patch with at most one active (unfixed) site is evaluated exactly and
//!    needs no engine. Otherwise the candidate pivots of the patch are
//!    sampled; if every sample is exactly zero the patch is a zero patch,
//!    reported in [`PatchedInterpolationReport::zero_projectors`] and omitted
//!    from the partition. This is a finite-sampling policy: a function that is
//!    nonzero only on a few points needs initial pivots in its support.
//! 5. The engine runs on the active sites with the absolute tolerance
//!    `rtol * reference_scale`. A [`InterpolationTermination::Converged`]
//!    outcome is accepted; any other verdict splits the patch at the next
//!    unfixed site of [`PatchedInterpolationOptions::patch_order`].
//! 6. An accepted network gets every fixed site re-attached to its node by a
//!    one-hot factor of the patch scalar type.
//!
//! # Error criterion
//!
//! Acceptance uses the engine's sampled error estimate compared with
//! `rtol * reference_scale`. It is not a verified error bound and makes no
//! claim about the L2 error of the result.
//!
//! # Randomness and determinism
//!
//! Every patch derives two sub-seeds from [`PatchedInterpolationOptions::seed`]
//! and its path, the (position in the derived site order, coordinate) pairs
//! of its fixed sites: one for its random candidate pivots and one for the
//! engine. The generator is SplitMix64, and a coordinate in `0..d` is drawn
//! with Lemire's unbiased multiply-shift method with rejection. Unlike other
//! randomized algorithms of this workspace, the driver offers no API taking a
//! caller-owned `&mut R`: one shared stream would make the randomness of a
//! patch depend on the processing order. For a fixed seed, a deterministic
//! evaluator, and a deterministic engine, the partition and the report are
//! identical across runs.
//!
//! # Examples
//!
//! Interpolate `f(x) = 1 / (1 + x)` on eight points, `x = b0 + 2 b1 + 4 b2`,
//! with one binary site per node of a three-node chain. A bond cap of two
//! only accepts rank-one patches, so the domain is split twice.
//!
//! ```
//! use std::collections::BTreeMap;
//! use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor};
//! use tensor4all_partitionedtreetn::adaptive_interpolation::{
//!     patched_interpolate, PatchedInterpolationOptions,
//! };
//! use tensor4all_treetci::TreeTciInterpolator;
//! use tensor4all_treetn::NodeNameNetwork;
//!
//! let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
//! let mut topology = NodeNameNetwork::new();
//! for node in 0..3usize {
//!     topology.add_node(node)?;
//! }
//! topology.add_edge(&0, &1)?;
//! topology.add_edge(&1, &2)?;
//! let node_sites: BTreeMap<usize, Vec<DynIndex>> =
//!     (0..3).map(|node| (node, vec![sites[node].clone()])).collect();
//!
//! let f = |point: &[usize]| 1.0 / (1.0 + (point[0] + 2 * point[1] + 4 * point[2]) as f64);
//! let evaluate = |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
//!     Ok(batch.data().chunks(3).map(f).collect())
//! };
//! let options = PatchedInterpolationOptions::new(2).with_rtol(1e-12);
//! let result = patched_interpolate(
//!     &TreeTciInterpolator::default(),
//!     topology,
//!     node_sites,
//!     ColMajorArray::new(vec![0, 0, 0], vec![3, 1])?,
//!     evaluate,
//!     &options,
//! )?;
//!
//! // The splits fix sites 0 and 1; every patch then has one active site.
//! assert_eq!(result.report.splits, 3);
//! assert_eq!(result.partition.len(), 4);
//! assert_eq!(result.report.function_evaluations, 8);
//!
//! // Compare with the dense function once: materialize, subtract, maxabs.
//! let values: Vec<f64> = (0..8).map(|x| 1.0 / (1.0 + x as f64)).collect();
//! let reference = IdxTensor::from_dense(sites.clone(), values)?;
//! let dense = result.partition.to_treetn()?.contract_to_tensor()?;
//! assert!(dense.sub(&reference)?.maxabs()? < 1e-12);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod cache;
mod embed;
mod layout;
mod sampling;
#[cfg(test)]
mod tests;

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use tensor4all_core::{
    ColMajorArray, ColMajorArrayRef, CommonScalar, DynIndex, IdxTensor, TensorElement,
};
use tensor4all_treetn::interpolation::{
    InterpolationError, InterpolationProblem, InterpolationTermination, TreeInterpolator,
};
use tensor4all_treetn::{NodeNameNetwork, TreeTN};

use crate::error::PartitionedTreeTNError;
use crate::{PartitionedTreeTN, Projector, SubDomainTreeTN};

use cache::{Counters, PatchCache, PatchSampler};
use layout::SiteLayout;
use sampling::{patch_candidates, patch_seeds, PatchDomain};

/// Options of [`patched_interpolate`].
///
/// Built with [`PatchedInterpolationOptions::new`], which takes the required
/// bond cap and sets every other field to its default, and refined with the
/// `with_*` builders. There is no `Default`: the bond cap has no sensible
/// default. The fields are validated by [`patched_interpolate`].
///
/// `rtol` and `max_bond_dim` trade off: a tighter tolerance or a smaller cap
/// produces more, smaller patches. When in doubt, keep the defaults, pass a
/// known `reference_scale`, and choose the cap from the rank the engine can
/// afford per patch.
///
/// # Examples
///
/// ```
/// use tensor4all_core::DynIndex;
/// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
///
/// let first = DynIndex::new_dyn(2);
/// let options = PatchedInterpolationOptions::new(16)
///     .with_rtol(1e-6)
///     .with_reference_scale(2.0)
///     .with_patch_order(vec![first.clone()])
///     .with_recycle_pivots(true);
/// assert_eq!(options.max_bond_dim, 16);
/// assert_eq!(options.rtol, 1e-6);
/// assert_eq!(options.reference_scale, Some(2.0));
/// assert_eq!(options.patch_order, vec![first]);
/// assert_eq!(options.n_initial_pivots, 5);
/// assert!(options.recycle_pivots);
/// assert_eq!(options.max_patches, None);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PatchedInterpolationOptions {
    /// Relative tolerance. The engine's absolute tolerance is
    /// `rtol * reference_scale`. Finite and nonnegative; `0` splits until
    /// every patch is represented exactly. Default `1e-8`.
    pub rtol: f64,
    /// Scale used for every patch. `None` (the default) pins it to the
    /// largest magnitude among the root patch's candidate samples (or its
    /// exact values when the root has at most one site), a sampled lower bound
    /// on `max |f|`. For a localized function that makes the tolerance tighter
    /// than intended, so passing a known scale is recommended. Finite and
    /// positive when given.
    pub reference_scale: Option<f64>,
    /// Bond cap of every patch, at least 2. A patch is accepted only when the
    /// engine converges with a rank strictly below the cap. Required; a
    /// smaller cap means more, smaller patches.
    pub max_bond_dim: usize,
    /// Sites fixed when a patch splits, in order, by full index identity.
    /// A partial order is allowed: a patch that does not converge after every
    /// listed site is fixed fails with
    /// [`PatchedInterpolationError::NoSplitIndexLeft`]. Empty (the default)
    /// means every site in the derived site order
    /// ([`InterpolationProblem::derive_site_order`]). For quantics grids, list
    /// the most significant bits first.
    pub patch_order: Vec<DynIndex>,
    /// Target number of distinct initial pivots per patch, at least 1.
    /// Compatible user (and recycled) pivots come first and random points of
    /// the patch fill the rest. Default `5`.
    pub n_initial_pivots: usize,
    /// Seed each child with the pivots of the parent's outcome. Default
    /// `false`.
    pub recycle_pivots: bool,
    /// Root seed of every per-patch sub-seed. Default `0`.
    pub seed: u64,
    /// Limit on processed patches (accepted, zero, and split alike); `None`
    /// (the default) means no limit, `Some(0)` is invalid.
    pub max_patches: Option<usize>,
}

impl PatchedInterpolationOptions {
    /// Create options with the given bond cap and the defaults of every
    /// other field (`rtol = 1e-8`, no reference scale, the derived site
    /// order, five initial pivots, no recycling, seed `0`, no patch limit).
    ///
    /// # Arguments
    ///
    /// * `max_bond_dim` - Bond cap of every patch; [`patched_interpolate`]
    ///   requires at least 2.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// let options = PatchedInterpolationOptions::new(8);
    /// assert_eq!(options.max_bond_dim, 8);
    /// assert_eq!(options.rtol, 1e-8);
    /// assert_eq!(options.reference_scale, None);
    /// assert!(options.patch_order.is_empty());
    /// assert_eq!(options.n_initial_pivots, 5);
    /// assert!(!options.recycle_pivots);
    /// assert_eq!(options.seed, 0);
    /// assert_eq!(options.max_patches, None);
    /// ```
    pub fn new(max_bond_dim: usize) -> Self {
        Self {
            rtol: 1e-8,
            reference_scale: None,
            max_bond_dim,
            patch_order: Vec::new(),
            n_initial_pivots: 5,
            recycle_pivots: false,
            seed: 0,
            max_patches: None,
        }
    }

    /// Set the relative tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// assert_eq!(PatchedInterpolationOptions::new(4).with_rtol(1e-4).rtol, 1e-4);
    /// ```
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol;
        self
    }

    /// Set a known reference scale, typically `max |f|`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// let options = PatchedInterpolationOptions::new(4).with_reference_scale(3.5);
    /// assert_eq!(options.reference_scale, Some(3.5));
    /// ```
    pub fn with_reference_scale(mut self, reference_scale: f64) -> Self {
        self.reference_scale = Some(reference_scale);
        self
    }

    /// Set the order in which sites are fixed when a patch splits.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::DynIndex;
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// let (a, b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    /// let options =
    ///     PatchedInterpolationOptions::new(4).with_patch_order(vec![b.clone(), a.clone()]);
    /// assert_eq!(options.patch_order, vec![b, a]);
    /// ```
    pub fn with_patch_order(mut self, patch_order: Vec<DynIndex>) -> Self {
        self.patch_order = patch_order;
        self
    }

    /// Set the target number of initial pivots per patch.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// let options = PatchedInterpolationOptions::new(4).with_n_initial_pivots(12);
    /// assert_eq!(options.n_initial_pivots, 12);
    /// ```
    pub fn with_n_initial_pivots(mut self, n_initial_pivots: usize) -> Self {
        self.n_initial_pivots = n_initial_pivots;
        self
    }

    /// Enable or disable pivot recycling.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// assert!(PatchedInterpolationOptions::new(4).with_recycle_pivots(true).recycle_pivots);
    /// ```
    pub fn with_recycle_pivots(mut self, recycle_pivots: bool) -> Self {
        self.recycle_pivots = recycle_pivots;
        self
    }

    /// Set the root seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// assert_eq!(PatchedInterpolationOptions::new(4).with_seed(42).seed, 42);
    /// ```
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Limit the number of processed patches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// let options = PatchedInterpolationOptions::new(4).with_max_patches(100);
    /// assert_eq!(options.max_patches, Some(100));
    /// ```
    pub fn with_max_patches(mut self, max_patches: usize) -> Self {
        self.max_patches = Some(max_patches);
        self
    }
}

/// The record of one accepted patch.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate, PatchedInterpolationOptions,
/// };
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::interpolation::InterpolationTermination;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// // A single node with one site of dimension 4 is evaluated exactly.
/// let site = DynIndex::new_dyn(4);
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node(0usize)?;
/// let result = patched_interpolate(
///     &TreeTciInterpolator::default(),
///     topology,
///     BTreeMap::from([(0usize, vec![site])]),
///     ColMajorArray::new(vec![], vec![1, 0])?,
///     |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///         Ok(batch.data().iter().map(|&x| x as f64 - 1.0).collect())
///     },
///     &PatchedInterpolationOptions::new(2),
/// )?;
/// let record = &result.report.accepted[0];
/// assert!(record.projector.is_empty());
/// assert_eq!(record.termination, InterpolationTermination::Converged);
/// assert_eq!(record.error_estimate, 0.0);
/// assert_eq!(record.max_sample_magnitude, 2.0);
/// assert_eq!(record.max_bond_dim, 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PatchRecord {
    /// Projector of the patch (its fixed sites).
    pub projector: Projector,
    /// The engine's verdict; always
    /// [`InterpolationTermination::Converged`] for an accepted patch.
    pub termination: InterpolationTermination,
    /// The engine's raw error estimate, `0` for an exactly evaluated patch.
    pub error_estimate: f64,
    /// Largest sampled magnitude of the patch reported by the engine, or the
    /// largest exact value of an exactly evaluated patch.
    pub max_sample_magnitude: f64,
    /// Largest bond dimension of the patch network (one for an exactly
    /// evaluated patch).
    pub max_bond_dim: usize,
}

/// Summary of one [`patched_interpolate`] run.
///
/// `accepted` and `zero_projectors` are sorted by patch path, the
/// lexicographic order of the (position in the derived site order,
/// coordinate) pairs of the fixed sites in split order; this canonical order
/// does not depend on the processing order. Accepted and zero projectors are
/// pairwise disjoint and together cover the domain.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate, PatchedInterpolationOptions,
/// };
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// // f(a, b) is b^2 for a = 0 and 1 + b for a = 1: rank two. With a cap of
/// // two the domain splits once at `a`; both halves are evaluated exactly.
/// let (a, b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node(0usize)?;
/// topology.add_node(1usize)?;
/// topology.add_edge(&0, &1)?;
/// let node_sites = BTreeMap::from([(0usize, vec![a.clone()]), (1, vec![b.clone()])]);
/// let f = |p: &[usize]| if p[0] == 1 { 1.0 + p[1] as f64 } else { (p[1] * p[1]) as f64 };
/// let result = patched_interpolate(
///     &TreeTciInterpolator::default(),
///     topology,
///     node_sites,
///     ColMajorArray::new(vec![1, 2], vec![2, 1])?,
///     |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///         Ok(batch.data().chunks(2).map(f).collect())
///     },
///     &PatchedInterpolationOptions::new(2).with_reference_scale(4.0),
/// )?;
/// let report = &result.report;
/// assert_eq!(report.reference_scale, 4.0);
/// assert_eq!(report.splits, 1);
/// assert_eq!(report.accepted.len(), 2);
/// assert!(report.zero_projectors.is_empty());
/// assert_eq!(report.accepted[0].projector.get(&a), Some(0));
/// assert_eq!(report.accepted[1].projector.get(&a), Some(1));
/// // Six points in total, each evaluated once.
/// assert_eq!(report.function_evaluations, 6);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PatchedInterpolationReport {
    /// The reference scale used for every patch: the given one, or the one
    /// pinned from the root patch (`0` when the root is an all-zero exact
    /// patch and no scale was given).
    pub reference_scale: f64,
    /// Records of the accepted patches, in canonical path order.
    pub accepted: Vec<PatchRecord>,
    /// Projectors of the zero patches, in canonical path order. They are not
    /// part of the partition, which treats an absent patch as zero.
    pub zero_projectors: Vec<Projector>,
    /// Number of patches that were split.
    pub splits: usize,
    /// Number of points passed to the evaluator.
    pub function_evaluations: usize,
    /// Number of requested points served from a patch cache instead of the
    /// evaluator.
    pub cache_hits: usize,
}

/// Result of [`patched_interpolate`].
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate, PatchedInterpolationOptions,
/// };
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// // f vanishes everywhere: the root is a zero patch and the partition is empty.
/// let site = DynIndex::new_dyn(3);
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node(0usize)?;
/// let result = patched_interpolate(
///     &TreeTciInterpolator::default(),
///     topology,
///     BTreeMap::from([(0usize, vec![site])]),
///     ColMajorArray::new(vec![], vec![1, 0])?,
///     |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///         Ok(vec![0.0; batch.shape()[1]])
///     },
///     &PatchedInterpolationOptions::new(2),
/// )?;
/// assert!(result.partition.is_empty());
/// assert_eq!(result.report.zero_projectors.len(), 1);
/// assert_eq!(result.report.reference_scale, 0.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct PatchedInterpolationResult<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// The accepted patches. Zero patches are absent (an absent patch is
    /// zero), so an all-zero function gives an empty partition.
    pub partition: PartitionedTreeTN<V>,
    /// Summary of the run.
    pub report: PatchedInterpolationReport,
}

/// Error returned by [`patched_interpolate`].
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate, PatchedInterpolationError, PatchedInterpolationOptions,
/// };
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node(0usize)?;
/// let error = patched_interpolate(
///     &TreeTciInterpolator::default(),
///     topology,
///     BTreeMap::from([(0usize, vec![DynIndex::new_dyn(2)])]),
///     ColMajorArray::new(vec![], vec![1, 0])?,
///     |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///         Ok(vec![1.0; batch.shape()[1]])
///     },
///     // A cap of one could only accept patches nonzero on a single node.
///     &PatchedInterpolationOptions::new(1),
/// )
/// .unwrap_err();
/// assert!(matches!(error, PatchedInterpolationError::InvalidInput { .. }));
/// assert!(error.to_string().contains("max_bond_dim"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PatchedInterpolationError {
    /// The inputs or options are invalid, or the reference scale cannot be
    /// pinned. Reported before any evaluation, except the unpinnable scale.
    #[error("invalid patched interpolation input: {message}")]
    InvalidInput {
        /// The violated condition and, where possible, the remedy.
        message: String,
    },
    /// Sampling or interpolating one patch failed: the evaluator failed or
    /// returned a wrong number of values or a non-finite value
    /// ([`InterpolationError::Evaluator`]), or the engine failed or returned
    /// an outcome that does not match the problem.
    #[error("interpolation of the patch {projector:?} failed: {source}")]
    Interpolation {
        /// Projector of the failing patch.
        projector: Projector,
        /// The underlying interpolation error.
        #[source]
        source: InterpolationError,
    },
    /// Building a patch or the partition failed.
    #[error("building the patched partition failed: {source}")]
    Partition {
        /// The underlying partition error.
        #[source]
        source: PartitionedTreeTNError,
    },
    /// A patch did not converge and every site of `patch_order` is already
    /// fixed in it.
    #[error(
        "the patch {projector:?} did not converge and every site of patch_order is fixed; \
         list more sites in patch_order or raise max_bond_dim"
    )]
    NoSplitIndexLeft {
        /// Projector of the patch that could not be split.
        projector: Projector,
    },
    /// A resource limit of the options was exceeded.
    #[error(
        "patched interpolation exceeded its {resource} limit of {limit}; raise the limit or \
         max_bond_dim"
    )]
    ResourceLimit {
        /// Name of the exceeded option.
        resource: &'static str,
        /// The limit.
        limit: usize,
    },
}

impl From<PartitionedTreeTNError> for PatchedInterpolationError {
    fn from(source: PartitionedTreeTNError) -> Self {
        Self::Partition { source }
    }
}

/// Adaptively interpolate a function on a tree into disjoint patches.
///
/// # Arguments
///
/// * `engine` - Tree interpolation engine run on every patch with at least
///   two active sites, for example `tensor4all_treetci::TreeTciInterpolator`.
/// * `topology` - Tree topology with named nodes. Its node set must equal the
///   keys of `node_sites`.
/// * `node_sites` - Site indices of every node, possibly none for a node. The
///   derived site order ([`InterpolationProblem::derive_site_order`]: nodes
///   in ascending name order, each node's sites in the given order) lays out
///   `initial_pivots` and every evaluator batch.
/// * `initial_pivots` - Column-major `[n_sites, n_pivots]` array of
///   full-domain points in site order; zero columns are allowed. Each patch
///   starts from those compatible with it. Pivots where the function is large
///   make the sampled scale and the zero screening reliable.
/// * `evaluate` - Batch evaluator. It receives a column-major
///   `[n_sites, n_points]` array of full-domain points in site order and
///   returns one finite value per point.
/// * `options` - See [`PatchedInterpolationOptions`].
///
/// # Returns
///
/// The partition of accepted patches (eagerly masked, every site index
/// retained, one dtype `T`) and a [`PatchedInterpolationReport`].
///
/// # Errors
///
/// - [`PatchedInterpolationError::InvalidInput`] before any evaluation when
///   [`validate_layout`](tensor4all_treetn::interpolation::validate_layout) rejects the topology or sites (with its message), a
///   `patch_order` entry is not a site of the problem (full identity and
///   dimension) or is repeated, `rtol` is negative or not finite,
///   `reference_scale` is not finite and positive, `max_bond_dim < 2`,
///   `n_initial_pivots == 0`, `max_patches == Some(0)`, or `initial_pivots` is
///   not a 2D array with one row per site and in-range coordinates; and when
///   no `reference_scale` is given and every candidate sample of a root that
///   needs the engine is exactly zero.
/// - [`PatchedInterpolationError::Interpolation`] when the evaluator fails,
///   returns a wrong number of values, or returns a non-finite value
///   ([`InterpolationError::Evaluator`]); when the engine fails (including
///   [`InterpolationError::AllSamplesZero`] after screening); or when an
///   engine outcome does not match the problem ([`InterpolationError::Engine`]).
/// - [`PatchedInterpolationError::NoSplitIndexLeft`] when a patch does not
///   converge and every site of `patch_order` is fixed.
/// - [`PatchedInterpolationError::ResourceLimit`] when more than
///   `max_patches` patches would be processed.
/// - [`PatchedInterpolationError::Partition`] when building a patch network
///   or the partition fails.
///
/// # Examples
///
/// A branched tree whose junction `"c"` has degree three and carries no
/// site. `f` vanishes where the leaf site `x` is zero; that half is reported
/// as a zero patch instead of stored.
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate, PatchedInterpolationOptions,
/// };
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// let (x, y, z) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3), DynIndex::new_dyn(3));
/// let mut topology = NodeNameNetwork::new();
/// for node in ["c", "x", "y", "z"] {
///     topology.add_node(node.to_string())?;
/// }
/// for leaf in ["x", "y", "z"] {
///     topology.add_edge(&"c".to_string(), &leaf.to_string())?;
/// }
/// let node_sites = BTreeMap::from([
///     ("c".to_string(), vec![]),
///     ("x".to_string(), vec![x.clone()]),
///     ("y".to_string(), vec![y.clone()]),
///     ("z".to_string(), vec![z.clone()]),
/// ]);
/// // Site order [x, y, z]; f = x (1 + y + z)^2 has rank three across the y edge.
/// let f = |p: &[usize]| (p[0] * (1 + p[1] + p[2]).pow(2)) as f64;
/// let options = PatchedInterpolationOptions::new(3)
///     .with_reference_scale(25.0)
///     .with_patch_order(vec![x.clone(), y.clone()]);
/// let result = patched_interpolate(
///     &TreeTciInterpolator::default(),
///     topology,
///     node_sites,
///     ColMajorArray::new(vec![1, 2, 2], vec![3, 1])?,
///     |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///         Ok(batch.data().chunks(3).map(f).collect())
///     },
///     &options,
/// )?;
/// assert_eq!(result.report.zero_projectors.len(), 1);
/// assert_eq!(result.report.zero_projectors[0].get(&x), Some(0));
///
/// let values: Vec<f64> = (0..18).map(|k| f(&[k % 2, (k / 2) % 3, k / 6])).collect();
/// let reference = IdxTensor::from_dense(vec![x, y, z], values)?;
/// let dense = result.partition.to_treetn()?.contract_to_tensor()?;
/// assert!(dense.sub(&reference)?.maxabs()? < 10.0 * options.rtol * 25.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn patched_interpolate<T, V, E, F>(
    engine: &E,
    topology: NodeNameNetwork<V>,
    node_sites: BTreeMap<V, Vec<DynIndex>>,
    initial_pivots: ColMajorArray<usize>,
    evaluate: F,
    options: &PatchedInterpolationOptions,
) -> Result<PatchedInterpolationResult<V>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T> + Sync,
    F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>> + Send + Sync,
{
    let layout = SiteLayout::validated(topology, node_sites, &initial_pivots, options)?;
    Driver {
        engine,
        evaluate: &evaluate,
        layout,
        initial_pivots,
        options,
        reference_scale: Cell::new(options.reference_scale),
        counters: Counters::default(),
        scalar: PhantomData,
    }
    .run()
}

fn invalid(message: impl Into<String>) -> PatchedInterpolationError {
    PatchedInterpolationError::InvalidInput {
        message: message.into(),
    }
}

/// One patch waiting in the queue.
struct Patch<T> {
    /// (position, coordinate) of every fixed site, in split order.
    path: Vec<(usize, usize)>,
    /// Fixed coordinate of every site of the site order, if any.
    fixed: Vec<Option<usize>>,
    cache: PatchCache<T>,
    /// Full-domain pivots recycled from the parent's outcome.
    recycled: Vec<Vec<usize>>,
}

/// What processing one patch produced.
enum Verdict<T, V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    Accepted(Box<(PatchRecord, SubDomainTreeTN<V>)>),
    Zero,
    Split(Vec<Patch<T>>),
}

fn max_magnitude<T: CommonScalar>(values: &[T]) -> f64 {
    values
        .iter()
        .map(|value| value.abs_val())
        .fold(0.0, f64::max)
}

fn evaluator_error(projector: &Projector, source: anyhow::Error) -> PatchedInterpolationError {
    PatchedInterpolationError::Interpolation {
        projector: projector.clone(),
        source: InterpolationError::Evaluator { source },
    }
}

fn engine_error(projector: &Projector, message: String) -> PatchedInterpolationError {
    PatchedInterpolationError::Interpolation {
        projector: projector.clone(),
        source: InterpolationError::Engine {
            source: anyhow::anyhow!(message),
        },
    }
}

/// A violated internal invariant, reported as a construction failure.
fn internal(source: impl Into<anyhow::Error>) -> PatchedInterpolationError {
    PatchedInterpolationError::Partition {
        source: PartitionedTreeTNError::TensorConstruction {
            source: source.into(),
        },
    }
}

/// Complete an outcome's active-site pivots with the fixed coordinates.
fn complete_pivots(
    pivots: Option<&ColMajorArray<usize>>,
    fixed: &[Option<usize>],
    active: &[usize],
    dims: &[usize],
) -> Result<Vec<Vec<usize>>, String> {
    let Some(pivots) = pivots else {
        return Ok(Vec::new());
    };
    let (Some(n_rows), Some(n_cols)) = (pivots.nrows(), pivots.ncols()) else {
        return Err(format!(
            "the outcome pivots have shape {:?}, expected a 2D array",
            pivots.shape()
        ));
    };
    if n_rows != active.len() {
        return Err(format!(
            "the outcome pivots have {n_rows} rows, expected {} active sites",
            active.len()
        ));
    }
    let mut points = Vec::with_capacity(n_cols);
    for column in 0..n_cols {
        let mut point: Vec<usize> = fixed.iter().map(|fixed| fixed.unwrap_or(0)).collect();
        let local = pivots.column(column).unwrap_or_default();
        for (&position, &value) in active.iter().zip(local) {
            if value >= dims[position] {
                return Err(format!(
                    "the outcome pivot {column} has coordinate {value} for a site of \
                     dimension {}",
                    dims[position]
                ));
            }
            point[position] = value;
        }
        points.push(point);
    }
    Ok(points)
}

/// State of one [`patched_interpolate`] run.
struct Driver<'a, T, V, E, F>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    engine: &'a E,
    evaluate: &'a F,
    layout: SiteLayout<V>,
    initial_pivots: ColMajorArray<usize>,
    options: &'a PatchedInterpolationOptions,
    /// The given scale, or the one pinned from the root patch.
    reference_scale: Cell<Option<f64>>,
    counters: Counters,
    scalar: PhantomData<T>,
}

impl<T, V, E, F> Driver<'_, T, V, E, F>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T>,
    F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>>,
{
    fn run(self) -> Result<PatchedInterpolationResult<V>, PatchedInterpolationError> {
        let mut queue = VecDeque::from([Patch {
            path: Vec::new(),
            fixed: vec![None; self.layout.sites.len()],
            cache: PatchCache::new(self.layout.dims.clone()),
            recycled: Vec::new(),
        }]);
        let mut processed = 0usize;
        let mut splits = 0usize;
        let mut accepted = Vec::new();
        let mut zeros = Vec::new();
        while let Some(patch) = queue.pop_front() {
            if let Some(limit) = self.options.max_patches {
                if processed == limit {
                    return Err(PatchedInterpolationError::ResourceLimit {
                        resource: "max_patches",
                        limit,
                    });
                }
            }
            processed += 1;
            let path = patch.path.clone();
            let projector = self.layout.projector(&path)?;
            match self.process(patch, &projector)? {
                Verdict::Accepted(entry) => {
                    let (record, subdomain) = *entry;
                    accepted.push((path, record, subdomain));
                }
                Verdict::Zero => zeros.push((path, projector)),
                Verdict::Split(children) => {
                    splits += 1;
                    queue.extend(children);
                }
            }
        }

        // Canonical order: lexicographic over the (position, coordinate) path.
        accepted.sort_by(|left, right| left.0.cmp(&right.0));
        zeros.sort_by(|left, right| left.0.cmp(&right.0));
        let (records, subdomains): (Vec<_>, Vec<_>) = accepted
            .into_iter()
            .map(|(_, record, subdomain)| (record, subdomain))
            .unzip();
        let partition = PartitionedTreeTN::from_disjoint_subdomains(subdomains)?;
        Ok(PatchedInterpolationResult {
            partition,
            report: PatchedInterpolationReport {
                reference_scale: self.reference_scale.get().unwrap_or(0.0),
                accepted: records,
                zero_projectors: zeros.into_iter().map(|(_, projector)| projector).collect(),
                splits,
                function_evaluations: self.counters.evaluations.get(),
                cache_hits: self.counters.cache_hits.get(),
            },
        })
    }

    fn sampler<'s>(
        &'s self,
        fixed: &'s [Option<usize>],
        n_active: usize,
        cache: PatchCache<T>,
    ) -> PatchSampler<'s, T, F> {
        PatchSampler {
            evaluate: self.evaluate,
            fixed,
            n_active,
            counters: &self.counters,
            cache: RefCell::new(cache),
        }
    }

    fn process(
        &self,
        patch: Patch<T>,
        projector: &Projector,
    ) -> Result<Verdict<T, V>, PatchedInterpolationError> {
        let Patch {
            path,
            fixed,
            cache,
            recycled,
        } = patch;
        let layout = &self.layout;
        let active: Vec<usize> = (0..layout.sites.len())
            .filter(|&position| fixed[position].is_none())
            .collect();
        if active.len() <= 1 {
            return self.exact_patch(&fixed, &active, cache, projector);
        }

        let seeds = patch_seeds(self.options.seed, &path);
        let candidates = patch_candidates(
            &PatchDomain {
                dims: &layout.dims,
                fixed: &fixed,
                active: &active,
                layout: cache.layout(),
            },
            &self.initial_pivots,
            &recycled,
            self.options.n_initial_pivots,
            seeds.candidates,
        );
        let sampler = self.sampler(&fixed, active.len(), cache);
        let shape = [active.len(), candidates.count];
        let batch = ColMajorArrayRef::new(&candidates.points, &shape).map_err(internal)?;
        let samples = sampler
            .sample(batch)
            .map_err(|source| evaluator_error(projector, source))?;
        let largest = max_magnitude(&samples);
        let scale = match self.reference_scale.get() {
            Some(scale) => scale,
            None if largest == 0.0 => {
                return Err(invalid(format!(
                    "the reference scale cannot be pinned: all {} candidate samples of the root \
                     patch are exactly zero; pass reference_scale or initial pivots in the \
                     support of the function",
                    candidates.count
                )));
            }
            None => {
                self.reference_scale.set(Some(largest));
                largest
            }
        };
        if largest == 0.0 {
            return Ok(Verdict::Zero);
        }

        let interpolation_error = |source| PatchedInterpolationError::Interpolation {
            projector: projector.clone(),
            source,
        };
        let problem = InterpolationProblem::new(
            layout.topology.clone(),
            layout.active_node_sites(&fixed),
            ColMajorArray::new(candidates.points, vec![active.len(), candidates.count])
                .map_err(internal)?,
            self.options.rtol * scale,
            NonZeroUsize::new(self.options.max_bond_dim),
            seeds.engine,
        )
        .map_err(interpolation_error)?;
        let outcome = self
            .engine
            .interpolate(&problem, |batch| sampler.sample(batch))
            .map_err(interpolation_error)?;

        if outcome.termination == InterpolationTermination::Converged {
            embed::check_outcome_layout(&outcome.network, layout, &fixed)
                .map_err(|message| engine_error(projector, message))?;
            let subdomain = self.subdomain(&outcome.network, &fixed, projector)?;
            let record = PatchRecord {
                projector: projector.clone(),
                termination: outcome.termination,
                error_estimate: outcome.error_estimate,
                max_sample_magnitude: outcome.max_sample_magnitude,
                max_bond_dim: subdomain.max_bond_dim(),
            };
            return Ok(Verdict::Accepted(Box::new((record, subdomain))));
        }

        let Some(&split_position) = layout
            .split_order
            .iter()
            .find(|&&position| fixed[position].is_none())
        else {
            return Err(PatchedInterpolationError::NoSplitIndexLeft {
                projector: projector.clone(),
            });
        };
        let recycled = if self.options.recycle_pivots {
            complete_pivots(outcome.pivots.as_ref(), &fixed, &active, &layout.dims)
                .map_err(|message| engine_error(projector, message))?
        } else {
            Vec::new()
        };
        let slot = active
            .iter()
            .position(|&position| position == split_position)
            .ok_or_else(|| internal(anyhow::anyhow!("the split site is not active")))?;
        let children = sampler
            .cache
            .into_inner()
            .split(slot)
            .into_iter()
            .enumerate()
            .map(|(value, cache)| {
                let mut child_path = path.clone();
                child_path.push((split_position, value));
                let mut child_fixed = fixed.clone();
                child_fixed[split_position] = Some(value);
                Patch {
                    path: child_path,
                    fixed: child_fixed,
                    cache,
                    recycled: recycled
                        .iter()
                        .filter(|point| point[split_position] == value)
                        .cloned()
                        .collect(),
                }
            })
            .collect();
        Ok(Verdict::Split(children))
    }

    /// Evaluate a patch with at most one active site on all its points and
    /// build its network without the engine.
    fn exact_patch(
        &self,
        fixed: &[Option<usize>],
        active: &[usize],
        cache: PatchCache<T>,
        projector: &Projector,
    ) -> Result<Verdict<T, V>, PatchedInterpolationError> {
        // Every point of the patch: `0..d` for one active site of dimension
        // `d`, or the single empty point when no site is active.
        let n_points: usize = active.iter().map(|&p| self.layout.dims[p]).product();
        let points: Vec<usize> = active.iter().flat_map(|_| 0..n_points).collect();
        let shape = [active.len(), n_points];
        let sampler = self.sampler(fixed, active.len(), cache);
        let batch = ColMajorArrayRef::new(&points, &shape).map_err(internal)?;
        let values = sampler
            .sample(batch)
            .map_err(|source| evaluator_error(projector, source))?;
        let largest = max_magnitude(&values);
        if self.reference_scale.get().is_none() {
            self.reference_scale.set(Some(largest));
        }
        if largest == 0.0 {
            return Ok(Verdict::Zero);
        }
        let network = embed::exact_active_network(&self.layout, active, values)?;
        let subdomain = self.subdomain(&network, fixed, projector)?;
        let record = PatchRecord {
            projector: projector.clone(),
            termination: InterpolationTermination::Converged,
            error_estimate: 0.0,
            max_sample_magnitude: largest,
            max_bond_dim: subdomain.max_bond_dim(),
        };
        Ok(Verdict::Accepted(Box::new((record, subdomain))))
    }

    /// Re-embed the fixed sites into an active-site network and wrap the
    /// already masked result as a patch.
    fn subdomain(
        &self,
        network: &TreeTN<IdxTensor, V>,
        fixed: &[Option<usize>],
        projector: &Projector,
    ) -> Result<SubDomainTreeTN<V>, PatchedInterpolationError> {
        let data = embed::embed_fixed_sites::<T, V>(network, &self.layout, fixed)?;
        Ok(SubDomainTreeTN::from_masked_data(
            data,
            projector.clone(),
            None,
        )?)
    }
}
