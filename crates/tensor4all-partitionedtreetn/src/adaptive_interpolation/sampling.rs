//! Reproducible randomness and candidate pivots of one patch.
//!
//! The stream is SplitMix64 (Steele, Lea, and Flood, 2014; the reference
//! constants of Vigna's implementation), implemented here so that the library
//! gains no random-number dependency and the stream never changes with a
//! dependency upgrade. Bounded draws use Lemire's multiply-shift method with
//! rejection, which is unbiased. Changing any constant below changes every
//! reproducible candidate list and engine seed.

use std::collections::HashSet;

use tensor4all_core::ColMajorArray;

use super::cache::KeyLayout;

/// SplitMix64 increment (the golden-ratio gamma).
const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
/// Domain separator mixed into the root seed before absorbing a patch path.
const PATH_DOMAIN: u64 = 0x7472_6565_7061_7463;
/// Stream selector of the candidate sub-seed.
const CANDIDATE_STREAM: u64 = 0x6361_6e64_6964_6174;
/// Stream selector of the engine sub-seed.
const ENGINE_STREAM: u64 = 0x656e_6769_6e65_7365;
/// Random attempts per missing candidate before the column-major fallback.
const ATTEMPTS_PER_CANDIDATE: usize = 20;
/// Random attempts added to every patch before the column-major fallback.
const BASE_ATTEMPTS: usize = 100;

/// The SplitMix64 generator.
#[derive(Clone, Debug)]
pub(super) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Draw a value in `0..bound` with Lemire's method; `bound` is positive.
    pub(super) fn below(&mut self, bound: u64) -> u64 {
        let mut product = u128::from(self.next_u64()) * u128::from(bound);
        let mut low = product as u64;
        if low < bound {
            let threshold = bound.wrapping_neg() % bound;
            while low < threshold {
                product = u128::from(self.next_u64()) * u128::from(bound);
                low = product as u64;
            }
        }
        (product >> 64) as u64
    }
}

/// One SplitMix64 output for the state `value`, used as a mixing function.
pub(super) fn mix64(value: u64) -> u64 {
    SplitMix64::new(value).next_u64()
}

/// The candidate and engine sub-seeds of a patch.
///
/// The root seed is mixed with the patch path, a sequence of (position in the
/// derived site order, coordinate) pairs in split order. The encoding uses no
/// index identities, so it does not depend on how the split sites were chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PatchSeeds {
    pub(super) candidates: u64,
    pub(super) engine: u64,
}

pub(super) fn patch_seeds(root_seed: u64, path: &[(usize, usize)]) -> PatchSeeds {
    let state = path.iter().fold(
        mix64(root_seed ^ PATH_DOMAIN),
        |state, &(position, value)| {
            let state = mix64(state ^ position as u64);
            mix64(state ^ value as u64)
        },
    );
    PatchSeeds {
        candidates: mix64(state ^ CANDIDATE_STREAM),
        engine: mix64(state ^ ENGINE_STREAM),
    }
}

/// Candidate pivots of a patch in active coordinates.
pub(super) struct Candidates {
    /// Column-major `[n_active, n_candidates]` coordinates.
    pub(super) points: Vec<usize>,
    pub(super) count: usize,
}

/// Inputs of [`patch_candidates`] that describe the patch.
pub(super) struct PatchDomain<'a> {
    /// Dimension of every site of the derived site order.
    pub(super) dims: &'a [usize],
    /// Fixed coordinate of every site of the derived site order, if any.
    pub(super) fixed: &'a [Option<usize>],
    /// Positions of the active sites, ascending.
    pub(super) active: &'a [usize],
    /// Key layout of the active coordinates.
    pub(super) layout: &'a KeyLayout,
}

impl PatchDomain<'_> {
    fn is_compatible(&self, point: &[usize]) -> bool {
        self.fixed
            .iter()
            .zip(point)
            .all(|(fixed, &value)| fixed.is_none_or(|fixed| fixed == value))
    }

    /// Number of points of the patch, saturating at `usize::MAX`.
    pub(super) fn point_count(&self) -> usize {
        self.active.iter().fold(1usize, |count, &position| {
            count.saturating_mul(self.dims[position])
        })
    }
}

/// Build the candidate pivots of a patch.
///
/// The user pivots and then the recycled pivots (full-domain points) that are
/// compatible with the patch are kept in order without duplicates. Random
/// points inside the patch are added until `target` distinct candidates exist
/// or the patch has no further point: each coordinate is drawn in `0..d` from
/// the SplitMix64 stream seeded with `seed`, in active-site order, and after a
/// bounded number of attempts the first unused points in column-major order
/// (first active site fastest) fill the rest.
pub(super) fn patch_candidates(
    domain: &PatchDomain<'_>,
    user_pivots: &ColMajorArray<usize>,
    recycled: &[Vec<usize>],
    target: usize,
    seed: u64,
) -> Candidates {
    build_candidates(domain, user_pivots, recycled, target, seed, random_attempts)
}

/// The number of random draws tried for `missing` candidates before the
/// column-major fallback.
pub(super) fn random_attempts(missing: usize) -> usize {
    missing
        .saturating_mul(ATTEMPTS_PER_CANDIDATE)
        .saturating_add(BASE_ATTEMPTS)
}

/// [`patch_candidates`] with an explicit attempt budget.
pub(super) fn build_candidates(
    domain: &PatchDomain<'_>,
    user_pivots: &ColMajorArray<usize>,
    recycled: &[Vec<usize>],
    target: usize,
    seed: u64,
    attempt_budget: fn(usize) -> usize,
) -> Candidates {
    let mut candidates = Candidates {
        points: Vec::new(),
        count: 0,
    };
    let mut seen = HashSet::new();
    let mut local = vec![0usize; domain.active.len()];
    let mut push = |local: &[usize], candidates: &mut Candidates| {
        if seen.insert(domain.layout.encode(local.iter().copied())) {
            candidates.points.extend_from_slice(local);
            candidates.count += 1;
        }
    };

    let n_user = user_pivots.ncols().unwrap_or(0);
    let user = (0..n_user).filter_map(|column| user_pivots.column(column));
    for point in user.chain(recycled.iter().map(Vec::as_slice)) {
        if domain.is_compatible(point) {
            for (slot, &position) in domain.active.iter().enumerate() {
                local[slot] = point[position];
            }
            push(&local, &mut candidates);
        }
    }

    let desired = target.min(domain.point_count());
    if candidates.count >= desired {
        return candidates;
    }

    let attempts = attempt_budget(desired - candidates.count);
    let mut rng = SplitMix64::new(seed);
    for _ in 0..attempts {
        if candidates.count >= desired {
            return candidates;
        }
        for (slot, &position) in domain.active.iter().enumerate() {
            local[slot] = rng.below(domain.dims[position] as u64) as usize;
        }
        push(&local, &mut candidates);
    }

    let mut flat = 0usize;
    while candidates.count < desired {
        let mut rest = flat;
        for (slot, &position) in domain.active.iter().enumerate() {
            let dim = domain.dims[position];
            local[slot] = rest % dim;
            rest /= dim;
        }
        push(&local, &mut candidates);
        flat += 1;
    }
    candidates
}
