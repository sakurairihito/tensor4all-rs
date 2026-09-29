//! Per-patch evaluation cache with variable-length packed keys.
//!
//! A key packs the active coordinates of one point into `u64` words. Each
//! coordinate of a site with dimension `d` takes `bits(d - 1)` bits (none for
//! `d = 1`) and never straddles two words, so any domain size is supported
//! without a width limit.

use std::cell::{Cell, RefCell};
use std::collections::hash_map::Entry;
use std::collections::HashMap;

use tensor4all_core::{ColMajorArrayRef, CommonScalar, TensorElement};

/// Packing of the active coordinates of a patch into key words.
#[derive(Clone, Debug)]
pub(super) struct KeyLayout {
    dims: Vec<usize>,
    /// `(word, shift, width)` of every active slot.
    slots: Vec<(usize, u32, u32)>,
    n_words: usize,
}

impl KeyLayout {
    /// Lay out the active coordinates with the given dimensions (all positive).
    pub(super) fn new(dims: Vec<usize>) -> Self {
        let mut slots = Vec::with_capacity(dims.len());
        let mut word = 0usize;
        let mut used = 0u32;
        for &dim in &dims {
            let width = usize::BITS - dim.saturating_sub(1).leading_zeros();
            if used + width > u64::BITS {
                word += 1;
                used = 0;
            }
            slots.push((word, used, width));
            used += width;
        }
        let n_words = if used == 0 { word } else { word + 1 };
        Self {
            dims,
            slots,
            n_words,
        }
    }

    /// Dimensions of the active slots.
    pub(super) fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Number of key words.
    #[cfg(test)]
    pub(super) fn n_words(&self) -> usize {
        self.n_words
    }

    /// Pack one point given in active-slot order.
    pub(super) fn encode(&self, coords: impl Iterator<Item = usize>) -> Box<[u64]> {
        let mut key = vec![0u64; self.n_words].into_boxed_slice();
        for (&(word, shift, width), value) in self.slots.iter().zip(coords) {
            if width > 0 {
                key[word] |= (value as u64) << shift;
            }
        }
        key
    }

    /// Unpack one key into active-slot coordinates.
    pub(super) fn decode(&self, key: &[u64]) -> Vec<usize> {
        self.slots
            .iter()
            .map(|&(word, shift, width)| {
                if width == 0 {
                    0
                } else {
                    let mask = u64::MAX >> (u64::BITS - width);
                    ((key[word] >> shift) & mask) as usize
                }
            })
            .collect()
    }
}

/// Evaluation cache of one patch, keyed by its active coordinates.
///
/// `tensor4all_core::CachedFunction` does not fit here: it wraps an
/// infallible point function (the driver's evaluator is fallible and batched),
/// its keys stop at 1024 bits, and its entries can only be cleared at once,
/// whereas this cache must be split among the children of a split patch.
/// Each lookup still allocates an owned key (see the "Owned Vector Cache
/// Keys" rule in `PERFORMANCE_TIPS.md`); removing that is left to the
/// parallel milestone (M7).
#[derive(Clone, Debug)]
pub(super) struct PatchCache<T> {
    layout: KeyLayout,
    entries: HashMap<Box<[u64]>, T>,
}

impl<T: Copy> PatchCache<T> {
    pub(super) fn new(active_dims: Vec<usize>) -> Self {
        Self {
            layout: KeyLayout::new(active_dims),
            entries: HashMap::new(),
        }
    }

    pub(super) fn layout(&self) -> &KeyLayout {
        &self.layout
    }

    pub(super) fn get(&self, key: &[u64]) -> Option<T> {
        self.entries.get(key).copied()
    }

    pub(super) fn insert(&mut self, key: Box<[u64]>, value: T) {
        self.entries.insert(key, value);
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Split the cache among the children of a split at active slot `slot`,
    /// in one pass: child `c` receives every entry whose coordinate at `slot`
    /// is `c`, keyed without that coordinate.
    pub(super) fn split(self, slot: usize) -> Vec<Self> {
        let mut child_dims = self.layout.dims.clone();
        let n_children = child_dims.remove(slot);
        let child_layout = KeyLayout::new(child_dims);
        let mut children: Vec<Self> = (0..n_children)
            .map(|_| Self {
                layout: child_layout.clone(),
                entries: HashMap::new(),
            })
            .collect();
        for (key, value) in self.entries {
            let mut coords = self.layout.decode(&key);
            let child = coords.remove(slot);
            let child_key = child_layout.encode(coords.into_iter());
            children[child].entries.insert(child_key, value);
        }
        children
    }
}

/// Evaluation counters of a run.
#[derive(Default)]
pub(super) struct Counters {
    pub(super) evaluations: Cell<usize>,
    pub(super) cache_hits: Cell<usize>,
}

impl Counters {
    fn add(counter: &Cell<usize>, amount: usize) {
        counter.set(counter.get().saturating_add(amount));
    }
}

/// The value of one requested point: cached, or the index of a new point.
enum Slot<T> {
    Known(T),
    Missing(usize),
}

/// Samples one patch through its cache.
pub(super) struct PatchSampler<'a, T, F> {
    pub(super) evaluate: &'a F,
    pub(super) fixed: &'a [Option<usize>],
    pub(super) n_active: usize,
    pub(super) counters: &'a Counters,
    pub(super) cache: RefCell<PatchCache<T>>,
}

impl<T, F> PatchSampler<'_, T, F>
where
    T: CommonScalar + TensorElement,
    F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>>,
{
    /// Values at a column-major `[n_active, n_points]` batch of active
    /// coordinates. Only points absent from the cache reach the evaluator,
    /// each once, completed with the fixed coordinates. The values are checked
    /// for count and finiteness before they are cached; nothing is cached when
    /// the evaluator fails.
    pub(super) fn sample(&self, batch: ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>> {
        let n_active = self.n_active;
        let shape = batch.shape();
        anyhow::ensure!(
            shape.len() == 2 && shape[0] == n_active,
            "batch shape {shape:?} does not match the {n_active} active sites of the patch"
        );
        let n_points = shape[1];
        let data = batch.data();

        let mut slots = Vec::with_capacity(n_points);
        let mut missing_keys: Vec<Box<[u64]>> = Vec::new();
        let mut missing_points: Vec<usize> = Vec::new();
        let mut hits = 0usize;
        {
            let cache = self.cache.borrow();
            let layout = cache.layout();
            let mut pending: HashMap<Box<[u64]>, usize> = HashMap::new();
            for point in 0..n_points {
                let local = &data[point * n_active..(point + 1) * n_active];
                if let Some((slot, (&value, &dim))) = local
                    .iter()
                    .zip(layout.dims())
                    .enumerate()
                    .find(|(_, (value, dim))| value >= dim)
                {
                    anyhow::bail!(
                        "point {point} has coordinate {value} at active site {slot}, out of \
                         range for dimension {dim}"
                    );
                }
                let key = layout.encode(local.iter().copied());
                if let Some(value) = cache.get(&key) {
                    hits += 1;
                    slots.push(Slot::Known(value));
                    continue;
                }
                match pending.entry(key) {
                    Entry::Occupied(entry) => {
                        hits += 1;
                        slots.push(Slot::Missing(*entry.get()));
                    }
                    Entry::Vacant(entry) => {
                        let index = missing_keys.len();
                        missing_keys.push(entry.key().clone());
                        entry.insert(index);
                        slots.push(Slot::Missing(index));
                        let mut active = local.iter().copied();
                        missing_points.extend(
                            self.fixed
                                .iter()
                                .map(|fixed| fixed.or_else(|| active.next()).unwrap_or_default()),
                        );
                    }
                }
            }
        }
        Counters::add(&self.counters.cache_hits, hits);

        let mut fresh = Vec::new();
        if !missing_keys.is_empty() {
            let n_sites = self.fixed.len();
            let n_missing = missing_keys.len();
            let full_shape = [n_sites, n_missing];
            fresh = (self.evaluate)(ColMajorArrayRef::new(&missing_points, &full_shape)?)?;
            anyhow::ensure!(
                fresh.len() == n_missing,
                "the evaluator returned {} values for {n_missing} points",
                fresh.len()
            );
            if let Some(index) = fresh.iter().position(|&value| !is_finite(value)) {
                anyhow::bail!(
                    "the evaluator returned a non-finite value at the point {:?}",
                    &missing_points[index * n_sites..(index + 1) * n_sites]
                );
            }
            Counters::add(&self.counters.evaluations, n_missing);
            let mut cache = self.cache.borrow_mut();
            for (key, &value) in missing_keys.into_iter().zip(&fresh) {
                cache.insert(key, value);
            }
        }

        Ok(slots
            .into_iter()
            .map(|slot| match slot {
                Slot::Known(value) => value,
                Slot::Missing(index) => fresh[index],
            })
            .collect())
    }
}

/// Whether every component (real and imaginary part) of `value` is finite.
///
/// Multiplying by zero gives exactly zero for finite components and NaN when
/// a component is infinite or NaN. Unlike `abs_val().is_finite()`, this
/// accepts a finite complex value whose magnitude overflows.
pub(super) fn is_finite<T: CommonScalar>(value: T) -> bool {
    (value * T::from_f64(0.0)).abs_val() == 0.0
}
