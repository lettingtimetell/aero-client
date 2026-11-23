//! Block models, blockstates, and baking — S09.
//!
//! `blockstate.rs` and `blockmodel.rs` parse vanilla JSON; `facebakery.rs` turns one element
//! face into a [`BakedQuad`]; `state_mapper.rs` resolves a [`mc_data::BlockState`] to the
//! blockstate file/key it bakes from (hazard **H12**); `item.rs` bakes item models including
//! the `builtin/generated` flat-icon expansion; `bake.rs` orchestrates all of the above plus
//! [`crate::atlas`] into the frozen [`model_for`]/[`sprite`] entry points.

mod bake;
pub mod blockmodel;
pub mod blockstate;
pub mod facebakery;
mod item;
pub mod state_mapper;
pub mod transforms;

pub use bake::{
    bake_all, bake_all_with_ctm, bake_all_with_mipmap_levels, ctm_table, install,
    installed_generation, item_frame_model, model_for, model_for_actual, model_for_actual_at,
    model_for_at, reinstall, sprite, BakeReport, BakedAssets, ModelBakeFailure, BUILTIN_TEXTURES,
};
// A reload with the same inputs reuses the
// generation it already published instead of leaking another copy.
pub use bake::{distinct_generations, install_keyed, reinstall_keyed};
// A private bake, read per thread.
pub use bake::{bake_all_unpublished, thread_registry_overridden, with_thread_registry};
pub use blockmodel::ModelResolveError;
pub use item::{
    item_model_for, item_model_for_damage, item_model_for_variant, item_model_override_for,
    item_model_override_for_variant, ItemModelOverride, ItemModelProperty,
};
// 1.7 Animations' "1.7 item edges".
pub use item::{bake_generated_layer_17, OLD_ITEM_EMPTY_ALPHA};
pub use transforms::{ItemCameraTransforms, ItemTransform, TransformType};

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

use mc_math::Facing;

use crate::sprite::UvRect;

/// A process-global slot that can be *replaced* after it is first filled, while readers keep
/// handing out `&'static` references.
///
/// # Why this exists
///
/// S09's frozen API is `pub fn model_for(state: BlockState) -> &'static BakedModel`, and S21's
/// resource-pack hot reload needs a second [`bake_all`] to take effect without a restart. Those
/// two requirements together rule out `OnceLock` (first call wins) *and* rule out any guard type
/// (`RwLockReadGuard`, `arc_swap::Guard`), because a guard is not `&'static` and changing the
/// return type would break every terrain-mesher call site. What is left is: publish the value at
/// `'static` lifetime and never take it back.
///
/// So a swap **leaks the value it replaces**. That is not an oversight, it is the price of the
/// frozen signature: a mesher job on a `rayon` worker may be holding a `&'static BakedModel` from
/// the previous generation at the moment the swap happens, and there is no sound way to free that
/// memory without `unsafe` (banned here — this crate is `#![forbid(unsafe_code)]`). The cost is
/// one baked-asset set per resource-pack reload (~10 MB against the real 1.8.9 assets), paid only
/// when the user actually swaps packs. With [`HotSwap::reinstall_keyed`]
/// (what the client's reload uses) it is one set per *distinct* pack stack instead: a reload
/// whose inputs match a set published before publishes that set again.
///
/// # Why not `arc-swap` or `crossbeam-epoch`
///
/// Checked directly, not assumed: `arc-swap` 1.7.1 and `crossbeam-epoch` 0.9.18/0.9.20 are both
/// present in this workspace's offline cargo registry cache, so "no dependency available" is not
/// actually why this stays leaky — neither crate closes the gap, and the reason is structural
/// rather than a missing `Cargo.toml` line.
///
/// `arc_swap::Guard<T>` (`arc-swap-1.7.1/src/lib.rs:168`) and `ArcSwapAny::load_full`
/// (`:403`) are safe and do reclaim promptly, but by construction they hand back an owned `T`
/// (typically `Arc<BakedModel>`) or a lifetime-free `Guard` wrapping one — never a bare
/// `&'static BakedModel`. Turning either into the `&'static` this signature demands needs the
/// same trick used here today (leak it, right back where this started) or an `unsafe` raw-pointer
/// deref of the `*const T` from `Arc::into_raw`.
///
/// `crossbeam_epoch::Shared<'g, T>` (`crossbeam-epoch-0.9.20/src/atomic.rs:1296`) is worse for
/// this purpose, not better: it is explicitly bound to the pinning `Guard`'s `'g`, not `'static`,
/// and *every* method that touches the pointee — `deref`, `deref_mut`, `as_ref`, and the
/// reclaiming `Guard::defer_destroy`/`Shared::into_owned` — is `pub unsafe fn`
/// (`atomic.rs:1418,1460,1497`, `guard.rs:268`). Epoch reclamation is not merely blocked at the
/// *free* step the way `Box::from_raw` blocks a manual leak; it needs `unsafe` to so much as read
/// the value back out, so it buys nothing under `forbid(unsafe_code)`.
///
/// A synchronization scheme instead of a crate — e.g. `reinstall` proving quiescence by draining
/// the mesher's job queue, or a `rayon::ThreadPool` broadcast barrier, before treating the old
/// generation as unreachable — does not change the answer. The blocker is not "how do we know
/// it's safe to free," it's "safe Rust has no operation that frees it even given that knowledge."
/// The value was published via `Box::leak`, and the only way back from that is `Box::from_raw`
/// (`unsafe fn`) to reconstitute the `Box` so `Drop` can run; there is no safe "un-leak" in `std`.
/// The same gap shows up in both crates above: `arc-swap` avoids it by never leaking in the first
/// place (at the cost of not returning `&'static`), and `crossbeam-epoch`'s reclaiming calls are
/// unsafe for the same reason. Proving no worker holds a stale reference is plausibly the easy
/// part — rayon's queue is not the obstacle — but that proof buys nothing without a safe
/// operation to act on it, and none exists for memory that was handed out as `&'static`.
///
/// # Concurrency
///
/// Reads are lock-free in the steady state *if the caller caches per thread on
/// [`HotSwap::generation`]* — see [`model_for`] for the pattern this crate uses. A read that does
/// not cache takes an uncontended `RwLock` read. Either way the reference handed out stays valid
/// forever, so a swap concurrent with a read is sound: the reader simply finishes with the
/// generation it started on, which is exactly the guarantee `OnceLock` gave and no weaker.
///
/// A swap is *not* atomic across threads: two threads reading at the same instant may straddle a
/// generation boundary. For a resource-pack reload that is harmless — the next frame is
/// consistent — and it is what vanilla does too (`refreshResources` reloads listeners one at a
/// time while the render thread is between frames).
#[derive(Debug)]
pub struct HotSwap<T: Send + Sync + 'static> {
    current: RwLock<Option<&'static T>>,
    /// `0` means "never installed". Monotonic, so a cached generation is never mistaken for a
    /// later one.
    generation: AtomicU64,
    /// Every value this slot has published, with the key it was
    /// published under (`None`: no key, never matched). They are all leaked, so each stays
    /// valid forever; keeping the list is what lets [`HotSwap::reinstall_keyed`] and
    /// [`HotSwap::reinstall_dedup`] publish one again instead of leaking an equal copy. The
    /// leak is then one value per *distinct* input (a resource-pack stack, say), not one per
    /// reload: a server resource pack reloads twice per visit (on join and on leave).
    published: Mutex<Vec<(Option<String>, &'static T)>>,
}

impl<T: Send + Sync + 'static> Default for HotSwap<T> {
    fn default() -> Self {
        HotSwap::new()
    }
}

impl<T: Send + Sync + 'static> HotSwap<T> {
    /// An empty slot. `const`, so this can back a `static`.
    #[must_use]
    pub const fn new() -> HotSwap<T> {
        HotSwap {
            current: RwLock::new(None),
            generation: AtomicU64::new(0),
            published: Mutex::new(Vec::new()),
        }
    }

    /// The list of published values. A poisoned lock still holds a valid list (every entry
    /// is a leaked `&'static`), so reading through the poison is safe.
    fn published(&self) -> std::sync::MutexGuard<'_, Vec<(Option<String>, &'static T)>> {
        self.published
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// How many distinct values this slot has published (and therefore keeps leaked).
    #[must_use]
    pub fn distinct_published(&self) -> usize {
        self.published().len()
    }

    /// How many times this slot has been filled. `0` means never; every successful
    /// [`HotSwap::install`] or [`HotSwap::reinstall`] increments it.
    ///
    /// Cheap enough (one relaxed atomic load of a shared, read-mostly cache line) to call on
    /// every read as a validity check for a thread-local cache.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// The current value, or `None` before anything is installed.
    ///
    /// Takes a read lock. Hot paths should call [`HotSwap::generation`] first and only come here
    /// when it changed.
    #[must_use]
    pub fn load(&self) -> Option<&'static T> {
        match self.current.read() {
            Ok(guard) => *guard,
            // A poisoned lock means a writer panicked mid-swap. The value it was replacing is
            // still valid (it is leaked), so reading through the poison is strictly better than
            // panicking a mesher worker.
            Err(poisoned) => *poisoned.into_inner(),
        }
    }

    /// Fill the slot **only if it is empty**, exactly reproducing `OnceLock::set`.
    ///
    /// Returns `true` if `value` was installed, `false` if the slot was already full (in which
    /// case `value` is dropped, again as `OnceLock::set` does).
    pub fn install(&self, value: T) -> bool {
        self.install_keyed(None, value)
    }

    /// [`HotSwap::install`], remembering `key` as what produced `value`, so a later
    /// [`HotSwap::reinstall_keyed`] with the same key can publish it again.
    pub fn install_keyed(&self, key: Option<String>, value: T) -> bool {
        let mut guard = match self.current.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if guard.is_some() {
            return false;
        }
        let leaked: &'static T = Box::leak(Box::new(value));
        self.published().push((key, leaked));
        *guard = Some(leaked);
        self.generation.fetch_add(1, Ordering::Release);
        true
    }

    /// Replace the slot's contents, whether or not it was already full.
    ///
    /// Returns the new generation. The previous value is **leaked** — see the type's docs for
    /// why that is required rather than sloppy.
    pub fn reinstall(&self, value: T) -> u64 {
        self.reinstall_keyed(None, value)
    }

    /// [`HotSwap::reinstall`], except that when this slot has already published a value under
    /// the same `key`, that value is published again and `value` is dropped (it was never
    /// handed out, so dropping it is safe): the slot then leaks one value per distinct key, not
    /// one per call. `None` never matches. The generation advances either way, so per-thread
    /// caches re-read the slot exactly as after any other swap.
    ///
    /// The caller vouches that equal keys mean equal values (for the block registry, the bake
    /// cache's own identity of a clean bake).
    pub fn reinstall_keyed(&self, key: Option<String>, value: T) -> u64 {
        self.publish(key, value, |published, key, _| {
            let key = key.as_ref()?;
            published
                .iter()
                .find(|(earlier, _)| earlier.as_ref() == Some(key))
                .map(|&(_, earlier)| earlier)
        })
    }

    /// [`HotSwap::reinstall`], except that when `same(&value, published)` holds for a value this
    /// slot published before, that one is published again and `value` is dropped. For values
    /// small enough to compare whole on every reload (the colormaps).
    pub fn reinstall_dedup(&self, value: T, same: impl Fn(&T, &T) -> bool) -> u64 {
        self.publish(None, value, |published, _, value| {
            published
                .iter()
                .find(|(_, earlier)| same(value, earlier))
                .map(|&(_, earlier)| earlier)
        })
    }

    /// Under the write lock: point the slot at what `find` picks among the published values
    /// (dropping `value`), or at `value`, leaked and recorded under `key`.
    fn publish(
        &self,
        key: Option<String>,
        value: T,
        find: impl FnOnce(&[(Option<String>, &'static T)], &Option<String>, &T) -> Option<&'static T>,
    ) -> u64 {
        let mut guard = match self.current.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut published = self.published();
        let unused = match find(&published, &key, &value) {
            Some(earlier) => {
                *guard = Some(earlier);
                Some(value)
            }
            None => {
                let leaked: &'static T = Box::leak(Box::new(value));
                published.push((key, leaked));
                *guard = Some(leaked);
                None
            }
        };
        drop(published);
        let generation = self.generation.fetch_add(1, Ordering::Release) + 1;
        drop(guard);
        // Never handed out: free it after the lock, so readers are not held up by the drop.
        drop(unused);
        generation
    }
}

/// One baked vertex: block-local position (`0..1` cube space, camera/world offset applied
/// downstream) and atlas UV. Frozen S09 API names `Vertex` without specifying its fields;
/// this is the minimal shape the mesher needs — no baked colour (see the `shade` note on
/// [`BakedQuad`] for why) and no normal (derivable from [`BakedQuad::face`] for axis-aligned
/// quads, or from vertex winding for the rare non-aligned ones).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vertex {
    /// Position in block-local `0..1` space (a full cube spans `[0,0,0]..[1,1,1]`).
    pub pos: [f32; 3],
    /// Atlas UV.
    pub uv: [f32; 2],
}

/// One baked quad. Frozen S09 API.
#[derive(Debug, Clone, PartialEq)]
pub struct BakedQuad {
    /// The four corners, in winding order.
    pub verts: [Vertex; 4],
    /// The quad's own geometric facing (java: `FaceBakery.getFacingFromVertexData`, rotated
    /// by the blockstate variant). `None` only for the rare element face whose plane, after
    /// an oblique element `rotation`, does not align with any of the six axes.
    pub face: Option<Facing>,
    /// The direction whose baked 0.5/0.6/0.8/1.0 shade byte FaceBakery selected.
    ///
    /// This is normally the same as [`Self::face`], but an element-local rotation is the
    /// important exception: Java stores `getFacingFromVertexData()` on `BakedQuad`, while it
    /// chooses the shade byte from `ModelRotation.rotateFace` before that local rotation.
    /// Keeping both values lets terrain use the geometric face for neighbour/AO sampling
    /// without accidentally rotating the already-baked directional shade.
    pub shade_face: Option<Facing>,
    /// Biome tint layer (grass/foliage/water colormap index), or `None` for an untinted quad.
    pub tint_index: Option<u8>,
    /// Whether this quad wants vanilla's directional per-face brightness applied. This is the
    /// element's raw JSON `shade` flag, not a precomputed colour — combining it with the
    /// quad's facing into an actual brightness multiplier, and blending that against ambient
    /// occlusion, happens once per world position in the terrain mesher (S11), not once per
    /// bake here.
    pub shade: bool,
}

/// One block state's baked geometry. Frozen S09 API.
#[derive(Debug, Clone)]
pub struct BakedModel {
    /// Quads culled when the neighbour in that direction is opaque — indexed by
    /// [`Facing::index`]. This is keyed by each quad's JSON `cullface` (rotated with the
    /// model), which is *usually* but not always the quad's own [`BakedQuad::face`].
    pub face_quads: [Vec<BakedQuad>; 6],
    /// Quads with no `cullface`: always drawn regardless of neighbours.
    pub general_quads: Vec<BakedQuad>,
    /// Whether the terrain mesher should compute smooth lighting/AO for this model.
    pub ambient_occlusion: bool,
    /// Java's `IBakedModel.isGui3d()` bit retained from the concrete baked-model kind.
    ///
    /// Ordinary JSON/block models and `BuiltInModel` report `true`; models produced by
    /// `ItemModelGenerator` (`builtin/generated`, compass, and clock) report `false`.
    pub gui_3d: bool,
    /// Atlas-space UV rectangle of the model JSON's resolved `"particle"` texture.
    ///
    /// This is retained independently of geometry because the particle sprite need not appear on
    /// any baked face. Built-in entity models intentionally leave it absent.
    pub particle_uv: Option<UvRect>,
    /// The model JSON's `display` block, resolved through the parent chain — the per-context
    /// rotation/translation/scale vanilla's `RenderItem` pushes before drawing an item
    /// (java: `ItemCameraTransforms`).
    ///
    /// Additive field, `ItemCameraTransforms::DEFAULT` (identity everywhere) for a model with
    /// no `display` anywhere in its chain, which is the correct value for every ordinary block
    /// model. Closes the S12 -> S09 request: `item::emit_dropped_item` previously received only
    /// `display.ground`'s y/z **scale**, supplied by its caller, so a tool whose `display.ground`
    /// also rotates rendered at the right size and the wrong orientation.
    pub transforms: ItemCameraTransforms,
}

impl Default for BakedModel {
    fn default() -> Self {
        BakedModel {
            face_quads: [
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            general_quads: Vec::new(),
            ambient_occlusion: true,
            gui_3d: false,
            particle_uv: None,
            transforms: ItemCameraTransforms::DEFAULT,
        }
    }
}

impl BakedModel {
    /// True when this model has no geometry at all — the correct, intentional result for
    /// air and every `render_type != Model` block (fluids, tile-entity-rendered blocks),
    /// which vanilla never bakes a JSON model for in the first place. Distinct from "failed
    /// to bake", which is reported through [`BakeReport`] instead of silently producing this.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.general_quads.is_empty() && self.face_quads.iter().all(std::vec::Vec::is_empty)
    }
}

#[cfg(test)]
mod hot_swap_tests {
    use super::HotSwap;

    /// The whole point of the type: `install` must behave exactly like the `OnceLock::set` it
    /// replaces, so nothing that has ever called `mc_assets::install` changes behaviour.
    #[test]
    fn install_is_first_wins_exactly_like_the_once_lock_it_replaces() {
        let slot: HotSwap<u32> = HotSwap::new();
        assert_eq!(slot.generation(), 0);
        assert_eq!(slot.load(), None, "nothing before the first install");

        assert!(slot.install(7));
        assert_eq!(slot.load(), Some(&7));
        assert_eq!(slot.generation(), 1);

        assert!(!slot.install(9), "a second install must be refused");
        assert_eq!(slot.load(), Some(&7), "and must not change the value");
        assert_eq!(slot.generation(), 1, "nor advance the generation");
    }

    /// The new capability, and the reason the generation exists.
    #[test]
    fn reinstall_replaces_and_advances_the_generation() {
        let slot: HotSwap<u32> = HotSwap::new();
        assert!(slot.install(1));
        assert_eq!(slot.reinstall(2), 2);
        assert_eq!(slot.load(), Some(&2));
        assert_eq!(slot.reinstall(3), 3);
        assert_eq!(slot.load(), Some(&3));
        // Monotonic, so a thread-local cache keyed on it can never confuse two generations.
        assert!(slot.generation() > 1);
    }

    /// A reload with the inputs of an earlier generation publishes that
    /// generation again (the very same `&'static`) and leaks nothing new; other keys still get
    /// their own value, and the generation advances on every call.
    #[test]
    fn a_keyed_reinstall_reuses_the_value_published_for_the_same_key() {
        let slot: HotSwap<String> = HotSwap::new();
        assert!(slot.install_keyed(Some("vanilla".to_owned()), "vanilla #1".to_owned()));
        let first: &'static String = slot.load().expect("installed");
        assert_eq!(
            slot.reinstall_keyed(Some("server pack".to_owned()), "server #1".to_owned()),
            2
        );
        assert_eq!(slot.load().map(String::as_str), Some("server #1"));
        // Back to the first stack, then the server pack again, many times over.
        for round in 0..10u64 {
            let generation =
                slot.reinstall_keyed(Some("vanilla".to_owned()), "vanilla again".to_owned());
            assert_eq!(generation, 3 + 2 * round);
            assert!(std::ptr::eq(slot.load().expect("installed"), first));
            slot.reinstall_keyed(Some("server pack".to_owned()), "server again".to_owned());
            assert_eq!(slot.load().map(String::as_str), Some("server #1"));
        }
        assert_eq!(
            slot.distinct_published(),
            2,
            "one value per distinct key, not per reload"
        );
        assert_eq!(
            first, "vanilla #1",
            "the reused value is the original, intact"
        );
    }

    /// The negative control: unkeyed reinstalls (and `None` keys) keep the old behaviour of
    /// leaking one value per call, and distinct keys never share a value.
    #[test]
    fn unkeyed_and_distinct_reinstalls_still_publish_a_new_value_each() {
        let slot: HotSwap<u32> = HotSwap::new();
        assert!(slot.install(1));
        slot.reinstall(1);
        slot.reinstall_keyed(None, 1);
        slot.reinstall_keyed(Some("a".to_owned()), 2);
        slot.reinstall_keyed(Some("b".to_owned()), 2);
        assert_eq!(slot.distinct_published(), 5);
        assert_eq!(slot.load(), Some(&2));
        assert_eq!(slot.generation(), 5);
    }

    /// Content deduplication, for values cheap to compare whole (the colormaps).
    #[test]
    fn a_dedup_reinstall_reuses_an_equal_value() {
        let slot: HotSwap<Vec<u32>> = HotSwap::new();
        assert!(slot.install(vec![1, 2, 3]));
        let first: &'static Vec<u32> = slot.load().expect("installed");
        slot.reinstall_dedup(vec![4, 5, 6], |a, b| a == b);
        slot.reinstall_dedup(vec![1, 2, 3], |a, b| a == b);
        assert!(std::ptr::eq(slot.load().expect("installed"), first));
        slot.reinstall_dedup(vec![4, 5, 6], |a, b| a == b);
        assert_eq!(slot.load(), Some(&vec![4, 5, 6]));
        assert_eq!(slot.distinct_published(), 2);
        assert_eq!(slot.generation(), 4);
    }

    /// A slot that was never installed can still be reinstalled — the reload path does not
    /// depend on a cold `install` having happened first.
    #[test]
    fn reinstall_works_on_an_empty_slot() {
        let slot: HotSwap<u32> = HotSwap::new();
        assert_eq!(slot.reinstall(5), 1);
        assert_eq!(slot.load(), Some(&5));
    }

    /// The soundness claim, exercised rather than asserted in prose: a reference taken before a
    /// swap must still be readable afterwards, because that is precisely what a `rayon` mesher
    /// worker holding a `&'static BakedModel` does.
    #[test]
    fn a_reference_taken_before_a_swap_stays_valid_after_it() {
        let slot: HotSwap<String> = HotSwap::new();
        slot.install("first".to_owned());
        let old: &'static String = slot.load().expect("installed");
        slot.reinstall("second".to_owned());
        assert_eq!(old, "first", "the old generation is leaked, not freed");
        assert_eq!(slot.load().map(String::as_str), Some("second"));
    }

    /// Concurrent readers while a writer swaps: nothing may panic, and every value read must be
    /// one of the ones actually installed — never a torn or dangling one.
    #[test]
    fn concurrent_reads_during_a_swap_are_sound() {
        static SLOT: HotSwap<u64> = HotSwap::new();
        SLOT.install(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..2_000 {
                        let value = *SLOT.load().expect("always installed");
                        assert!(value < 64, "read a value that was never installed: {value}");
                    }
                });
            }
            scope.spawn(|| {
                for generation in 1..64u64 {
                    SLOT.reinstall(generation);
                }
            });
        });
        assert_eq!(SLOT.load(), Some(&63));
    }
}
