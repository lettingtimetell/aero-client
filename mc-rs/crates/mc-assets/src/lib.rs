#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Resource location, resource packs, and the baked model/atlas pipeline.
//!
//! Its modules:
//!
//! * [`ResourceLocation`], the pack implementations, the stacked
//!   [`ResourceManager`], the [`AssetIndex`], and [`Language`]. Its job stops at "raw bytes
//!   and translated strings load by [`ResourceLocation`]".
//! * `model`, `atlas`, `stitcher`, `sprite` modules: blockstate/model JSON parsing
//!   and quad baking ([`model`], hazard **H12**), texture bin-packing ([`stitcher`]),
//!   animated-texture decode ([`sprite`]), and the assembled RGBA8 [`Atlas`]. The frozen
//!   entry points are [`model_for`] (a block state's baked geometry) and [`sprite`] (a
//!   texture's atlas placement + animation frames); [`bake_all`] is the real constructor and
//!   [`install`] publishes its result for those two to read.
//!
//! # Where assets come from
//!
//! Vanilla stacks three kinds of pack, lowest priority first:
//!
//! 1. [`ObjectStorePack`] — the launcher's content-addressed store, `assets/indexes/1.8.json`
//!    plus `assets/objects/<ab>/<sha1>`. Sounds, non-US languages, icons.
//! 2. [`ZipResourcePack`] — `versions/1.8.9/1.8.9.jar`. Models, blockstates, textures,
//!    `en_US.lang`. Also how user `.zip` resource packs are read.
//! 3. [`FolderResourcePack`] — an unpacked directory, for development and folder-style packs.
//!
//! # Example
//!
//! ```no_run
//! use mc_assets::{Language, ObjectStorePack, ResourceLocation, ResourceManager, ZipResourcePack};
//!
//! let mut assets = ResourceManager::new();
//! assets.push_pack(Box::new(ObjectStorePack::open("jars/assets", "1.8")?));
//! assets.push_pack(Box::new(ZipResourcePack::open("jars/versions/1.8.9/1.8.9.jar")?));
//!
//! let png = assets.load(&ResourceLocation::parse("textures/blocks/stone.png")?)?;
//! assert_eq!(&png[1..4], b"PNG");
//!
//! let lang = Language::load(&assets, &["en_US"]);
//! assert_eq!(lang.tr("gui.done"), "Done");
//! # Ok::<_, mc_assets::AssetError>(())
//! ```
//!
//! # Untrusted input
//!
//! A server can hand the client a resource-pack URL, so packs are hostile input. Locations
//! are validated against path traversal ([`ResourceLocation::is_safe`]), zip entry names are
//! rejected at open time, entry counts and decompressed sizes are capped ([`ZipLimits`]), and
//! nothing here panics on malformed pack data.

pub mod atlas;
pub mod cache;
// Connected textures.
pub mod ctm;
// Resource-pack custom sky layers.
pub mod custom_sky;
mod error;
mod index;
mod lang;
mod location;
mod manager;
pub mod model;
mod pack;
pub mod sprite;
pub mod stitcher;
// Pack PNGs larger than the graphics device allows, refused at decode.
pub mod texture_size;

pub use atlas::{
    coalesce_animation_uploads, Atlas, AtlasAnimationContext, AtlasAnimationUpload, AtlasAnimator,
    AtlasClockInput, AtlasCompassInput,
};
pub use error::AssetError;
pub use index::{AssetIndex, AssetObject};
pub use lang::{
    format_template, parse_language_sections, Language, LanguageInfo, DEFAULT_LANGUAGE,
};
pub use location::{is_windows_device_name, ResourceLocation, DEFAULT_NAMESPACE};
pub use manager::{Presence, ReloadListener, Resource, ResourceManager};
pub use model::{
    bake_all, bake_all_with_mipmap_levels, install, item_frame_model, item_model_for,
    item_model_for_damage, item_model_for_variant, item_model_override_for,
    item_model_override_for_variant, model_for, model_for_actual, model_for_actual_at,
    model_for_at, sprite, BakeReport, BakedAssets, BakedModel, BakedQuad, ItemModelOverride,
    ItemModelProperty, ModelBakeFailure, Vertex,
};
pub use pack::{
    FolderResourcePack, ObjectStorePack, PackMetadata, ResourcePack, ZipLimits, ZipResourcePack,
    MAX_MCMETA_BYTES, PACK_FORMAT_1_8_9,
};
pub use sprite::{parse_texture_metadata, Sprite, TextureMetadata};
pub use texture_size::{decode_texture_png, limit_texture_side, texture_side_limit};
