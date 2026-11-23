//! The block/item texture atlas (java: `TextureMap.java`'s stitching orchestration).
//!
//! [`Atlas`] is the frozen S09 deliverable: a plain RGBA8 `Vec<u8>` plus dimensions. GPU
//! upload is downstream (S11/S12); this module's job stops at "one correctly-packed RGBA8
//! buffer, and every sprite knows where in it it landed."

use std::collections::HashMap;

use crate::sprite::{
    anaglyph_rgba, apply_anaglyph_rgba, load_sprite_with_atlas_frame, LoadedSprite, Sprite, UvRect,
};
use crate::stitcher::{stitch_with_mipmap_named, NamedSize, StitchError};
use crate::{AssetError, ResourceLocation, ResourceManager};

/// A baked RGBA8 texture atlas. Frozen S09 API.
#[derive(Debug, Clone)]
pub struct Atlas {
    /// Tightly packed RGBA8 pixels, `w * h * 4` bytes.
    pub texture: Vec<u8>,
    /// Additional atlas levels, in ascending order (level 1 first).  Each is stitched from
    /// sprite-local mip data rather than downsampled from the completed atlas, matching
    /// `TextureMap` and avoiding cross-sprite colour bleed.
    pub mipmaps: Vec<Vec<u8>>,
    /// Atlas width in pixels. Always a power of two.
    pub w: u32,
    /// Atlas height in pixels. Always a power of two.
    pub h: u32,
}

impl Atlas {
    /// Return the bytes that Java would hand to a texture upload for this atlas.
    ///
    /// Atlas construction keeps the decoded source colours intact so model baking, cache keys,
    /// and non-anaglyph renderers remain byte-identical.  The conversion is therefore exposed as
    /// an explicit upload-boundary copy and covers every resident mip level, not just level zero.
    /// Callers that retain an atlas for animation should pair this with
    /// [`AtlasAnimator::new_with_anaglyph`].
    #[must_use]
    pub fn upload_pixels(&self, anaglyph: bool) -> (Vec<u8>, Vec<Vec<u8>>) {
        (
            anaglyph_rgba(&self.texture, anaglyph),
            self.mipmaps
                .iter()
                .map(|mip| anaglyph_rgba(mip, anaglyph))
                .collect(),
        )
    }

    /// Clone this atlas with Java's optional upload-time anaglyph transform applied to every
    /// level.  The source atlas is left untouched, making this the safe form for a live setting
    /// toggle when a caller keeps one raw bake around.
    #[must_use]
    pub fn with_anaglyph(&self, enabled: bool) -> Self {
        let mut upload = self.clone();
        upload.apply_anaglyph(enabled);
        upload
    }

    /// Apply Java's optional anaglyph transform to an atlas copy in place.
    ///
    /// This is useful for a short-lived atlas whose pixels are about to be uploaded once.  It is
    /// intentionally not used by [`build_atlas`], because the canonical baked atlas must remain
    /// raw and reusable when the setting changes.
    pub fn apply_anaglyph(&mut self, enabled: bool) {
        apply_anaglyph_rgba(&mut self.texture, enabled);
        for mip in &mut self.mipmaps {
            apply_anaglyph_rgba(mip, enabled);
        }
    }
}

/// One changed rectangle of an animated atlas texture.
///
/// `TextureMap.updateAnimations` rewrites just a sprite's rectangle (and its already-generated
/// mip levels) through `TextureUtil.uploadTextureMipmap`; it does not replace the complete
/// atlas.  Render owners apply these records with one `Queue::write_texture` per rectangle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtlasAnimationUpload {
    /// Atlas mip level, where zero is the base atlas image.
    pub mip_level: u32,
    /// Left edge in texels at [`Self::mip_level`].
    pub x: u32,
    /// Top edge in texels at [`Self::mip_level`].
    pub y: u32,
    /// Rectangle width in texels at [`Self::mip_level`].
    pub width: u32,
    /// Rectangle height in texels at [`Self::mip_level`].
    pub height: u32,
    /// Tightly packed RGBA8 rectangle pixels.
    pub rgba: Vec<u8>,
}

impl AtlasAnimationUpload {
    /// Clone one animation upload with Java's optional anaglyph transform applied.
    #[must_use]
    pub fn upload_pixels(&self, anaglyph: bool) -> Self {
        Self {
            mip_level: self.mip_level,
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
            rgba: anaglyph_rgba(&self.rgba, anaglyph),
        }
    }

    /// Whether `self` and `other` cover exactly the same texels of the same level.
    fn same_rect(&self, other: &Self) -> bool {
        (self.mip_level, self.x, self.y, self.width, self.height)
            == (other.mip_level, other.x, other.y, other.width, other.height)
    }
}

/// Append one tick's dirty rectangles to the records still waiting for the next frame's flush.
///
/// java: `Minecraft.runGameLoop` runs every catch-up `runTick` (each ending in
/// `TextureMap.updateAnimations`) before its single render, so a texel written by an earlier
/// catch-up tick and rewritten by a later one is never sampled. When `queue` already holds an
/// earlier record for the identical rectangle, that record is dropped and the new one appended.
/// This is exact for any record sequence: the final value of a texel is the last record covering
/// it, and a dropped record is always followed by one covering every texel it covered, while the
/// relative order of all remaining records is unchanged. Every owner therefore ends a flush with
/// the same texture and raw-shadow bytes, and a ten-tick catch-up frame uploads one record per
/// animated sprite level instead of up to ten (`MAX_CATCH_UP_TICKS`).
///
/// The ordinary one-tick frame (an empty `queue`) is a plain append.
pub fn coalesce_animation_uploads(
    queue: &mut Vec<AtlasAnimationUpload>,
    uploads: impl IntoIterator<Item = AtlasAnimationUpload>,
) {
    if queue.is_empty() {
        queue.extend(uploads);
        return;
    }
    for upload in uploads {
        if let Some(index) = queue.iter().position(|queued| queued.same_rect(&upload)) {
            queue.remove(index);
        }
        queue.push(upload);
    }
}

/// Dynamic inputs for vanilla's two special atlas sprites, `items/clock` and `items/compass`.
///
/// The caller supplies the random samples instead of this type reaching for process randomness:
/// Java calls `Math.random()` in dimensions without a surface sky, and the client is the owner
/// of both that nondeterminism and the world/player snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AtlasAnimationContext {
    /// Clock-specific world input. `None` is vanilla's no-world/no-player path.
    pub clock: AtlasClockInput,
    /// Compass-specific world/player input. `None` is vanilla's no-world/no-player path.
    pub compass: Option<AtlasCompassInput>,
}

/// The context `TextureClock.updateAnimation` reads once per tick.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AtlasClockInput {
    /// `World.getWorldTime()`, or `None` when either vanilla `theWorld` or `thePlayer` is absent.
    pub world_time: Option<i64>,
    /// `WorldProvider.isSurfaceWorld()`.
    pub is_surface_world: bool,
    /// One `Math.random()` result in `0.0..1.0`, used only outside a surface world.
    pub random: f64,
}

/// The context `TextureCompass.updateCompass` reads once per tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtlasCompassInput {
    /// Player X position.
    pub player_x: f64,
    /// Player Z position.
    pub player_z: f64,
    /// `EntityPlayer.rotationYaw`, in degrees.
    pub yaw_degrees: f64,
    /// `World.getSpawnPoint().getX()`.
    pub spawn_x: i32,
    /// `World.getSpawnPoint().getZ()`.
    pub spawn_z: i32,
    /// `WorldProvider.isSurfaceWorld()`.
    pub is_surface_world: bool,
    /// One `Math.random()` result in `0.0..1.0`, used only outside a surface world.
    pub random: f64,
}

/// Playback state for every `.png.mcmeta` animation in a stitched atlas, including vanilla's
/// clock/compass specializations when a dynamic context is supplied.
///
/// This is deliberately CPU/GPU-neutral: call [`AtlasAnimator::tick`] once per client tick,
/// then send each returned [`AtlasAnimationUpload`] to every GPU copy of the block atlas
/// (terrain, particles, GUI, held-item/entity rendering).  Keeping that fan-out at the render
/// owner prevents this asset crate from depending on `wgpu`.
///
/// `items/clock` and `items/compass` have an additional vanilla dynamic-frame selector driven
/// by world time / bearing.  They are ordinary atlas animations at decode time, but their live
/// selector must be supplied by the client before they can be considered fully faithful.
#[derive(Debug, Clone)]
pub struct AtlasAnimator {
    sprites: Vec<AnimatedSprite>,
    anaglyph: bool,
}

#[derive(Debug, Clone)]
struct AnimatedSprite {
    location: ResourceLocation,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    /// `[playback frame][mip level]`, including base level zero.
    frames: Vec<Vec<Vec<u8>>>,
    durations: Vec<u32>,
    interpolate: bool,
    kind: AnimationKind,
    frame: usize,
    tick_in_frame: u32,
    anaglyph: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AnimationKind {
    Metadata,
    Clock {
        current_angle: f64,
        angle_delta: f64,
    },
    Compass {
        current_angle: f64,
        angle_delta: f64,
    },
}

fn animation_kind(sprite: &Sprite) -> AnimationKind {
    // java: TextureMap.registerSprite replaces precisely these two built-in item locations with
    // TextureClock/TextureCompass subclasses. A resource-pack texture at the same location is
    // still driven by that subclass, so identify the registered name, not its pixels/metadata.
    if sprite.location.namespace == "minecraft" && sprite.location.path == "textures/items/clock" {
        AnimationKind::Clock {
            current_angle: 0.0,
            angle_delta: 0.0,
        }
    } else if sprite.location.namespace == "minecraft"
        && sprite.location.path == "textures/items/compass"
    {
        AnimationKind::Compass {
            current_angle: 0.0,
            angle_delta: 0.0,
        }
    } else {
        AnimationKind::Metadata
    }
}

/// java: `WorldProvider.calculateCelestialAngle(worldTime, 1.0F)`, including its f32
/// intermediates. `TextureClock` passes that value straight into its angle smoother.
fn celestial_angle(world_time: i64) -> f64 {
    let mut angle = ((world_time % 24_000) as f32 + 1.0) / 24_000.0 - 0.25;
    if angle < 0.0 {
        angle += 1.0;
    }
    if angle > 1.0 {
        angle -= 1.0;
    }
    let linear = angle;
    // Java widens the f32 phase for `Math.cos`, then narrows the result before the final f32
    // blend. Do not use `f32::cos`: close values can straddle a clock-frame boundary.
    let curved = (1.0 - ((f64::from(angle) * std::f64::consts::PI).cos() + 1.0) / 2.0) as f32;
    f64::from(linear + (curved - linear) / 3.0)
}

fn clock_target(input: AtlasClockInput) -> f64 {
    match input.world_time {
        None => 0.0,
        Some(_) if !input.is_surface_world => input.random,
        Some(world_time) => celestial_angle(world_time),
    }
}

fn compass_target(input: Option<AtlasCompassInput>) -> f64 {
    let Some(input) = input else {
        return 0.0;
    };
    if !input.is_surface_world {
        // java: `Math.random() * Math.PI * 2.0D`, compiled with the float-widened constant
        // `3.1415927410125732` (bytecode of `bmp`); the other angles here use the exact `PI`.
        return input.random * f64::from(std::f32::consts::PI) * 2.0;
    }
    let dx = f64::from(input.spawn_x) - input.player_x;
    let dz = f64::from(input.spawn_z) - input.player_z;
    let yaw = input.yaw_degrees % 360.0;
    -((yaw - 90.0) * std::f64::consts::PI / 180.0 - dz.atan2(dx))
}

fn advance_clock(current_angle: &mut f64, angle_delta: &mut f64, target: f64) {
    let mut delta = target - *current_angle;
    while delta < -0.5 {
        delta += 1.0;
    }
    while delta >= 0.5 {
        delta -= 1.0;
    }
    delta = delta.clamp(-1.0, 1.0);
    *angle_delta += delta * 0.1;
    *angle_delta *= 0.8;
    *current_angle += *angle_delta;
}

fn advance_compass(current_angle: &mut f64, angle_delta: &mut f64, target: f64) {
    let mut delta = target - *current_angle;
    while delta < -std::f64::consts::PI {
        delta += std::f64::consts::TAU;
    }
    while delta >= std::f64::consts::PI {
        delta -= std::f64::consts::TAU;
    }
    // Yes, vanilla clamps radians to +/- 1.0 rather than +/- PI. This is visible as the
    // compass needle's maximum turn speed and must not be "corrected" to a full semicircle.
    delta = delta.clamp(-1.0, 1.0);
    *angle_delta += delta * 0.1;
    *angle_delta *= 0.8;
    *current_angle += *angle_delta;
}

fn positive_frame(value: f64, frame_count: i32) -> i32 {
    let frame = value as i32 % frame_count;
    if frame < 0 {
        (frame + frame_count) % frame_count
    } else {
        frame
    }
}

impl AtlasAnimator {
    /// Build playback state from already-atlas-placed sprites.
    ///
    /// Static sprites are ignored.  The placement is recovered from the same half-texel-inset
    /// UV values [`build_atlas`] writes, avoiding a duplicate public placement table solely for
    /// animation.  A malformed externally-built sprite whose rectangle lies outside `atlas` is
    /// ignored rather than making a game tick panic.
    #[must_use]
    pub fn new<'a>(atlas: &Atlas, sprites: impl IntoIterator<Item = &'a Sprite>) -> Self {
        Self::new_with_anaglyph(atlas, sprites, false)
    }

    /// Build playback state with Java's optional upload-time anaglyph conversion enabled.
    ///
    /// The atlas itself remains raw: its initial upload should use [`Atlas::upload_pixels`], while
    /// the dirty rectangles returned by [`Self::tick`] carry their one converted upload copy.
    /// Keeping raw frame data in the animator avoids accumulating colour-space loss when the
    /// option changes.
    #[must_use]
    pub fn new_with_anaglyph<'a>(
        atlas: &Atlas,
        sprites: impl IntoIterator<Item = &'a Sprite>,
        anaglyph: bool,
    ) -> Self {
        let mip_levels = atlas.mipmaps.len() as u32;
        let sprites = sprites
            .into_iter()
            .filter_map(|sprite| {
                if sprite.frames.len() <= 1 || sprite.width == 0 || sprite.height == 0 {
                    return None;
                }
                // `uv_for_placement` adds +0.01 / side, so truncation recovers the exact
                // integer placement even after the values pass through f32.
                let x = (sprite.uv.min_u * atlas.w as f32) as u32;
                let y = (sprite.uv.min_v * atlas.h as f32) as u32;
                if x.checked_add(sprite.width)? > atlas.w || y.checked_add(sprite.height)? > atlas.h
                {
                    return None;
                }
                let frames = sprite.generate_mipmaps(mip_levels);
                if frames
                    .iter()
                    .any(|chain| chain.len() != mip_levels as usize + 1)
                {
                    return None;
                }
                Some(AnimatedSprite {
                    location: sprite.location.clone(),
                    x,
                    y,
                    width: sprite.width,
                    height: sprite.height,
                    frames,
                    durations: sprite
                        .frames
                        .iter()
                        .map(|frame| {
                            frame
                                .duration_ticks
                                .unwrap_or(sprite.default_duration_ticks)
                                .max(1)
                        })
                        .collect(),
                    interpolate: sprite.interpolate,
                    kind: animation_kind(sprite),
                    // `TextureAtlasSprite` begins here even when explicit metadata's first
                    // playback entry is not sheet frame zero.  `build_atlas` has already put
                    // sheet frame zero in the atlas; the first update advances from playback 0.
                    frame: 0,
                    tick_in_frame: 0,
                    anaglyph,
                })
            })
            .collect();
        Self { sprites, anaglyph }
    }

    /// Change the upload-time anaglyph switch for subsequent animation rectangles.
    ///
    /// The caller must re-upload the static atlas levels with [`Atlas::upload_pixels`] when this
    /// setting changes; this setter only controls future dirty rectangles and never mutates a
    /// caller-owned GPU resource.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.anaglyph = enabled;
        for sprite in &mut self.sprites {
            sprite.anaglyph = enabled;
        }
    }

    /// Whether animation rectangles are currently converted at their upload boundary.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Number of animated sprites this controller owns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sprites.len()
    }

    /// Whether no animated sprites were registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sprites.is_empty()
    }

    /// Advance metadata-driven animations by one 20 Hz client tick and patch `atlas` in place.
    ///
    /// This deliberately leaves `items/clock` and `items/compass` unchanged: their vanilla
    /// subclasses need a world/player snapshot. Use [`AtlasAnimator::tick_with_context`] when
    /// those sprites are present in a live atlas.
    ///
    /// The returned rectangles contain exactly the bytes written to the CPU atlas.  Upload them
    /// to each live GPU atlas before its next draw.  If no frame boundary or interpolation update
    /// occurs this returns an empty vector.
    pub fn tick(&mut self, atlas: &mut Atlas) -> Vec<AtlasAnimationUpload> {
        self.tick_inner(atlas, None)
    }

    /// Advance normal metadata sprites and the special clock/compass sprites with an explicit
    /// world snapshot. Call this instead of [`AtlasAnimator::tick`] in the live client.
    pub fn tick_with_context(
        &mut self,
        atlas: &mut Atlas,
        context: AtlasAnimationContext,
    ) -> Vec<AtlasAnimationUpload> {
        self.tick_inner(atlas, Some(context))
    }

    fn tick_inner(
        &mut self,
        atlas: &mut Atlas,
        context: Option<AtlasAnimationContext>,
    ) -> Vec<AtlasAnimationUpload> {
        let mut uploads = Vec::new();
        for sprite in &mut self.sprites {
            sprite.tick(atlas, context, &mut uploads);
        }
        uploads
    }

    /// Locations of every animated atlas entry, including the special clock/compass sprites.
    #[must_use]
    pub fn locations(&self) -> impl ExactSizeIterator<Item = &ResourceLocation> {
        self.sprites.iter().map(|sprite| &sprite.location)
    }

    /// Write every animated sprite's current picture (the bytes its last
    /// [`Self::tick_with_context`] left, or frame 0 for a fresh animator) into `atlas`, and return
    /// the rectangles to upload. A replay checkpoint restores an animator's counters while the
    /// atlas still shows whatever later frames were drawn since; this puts the atlas back in step
    /// with the counters.
    pub fn write_current_frames(&self, atlas: &mut Atlas) -> Vec<AtlasAnimationUpload> {
        let mut uploads = Vec::new();
        for sprite in &self.sprites {
            sprite.write_current(atlas, &mut uploads);
        }
        uploads
    }
}

impl AnimatedSprite {
    fn tick(
        &mut self,
        atlas: &mut Atlas,
        context: Option<AtlasAnimationContext>,
        uploads: &mut Vec<AtlasAnimationUpload>,
    ) {
        if !matches!(self.kind, AnimationKind::Metadata) {
            if let Some(context) = context {
                self.tick_dynamic(atlas, context, uploads);
            }
            return;
        }
        self.tick_in_frame += 1;
        let duration = self.durations[self.frame];
        if self.tick_in_frame >= duration {
            let old = self.frame;
            self.frame = (self.frame + 1) % self.frames.len();
            self.tick_in_frame = 0;
            // Java compares the source sheet indices before upload.  Decoded SpriteFrame does
            // not retain those indices, but equal decoded bytes are observationally identical;
            // skipping them avoids a needless upload while preserving visible output.
            if self.frames[old][0] != self.frames[self.frame][0] {
                self.write_frame(atlas, self.frame, uploads);
            }
        } else if self.interpolate {
            let next = (self.frame + 1) % self.frames.len();
            if self.frames[self.frame][0] != self.frames[next][0] {
                self.write_interpolated(atlas, next, uploads);
            }
        }
    }

    fn tick_dynamic(
        &mut self,
        atlas: &mut Atlas,
        context: AtlasAnimationContext,
        uploads: &mut Vec<AtlasAnimationUpload>,
    ) {
        let frame_count = self.frames.len() as i32;
        let next_frame = match &mut self.kind {
            AnimationKind::Metadata => unreachable!("checked by caller"),
            AnimationKind::Clock {
                current_angle,
                angle_delta,
            } => {
                let target = clock_target(context.clock);
                advance_clock(current_angle, angle_delta, target);
                positive_frame((*current_angle + 1.0) * f64::from(frame_count), frame_count)
            }
            AnimationKind::Compass {
                current_angle,
                angle_delta,
            } => {
                let target = compass_target(context.compass);
                advance_compass(current_angle, angle_delta, target);
                positive_frame(
                    (*current_angle / std::f64::consts::TAU + 1.0) * f64::from(frame_count),
                    frame_count,
                )
            }
        } as usize;
        if next_frame != self.frame {
            self.frame = next_frame;
            self.write_frame(atlas, next_frame, uploads);
        }
    }

    /// What the atlas shows for this sprite after its last tick: an interpolated sprite between
    /// two different frames is blended by `tick_in_frame` (exactly `frames[frame]` at 0), every
    /// other sprite is `frames[frame]` (a tick that moved to an identical frame skipped the
    /// write, and those bytes are equal).
    fn write_current(&self, atlas: &mut Atlas, uploads: &mut Vec<AtlasAnimationUpload>) {
        if matches!(self.kind, AnimationKind::Metadata) && self.interpolate {
            let next = (self.frame + 1) % self.frames.len();
            if self.frames[self.frame][0] != self.frames[next][0] {
                self.write_interpolated(atlas, next, uploads);
                return;
            }
        }
        self.write_frame(atlas, self.frame, uploads);
    }

    fn write_frame(
        &self,
        atlas: &mut Atlas,
        frame: usize,
        uploads: &mut Vec<AtlasAnimationUpload>,
    ) {
        for (level, pixels) in self.frames[frame].iter().enumerate() {
            self.write_level(atlas, level as u32, pixels.clone(), uploads);
        }
    }

    fn write_interpolated(
        &self,
        atlas: &mut Atlas,
        next: usize,
        uploads: &mut Vec<AtlasAnimationUpload>,
    ) {
        // java: `TextureAtlasSprite.updateAnimationInterpolated` weighs red, green and blue in
        // `double` (`d0 = 1 - tickCounter / frameTime`, each channel `(int)(cur * d0 + next *
        // (1 - d0))`) and keeps the current frame's alpha (`j1 & -16777216`).
        let current_weight =
            1.0 - f64::from(self.tick_in_frame) / f64::from(self.durations[self.frame]);
        let next_weight = 1.0 - current_weight;
        for (level, current) in self.frames[self.frame].iter().enumerate() {
            let blended = current
                .chunks_exact(4)
                .zip(self.frames[next][level].chunks_exact(4))
                .flat_map(|(a, b)| {
                    let mix = |c: usize| {
                        (f64::from(a[c]) * current_weight + f64::from(b[c]) * next_weight) as u8
                    };
                    [mix(0), mix(1), mix(2), a[3]]
                })
                .collect();
            self.write_level(atlas, level as u32, blended, uploads);
        }
    }

    fn write_level(
        &self,
        atlas: &mut Atlas,
        level: u32,
        rgba: Vec<u8>,
        uploads: &mut Vec<AtlasAnimationUpload>,
    ) {
        let width = self.width >> level;
        let height = self.height >> level;
        let x = self.x >> level;
        let y = self.y >> level;
        let (target, target_width) = if level == 0 {
            (&mut atlas.texture, atlas.w)
        } else {
            (&mut atlas.mipmaps[(level - 1) as usize], atlas.w >> level)
        };
        blit(target, target_width, &rgba, width, height, x, y, false);
        // Keep the CPU atlas canonical/raw.  Java's transform belongs to the temporary upload
        // array, so an enabled animator must not leave one mixed raw/converted atlas behind for a
        // later setting toggle or another GPU owner.  ATLAS-6: with the transform off (the live
        // animator is always built that way) the raw bytes move into the record uncopied.
        let upload_rgba = if self.anaglyph {
            anaglyph_rgba(&rgba, true)
        } else {
            rgba
        };
        uploads.push(AtlasAnimationUpload {
            mip_level: level,
            x,
            y,
            width,
            height,
            rgba: upload_rgba,
        });
    }
}

/// One pixel used for the well-known magenta/black "missing texture" checkerboard, and the
/// dimensions it's generated at.
const MISSING_TEXTURE_SIZE: u32 = 16;
/// `GameSettings.mipmapLevels` defaults to four in 1.8.9.  The loader may lower this globally
/// when a pack contains a sprite whose dimensions cannot support all four levels.
pub const DEFAULT_MIPMAP_LEVELS: u32 = 4;

/// The synthetic sprite `TextureMap` installs before it attempts to load pack textures.
///
/// This is deliberately *not* `textures/missingno`: in 1.8.9 the resource location is the
/// atlas key `minecraft:missingno`; its pixels are generated by `TextureUtil.missingTextureData`
/// rather than read from a resource pack file.
#[must_use]
pub(crate) fn missing_texture_location() -> ResourceLocation {
    ResourceLocation::minecraft("missingno")
}

pub(crate) fn missing_texture_pixels() -> Vec<u8> {
    let mut out = Vec::with_capacity((MISSING_TEXTURE_SIZE * MISSING_TEXTURE_SIZE * 4) as usize);
    for y in 0..MISSING_TEXTURE_SIZE {
        for x in 0..MISSING_TEXTURE_SIZE {
            let half = MISSING_TEXTURE_SIZE / 2;
            let magenta = (x < half) == (y < half);
            if magenta {
                out.extend_from_slice(&[0xF8, 0x00, 0xF8, 0xFF]);
            } else {
                out.extend_from_slice(&[0x00, 0x00, 0x00, 0xFF]);
            }
        }
    }
    out
}

/// Loads every texture in `locations` (each e.g. `minecraft:textures/blocks/stone`, no `.png`
/// suffix — [`crate::model`] resolves these), stitches them into one [`Atlas`], and returns
/// the atlas together with every sprite's placement. Locations that fail to load fall back to
/// the missing-texture checkerboard rather than aborting the whole bake, since one broken
/// resource pack texture shouldn't take down every other block's rendering.
///
/// # Errors
/// [`AssetError`] only for the packing itself failing (a texture too large for the configured
/// cap) — individual missing/corrupt textures are logged and substituted, not propagated.
pub fn build_atlas(
    rm: &ResourceManager,
    locations: &[ResourceLocation],
    max_dimension: u32,
) -> Result<(Atlas, HashMap<ResourceLocation, Sprite>), AssetError> {
    build_atlas_with_mipmap_levels(rm, locations, max_dimension, DEFAULT_MIPMAP_LEVELS)
}

/// [`build_atlas`] with the user's `GameSettings.mipmapLevels` instead of the 1.8.9 default.
///
/// java: `Minecraft.startGame` and `GameSettings.setOptionFloatValue(MIPMAP_LEVELS)` call
/// `TextureMap.setMipmapLevels(gameSettings.mipmapLevels)` before `loadTextureAtlas`, which
/// sizes the `Stitcher` alignment, every sprite's `generateMipmaps` and the GL texture's
/// level count from it (lowered further by the smallest sprite, see `atlas_mipmap_levels`).
/// Zero therefore stitches without mip alignment and uploads level 0 only. §7.6 / ATLAS-4.
pub fn build_atlas_with_mipmap_levels(
    rm: &ResourceManager,
    locations: &[ResourceLocation],
    max_dimension: u32,
    mipmap_levels: u32,
) -> Result<(Atlas, HashMap<ResourceLocation, Sprite>), AssetError> {
    let mut sprites: Vec<LoadedSprite> = Vec::with_capacity(locations.len());
    for loc in locations {
        let (sprite, atlas_frame, atlas_mipmaps) = if *loc == missing_texture_location() {
            let pixels = missing_texture_pixels();
            (
                Sprite {
                    location: loc.clone(),
                    width: MISSING_TEXTURE_SIZE,
                    height: MISSING_TEXTURE_SIZE,
                    uv: UvRect::FULL,
                    frames: vec![crate::sprite::SpriteFrame {
                        rgba: pixels.clone(),
                        duration_ticks: None,
                    }],
                    default_duration_ticks: 1,
                    interpolate: false,
                    texture_metadata: Default::default(),
                },
                pixels,
                Vec::new(),
            )
        } else {
            match load_sprite_with_atlas_frame(rm, loc, mipmap_levels) {
                Ok(loaded) => loaded,
                Err(e) => {
                    tracing::warn!(location = %loc, error = %e, "texture failed to load; using missing-texture placeholder");
                    let pixels = missing_texture_pixels();
                    (
                        Sprite {
                            location: loc.clone(),
                            width: MISSING_TEXTURE_SIZE,
                            height: MISSING_TEXTURE_SIZE,
                            uv: UvRect::FULL,
                            frames: vec![crate::sprite::SpriteFrame {
                                rgba: pixels.clone(),
                                duration_ticks: None,
                            }],
                            default_duration_ticks: 1,
                            interpolate: false,
                            texture_metadata: Default::default(),
                        },
                        pixels,
                        Vec::new(),
                    )
                }
            }
        };
        sprites.push((sprite, atlas_frame, atlas_mipmaps));
    }

    let mipmap_levels = atlas_mipmap_levels(&sprites, mipmap_levels);
    // Java's TextureMap registers sprites by icon name and Stitcher.Holder compares that
    // name after the mipmap-rounded dimensions.  Preserve that tie-breaker here instead of
    // letting a caller's `locations` order choose coordinates for equal-sized custom-pack
    // sprites.
    let sizes: Vec<NamedSize> = sprites
        .iter()
        .map(|(sprite, _, _)| {
            NamedSize::new(sprite.location.to_string(), sprite.width, sprite.height)
        })
        .collect();
    let layout =
        stitch_with_mipmap_named(&sizes, max_dimension, max_dimension, true, mipmap_levels)
            .map_err(stitch_error_to_asset)?;

    let mut texture = vec![0u8; (layout.width * layout.height * 4) as usize];
    let mut mipmaps: Vec<Vec<u8>> = (1..=mipmap_levels)
        .map(|level| vec![0u8; ((layout.width >> level) * (layout.height >> level) * 4) as usize])
        .collect();
    for placement in &layout.placements {
        let (sprite, atlas_frame, atlas_mipmaps) = &mut sprites[placement.index];
        blit(
            &mut texture,
            layout.width,
            atlas_frame,
            sprite.width,
            sprite.height,
            placement.x,
            placement.y,
            placement.rotated,
        );
        let chain = crate::sprite::generate_mipmap_chain_with_overrides(
            atlas_frame,
            sprite.width,
            sprite.height,
            mipmap_levels,
            atlas_mipmaps,
        );
        for (level, mip) in chain.iter().enumerate().skip(1) {
            let level = level as u32;
            blit(
                &mut mipmaps[(level - 1) as usize],
                layout.width >> level,
                mip,
                sprite.width >> level,
                sprite.height >> level,
                placement.x >> level,
                placement.y >> level,
                placement.rotated,
            );
        }
        sprite.uv = uv_for_placement(
            placement.x,
            placement.y,
            sprite.width,
            sprite.height,
            layout.width,
            layout.height,
        );
    }

    let mut by_location = HashMap::with_capacity(sprites.len());
    for (s, _, _) in sprites {
        by_location.insert(s.location.clone(), s);
    }

    Ok((
        Atlas {
            texture,
            mipmaps,
            w: layout.width,
            h: layout.height,
        },
        by_location,
    ))
}

/// Build an atlas and return a one-shot upload copy with Java's optional anaglyph conversion.
///
/// The ordinary [`build_atlas`] API remains the canonical raw bake.  This additive entry point is
/// the terrain upload seam for callers that already know `GameSettings::anaglyph`; it converts
/// level zero and every generated/explicit mip only after stitching and mip generation, matching
/// `TextureUtil.uploadTextureMipmap` rather than changing the source pixels used by the mip filter.
pub fn build_atlas_with_anaglyph(
    rm: &ResourceManager,
    locations: &[ResourceLocation],
    max_dimension: u32,
    anaglyph: bool,
) -> Result<(Atlas, HashMap<ResourceLocation, Sprite>), AssetError> {
    let (mut atlas, sprites) = build_atlas(rm, locations, max_dimension)?;
    atlas.apply_anaglyph(anaglyph);
    Ok((atlas, sprites))
}

/// Java lowers `TextureMap.mipmapLevels` to the smallest lowest-one-bit among loaded sprites,
/// capped by the configured `mipmapLevels`.  The global reduction matters: every atlas placement
/// must share the same alignment and every uploaded level the same dimensions.
fn atlas_mipmap_levels(sprites: &[LoadedSprite], configured: u32) -> u32 {
    sprites
        .iter()
        .map(|(sprite, _, _)| sprite.width.min(sprite.height).trailing_zeros())
        .min()
        .unwrap_or(0)
        .min(configured)
}

fn stitch_error_to_asset(e: StitchError) -> AssetError {
    AssetError::ModelResolve {
        location: ResourceLocation::minecraft("atlas"),
        reason: e.to_string(),
    }
}

/// java: `TextureAtlasSprite.initSprite` — a half-texel epsilon inset keeps neighbouring
/// sprites from bleeding into this one under bilinear filtering.
fn uv_for_placement(x: u32, y: u32, w: u32, h: u32, sheet_w: u32, sheet_h: u32) -> UvRect {
    let eps_u = 0.01 / f64::from(sheet_w.max(1));
    let eps_v = 0.01 / f64::from(sheet_h.max(1));
    UvRect {
        min_u: (f64::from(x) / f64::from(sheet_w.max(1)) + eps_u) as f32,
        max_u: (f64::from(x + w) / f64::from(sheet_w.max(1)) - eps_u) as f32,
        min_v: (f64::from(y) / f64::from(sheet_h.max(1)) + eps_v) as f32,
        max_v: (f64::from(y + h) / f64::from(sheet_h.max(1)) - eps_v) as f32,
    }
}

#[allow(clippy::too_many_arguments)]
fn blit(
    dst: &mut [u8],
    dst_w: u32,
    src: &[u8],
    src_w: u32,
    src_h: u32,
    x: u32,
    y: u32,
    _rotated: bool,
) {
    for row in 0..src_h {
        for col in 0..src_w {
            let src_o = ((row * src_w + col) * 4) as usize;
            if src_o + 4 > src.len() {
                continue;
            }
            // Vanilla records Stitcher.Holder's rotation flag but uploads the original pixel
            // dimensions and derives UVs from them; the flag is otherwise vestigial.
            let (dx, dy) = (col, row);
            let dst_o = (((y + dy) * dst_w + (x + dx)) * 4) as usize;
            if dst_o + 4 > dst.len() {
                continue;
            }
            dst[dst_o..dst_o + 4].copy_from_slice(&src[src_o..src_o + 4]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FolderResourcePack;

    #[test]
    fn missing_texture_is_a_16x16_checkerboard() {
        let px = missing_texture_pixels();
        assert_eq!(px.len(), (16 * 16 * 4) as usize);
        // top-left quadrant magenta, adjacent quadrant black
        assert_eq!(&px[0..4], &[0xF8, 0x00, 0xF8, 0xFF]);
        let half_row_offset = (8 * 4) as usize; // pixel (8,0)
        assert_eq!(
            &px[half_row_offset..half_row_offset + 4],
            &[0x00, 0x00, 0x00, 0xFF]
        );
    }

    #[test]
    fn atlas_upload_copies_convert_every_mip_and_animation_rect_once() {
        let atlas = Atlas {
            texture: vec![255, 0, 0, 7, 128, 128, 128, 64],
            mipmaps: vec![vec![0, 255, 0, 19]],
            w: 2,
            h: 1,
        };
        let (base, mips) = atlas.upload_pixels(true);
        assert_eq!(base, [76, 76, 76, 7, 128, 128, 128, 64]);
        assert_eq!(mips, vec![vec![178, 150, 0, 19]]);
        assert_eq!(atlas.texture, [255, 0, 0, 7, 128, 128, 128, 64]);

        let upload = AtlasAnimationUpload {
            mip_level: 0,
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            rgba: vec![0, 0, 255, 31],
        };
        assert_eq!(
            upload.upload_pixels(true).rgba,
            [0, 28, 178, 31],
            "dirty rectangles use the same boundary transform as the static atlas"
        );
        assert_eq!(upload.upload_pixels(false).rgba, upload.rgba);
    }

    #[test]
    fn synthetic_missingno_is_stitched_as_a_checkerboard_sprite() {
        let resources = ResourceManager::new();
        let missing = missing_texture_location();
        // Keep a normal failed pack texture beside it: both have checker pixels, but only the
        // special `missingno` key is registered as vanilla's reusable fallback sprite.
        let other = ResourceLocation::minecraft("textures/blocks/not_present");
        let (atlas, sprites) = build_atlas(&resources, &[other, missing.clone()], 64).unwrap();
        let sprite = &sprites[&missing];
        assert_ne!(sprite.uv, UvRect::FULL);
        assert_eq!((sprite.width, sprite.height), (16, 16));

        let x = (sprite.uv.min_u * atlas.w as f32).round() as u32;
        let y = (sprite.uv.min_v * atlas.h as f32).round() as u32;
        let pixel = |x: u32, y: u32| {
            let at = ((y * atlas.w + x) * 4) as usize;
            [
                atlas.texture[at],
                atlas.texture[at + 1],
                atlas.texture[at + 2],
                atlas.texture[at + 3],
            ]
        };
        assert_eq!(pixel(x, y), [0xF8, 0x00, 0xF8, 0xFF]);
        assert_eq!(pixel(x + 8, y), [0x00, 0x00, 0x00, 0xFF]);
    }

    #[test]
    fn atlas_starts_explicit_animations_from_sheet_frame_zero() {
        let temp = tempfile::tempdir().unwrap();
        let texture_dir = temp.path().join("assets/minecraft/textures/blocks");
        std::fs::create_dir_all(&texture_dir).unwrap();

        let mut image = image::RgbaImage::new(2, 4);
        for y in 0..2 {
            for x in 0..2 {
                image.put_pixel(x, y, image::Rgba([10, 20, 30, 255]));
                image.put_pixel(x, y + 2, image::Rgba([200, 210, 220, 255]));
            }
        }
        image.save(texture_dir.join("reordered.png")).unwrap();
        std::fs::write(
            texture_dir.join("reordered.png.mcmeta"),
            br#"{"animation":{"frames":[1,0]}}"#,
        )
        .unwrap();

        let mut resources = ResourceManager::new();
        resources.push_pack(Box::new(FolderResourcePack::open(temp.path()).unwrap()));
        let location = ResourceLocation::minecraft("textures/blocks/reordered");
        let (atlas, sprites) =
            build_atlas(&resources, std::slice::from_ref(&location), 64).unwrap();

        assert_eq!(&atlas.texture[..4], &[10, 20, 30, 255]);
        assert_eq!(
            &sprites[&location].frames[0].rgba[..4],
            &[200, 210, 220, 255]
        );
    }

    fn write_solid_png(path: &std::path::Path, width: u32, height: u32, color: [u8; 4]) {
        image::RgbaImage::from_pixel(width, height, image::Rgba(color))
            .save(path)
            .unwrap();
    }

    fn oracle_pack(
        base_size: (u32, u32),
        metadata: &[u8],
        explicit_mip: Option<(u32, u32, [u8; 4])>,
    ) -> (tempfile::TempDir, ResourceManager, ResourceLocation) {
        let temp = tempfile::tempdir().unwrap();
        let texture_dir = temp.path().join("assets/minecraft/textures/blocks");
        std::fs::create_dir_all(&texture_dir).unwrap();
        write_solid_png(
            &texture_dir.join("oracle.png"),
            base_size.0,
            base_size.1,
            [255, 0, 0, 255],
        );
        std::fs::write(texture_dir.join("oracle.png.mcmeta"), metadata).unwrap();
        if let Some((width, height, color)) = explicit_mip {
            let mip_dir = temp.path().join("assets/minecraft/textures/mipmaps/blocks");
            std::fs::create_dir_all(&mip_dir).unwrap();
            write_solid_png(&mip_dir.join("oracle.1.png"), width, height, color);
        }
        let mut resources = ResourceManager::new();
        resources.push_pack(Box::new(FolderResourcePack::open(temp.path()).unwrap()));
        (
            temp,
            resources,
            ResourceLocation::minecraft("textures/blocks/oracle"),
        )
    }

    fn atlas_mip_pixel(atlas: &Atlas, sprite: &Sprite, level: u32) -> [u8; 4] {
        let x = ((sprite.uv.min_u * atlas.w as f32) as u32) >> level;
        let y = ((sprite.uv.min_v * atlas.h as f32) as u32) >> level;
        let width = atlas.w >> level;
        let mip = if level == 0 {
            &atlas.texture
        } else {
            &atlas.mipmaps[(level - 1) as usize]
        };
        let offset = ((y * width + x) * 4) as usize;
        [
            mip[offset],
            mip[offset + 1],
            mip[offset + 2],
            mip[offset + 3],
        ]
    }

    /// §7.6 / ATLAS-4: `mipmapLevels` caps the stitched level count (vanilla's 0–4 slider), the
    /// default entry point is exactly the explicit level-4 bake, and the smallest sprite still
    /// lowers an over-large setting the way `TextureMap.loadTextureAtlas` does.
    #[test]
    fn configured_mipmap_levels_cap_the_atlas_level_count() {
        let (_temp, resources, location) =
            oracle_pack((16, 16), br#"{"texture":{"mipmaps":[1]}}"#, None);
        let locations = [location.clone(), missing_texture_location()];
        let (default, default_sprites) = build_atlas(&resources, &locations, 64).unwrap();
        let (explicit, explicit_sprites) =
            build_atlas_with_mipmap_levels(&resources, &locations, 64, DEFAULT_MIPMAP_LEVELS)
                .unwrap();
        assert_eq!((default.w, default.h), (explicit.w, explicit.h));
        assert_eq!(default.texture, explicit.texture);
        assert_eq!(default.mipmaps, explicit.mipmaps);
        assert_eq!(default_sprites, explicit_sprites);
        for levels in 0..=DEFAULT_MIPMAP_LEVELS {
            let (atlas, sprites) =
                build_atlas_with_mipmap_levels(&resources, &locations, 64, levels).unwrap();
            assert_eq!(atlas.mipmaps.len() as u32, levels, "mipmapLevels={levels}");
            assert_eq!(
                atlas_mip_pixel(&atlas, &sprites[&location], 0),
                [255, 0, 0, 255]
            );
        }
        let (capped, _) = build_atlas_with_mipmap_levels(&resources, &locations, 64, 9).unwrap();
        assert_eq!(
            capped.mipmaps.len(),
            4,
            "16x16 sprites support at most four levels"
        );
    }

    /// With `mipmapLevels=0` a frame change uploads one record instead of five.
    #[test]
    fn level_zero_atlas_animates_with_one_record_per_frame_change() {
        let mut atlas = Atlas {
            texture: solid_frame([255, 0, 0, 255]).rgba,
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        };
        let sprite = two_frame_sprite(false);
        let mut animator = AtlasAnimator::new(&atlas, [&sprite]);
        assert_eq!(animator.len(), 1);
        assert!(animator.tick(&mut atlas).is_empty());
        let uploads = animator.tick(&mut atlas);
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].mip_level, 0);
        assert_eq!(atlas.texture, solid_frame([0, 0, 255, 255]).rgba);
    }

    #[test]
    fn atlas_explicit_mip_png_wins_over_generated_red() {
        let (_temp, resources, location) = oracle_pack(
            (16, 16),
            br#"{"texture":{"mipmaps":[1]}}"#,
            Some((8, 8, [0, 0, 255, 255])),
        );
        let (atlas, sprites) =
            build_atlas(&resources, std::slice::from_ref(&location), 64).unwrap();
        assert_eq!(
            atlas_mip_pixel(&atlas, &sprites[&location], 0),
            [255, 0, 0, 255]
        );
        assert_eq!(
            atlas_mip_pixel(&atlas, &sprites[&location], 1),
            [0, 0, 255, 255]
        );
        assert_eq!(
            sprites[&location].texture_metadata.explicit_mipmaps[0][0].0,
            1
        );
    }

    /// Phase 4d review S1: `TextureMap` loads an explicit level only below the configured
    /// `mipmapLevels` (`i2 < abufferedimage.length - 1`); at a setting of 1 a pack's level-1
    /// image is ignored and the level is generated, at 2 it is used.
    #[test]
    fn explicit_mips_load_only_below_the_configured_mipmap_levels() {
        let (_temp, resources, location) = oracle_pack(
            (16, 16),
            br#"{"texture":{"mipmaps":[1]}}"#,
            Some((8, 8, [0, 0, 255, 255])),
        );
        let build = |levels| {
            build_atlas_with_mipmap_levels(&resources, std::slice::from_ref(&location), 64, levels)
                .unwrap()
        };
        let (atlas, sprites) = build(1);
        assert_eq!(
            atlas_mip_pixel(&atlas, &sprites[&location], 1),
            [255, 0, 0, 255],
            "generated at mipmapLevels 1"
        );
        let (atlas, sprites) = build(2);
        assert_eq!(
            atlas_mip_pixel(&atlas, &sprites[&location], 1),
            [0, 0, 255, 255],
            "explicit at mipmapLevels 2"
        );
    }

    #[test]
    fn atlas_explicit_mip_missing_png_falls_back_to_generated_red() {
        let (_temp, resources, location) =
            oracle_pack((16, 16), br#"{"texture":{"mipmaps":[1]}}"#, None);
        let (atlas, sprites) =
            build_atlas(&resources, std::slice::from_ref(&location), 64).unwrap();
        assert_eq!(
            atlas_mip_pixel(&atlas, &sprites[&location], 1),
            [255, 0, 0, 255]
        );
        assert!(sprites[&location]
            .texture_metadata
            .explicit_mipmaps
            .is_empty());
    }

    #[test]
    fn atlas_explicit_mip_wrong_dimension_rejects_the_sprite() {
        let (_temp, resources, location) = oracle_pack(
            (16, 16),
            br#"{"texture":{"mipmaps":[1]}}"#,
            Some((4, 4, [0, 0, 255, 255])),
        );
        let (_atlas, sprites) =
            build_atlas(&resources, std::slice::from_ref(&location), 64).unwrap();
        assert_eq!(
            &sprites[&location].frames[0].rgba[..4],
            &[0xF8, 0x00, 0xF8, 0xFF]
        );
    }

    #[test]
    fn atlas_explicit_mip_non_power_of_two_rejects_but_metadata_coercion_survives() {
        let (_temp, non_power, non_power_location) = oracle_pack(
            (3, 3),
            br#"{"texture":{"mipmaps":[1]}}"#,
            Some((1, 1, [0, 0, 255, 255])),
        );
        let (_atlas, sprites) =
            build_atlas(&non_power, std::slice::from_ref(&non_power_location), 64).unwrap();
        assert_eq!(
            &sprites[&non_power_location].frames[0].rgba[..4],
            &[0xF8, 0x00, 0xF8, 0xFF]
        );

        let (_temp, malformed, malformed_location) =
            oracle_pack((16, 16), br#"{"texture":{"mipmaps":[1.5]}}"#, None);
        let (_atlas, sprites) =
            build_atlas(&malformed, std::slice::from_ref(&malformed_location), 64).unwrap();
        assert_eq!(
            &sprites[&malformed_location].frames[0].rgba[..4],
            &[255, 0, 0, 255],
            "Java parses numeric 1.5 as mip level 1, then generates it when the override is absent"
        );
    }

    #[test]
    fn atlas_malformed_explicit_mip_png_falls_back_to_generated_red() {
        let temp = tempfile::tempdir().unwrap();
        let texture_dir = temp.path().join("assets/minecraft/textures/blocks");
        let mip_dir = temp.path().join("assets/minecraft/textures/mipmaps/blocks");
        std::fs::create_dir_all(&texture_dir).unwrap();
        std::fs::create_dir_all(&mip_dir).unwrap();
        write_solid_png(&texture_dir.join("oracle.png"), 16, 16, [255, 0, 0, 255]);
        std::fs::write(
            texture_dir.join("oracle.png.mcmeta"),
            br#"{"texture":{"mipmaps":[1]}}"#,
        )
        .unwrap();
        std::fs::write(mip_dir.join("oracle.1.png"), b"not a png").unwrap();

        let mut resources = ResourceManager::new();
        resources.push_pack(Box::new(FolderResourcePack::open(temp.path()).unwrap()));
        let location = ResourceLocation::minecraft("textures/blocks/oracle");
        let (atlas, sprites) =
            build_atlas(&resources, std::slice::from_ref(&location), 64).unwrap();
        assert_eq!(
            atlas_mip_pixel(&atlas, &sprites[&location], 1),
            [255, 0, 0, 255]
        );
        assert!(sprites[&location]
            .texture_metadata
            .explicit_mipmaps
            .is_empty());
    }

    #[test]
    fn animated_explicit_mips_follow_playback_order_and_uploads() {
        let temp = tempfile::tempdir().unwrap();
        let texture_dir = temp.path().join("assets/minecraft/textures/blocks");
        let mip_dir = temp.path().join("assets/minecraft/textures/mipmaps/blocks");
        std::fs::create_dir_all(&texture_dir).unwrap();
        std::fs::create_dir_all(&mip_dir).unwrap();

        let mut base = image::RgbaImage::new(2, 4);
        for y in 0..2 {
            for x in 0..2 {
                base.put_pixel(x, y, image::Rgba([255, 0, 0, 255]));
                base.put_pixel(x, y + 2, image::Rgba([0, 255, 0, 255]));
            }
        }
        base.save(texture_dir.join("animated.png")).unwrap();
        std::fs::write(
            texture_dir.join("animated.png.mcmeta"),
            br#"{"animation":{"frames":[1,0],"frametime":1},"texture":{"mipmaps":[1]}}"#,
        )
        .unwrap();
        let mut explicit = image::RgbaImage::new(1, 2);
        explicit.put_pixel(0, 0, image::Rgba([0, 0, 255, 255]));
        explicit.put_pixel(0, 1, image::Rgba([255, 255, 0, 255]));
        explicit.save(mip_dir.join("animated.1.png")).unwrap();

        let mut resources = ResourceManager::new();
        resources.push_pack(Box::new(FolderResourcePack::open(temp.path()).unwrap()));
        let location = ResourceLocation::minecraft("textures/blocks/animated");
        let (mut atlas, sprites) =
            build_atlas(&resources, std::slice::from_ref(&location), 64).unwrap();
        let sprite = &sprites[&location];
        assert_eq!(atlas_mip_pixel(&atlas, sprite, 1), [0, 0, 255, 255]);

        let mut animator = AtlasAnimator::new(&atlas, [sprite]);
        let uploads = animator.tick(&mut atlas);
        assert_eq!(uploads.len(), 2);
        assert_eq!(uploads[0].rgba[..4], [255, 0, 0, 255]);
        assert_eq!(uploads[1].mip_level, 1);
        assert_eq!(uploads[1].rgba, vec![0, 0, 255, 255]);
        assert_eq!(atlas_mip_pixel(&atlas, sprite, 1), [0, 0, 255, 255]);
    }

    #[test]
    fn atlas_retains_texture_sampler_metadata_from_the_winning_sidecar() {
        let temp = tempfile::tempdir().unwrap();
        let texture_dir = temp.path().join("assets/minecraft/textures/entity");
        std::fs::create_dir_all(&texture_dir).unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([80, 90, 100, 255]))
            .save(texture_dir.join("layer.png"))
            .unwrap();
        std::fs::write(
            texture_dir.join("layer.png.mcmeta"),
            br#"{"texture":{"blur":true,"clamp":true,"mipmaps":[1,2]}}"#,
        )
        .unwrap();

        let mut resources = ResourceManager::new();
        resources.push_pack(Box::new(FolderResourcePack::open(temp.path()).unwrap()));
        let location = ResourceLocation::minecraft("textures/entity/layer");
        let (_, sprites) = build_atlas(&resources, std::slice::from_ref(&location), 64).unwrap();
        assert_eq!(
            sprites[&location].texture_metadata,
            crate::TextureMetadata {
                blur: true,
                clamp: true,
                mipmaps: vec![1, 2],
                explicit_mipmaps: Vec::new(),
            }
        );
    }

    #[test]
    fn atlas_stitches_sprite_local_mips_without_neighbour_bleed() {
        let temp = tempfile::tempdir().unwrap();
        let texture_dir = temp.path().join("assets/minecraft/textures/blocks");
        std::fs::create_dir_all(&texture_dir).unwrap();

        let write = |name: &str, color: [u8; 4]| {
            let image = image::RgbaImage::from_pixel(16, 16, image::Rgba(color));
            image.save(texture_dir.join(format!("{name}.png"))).unwrap();
        };
        write("red", [255, 0, 0, 255]);
        write("blue", [0, 0, 255, 255]);

        let mut resources = ResourceManager::new();
        resources.push_pack(Box::new(FolderResourcePack::open(temp.path()).unwrap()));
        let red = ResourceLocation::minecraft("textures/blocks/red");
        let blue = ResourceLocation::minecraft("textures/blocks/blue");
        let (atlas, sprites) = build_atlas(&resources, &[red.clone(), blue.clone()], 64).unwrap();

        assert_eq!(
            atlas.mipmaps.len(),
            4,
            "16px sprites retain vanilla's default four mips"
        );
        // The two 16px base sprites occupy adjacent 8px regions at mip 1. If the atlas were
        // downsampled after stitching, their boundary texel would be purple; sprite-local mips
        // keep it exactly blue.
        let mip1 = &atlas.mipmaps[0];
        let pixel = |x: u32, y: u32| {
            let offset = ((y * (atlas.w >> 1) + x) * 4) as usize;
            [
                mip1[offset],
                mip1[offset + 1],
                mip1[offset + 2],
                mip1[offset + 3],
            ]
        };
        let first_mip_x = |location: &ResourceLocation| {
            (sprites[location].uv.min_u * atlas.w as f32 / 2.0).round() as u32
        };
        assert_eq!(pixel(first_mip_x(&red), 0), [255, 0, 0, 255]);
        assert_eq!(pixel(first_mip_x(&blue), 0), [0, 0, 255, 255]);
    }

    #[test]
    fn custom_pack_equal_size_coordinates_follow_icon_name_not_input_order() {
        let temp = tempfile::tempdir().unwrap();
        let texture_dir = temp.path().join("assets/minecraft/textures/blocks");
        std::fs::create_dir_all(&texture_dir).unwrap();
        write_solid_png(&texture_dir.join("zulu.png"), 16, 16, [255, 0, 0, 255]);
        write_solid_png(&texture_dir.join("alpha.png"), 16, 16, [0, 0, 255, 255]);

        let mut resources = ResourceManager::new();
        resources.push_pack(Box::new(FolderResourcePack::open(temp.path()).unwrap()));
        let zulu = ResourceLocation::minecraft("textures/blocks/zulu");
        let alpha = ResourceLocation::minecraft("textures/blocks/alpha");

        let (_, first) = build_atlas(&resources, &[zulu.clone(), alpha.clone()], 64).unwrap();
        let (_, second) = build_atlas(&resources, &[alpha.clone(), zulu.clone()], 64).unwrap();

        assert_eq!(first[&alpha].uv, second[&alpha].uv);
        assert_eq!(first[&zulu].uv, second[&zulu].uv);
        assert!(first[&alpha].uv.min_u < first[&zulu].uv.min_u);
    }

    #[test]
    fn atlas_mipmap_level_drops_to_smallest_sprite_lowest_bit() {
        let sprites = vec![
            (test_sprite(16, 16), vec![255; 16 * 16 * 4], Vec::new()),
            (test_sprite(24, 24), vec![255; 24 * 24 * 4], Vec::new()),
        ];
        assert_eq!(atlas_mipmap_levels(&sprites, DEFAULT_MIPMAP_LEVELS), 3);
        // §7.6 / ATLAS-4: the configured `mipmapLevels` is the other cap.
        assert_eq!(atlas_mipmap_levels(&sprites, 1), 1);
        assert_eq!(atlas_mipmap_levels(&sprites, 0), 0);
    }

    fn test_sprite(width: u32, height: u32) -> Sprite {
        Sprite {
            location: ResourceLocation::minecraft("textures/blocks/test"),
            width,
            height,
            uv: UvRect::FULL,
            frames: Vec::new(),
            default_duration_ticks: 1,
            interpolate: false,
            texture_metadata: Default::default(),
        }
    }

    #[test]
    fn blit_places_pixels_at_the_right_offset() {
        let mut dst = vec![0u8; 4 * 4 * 4];
        let src = vec![9u8; 2 * 2 * 4];
        blit(&mut dst, 4, &src, 2, 2, 1, 1, false);
        let at = |x: u32, y: u32| -> [u8; 4] {
            let o = ((y * 4 + x) * 4) as usize;
            [dst[o], dst[o + 1], dst[o + 2], dst[o + 3]]
        };
        assert_eq!(at(1, 1), [9, 9, 9, 9]);
        assert_eq!(at(2, 2), [9, 9, 9, 9]);
        assert_eq!(at(0, 0), [0, 0, 0, 0]);
        assert_eq!(at(3, 3), [0, 0, 0, 0]);
    }

    #[test]
    fn rotated_placements_keep_vanillas_unrotated_pixels() {
        let mut dst = vec![0u8; 4 * 4 * 4];
        let src: Vec<u8> = [
            [1, 0, 0, 255],
            [2, 0, 0, 255],
            [3, 0, 0, 255],
            [4, 0, 0, 255],
            [5, 0, 0, 255],
            [6, 0, 0, 255],
        ]
        .into_iter()
        .flatten()
        .collect();

        blit(&mut dst, 4, &src, 3, 2, 0, 0, true);
        let red_at = |x: u32, y: u32| dst[((y * 4 + x) * 4) as usize];

        assert_eq!([red_at(0, 0), red_at(1, 0), red_at(2, 0)], [1, 2, 3]);
        assert_eq!([red_at(0, 1), red_at(1, 1), red_at(2, 1)], [4, 5, 6]);
    }

    #[test]
    fn uv_rect_is_inset_and_within_bounds() {
        let uv = uv_for_placement(16, 0, 16, 16, 64, 64);
        assert!(uv.min_u > 0.25 && uv.min_u < 0.26);
        assert!(uv.max_u > 0.49 && uv.max_u < 0.5);
        assert!(uv.min_u < uv.max_u);
        assert!(uv.min_v < uv.max_v);
    }

    fn solid_frame(color: [u8; 4]) -> crate::sprite::SpriteFrame {
        crate::sprite::SpriteFrame {
            rgba: color.into_iter().cycle().take(2 * 2 * 4).collect(),
            duration_ticks: Some(2),
        }
    }

    fn two_frame_sprite(interpolate: bool) -> Sprite {
        Sprite {
            location: ResourceLocation::minecraft("textures/blocks/animated"),
            width: 2,
            height: 2,
            uv: UvRect::FULL,
            frames: vec![solid_frame([255, 0, 0, 255]), solid_frame([0, 0, 255, 255])],
            default_duration_ticks: 1,
            interpolate,
            texture_metadata: Default::default(),
        }
    }

    fn two_pixel_atlas() -> Atlas {
        Atlas {
            texture: solid_frame([255, 0, 0, 255]).rgba,
            mipmaps: vec![vec![255, 0, 0, 255]],
            w: 2,
            h: 2,
        }
    }

    #[test]
    fn atlas_animator_waits_for_the_frame_duration_then_patches_all_mips() {
        let mut atlas = two_pixel_atlas();
        let sprite = two_frame_sprite(false);
        let mut animator = AtlasAnimator::new(&atlas, [&sprite]);
        assert_eq!(animator.len(), 1);

        assert!(
            animator.tick(&mut atlas).is_empty(),
            "frame time is two ticks"
        );
        let uploads = animator.tick(&mut atlas);
        assert_eq!(
            uploads.len(),
            2,
            "level zero and the stitched mip both update"
        );
        assert_eq!(uploads[0].mip_level, 0);
        assert_eq!(uploads[0].rgba, solid_frame([0, 0, 255, 255]).rgba);
        assert_eq!(uploads[1].mip_level, 1);
        assert_eq!(&atlas.texture[..4], &[0, 0, 255, 255]);
        assert_eq!(atlas.mipmaps[0], uploads[1].rgba);
    }

    #[test]
    fn atlas_animator_interpolates_before_the_frame_boundary_with_java_truncation() {
        let mut atlas = two_pixel_atlas();
        let sprite = two_frame_sprite(true);
        let mut animator = AtlasAnimator::new(&atlas, [&sprite]);

        let uploads = animator.tick(&mut atlas);
        assert_eq!(uploads.len(), 2);
        // TextureAtlasSprite's channel blend is an `int` cast, not a round-to-nearest.
        assert_eq!(&uploads[0].rgba[..4], &[127, 0, 127, 255]);
        assert_eq!(&atlas.texture[..4], &[127, 0, 127, 255]);

        let uploads = animator.tick(&mut atlas);
        assert_eq!(uploads[0].rgba, solid_frame([0, 0, 255, 255]).rgba);
    }

    /// The first texel of the level-0 record one more tick produces.
    fn next_base_texel(animator: &mut AtlasAnimator, atlas: &mut Atlas) -> [u8; 4] {
        let uploads = animator.tick(atlas);
        let base = uploads
            .iter()
            .find(|upload| upload.mip_level == 0)
            .expect("a level-0 record every tick");
        [base.rgba[0], base.rgba[1], base.rgba[2], base.rgba[3]]
    }

    fn interpolated_sprite(frames: [([u8; 4], u32); 2]) -> Sprite {
        Sprite {
            location: ResourceLocation::minecraft("textures/blocks/animated"),
            width: 2,
            height: 2,
            uv: UvRect::FULL,
            frames: frames
                .iter()
                .map(|&(color, ticks)| crate::sprite::SpriteFrame {
                    rgba: color.repeat(2 * 2),
                    duration_ticks: Some(ticks),
                })
                .collect(),
            default_duration_ticks: 1,
            interpolate: true,
            texture_metadata: Default::default(),
        }
    }

    /// java: `TextureAtlasSprite.updateAnimationInterpolated:195-226` weighs red, green and blue
    /// in `double` (`d0 = 1 - tickCounter / frameTime`, each channel `(int)(cur * d0 + next *
    /// (1 - d0))`) and keeps the current frame's alpha as is (`j1 & -16777216`): alpha is never
    /// blended. Every expected texel below is that Java expression evaluated on JDK 21.
    #[test]
    fn interpolation_uses_java_double_weights_and_keeps_the_current_alpha() {
        let opaque = [200, 100, 0, 255];
        let translucent = [0, 100, 200, 51];
        let mut atlas = Atlas {
            texture: opaque.repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        };
        let sprite = interpolated_sprite([(opaque, 3), (translucent, 3)]);
        let mut animator = AtlasAnimator::new(&atlas, [&sprite]);
        let ticks: Vec<[u8; 4]> = (0..6)
            .map(|_| next_base_texel(&mut animator, &mut atlas))
            .collect();
        assert_eq!(
            ticks,
            [
                [133, 100, 66, 255],
                [66, 100, 133, 255],
                translucent,
                [66, 100, 133, 51],
                [133, 100, 66, 51],
                opaque,
            ]
        );

        // `prismarine_rough`'s 300-tick frames: at tick 297 Java's double weights give 1 and 2
        // where a float blend truncates to 0 and 1.
        let mut atlas = Atlas {
            texture: [100, 200, 0, 255].repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        };
        let sprite = interpolated_sprite([([100, 200, 0, 255], 300), ([0, 0, 0, 255], 300)]);
        let mut animator = AtlasAnimator::new(&atlas, [&sprite]);
        let texel = (0..297)
            .map(|_| next_base_texel(&mut animator, &mut atlas))
            .last();
        assert_eq!(texel, Some([1, 2, 0, 255]));
    }

    /// java: outside a surface world `TextureCompass.updateCompass` aims at `Math.random() *
    /// Math.PI * 2.0D`, which the 1.8.9 bytecode (`bmp`) compiles with the float-widened
    /// constant `3.1415927410125732`, not `Math.PI` (its other angle constants are exact).
    #[test]
    fn compass_random_target_uses_the_bytecode_pi() {
        let input = AtlasCompassInput {
            player_x: 0.0,
            player_z: 0.0,
            yaw_degrees: 0.0,
            spawn_x: 0,
            spawn_z: 0,
            is_surface_world: false,
            random: 0.75,
        };
        assert_eq!(
            compass_target(Some(input)),
            0.75 * f64::from(std::f32::consts::PI) * 2.0
        );
    }

    #[test]
    fn anaglyph_animator_converts_dirty_uploads_without_mixing_the_cpu_atlas() {
        let mut atlas = two_pixel_atlas();
        let sprite = two_frame_sprite(false);
        let mut animator = AtlasAnimator::new_with_anaglyph(&atlas, [&sprite], true);

        assert!(animator.tick(&mut atlas).is_empty());
        let uploads = animator.tick(&mut atlas);
        assert_eq!(&uploads[0].rgba[..4], &[0, 28, 178, 255]);
        assert_eq!(
            &atlas.texture[..4],
            &[0, 0, 255, 255],
            "the source atlas remains raw while the returned upload is converted"
        );
        assert!(animator.anaglyph_enabled());
        animator.set_anaglyph(false);
        assert!(!animator.anaglyph_enabled());
    }

    fn special_sprite(path: &str, frames: u8) -> Sprite {
        let frames = (0..frames)
            .map(|index| crate::sprite::SpriteFrame {
                rgba: [index, 0, 0, 255]
                    .into_iter()
                    .cycle()
                    .take(2 * 2 * 4)
                    .collect(),
                duration_ticks: None,
            })
            .collect();
        Sprite {
            location: ResourceLocation::minecraft(format!("textures/items/{path}")),
            width: 2,
            height: 2,
            uv: UvRect::FULL,
            frames,
            default_duration_ticks: 1,
            interpolate: false,
            texture_metadata: Default::default(),
        }
    }

    #[test]
    fn clock_uses_vanillas_surface_random_and_half_turn_boundaries() {
        // TextureClock's `d1 >= 0.5` loop makes exactly +0.5 turn backwards, not forwards.
        let mut angle = 0.0;
        let mut velocity = 0.0;
        advance_clock(&mut angle, &mut velocity, 0.5);
        assert!((velocity + 0.04).abs() < 1e-12);
        assert!((angle + 0.04).abs() < 1e-12);

        // `WorldProvider.calculateCelestialAngle` is f32 throughout and TextureClock samples
        // partialTicks=1.0. Noon is nearly zero rather than exactly zero because of that +1.
        assert!(celestial_angle(6_000).abs() < 0.000_03);
        assert_eq!(clock_target(AtlasClockInput::default()), 0.0);
        assert_eq!(
            clock_target(AtlasClockInput {
                world_time: Some(1),
                is_surface_world: false,
                random: 0.375,
            }),
            0.375,
            "non-surface worlds replace the celestial angle with Math.random()"
        );
    }

    #[test]
    fn compass_uses_spawn_bearing_radian_wrap_and_one_radian_speed_cap() {
        let input = AtlasCompassInput {
            player_x: 0.0,
            player_z: 0.0,
            yaw_degrees: 0.0,
            spawn_x: 10,
            spawn_z: 0,
            is_surface_world: true,
            random: 0.0,
        };
        // Java: -((yaw - 90) * PI / 180 - atan2(spawnZ-z, spawnX-x)).
        assert!((compass_target(Some(input)) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);

        let mut angle = 0.0;
        let mut velocity = 0.0;
        advance_compass(&mut angle, &mut velocity, compass_target(Some(input)));
        assert!(
            (velocity - 0.08).abs() < 1e-12,
            "the radians delta clamps at one, not PI"
        );
        assert!((angle - 0.08).abs() < 1e-12);
        assert_eq!(
            compass_target(None),
            0.0,
            "no world/player eases toward zero"
        );
        // This used to expect exactly PI / 2. The 1.8.9 bytecode (`bmp`)
        // multiplies by the float-widened 3.1415927410125732 here; see
        // `compass_random_target_uses_the_bytecode_pi`.
        assert_eq!(
            compass_target(Some(AtlasCompassInput {
                is_surface_world: false,
                random: 0.25,
                ..input
            })),
            0.25 * f64::from(std::f32::consts::PI) * 2.0,
            "non-surface compass input is Math.random() * (double)(float)PI * 2"
        );
    }

    #[test]
    fn compass_live_context_consumes_retained_spawn_instead_of_zero_fallback() {
        let sprite = special_sprite("compass", 16);
        let mut atlas = Atlas {
            texture: sprite.frames[0].rgba.clone(),
            mipmaps: vec![vec![0, 0, 0, 255]],
            w: 2,
            h: 2,
        };
        let mut animator = AtlasAnimator::new(&atlas, [&sprite]);

        // A missing world/player snapshot is intentionally inert; the live caller must opt into
        // the context path rather than silently cycling the compass.
        assert!(animator.tick(&mut atlas).is_empty());
        assert_eq!(atlas.texture[0], 0);

        // With yaw zero, this retained spawn eases toward a negative angle and selects frame 15
        // on the first update. A stale `(0, 0)` fallback points the other way and selects frame 0.
        let uploads = animator.tick_with_context(
            &mut atlas,
            AtlasAnimationContext {
                clock: AtlasClockInput::default(),
                compass: Some(AtlasCompassInput {
                    player_x: -1.0,
                    player_z: 2.0,
                    yaw_degrees: 0.0,
                    spawn_x: -11,
                    spawn_z: -8,
                    is_surface_world: true,
                    random: 0.0,
                }),
            },
        );
        assert_eq!(uploads[0].rgba[0], 15);
        assert_eq!(atlas.texture[0], 15);
    }

    #[test]
    fn compass_nether_context_uses_random_spin_and_no_world_context_is_inert() {
        let sprite = special_sprite("compass", 16);
        let mut atlas = Atlas {
            texture: sprite.frames[0].rgba.clone(),
            mipmaps: vec![vec![0, 0, 0, 255]],
            w: 2,
            h: 2,
        };
        let mut animator = AtlasAnimator::new(&atlas, [&sprite]);

        // No WorldClient/thePlayer means TextureCompass returns before choosing a frame.
        assert!(animator.tick(&mut atlas).is_empty());
        assert_eq!(atlas.texture[0], 0);

        // In a Nether-like world the spawn coordinates are ignored and Math.random() drives the
        // target. random=.75 is 3PI/2, which wraps to a small negative turn and frame 15.
        let uploads = animator.tick_with_context(
            &mut atlas,
            AtlasAnimationContext {
                clock: AtlasClockInput::default(),
                compass: Some(AtlasCompassInput {
                    player_x: 100.0,
                    player_z: -50.0,
                    yaw_degrees: 173.0,
                    spawn_x: -999,
                    spawn_z: 777,
                    is_surface_world: false,
                    random: 0.75,
                }),
            },
        );
        assert_eq!(uploads[0].rgba[0], 15);
        assert_eq!(atlas.texture[0], 15);
    }

    #[test]
    fn dynamic_context_drives_clock_but_plain_tick_does_not_blindly_cycle_it() {
        let sprite = special_sprite("clock", 16);
        let mut atlas = Atlas {
            texture: sprite.frames[0].rgba.clone(),
            mipmaps: vec![vec![0, 0, 0, 255]],
            w: 2,
            h: 2,
        };
        let mut animator = AtlasAnimator::new(&atlas, [&sprite]);

        assert!(
            animator.tick(&mut atlas).is_empty(),
            "special sprites need world context"
        );
        let uploads = animator.tick_with_context(
            &mut atlas,
            AtlasAnimationContext {
                clock: AtlasClockInput {
                    world_time: Some(0),
                    is_surface_world: true,
                    random: 0.0,
                },
                compass: None,
            },
        );
        assert_eq!(uploads.len(), 2);
        assert_eq!(
            uploads[0].rgba[0], 15,
            "sunrise's wrapped first step selects frame 15"
        );
    }

    /// Apply records in order to a GPU-like mirror of every atlas level.
    fn apply_uploads(levels: &mut [Vec<u8>], base_width: u32, uploads: &[AtlasAnimationUpload]) {
        for upload in uploads {
            let level = upload.mip_level as usize;
            blit(
                &mut levels[level],
                base_width >> upload.mip_level,
                &upload.rgba,
                upload.width,
                upload.height,
                upload.x,
                upload.y,
                false,
            );
        }
    }

    fn atlas_levels(atlas: &Atlas) -> Vec<Vec<u8>> {
        std::iter::once(atlas.texture.clone())
            .chain(atlas.mipmaps.iter().cloned())
            .collect()
    }

    fn patterned_frame(seed: u32, duration_ticks: Option<u32>) -> crate::sprite::SpriteFrame {
        crate::sprite::SpriteFrame {
            rgba: (0..4 * 4 * 4)
                .map(|index| (seed.wrapping_mul(97).wrapping_add(index * 7) % 256) as u8)
                .collect(),
            duration_ticks,
        }
    }

    fn patterned_sprite(
        name: &str,
        x: u32,
        frames: Vec<crate::sprite::SpriteFrame>,
        interpolate: bool,
    ) -> Sprite {
        Sprite {
            location: ResourceLocation::minecraft(name),
            width: 4,
            height: 4,
            uv: uv_for_placement(x, 0, 4, 4, 8, 4),
            frames,
            default_duration_ticks: 3,
            interpolate,
            texture_metadata: Default::default(),
        }
    }

    /// A replay checkpoint keeps a copy of the animator; restoring it
    /// over an atlas that animated further must put back exactly the texels the copy's moment
    /// showed, on every level, for stepped and interpolated sprites, wherever in a frame the
    /// copy was taken. The records it returns must bring a GPU mirror to the same texels.
    #[test]
    fn writing_the_current_frames_restores_a_copied_animators_moment() {
        let stepped = patterned_sprite(
            "textures/blocks/stepped",
            0,
            vec![
                patterned_frame(1, Some(1)),
                patterned_frame(2, Some(2)),
                patterned_frame(3, Some(3)),
            ],
            false,
        );
        let blended = patterned_sprite(
            "textures/blocks/blended",
            4,
            vec![
                patterned_frame(4, None),
                patterned_frame(5, None),
                patterned_frame(6, Some(1)),
            ],
            true,
        );
        let mut atlas = Atlas {
            texture: vec![0; 8 * 4 * 4],
            mipmaps: vec![vec![0; 4 * 2 * 4], vec![0; 2 * 4]],
            w: 8,
            h: 4,
        };
        let mut animator = AtlasAnimator::new(&atlas, [&stepped, &blended]);
        let mut checked = 0;
        // This fixture's atlas starts blank rather than stitched, so the first copy is taken after
        // one tick has drawn both sprites.
        for (copy_after, run_on) in [(1, 5), (1, 3), (2, 7), (4, 1), (5, 9), (7, 2), (11, 4)] {
            for _ in 0..copy_after {
                animator.tick(&mut atlas);
            }
            let copy = animator.clone();
            let moment = atlas_levels(&atlas);
            for _ in 0..run_on {
                animator.tick(&mut atlas);
            }
            let mut mirror = atlas_levels(&atlas);
            if mirror != moment {
                checked += 1;
            }
            let uploads = copy.write_current_frames(&mut atlas);
            assert_eq!(
                atlas_levels(&atlas),
                moment,
                "copied after {copy_after} more ticks"
            );
            apply_uploads(&mut mirror, atlas.w, &uploads);
            assert_eq!(mirror, moment, "the records bring a GPU mirror back too");
            animator = copy;
        }
        assert!(checked >= 4, "the fixture must restore over changed texels");
    }

    /// ATLAS-2: frames that follow 1..=10 catch-up ticks must flush exactly the texels that the
    /// uncoalesced per-tick fan-out would have left behind, for stepped and interpolated sprites
    /// across every mip level, and never more than one record per sprite level.
    #[test]
    fn coalesced_catch_up_flushes_leave_bit_identical_levels() {
        let stepped = patterned_sprite(
            "textures/blocks/stepped",
            0,
            vec![
                patterned_frame(1, Some(1)),
                patterned_frame(2, Some(2)),
                patterned_frame(3, Some(3)),
            ],
            false,
        );
        let blended = patterned_sprite(
            "textures/blocks/blended",
            4,
            vec![
                patterned_frame(4, None),
                patterned_frame(5, None),
                patterned_frame(6, Some(1)),
            ],
            true,
        );
        let mut atlas = Atlas {
            texture: vec![0; 8 * 4 * 4],
            mipmaps: vec![vec![0; 4 * 2 * 4], vec![0; 2 * 4]],
            w: 8,
            h: 4,
        };
        let mut animator = AtlasAnimator::new(&atlas, [&stepped, &blended]);
        assert_eq!(animator.len(), 2);
        let mut every_tick = atlas_levels(&atlas);
        let mut coalesced = every_tick.clone();
        let mut queue = Vec::new();
        let mut uncoalesced_records = 0;
        for ticks in [1, 3, 10, 2, 7, 1, 1, 5, 10, 4, 9, 6, 8, 1, 10] {
            for _ in 0..ticks {
                let uploads = animator.tick(&mut atlas);
                uncoalesced_records += uploads.len();
                apply_uploads(&mut every_tick, atlas.w, &uploads);
                coalesce_animation_uploads(&mut queue, uploads);
            }
            // Two sprites times three levels bound any flush, however many ticks it follows.
            assert!(
                queue.len() <= 6,
                "{} records after {ticks} ticks",
                queue.len()
            );
            apply_uploads(&mut coalesced, atlas.w, &queue);
            queue.clear();
            assert_eq!(coalesced, every_tick, "after a {ticks}-tick frame");
            assert_eq!(
                coalesced,
                atlas_levels(&atlas),
                "the CPU atlas is the canonical state"
            );
        }
        assert!(
            uncoalesced_records > 6 * 15,
            "the fixture must exercise real catch-up"
        );
    }

    /// The exactness argument does not depend on disjoint sprites: even partially overlapping
    /// rectangles, mixed levels and repeats within one tick flush to the same texels.
    #[test]
    fn coalescing_is_exact_for_arbitrary_overlapping_record_streams() {
        let mut state = 0x2545_f491_u32;
        let mut next = move |bound: u32| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 8) % bound
        };
        let rects = [
            (0, 0, 0, 4, 4),
            (0, 2, 1, 4, 2),
            (0, 4, 0, 4, 4),
            (1, 0, 0, 2, 2),
            (1, 1, 0, 3, 1),
        ];
        for _ in 0..200 {
            let mut every_tick = vec![vec![0_u8; 8 * 4 * 4], vec![0_u8; 4 * 2 * 4]];
            let mut coalesced = every_tick.clone();
            let mut queue = Vec::new();
            for _ in 0..1 + next(10) {
                let tick: Vec<_> = (0..next(6))
                    .map(|_| {
                        let (mip_level, x, y, width, height) =
                            rects[next(rects.len() as u32) as usize];
                        let value = next(256) as u8;
                        AtlasAnimationUpload {
                            mip_level,
                            x,
                            y,
                            width,
                            height,
                            rgba: vec![value; (width * height * 4) as usize],
                        }
                    })
                    .collect();
                apply_uploads(&mut every_tick, 8, &tick);
                coalesce_animation_uploads(&mut queue, tick);
            }
            apply_uploads(&mut coalesced, 8, &queue);
            assert_eq!(coalesced, every_tick);
        }
    }
}
