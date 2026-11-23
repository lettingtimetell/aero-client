//! Decoded textures: a sprite's atlas UV rect, its animation frames, and vanilla's
//! alpha-aware mipmap blend.
//!
// java: TextureAtlasSprite.java, AnimationMetadataSection.java, TextureUtil.java
//!
//! A [`Sprite`] holds every animation frame's raw RGBA pixels (not just the one baked into the
//! atlas) plus per-frame timing, because vanilla animates a texture by periodically
//! *re-uploading* a sub-rectangle of the atlas rather than by shifting UVs — S11/S12 need the
//! frame data to do the same. [`Atlas`] itself initially holds sheet frame zero.

use serde_json::Value;

use crate::{AssetError, ResourceLocation, ResourceManager};

/// A sprite's rectangle within the atlas, in normalised `0..1` UV space, plus the
/// `0..16`-texel-space interpolation vanilla's model baking targets.
// java: TextureAtlasSprite's minU/maxU/minV/maxV + getInterpolatedU/V
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvRect {
    /// Minimum U.
    pub min_u: f32,
    /// Maximum U.
    pub max_u: f32,
    /// Minimum V.
    pub min_v: f32,
    /// Maximum V.
    pub max_v: f32,
}

impl UvRect {
    /// The whole atlas / an unstitched standalone texture.
    pub const FULL: UvRect = UvRect {
        min_u: 0.0,
        max_u: 1.0,
        min_v: 0.0,
        max_v: 1.0,
    };

    /// Maps a `0..16` texel-space U (as `elements[].faces.*.uv` uses) into atlas UV space.
    /// java: `getInterpolatedU`
    #[must_use]
    pub fn interpolate_u(&self, u: f32) -> f32 {
        self.min_u + (self.max_u - self.min_u) * (u / 16.0)
    }

    /// java: `getInterpolatedV`
    #[must_use]
    pub fn interpolate_v(&self, v: f32) -> f32 {
        self.min_v + (self.max_v - self.min_v) * (v / 16.0)
    }
}

/// One animation frame: RGBA8 pixels at the sprite's native resolution, plus how many client
/// ticks it displays for.
#[derive(Debug, Clone, PartialEq)]
pub struct SpriteFrame {
    /// Tightly packed RGBA8, `width * height * 4` bytes.
    pub rgba: Vec<u8>,
    /// Ticks this frame displays for. `None` means "use the animation's default frametime".
    pub duration_ticks: Option<u32>,
}

/// A decoded, atlas-placed texture.
#[derive(Debug, Clone, PartialEq)]
pub struct Sprite {
    /// The location it was loaded from (`textures/blocks/...`, without `.png`).
    pub location: ResourceLocation,
    /// Native width in pixels (one frame).
    pub width: u32,
    /// Native height in pixels (one frame).
    pub height: u32,
    /// This sprite's rectangle within the atlas.
    pub uv: UvRect,
    /// Animation frames in playback order. Explicit metadata can make `frames[0]` differ from
    /// sheet frame zero; [`crate::Atlas`] nevertheless starts with sheet frame zero like vanilla.
    /// Non-animated textures have exactly one frame.
    pub frames: Vec<SpriteFrame>,
    /// Default per-frame duration in ticks, from `.mcmeta`'s `frametime` (vanilla default 1).
    pub default_duration_ticks: u32,
    /// Whether consumers should cross-fade between frames rather than cutting.
    pub interpolate: bool,
    /// Texture sampler and explicit-mipmap options from `<location>.png.mcmeta`.
    ///
    /// Java carries these through `TextureMetadataSection` even for atlas sprites.  The atlas
    /// bake/cache retains them so a renderer can choose the same sampler after a cold bake or a
    /// cache hit; animation metadata remains represented by the fields above.
    pub texture_metadata: TextureMetadata,
}

/// The `texture` section of a PNG sidecar (`TextureMetadataSection`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TextureMetadata {
    /// `SimpleTexture` selects linear magnification/minification when true.
    pub blur: bool,
    /// `SimpleTexture` selects `CLAMP_TO_EDGE` wrapping when true.
    pub clamp: bool,
    /// Explicit atlas mip levels requested by the sidecar. Values outside the usable range are
    /// retained because Java parses them and ignores them at the load boundary.
    pub mipmaps: Vec<i32>,
    /// Explicit mip PNG pixels, grouped by playback frame and keyed by mip level.
    ///
    /// Java's `TextureAtlasSprite` keeps these arrays alongside each frame's base pixels.  The
    /// public `SpriteFrame` shape predates explicit mips, so the bytes live here instead; an empty
    /// outer vector means that no explicit mip image was loaded.  Levels not present in the pack
    /// are absent and are generated from the preceding level just like vanilla.
    pub explicit_mipmaps: Vec<Vec<(u32, Vec<u8>)>>,
}

/// Apply vanilla's `TextureUtil.anaglyphColor` transform to one RGBA8 pixel.
///
/// Java performs the arithmetic on the integer colour channels and truncates each weighted sum
/// with integer division by 100.  The alpha byte is not part of the transform.  Keeping this
/// helper in the asset crate gives every upload owner the same byte oracle instead of allowing a
/// float implementation (or a subtly different rounding rule) to drift between atlas, entity,
/// and UI paths.
///
/// java: `TextureUtil.anaglyphColor`
#[must_use]
pub fn anaglyph_color(pixel: [u8; 4]) -> [u8; 4] {
    let r = i32::from(pixel[0]);
    let g = i32::from(pixel[1]);
    let b = i32::from(pixel[2]);
    let luminance = (r * 30 + g * 59 + b * 11) / 100;
    let red = (r * 30 + g * 70) / 100;
    let blue = (r * 30 + b * 70) / 100;
    [red as u8, luminance as u8, blue as u8, pixel[3]]
}

/// Convert an RGBA8 byte buffer in place when Java's anaglyph setting is enabled.
///
/// Upload callers own their source buffers, so they should clone the bytes before calling this
/// helper when the same source will be used by a later non-anaglyph upload.  A disabled call is a
/// strict no-op, preserving the existing byte-for-byte path.  Texture upload boundaries validate
/// their dimensions before reaching this helper; the assertion keeps malformed internal buffers
/// from silently leaving a trailing partial pixel untouched.
pub fn apply_anaglyph_rgba(rgba: &mut [u8], enabled: bool) {
    if !enabled {
        return;
    }
    assert!(
        rgba.len().is_multiple_of(4),
        "anaglyph conversion requires RGBA8 pixels, got {} bytes",
        rgba.len()
    );
    for pixel in rgba.chunks_exact_mut(4) {
        let converted = anaglyph_color([pixel[0], pixel[1], pixel[2], pixel[3]]);
        pixel.copy_from_slice(&converted);
    }
}

/// Return an upload-owned RGBA8 buffer with Java's optional anaglyph transform applied.
///
/// This is intentionally an allocation boundary: upload owners must never mutate decoded pack
/// bytes because the same source can feed another GPU copy or a resource reload.  With the option
/// disabled the returned bytes are an exact clone of `rgba`; with it enabled each pixel is passed
/// through [`anaglyph_color`] once.
#[must_use]
pub fn anaglyph_rgba(rgba: &[u8], enabled: bool) -> Vec<u8> {
    let mut upload = rgba.to_vec();
    apply_anaglyph_rgba(&mut upload, enabled);
    upload
}

/// Parse the optional `texture` section in a resource's `.mcmeta` bytes.
///
/// `None` means the Java metadata lookup returned no section and therefore yields defaults. The
/// coercions here intentionally mirror 1.8.9's Gson calls: `JsonUtils.getBoolean` accepts any
/// primitive, while the mipmap array calls `JsonElement.getAsInt` directly (including its
/// decimal-number truncation and numeric-string path).
pub fn parse_texture_metadata(
    location: &ResourceLocation,
    metadata: Option<&[u8]>,
) -> Result<TextureMetadata, AssetError> {
    let Some(bytes) = metadata else {
        return Ok(TextureMetadata::default());
    };
    let root: Value = serde_json::from_slice(bytes)
        .map_err(|e| AssetError::json(format!("{location}.png.mcmeta"), e))?;
    let Some(root) = root.as_object() else {
        return Err(AssetError::BadTextureMeta {
            location: location.to_string(),
            reason: "metadata root is not an object".to_owned(),
        });
    };
    let Some(section) = root.get("texture") else {
        return Ok(TextureMetadata::default());
    };
    let Some(section) = section.as_object() else {
        return Err(AssetError::BadTextureMeta {
            location: location.to_string(),
            reason: "texture section is not an object".to_owned(),
        });
    };
    let bool_field = |name: &str| -> Result<bool, AssetError> {
        match section.get(name) {
            None => Ok(false),
            Some(value) => gson_primitive_bool(value).ok_or_else(|| AssetError::BadTextureMeta {
                location: location.to_string(),
                reason: format!("texture.{name} is not a boolean primitive"),
            }),
        }
    };
    let mipmaps = match section.get("mipmaps") {
        None => Vec::new(),
        Some(value) => {
            let Some(values) = value.as_array() else {
                return Err(AssetError::BadTextureMeta {
                    location: location.to_string(),
                    reason: "texture.mipmaps is not an array".to_owned(),
                });
            };
            let mut mipmaps = Vec::new();
            for (index, value) in values.iter().enumerate() {
                // TextureMetadataSectionSerializer has no final `else`: JSON nulls and nested
                // arrays are silently skipped. Objects are rejected explicitly, while every
                // primitive reaches getAsInt (and therefore errors if it cannot become an int).
                if value.is_null() || value.is_array() {
                    continue;
                }
                if value.is_object() {
                    return Err(AssetError::BadTextureMeta {
                        location: location.to_string(),
                        reason: format!("texture.mipmaps[{index}] is an object, not a number"),
                    });
                }
                let mipmap =
                    gson_texture_mipmap_int(value).ok_or_else(|| AssetError::BadTextureMeta {
                        location: location.to_string(),
                        reason: format!("texture.mipmaps[{index}] is not an integer primitive"),
                    })?;
                mipmaps.push(mipmap);
            }
            mipmaps
        }
    };
    Ok(TextureMetadata {
        blur: bool_field("blur")?,
        clamp: bool_field("clamp")?,
        mipmaps,
        explicit_mipmaps: Vec::new(),
    })
}

/// Gson's `JsonPrimitive#getAsBoolean`: booleans retain their value, while every other
/// primitive is converted to text and passed through `Boolean.parseBoolean`.
fn gson_primitive_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        Value::String(value) => Some(value.eq_ignore_ascii_case("true")),
        Value::Number(_) => Some(false),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

/// The mipmap serializer calls `JsonElement.getAsInt()` directly. JSON numbers use Gson's
/// `LazilyParsedNumber.intValue` (decimal truncation plus Java's low-32-bit narrowing), while
/// JSON strings use `Integer.parseInt` and therefore accept only ordinary signed integers.
fn gson_texture_mipmap_int(value: &Value) -> Option<i32> {
    match value {
        Value::Number(value) => parse_java_int_text(&value.to_string()),
        Value::String(value) => value.parse::<i32>().ok(),
        Value::Bool(_) | Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

/// The low 32 bits of a decimal integer after truncation toward zero. This is the observable
/// contract of Gson 2.2.4's `LazilyParsedNumber.intValue` for JSON numeric primitives.
fn parse_java_int_text(text: &str) -> Option<i32> {
    if let Ok(value) = text.parse::<i32>() {
        return Some(value);
    }
    if let Ok(value) = text.parse::<i64>() {
        return Some(value as i32);
    }
    if let Ok(value) = text.parse::<u64>() {
        return Some(value as i32);
    }

    let (negative, unsigned) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    if unsigned.is_empty() {
        return None;
    }
    let mut exponent_parts = unsigned.split(['e', 'E']);
    let mantissa = exponent_parts.next()?;
    let exponent = match exponent_parts.next() {
        Some(value) => value.parse::<i64>().ok()?,
        None => 0,
    };
    if exponent_parts.next().is_some() {
        return None;
    }

    let mut digits = Vec::with_capacity(mantissa.len());
    let mut decimal_at = None;
    for byte in mantissa.bytes() {
        match byte {
            b'0'..=b'9' => digits.push(byte - b'0'),
            b'.' if decimal_at.is_none() => decimal_at = Some(digits.len()),
            _ => return None,
        }
    }
    if digits.is_empty() {
        return None;
    }
    let decimal_at = decimal_at.unwrap_or(digits.len());
    let integer_len = i64::try_from(decimal_at).ok()?.checked_add(exponent)?;
    if integer_len <= 0 {
        return Some(0);
    }

    let retained = usize::try_from(integer_len)
        .ok()
        .map_or(digits.len(), |len| len.min(digits.len()));
    let mut low = 0_u32;
    for &digit in &digits[..retained] {
        low = low.wrapping_mul(10).wrapping_add(u32::from(digit));
    }
    let appended_zeros = integer_len.saturating_sub(i64::try_from(digits.len()).ok()?);
    low = low.wrapping_mul(pow10_wrapping(appended_zeros));
    Some(if negative {
        0_u32.wrapping_sub(low) as i32
    } else {
        low as i32
    })
}

fn pow10_wrapping(mut exponent: i64) -> u32 {
    let mut result = 1_u32;
    let mut base = 10_u32;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result.wrapping_mul(base);
        }
        base = base.wrapping_mul(base);
        exponent >>= 1;
    }
    result
}

impl Sprite {
    /// Generates `levels` additional mip levels for every frame, using vanilla's alpha-aware
    /// blend (see [`generate_mipmap_level`]). Returns `[frame0_mips, frame1_mips, ...]`, each
    /// inner `Vec` being `[level0, level1, ..., levelN]` with `level0` the original pixels.
    #[must_use]
    pub fn generate_mipmaps(&self, levels: u32) -> Vec<Vec<Vec<u8>>> {
        self.frames
            .iter()
            .enumerate()
            .map(|(frame, f)| {
                let explicit = self
                    .texture_metadata
                    .explicit_mipmaps
                    .get(frame)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                generate_mipmap_chain_with_overrides(
                    &f.rgba,
                    self.width,
                    self.height,
                    levels,
                    explicit,
                )
            })
            .collect()
    }
}

/// `assets/<ns>/<path>.png.mcmeta`'s `"animation"` section, already validated the way
/// `AnimationMetadataSectionSerializer` validates it.
// java: AnimationMetadataSectionSerializer
#[derive(Debug, Clone)]
struct AnimationSection {
    frametime: u32,
    width: Option<i32>,
    height: Option<i32>,
    frames: Vec<FrameEntry>,
    interpolate: bool,
}

/// One `frames` entry. java: `AnimationFrame`; a bare number's index is not range-checked
/// there (a negative one fails later, in `TextureAtlasSprite.loadSprite`).
#[derive(Debug, Clone, Copy)]
struct FrameEntry {
    index: i32,
    time: Option<i32>,
}

/// java: `JsonUtils.getInt(JsonElement, name)`: a JSON number only (a numeric string throws),
/// read with Gson 2.2.4's truncating, wrapping `LazilyParsedNumber.intValue()`.
fn mcmeta_int(value: &Value, field: &str) -> Result<i32, String> {
    match value {
        Value::Number(number) => crate::model::blockstate::parse_java_int_text(&number.to_string())
            .ok_or_else(|| format!("expected {field} to be an int, was {number}")),
        other => Err(format!("expected {field} to be an int, was {other}")),
    }
}

/// java: `JsonUtils.getInt(object, key, default)`.
fn mcmeta_int_or(
    object: &serde_json::Map<String, Value>,
    key: &str,
    default: i32,
) -> Result<i32, String> {
    object
        .get(key)
        .map_or(Ok(default), |value| mcmeta_int(value, key))
}

/// Reads the `.mcmeta` sidecar's `animation` section exactly as vanilla does: `None` when the
/// file has no `animation` member, `Err` wherever vanilla throws (the sprite is then skipped).
///
/// java: `IMetadataSerializer.parseMetadataSection` (a present section must be an object) and
/// `AnimationMetadataSectionSerializer.deserialize`/`parseAnimationFrame`. PA:
/// this replaced serde structs that rejected the `null`/array `frames` entries vanilla skips,
/// rejected decimal numbers vanilla truncates (`"frametime": 2.5` is 2), rejected a string
/// `interpolate` that `getBoolean` reads, and read a `null` time, width, height or section as
/// absent where vanilla throws.
fn parse_animation_metadata(bytes: &[u8]) -> Result<Option<AnimationSection>, String> {
    let root: Value = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    let Value::Object(root) = root else {
        return Err("metadata is not a JSON object".to_owned());
    };
    let Some(section) = root.get("animation") else {
        return Ok(None);
    };
    let Value::Object(section) = section else {
        return Err(format!(
            "invalid metadata for 'animation' - expected object, found {section}"
        ));
    };
    let frametime = mcmeta_int_or(section, "frametime", 1)?;
    if frametime != 1 && frametime < 1 {
        return Err("invalid default frame time".to_owned());
    }
    let mut frames = Vec::new();
    if let Some(list) = section.get("frames") {
        let Value::Array(list) = list else {
            return Err(format!("expected frames to be a JsonArray, was {list}"));
        };
        for (position, entry) in list.iter().enumerate() {
            match entry {
                Value::Object(frame) => {
                    let time = mcmeta_int_or(frame, "time", -1)?;
                    if frame.contains_key("time") && time < 1 {
                        return Err("invalid frame time".to_owned());
                    }
                    let index = frame
                        .get("index")
                        .ok_or_else(|| format!("missing frames[{position}].index"))
                        .and_then(|index| mcmeta_int(index, "index"))?;
                    if index < 0 {
                        return Err("invalid frame index".to_owned());
                    }
                    frames.push(FrameEntry {
                        index,
                        time: frame.contains_key("time").then_some(time),
                    });
                }
                // java: `parseAnimationFrame` returns null for anything that is neither a
                // primitive nor an object, and the caller skips it.
                Value::Null | Value::Array(_) => {}
                primitive => frames.push(FrameEntry {
                    index: mcmeta_int(primitive, &format!("frames[{position}]"))?,
                    time: None,
                }),
            }
        }
    }
    let width = mcmeta_int_or(section, "width", -1)?;
    let height = mcmeta_int_or(section, "height", -1)?;
    if width != -1 && width < 1 {
        return Err("invalid width".to_owned());
    }
    if height != -1 && height < 1 {
        return Err("invalid height".to_owned());
    }
    // java: `JsonUtils.getBoolean(object, "interpolate", false)`: any primitive, through
    // `JsonPrimitive.getAsBoolean` (`Boolean.parseBoolean` of its text unless it is a boolean).
    let interpolate = match section.get("interpolate") {
        None => false,
        Some(Value::Bool(value)) => *value,
        Some(Value::String(value)) => value.eq_ignore_ascii_case("true"),
        Some(Value::Number(_)) => false,
        Some(other) => return Err(format!("expected interpolate to be a Boolean, was {other}")),
    };
    Ok(Some(AnimationSection {
        frametime: frametime.unsigned_abs(),
        width: (width != -1).then_some(width),
        height: (height != -1).then_some(height),
        frames,
        interpolate,
    }))
}

/// Loads and decodes a texture (`<location>.png`, e.g. `minecraft:textures/blocks/lava_still`)
/// plus its optional `.mcmeta` animation sidecar, splitting a vertically-stacked animation
/// strip into individual frames.
///
/// # Errors
/// [`AssetError::NotFound`] if the PNG is missing, or a decode/shape error if the PNG or
/// `.mcmeta` is malformed. Never panics on malformed input.
pub fn load_sprite(
    rm: &ResourceManager,
    location: &ResourceLocation,
) -> Result<Sprite, AssetError> {
    load_sprite_with_atlas_frame(rm, location, crate::atlas::DEFAULT_MIPMAP_LEVELS)
        .map(|(sprite, _, _)| sprite)
}

/// Loads a sprite together with sheet frame zero, which is the frame vanilla initially
/// uploads into the atlas even when an explicit animation sequence starts somewhere else.
///
/// The public [`load_sprite`] API keeps exposing the animation sequence through
/// [`Sprite::frames`]. Atlas construction needs the separate sheet-frame-zero bytes because
/// those are distinct index spaces in `TextureAtlasSprite`.
pub(crate) fn load_sprite_with_atlas_frame(
    rm: &ResourceManager,
    location: &ResourceLocation,
    mipmap_levels: u32,
) -> Result<LoadedSprite, AssetError> {
    let png_loc = location.with_suffix(".png");
    let resource = rm.resource(&png_loc)?;

    let decoded = decode_sprite_with_info(
        location,
        &png_loc,
        &resource.data,
        resource.metadata.as_deref(),
    )?;
    let mut sprite = decoded.sprite;
    let mut texture_metadata = parse_texture_metadata(location, resource.metadata.as_deref())?;
    let (frame_mips, atlas_mips) = load_explicit_mipmaps(
        rm,
        location,
        decoded.source_width,
        decoded.source_height,
        &decoded.playback_indices,
        &texture_metadata.mipmaps,
        mipmap_levels,
    )?;
    texture_metadata.explicit_mipmaps = frame_mips;
    sprite.texture_metadata = texture_metadata;
    Ok((sprite, decoded.atlas_frame, atlas_mips))
}

/// The extra information needed while converting full-sheet explicit mip PNGs into the two
/// distinct index spaces vanilla exposes: sheet frame zero for the initial atlas upload, and the
/// metadata playback order for later animation uploads.
struct DecodedSprite {
    sprite: Sprite,
    atlas_frame: Vec<u8>,
    source_width: u32,
    source_height: u32,
    playback_indices: Vec<u32>,
}

/// A decoded sprite plus the sheet-frame-zero pixels and explicit atlas-level overrides.
pub(crate) type LoadedSprite = (Sprite, Vec<u8>, Vec<(u32, Vec<u8>)>);

type ExplicitMipmapData = (Vec<Vec<(u32, Vec<u8>)>>, Vec<(u32, Vec<u8>)>);

#[cfg(test)]
fn decode_sprite(
    location: &ResourceLocation,
    png_loc: &ResourceLocation,
    png: &[u8],
    metadata: Option<&[u8]>,
) -> Result<(Sprite, Vec<u8>), AssetError> {
    let decoded = decode_sprite_with_info(location, png_loc, png, metadata)?;
    Ok((decoded.sprite, decoded.atlas_frame))
}

/// How many times its own sheet an animation's per-entry frame copies may take (see
/// [`MAX_PLAYBACK_COPY_FLOOR`]).
const MAX_PLAYBACK_COPIES_PER_SHEET: usize = 8;
/// The frame copies an animation may always take, whatever its sheet: 64 MiB.
const MAX_PLAYBACK_COPY_FLOOR: usize = 64 << 20;

fn decode_sprite_with_info(
    location: &ResourceLocation,
    png_loc: &ResourceLocation,
    png: &[u8],
    metadata: Option<&[u8]>,
) -> Result<DecodedSprite, AssetError> {
    let img = image::load_from_memory_with_format(png, image::ImageFormat::Png)
        .map_err(|e| AssetError::Image {
            location: png_loc.to_string(),
            message: e.to_string(),
        })?
        .into_rgba8();
    let (width, height) = img.dimensions();
    let pixels = img.into_raw();

    let animation = match metadata {
        Some(bytes) => {
            parse_animation_metadata(bytes).map_err(|reason| AssetError::ModelResolve {
                location: ResourceLocation::new(
                    png_loc.namespace.clone(),
                    format!("{}.mcmeta", png_loc.path),
                ),
                reason,
            })?
        }
        None => None,
    };

    let Some(anim) = animation else {
        if width != height {
            return Err(invalid_animation(
                location,
                "broken aspect ratio and not an animation",
            ));
        }
        let atlas_frame = pixels.clone();
        return Ok(DecodedSprite {
            sprite: Sprite {
                location: location.clone(),
                width,
                height,
                uv: UvRect::FULL,
                frames: vec![SpriteFrame {
                    rgba: pixels,
                    duration_ticks: None,
                }],
                default_duration_ticks: 1,
                interpolate: false,
                texture_metadata: TextureMetadata::default(),
            },
            atlas_frame,
            source_width: width,
            source_height: height,
            playback_indices: vec![0],
        });
    };

    if anim.frametime == 0 {
        return Err(invalid_animation(location, "invalid default frame time"));
    }
    if anim.width.is_some_and(|width| width < 1) {
        return Err(invalid_animation(location, "invalid animation width"));
    }
    if anim.height.is_some_and(|height| height < 1) {
        return Err(invalid_animation(location, "invalid animation height"));
    }

    let frame_side = width; // frames stack vertically; each frame is width x width
    let frame_count_in_sheet = height.checked_div(frame_side).unwrap_or(0);
    // A sheet shorter than one frame holds no frame (java: `j / i` is 0,
    // so `loadSprite` stores none and a listed frame is an "invalid frameindex"). It used to get
    // a zero-filled `width x width` frame, twice: a 1,000,000x1 PNG of a few hundred bytes asked
    // for 4 TB and the allocation failure aborted the process on every start. Now the sprite is
    // skipped like any broken animation, before anything is allocated.
    if frame_count_in_sheet == 0 {
        return Err(invalid_animation(
            location,
            "the sheet is shorter than one frame",
        ));
    }
    let stride = (frame_side * 4) as usize;
    let frame_bytes = (frame_side as usize) * stride;
    // Each playback entry below gets its own copy of its
    // frame's pixels (java keeps each sheet frame once and plays the list by index), so the
    // `frames` list multiplies memory: `[0,0,0,...]` in a 1 MiB `.mcmeta` asked for gigabytes.
    // The copies may take up to MAX_PLAYBACK_COPIES_PER_SHEET times the sheet itself, or
    // MAX_PLAYBACK_COPY_FLOOR (65,536 entries of a 16x16 frame), whichever is more; past that
    // the animation is refused like any other broken one.
    let playback_entries = if anim.frames.is_empty() {
        frame_count_in_sheet as usize
    } else {
        anim.frames.len()
    };
    let copied = playback_entries.saturating_mul(frame_bytes);
    let allowed = pixels
        .len()
        .saturating_mul(MAX_PLAYBACK_COPIES_PER_SHEET)
        .max(MAX_PLAYBACK_COPY_FLOOR);
    if copied > allowed {
        return Err(invalid_animation(
            location,
            "the animation lists too many frames",
        ));
    }
    let atlas_frame = if frame_bytes <= pixels.len() {
        pixels[..frame_bytes].to_vec()
    } else {
        vec![0u8; frame_bytes]
    };

    let slice_frame = |index: u32| -> Vec<u8> {
        let start = index as usize * frame_bytes;
        let end = start + frame_bytes;
        if end <= pixels.len() {
            pixels[start..end].to_vec()
        } else {
            vec![0u8; frame_bytes]
        }
    };

    let mut frames = Vec::new();
    let mut playback_indices = Vec::new();
    if anim.frames.is_empty() {
        for i in 0..frame_count_in_sheet {
            frames.push(SpriteFrame {
                rgba: slice_frame(i),
                duration_ticks: None,
            });
            playback_indices.push(i);
        }
    } else {
        for &FrameEntry { index, time } in &anim.frames {
            // java: `loadSprite` throws for `index >= frameCount`, and `List.set` for a negative
            // bare-number index; either way the sprite is skipped.
            let Some(index) = u32::try_from(index)
                .ok()
                .filter(|&index| index < frame_count_in_sheet)
            else {
                return Err(invalid_animation(location, "invalid animation frame index"));
            };
            if time.is_some_and(|time| time < 1) {
                return Err(invalid_animation(location, "invalid animation frame time"));
            }
            frames.push(SpriteFrame {
                rgba: slice_frame(index),
                duration_ticks: time.map(|time| time as u32),
            });
            playback_indices.push(index);
        }
    }
    if frames.is_empty() {
        frames.push(SpriteFrame {
            rgba: slice_frame(0),
            duration_ticks: None,
        });
        playback_indices.push(0);
    }

    Ok(DecodedSprite {
        sprite: Sprite {
            location: location.clone(),
            width: frame_side,
            height: frame_side,
            uv: UvRect::FULL,
            frames,
            default_duration_ticks: anim.frametime,
            interpolate: anim.interpolate,
            texture_metadata: TextureMetadata::default(),
        },
        atlas_frame,
        source_width: width,
        source_height: height,
        playback_indices,
    })
}

/// Load the listed `<basePath>/mipmaps/<name>.<level>.png` files and split each image into the
/// sheet frame zero plus the animation's playback frames.  A missing or undecodable override is
/// ignored (vanilla logs the IOException and lets `generateMipmaps` fill it); a present image
/// with the wrong dimensions rejects the whole sprite, matching `TextureAtlasSprite.loadSprite`.
fn load_explicit_mipmaps(
    rm: &ResourceManager,
    location: &ResourceLocation,
    source_width: u32,
    source_height: u32,
    playback_indices: &[u32],
    listed_levels: &[i32],
    mipmap_levels: u32,
) -> Result<ExplicitMipmapData, AssetError> {
    if listed_levels.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    if !source_width.is_power_of_two() || !source_height.is_power_of_two() {
        return Err(AssetError::ModelResolve {
            location: location.clone(),
            reason: "Unable to load extra miplevels, source-texture is not power of two".to_owned(),
        });
    }

    let mut per_frame = vec![Vec::new(); playback_indices.len()];
    let mut sheet_frame_zero = Vec::new();
    let mut loaded_levels = std::collections::HashSet::new();
    for &listed in listed_levels {
        let Ok(level) = u32::try_from(listed) else {
            continue;
        };
        // java: `TextureMap.loadTextureAtlas` sizes its image array `1 + mipmapLevels` from the
        // configured setting and loads level `i2` only while `i2 < abufferedimage.length - 1`,
        // i.e. below `mipmapLevels`: levels 1-3 at the default 4 (level 4 stays generated), none
        // at 0 or 1 (phase 4d review S1). Levels past the atlas's final count are ignored by the
        // mip chain, as vanilla's `generateMipmaps` ignores them.
        if level == 0 || level >= mipmap_levels || !loaded_levels.insert(level) {
            continue;
        }
        let mip_location = explicit_mipmap_location(location, level);
        let resource = match rm.resource(&mip_location) {
            Ok(resource) => resource,
            Err(AssetError::NotFound(_)) => continue,
            Err(error) => {
                tracing::warn!(
                    location = %mip_location,
                    error = %error,
                    "explicit mipmap could not be read; generating it"
                );
                continue;
            }
        };
        let image =
            match image::load_from_memory_with_format(&resource.data, image::ImageFormat::Png) {
                Ok(image) => image.into_rgba8(),
                Err(error) => {
                    tracing::warn!(
                        location = %mip_location,
                        error = %error,
                        "explicit mipmap could not be decoded; generating it"
                    );
                    continue;
                }
            };
        let (width, height) = image.dimensions();
        let expected_width = source_width >> level;
        let expected_height = source_height >> level;
        if width != expected_width || height != expected_height {
            return Err(AssetError::ModelResolve {
                location: location.clone(),
                reason: format!(
                    "Unable to load miplevel: {level}, image is size: {width}x{height}, expected {expected_width}x{expected_height}"
                ),
            });
        }
        let pixels = image.into_raw();
        let frame_width = source_width >> level;
        let frame_height = frame_width;
        let frame_stride = (frame_width * frame_height * 4) as usize;
        let sheet_frame = pixels[..frame_stride].to_vec();
        sheet_frame_zero.push((level, sheet_frame));
        for (playback_frame, &sheet_index) in playback_indices.iter().enumerate() {
            let start = (sheet_index as usize)
                .saturating_mul(frame_stride)
                .min(pixels.len());
            let end = start.saturating_add(frame_stride);
            if end > pixels.len() {
                // The base image's frame count/index checks should make this unreachable, but a
                // checked fallback keeps malformed hostile data from panicking here.
                return Err(AssetError::ModelResolve {
                    location: location.clone(),
                    reason: format!("Unable to load miplevel: {level}, frame data is truncated"),
                });
            }
            per_frame[playback_frame].push((level, pixels[start..end].to_vec()));
        }
    }
    if sheet_frame_zero.is_empty() {
        Ok((Vec::new(), Vec::new()))
    } else {
        // Java stores these in an image array keyed by level, so metadata order is not
        // observable. Canonicalising here keeps the baked-cache reader's ascending-level
        // invariant true even for a pack that lists `[2, 1]`.
        for frame in &mut per_frame {
            frame.sort_by_key(|(level, _)| *level);
        }
        sheet_frame_zero.sort_by_key(|(level, _)| *level);
        Ok((per_frame, sheet_frame_zero))
    }
}

fn explicit_mipmap_location(location: &ResourceLocation, level: u32) -> ResourceLocation {
    let path = location
        .path
        .strip_prefix("textures/")
        .unwrap_or(location.path.as_str());
    ResourceLocation::new(
        location.namespace.clone(),
        format!("textures/mipmaps/{path}.{level}.png"),
    )
}

fn invalid_animation(location: &ResourceLocation, reason: &str) -> AssetError {
    AssetError::ModelResolve {
        location: location.clone(),
        reason: reason.to_owned(),
    }
}

/// Vanilla's alpha-aware 2x2 box-filter mip blend (java: `TextureUtil.generateMipmapData` /
/// `blendColors` / `blendColorComponent`).
///
/// Ordinary box filtering of premultiplied-nowhere RGBA produces a visible dark fringe around
/// cutout edges (leaves, glass panes) because fully-transparent texels still contribute their
/// (usually black) RGB to the average. Vanilla avoids this two ways, chosen once per sprite
/// from whether *any* texel anywhere in the base image is fully transparent:
/// - **no transparency anywhere**: gamma-correct (2.2) average of all four texels, every
///   channel including alpha treated identically.
/// - **some transparency somewhere**: still averages all four texels' gamma-corrected
///   channels unconditionally (dividing by 4, not by the transparent count — this is
///   deliberately not a "skip transparent neighbours" average), but then **snaps the result
///   fully transparent** if the blended alpha comes out below `96/255`, which is what removes
///   the fringe: a mip texel surrounded mostly by transparent neighbours disappears instead of
///   going dark-and-translucent.
#[must_use]
pub fn generate_mipmap_level(rgba: &[u8], width: u32, has_transparency: bool) -> Vec<u8> {
    debug_assert!(width >= 2 && width.is_multiple_of(2));
    let w = width as usize;
    let out_w = w / 2;
    let mut out = vec![0u8; out_w * out_w * 4];

    for row in 0..out_w {
        for col in 0..out_w {
            let idx = |dx: usize, dy: usize| -> [u8; 4] {
                let x = col * 2 + dx;
                let y = row * 2 + dy;
                let o = (y * w + x) * 4;
                [rgba[o], rgba[o + 1], rgba[o + 2], rgba[o + 3]]
            };
            let texels = [idx(0, 0), idx(1, 0), idx(0, 1), idx(1, 1)];
            let blended = blend_texels(texels, has_transparency);
            let o = (row * out_w + col) * 4;
            out[o..o + 4].copy_from_slice(&blended);
        }
    }
    out
}

fn gamma_decode(c: u8) -> f32 {
    // ATLAS-7: a `u8` input has 256 possible results, so mip generation (bake and every
    // animator rebuild, about 0.8M calls for vanilla) reads a table built once from the exact
    // expression below instead of calling `powf` per channel per texel. Bit-identical.
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|index| gamma_decode_exact(index as u8)))
        [usize::from(c)]
}

fn gamma_decode_exact(c: u8) -> f32 {
    // Java first performs the `/ 255.0F` in float, widens that result for Math.pow, then
    // narrows the decoded channel back to float.
    f64::from(f32::from(c) / 255.0).powf(2.2) as f32
}

fn gamma_encode_plain_average(sum: f32) -> u8 {
    // TextureUtil.blendColorComponent widens the float sum before multiplying by 0.25,
    // evaluates pow in double, narrows to float, then truncates the final double product.
    let encoded = (f64::from(sum) * 0.25).powf(0.454_545_454_545_454_53) as f32;
    (f64::from(encoded) * 255.0) as u8
}

fn gamma_encode_alpha_aware_sum(sum: f32) -> u8 {
    // TextureUtil.blendColors divides in float but does not narrow the Math.pow result before
    // multiplying by 255. Both Java casts truncate rather than round.
    (f64::from(sum / 4.0).powf(0.454_545_454_545_454_53) * 255.0) as u8
}

fn blend_texels(texels: [[u8; 4]; 4], has_transparency: bool) -> [u8; 4] {
    if !has_transparency {
        // Every channel, including alpha, averaged independently and identically.
        let mut out = [0u8; 4];
        for c in 0..4 {
            let sum: f32 = texels.iter().map(|t| gamma_decode(t[c])).sum();
            out[c] = gamma_encode_plain_average(sum);
        }
        out
    } else {
        // A texel whose *own* alpha is 0 contributes nothing to any channel's sum — not even
        // its RGB — but the divisor stays a fixed 4 regardless of how many texels actually
        // contributed. That fixed divisor (not "average of the survivors") is what makes a
        // mostly-transparent 2x2 block blend to a low alpha instead of a full-strength one,
        // which is what lets the <96 snap below reliably catch it.
        let mut sums = [0.0f32; 4]; // A, R, G, B accumulation order, matching Java's shifts
        for t in &texels {
            if t[3] != 0 {
                sums[0] += gamma_decode(t[3]); // alpha
                sums[1] += gamma_decode(t[0]); // red
                sums[2] += gamma_decode(t[1]); // green
                sums[3] += gamma_decode(t[2]); // blue
            }
        }
        let a = gamma_encode_alpha_aware_sum(sums[0]);
        let r = gamma_encode_alpha_aware_sum(sums[1]);
        let g = gamma_encode_alpha_aware_sum(sums[2]);
        let b = gamma_encode_alpha_aware_sum(sums[3]);
        let a = if a < 96 { 0 } else { a };
        [r, g, b, a]
    }
}

/// Builds a full mip chain `[level0, level1, ..., levelN]` for one RGBA8 image, `level0` being
/// the unmodified input. Stops early (chain shorter than `levels + 1`) if the image can no
/// longer be halved.
#[must_use]
pub fn generate_mipmap_chain(rgba: &[u8], width: u32, height: u32, levels: u32) -> Vec<Vec<u8>> {
    generate_mipmap_chain_with_overrides(rgba, width, height, levels, &[])
}

/// As [`generate_mipmap_chain`], retaining any valid explicit PNG overrides keyed by level.
///
/// `TextureUtil.generateMipmapData` checks the source array for each level before generating it;
/// that means a supplied level also becomes the input to generation of the next missing level.
/// Keeping that order is what makes a blue explicit level one continue into a generated level two
/// instead of silently regenerating level one from a red base.
#[must_use]
pub(crate) fn generate_mipmap_chain_with_overrides(
    rgba: &[u8],
    width: u32,
    height: u32,
    levels: u32,
    overrides: &[(u32, Vec<u8>)],
) -> Vec<Vec<u8>> {
    debug_assert_eq!(width, height, "sprite frames baked here are always square");
    // Vanilla accidentally bounds this probe by the number of mip arrays rather than by the
    // number of base-level texels. Preserve that visible bug: with N requested mip levels only
    // the first N+1 base texels participate in the alpha-aware/plain blend decision.
    let has_transparency = rgba
        .chunks_exact(4)
        .take(levels as usize + 1)
        .any(|px| px[3] == 0);
    let mut chain = vec![rgba.to_vec()];
    let mut w = width;
    for level in 1..=levels {
        if w < 2 || !w.is_multiple_of(2) {
            break;
        }
        let expected = ((width >> level) * (height >> level) * 4) as usize;
        let next = overrides
            .iter()
            .find(|(override_level, pixels)| *override_level == level && pixels.len() == expected)
            .map(|(_, pixels)| pixels.clone())
            .unwrap_or_else(|| generate_mipmap_level(chain.last().unwrap(), w, has_transparency));
        chain.push(next);
        w /= 2;
    }
    chain
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// [`parse_animation_metadata`] reads exactly what the real 1.8.9
    /// `AnimationMetadataSectionSerializer` (through `IMetadataSerializer`) reads, case for case:
    /// the `animation` rows of `tests/fixtures/model_json_ref.tsv` come from
    /// `javaref/ModelJsonRef.java`. Among them: `null`/array `frames` entries are skipped,
    /// decimals truncate, a `null` time/width/section throws, a string `interpolate` is read.
    /// (A bare negative index parses there and fails later, in `loadSprite`, as it does here.)
    #[test]
    fn animation_metadata_matches_the_vanilla_serializer() {
        let fixture = include_str!("../tests/fixtures/model_json_ref.tsv");
        let mut checked = 0;
        for line in fixture
            .lines()
            .filter(|line| line.starts_with("animation\t"))
        {
            let fields: Vec<&str> = line.split('\t').collect();
            let (name, json, verdict, java) = (fields[1], fields[2], fields[3], fields[4]);
            let rust = parse_animation_metadata(json.as_bytes()).map(|section| match section {
                None => "none".to_owned(),
                Some(a) => format!(
                    "frametime={},width={},height={},interpolate={},frames={}",
                    a.frametime,
                    a.width.unwrap_or(-1),
                    a.height.unwrap_or(-1),
                    a.interpolate,
                    a.frames
                        .iter()
                        .map(|frame| format!("{}:{}", frame.index, frame.time.unwrap_or(-1)))
                        .collect::<Vec<_>>()
                        .join(" ")
                ),
            });
            match (verdict, rust) {
                ("ACCEPT", Ok(rust)) => assert_eq!(rust, java, "{name}"),
                ("REJECT", Err(_)) => {}
                (verdict, rust) => panic!("{name}: java {verdict} ({java}), rust {rust:?}"),
            }
            checked += 1;
        }
        assert!(checked >= 20, "the fixture lost animation cases: {checked}");
    }

    fn solid(color: [u8; 4], w: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * w * 4);
        for _ in 0..(w * w) {
            v.extend_from_slice(&color);
        }
        v
    }

    #[test]
    fn explicit_animation_sequence_does_not_choose_the_initial_atlas_frame() {
        let mut image = image::RgbaImage::new(2, 4);
        for y in 0..2 {
            for x in 0..2 {
                image.put_pixel(x, y, image::Rgba([10, 20, 30, 255]));
                image.put_pixel(x, y + 2, image::Rgba([200, 210, 220, 255]));
            }
        }
        let mut png = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let location = ResourceLocation::minecraft("textures/blocks/reordered");
        let png_location = location.with_suffix(".png");
        let metadata = br#"{"animation":{"frames":[1,0]}}"#;

        let (sprite, atlas_frame) =
            decode_sprite(&location, &png_location, png.get_ref(), Some(metadata)).unwrap();

        assert_eq!(&sprite.frames[0].rgba[..4], &[200, 210, 220, 255]);
        assert_eq!(&atlas_frame[..4], &[10, 20, 30, 255]);
    }

    #[test]
    fn malformed_animation_values_are_rejected_instead_of_clamped_or_skipped() {
        let mut image = image::RgbaImage::new(2, 4);
        for pixel in image.pixels_mut() {
            *pixel = image::Rgba([10, 20, 30, 255]);
        }
        let mut png = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let location = ResourceLocation::minecraft("textures/blocks/malformed");
        let png_location = location.with_suffix(".png");

        for metadata in [
            br#"{"animation":{"frametime":0}}"#.as_slice(),
            br#"{"animation":{"width":0}}"#.as_slice(),
            br#"{"animation":{"height":0}}"#.as_slice(),
            br#"{"animation":{"frames":[{"index":0,"time":0}]}}"#.as_slice(),
            br#"{"animation":{"frames":[2]}}"#.as_slice(),
        ] {
            assert!(
                decode_sprite(&location, &png_location, png.get_ref(), Some(metadata)).is_err()
            );
        }
    }

    /// An animated sheet shorter than one frame (height < width) holds
    /// no frame: refused before any frame is allocated. The 1,000,000x1 PNG is 4 MB decoded (the
    /// image crate allows it); a `width x width` frame for it would be 4 TB, which aborted the
    /// process. The control: the same width with one whole frame still decodes.
    #[test]
    fn an_animated_sheet_shorter_than_one_frame_is_refused_before_allocating() {
        let encode = |width: u32, height: u32| {
            let image = image::RgbaImage::from_pixel(width, height, image::Rgba([1, 2, 3, 255]));
            let mut png = Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(image)
                .write_to(&mut png, image::ImageFormat::Png)
                .unwrap();
            png.into_inner()
        };
        let location = ResourceLocation::minecraft("textures/blocks/stone");
        let png_location = location.with_suffix(".png");
        let short = encode(16, 8);
        assert!(decode_sprite(
            &location,
            &png_location,
            &short,
            Some(br#"{"animation":{}}"#)
        )
        .is_err());
        let wide = encode(1_000_000, 1);
        for metadata in [
            br#"{"animation":{}}"#.as_slice(),
            br#"{"animation":{"frames":[0]}}"#.as_slice(),
        ] {
            let result = decode_sprite(&location, &png_location, &wide, Some(metadata));
            assert!(result.is_err(), "a sheet with no whole frame is refused");
        }
        let one_frame = encode(16, 16);
        let (sprite, atlas_frame) = decode_sprite(
            &location,
            &png_location,
            &one_frame,
            Some(br#"{"animation":{}}"#),
        )
        .expect("one whole frame decodes");
        assert_eq!(
            (sprite.width, sprite.height, sprite.frames.len()),
            (16, 16, 1)
        );
        assert_eq!(atlas_frame.len(), 16 * 16 * 4);
    }

    /// ATLAS-7: the table is the exact expression for every input, bit for bit.
    #[test]
    fn gamma_decode_table_is_bit_identical_to_the_exact_pow() {
        for c in 0..=u8::MAX {
            assert_eq!(
                gamma_decode(c).to_bits(),
                gamma_decode_exact(c).to_bits(),
                "c={c}"
            );
        }
    }

    #[test]
    fn texture_metadata_preserves_sampler_flags_and_mipmap_levels() {
        let location = ResourceLocation::minecraft("textures/entity/test");
        let metadata = br#"{
            "texture": {"blur": true, "clamp": true, "mipmaps": [1, 3, -1]}
        }"#;
        assert_eq!(
            parse_texture_metadata(&location, Some(metadata)).unwrap(),
            TextureMetadata {
                blur: true,
                clamp: true,
                mipmaps: vec![1, 3, -1],
                explicit_mipmaps: Vec::new(),
            }
        );
        assert_eq!(
            parse_texture_metadata(&location, None).unwrap(),
            TextureMetadata::default()
        );
    }

    #[test]
    fn anaglyph_color_matches_java_colored_and_gray_oracles() {
        // Java's integer formula: red, green and blue weights are [30, 59, 11], while the
        // output red/blue channels use [30, 70] and truncate rather than round.
        assert_eq!(anaglyph_color([255, 0, 0, 7]), [76, 76, 76, 7]);
        assert_eq!(anaglyph_color([0, 255, 0, 19]), [178, 150, 0, 19]);
        assert_eq!(anaglyph_color([0, 0, 255, 31]), [0, 28, 178, 31]);
        assert_eq!(
            anaglyph_color([128, 128, 128, 64]),
            [128, 128, 128, 64],
            "equal channels and alpha survive the exact integer transform"
        );
    }

    #[test]
    fn anaglyph_upload_preserves_disabled_bytes_and_alpha() {
        let source = [255, 0, 0, 3, 17, 91, 201, 0];
        assert_eq!(anaglyph_rgba(&source, false), source);
        assert_eq!(
            anaglyph_rgba(&source, true),
            [76, 76, 76, 3, 68, 80, 145, 0]
        );

        let mut disabled = source;
        apply_anaglyph_rgba(&mut disabled, false);
        assert_eq!(disabled, source);
    }

    #[test]
    fn texture_metadata_uses_gson_primitive_coercions() {
        let location = ResourceLocation::minecraft("textures/entity/test");
        let metadata = br#"{
            "texture": {"blur": "TrUe", "clamp": 1, "mipmaps": [1.5, "2", 3e0]}
        }"#;
        assert_eq!(
            parse_texture_metadata(&location, Some(metadata)).unwrap(),
            TextureMetadata {
                blur: true,
                clamp: false,
                mipmaps: vec![1, 2, 3],
                explicit_mipmaps: Vec::new(),
            }
        );
    }

    #[test]
    fn malformed_texture_metadata_is_rejected_at_java_error_boundaries() {
        let location = ResourceLocation::minecraft("textures/entity/test");
        for metadata in [
            br#"{"texture":{"blur":null}}"#.as_slice(),
            br#"{"texture":{"clamp":[]}}"#.as_slice(),
            br#"{"texture":{"mipmaps":{}}}"#.as_slice(),
            br#"{"texture":{"mipmaps":[true]}}"#.as_slice(),
            br#"{"texture":{"mipmaps":[{}]}}"#.as_slice(),
        ] {
            assert!(parse_texture_metadata(&location, Some(metadata)).is_err());
        }
    }

    #[test]
    fn texture_mipmap_metadata_skips_nested_arrays_and_null_like_java() {
        let location = ResourceLocation::minecraft("textures/entity/test");
        let metadata = br#"{
            "texture": {"mipmaps": [null, [], [1], 2, "3"]}
        }"#;
        assert_eq!(
            parse_texture_metadata(&location, Some(metadata))
                .unwrap()
                .mipmaps,
            vec![2, 3]
        );
    }

    #[test]
    fn non_animated_non_square_texture_is_rejected() {
        let image = image::RgbaImage::new(2, 3);
        let mut png = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let location = ResourceLocation::minecraft("textures/blocks/non_square");
        let png_location = location.with_suffix(".png");

        assert!(decode_sprite(&location, &png_location, png.get_ref(), None).is_err());
    }

    #[test]
    fn uv_rect_interpolation_matches_atlas_placement() {
        let uv = UvRect {
            min_u: 0.25,
            max_u: 0.5,
            min_v: 0.0,
            max_v: 0.25,
        };
        assert!((uv.interpolate_u(0.0) - 0.25).abs() < 1e-6);
        assert!((uv.interpolate_u(16.0) - 0.5).abs() < 1e-6);
        assert!((uv.interpolate_u(8.0) - 0.375).abs() < 1e-6);
    }

    #[test]
    fn solid_color_mipmaps_to_the_same_color() {
        let base = solid([200, 100, 50, 255], 4);
        let mip = generate_mipmap_level(&base, 4, false);
        assert_eq!(mip.len(), 2 * 2 * 4);
        for px in mip.chunks_exact(4) {
            // Gamma round-trip is not bit-exact, but should be within a texel of correct.
            assert!((px[0] as i32 - 200).abs() <= 1, "{px:?}");
            assert!((px[1] as i32 - 100).abs() <= 1, "{px:?}");
            assert!((px[2] as i32 - 50).abs() <= 1, "{px:?}");
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn mipmap_gamma_encoding_matches_java_truncation() {
        let texels = [
            [0, 0, 0, 255],
            [0, 0, 0, 255],
            [0, 0, 0, 255],
            [170, 0, 0, 255],
        ];
        assert_eq!(blend_texels(texels, false), [90, 0, 0, 255]);
    }

    #[test]
    fn mipmap_alpha_probe_only_reads_the_first_levels_plus_one_texels() {
        let mut base = solid([40, 80, 120, 255], 4);
        base[2 * 4..3 * 4].copy_from_slice(&[255, 255, 255, 0]);

        let chain = generate_mipmap_chain(&base, 4, 4, 1);
        assert_eq!(chain[1], generate_mipmap_level(&base, 4, false));
        assert_ne!(chain[1], generate_mipmap_level(&base, 4, true));
    }

    #[test]
    fn fully_transparent_neighbourhood_stays_transparent() {
        // All 4 texels transparent black: blended alpha is 0, well under the 96 cutoff.
        let base = solid([0, 0, 0, 0], 4);
        let mip = generate_mipmap_level(&base, 4, true);
        for px in mip.chunks_exact(4) {
            assert_eq!(px[3], 0, "{px:?}");
        }
    }

    #[test]
    fn low_alpha_blend_snaps_to_zero_avoiding_dark_fringe() {
        // 1-of-4 texels dimly translucent (alpha 64), 3 fully transparent: the fixed /4
        // divisor pulls the blended alpha well under the 96 cutoff, so it must snap to fully
        // transparent rather than leaving a faint grey fringe texel.
        let mut base = solid([0, 0, 0, 0], 2);
        base[0..4].copy_from_slice(&[255, 255, 255, 64]);
        let mip = generate_mipmap_level(&base, 2, true);
        assert_eq!(
            &mip[3], &0u8,
            "blended alpha must snap to 0 under the 96 threshold"
        );
    }

    #[test]
    fn moderate_partial_coverage_does_not_snap() {
        // 2-of-4 texels fully opaque: blended alpha ~186/255, comfortably above the cutoff —
        // this must stay visible, only the fully-sparse case should vanish.
        let mut base = solid([0, 0, 0, 0], 2);
        base[0..4].copy_from_slice(&[10, 20, 30, 255]);
        base[4..8].copy_from_slice(&[10, 20, 30, 255]);
        let mip = generate_mipmap_level(&base, 2, true);
        assert!(mip[3] > 96, "alpha={} should stay above the cutoff", mip[3]);
    }

    #[test]
    fn mipmap_chain_halves_each_level_and_stops_at_1x1() {
        let base = solid([10, 20, 30, 255], 8);
        let chain = generate_mipmap_chain(&base, 8, 8, 10);
        // 8 -> 4 -> 2 -> 1 = 4 levels total (level0..level3); level 1 has odd width 1 so it stops.
        assert_eq!(chain.len(), 4);
        assert_eq!(chain[0].len(), 8 * 8 * 4);
        assert_eq!(chain[1].len(), 4 * 4 * 4);
        assert_eq!(chain[2].len(), 2 * 2 * 4);
        assert_eq!(chain[3].len(), 4);
    }
}
