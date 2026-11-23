//! Connected textures (CTM): the resource pack's `mcpatcher/ctm/**.properties` rules, resolved
//! into a [`CtmTable`] that the terrain mesher reads without hashing, locking or allocating.
//!
//! The *behaviour* is the 1.8.9-era CTM format's, written from its documented rules; no
//! third-party code is used, and each claim was checked against a real pack.
//!
//! Only `method=ctm` (the 47-tile template every glass pack uses) is implemented. A rule with
//! another method, or with `biomes`/`heights`, is skipped with one warning, never half-applied.
//!
//! # Pipeline
//! 1. [`CtmRules::load`] lists each pack's rule files (highest-priority pack first, Java string
//!    order within a pack), parses and validates them, and drops every rule an
//!    earlier rule always shadows, so their tiles are never stitched.
//! 2. [`CtmRules::atlas_textures`] names the tiles the bake adds to the block atlas: `0..=46` of
//!    each surviving rule (tile 47 of a `0-47` set is never used by `method=ctm`).
//! 3. [`CtmTable::build`] runs after the stitch and the model bake. It resolves the format's rule
//!    precedence once per (block state, source sprite, face), so meshing never searches rules.

use std::collections::{HashMap, HashSet};

use mc_data::{BlockState, RenderType};
use mc_math::Facing;

use crate::model::state_mapper::{self, StateMapping};
use crate::model::{BakedModel, BakedQuad};
use crate::sprite::{Sprite, UvRect};
use crate::{ResourceLocation, ResourceManager};

/// Tiles of the `method=ctm` template that a rule must provide and that are ever drawn.
pub const CTM_TILE_COUNT: usize = 47;

/// What each tile of the `method=ctm` template draws as joined to its neighbours, in [`mask`]
/// bits: a side bit is an edge drawn open (no frame toward that neighbour), a corner bit a corner
/// drawn without its small inner notch. Entry `i` is tile `i` of the template's 12x4 grid
/// (row-major), the layout every pack's tiles follow.
const TEMPLATE_JOINS: [u8; CTM_TILE_COUNT] = {
    use mask::{D, DL, DR, L, R, U, UL, UR};
    const SIDES: u8 = L | R | D | U;
    [
        0,                         // 0: framed on every side
        R,                         // 1
        L | R,                     // 2
        L,                         // 3
        R | D,                     // 4
        L | D,                     // 5
        R | D | U,                 // 6
        L | R | D,                 // 7
        SIDES | UR,                // 8
        SIDES | DR,                // 9
        SIDES | DL | UL,           // 10
        SIDES | UL | UR,           // 11
        D,                         // 12
        R | D | DR,                // 13
        L | R | D | DL | DR,       // 14
        L | D | DL,                // 15
        R | U,                     // 16
        L | U,                     // 17
        L | R | U,                 // 18
        L | D | U,                 // 19
        SIDES | UL,                // 20
        SIDES | DL,                // 21
        SIDES | DL | DR,           // 22
        SIDES | DR | UR,           // 23
        D | U,                     // 24
        R | D | U | DR | UR,       // 25
        SIDES | DL | DR | UR | UL, // 26: joined all round, no frame at all
        L | D | U | DL | UL,       // 27
        R | D | U | DR,            // 28
        L | R | D | DL,            // 29
        R | D | U | UR,            // 30
        L | R | D | DR,            // 31
        SIDES | DL | UL | UR,      // 32
        SIDES | DR | UL | UR,      // 33
        SIDES | DL | UR,           // 34
        SIDES | DR | UL,           // 35
        U,                         // 36
        R | U | UR,                // 37
        L | R | U | UL | UR,       // 38
        L | U | UL,                // 39
        L | R | U | UR,            // 40
        L | D | U | UL,            // 41
        L | R | U | UL,            // 42
        L | D | U | DL,            // 43
        SIDES | DL | DR | UL,      // 44
        SIDES | DL | DR | UR,      // 45
        SIDES,                     // 46: open sides, four notched corners
    ]
};

/// The part of neighbour mask `m` a tile can show: its sides, plus each corner whose two sides
/// are both joined (a corner beside a framed side changes nothing the tile draws).
const fn joined_part(m: u8) -> u8 {
    use mask::{D, DL, DR, L, R, U, UL, UR};
    let corners = [(DL, D, L), (DR, D, R), (UR, U, R), (UL, U, L)];
    let mut out = m & (L | R | D | U);
    let mut i = 0;
    while i < corners.len() {
        let (corner, a, b) = corners[i];
        if m & corner != 0 && m & a != 0 && m & b != 0 {
            out |= corner;
        }
        i += 1;
    }
    out
}

/// For every neighbour mask, the template tile that joins exactly what the mask joins. A mask
/// with no such tile stops the build, so the template covers all 256 masks.
const fn build_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    let mut m = 0;
    while m < 256 {
        let want = joined_part(m as u8);
        let mut found = false;
        let mut tile = 0;
        while tile < CTM_TILE_COUNT {
            if TEMPLATE_JOINS[tile] == want {
                lut[m] = tile as u8;
                found = true;
            }
            tile += 1;
        }
        assert!(found, "a neighbour mask has no tile in the ctm template");
        m += 1;
    }
    lut
}

/// `method=ctm`'s tile for each neighbour mask, computed at compile time from
/// [`TEMPLATE_JOINS`].
///
/// Mask bits, in the face's own texture orientation: L=1, DL=2, D=4, DR=8, R=16, UR=32, U=64,
/// UL=128. A corner bit only matters when both of its sides are set; every "dangling" corner
/// maps to the same tile as the mask without it.
pub const CTM_LUT: [u8; 256] = build_lut();

/// Neighbour-mask bits, see [`CTM_LUT`].
pub mod mask {
    /// Left.
    pub const L: u8 = 1;
    /// Down-left corner.
    pub const DL: u8 = 2;
    /// Down.
    pub const D: u8 = 4;
    /// Down-right corner.
    pub const DR: u8 = 8;
    /// Right.
    pub const R: u8 = 16;
    /// Up-right corner.
    pub const UR: u8 = 32;
    /// Up.
    pub const U: u8 = 64;
    /// Up-left corner.
    pub const UL: u8 = 128;
}

/// "No rule applies" in [`CtmEntry::rule_by_face`].
pub const NO_RULE: u16 = u16::MAX;
/// "Not a CTM source sprite" in [`CtmTable::face_src`].
pub const NO_SRC: u16 = u16::MAX;

/// Rule files live under these directories of the `minecraft` namespace. 1.8.9-era packs use
/// `mcpatcher/`; `optifine/` is what packs backported from later versions use. Listing it as
/// well is harmless and `mcpatcher/` still sorts first within a pack.
const RULE_DIRS: [&str; 2] = ["mcpatcher/ctm/", "optifine/ctm/"];
const ANY_META: u16 = u16::MAX;
const ALL_FACES: u8 = 0x3F;
/// A hostile `tiles=0-4000000000` must not allocate the machine.
const MAX_TILE_RANGE: u32 = 4096;

/// The most `matchTiles` entries a rule may list (real rules list a
/// handful; the block atlas has a few hundred textures); a longer list refuses the rule.
const MAX_MATCH_TILES: usize = 256;

/// The most valid rules kept from a pack stack, in precedence order
/// (real packs have tens to a few hundred `method=ctm` rules); the rest are left out, logged.
const MAX_CTM_RULES: usize = 512;

/// How a rule decides that a neighbour "connects" (`connect=`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtmConnect {
    /// Same block id and metadata. The default for `matchBlocks` rules.
    Block,
    /// The neighbour's model shows the same (un-CTM'd) sprite on this face. Default otherwise.
    Tile,
    /// Same block material, and not air.
    Material,
}

/// One tile of a rule.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TileRef {
    /// A texture to stitch into the block atlas.
    Sprite(ResourceLocation),
    /// `<default>`/`<skip>`: keep the quad's own sprite.
    Keep,
}

/// One valid `method=ctm` rule, as parsed from its `.properties` file.
#[derive(Debug, Clone)]
pub struct CtmRule {
    source: String,
    match_tiles: Vec<ResourceLocation>,
    /// `(block id, metadata mask)`; the mask is [`ANY_META`] when the entry names none.
    match_blocks: Vec<(u16, u16)>,
    /// `match_tiles` as a set and `match_blocks` merged by block
    /// (the masks of one block's entries OR-ed), for the lookups the shadowing check and the
    /// table build make per rule and per quad: linear scans of the lists made those multiply.
    tile_set: HashSet<ResourceLocation>,
    block_masks: HashMap<u16, u16>,
    metadata: u16,
    faces: u8,
    connect: CtmConnect,
    inner_seams: bool,
    /// Exactly [`CTM_TILE_COUNT`] entries.
    tiles: Vec<TileRef>,
}

impl CtmRule {
    /// The rule file, `<namespace>:<path>`, e.g. `minecraft:mcpatcher/ctm/glass/Normal/glass.properties`.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The connect mode in effect (explicit or the format's default).
    #[must_use]
    pub fn connect(&self) -> CtmConnect {
        self.connect
    }

    /// The sprites this rule matches (`matchTiles`, explicit or derived from the file name).
    #[must_use]
    pub fn match_tiles(&self) -> &[ResourceLocation] {
        &self.match_tiles
    }

    /// The block ids this rule matches (`matchBlocks`).
    pub fn match_block_ids(&self) -> impl Iterator<Item = u16> + '_ {
        self.match_blocks.iter().map(|&(id, _)| id)
    }

    /// The atlas location of tile `index`, or `None` for a `<default>` tile or out of range.
    #[must_use]
    pub fn tile(&self, index: usize) -> Option<&ResourceLocation> {
        match self.tiles.get(index)? {
            TileRef::Sprite(location) => Some(location),
            TileRef::Keep => None,
        }
    }

    fn matches_block(&self, block: u16, meta: u8) -> bool {
        self.block_masks
            .get(&block)
            .is_some_and(|&metas| metas & (1 << meta) != 0)
    }

    /// The per-quad test for a rule found through `tileRules[sprite]`.
    fn applies_as_tile_rule(&self, block: u16, meta: u8, face: u8) -> bool {
        (self.match_blocks.is_empty() || self.matches_block(block, meta))
            && self.metadata & (1 << meta) != 0
            && self.faces & (1 << face) != 0
    }

    /// The per-quad test for a rule found through `blockRules[block]` (§3.5 step 2).
    fn applies_as_block_rule(
        &self,
        block: u16,
        meta: u8,
        sprite: &ResourceLocation,
        face: u8,
    ) -> bool {
        (self.match_tiles.is_empty() || self.tile_set.contains(sprite))
            && self.matches_block(block, meta)
            && self.metadata & (1 << meta) != 0
            && self.faces & (1 << face) != 0
    }
}

/// Every valid rule of the current pack stack, in the format's precedence order.
#[derive(Debug, Clone, Default)]
pub struct CtmRules {
    rules: Vec<CtmRule>,
    files: usize,
}

impl CtmRules {
    /// Find, parse, validate and prune every CTM rule in `rm` (spec §3.4–3.5).
    ///
    /// `model_textures` is the set of block-atlas textures the model bake references
    /// (`minecraft:textures/blocks/glass`, ...): a rule that names no `matchTiles` or
    /// `matchBlocks` matches the sprite its file is named after only when that sprite is in it.
    ///
    /// Never fails: a broken rule is logged and skipped.
    #[must_use]
    pub fn load(rm: &ResourceManager, model_textures: &HashSet<ResourceLocation>) -> CtmRules {
        let mut per_pack: Vec<Vec<String>> = Vec::new();
        for dir in RULE_DIRS {
            for (index, files) in rm
                .list_files_by_pack("minecraft", dir)
                .into_iter()
                .enumerate()
            {
                if per_pack.len() <= index {
                    per_pack.resize_with(index + 1, Vec::new);
                }
                per_pack[index].extend(files);
            }
        }
        // Every listed file, for tile existence checks without a filesystem probe per tile.
        let listed: HashSet<&str> = per_pack.iter().flatten().map(String::as_str).collect();
        let exists = |location: &ResourceLocation| {
            let png = format!("{}.png", location.path);
            (location.namespace == "minecraft" && listed.contains(png.as_str()))
                || rm.contains(&location.with_suffix(".png"))
        };

        let mut rules = Vec::new();
        let mut files = 0usize;
        let mut left_out = 0usize;
        for (pack, listing) in per_pack.iter().enumerate() {
            let mut properties: Vec<&String> = listing
                .iter()
                .filter(|path| path.ends_with(".properties"))
                .collect();
            // Each pack's file list in Java `String.compareTo` order (the format's). Byte order is
            // the same for the ASCII paths packs use.
            properties.sort();
            properties.dedup();
            for path in properties {
                files += 1;
                let location = ResourceLocation::minecraft(path.clone());
                let text = match rm.load_from_pack(pack, &location) {
                    Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                    Err(error) => {
                        tracing::warn!(rule = %location, %error, "CTM rule unreadable; skipped");
                        continue;
                    }
                };
                match parse_rule(path, &text, model_textures) {
                    Ok(rule) => {
                        let missing = rule.tiles.iter().find_map(|tile| match tile {
                            TileRef::Sprite(location) if !exists(location) => Some(location),
                            _ => None,
                        });
                        if let Some(missing) = missing {
                            tracing::warn!(rule = %location, tile = %missing, "CTM tile missing; rule skipped");
                            continue;
                        }
                        if rules.len() >= MAX_CTM_RULES {
                            left_out += 1;
                            continue;
                        }
                        rules.push(rule);
                    }
                    Err(RuleError::Unsupported(why)) => {
                        tracing::info!(rule = %location, why, "CTM rule not supported; skipped");
                    }
                    Err(RuleError::Invalid(why)) => {
                        tracing::warn!(rule = %location, why, "invalid CTM rule; skipped");
                    }
                }
            }
        }
        if left_out > 0 {
            tracing::warn!(
                left_out,
                kept = MAX_CTM_RULES,
                "more CTM rules than any real pack has; the rest are left out"
            );
        }
        let parsed = rules.len();
        let rules = prune_shadowed(rules);
        tracing::info!(
            files,
            valid = parsed,
            active = rules.len(),
            tiles = rules.len() * CTM_TILE_COUNT,
            "connected-texture rules loaded"
        );
        CtmRules { rules, files }
    }

    /// The surviving rules, in precedence order.
    #[must_use]
    pub fn rules(&self) -> &[CtmRule] {
        &self.rules
    }

    /// How many `.properties` files were found, valid or not.
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.files
    }

    /// No rule survived (the pack has no CTM, or none that this port supports).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// The tile textures to stitch, deduplicated, in a deterministic order.
    #[must_use]
    pub fn atlas_textures(&self) -> Vec<ResourceLocation> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for rule in &self.rules {
            for tile in &rule.tiles {
                if let TileRef::Sprite(location) = tile {
                    if seen.insert(location.clone()) {
                        out.push(location.clone());
                    }
                }
            }
        }
        out
    }

    /// The rule that applies to one quad, or `None`. `sprite` is the quad's
    /// own atlas texture, e.g. `minecraft:textures/blocks/glass`.
    #[must_use]
    pub fn winner(
        &self,
        state: BlockState,
        sprite: &ResourceLocation,
        face: Facing,
    ) -> Option<&CtmRule> {
        let (block, meta, face) = (state.block().0, state.meta(), face.index());
        let tile_rule = self.rules.iter().find(|rule| {
            rule.tile_set.contains(sprite) && rule.applies_as_tile_rule(block, meta, face)
        });
        tile_rule.or_else(|| {
            self.rules.iter().find(|rule| {
                !rule.match_blocks.is_empty()
                    && rule.applies_as_block_rule(block, meta, sprite, face)
            })
        })
    }
}

#[derive(Debug)]
enum RuleError {
    Unsupported(&'static str),
    Invalid(&'static str),
}

/// A Java-`Properties`-style parse: `#`/`!` comments, `=`/`:`/whitespace separators, CR, LF or
/// CRLF line ends, a missing final newline, trailing-backslash continuations. The last duplicate
/// key wins. Every value is trimmed.
fn parse_properties(text: &str) -> Vec<(String, String)> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    // CRLF is one terminator, so a continuation never joins an empty half-line.
    let text = text.replace("\r\n", "\n");
    // `out` holds each key once, where its last value was set;
    // `slots` finds a repeated key in one step (a `retain` over every earlier key made a file of
    // many distinct keys quadratic). Removed entries are `None` until the end.
    let mut out: Vec<Option<(String, String)>> = Vec::new();
    let mut slots: HashMap<String, usize> = HashMap::new();
    let mut logical = String::new();
    let mut lines = text.split(['\n', '\r']).peekable();
    while let Some(line) = lines.next() {
        let line = line.trim_start_matches([' ', '\t', '\u{c}']);
        if logical.is_empty() && (line.is_empty() || line.starts_with('#') || line.starts_with('!'))
        {
            continue;
        }
        // A line ending in an odd number of backslashes continues on the next non-empty line.
        let trailing = line.len() - line.trim_end_matches('\\').len();
        if trailing % 2 == 1 && lines.peek().is_some() {
            logical.push_str(&line[..line.len() - 1]);
            continue;
        }
        logical.push_str(line);
        let entry = std::mem::take(&mut logical);
        let mut key = String::new();
        let mut chars = entry.char_indices();
        let mut value_start = entry.len();
        while let Some((index, c)) = chars.next() {
            match c {
                '\\' => {
                    if let Some((_, escaped)) = chars.next() {
                        key.push(escaped);
                    }
                }
                '=' | ':' => {
                    value_start = index + 1;
                    break;
                }
                ' ' | '\t' | '\u{c}' => {
                    // Whitespace ends the key; an `=`/`:` may still follow it.
                    let rest = entry[index..].trim_start_matches([' ', '\t', '\u{c}']);
                    let skipped = entry.len() - rest.len();
                    value_start = if rest.starts_with(['=', ':']) {
                        skipped + 1
                    } else {
                        skipped
                    };
                    break;
                }
                _ => key.push(c),
            }
        }
        let value = entry[value_start.min(entry.len())..]
            .trim()
            .replace("\\\\", "\\");
        if key.is_empty() {
            continue;
        }
        if let Some(earlier) = slots.insert(key.clone(), out.len()) {
            out[earlier] = None;
        }
        out.push(Some((key, value)));
    }
    out.into_iter().flatten().collect()
}

fn property<'a>(props: &'a [(String, String)], key: &str) -> Option<&'a str> {
    props
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .filter(|v| !v.is_empty())
}

/// `path` is relative to `assets/minecraft/`, e.g. `mcpatcher/ctm/glass/Normal/glass.properties`.
fn parse_rule(
    path: &str,
    text: &str,
    model_textures: &HashSet<ResourceLocation>,
) -> Result<CtmRule, RuleError> {
    let props = parse_properties(text);
    let (base, file) = path.rsplit_once('/').unwrap_or(("", path));
    let name = file.strip_suffix(".properties").unwrap_or(file);

    for unsupported in ["biomes", "heights", "minHeight", "maxHeight"] {
        if property(&props, unsupported).is_some() {
            return Err(RuleError::Unsupported("biomes/heights"));
        }
    }
    match property(&props, "method")
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("ctm" | "glass") => {}
        Some(
            "ctm_compact"
            | "horizontal"
            | "bookshelf"
            | "vertical"
            | "horizontal+vertical"
            | "h+v"
            | "vertical+horizontal"
            | "v+h"
            | "top"
            | "sandstone"
            | "random"
            | "repeat"
            | "pattern"
            | "fixed"
            | "static"
            | "overlay"
            | "overlay_ctm"
            | "overlay_random"
            | "overlay_repeat"
            | "overlay_fixed",
        ) => return Err(RuleError::Unsupported("method other than ctm")),
        Some(_) => return Err(RuleError::Invalid("unknown method")),
        None => return Err(RuleError::Invalid("no method")),
    }

    let mut match_tiles: Vec<ResourceLocation> = property(&props, "matchTiles")
        .map(|value| {
            value
                .split_whitespace()
                .filter_map(match_tile_location)
                .collect()
        })
        .unwrap_or_default();
    let mut match_blocks: Vec<(u16, u16)> = property(&props, "matchBlocks")
        .map(|value| {
            value
                .split_whitespace()
                .filter_map(parse_match_block)
                .collect()
        })
        .unwrap_or_default();
    let explicit_blocks = property(&props, "matchBlocks").is_some();
    if match_tiles.is_empty() && match_blocks.is_empty() {
        if property(&props, "matchTiles").is_some() || explicit_blocks {
            return Err(RuleError::Invalid(
                "matchTiles/matchBlocks name nothing known",
            ));
        }
        // No match list: a file named `block<id>` matches that block, any other name the block
        // texture it is named after.
        if let Some(id) = name
            .strip_prefix("block")
            .and_then(|n| n.parse::<u16>().ok())
        {
            match_blocks.push((id, ANY_META));
        } else {
            let sprite = ResourceLocation::minecraft(format!("textures/blocks/{name}"));
            if !model_textures.contains(&sprite) {
                return Err(RuleError::Invalid("file name is not a block texture"));
            }
            match_tiles.push(sprite);
        }
    }

    let metadata = property(&props, "metadata").map_or(ANY_META, |value| {
        let mask = parse_int_mask(value, 15);
        if mask == 0 {
            ANY_META
        } else {
            mask
        }
    });
    let faces = property(&props, "faces").map_or(ALL_FACES, |value| {
        let faces = parse_faces(value);
        if faces == 0 {
            ALL_FACES
        } else {
            faces
        }
    });
    // The format's documented default: a rule that lists blocks connects by block, a rule that
    // lists tiles connects by tile.
    let default_connect = if match_blocks.is_empty() {
        CtmConnect::Tile
    } else {
        CtmConnect::Block
    };
    let connect = match property(&props, "connect")
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("block") => CtmConnect::Block,
        Some("tile") => CtmConnect::Tile,
        Some("material") => CtmConnect::Material,
        _ => default_connect,
    };
    let inner_seams =
        property(&props, "innerSeams").is_some_and(|value| value.eq_ignore_ascii_case("true"));

    let Some(tiles_value) = property(&props, "tiles") else {
        return Err(RuleError::Invalid("no tiles"));
    };
    let mut tiles = parse_tiles(tiles_value, base).ok_or(RuleError::Invalid("bad tile name"))?;
    if tiles.len() < CTM_TILE_COUNT {
        return Err(RuleError::Invalid("method=ctm needs 47 tiles"));
    }
    tiles.truncate(CTM_TILE_COUNT);

    if match_tiles.len() > MAX_MATCH_TILES {
        return Err(RuleError::Invalid(
            "matchTiles lists more tiles than a rule can use",
        ));
    }
    let tile_set = match_tiles.iter().cloned().collect();
    let mut block_masks: HashMap<u16, u16> = HashMap::new();
    for &(id, metas) in &match_blocks {
        *block_masks.entry(id).or_default() |= metas;
    }
    Ok(CtmRule {
        source: format!("minecraft:{path}"),
        match_tiles,
        match_blocks,
        tile_set,
        block_masks,
        metadata,
        faces,
        connect,
        inner_seams,
        tiles,
    })
}

/// `matchTiles` entry → atlas key: `glass` and `blocks/glass` both mean
/// `minecraft:textures/blocks/glass`.
fn match_tile_location(token: &str) -> Option<ResourceLocation> {
    let token = token.strip_suffix(".png").unwrap_or(token);
    let (namespace, path) = match token.split_once(':') {
        Some((namespace, path)) => (namespace, path),
        None => ("minecraft", token),
    };
    let path = if path.starts_with("textures/")
        || path.starts_with("mcpatcher/")
        || path.starts_with("optifine/")
    {
        path.to_owned()
    } else if path.contains('/') {
        format!("textures/{path}")
    } else {
        format!("textures/blocks/{path}")
    };
    let location = ResourceLocation::new(namespace, path);
    location.is_safe().then_some(location)
}

/// One `matchBlocks` token: `20`, `glass`, `minecraft:glass`, each with an optional
/// `:1,2,4-6` metadata list.
fn parse_match_block(token: &str) -> Option<(u16, u16)> {
    let parts: Vec<&str> = token.split(':').collect();
    let is_meta = |s: &str| {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_digit() || c == ',' || c == '-')
    };
    let (id, metas) = if parts[0].chars().all(|c| c.is_ascii_digit()) && !parts[0].is_empty() {
        (parts[0].parse::<u16>().ok()?, parts.get(1).copied())
    } else if parts.len() >= 2 && !is_meta(parts[1]) {
        let id = mc_data::block_by_name(&format!("{}:{}", parts[0], parts[1]))?;
        (id.0, parts.get(2).copied())
    } else {
        (mc_data::block_by_name(parts[0])?.0, parts.get(1).copied())
    };
    let mask = metas.map_or(ANY_META, |m| parse_int_mask(m, 15));
    (mask != 0).then_some((id, mask))
}

/// `1 2 4-6` / `1,2,4-6` → a bit mask over `0..=max`. Out-of-range values are ignored.
fn parse_int_mask(value: &str, max: u32) -> u16 {
    let mut mask = 0u16;
    for token in value.split([' ', ',', '\t']).filter(|t| !t.is_empty()) {
        let (from, to) = match token.split_once('-') {
            Some((a, b)) => (a.trim().parse::<u32>(), b.trim().parse::<u32>()),
            None => (token.parse::<u32>(), token.parse::<u32>()),
        };
        if let (Ok(from), Ok(to)) = (from, to) {
            for v in from..=to.min(max) {
                mask |= 1 << v;
            }
        }
    }
    mask
}

fn parse_faces(value: &str) -> u8 {
    let mut faces = 0u8;
    for token in value.split([' ', ',', '\t']).filter(|t| !t.is_empty()) {
        faces |= match token.to_ascii_lowercase().as_str() {
            "bottom" | "down" => 1 << Facing::Down.index(),
            "top" | "up" => 1 << Facing::Up.index(),
            "north" => 1 << Facing::North.index(),
            "south" => 1 << Facing::South.index(),
            "west" => 1 << Facing::West.index(),
            "east" => 1 << Facing::East.index(),
            "sides" => 0b11_1100,
            "all" => ALL_FACES,
            _ => 0,
        };
    }
    faces
}

/// `tiles=`: numbers, ranges and names, relative to the rule's directory `base` unless they
/// start at a texture root. `None` when a name could escape the pack.
fn parse_tiles(value: &str, base: &str) -> Option<Vec<TileRef>> {
    // A rule uses its first CTM_TILE_COUNT tiles, so only those are
    // kept (`parse_rule` checks the count and keeps 47): expanding every range in full let a few
    // kilobytes of `0-4095` tokens hold millions of tile names. Every token is still checked, so
    // an unsafe name anywhere refuses the rule as before.
    let mut out = Vec::with_capacity(CTM_TILE_COUNT);
    for token in value.split([' ', ',', '\t']).filter(|t| !t.is_empty()) {
        if let Some((from, to)) = token.split_once('-') {
            if let (Ok(from), Ok(to)) = (from.parse::<u32>(), to.parse::<u32>()) {
                if from <= to && to - from < MAX_TILE_RANGE {
                    // A numbered tile is `<base>/<n>`: safe exactly when the rule's own folder
                    // is, whatever the number, so the first tile answers for the whole range.
                    let first = tile_ref(&from.to_string(), base)?;
                    if out.len() < CTM_TILE_COUNT {
                        out.push(first);
                    }
                    let mut n = from;
                    while n < to && out.len() < CTM_TILE_COUNT {
                        n += 1;
                        out.push(tile_ref(&n.to_string(), base)?);
                    }
                }
                continue;
            }
        }
        let tile = tile_ref(token, base)?;
        if out.len() < CTM_TILE_COUNT {
            out.push(tile);
        }
    }
    Some(out)
}

fn tile_ref(token: &str, base: &str) -> Option<TileRef> {
    if token == "<skip>" || token == "<default>" {
        return Some(TileRef::Keep);
    }
    let token = token.strip_suffix(".png").unwrap_or(token);
    let location = if let Some(rest) = token.strip_prefix("./") {
        ResourceLocation::minecraft(format!("{base}/{rest}"))
    } else if let Some(rest) = token.strip_prefix("~/") {
        ResourceLocation::minecraft(format!("mcpatcher/{rest}"))
    } else if token.contains(':') {
        ResourceLocation::parse(token).ok()?
    } else if token.starts_with("textures/")
        || token.starts_with("mcpatcher/")
        || token.starts_with("optifine/")
    {
        ResourceLocation::minecraft(token)
    } else {
        ResourceLocation::minecraft(format!("{base}/{token}"))
    };
    location.is_safe().then_some(TileRef::Sprite(location))
}

/// Drop each tile rule that an earlier tile rule for all of its sprites always beats: the earlier
/// rule matches any block and its `faces` and `metadata` are supersets. The dropped
/// rule could never win a quad, so this changes no output and keeps its tiles out of the atlas.
fn prune_shadowed(rules: Vec<CtmRule>) -> Vec<CtmRule> {
    let mut kept: Vec<CtmRule> = Vec::with_capacity(rules.len());
    for rule in rules {
        let shadowed = rule.match_blocks.is_empty()
            && !rule.match_tiles.is_empty()
            && rule.match_tiles.iter().all(|sprite| {
                kept.iter().any(|earlier| {
                    earlier.match_blocks.is_empty()
                        && earlier.tile_set.contains(sprite)
                        && earlier.faces & rule.faces == rule.faces
                        && earlier.metadata & rule.metadata == rule.metadata
                })
            });
        if shadowed {
            tracing::debug!(rule = rule.source, "CTM rule shadowed by an earlier one");
        } else {
            kept.push(rule);
        }
    }
    kept
}

// ---- the mesher's table -------------------------------------------------------------------

/// One rule, ready for meshing: its 47 tile rectangles in the stitched atlas.
#[derive(Debug, Clone)]
pub struct CtmRuleRt {
    /// Atlas rectangle of each tile.
    pub tiles: [UvRect; CTM_TILE_COUNT],
    /// Bit `i` set: tile `i` is `<default>`, so the quad keeps its own sprite.
    pub keep: u64,
    /// The connect mode.
    pub connect: CtmConnect,
    /// `innerSeams=true`.
    pub inner_seams: bool,
    /// The rule file, for logs and tests.
    pub source: String,
}

/// One source sprite a block state shows, and what CTM does to it on each face.
#[derive(Debug, Clone, Copy)]
pub struct CtmEntry {
    /// The sprite's atlas rectangle; a quad belongs to this entry when its UV centroid is inside.
    pub src: UvRect,
    /// Index into the table's source-sprite list, compared by `connect=tile` neighbour checks.
    pub src_id: u16,
    /// Index into [`CtmTable::rule`] per [`Facing::index`], or [`NO_RULE`].
    pub rule_by_face: [u16; 6],
    /// A `glass_pane_top*` sprite on a pane: its Up/Down quads are hidden between two panes of
    /// the same stored state.
    pub pane_edge: bool,
}

/// The CTM entries of one block state.
#[derive(Debug, Clone, Default)]
pub struct CtmSlot {
    entries: Vec<CtmEntry>,
}

impl CtmSlot {
    /// Every entry, in model quad order.
    #[must_use]
    pub fn entries(&self) -> &[CtmEntry] {
        &self.entries
    }

    /// The entry whose source sprite contains `uv` (a quad's UV centroid).
    #[inline]
    #[must_use]
    pub fn entry_at(&self, uv: [f32; 2]) -> Option<&CtmEntry> {
        self.entries
            .iter()
            .find(|entry| rect_contains(&entry.src, uv))
    }
}

/// Everything the mesher needs, resolved once per bake. Indexed by [`BlockState::id`], so a
/// lookup is an array load; nearly every state has no slot.
#[derive(Debug, Clone)]
pub struct CtmTable {
    /// `0` = no CTM for this state, else slot index + 1.
    state_slot: Box<[u16]>,
    slots: Vec<CtmSlot>,
    rules: Vec<CtmRuleRt>,
    /// Per state and face: the source-sprite id of the *base* model's first quad on that face
    /// (`getNeighbourIcon`), or [`NO_SRC`].
    face_src: Box<[[u16; 6]]>,
    /// Per state: bit `f` set when any model variant of the state (including actual-state ones)
    /// shows a source sprite as its first quad on face `f`.
    face_any: Box<[u8]>,
    src_rects: Vec<UvRect>,
}

impl Default for CtmTable {
    fn default() -> Self {
        CtmTable {
            state_slot: vec![0; mc_data::STATE_COUNT].into_boxed_slice(),
            slots: Vec::new(),
            rules: Vec::new(),
            face_src: vec![[NO_SRC; 6]; mc_data::STATE_COUNT].into_boxed_slice(),
            face_any: vec![0; mc_data::STATE_COUNT].into_boxed_slice(),
            src_rects: Vec::new(),
        }
    }
}

impl CtmTable {
    /// The state's CTM entries, or `None` for the overwhelmingly common non-CTM state.
    #[inline]
    #[must_use]
    pub fn slot(&self, state: BlockState) -> Option<&CtmSlot> {
        match self.state_slot.get(state.id as usize) {
            Some(&0) | None => None,
            Some(&index) => self.slots.get(usize::from(index) - 1),
        }
    }

    /// A rule by its [`CtmEntry::rule_by_face`] index.
    #[inline]
    #[must_use]
    pub fn rule(&self, index: u16) -> Option<&CtmRuleRt> {
        self.rules.get(usize::from(index))
    }

    /// The source-sprite id of `state`'s base-model face, or [`NO_SRC`].
    #[inline]
    #[must_use]
    pub fn face_src(&self, state: BlockState, face: Facing) -> u16 {
        self.face_src
            .get(state.id as usize)
            .map_or(NO_SRC, |faces| faces[face.index() as usize])
    }

    /// Whether any model of `state` (actual-state variants included) can show a source sprite
    /// on `face`. `false` lets a `connect=tile` check skip the actual-state model lookup.
    #[inline]
    #[must_use]
    pub fn face_may_show_src(&self, state: BlockState, face: Facing) -> bool {
        self.face_any
            .get(state.id as usize)
            .is_some_and(|bits| bits & (1 << face.index()) != 0)
    }

    /// Number of rules with stitched tiles.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Number of block states with CTM entries.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Resolve every rule against the finished bake.
    ///
    /// `sprites` must be the bake's stitched sprites; a rule whose tiles are not all among them
    /// (a stitch that fell back to the vanilla atlas, a cache written without them) is dropped.
    /// Pane blocks get a slot even without rules, for the stacked-pane edge merge.
    #[must_use]
    pub fn build(
        rules: &CtmRules,
        sprites: &HashMap<ResourceLocation, Sprite>,
        models: &[BakedModel],
        weighted: &HashMap<u16, Vec<(i32, BakedModel)>>,
        actual_models: &HashMap<StateMapping, BakedModel>,
    ) -> CtmTable {
        let mut table = CtmTable::default();

        // Runtime rules, in precedence order; `None` for a rule whose tiles are not stitched.
        let mut runtime_index: Vec<Option<u16>> = Vec::with_capacity(rules.rules.len());
        for rule in &rules.rules {
            let mut tiles = [UvRect::FULL; CTM_TILE_COUNT];
            let mut keep = 0u64;
            let mut complete = true;
            for (index, tile) in rule.tiles.iter().enumerate() {
                match tile {
                    TileRef::Keep => keep |= 1 << index,
                    TileRef::Sprite(location) => match sprites.get(location) {
                        Some(sprite) => tiles[index] = sprite.uv,
                        None => complete = false,
                    },
                }
            }
            if complete && table.rules.len() < usize::from(NO_RULE) {
                runtime_index.push(Some(table.rules.len() as u16));
                table.rules.push(CtmRuleRt {
                    tiles,
                    keep,
                    connect: rule.connect,
                    inner_seams: rule.inner_seams,
                    source: rule.source.clone(),
                });
            } else {
                tracing::warn!(
                    rule = rule.source,
                    "CTM tiles not in the atlas; rule dropped"
                );
                runtime_index.push(None);
            }
        }
        let live = |index: usize| runtime_index[index].is_some();
        let has_block_rules: HashSet<u16> = rules
            .rules
            .iter()
            .enumerate()
            .filter(|(index, _)| live(*index))
            .flat_map(|(_, rule)| rule.match_block_ids())
            .collect();

        // Sprite identification by UV centroid. Rule sprites cover tile rules; block-rule blocks
        // need every sprite; pane edges need the `glass_pane_top*` family.
        let mut sorted: Vec<(&ResourceLocation, &Sprite)> = sprites.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(b.0));
        let rule_sprites: HashSet<&ResourceLocation> = rules
            .rules
            .iter()
            .enumerate()
            .filter(|(index, _)| live(*index))
            .flat_map(|(_, rule)| rule.match_tiles.iter())
            .collect();
        let tile_sprites: Vec<(UvRect, &ResourceLocation)> = sorted
            .iter()
            .filter(|(location, _)| rule_sprites.contains(location))
            .map(|(location, sprite)| (sprite.uv, *location))
            .collect();
        let pane_edge_sprites: Vec<(UvRect, &ResourceLocation)> = sorted
            .iter()
            .filter(|(location, _)| {
                location.namespace == "minecraft"
                    && location.path.starts_with("textures/blocks/glass_pane_top")
            })
            .map(|(location, sprite)| (sprite.uv, *location))
            .collect();
        // Every model texture; the CTM tiles themselves are never on a model.
        let all_sprites: Vec<(UvRect, &ResourceLocation)> = sorted
            .iter()
            .filter(|(location, _)| {
                !location.path.starts_with("mcpatcher/") && !location.path.starts_with("optifine/")
            })
            .map(|(location, sprite)| (sprite.uv, *location))
            .collect();
        let find = find_sprite;

        // Actual-state model variants by blockstate file.
        let mut by_stem: HashMap<&str, Vec<(&str, &BakedModel)>> = HashMap::new();
        for (mapping, model) in actual_models {
            by_stem
                .entry(mapping.file_stem.as_str())
                .or_default()
                .push((mapping.variant_key.as_str(), model));
        }
        for variants in by_stem.values_mut() {
            variants.sort_by(|a, b| a.0.cmp(b.0));
        }

        let resolve = |block: u16, meta: u8, sprite: &ResourceLocation, face: u8| -> u16 {
            let tile_rule = rules.rules.iter().enumerate().find(|(index, rule)| {
                live(*index)
                    && rule.tile_set.contains(sprite)
                    && rule.applies_as_tile_rule(block, meta, face)
            });
            let winner = tile_rule.or_else(|| {
                rules.rules.iter().enumerate().find(|(index, rule)| {
                    live(*index)
                        && !rule.match_blocks.is_empty()
                        && rule.applies_as_block_rule(block, meta, sprite, face)
                })
            });
            winner
                .and_then(|(index, _)| runtime_index[index])
                .unwrap_or(NO_RULE)
        };

        let mut src_ids: HashMap<&ResourceLocation, u16> = HashMap::new();
        let mut state_models: Vec<(BlockState, Vec<&BakedModel>)> = Vec::new();
        for state in mc_data::legal_states() {
            if state.props().render_type != RenderType::Model {
                continue;
            }
            let Some(base) = models.get(state.id as usize) else {
                continue;
            };
            let mut variants: Vec<&BakedModel> = vec![base];
            if let Some(candidates) = weighted.get(&state.id) {
                variants.extend(candidates.iter().map(|(_, model)| model));
            }
            let mapping = state_mapper::resolve(state);
            let stored = mc_data::state_properties(state);
            if let Some(list) = by_stem.get(mapping.file_stem.as_str()) {
                variants.extend(
                    list.iter()
                        .filter(|(key, _)| variant_compatible(key, stored))
                        .map(|(_, model)| *model),
                );
            }

            let block = state.block().0;
            let meta = state.meta();
            let block_rules = has_block_rules.contains(&block);
            let pane = is_pane(state);
            if !table.rules.is_empty() || pane {
                let mut entries: Vec<CtmEntry> = Vec::new();
                let mut entry_sprites: Vec<&ResourceLocation> = Vec::new();
                for model in &variants {
                    for quad in model
                        .face_quads
                        .iter()
                        .flatten()
                        .chain(&model.general_quads)
                    {
                        if quad.face.is_none() {
                            continue;
                        }
                        let uv = uv_centroid(quad);
                        let edge = if pane {
                            find(&pane_edge_sprites, uv)
                        } else {
                            None
                        };
                        let sprite = if block_rules {
                            find(&all_sprites, uv)
                        } else {
                            find(&tile_sprites, uv).or(edge)
                        };
                        let Some(sprite) = sprite else {
                            continue;
                        };
                        if entry_sprites.contains(&sprite) {
                            continue;
                        }
                        let mut rule_by_face = [NO_RULE; 6];
                        for (face, slot) in rule_by_face.iter_mut().enumerate() {
                            *slot = resolve(block, meta, sprite, face as u8);
                        }
                        let pane_edge = edge.is_some_and(|edge| edge == sprite);
                        if rule_by_face.iter().all(|&rule| rule == NO_RULE) && !pane_edge {
                            continue;
                        }
                        let next_id = src_ids.len() as u16;
                        let src_id = *src_ids.entry(sprite).or_insert(next_id);
                        entry_sprites.push(sprite);
                        entries.push(CtmEntry {
                            src: sprites[sprite].uv,
                            src_id,
                            rule_by_face,
                            pane_edge,
                        });
                    }
                }
                if !entries.is_empty() && table.slots.len() < usize::from(u16::MAX) - 1 {
                    table.slots.push(CtmSlot { entries });
                    table.state_slot[state.id as usize] = table.slots.len() as u16;
                }
            }
            state_models.push((state, variants));
        }

        // Source sprites by id, then each state's face sprites for `connect=tile` checks.
        let mut src_rects = vec![UvRect::FULL; src_ids.len()];
        for (location, &id) in &src_ids {
            src_rects[usize::from(id)] = sprites[*location].uv;
        }
        let src_of = |uv: [f32; 2]| -> u16 {
            src_rects
                .iter()
                .position(|rect| rect_contains(rect, uv))
                .map_or(NO_SRC, |index| index as u16)
        };
        if !src_rects.is_empty() {
            for (state, variants) in &state_models {
                let id = state.id as usize;
                for face in Facing::VALUES {
                    let f = face.index() as usize;
                    if let Some(quad) = first_face_quad(variants[0], face) {
                        table.face_src[id][f] = src_of(uv_centroid(quad));
                    }
                    let any = variants.iter().any(|model| {
                        first_face_quad(model, face)
                            .is_some_and(|quad| src_of(uv_centroid(quad)) != NO_SRC)
                    });
                    if any {
                        table.face_any[id] |= 1 << f;
                    }
                }
            }
        }
        table.src_rects = src_rects;
        tracing::info!(
            rules = table.rules.len(),
            states = table.slots.len(),
            sprites = table.src_rects.len(),
            "connected-textures table built"
        );
        table
    }
}

fn find_sprite<'a>(
    list: &[(UvRect, &'a ResourceLocation)],
    uv: [f32; 2],
) -> Option<&'a ResourceLocation> {
    list.iter()
        .find(|(rect, _)| rect_contains(rect, uv))
        .map(|(_, location)| *location)
}

/// Whether a blockstate variant key (`east=false,north=true`) can belong to a stored state: no
/// stored property disagrees. Derived (actual-state) properties are not stored, so they pass.
fn variant_compatible(key: &str, stored: &[(&str, &str)]) -> bool {
    key.split(',').all(|pair| match pair.split_once('=') {
        Some((name, value)) => stored
            .iter()
            .find(|(stored_name, _)| *stored_name == name)
            .is_none_or(|(_, stored_value)| *stored_value == value),
        None => true,
    })
}

/// The pane rule applies to `BlockPane` and its stained subclass.
fn is_pane(state: BlockState) -> bool {
    let class = mc_data::block_info(state.block()).java_class;
    matches!(
        class.rsplit('.').next().unwrap_or(class),
        "BlockPane" | "BlockStainedGlassPane"
    )
}

/// The quad the neighbour test reads for `face`: the first culled quad of that face, else the
/// first general quad facing that way.
#[must_use]
pub fn first_face_quad(model: &BakedModel, face: Facing) -> Option<&BakedQuad> {
    model.face_quads[face.index() as usize].first().or_else(|| {
        model
            .general_quads
            .iter()
            .find(|quad| quad.face == Some(face))
    })
}

/// The mean of a quad's four atlas UVs: always inside the sprite the quad samples.
#[inline]
#[must_use]
pub fn uv_centroid(quad: &BakedQuad) -> [f32; 2] {
    let [a, b, c, d] = &quad.verts;
    [
        (a.uv[0] + b.uv[0] + c.uv[0] + d.uv[0]) * 0.25,
        (a.uv[1] + b.uv[1] + c.uv[1] + d.uv[1]) * 0.25,
    ]
}

/// Whether `uv` lies inside `rect` (edges included).
#[inline]
#[must_use]
pub fn rect_contains(rect: &UvRect, uv: [f32; 2]) -> bool {
    uv[0] >= rect.min_u && uv[0] <= rect.max_u && uv[1] >= rect.min_v && uv[1] <= rect.max_v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 47-row template table: (tile, connected cardinals, connected corners).
    const TEMPLATE: [(u8, &str, &str); 47] = [
        (0, "", ""),
        (1, "R", ""),
        (2, "L R", ""),
        (3, "L", ""),
        (4, "R D", ""),
        (5, "L D", ""),
        (6, "R D U", ""),
        (7, "L R D", ""),
        (8, "L R D U", "UR"),
        (9, "L R D U", "DR"),
        (10, "L R D U", "DL UL"),
        (11, "L R D U", "UL UR"),
        (12, "D", ""),
        (13, "R D", "DR"),
        (14, "L R D", "DL DR"),
        (15, "L D", "DL"),
        (16, "R U", ""),
        (17, "L U", ""),
        (18, "L R U", ""),
        (19, "L D U", ""),
        (20, "L R D U", "UL"),
        (21, "L R D U", "DL"),
        (22, "L R D U", "DL DR"),
        (23, "L R D U", "DR UR"),
        (24, "D U", ""),
        (25, "R D U", "DR UR"),
        (26, "L R D U", "DL DR UR UL"),
        (27, "L D U", "DL UL"),
        (28, "R D U", "DR"),
        (29, "L R D", "DL"),
        (30, "R D U", "UR"),
        (31, "L R D", "DR"),
        (32, "L R D U", "DL UL UR"),
        (33, "L R D U", "DR UL UR"),
        (34, "L R D U", "DL UR"),
        (35, "L R D U", "DR UL"),
        (36, "U", ""),
        (37, "R U", "UR"),
        (38, "L R U", "UL UR"),
        (39, "L U", "UL"),
        (40, "L R U", "UR"),
        (41, "L D U", "UL"),
        (42, "L R U", "UL"),
        (43, "L D U", "DL"),
        (44, "L R D U", "DL DR UL"),
        (45, "L R D U", "DL DR UR"),
        (46, "L R D U", ""),
    ];

    fn bits(names: &str) -> u8 {
        names
            .split_whitespace()
            .map(|name| match name {
                "L" => mask::L,
                "DL" => mask::DL,
                "D" => mask::D,
                "DR" => mask::DR,
                "R" => mask::R,
                "UR" => mask::UR,
                "U" => mask::U,
                "UL" => mask::UL,
                other => panic!("bad direction {other}"),
            })
            .fold(0, |acc, bit| acc | bit)
    }

    /// The meaningful part of a mask: its cardinals, plus each corner whose two cardinals are set.
    fn effective(m: u8) -> u8 {
        let mut out = m & (mask::L | mask::D | mask::R | mask::U);
        for (corner, a, b) in [
            (mask::DL, mask::D, mask::L),
            (mask::DR, mask::D, mask::R),
            (mask::UR, mask::U, mask::R),
            (mask::UL, mask::U, mask::L),
        ] {
            if m & corner != 0 && m & a != 0 && m & b != 0 {
                out |= corner;
            }
        }
        out
    }

    #[test]
    fn the_lut_matches_the_47_tile_template_for_all_256_masks() {
        let template: Vec<(u8, u8)> = TEMPLATE
            .iter()
            .map(|(tile, cardinals, corners)| (*tile, bits(cardinals) | bits(corners)))
            .collect();
        // The template is complete and unambiguous: 47 distinct effective masks.
        let distinct: HashSet<u8> = template.iter().map(|(_, m)| *m).collect();
        assert_eq!(distinct.len(), 47);
        for (tile, m) in &template {
            assert_eq!(effective(*m), *m, "tile {tile} names a dangling corner");
            // The table the LUT is built from says the same as this one.
            assert_eq!(TEMPLATE_JOINS[usize::from(*tile)], *m, "tile {tile}");
            assert_eq!(joined_part(*m), *m, "tile {tile}");
        }
        for m in 0..=255u8 {
            assert_eq!(joined_part(m), effective(m), "mask {m}");
        }
        for m in 0..=255u8 {
            let want = effective(m);
            let matches: Vec<u8> = template
                .iter()
                .filter(|(_, tm)| *tm == want)
                .map(|(tile, _)| *tile)
                .collect();
            assert_eq!(matches.len(), 1, "mask {m}: {matches:?}");
            assert_eq!(CTM_LUT[usize::from(m)], matches[0], "mask {m}");
        }
        // Worked examples (3×3 wall seen from the south).
        assert_eq!(CTM_LUT[0], 0);
        assert_eq!(CTM_LUT[17], 2);
        assert_eq!(CTM_LUT[16], 1);
        assert_eq!(CTM_LUT[255], 26);
        assert_eq!(CTM_LUT[31], 14);
        assert_eq!(CTM_LUT[28], 13);
        assert_eq!(CTM_LUT[193], 39);
        assert_eq!(CTM_LUT[247], 32);
    }

    #[test]
    fn properties_parse_like_java_with_crlf_junk_and_no_final_newline() {
        let props = parse_properties(
            "method=ctm\r\ntiles=60-71 76-87 92-103 108-118\r\n\r\nGlass and Glass Panes\r\n\
             # comment\r\n! also comment\r\n  connect : tile\r\nmatchBlocks=160 95\\\r\n 102",
        );
        assert_eq!(property(&props, "method"), Some("ctm"));
        assert_eq!(
            property(&props, "tiles"),
            Some("60-71 76-87 92-103 108-118")
        );
        assert_eq!(property(&props, "Glass"), Some("and Glass Panes"));
        assert_eq!(property(&props, "connect"), Some("tile"));
        assert_eq!(property(&props, "matchBlocks"), Some("160 95102"));
        assert_eq!(property(&props, "missing"), None);
    }

    fn textures(names: &[&str]) -> HashSet<ResourceLocation> {
        names
            .iter()
            .map(|n| ResourceLocation::minecraft(format!("textures/blocks/{n}")))
            .collect()
    }

    #[test]
    fn a_rule_derives_match_tiles_from_its_file_name_and_expands_ranges() {
        let rule = parse_rule(
            "mcpatcher/ctm/glass/Normal/glass.properties",
            "method=ctm\r\ntiles=60-71 76-87 92-103 108-118",
            &textures(&["glass"]),
        )
        .expect("valid rule");
        assert_eq!(
            rule.match_tiles,
            [ResourceLocation::minecraft("textures/blocks/glass")]
        );
        assert_eq!(rule.connect, CtmConnect::Tile);
        assert_eq!(rule.tiles.len(), CTM_TILE_COUNT);
        assert_eq!(
            rule.tile(0).unwrap().to_string(),
            "minecraft:mcpatcher/ctm/glass/Normal/60"
        );
        assert_eq!(
            rule.tile(12).unwrap().to_string(),
            "minecraft:mcpatcher/ctm/glass/Normal/76"
        );
        assert_eq!(
            rule.tile(46).unwrap().to_string(),
            "minecraft:mcpatcher/ctm/glass/Normal/118"
        );
        // A name that is not a model texture makes no rule.
        assert!(parse_rule(
            "mcpatcher/ctm/x/nope.properties",
            "method=ctm\ntiles=0-46",
            &textures(&[])
        )
        .is_err());
        // Fewer than 47 tiles is invalid; other methods are unsupported.
        assert!(matches!(
            parse_rule(
                "mcpatcher/ctm/glass.properties",
                "method=ctm\ntiles=0-45",
                &textures(&["glass"])
            ),
            Err(RuleError::Invalid(_))
        ));
        assert!(matches!(
            parse_rule(
                "mcpatcher/ctm/glass.properties",
                "method=horizontal\ntiles=0-3",
                &textures(&["glass"])
            ),
            Err(RuleError::Unsupported(_))
        ));
        assert!(matches!(
            parse_rule(
                "mcpatcher/ctm/glass.properties",
                "method=ctm\ntiles=0-46\nbiomes=plains",
                &textures(&["glass"])
            ),
            Err(RuleError::Unsupported(_))
        ));
    }

    #[test]
    fn block_rules_parse_ids_names_metadata_and_default_to_block_connect() {
        let rule = parse_rule(
            "mcpatcher/ctm/glass_stained/glass_black/glass_black.properties",
            "matchBlocks=160 95 minecraft:glass stone:1,3-4\nmetadata=15\nmethod=ctm\ntiles=0-47",
            &textures(&[]),
        )
        .expect("valid block rule");
        assert_eq!(rule.connect, CtmConnect::Block);
        assert_eq!(
            rule.match_blocks,
            [
                (160, ANY_META),
                (95, ANY_META),
                (20, ANY_META),
                (1, 0b1_1010)
            ]
        );
        assert_eq!(rule.metadata, 1 << 15);
        assert!(rule.match_tiles.is_empty());
        let block_named = parse_rule(
            "mcpatcher/ctm/bookshelf/block47.properties",
            "method=ctm\ntiles=0-46",
            &textures(&[]),
        )
        .expect("block<id> file name");
        assert_eq!(block_named.match_blocks, [(47, ANY_META)]);
        assert_eq!(parse_faces("sides top"), 0b11_1110);
        assert_eq!(parse_faces("bottom"), 1);
    }

    /// Writes `files` (relative to `assets/minecraft/mcpatcher/ctm/`) as a folder pack.
    fn folder_pack(files: &[(&str, &str)]) -> (tempfile::TempDir, crate::FolderResourcePack) {
        let temp = tempfile::tempdir().expect("tempdir");
        let ctm = temp.path().join("assets/minecraft/mcpatcher/ctm");
        for (path, contents) in files {
            let full = ctm.join(path);
            std::fs::create_dir_all(full.parent().unwrap()).unwrap();
            std::fs::write(full, contents).unwrap();
        }
        let pack = crate::FolderResourcePack::open(temp.path()).expect("open folder pack");
        (temp, pack)
    }

    fn tiles_for(dir: &str, range: std::ops::RangeInclusive<u32>) -> Vec<(String, &'static str)> {
        range.map(|n| (format!("{dir}/{n}.png"), "png")).collect()
    }

    /// A pack laid out like the user's (spaces, uppercase, duplicate sets, block rules).
    #[test]
    fn precedence_on_a_fixture_shaped_like_the_users_pack() {
        let mut owned: Vec<(String, &str)> = vec![
            (
                "glass/Normal/glass.properties".into(),
                "method=ctm\r\ntiles=60-71 76-87 92-103 108-118",
            ),
            (
                "glass/glass.properties".into(),
                "method=ctm\ntiles=0-46\n\nGlass and Glass Panes",
            ),
            (
                "glass_clear/glass.properties".into(),
                "method=ctm\ntiles=0-46",
            ),
            (
                "glass black/glass_black.properties".into(),
                "metadata=15\nmethod=ctm\ntiles=0-47",
            ),
            (
                "glass/Black Glass/glass_black.properties".into(),
                "metadata=15\nmethod=ctm\ntiles=0-47",
            ),
            (
                "glass light gray/glass_silver.properties".into(),
                "metadata=8\nmethod=ctm\ntiles=0-47",
            ),
            (
                "glass silver/glass_silver.properties".into(),
                "method=ctm\ntiles=0-47\nconnect=tile",
            ),
            (
                "glass_stained/glass_black/glass_black.properties".into(),
                "matchBlocks=160 95\nmetadata=15\nmethod=ctm\ntiles=0-47",
            ),
            ("Thumbs.db".into(), "junk"),
        ];
        for (dir, range) in [
            ("glass/Normal", 60..=118),
            ("glass", 0..=46),
            ("glass_clear", 0..=46),
            ("glass black", 0..=47),
            ("glass/Black Glass", 0..=47),
            ("glass light gray", 0..=47),
            ("glass silver", 0..=47),
            ("glass_stained/glass_black", 0..=47),
        ] {
            owned.extend(tiles_for(dir, range));
        }
        let files: Vec<(&str, &str)> = owned.iter().map(|(p, c)| (p.as_str(), *c)).collect();
        let (_temp, pack) = folder_pack(&files);
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(pack));
        let model_textures = textures(&[
            "glass",
            "glass_black",
            "glass_silver",
            "glass_pane_top_black",
        ]);
        let rules = CtmRules::load(&rm, &model_textures);
        assert_eq!(rules.file_count(), 8);

        let glass = BlockState::new(mc_data::block_by_name("glass").unwrap(), 0);
        let pane = BlockState::new(mc_data::block_by_name("glass_pane").unwrap(), 0);
        let stained =
            |meta| BlockState::new(mc_data::block_by_name("stained_glass").unwrap(), meta);
        let stained_pane =
            |meta| BlockState::new(mc_data::block_by_name("stained_glass_pane").unwrap(), meta);
        let sprite = |name: &str| ResourceLocation::minecraft(format!("textures/blocks/{name}"));
        let winner = |state, name: &str, face| {
            rules
                .winner(state, &sprite(name), face)
                .map(CtmRule::source)
        };
        // `N` sorts before `g`: the 64x Normal set wins clear glass and panes.
        for state in [glass, pane] {
            assert_eq!(
                winner(state, "glass", Facing::South),
                Some("minecraft:mcpatcher/ctm/glass/Normal/glass.properties")
            );
        }
        // ' ' sorts before '/': `glass black/` beats `glass/Black Glass/`.
        for state in [stained(15), stained_pane(15)] {
            assert_eq!(
                winner(state, "glass_black", Facing::Up),
                Some("minecraft:mcpatcher/ctm/glass black/glass_black.properties")
            );
        }
        // `glass light gray/` (metadata 8) beats `glass silver/`.
        assert_eq!(
            winner(stained(8), "glass_silver", Facing::North),
            Some("minecraft:mcpatcher/ctm/glass light gray/glass_silver.properties")
        );
        // The block rule reaches only the stained pane's edge sprite.
        assert_eq!(
            winner(stained_pane(15), "glass_pane_top_black", Facing::Up),
            Some("minecraft:mcpatcher/ctm/glass_stained/glass_black/glass_black.properties")
        );
        assert_eq!(
            winner(stained_pane(14), "glass_pane_top_black", Facing::Up),
            None
        );
        assert_eq!(winner(glass, "stone", Facing::Up), None);

        // Pruned: the 16x clear-glass copies and the duplicate black set.
        let sources: Vec<&str> = rules.rules().iter().map(CtmRule::source).collect();
        assert_eq!(
            sources,
            [
                "minecraft:mcpatcher/ctm/glass black/glass_black.properties",
                "minecraft:mcpatcher/ctm/glass light gray/glass_silver.properties",
                "minecraft:mcpatcher/ctm/glass silver/glass_silver.properties",
                "minecraft:mcpatcher/ctm/glass/Normal/glass.properties",
                "minecraft:mcpatcher/ctm/glass_stained/glass_black/glass_black.properties",
            ]
        );
        // Tiles 0..=46 only: tile 47 of a 0-47 set is never stitched.
        let atlas = rules.atlas_textures();
        assert_eq!(atlas.len(), 5 * CTM_TILE_COUNT);
        assert!(!atlas.iter().any(|l| l.path.ends_with("/47")));
    }

    /// The reference pack, read-only, when this machine has it: `MC_REFERENCE_PACK`, or else
    /// the only folder in the workspace's `resourcepacks/` (never checked in).
    #[test]
    fn precedence_on_the_users_real_pack() {
        let packs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resourcepacks");
        let only_folder = || {
            let mut dirs = std::fs::read_dir(&packs)
                .ok()?
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.is_dir());
            let only = dirs.next()?;
            dirs.next().is_none().then_some(only)
        };
        let Some(root) = std::env::var_os("MC_REFERENCE_PACK")
            .map(std::path::PathBuf::from)
            .or_else(only_folder)
        else {
            eprintln!(
                "SKIP: no reference pack (MC_REFERENCE_PACK, or one folder in {})",
                packs.display()
            );
            return;
        };
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(
            crate::FolderResourcePack::open(&root).expect("open pack"),
        ));
        let colours = [
            "white",
            "orange",
            "magenta",
            "light_blue",
            "yellow",
            "lime",
            "pink",
            "gray",
            "silver",
            "cyan",
            "purple",
            "blue",
            "brown",
            "green",
            "red",
            "black",
        ];
        let mut names = vec!["glass".to_owned(), "glass_pane_top".to_owned()];
        for colour in colours {
            names.push(format!("glass_{colour}"));
            names.push(format!("glass_pane_top_{colour}"));
        }
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let rules = CtmRules::load(&rm, &textures(&name_refs));
        assert_eq!(rules.file_count(), 52, "the pack ships 52 rule files");
        let tile_rules = rules
            .rules()
            .iter()
            .filter(|r| r.match_blocks.is_empty())
            .count();
        let block_rules = rules.rules().len() - tile_rules;
        // 17 active tile rules plus `glass silver/` (metadata any is not a subset of 8).
        assert_eq!(tile_rules, 18);
        assert_eq!(block_rules, 16);

        let glass = BlockState::new(mc_data::block_by_name("glass").unwrap(), 0);
        let sprite = |name: &str| ResourceLocation::minecraft(format!("textures/blocks/{name}"));
        assert_eq!(
            rules
                .winner(glass, &sprite("glass"), Facing::South)
                .map(CtmRule::source),
            Some("minecraft:mcpatcher/ctm/glass/Normal/glass.properties")
        );
        let stained =
            |meta| BlockState::new(mc_data::block_by_name("stained_glass").unwrap(), meta);
        for (meta, colour) in colours.iter().enumerate() {
            let dir = if *colour == "silver" {
                "glass light gray".to_owned()
            } else {
                format!("glass {}", colour.replace('_', " "))
            };
            let want = format!("minecraft:mcpatcher/ctm/{dir}/glass_{colour}.properties");
            assert_eq!(
                rules
                    .winner(
                        stained(meta as u8),
                        &sprite(&format!("glass_{colour}")),
                        Facing::Up
                    )
                    .map(CtmRule::source),
                Some(want.as_str()),
                "{colour}"
            );
        }
        let pane = BlockState::new(mc_data::block_by_name("stained_glass_pane").unwrap(), 15);
        assert_eq!(
            rules
                .winner(pane, &sprite("glass_pane_top_black"), Facing::Down)
                .map(CtmRule::source),
            Some("minecraft:mcpatcher/ctm/glass_stained/glass_black/glass_black.properties")
        );
        assert_eq!(rules.atlas_textures().len(), 34 * CTM_TILE_COUNT);
    }
}
