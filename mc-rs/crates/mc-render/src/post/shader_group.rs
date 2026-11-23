//! Parses vanilla's shader-group JSON format. java: `client/shader/ShaderGroup.java`
//! (`parseGroup`/`initTarget`/`parsePass`/`initUniform`) and `client/shader/ShaderManager.java`
//! (the per-program `vertex`/`fragment`/`samplers`/`uniforms`/`blend`/`cull` schema).
//!
//! This is a faithful **parser** for the real files vanilla ships
//! (`assets/minecraft/shaders/post/*.json`, `assets/minecraft/shaders/program/*.json`) — see
//! `tests/s20_post.rs` for round-trips against the actual `1.8.9.jar`. It is deliberately not a
//! GLSL interpreter: [`super::pipeline::PostChain`] is a from-scratch `wgpu` multi-pass
//! compositor driven by the *structure* this module recovers (targets, pass chaining,
//! `auxtargets`, declared uniform values), not by executing the referenced `.vsh`/`.fsh` source.
//! See `super::pipeline`'s module docs for exactly which programs that chain can actually run and
//! why arbitrary vanilla GLSL cannot be replayed unmodified in `wgpu`.
//!
//! # Why a hand-rolled parser
//!
//! `mc-render` does not depend on `serde`/`serde_json` (see the crate's `Cargo.toml`; S11/S12/S20
//! share it and it is frozen for the run — see this crate's rules), so this is a small
//! purpose-built JSON reader rather than `#[derive(Deserialize)]`, in the same spirit as
//! `settings.rs`'s hand-rolled `parse_json_string_array`/`json_string_array` ("pulling
//! `serde_json` into the renderer to read two lines is not a good trade" — the same logic applies
//! here, just for a slightly less trivial schema). [`Json`] is a minimal recursive-descent parser
//! for the JSON subset these files use (objects, arrays, strings with `\"`/`\\`/`\/`/`\n`-style
//! escapes, numbers, `true`/`false`/`null`) — general enough for any file in
//! `assets/minecraft/shaders/**`, not a full RFC 8259 implementation (no `\uXXXX` escapes, no
//! exponent-form edge cases beyond what `f64::from_str` accepts).

use std::fmt;

use mc_assets::{ResourceLocation, ResourceManager};

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true`/`false`.
    Bool(bool),
    /// Any JSON number, stored widened to `f64`.
    Number(f64),
    /// A string, with escapes already resolved.
    String(String),
    /// `[...]`, order preserved.
    Array(Vec<Json>),
    /// `{...}`, insertion order preserved (a `BTreeMap` would silently reorder / dedupe on a
    /// pathological duplicate key, which is closer to `Gson`'s own last-write-wins behaviour).
    Object(Vec<(String, Json)>),
}

/// A malformed document. Untrusted input (a resource pack can ship any bytes here), so this is a
/// typed error rather than a panic — same policy as every other JSON-shaped asset in this
/// workspace.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonError {
    /// Human-readable description, including a byte offset when useful.
    pub message: String,
    /// Outer-to-inner JSON keys and array indices, matching vanilla's `JsonException` chain.
    path: Vec<String>,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.path.is_empty() {
            write!(f, "malformed json: {}", self.message)
        } else {
            write!(
                f,
                "malformed json at {}: {}",
                self.path.join("->"),
                self.message
            )
        }
    }
}

impl std::error::Error for JsonError {}

impl JsonError {
    /// The outer-to-inner JSON key/index chain attached while nested values were parsed.
    #[must_use]
    pub fn path(&self) -> &[String] {
        &self.path
    }

    fn prepend_path(mut self, segment: impl Into<String>) -> Self {
        self.path.insert(0, segment.into());
        self
    }
}

/// The two source stages loaded by vanilla's [`crate::post::ShaderLoader`] replacement.
///
/// Java addresses these as `shaders/program/<name>.vsh` and `.fsh`; the extension is part of the
/// resource lookup contract, not a hint that the bytes are interchangeable with WGSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    /// A vertex-stage GLSL source (`.vsh`).
    Vertex,
    /// A fragment-stage GLSL source (`.fsh`).
    Fragment,
}

impl ShaderStage {
    /// The Java resource extension for this stage.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Vertex => ".vsh",
            Self::Fragment => ".fsh",
        }
    }

    /// The human-readable stage name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }
}

impl fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A resource-backed GLSL source discovered while resolving one program.
///
/// The source text is deliberately not exposed as executable WGSL. Keeping the pack name and a
/// stable fingerprint makes an override visible to callers and tests without pretending that this
/// renderer can run Java's GLSL through its WGSL-only backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderSourceInfo {
    /// Which Java shader stage supplied the source.
    pub stage: ShaderStage,
    /// Exact namespaced location used for the resource lookup.
    pub location: ResourceLocation,
    /// Name of the highest-priority pack supplying the source.
    pub pack_name: String,
    /// Number of bytes read from the resource.
    pub byte_len: usize,
    /// Deterministic FNV-1a fingerprint of the source bytes.
    pub fingerprint: u64,
    /// Whether another lower-priority pack also supplied this same location.
    ///
    /// This is the observable resource-pack override oracle. A single source in a standalone
    /// manager is still reported by [`ProgramSourceResolution`], but cannot be proven to shadow a
    /// lower pack without a second copy to compare against.
    pub is_override: bool,
}

/// A source-loading/validation failure with the same program/stage/location breadcrumbs that
/// Java's `ShaderLoader`/`JsonException` chain exposes to the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderSourceError {
    /// A `program/*.json` entry named a stage but the corresponding source was absent.
    Missing {
        /// Program name whose manifest referenced the missing source.
        program: String,
        /// Missing stage.
        stage: ShaderStage,
        /// Exact source location.
        location: ResourceLocation,
    },
    /// A stage name was absent from the program manifest.
    MissingStageName {
        /// Program name whose manifest was incomplete.
        program: String,
        /// Stage whose name was absent.
        stage: ShaderStage,
    },
    /// The resource manager could not read an otherwise-present source.
    Read {
        /// Program name being resolved.
        program: String,
        /// Stage being read.
        stage: ShaderStage,
        /// Exact source location.
        location: ResourceLocation,
        /// Underlying resource-manager message.
        message: String,
    },
    /// Source bytes were not valid UTF-8. Java hands raw bytes to OpenGL; this backend refuses an
    /// ambiguous replacement-decoded string so failures are deterministic.
    InvalidUtf8 {
        /// Program name being resolved.
        program: String,
        /// Stage being validated.
        stage: ShaderStage,
        /// Exact source location.
        location: ResourceLocation,
    },
    /// The bytes decoded as text but failed the small, deterministic GLSL sanity validator.
    Malformed {
        /// Program name being resolved.
        program: String,
        /// Stage being validated.
        stage: ShaderStage,
        /// Exact source location.
        location: ResourceLocation,
        /// Validation reason.
        reason: String,
    },
}

impl fmt::Display for ShaderSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing {
                program,
                stage,
                location,
            } => write!(
                f,
                "shader program {program} {stage} source {location} is missing"
            ),
            Self::MissingStageName { program, stage } => {
                write!(f, "shader program {program} has no {stage} source name")
            }
            Self::Read {
                program,
                stage,
                location,
                message,
            } => write!(
                f,
                "shader program {program} {stage} source {location} could not be read: {message}"
            ),
            Self::InvalidUtf8 {
                program,
                stage,
                location,
            } => write!(
                f,
                "shader program {program} {stage} source {location} is not valid UTF-8"
            ),
            Self::Malformed {
                program,
                stage,
                location,
                reason,
            } => write!(
                f,
                "couldn't compile {stage} program {program} from {location}: {reason}"
            ),
        }
    }
}

impl std::error::Error for ShaderSourceError {}

/// The vertex/fragment source pair resolved from one `program/*.json` manifest.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProgramSourceResolution {
    /// The resolved vertex source, when this program has resource-backed sources.
    pub vertex: Option<ShaderSourceInfo>,
    /// The resolved fragment source, when this program has resource-backed sources.
    pub fragment: Option<ShaderSourceInfo>,
}

impl ProgramSourceResolution {
    /// Whether either stage was supplied by the active resource-pack stack.
    #[must_use]
    pub fn has_sources(&self) -> bool {
        self.vertex.is_some() || self.fragment.is_some()
    }

    /// Whether either stage is proven to shadow a lower-priority pack.
    #[must_use]
    pub fn has_overrides(&self) -> bool {
        self.vertex
            .as_ref()
            .is_some_and(|source| source.is_override)
            || self
                .fragment
                .as_ref()
                .is_some_and(|source| source.is_override)
    }

    /// Iterate over the present stages in deterministic vertex-then-fragment order.
    pub fn iter(&self) -> impl Iterator<Item = &ShaderSourceInfo> {
        self.vertex.iter().chain(self.fragment.iter())
    }
}

fn shader_source_location(name: &str, stage: ShaderStage) -> ResourceLocation {
    // java: ShaderLoader.loadShader constructs `new ResourceLocation("shaders/program/"
    // + filename + extension)`, whose first-colon split is significant for malformed/custom
    // names.  Do not force every filename into the minecraft namespace here.
    ResourceLocation::parse(&format!("shaders/program/{name}{}", stage.extension()))
        .expect("constructed shader source location")
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

/// Check the source properties that can be validated without claiming GLSL-to-WGSL support.
///
/// This is intentionally a small lexical check rather than a fake compiler: it catches empty,
/// binary, unterminated, unbalanced, and missing-entry-point sources with stable diagnostics, but
/// does not assert that a GLSL dialect accepted by OpenGL is accepted by naga/wgpu.
fn validate_glsl_source(text: &str) -> Result<(), String> {
    if text.trim().is_empty() {
        return Err("source is empty".to_owned());
    }

    let bytes = text.as_bytes();
    let mut stack: Vec<(u8, usize)> = Vec::new();
    let mut previous_identifier: Option<&str> = None;
    let mut has_main = false;
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'/' if bytes.get(index + 1) == Some(&b'/') => {
                index += 2;
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
            }
            b'/' if bytes.get(index + 1) == Some(&b'*') => {
                let start = index;
                index += 2;
                let mut closed = false;
                while index + 1 < bytes.len() {
                    if bytes[index] == b'*' && bytes[index + 1] == b'/' {
                        index += 2;
                        closed = true;
                        break;
                    }
                    index += 1;
                }
                if !closed {
                    return Err(format!("unterminated block comment at byte {start}"));
                }
            }
            b'"' | b'\'' => {
                let quote = bytes[index];
                let start = index;
                index += 1;
                let mut escaped = false;
                let mut closed = false;
                while index < bytes.len() {
                    let byte = bytes[index];
                    index += 1;
                    if escaped {
                        escaped = false;
                    } else if byte == b'\\' {
                        escaped = true;
                    } else if byte == quote {
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    return Err(format!("unterminated string at byte {start}"));
                }
            }
            b'{' | b'(' | b'[' => {
                stack.push((bytes[index], index));
                index += 1;
            }
            b'}' | b')' | b']' => {
                let expected = match bytes[index] {
                    b'}' => b'{',
                    b')' => b'(',
                    _ => b'[',
                };
                match stack.pop() {
                    Some((open, _)) if open == expected => index += 1,
                    Some((open, at)) => {
                        return Err(format!(
                            "mismatched delimiter '{}' at byte {} (opened '{}' at byte {at})",
                            bytes[index] as char, index, open as char
                        ));
                    }
                    None => {
                        return Err(format!("unmatched delimiter at byte {index}"));
                    }
                }
            }
            byte if byte.is_ascii_alphabetic() || byte == b'_' => {
                let start = index;
                index += 1;
                while index < bytes.len()
                    && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
                {
                    index += 1;
                }
                let identifier = &text[start..index];
                if previous_identifier == Some("void") && identifier == "main" {
                    has_main = text[index..]
                        .trim_start_matches(char::is_whitespace)
                        .starts_with('(');
                }
                previous_identifier = Some(identifier);
            }
            _ => {
                if !bytes[index].is_ascii_whitespace() {
                    previous_identifier = None;
                }
                index += 1;
            }
        }
    }

    if let Some((open, at)) = stack.last().copied() {
        return Err(format!(
            "unclosed delimiter '{}' at byte {at}",
            open as char
        ));
    }
    if !has_main {
        return Err("missing `void main(...)` entry point".to_owned());
    }
    Ok(())
}

fn resolve_shader_source(
    rm: &ResourceManager,
    program: &str,
    stage: ShaderStage,
    name: &str,
) -> Result<ShaderSourceInfo, ShaderSourceError> {
    let location = shader_source_location(name, stage);
    if !rm.contains(&location) {
        return Err(ShaderSourceError::Missing {
            program: program.to_owned(),
            stage,
            location,
        });
    }
    let resource = rm
        .resource(&location)
        .map_err(|error| ShaderSourceError::Read {
            program: program.to_owned(),
            stage,
            location: location.clone(),
            message: error.to_string(),
        })?;
    let text = std::str::from_utf8(&resource.data).map_err(|_| ShaderSourceError::InvalidUtf8 {
        program: program.to_owned(),
        stage,
        location: location.clone(),
    })?;
    if resource.data.contains(&0) {
        return Err(ShaderSourceError::Malformed {
            program: program.to_owned(),
            stage,
            location,
            reason: "source contains a NUL byte".to_owned(),
        });
    }
    validate_glsl_source(text).map_err(|reason| ShaderSourceError::Malformed {
        program: program.to_owned(),
        stage,
        location: location.clone(),
        reason,
    })?;
    let is_override = rm.all_resources(&location).len() > 1;
    Ok(ShaderSourceInfo {
        stage,
        location,
        byte_len: resource.data.len(),
        fingerprint: fnv1a(&resource.data),
        pack_name: resource.pack_name,
        is_override,
    })
}

fn err(msg: impl Into<String>) -> JsonError {
    JsonError {
        message: msg.into(),
        path: Vec::new(),
    }
}

fn parse_indexed<T>(
    values: &[Json],
    label: &str,
    mut parse: impl FnMut(&Json) -> Result<T, JsonError>,
) -> Result<Vec<T>, JsonError> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            parse(value).map_err(|error| error.prepend_path(format!("{label}[{index}]")))
        })
        .collect()
}

/// How deeply arrays and objects may nest (shipped groups nest 5
/// deep). The reader recurses per level, and these files come from resource packs: 200,000
/// opening brackets overflowed the stack, which ends the process at every resource load.
const MAX_JSON_DEPTH: usize = 128;

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            bytes: s.as_bytes(),
            pos: 0,
            depth: 0,
        }
    }

    /// One level deeper for `parse` (an array or object), refused past [`MAX_JSON_DEPTH`].
    fn nested(
        &mut self,
        parse: fn(&mut Self) -> Result<Json, JsonError>,
    ) -> Result<Json, JsonError> {
        if self.depth >= MAX_JSON_DEPTH {
            return Err(err(format!(
                "nested deeper than {MAX_JSON_DEPTH} at byte {}",
                self.pos
            )));
        }
        self.depth += 1;
        let value = parse(self);
        self.depth -= 1;
        value
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), JsonError> {
        if self.bump() == Some(b) {
            Ok(())
        } else {
            Err(err(format!(
                "expected '{}' at byte {}",
                b as char, self.pos
            )))
        }
    }

    fn parse_value(&mut self) -> Result<Json, JsonError> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.nested(Self::parse_object),
            Some(b'[') => self.nested(Self::parse_array),
            Some(b'"') => Ok(Json::String(self.parse_string()?)),
            Some(b't') => self.parse_lit("true", Json::Bool(true)),
            Some(b'f') => self.parse_lit("false", Json::Bool(false)),
            Some(b'n') => self.parse_lit("null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            _ => Err(err(format!("unexpected byte at {}", self.pos))),
        }
    }

    fn parse_lit(&mut self, lit: &str, value: Json) -> Result<Json, JsonError> {
        for expected in lit.bytes() {
            if self.bump() != Some(expected) {
                return Err(err(format!("expected literal {lit:?}")));
            }
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<Json, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| err("non-utf8 number"))?;
        text.parse::<f64>()
            .map(Json::Number)
            .map_err(|_| err(format!("bad number literal {text:?}")))
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            match self.bump().ok_or_else(|| err("unterminated string"))? {
                b'"' => return Ok(out),
                b'\\' => match self.bump().ok_or_else(|| err("unterminated escape"))? {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    b'b' => out.push('\u{8}'),
                    b'f' => out.push('\u{c}'),
                    b'u' => {
                        // \uXXXX is rare in these assets (no non-ASCII shader names); handled as
                        // a literal codepoint without surrogate-pair joining.
                        let hex = std::str::from_utf8(
                            self.bytes
                                .get(self.pos..self.pos + 4)
                                .ok_or_else(|| err("truncated \\u escape"))?,
                        )
                        .map_err(|_| err("non-utf8 \\u escape"))?;
                        let code =
                            u32::from_str_radix(hex, 16).map_err(|_| err("bad \\u escape"))?;
                        out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                        self.pos += 4;
                    }
                    other => return Err(err(format!("bad escape '{}'", other as char))),
                },
                b => {
                    // Re-decode from the byte boundary so multi-byte UTF-8 survives intact.
                    let start = self.pos - 1;
                    let ch_len = utf8_len(b);
                    let slice = self
                        .bytes
                        .get(start..start + ch_len)
                        .ok_or_else(|| err("truncated utf-8"))?;
                    out.push_str(std::str::from_utf8(slice).map_err(|_| err("invalid utf-8"))?);
                    self.pos = start + ch_len;
                }
            }
        }
    }

    fn parse_array(&mut self) -> Result<Json, JsonError> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Array(out));
        }
        loop {
            out.push(self.parse_value()?);
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => return Ok(Json::Array(out)),
                _ => return Err(err("expected ',' or ']' in array")),
            }
        }
    }

    fn parse_object(&mut self) -> Result<Json, JsonError> {
        self.expect(b'{')?;
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Object(out));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            let value = self.parse_value()?;
            out.push((key, value));
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => return Ok(Json::Object(out)),
                _ => return Err(err("expected ',' or '}' in object")),
            }
        }
    }
}

fn utf8_len(first_byte: u8) -> usize {
    if first_byte & 0x80 == 0 {
        1
    } else if first_byte & 0xE0 == 0xC0 {
        2
    } else if first_byte & 0xF0 == 0xE0 {
        3
    } else {
        4
    }
}

impl Json {
    /// Parse a complete JSON document.
    ///
    /// # Errors
    /// [`JsonError`] if the text is not well-formed JSON.
    pub fn parse(text: &str) -> Result<Json, JsonError> {
        let mut p = Parser::new(text);
        let value = p.parse_value()?;
        p.skip_ws();
        if p.pos != p.bytes.len() {
            return Err(err("trailing data after top-level value"));
        }
        Ok(value)
    }

    fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(o) => Some(o),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }
}

/// `"minecraft:main"` — the literal name for the game's own framebuffer.
pub const MAIN_TARGET: &str = "minecraft:main";

/// One `targets[]` entry. java: `ShaderGroup.initTarget`
#[derive(Debug, Clone, PartialEq)]
pub enum TargetSpec {
    /// A bare string (`"swap"`): sized to the main framebuffer, resized with the window.
    Named(String),
    /// An explicit `{"name", "width", "height"}` object. `width`/`height` default to the main
    /// framebuffer's size when omitted, exactly as `JsonUtils.getInt(.., default)` does.
    Sized {
        /// Target name, referenced by passes' `intarget`/`outtarget`/`auxtargets`.
        name: String,
        /// Fixed width, or `None` to track the main framebuffer.
        width: Option<u32>,
        /// Fixed height, or `None` to track the main framebuffer.
        height: Option<u32>,
    },
}

impl TargetSpec {
    /// The target's name, regardless of which JSON shape it was.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            TargetSpec::Named(n) => n,
            TargetSpec::Sized { name, .. } => name,
        }
    }

    fn parse(value: &Json) -> Result<Self, JsonError> {
        if let Some(s) = value.as_str() {
            return Ok(TargetSpec::Named(s.to_owned()));
        }
        // `JsonUtils.getJsonObject(value, "target")` consumes `value` itself; its second
        // argument is only an error-message label. Vanilla therefore expects this object flat.
        // Keep accepting the old wrapper as a harmless compatibility extension.
        let target = value.get("target").unwrap_or(value);
        let name = target
            .get("name")
            .and_then(Json::as_str)
            .ok_or_else(|| err("target.name missing"))?
            .to_owned();
        let width = target.get("width").and_then(Json::as_f64).map(|n| n as u32);
        let height = target
            .get("height")
            .and_then(Json::as_f64)
            .map(|n| n as u32);
        Ok(TargetSpec::Sized {
            name,
            width,
            height,
        })
    }
}

/// One `auxtargets[]` entry. java: `ShaderGroup.parsePass`'s `auxtargets` handling.
#[derive(Debug, Clone, PartialEq)]
pub struct AuxTargetSpec {
    /// The sampler name this is bound to in the program (e.g. `"DitherSampler"`).
    pub name: String,
    /// Either another declared target's name (its colour texture is bound), or — when it does
    /// not match any target — a static PNG at `textures/effect/<id>.png`.
    pub id: String,
    /// Declared width, used when `id` resolves to a static texture.
    pub width: Option<u32>,
    /// Declared height, used when `id` resolves to a static texture.
    pub height: Option<u32>,
    /// Bilinear vs nearest filtering for a static aux texture.
    pub bilinear: bool,
    /// Whether the static-texture JSON supplied the Java-required `bilinear` member. Framebuffer
    /// aux targets never consult this field.
    pub(crate) bilinear_specified: bool,
}

impl AuxTargetSpec {
    fn parse(value: &Json) -> Result<Self, JsonError> {
        // `JsonUtils.getJsonObject(value, "auxtarget")` consumes `value` itself; "auxtarget"
        // is only an error-message label. Vanilla and its shipped files therefore use the flat
        // shape. Keep accepting the wrapper as a harmless compatibility extension.
        let aux = value.get("auxtarget").unwrap_or(value);
        let bilinear = aux
            .get("bilinear")
            .map(|value| {
                value
                    .as_bool()
                    .ok_or_else(|| err("auxtarget.bilinear must be a boolean"))
            })
            .transpose()?;
        Ok(Self {
            name: aux
                .get("name")
                .and_then(Json::as_str)
                .ok_or_else(|| err("auxtarget.name missing"))?
                .to_owned(),
            id: aux
                .get("id")
                .and_then(Json::as_str)
                .ok_or_else(|| err("auxtarget.id missing"))?
                .to_owned(),
            width: aux.get("width").and_then(Json::as_f64).map(|n| n as u32),
            height: aux.get("height").and_then(Json::as_f64).map(|n| n as u32),
            bilinear: bilinear.unwrap_or(false),
            bilinear_specified: bilinear.is_some(),
        })
    }
}

/// One `uniforms[]` entry inside a pass. java: `ShaderGroup.initUniform`
#[derive(Debug, Clone, PartialEq)]
pub struct UniformOverride {
    /// The uniform's name, matched against the program's own declared uniforms.
    pub name: String,
    /// 1-4 float components.
    pub values: Vec<f32>,
}

impl UniformOverride {
    fn parse(value: &Json) -> Result<Self, JsonError> {
        // `JsonUtils.getJsonObject(value, "uniform")` consumes `value` itself; "uniform" is
        // only an error-message label. Vanilla and its shipped files therefore use the flat
        // shape. Keep accepting the wrapper as a harmless compatibility extension.
        let obj = value.get("uniform").unwrap_or(value);
        let name = obj
            .get("name")
            .and_then(Json::as_str)
            .ok_or_else(|| err("uniform.name missing"))?
            .to_owned();
        let values = obj
            .get("values")
            .and_then(Json::as_array)
            .ok_or_else(|| err("uniform.values missing"))?;
        let values = parse_indexed(values, "values", |value| {
            value
                .as_f64()
                .map(|number| number as f32)
                .ok_or_else(|| err("uniform value must be a number"))
        })?;
        Ok(Self { name, values })
    }
}

/// One `passes[]` entry. java: `ShaderGroup.parsePass`
#[derive(Debug, Clone, PartialEq)]
pub struct PassSpec {
    /// Which `program/<name>.json` this pass runs.
    pub name: String,
    /// Input target name, or the literal `"minecraft:main"`.
    pub intarget: String,
    /// Output target name, or the literal `"minecraft:main"`.
    pub outtarget: String,
    /// Extra named samplers this pass reads besides the implicit `DiffuseSampler` input.
    pub auxtargets: Vec<AuxTargetSpec>,
    /// Uniform value overrides for this specific pass instance.
    pub uniforms: Vec<UniformOverride>,
}

impl PassSpec {
    fn parse(value: &Json) -> Result<Self, JsonError> {
        // `JsonUtils.getJsonObject(value, "pass")` consumes `value` itself; "pass" is only an
        // error-message label. The fallback merely preserves our older wrapper extension.
        let obj = value.get("pass").unwrap_or(value);
        let name = obj
            .get("name")
            .and_then(Json::as_str)
            .ok_or_else(|| err("pass.name missing"))?
            .to_owned();
        let intarget = obj
            .get("intarget")
            .and_then(Json::as_str)
            .ok_or_else(|| err("pass.intarget missing"))?
            .to_owned();
        let outtarget = obj
            .get("outtarget")
            .and_then(Json::as_str)
            .ok_or_else(|| err("pass.outtarget missing"))?
            .to_owned();
        let auxtargets = obj
            .get("auxtargets")
            .and_then(Json::as_array)
            .map(|a| parse_indexed(a, "auxtargets", AuxTargetSpec::parse))
            .transpose()?
            .unwrap_or_default();
        let uniforms = obj
            .get("uniforms")
            .and_then(Json::as_array)
            .map(|a| parse_indexed(a, "uniforms", UniformOverride::parse))
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            name,
            intarget,
            outtarget,
            auxtargets,
            uniforms,
        })
    }
}

/// A `post/*.json` shader group: named framebuffer targets plus an ordered pass chain.
/// java: `ShaderGroup.parseGroup`'s `targets`/`passes` top-level fields.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderGroupSpec {
    /// Declared framebuffer targets, `"swap"`-style bare names or `{"name", ...}` objects.
    pub targets: Vec<TargetSpec>,
    /// The ordered pass chain. Execution order is JSON array order — there is no separate
    /// dependency solver in vanilla, and none here either.
    pub passes: Vec<PassSpec>,
}

impl ShaderGroupSpec {
    /// Parse a `post/*.json` document.
    ///
    /// # Errors
    /// [`JsonError`] if the document is not valid JSON or does not match the expected shape.
    pub fn parse(text: &str) -> Result<Self, JsonError> {
        let root = Json::parse(text)?;
        let targets = root
            .get("targets")
            .and_then(Json::as_array)
            .map(|a| parse_indexed(a, "targets", TargetSpec::parse))
            .transpose()?
            .unwrap_or_default();
        let passes = root
            .get("passes")
            .and_then(Json::as_array)
            .map(|a| parse_indexed(a, "passes", PassSpec::parse))
            .transpose()?
            .unwrap_or_default();
        Ok(Self { targets, passes })
    }
}

/// java: `ShaderManager.parseSampler` — `{"name", "file"}` (`file` absent = a framebuffer input
/// bound at pass-record time, e.g. the implicit `DiffuseSampler`).
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerSpec {
    /// Sampler name as declared in the GLSL source.
    pub name: String,
    /// A static texture path, when this sampler is not fed by a framebuffer/aux target.
    pub file: Option<String>,
}

/// java: `ShaderManager.parseUniform` / `ShaderUniform.parseType`.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformDecl {
    /// Uniform name.
    pub name: String,
    /// `"int"`, `"float"`, or `"matrixNxN"`.
    pub kind: String,
    /// Component count (1-4 for vectors; 4/9/16 for matrices).
    pub count: u32,
    /// Initial values.
    pub values: Vec<f32>,
}

/// A GL blend equation resolved by vanilla's `JsonBlendingMode.func_148108_a`.
///
/// The discriminants are the OpenGL constants vanilla passes to `glBlendEquation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum BlendEquation {
    /// `GL_FUNC_ADD` / the fallback for an unknown equation name.
    #[default]
    Add = 32_774,
    /// `GL_FUNC_SUBTRACT`.
    Subtract = 32_778,
    /// `GL_FUNC_REVERSE_SUBTRACT`.
    ReverseSubtract = 32_779,
    /// `GL_MIN`.
    Min = 32_775,
    /// `GL_MAX`.
    Max = 32_776,
}

impl BlendEquation {
    /// The OpenGL value used by 1.8.9.
    #[must_use]
    pub const fn gl_enum(self) -> i32 {
        self as i32
    }
}

/// A GL blend factor resolved by vanilla's `JsonBlendingMode.func_148107_b`.
///
/// An unrecognised string deliberately becomes [`Self::Invalid`]: vanilla returns `-1` and lets
/// the subsequent GL call report `GL_INVALID_ENUM`; it does not silently substitute a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum BlendFactor {
    /// `GL_ZERO`.
    Zero = 0,
    /// `GL_ONE`.
    #[default]
    One = 1,
    /// `GL_SRC_COLOR`.
    SrcColor = 768,
    /// `GL_ONE_MINUS_SRC_COLOR`.
    OneMinusSrcColor = 769,
    /// `GL_DST_COLOR`.
    DstColor = 774,
    /// `GL_ONE_MINUS_DST_COLOR`.
    OneMinusDstColor = 775,
    /// `GL_SRC_ALPHA`.
    SrcAlpha = 770,
    /// `GL_ONE_MINUS_SRC_ALPHA`.
    OneMinusSrcAlpha = 771,
    /// `GL_DST_ALPHA`.
    DstAlpha = 772,
    /// `GL_ONE_MINUS_DST_ALPHA`.
    OneMinusDstAlpha = 773,
    /// Vanilla's `-1` result for a name outside its ten recognised factor aliases.
    Invalid = -1,
}

impl BlendFactor {
    /// The OpenGL value used by 1.8.9 (or `-1` for an invalid resource-pack factor name).
    #[must_use]
    pub const fn gl_enum(self) -> i32 {
        self as i32
    }
}

/// java: `ShaderManager`'s `blend` block (`func`/`srcrgb`/`dstrgb`/`srcalpha`/`dstalpha`), after
/// `JsonBlendingMode.func_148110_a` has resolved all aliases and collapsed default state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendSpec {
    /// `false` when the five resolved values are exactly vanilla's defaults, which makes vanilla
    /// call `GlStateManager.disableBlend()` even when an explicit `blend` object was present.
    pub enabled: bool,
    /// Whether the enabled mode calls `tryBlendFuncSeparate` rather than `blendFunc`.
    pub separate_alpha: bool,
    /// Blend equation (`func`, default `add`; unknown names also resolve to `add`).
    pub func: BlendEquation,
    /// Source RGB factor (`srcrgb`, default `one`).
    pub srcrgb: BlendFactor,
    /// Destination RGB factor (`dstrgb`, default `zero`).
    pub dstrgb: BlendFactor,
    /// Source alpha factor (`srcalpha`, default `one`).
    pub srcalpha: BlendFactor,
    /// Destination alpha factor (`dstalpha`, default `zero`).
    pub dstalpha: BlendFactor,
}

impl Default for BlendSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            separate_alpha: false,
            func: BlendEquation::Add,
            srcrgb: BlendFactor::One,
            dstrgb: BlendFactor::Zero,
            srcalpha: BlendFactor::One,
            dstalpha: BlendFactor::Zero,
        }
    }
}

impl BlendSpec {
    /// Alpha factors for an enabled separate-alpha blend; `None` means either blending is
    /// disabled or vanilla takes the ordinary two-factor `blendFunc` path.
    #[must_use]
    pub const fn separate_alpha_factors(&self) -> Option<(BlendFactor, BlendFactor)> {
        if self.enabled && self.separate_alpha {
            Some((self.srcalpha, self.dstalpha))
        } else {
            None
        }
    }

    fn parse(value: &Json) -> Result<Self, JsonError> {
        // `JsonUtils.getJsonObject(root, "blend", null)` throws for a present non-object; it
        // does not treat that malformed member like an absent blend declaration.
        if value.as_object().is_none() {
            return Err(err("blend must be an object"));
        }

        let func = value
            .get("func")
            .and_then(Json::as_str)
            .map(blend_equation_from_name)
            .unwrap_or_default();
        let srcrgb = value
            .get("srcrgb")
            .and_then(Json::as_str)
            .map(blend_factor_from_name)
            .unwrap_or(BlendFactor::One);
        let dstrgb = value
            .get("dstrgb")
            .and_then(Json::as_str)
            .map(blend_factor_from_name)
            .unwrap_or(BlendFactor::Zero);
        let srcalpha_name = value.get("srcalpha").and_then(Json::as_str);
        let dstalpha_name = value.get("dstalpha").and_then(Json::as_str);
        let srcalpha = srcalpha_name
            .map(blend_factor_from_name)
            .unwrap_or(BlendFactor::One);
        let dstalpha = dstalpha_name
            .map(blend_factor_from_name)
            .unwrap_or(BlendFactor::Zero);
        let enabled = func != BlendEquation::Add
            || srcrgb != BlendFactor::One
            || dstrgb != BlendFactor::Zero
            || srcalpha != BlendFactor::One
            || dstalpha != BlendFactor::Zero;

        Ok(Self {
            enabled,
            // Vanilla sets this only when a *string* alpha member is present.  If all five
            // values are defaults it returns its disabled default mode before this flag matters.
            separate_alpha: enabled && (srcalpha_name.is_some() || dstalpha_name.is_some()),
            func,
            srcrgb,
            dstrgb,
            srcalpha,
            dstalpha,
        })
    }
}

/// Equation names accepted by `JsonBlendingMode`: trim, lowercase, then match its six aliases;
/// an unknown spelling falls back to `add`.
fn blend_equation_from_name(name: &str) -> BlendEquation {
    match name.trim().to_ascii_lowercase().as_str() {
        "add" => BlendEquation::Add,
        "subtract" => BlendEquation::Subtract,
        "reversesubtract" | "reverse_subtract" => BlendEquation::ReverseSubtract,
        "min" => BlendEquation::Min,
        "max" => BlendEquation::Max,
        _ => BlendEquation::Add,
    }
}

/// Vanilla's factor-name normalisation: `trim`, lowercase, then `_`, `one`, `zero`, `minus`
/// replacements in that exact order.  Each replacement is global, just like Java `replaceAll`.
fn blend_factor_from_name(name: &str) -> BlendFactor {
    let normalized = name
        .trim()
        .to_lowercase()
        .replace('_', "")
        .replace("one", "1")
        .replace("zero", "0")
        .replace("minus", "-");
    match normalized.as_str() {
        "0" => BlendFactor::Zero,
        "1" => BlendFactor::One,
        "srccolor" => BlendFactor::SrcColor,
        "1-srccolor" => BlendFactor::OneMinusSrcColor,
        "dstcolor" => BlendFactor::DstColor,
        "1-dstcolor" => BlendFactor::OneMinusDstColor,
        "srcalpha" => BlendFactor::SrcAlpha,
        "1-srcalpha" => BlendFactor::OneMinusSrcAlpha,
        "dstalpha" => BlendFactor::DstAlpha,
        "1-dstalpha" => BlendFactor::OneMinusDstAlpha,
        _ => BlendFactor::Invalid,
    }
}

/// A `program/*.json` file: one pass's GLSL program description.
/// java: `ShaderManager`'s constructor (field names `vertex`, `fragment`, `attributes`,
/// `samplers`, `uniforms`, `blend`, `cull`).
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramSpec {
    /// `.vsh` file stem (no extension), e.g. `"sobel"`.
    pub vertex: Option<String>,
    /// `.fsh` file stem (no extension).
    pub fragment: Option<String>,
    /// Declared vertex attribute names.
    pub attributes: Vec<String>,
    /// Declared texture samplers.
    pub samplers: Vec<SamplerSpec>,
    /// Declared uniforms with their initial values.
    pub uniforms: Vec<UniformDecl>,
    /// Blend function, if this pass blends rather than replaces.
    pub blend: Option<BlendSpec>,
    /// Whether back-face culling is enabled. Vanilla defaults this to `true`.
    pub cull: bool,
}

impl Default for ProgramSpec {
    fn default() -> Self {
        Self {
            vertex: None,
            fragment: None,
            attributes: Vec::new(),
            samplers: Vec::new(),
            uniforms: Vec::new(),
            blend: None,
            cull: true,
        }
    }
}

impl ProgramSpec {
    /// Parse a `program/*.json` document.
    ///
    /// # Errors
    /// [`JsonError`] if the document is not valid JSON or does not match the expected shape.
    pub fn parse(text: &str) -> Result<Self, JsonError> {
        let root = Json::parse(text)?;
        let vertex = root.get("vertex").and_then(Json::as_str).map(str::to_owned);
        let fragment = root
            .get("fragment")
            .and_then(Json::as_str)
            .map(str::to_owned);
        let attributes = root
            .get("attributes")
            .and_then(Json::as_array)
            .map(|a| {
                parse_indexed(a, "attributes", |value| {
                    value
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| err("attribute must be a string"))
                })
            })
            .transpose()?
            .unwrap_or_default();
        let samplers = root
            .get("samplers")
            .and_then(Json::as_array)
            .map(|a| {
                parse_indexed(a, "samplers", |s| {
                    Ok(SamplerSpec {
                        name: s
                            .get("name")
                            .and_then(Json::as_str)
                            .ok_or_else(|| err("sampler.name missing"))?
                            .to_owned(),
                        file: s.get("file").and_then(Json::as_str).map(str::to_owned),
                    })
                })
            })
            .transpose()?
            .unwrap_or_default();
        let uniforms = root
            .get("uniforms")
            .and_then(Json::as_array)
            .map(|a| {
                parse_indexed(a, "uniforms", |u| {
                    let values_json = u
                        .get("values")
                        .and_then(Json::as_array)
                        .ok_or_else(|| err("uniform.values missing"))?;
                    let mut values = parse_indexed(values_json, "values", |value| {
                        value
                            .as_f64()
                            .map(|number| number as f32)
                            .ok_or_else(|| err("uniform value must be a number"))
                    })?;
                    let count_number = u
                        .get("count")
                        .and_then(Json::as_f64)
                        .ok_or_else(|| err("uniform.count missing"))?;
                    if count_number < 0.0
                        || count_number > f64::from(u32::MAX)
                        || count_number.fract() != 0.0
                    {
                        return Err(err("uniform.count must be a non-negative integer"));
                    }
                    let count = count_number as u32;
                    // ShaderManager accepts exactly `count` values, or a single value which it
                    // broadcasts across a vector/matrix. Zero values are retained as Java does;
                    // ShaderGroup's per-pass override path treats an empty array as a no-op.
                    if values.len() != count as usize && values.len() > 1 {
                        return Err(err(format!(
                            "invalid amount of values specified (expected {}, found {})",
                            count,
                            values.len()
                        )));
                    }
                    if count > 1 && values.len() == 1 {
                        values.resize(count as usize, values[0]);
                    }
                    Ok(UniformDecl {
                        name: u
                            .get("name")
                            .and_then(Json::as_str)
                            .ok_or_else(|| err("uniform.name missing"))?
                            .to_owned(),
                        kind: u
                            .get("type")
                            .and_then(Json::as_str)
                            .ok_or_else(|| err("uniform.type missing"))?
                            .to_owned(),
                        count,
                        values,
                    })
                })
            })
            .transpose()?
            .unwrap_or_default();
        let blend = root
            .get("blend")
            .map(|value| BlendSpec::parse(value).map_err(|error| error.prepend_path("blend")))
            .transpose()?;
        let cull = root.get("cull").and_then(Json::as_bool).unwrap_or(true);
        Ok(Self {
            vertex,
            fragment,
            attributes,
            samplers,
            uniforms,
            blend,
            cull,
        })
    }

    /// Resolve the Java `.vsh`/`.fsh` resources named by this manifest.
    ///
    /// A manifest must name both stages, and both named resources are loaded and validated. This
    /// mirrors Java's `ShaderManager` constructor, which calls `JsonUtils.getString` for both
    /// fields before `ShaderLoader.loadShader` opens either resource. The baked WGSL path stays
    /// usable without a vanilla resource manager because callers skip this method when the
    /// manifest itself is absent.
    ///
    /// This method does not translate or execute GLSL. A caller that needs to run a chain on the
    /// WGSL backend must treat a non-empty result as an explicit source override and decide how to
    /// report the unsupported language; [`super::pipeline::PostChain::try_new`] does exactly that.
    ///
    /// # Errors
    /// [`ShaderSourceError`] when a stage name is absent, a named source is missing/cannot be read,
    /// or a source fails deterministic lexical validation.
    pub fn resolve_shader_sources(
        &self,
        rm: &ResourceManager,
        program_name: &str,
    ) -> Result<ProgramSourceResolution, ShaderSourceError> {
        let vertex_name =
            self.vertex
                .as_deref()
                .ok_or_else(|| ShaderSourceError::MissingStageName {
                    program: program_name.to_owned(),
                    stage: ShaderStage::Vertex,
                })?;
        let fragment_name =
            self.fragment
                .as_deref()
                .ok_or_else(|| ShaderSourceError::MissingStageName {
                    program: program_name.to_owned(),
                    stage: ShaderStage::Fragment,
                })?;

        Ok(ProgramSourceResolution {
            vertex: Some(resolve_shader_source(
                rm,
                program_name,
                ShaderStage::Vertex,
                vertex_name,
            )?),
            fragment: Some(resolve_shader_source(
                rm,
                program_name,
                ShaderStage::Fragment,
                fragment_name,
            )?),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::{Cursor, Read};

    use mc_assets::{ResourceManager, ResourcePack};

    use super::*;

    struct SourcePack {
        name: String,
        files: HashMap<String, Vec<u8>>,
    }

    impl SourcePack {
        fn new(name: &str, files: impl IntoIterator<Item = (&'static str, Vec<u8>)>) -> Self {
            Self {
                name: name.to_owned(),
                files: files
                    .into_iter()
                    .map(|(path, bytes)| (format!("assets/minecraft/{path}"), bytes))
                    .collect(),
            }
        }
    }

    impl ResourcePack for SourcePack {
        fn open(&self, location: &ResourceLocation) -> Option<Box<dyn Read>> {
            self.files
                .get(&location.pack_entry_name())
                .cloned()
                .map(|bytes| Box::new(Cursor::new(bytes)) as Box<dyn Read>)
        }

        fn contains(&self, location: &ResourceLocation) -> bool {
            self.files.contains_key(&location.pack_entry_name())
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn namespaces(&self) -> Vec<String> {
            vec!["minecraft".to_owned()]
        }
    }

    fn source_program() -> ProgramSpec {
        ProgramSpec::parse(r#"{"vertex":"override","fragment":"override"}"#).unwrap()
    }

    #[test]
    fn shader_source_locations_use_java_first_colon_split() {
        let vertex = shader_source_location("custom:stage", ShaderStage::Vertex);
        assert_eq!(vertex.namespace, "shaders/program/custom");
        assert_eq!(vertex.path, "stage.vsh");

        let fragment = shader_source_location("custom:stage", ShaderStage::Fragment);
        assert_eq!(fragment.namespace, "shaders/program/custom");
        assert_eq!(fragment.path, "stage.fsh");
    }

    #[test]
    fn namespaced_shader_filename_reads_the_java_pack_entry() {
        let pack = SourcePack {
            name: "namespaced.zip".to_owned(),
            files: HashMap::from([
                (
                    "assets/shaders/program/custom/stage.vsh".to_owned(),
                    valid_vertex(),
                ),
                (
                    "assets/shaders/program/custom/stage.fsh".to_owned(),
                    valid_fragment(),
                ),
            ]),
        };
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(pack));
        let program =
            ProgramSpec::parse(r#"{"vertex":"custom:stage","fragment":"custom:stage"}"#).unwrap();
        let resolved = program
            .resolve_shader_sources(&rm, "custom_program")
            .unwrap();
        assert_eq!(resolved.vertex.unwrap().pack_name, "namespaced.zip");
        assert_eq!(resolved.fragment.unwrap().pack_name, "namespaced.zip");
    }

    fn valid_vertex() -> Vec<u8> {
        b"#version 120\nvoid main() { gl_Position = vec4(0.0); }".to_vec()
    }

    fn valid_fragment() -> Vec<u8> {
        b"#version 120\nvoid main() { gl_FragColor = vec4(1.0); }".to_vec()
    }

    #[test]
    fn shader_sources_follow_java_locations_and_expose_pack_fingerprint() {
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(SourcePack::new(
            "pack.zip",
            [
                ("shaders/program/override.vsh", valid_vertex()),
                ("shaders/program/override.fsh", valid_fragment()),
            ],
        )));

        let resolved = source_program()
            .resolve_shader_sources(&rm, "custom_program")
            .unwrap();
        assert_eq!(resolved.vertex.as_ref().unwrap().stage, ShaderStage::Vertex);
        assert_eq!(
            resolved.vertex.as_ref().unwrap().location.path,
            "shaders/program/override.vsh"
        );
        assert_eq!(
            resolved.fragment.as_ref().unwrap().location.path,
            "shaders/program/override.fsh"
        );
        assert_eq!(resolved.vertex.as_ref().unwrap().pack_name, "pack.zip");
        assert_eq!(
            resolved.vertex.as_ref().unwrap().byte_len,
            valid_vertex().len()
        );
        assert_ne!(
            resolved.vertex.as_ref().unwrap().fingerprint,
            resolved.fragment.as_ref().unwrap().fingerprint
        );
        assert!(!resolved.has_overrides());
    }

    #[test]
    fn two_pack_source_stack_marks_the_top_source_as_an_override() {
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(SourcePack::new(
            "vanilla.jar",
            [
                ("shaders/program/override.vsh", valid_vertex()),
                ("shaders/program/override.fsh", valid_fragment()),
            ],
        )));
        let mut changed = valid_fragment();
        changed.extend_from_slice(b"\n// resource-pack override");
        rm.push_pack(Box::new(SourcePack::new(
            "custom.zip",
            [("shaders/program/override.fsh", changed)],
        )));

        let resolved = source_program()
            .resolve_shader_sources(&rm, "custom_program")
            .unwrap();
        assert!(resolved.has_overrides());
        assert_eq!(resolved.fragment.as_ref().unwrap().pack_name, "custom.zip");
        assert!(resolved.fragment.as_ref().unwrap().is_override);
        assert_eq!(resolved.vertex.as_ref().unwrap().pack_name, "vanilla.jar");
        assert!(!resolved.vertex.as_ref().unwrap().is_override);
    }

    #[test]
    fn reload_recomputes_shader_provenance_and_fingerprint() {
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(SourcePack::new(
            "before.zip",
            [
                ("shaders/program/override.vsh", valid_vertex()),
                ("shaders/program/override.fsh", valid_fragment()),
            ],
        )));
        let before = source_program()
            .resolve_shader_sources(&rm, "custom_program")
            .unwrap();
        let before_fingerprint = before.vertex.as_ref().unwrap().fingerprint;

        let mut changed_vertex = valid_vertex();
        changed_vertex.extend_from_slice(b"\n// reloaded");
        rm.reload(vec![Box::new(SourcePack::new(
            "after.zip",
            [
                ("shaders/program/override.vsh", changed_vertex),
                ("shaders/program/override.fsh", valid_fragment()),
            ],
        ))]);
        let after = source_program()
            .resolve_shader_sources(&rm, "custom_program")
            .unwrap();
        assert_eq!(rm.generation(), 1);
        assert_eq!(after.vertex.as_ref().unwrap().pack_name, "after.zip");
        assert_ne!(
            after.vertex.as_ref().unwrap().fingerprint,
            before_fingerprint
        );
    }

    #[test]
    fn malformed_source_reports_stage_and_exact_resource_path() {
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(SourcePack::new(
            "broken.zip",
            [
                ("shaders/program/override.vsh", valid_vertex()),
                (
                    "shaders/program/override.fsh",
                    b"void main() { gl_FragColor = vec4(1.0);".to_vec(),
                ),
            ],
        )));

        let error = source_program()
            .resolve_shader_sources(&rm, "custom_program")
            .unwrap_err();
        let text = error.to_string();
        assert!(text.contains("fragment"), "{text}");
        assert!(text.contains("shaders/program/override.fsh"), "{text}");
        assert!(text.contains("unclosed delimiter"), "{text}");
    }

    #[test]
    fn invalid_utf8_source_is_rejected_without_lossy_replacement() {
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(SourcePack::new(
            "binary.zip",
            [
                (
                    "shaders/program/override.vsh",
                    vec![
                        b'v', b'o', b'i', b'd', b' ', 0xff, b' ', b'm', b'a', b'i', b'n',
                    ],
                ),
                ("shaders/program/override.fsh", valid_fragment()),
            ],
        )));

        let error = source_program()
            .resolve_shader_sources(&rm, "custom_program")
            .unwrap_err();
        assert!(matches!(error, ShaderSourceError::InvalidUtf8 { .. }));
        assert!(error.to_string().contains("override.vsh"));
    }

    #[test]
    fn manifest_missing_stage_name_fails_before_any_source_lookup() {
        let program = ProgramSpec::parse(r#"{"vertex":"override"}"#).unwrap();
        let error = program
            .resolve_shader_sources(&ResourceManager::new(), "custom_program")
            .unwrap_err();
        assert!(matches!(
            error,
            ShaderSourceError::MissingStageName {
                stage: ShaderStage::Fragment,
                ..
            }
        ));
    }

    #[test]
    fn manifest_named_but_missing_source_is_not_silently_replaced_by_baked_wgsl() {
        let error = source_program()
            .resolve_shader_sources(&ResourceManager::new(), "custom_program")
            .unwrap_err();
        assert!(matches!(
            error,
            ShaderSourceError::Missing {
                stage: ShaderStage::Vertex,
                location,
                ..
            } if location.path == "shaders/program/override.vsh"
        ));
    }

    #[test]
    fn parses_blurs_two_pass_ping_pong() {
        // java: assets/minecraft/shaders/post/blur.json, transcribed from the real jar.
        let json = r#"
        {
            "targets": [ "swap" ],
            "passes": [
                { "name": "blur", "intarget": "minecraft:main", "outtarget": "swap",
                  "uniforms": [ {"name":"BlurDir","values":[1.0,0.0]}, {"name":"Radius","values":[20.0]} ] },
                { "name": "blur", "intarget": "swap", "outtarget": "minecraft:main",
                  "uniforms": [ {"name":"BlurDir","values":[0.0,1.0]}, {"name":"Radius","values":[20.0]} ] }
            ]
        }"#;
        let group = ShaderGroupSpec::parse(json).unwrap();
        assert_eq!(group.targets.len(), 1);
        assert_eq!(group.targets[0].name(), "swap");
        assert_eq!(group.passes.len(), 2);
        assert_eq!(group.passes[0].intarget, MAIN_TARGET);
        assert_eq!(group.passes[0].outtarget, "swap");
        assert_eq!(group.passes[1].intarget, "swap");
        assert_eq!(group.passes[1].outtarget, MAIN_TARGET);
        assert_eq!(group.passes[0].uniforms[0].name, "BlurDir");
        assert_eq!(group.passes[0].uniforms[0].values, vec![1.0, 0.0]);
        assert_eq!(group.passes[0].uniforms[1].values, vec![20.0]);
    }

    #[test]
    fn parses_notchs_fixed_size_aux_texture() {
        // java: assets/minecraft/shaders/post/notch.json
        let json = r#"
        {
            "targets": [ "swap" ],
            "passes": [
                { "name": "notch", "intarget": "minecraft:main", "outtarget": "swap",
                  "auxtargets": [ {"name":"DitherSampler","id":"dither","width":4,"height":4,"bilinear":false} ] },
                { "name": "blit", "intarget": "swap", "outtarget": "minecraft:main" }
            ]
        }"#;
        let group = ShaderGroupSpec::parse(json).unwrap();
        assert_eq!(group.passes[0].auxtargets.len(), 1);
        let aux = &group.passes[0].auxtargets[0];
        assert_eq!(aux.name, "DitherSampler");
        assert_eq!(aux.id, "dither");
        assert_eq!(aux.width, Some(4));
        assert_eq!(aux.height, Some(4));
        assert!(!aux.bilinear);
        assert!(group.passes[1].uniforms.is_empty());
        assert!(group.passes[1].auxtargets.is_empty());
    }

    #[test]
    fn parses_a_sized_target_object() {
        let json = r#"
        {
            "targets": [ {"name":"fixed","width":256,"height":128} ],
            "passes": []
        }"#;
        let group = ShaderGroupSpec::parse(json).unwrap();
        match &group.targets[0] {
            TargetSpec::Sized {
                name,
                width,
                height,
            } => {
                assert_eq!(name, "fixed");
                assert_eq!(*width, Some(256));
                assert_eq!(*height, Some(128));
            }
            other => panic!("expected a sized target, got {other:?}"),
        }
    }

    #[test]
    fn a_sized_target_may_omit_dimensions() {
        let json = r#"{ "targets": [ {"name":"tracks_main"} ], "passes": [] }"#;
        let group = ShaderGroupSpec::parse(json).unwrap();
        match &group.targets[0] {
            TargetSpec::Sized { width, height, .. } => {
                assert_eq!(*width, None);
                assert_eq!(*height, None);
            }
            other => panic!("expected a sized target, got {other:?}"),
        }
    }

    #[test]
    fn parses_a_program_json_with_typed_uniforms() {
        // java: assets/minecraft/shaders/program/blur.json
        let json = r#"
        {
            "blend": { "func": "add", "srcrgb": "one", "dstrgb": "zero" },
            "vertex": "sobel",
            "fragment": "blur",
            "attributes": [ "Position" ],
            "samplers": [ { "name": "DiffuseSampler" } ],
            "uniforms": [
                { "name": "ProjMat", "type": "matrix4x4", "count": 16, "values": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] },
                { "name": "InSize",  "type": "float", "count": 2, "values": [1.0,1.0] },
                { "name": "BlurDir", "type": "float", "count": 2, "values": [1.0,1.0] },
                { "name": "Radius",  "type": "float", "count": 1, "values": [5.0] }
            ]
        }"#;
        let program = ProgramSpec::parse(json).unwrap();
        assert_eq!(program.vertex.as_deref(), Some("sobel"));
        assert_eq!(program.fragment.as_deref(), Some("blur"));
        assert_eq!(program.attributes, vec!["Position"]);
        assert_eq!(program.samplers[0].name, "DiffuseSampler");
        assert!(program.samplers[0].file.is_none());
        assert_eq!(program.uniforms.len(), 4);
        assert_eq!(program.uniforms[3].name, "Radius");
        assert_eq!(program.uniforms[3].count, 1);
        assert_eq!(program.uniforms[3].values, vec![5.0]);
        let blend = program.blend.unwrap();
        assert!(!blend.enabled, "explicit vanilla defaults disable blending");
        assert_eq!(blend.func, BlendEquation::Add);
        assert_eq!(blend.srcrgb, BlendFactor::One);
        assert_eq!(blend.dstrgb, BlendFactor::Zero);
        assert_eq!(blend.separate_alpha_factors(), None);
        assert!(program.cull);
    }

    #[test]
    fn program_uniform_values_follow_java_count_broadcast_and_arity_rules() {
        let program = ProgramSpec::parse(
            r#"{"uniforms":[
                {"name":"Tint","type":"float","count":3,"values":[0.5]},
                {"name":"Empty","type":"float","count":3,"values":[]}
            ]}"#,
        )
        .unwrap();
        assert_eq!(program.uniforms[0].values, vec![0.5, 0.5, 0.5]);
        assert!(program.uniforms[1].values.is_empty());

        let err = ProgramSpec::parse(
            r#"{"uniforms":[{"name":"Tint","type":"float","count":3,"values":[0.0,1.0]}]}"#,
        )
        .unwrap_err();
        assert_eq!(err.path(), &["uniforms[0]"]);
        assert!(err.message.contains("expected 3"));
    }

    #[test]
    fn program_cull_is_boolean_and_defaults_true() {
        let omitted = ProgramSpec::parse(r#"{"vertex":"blit"}"#).unwrap();
        let disabled = ProgramSpec::parse(r#"{"vertex":"blit","cull":false}"#).unwrap();

        assert!(omitted.cull, "ShaderManager defaults missing cull to true");
        assert!(
            !disabled.cull,
            "a resource pack can explicitly disable culling"
        );
    }

    #[test]
    fn either_alpha_factor_selects_separate_blending_with_vanilla_defaults() {
        let source_only = ProgramSpec::parse(
            r#"{"blend":{"func":"add","srcrgb":"one","dstrgb":"zero","srcalpha":"zero"}}"#,
        )
        .unwrap();
        let destination_only = ProgramSpec::parse(
            r#"{"blend":{"func":"add","srcrgb":"one","dstrgb":"zero","dstalpha":"one"}}"#,
        )
        .unwrap();

        let source_only = source_only.blend.unwrap();
        assert!(source_only.enabled);
        assert_eq!(source_only.srcalpha, BlendFactor::Zero);
        assert_eq!(source_only.dstalpha, BlendFactor::Zero);
        assert_eq!(
            source_only.separate_alpha_factors(),
            Some((BlendFactor::Zero, BlendFactor::Zero))
        );

        let destination_only = destination_only.blend.unwrap();
        assert_eq!(destination_only.srcalpha, BlendFactor::One);
        assert_eq!(destination_only.dstalpha, BlendFactor::One);
        assert_eq!(
            destination_only.separate_alpha_factors(),
            Some((BlendFactor::One, BlendFactor::One))
        );
    }

    #[test]
    fn blend_factor_aliases_follow_vanillas_ordered_normalisation() {
        // Java applies trim/lowercase and then `_`, `one`, `zero`, `minus` replacements. The
        // common OpenGL spelling therefore reaches the same `1-srcalpha` lookup key.
        assert_eq!(
            blend_factor_from_name("  ONE_MINUS_SRC_ALPHA\t"),
            BlendFactor::OneMinusSrcAlpha
        );
        assert_eq!(
            blend_factor_from_name("one_minus_dst_color"),
            BlendFactor::OneMinusDstColor
        );
        assert_eq!(blend_factor_from_name("SRC_COLOR"), BlendFactor::SrcColor);
        assert_eq!(blend_factor_from_name("not-a-factor"), BlendFactor::Invalid);
        assert_eq!(BlendFactor::Invalid.gl_enum(), -1);
    }

    #[test]
    fn blend_equations_trim_and_case_fold_java_aliases_with_add_fallback() {
        assert_eq!(
            blend_equation_from_name("reverse_subtract"),
            BlendEquation::ReverseSubtract
        );
        assert_eq!(
            blend_equation_from_name("reversesubtract"),
            BlendEquation::ReverseSubtract
        );
        assert_eq!(
            blend_equation_from_name("  ReVeRsE SuBtRaCt  "),
            BlendEquation::Add,
            "Java only accepts the underscore and non-underscore aliases"
        );
        assert_eq!(
            blend_equation_from_name("  MAX\t"),
            BlendEquation::Max,
            "equation names trim and lowercase before alias lookup"
        );
        assert_eq!(blend_equation_from_name("MIN"), BlendEquation::Min);
        assert_eq!(
            blend_equation_from_name("SUBTRACT"),
            BlendEquation::Subtract
        );
        assert_eq!(blend_equation_from_name(" add \n"), BlendEquation::Add);
        assert_eq!(
            blend_equation_from_name("not-an-equation"),
            BlendEquation::Add
        );
        assert_eq!(BlendEquation::Max.gl_enum(), 32_776);
    }

    #[test]
    fn alpha_key_presence_only_selects_separate_path_when_blending_is_enabled() {
        let default_alpha = ProgramSpec::parse(
            r#"{"blend":{"func":"add","srcrgb":"one","dstrgb":"zero","srcalpha":"one","dstalpha":"zero"}}"#,
        )
        .unwrap()
        .blend
        .unwrap();
        assert!(!default_alpha.enabled);
        assert!(!default_alpha.separate_alpha);

        let combined =
            ProgramSpec::parse(r#"{"blend":{"srcrgb":"srcalpha","dstrgb":"1-srcalpha"}}"#)
                .unwrap()
                .blend
                .unwrap();
        assert!(combined.enabled);
        assert!(!combined.separate_alpha);
        assert_eq!(combined.separate_alpha_factors(), None);
    }

    #[test]
    fn non_string_blend_fields_are_ignored_but_non_object_blend_is_rejected() {
        // `JsonUtils.isString` treats a present number exactly like an absent member.
        let ignored = ProgramSpec::parse(r#"{"blend":{"srcrgb":0,"srcalpha":0}}"#)
            .unwrap()
            .blend
            .unwrap();
        assert_eq!(ignored, BlendSpec::default());

        let err = ProgramSpec::parse(r#"{"blend":false}"#).unwrap_err();
        assert!(err.message.contains("blend must be an object"));
    }

    #[test]
    fn a_program_with_no_blend_block_parses_as_replace() {
        let json =
            r#"{ "vertex": "blit", "fragment": "blit", "samplers": [{"name":"DiffuseSampler"}] }"#;
        let program = ProgramSpec::parse(json).unwrap();
        assert!(program.blend.is_none());
    }

    #[test]
    fn malformed_json_is_a_typed_error_not_a_panic() {
        let err = ShaderGroupSpec::parse("{ not json").unwrap_err();
        assert!(err.message.contains("json") || !err.message.is_empty());
        assert!(err.path().is_empty());
    }

    #[test]
    fn shader_group_errors_preserve_the_full_nested_json_context_chain() {
        let err = ShaderGroupSpec::parse(
            r#"{
                "passes": [{
                    "name": "blur",
                    "intarget": "minecraft:main",
                    "outtarget": "swap",
                    "uniforms": [
                        {"name": "Radius", "values": [1.0]},
                        {"name": "BlurDir", "values": [0.0, false]}
                    ]
                }]
            }"#,
        )
        .unwrap_err();

        assert_eq!(err.path(), &["passes[0]", "uniforms[1]", "values[1]"]);
        assert_eq!(err.message, "uniform value must be a number");
        assert_eq!(
            err.to_string(),
            "malformed json at passes[0]->uniforms[1]->values[1]: uniform value must be a number"
        );
    }

    #[test]
    fn program_json_context_attaches_array_indices_and_top_level_keys() {
        let uniform = ProgramSpec::parse(
            r#"{"uniforms":[{"name":"Ok","type":"float","count":1,"values":[1]},
                             {"name":"Bad","type":"float","count":1,"values":["nope"]}]}"#,
        )
        .unwrap_err();
        assert_eq!(
            uniform.path(),
            &["uniforms[1]", "values[0]"],
            "dropping either catch boundary reproduces the missing JsonException chain"
        );

        let blend = ProgramSpec::parse(r#"{"blend":false}"#).unwrap_err();
        assert_eq!(blend.path(), &["blend"]);
        assert!(blend.to_string().contains("at blend:"));
    }

    #[test]
    fn strings_with_escapes_round_trip() {
        let v = Json::parse(r#"{"a": "line1\nline2\t\"quoted\""}"#).unwrap();
        assert_eq!(
            v.get("a").unwrap().as_str().unwrap(),
            "line1\nline2\t\"quoted\""
        );
    }

    #[test]
    fn numbers_parse_including_negative_and_fractional() {
        let v = Json::parse("[-1.5, 20, 0.00390625]").unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr[0].as_f64(), Some(-1.5));
        assert_eq!(arr[1].as_f64(), Some(20.0));
        assert_eq!(arr[2].as_f64(), Some(0.00390625));
    }

    #[test]
    fn empty_object_and_array_parse() {
        assert_eq!(Json::parse("{}").unwrap(), Json::Object(Vec::new()));
        assert_eq!(Json::parse("[]").unwrap(), Json::Array(Vec::new()));
    }

    #[test]
    fn trailing_garbage_is_rejected() {
        assert!(Json::parse("{} garbage").is_err());
    }
}
