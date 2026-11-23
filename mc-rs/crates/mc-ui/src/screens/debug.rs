//! The F3 debug overlay.
//!
//! java: `client/gui/GuiOverlayDebug.java`
//!
//! Two columns of one-line strings on translucent black plates: the left one is world state,
//! the right one is machine state. The formatting is the load-bearing part — `XYZ` is
//! `%.3f / %.5f / %.3f` (note the *five* decimals on Y, which is deliberate: it is how you see
//! the 1/64 step-height quantisation), memory is `% 2d%% %03d/%03dMB`, and local difficulty is
//! `%.2f (Day %d)`.
//!
//! [`DebugInfo`] is the whole input, so every string below is testable from a literal — and the
//! formatting is cross-checked in `tests/vanilla_debug.rs` against values transcribed
//! independently from the Java's format strings.

use mc_math::{BlockPos, Facing};
use mc_world::World;

use crate::font::FontRenderer;
use crate::frame_timer::FrameTimer;
use crate::screen_stack::ScreenCtx;
use crate::screens::common::INFO_COLOR;
use crate::widget::draw::DrawList;
use crate::widget::keys::Modifiers;
use crate::widget::{unpack_argb, UiCull, UiDepth, UiTexture};

/// The backdrop behind each line. java: `drawRect(..., -1873784752)`
pub const PLATE_COLOR: u32 = 0x9050_5050;

/// The lagometer is always 240 GUI pixels wide: one column per `FrameTimer` slot.
pub const LAGOMETER_WIDTH: i32 = 240;
/// The graph's 30 FPS reference line is thirty pixels above its bottom edge.
pub const LAGOMETER_HALF_HEIGHT: i32 = 30;
/// The graph's 30 ms / 60 ms range, in GUI pixels.
pub const LAGOMETER_HEIGHT: i32 = LAGOMETER_HALF_HEIGHT * 2;

/// Horizontal radius of vanilla's F3+Shift profiler pie.
pub const PROFILER_RADIUS: i32 = 160;
/// Vertical radius of the profiler pie. Java uses `radius * 0.5F`.
pub const PROFILER_VERTICAL_RADIUS: f32 = 80.0;
/// The chart's dark backing rectangle uses this alpha before ordinary GUI blending.
pub const PROFILER_PANEL_COLOR: u32 = 0xc800_0000;

/// One one-pixel-wide vertical lagometer column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LagometerBar {
    /// X position from the graph's left edge.
    pub x: i32,
    /// Height calculated by `FrameTimer.getLagometerValue(time, 30)`.
    pub height: i32,
    /// Vanilla's green → yellow → red ARGB colour.
    pub color: u32,
}

/// Pure geometry and guide state for the F3+Alt frame graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LagometerData {
    /// Oldest-to-newest columns, exactly as `GuiOverlayDebug` walks `FrameTimer`.
    pub bars: Vec<LagometerBar>,
    /// Green 120-FPS cap guide in graph-local Y coordinates, when vanilla draws it.
    pub frame_limit_y: Option<i32>,
}

/// One result row from vanilla's `Profiler.getProfilingData`.
///
/// `use_percentage` is the slice's share of the selected profiler parent, while
/// `total_use_percentage` is its share of the root.  The driver supplies these values from the
/// existing CPU profiler snapshot; keeping the result shape here avoids making the UI depend on
/// the profiler subscriber feature of `mc-core`.
// java: net/minecraft/profiler/Profiler.Result
#[derive(Debug, Clone, PartialEq)]
pub struct ProfilerResult {
    /// The section name. An empty name is the root row; `"unspecified"` is vanilla's remainder.
    pub name: String,
    /// The percentage used by this section inside its selected parent.
    pub use_percentage: f64,
    /// The percentage used by this section inside the root profiler result.
    pub total_use_percentage: f64,
    /// Vanilla's stable colour derived from [`Self::name`].
    pub color: u32,
}

impl ProfilerResult {
    /// Build one result and derive the same colour as `Profiler.Result.getColor()`.
    #[must_use]
    pub fn new(name: impl Into<String>, use_percentage: f64, total_use_percentage: f64) -> Self {
        let name = name.into();
        Self {
            color: profiler_color(&name),
            name,
            use_percentage,
            total_use_percentage,
        }
    }

    /// The synthetic root row returned by vanilla before its child rows.
    #[must_use]
    pub fn root() -> Self {
        Self::new("", 100.0, 100.0)
    }
}

/// A pie-chart slice with its accumulated percentage interval.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfilerSlice {
    /// The displayed result for this slice.
    pub result: ProfilerResult,
    /// Start of the slice in percentage points around the pie.
    pub start_percentage: f64,
    /// End of the slice in percentage points around the pie.
    pub end_percentage: f64,
}

/// CPU-only geometry input for the F3+Shift profiler chart.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfilerChartData {
    /// The selected parent row shown above the pie.
    pub root: ProfilerResult,
    /// Child slices in vanilla's already-sorted order.
    pub slices: Vec<ProfilerSlice>,
}

/// Convert Java profiler results into the cumulative intervals consumed by the pie renderer.
///
/// Vanilla's `displayDebugInfo` keeps a running `d0` in percentage points and emits one fan for
/// each row.  Invalid or non-positive percentages cannot produce a visible Java slice and are
/// ignored here as a defensive boundary for host-side profiler data.
#[must_use]
pub fn profiler_chart_data(results: &[ProfilerResult]) -> Option<ProfilerChartData> {
    let root = results.first()?.clone();
    let mut cursor = 0.0;
    let mut slices = Vec::new();
    for result in results.iter().skip(1) {
        if !result.use_percentage.is_finite() || result.use_percentage <= 0.0 {
            continue;
        }
        let end = cursor + result.use_percentage;
        if !end.is_finite() {
            continue;
        }
        slices.push(ProfilerSlice {
            result: result.clone(),
            start_percentage: cursor,
            end_percentage: end,
        });
        cursor = end;
    }
    Some(ProfilerChartData { root, slices })
}

/// Vanilla's `Profiler.Result.getColor()` hash, including Java's UTF-16 string semantics.
// java: Profiler.java:181-184
#[must_use]
pub fn profiler_color(name: &str) -> u32 {
    let hash = name.encode_utf16().fold(0_i32, |hash, unit| {
        hash.wrapping_mul(31).wrapping_add(i32::from(unit))
    });
    ((hash as u32) & 0x00aa_aaaa).wrapping_add(0x0044_4444)
}

/// Build the data consumed by the vanilla F3+Alt lagometer.
///
/// `frame_limit` is `GameSettings.limitFramerate`.  Values above 120 omit the cap guide, as
/// vanilla does for unlimited/large caps.
// java: GuiOverlayDebug.renderLagometer:219-255
#[must_use]
pub fn lagometer_data(timer: &FrameTimer, frame_limit: i32) -> LagometerData {
    let bars = timer
        .lagometer_samples()
        .enumerate()
        .map(|(x, time)| {
            let height = FrameTimer::lagometer_value(time, LAGOMETER_HALF_HEIGHT);
            let colour_value = height.clamp(0, LAGOMETER_HEIGHT);
            LagometerBar {
                x: x as i32,
                height,
                color: frame_color(colour_value, 0, LAGOMETER_HALF_HEIGHT, LAGOMETER_HEIGHT),
            }
        })
        .collect();
    LagometerData {
        bars,
        frame_limit_y: (frame_limit <= 120).then_some(frame_limit / 2),
    }
}

/// Vanilla's F3 lag colour ramp.
// java: GuiOverlayDebug.getFrameColor
#[must_use]
pub fn frame_color(value: i32, _minimum: i32, middle: i32, maximum: i32) -> u32 {
    if value < middle {
        blend_colors(0xff00_ff00, 0xffff_ff00, value as f32 / middle as f32)
    } else {
        blend_colors(
            0xffff_ff00,
            0xffff_0000,
            (value - middle) as f32 / (maximum - middle) as f32,
        )
    }
}

/// Java's component-wise ARGB interpolation used by [`frame_color`].
// java: GuiOverlayDebug.blendColors
#[must_use]
pub fn blend_colors(first: u32, second: u32, amount: f32) -> u32 {
    let channel = |shift| {
        let from = ((first >> shift) & 0xff_u32) as f32;
        let to = ((second >> shift) & 0xff_u32) as f32;
        (from + (to - from) * amount) as i32
    };
    let alpha = channel(24).clamp(0, 255) as u32;
    let red = channel(16).clamp(0, 255) as u32;
    let green = channel(8).clamp(0, 255) as u32;
    let blue = channel(0).clamp(0, 255) as u32;
    alpha << 24 | red << 16 | green << 8 | blue
}

/// Everything the overlay reports that the UI layer cannot work out for itself.
#[derive(Debug, Clone, PartialEq)]
pub struct DebugInfo {
    /// The client version string, e.g. `"1.8.9"`.
    pub version: String,
    /// The brand, e.g. `"vanilla"`. java: `ClientBrandRetriever.getClientModName()`
    pub brand: String,
    /// The host runtime version shown in Java's `System.getProperty("java.version")` slot.
    ///
    /// The client is Rust rather than a JVM, so the caller supplies the best available host
    /// runtime label. It is deliberately separate from [`Self::version`], which is the launcher's
    /// `--version` value shown in the left-column header.
    pub java_version: String,
    /// The host runtime data-model width shown beside [`Self::java_version`].
    pub java_bits: u8,
    /// The FPS line. java: `Minecraft.debug` — `"120 fps (0 chunk updates) T: inf vsync"`. Note
    /// the plural: vanilla's pluraliser drops the `s` at exactly one update, so zero keeps it.
    pub fps: String,
    /// The chunk-renderer line. java: `RenderGlobal.getDebugInfoRenders()`
    pub renders: String,
    /// The entity line. java: `RenderGlobal.getDebugInfoEntities()`
    pub entities: String,
    /// The particle line. java: `"P: <n>. T: <n>"`
    pub particles: String,
    /// Fallback chunk-provider status line when no [`World`] is available.
    /// java: `World.getProviderName()`
    pub dimension: String,
    /// The camera position.
    pub position: (f64, f64, f64),
    /// Yaw and pitch in degrees, before wrapping.
    pub rotation: (f32, f32),
    /// The block the crosshair is on, if any.
    pub looking_at: Option<BlockPos>,
    /// That block's world/neighbour-resolved state, for the right column's registry name +
    /// property list. `None` even when `looking_at` is `Some` means the state was not surfaced —
    /// the two are independent so a caller that only has the position can still get the left
    /// column's line. java: `GuiOverlayDebug.getDebugInfoRight`'s `objectMouseOver` block,
    /// `:185`-`:212`; the vanilla `getActualState` call is represented by
    /// [`mc_data::block::ActualBlockState`].
    pub looking_at_state: Option<mc_data::block::ActualBlockState>,
    /// Local difficulty and day number.
    pub difficulty: Option<(f32, i64)>,
    /// The active post shader group resource, in Java's `ResourceLocation.toString()` form.
    ///
    /// This is deliberately separate from the renderer's visibility bit: Java's F4 toggle hides
    /// a loaded `ShaderGroup` without making `EntityRenderer.isShaderActive()` false.
    pub shader: Option<String>,
    /// Java-side memory numbers: `(used, allocated, max)` in bytes.
    ///
    /// Rust has no `Runtime.maxMemory()`, so these arrive from the caller — `mc-client` can
    /// supply real numbers from an allocator hook or zeros. Zeros render as `0% 000/000MB`
    /// rather than dividing by zero.
    pub memory: (u64, u64, u64),
    /// The CPU description.
    pub cpu: String,
    /// The window size in real pixels, and the GPU vendor.
    pub display: (u32, u32, String),
    /// The adapter name.
    pub renderer: String,
    /// The graphics API version.
    pub api_version: String,
    /// The root row followed by the current F3+Shift profiler children.
    ///
    /// An empty vector means profiling was not selected on this frame.  The root-only value is
    /// useful while the CPU profiler is warming up: vanilla still paints the chart frame before
    /// its first child section has accumulated.
    pub profiler: Vec<ProfilerResult>,
    /// Whether the server asked for `reducedDebugInfo` or the player set it.
    pub reduced: bool,
}

impl Default for DebugInfo {
    fn default() -> Self {
        Self {
            version: "1.8.9".into(),
            brand: "vanilla".into(),
            java_version: "<unknown>".into(),
            java_bits: usize::BITS as u8,
            fps: String::new(),
            renders: String::new(),
            entities: String::new(),
            particles: String::new(),
            dimension: "MultiplayerChunkCache: 0, 0".into(),
            position: (0.0, 0.0, 0.0),
            rotation: (0.0, 0.0),
            looking_at: None,
            looking_at_state: None,
            difficulty: None,
            shader: None,
            memory: (0, 0, 0),
            cpu: "<unknown>".into(),
            display: (854, 480, "<unknown>".into()),
            renderer: "<unknown>".into(),
            api_version: "<unknown>".into(),
            profiler: Vec::new(),
            reduced: false,
        }
    }
}

/// The compass direction string vanilla prints beside the facing.
// java: GuiOverlayDebug.java:116
#[must_use]
pub fn facing_description(facing: Facing) -> &'static str {
    match facing {
        Facing::North => "Towards negative Z",
        Facing::South => "Towards positive Z",
        Facing::West => "Towards negative X",
        Facing::East => "Towards positive X",
        _ => "Invalid",
    }
}

/// The horizontal facing reported by `Entity.getHorizontalFacing` for an entity yaw.
///
/// Keep the multiply and divide in `f32`: Java evaluates `rotationYaw * 4.0F / 360.0F` before
/// widening that result for `floor_double`. Widening the yaw first and calling
/// `EnumFacing.fromAngle(double)` differs at valid large `f32` values near a quadrant boundary.
// java: Entity.java:2654-2657
fn entity_horizontal_facing(yaw: f32) -> Facing {
    let quarter_turns = yaw * 4.0 / 360.0;
    Facing::from_horizontal(mc_math::floor_f64(f64::from(quarter_turns) + 0.5) & 3)
}

/// Bytes to whole megabytes, Java's two integer divisions.
// java: GuiOverlayDebug.bytesToMb
#[must_use]
pub fn bytes_to_mb(bytes: u64) -> u64 {
    bytes / 1024 / 1024
}

/// Java's `String.format("% 2d", n)`.
///
/// The `' '` flag emits a space where a sign would go, and the width-2 field is already full
/// once that space and one digit are in it — so this is a leading space, not padding to three
/// columns. `format!("{n:>3}")` looks equivalent and is not: it pads `5` to `"  5"` where Java
/// gives `" 5"`, and every memory line comes out a column wide.
#[must_use]
fn space_padded_2(n: u64) -> String {
    format!(" {n}")
}

/// Java `Formatter`'s fixed-point conversion rounds an exact halfway value away from zero,
/// while Rust's formatter rounds it to an even last digit. Detect that narrow case from the
/// float's binary representation and advance Rust's otherwise-identical fixed-point result.
///
/// Debug coordinates are deliberately dyadic (slabs and the 1/64 movement grid), so this is not
/// merely theoretical here: `-3.8125` at three places is `-3.813` in Java and `-3.812` with a
/// bare Rust format string.
fn format_fixed_half_up(value: f64, digits: usize) -> String {
    if !value.is_finite() {
        return format!("{value:.digits$}");
    }

    let negative = value.is_sign_negative();
    let magnitude = value.abs();
    let mut rendered = format!("{magnitude:.digits$}");

    let bits = magnitude.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (mantissa, exponent) = if exponent_bits == 0 {
        (u128::from(fraction), -1074)
    } else {
        (
            u128::from((1_u64 << 52) | fraction),
            exponent_bits - 1023 - 52,
        )
    };

    // value * 10^digits = mantissa * 5^digits * 2^(exponent + digits).
    // It is exactly `integer + 1/2` iff the low denominator bits have that one-bit shape.
    let halfway_with_even_floor = u32::try_from(digits)
        .ok()
        .and_then(|digits| 5_u128.checked_pow(digits).map(|five| (digits, five)))
        .and_then(|(digits, five)| {
            mantissa
                .checked_mul(five)
                .map(|numerator| (digits, numerator))
        })
        .is_some_and(|(digits, numerator)| {
            let binary_exponent = exponent.saturating_add(digits as i32);
            if binary_exponent >= 0 {
                return false;
            }
            let shift = binary_exponent.unsigned_abs();
            if shift == 0 || shift >= 128 {
                return false;
            }
            let halfway_bit = 1_u128 << (shift - 1);
            let remainder_mask = (1_u128 << shift) - 1;
            numerator & remainder_mask == halfway_bit && (numerator >> shift) & 1 == 0
        });

    if halfway_with_even_floor {
        let mut bytes = rendered.into_bytes();
        let mut carried = true;
        for byte in bytes.iter_mut().rev() {
            match *byte {
                b'0'..=b'8' => {
                    *byte += 1;
                    carried = false;
                    break;
                }
                b'9' => *byte = b'0',
                _ => {}
            }
        }
        if carried {
            bytes.insert(0, b'1');
        }
        rendered = String::from_utf8(bytes).expect("fixed-point output is ASCII");
    }

    if negative {
        rendered.insert(0, '-');
    }
    rendered
}

/// The F3 overlay.
#[derive(Debug, Clone)]
pub struct DebugOverlay {
    /// Whether F3 is held down.
    pub visible: bool,
    /// Whether the one-shot F3+Shift profiler mode was selected on the most recent F3 edge.
    ///
    /// The chart itself is a separate profiler-core/render seam; retaining this vanilla flag here
    /// keeps the key edge observable and prevents the Alt and Shift modes from being conflated.
    show_profiler_chart: bool,
    /// Whether the one-shot F3+Alt lagometer mode was selected on the most recent F3 edge.
    show_lagometer: bool,
    /// The fixed-size history consumed by the F3+Alt lagometer.
    frame_timer: FrameTimer,
    /// Java's `Minecraft.debugProfilerName`, represented with `/` between Rust span names so a
    /// span name containing dots remains one selectable section.
    profiler_path: String,
}

impl Default for DebugOverlay {
    fn default() -> Self {
        Self {
            visible: false,
            show_profiler_chart: false,
            show_lagometer: false,
            frame_timer: FrameTimer::default(),
            profiler_path: "root".to_owned(),
        }
    }
}

impl DebugOverlay {
    /// A hidden overlay.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply the F3 edge and its modifier snapshot.
    ///
    /// java: `Minecraft.runTick`'s `k == 61` branch. The auxiliary flags are assigned on every
    /// F3 press, rather than toggled independently, so a plain F3 immediately clears a prior
    /// Alt/Shift mode just like vanilla.
    pub fn on_f3(&mut self, mods: Modifiers) {
        self.visible = !self.visible;
        self.show_profiler_chart = mods.shift;
        self.show_lagometer = mods.alt;
    }

    /// Record one displayed frame for the lagometer's `FrameTimer` ring.
    ///
    /// The shell exposes the completed frame duration as seconds. Convert it at this boundary so
    /// the UI timer remains a Java-shaped nanosecond ring and does not need to depend on the
    /// renderer's frame-stat type.
    pub fn record_frame_time(&mut self, seconds: f32) {
        let nanos = if seconds.is_finite() && seconds >= 0.0 {
            (f64::from(seconds) * 1_000_000_000.0) as i64
        } else {
            0
        };
        self.frame_timer.add_frame(nanos);
    }

    /// Whether the most recent F3 edge selected the profiler chart.
    #[must_use]
    pub const fn profiler_chart_visible(&self) -> bool {
        self.show_profiler_chart
    }

    /// Whether the most recent F3 edge selected the lagometer.
    #[must_use]
    pub const fn lagometer_visible(&self) -> bool {
        self.show_lagometer
    }

    /// The frame-time history for the F3+Alt renderer.
    #[must_use]
    pub const fn frame_timer(&self) -> &FrameTimer {
        &self.frame_timer
    }

    /// The currently selected profiler path. `root` is the initial Java selection.
    #[must_use]
    pub fn profiler_path(&self) -> &str {
        &self.profiler_path
    }

    /// Apply Java's number-key profiler navigation to the current result list.
    ///
    /// `0` backs up one parent (where one exists); `1` through `9` select the corresponding
    /// immediate child, with the synthetic `unspecified` remainder deliberately not selectable.
    pub fn update_profiler_name(&mut self, key_count: u8, results: &[ProfilerResult]) {
        let Some(root) = results.first() else { return };
        if key_count == 0 {
            if !root.name.is_empty() {
                if let Some(parent) = self.profiler_path.rsplit_once('/') {
                    self.profiler_path.truncate(parent.0.len());
                }
            }
            return;
        }
        let index = usize::from(key_count - 1);
        let Some(child) = results.get(index + 1) else {
            return;
        };
        if child.name == "unspecified" {
            return;
        }
        if self.profiler_path == "root" || !self.profiler_path.ends_with('/') {
            self.profiler_path.push('/');
        }
        self.profiler_path.push_str(&child.name);
    }

    /// The left column.
    ///
    /// `world` supplies the biome, light levels and chunk state. A position in an unloaded
    /// chunk simply omits those three lines, exactly as vanilla's `isBlockLoaded` guard does.
    // java: GuiOverlayDebug.java:102 (call)
    #[must_use]
    pub fn left_column(&self, info: &DebugInfo, world: Option<&World>) -> Vec<String> {
        let (x, y, z) = info.position;
        let pos = BlockPos::from_f64(x, y, z);
        // `ChunkProviderClient` keeps the mapping and listing in lock-step. `World` has one
        // authoritative map, so its size supplies both values of vanilla's provider line.
        let provider = world.map_or_else(
            || info.dimension.clone(),
            |world| {
                let count = world.loaded_chunk_count();
                format!("MultiplayerChunkCache: {count}, {count}")
            },
        );
        let header = vec![
            format!("Minecraft 1.8.9 ({}/{})", info.version, info.brand),
            info.fps.clone(),
            info.renders.clone(),
            info.entities.clone(),
            info.particles.clone(),
            provider,
            String::new(),
        ];
        if info.reduced {
            let mut list = header;
            list.push(format!(
                "Chunk-relative: {} {} {}",
                pos.x & 15,
                pos.y & 15,
                pos.z & 15
            ));
            return list;
        }
        let facing = entity_horizontal_facing(info.rotation.0);
        let mut list = header;
        list.push(format!(
            "XYZ: {} / {} / {}",
            format_fixed_half_up(x, 3),
            format_fixed_half_up(y, 5),
            format_fixed_half_up(z, 3)
        ));
        list.push(format!("Block: {} {} {}", pos.x, pos.y, pos.z));
        list.push(format!(
            "Chunk: {} {} {} in {} {} {}",
            pos.x & 15,
            pos.y & 15,
            pos.z & 15,
            pos.x >> 4,
            pos.y >> 4,
            pos.z >> 4
        ));
        list.push(format!(
            "Facing: {} ({}) ({} / {})",
            facing_name(facing),
            facing_description(facing),
            format_fixed_half_up(f64::from(mc_math::wrap_degrees_f32(info.rotation.0)), 1),
            format_fixed_half_up(f64::from(mc_math::wrap_degrees_f32(info.rotation.1)), 1)
        ));
        // java: `World.isBlockLoaded(BlockPos)` rejects positions outside the build-height range
        // before consulting the chunk provider. Difficulty lives inside this same guard.
        if let Some(world) = world
            .filter(|world| (0..256).contains(&pos.y) && world.is_chunk_loaded(pos.chunk_pos()))
        {
            let biome = world
                .chunk(pos.chunk_pos())
                .map_or(0, |c| c.biome((pos.x & 15) as u8, (pos.z & 15) as u8));
            list.push(format!("Biome: {}", mc_data::biome(biome).name));
            list.push(format!(
                "Light: {} ({} sky, {} block)",
                world.light_at(pos),
                world.sky_light_at(pos),
                world.block_light_at(pos)
            ));
            if let Some((difficulty, day)) = info.difficulty {
                list.push(format!(
                    "Local Difficulty: {} (Day {day})",
                    format_fixed_half_up(f64::from(difficulty), 2)
                ));
            }
        }
        if let Some(shader) = info.shader.as_deref() {
            list.push(format!("Shader: {shader}"));
        }
        if let Some(p) = info.looking_at {
            list.push(format!("Looking at: {} {} {}", p.x, p.y, p.z));
        }
        list
    }

    /// The right column.
    // java: GuiOverlayDebug.java:171 (getDebugInfoRight)
    #[must_use]
    pub fn right_column(&self, info: &DebugInfo) -> Vec<String> {
        let (used, allocated, max) = info.memory;
        // Vanilla divides by maxMemory unguarded; a zero there would panic in Rust, so it is
        // reported as 0% instead. Every other digit is vanilla's.
        let pct = |n: u64| {
            n.checked_mul(100)
                .and_then(|v| v.checked_div(max))
                .unwrap_or(0)
        };
        let mut list = vec![
            format!("Java: {} {}bit", info.java_version, info.java_bits),
            format!(
                "Mem: {}% {:03}/{:03}MB",
                space_padded_2(pct(used)),
                bytes_to_mb(used),
                bytes_to_mb(max)
            ),
            format!(
                "Allocated: {}% {:03}MB",
                space_padded_2(pct(allocated)),
                bytes_to_mb(allocated)
            ),
            String::new(),
            format!("CPU: {}", info.cpu),
            String::new(),
            format!(
                "Display: {}x{} ({})",
                info.display.0, info.display.1, info.display.2
            ),
            info.renderer.clone(),
            info.api_version.clone(),
        ];
        // java: `GuiOverlayDebug.getDebugInfoRight` (`:179`-`:213`) — gated the same way the left
        // column's extra lines are, and skipped entirely when nothing is targeted.
        if !info.reduced {
            if let Some(state) = info.looking_at_state.as_ref() {
                list.push(String::new());
                let block_id = mc_data::block_of(state.stored()).0;
                list.push(
                    mc_data::block_info(mc_data::BlockId(block_id))
                        .name
                        .to_owned(),
                );
                for (key, value) in state.properties() {
                    // GuiOverlayDebug prints Comparable.toString(), not the serialized model
                    // name. BlockQuartz.EnumType keeps `lines_y/x/z` for model lookup but
                    // stringifies every directional line variant as `lines`.
                    let value = if block_id == 155
                        && *key == "variant"
                        && matches!(*value, "lines_y" | "lines_x" | "lines_z")
                    {
                        "lines"
                    } else if matches!(block_id, 35 | 95 | 159 | 160 | 171)
                        && *key == "color"
                        && *value == "light_blue"
                    {
                        // `EnumDyeColor` serializes model keys as `light_blue`, but its
                        // `toString()` value (which F3 prints) is the camel-cased `lightBlue`.
                        "lightBlue"
                    } else {
                        *value
                    };
                    // java: `EnumChatFormatting.GREEN`/`RED` (`§a`/`§c`) prefixed onto boolean
                    // property values only — `state_properties` already renders every value as
                    // vanilla's `Comparable.toString()` would, and `"true"`/`"false"` are exactly
                    // and only what a boolean property stringifies to.
                    let coloured = match value {
                        "true" => format!("§a{value}"),
                        "false" => format!("§c{value}"),
                        _ => value.to_owned(),
                    };
                    list.push(format!("{key}: {coloured}"));
                }
            }
        }
        list
    }

    /// Draw both columns.
    // java: GuiOverlayDebug.java:61
    pub fn draw(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        info: &DebugInfo,
        world: Option<&World>,
    ) {
        if !self.visible {
            return;
        }
        let h = font.font_height();
        for (i, line) in self.left_column(info, world).into_iter().enumerate() {
            if line.is_empty() {
                continue;
            }
            let w = font.string_width(&line);
            let y = 2 + h * i as i32;
            d.rect(1, y - 1, 2 + w + 1, y + h - 1, PLATE_COLOR);
            font.draw_string(d, &line, 2.0, y as f32, INFO_COLOR, false);
        }
        for (i, line) in self.right_column(info).into_iter().enumerate() {
            if line.is_empty() {
                continue;
            }
            let w = font.string_width(&line);
            let x = ctx.width - 2 - w;
            let y = 2 + h * i as i32;
            d.rect(x - 1, y - 1, x + w + 1, y + h - 1, PLATE_COLOR);
            font.draw_string(d, &line, x as f32, y as f32, INFO_COLOR, false);
        }
        if self.show_profiler_chart {
            self.draw_profiler_chart(d, font, ctx, &info.profiler);
        }
    }

    /// Draw the F3+Alt frame-time graph below the normal debug columns.
    ///
    /// The caller owns the F3/Alt/settings gate and feeds this once per displayed debug frame.
    /// Keeping that live input outside `mc-ui` lets the graph itself remain deterministic and
    /// makes the Java ring's cursor boundary independently testable.
    // java: GuiOverlayDebug.renderLagometer
    pub fn draw_lagometer(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        timer: &FrameTimer,
        frame_limit: i32,
    ) {
        let bottom = ctx.height;
        let top = bottom - LAGOMETER_HEIGHT;
        let data = lagometer_data(timer, frame_limit);

        d.rect(0, top, LAGOMETER_WIDTH, bottom, PLATE_COLOR);
        for bar in data.bars {
            // `drawVerticalLine(x, bottom, bottom - height, colour)` eventually calls
            // `drawRect(x, bottom - height, x + 1, bottom + 1, colour)`, including the bottom
            // endpoint even when the calculated height is zero.
            d.rect(bar.x, bottom - bar.height, bar.x + 1, bottom + 1, bar.color);
        }

        d.rect(
            1,
            bottom - LAGOMETER_HALF_HEIGHT + 1,
            14,
            bottom - LAGOMETER_HALF_HEIGHT + 10,
            PLATE_COLOR,
        );
        font.draw_string(
            d,
            "60",
            2.0,
            (bottom - LAGOMETER_HALF_HEIGHT + 2) as f32,
            INFO_COLOR,
            false,
        );
        horizontal_lagometer_line(d, bottom - LAGOMETER_HALF_HEIGHT, 0xffff_ffff);
        d.rect(1, top + 1, 14, top + 10, PLATE_COLOR);
        font.draw_string(d, "30", 2.0, (top + 2) as f32, INFO_COLOR, false);
        horizontal_lagometer_line(d, top, 0xffff_ffff);
        horizontal_lagometer_line(d, bottom - 1, 0xffff_ffff);
        d.rect(0, top, 1, bottom + 1, 0xffff_ffff);
        d.rect(
            LAGOMETER_WIDTH - 1,
            top,
            LAGOMETER_WIDTH,
            bottom + 1,
            0xffff_ffff,
        );
        if let Some(y) = data.frame_limit_y {
            horizontal_lagometer_line(d, top + y, 0xff00_ffff);
        }
    }

    /// Draw the F3+Shift profiler pie and its legend.
    ///
    /// Java's Minecraft.displayDebugInfo uses a 160-pixel horizontal radius, a half-height
    /// vertical radius, a dark 200-alpha panel, and one triangle fan plus triangle strip per
    /// result. DrawList is a triangle list, so the fans and strips are expanded into CPU-side
    /// triangles here; the resulting geometry is still entirely inspectable without a live GPU.
    // java: Minecraft.displayDebugInfo:1331-1440
    pub fn draw_profiler_chart(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        results: &[ProfilerResult],
    ) {
        let Some(data) = profiler_chart_data(results) else {
            return;
        };

        let radius = PROFILER_RADIUS as f32;
        let center_x = ctx.width as f32 - radius - 10.0;
        let center_y = ctx.height as f32 - radius * 2.0;
        let panel_left = center_x - radius * 1.1;
        let panel_right = center_x + radius * 1.1;
        let panel_top = center_y - radius * 0.6 - 16.0;
        let panel_bottom = center_y + radius * 2.0;
        d.quad_rect(
            &UiTexture::Solid,
            (panel_left, panel_top, panel_right, panel_bottom),
            (0.0, 0.0, 1.0, 1.0),
            PROFILER_PANEL_COLOR,
        );

        for slice in &data.slices {
            let steps = profiler_arc_steps(slice.end_percentage - slice.start_percentage);
            let points: Vec<[f32; 2]> = (0..=steps)
                .map(|step| {
                    let fraction = step as f64 / steps as f64;
                    profiler_point(
                        center_x,
                        center_y,
                        slice.end_percentage
                            - (slice.end_percentage - slice.start_percentage) * fraction,
                    )
                })
                .collect();
            let color = 0xff00_0000 | slice.result.color;
            let rgba = unpack_argb(color);
            for pair in points.windows(2) {
                d.triangle(
                    &UiTexture::Solid,
                    [[center_x, center_y], pair[0], pair[1]],
                    [[0.0, 0.0]; 3],
                    [rgba; 3],
                );
            }

            // java: `begin(5, ...)` is GL_TRIANGLE_STRIP: [point, point + 10]
            // for every angle sample is a filled 10-pixel band in the half-bright colour, the
            // pie's side. The strip's triangles go out in GL's vertex order, and the GUI's
            // back-face cull (vanilla's GL_CULL_FACE, still on from the world pass, which ends
            // with `enableCull()`; `displayDebugInfo` never changes it) keeps the front half
            // only: the band shows under the pie's near edge, never over its top.
            let mut strip = Vec::with_capacity(points.len() * 2);
            for point in points {
                strip.push(point);
                strip.push([point[0], point[1] + 10.0]);
            }
            let outline = unpack_argb(0xff00_0000 | profiler_half_color(slice.result.color));
            d.with_render_state(UiDepth::Overlay, UiCull::Back, |d| {
                for triangle in profiler_strip_triangles(&strip) {
                    d.triangle(&UiTexture::Solid, triangle, [[0.0, 0.0]; 3], [outline; 3]);
                }
            });
        }

        let root_name = if data.root.name == "unspecified" {
            "unspecified ".to_owned()
        } else if data.root.name.is_empty() {
            "ROOT ".to_owned()
        } else {
            format!("[0] {} ", data.root.name)
        };
        let root_y = center_y - radius * 0.5 - 16.0;
        font.draw_string_with_shadow(d, &root_name, center_x - radius, root_y, 0x00ff_ffff);
        let root_percentage = profiler_percent(data.root.total_use_percentage);
        font.draw_string_with_shadow(
            d,
            &root_percentage,
            center_x + radius - font.string_width(&root_percentage) as f32,
            root_y,
            0x00ff_ffff,
        );

        for (index, slice) in data.slices.iter().enumerate() {
            let label = if slice.result.name == "unspecified" {
                format!("[?] {}", slice.result.name)
            } else {
                format!("[{}] {}", index + 1, slice.result.name)
            };
            let y = center_y + radius * 0.5 + index as f32 * 8.0 + 20.0;
            let color = (slice.result.color & 0x00ff_ffff) as i32;
            font.draw_string_with_shadow(d, &label, center_x - radius, y, color);
            let self_percentage = profiler_percent(slice.result.use_percentage);
            let total_percentage = profiler_percent(slice.result.total_use_percentage);
            font.draw_string_with_shadow(
                d,
                &self_percentage,
                center_x + radius - 50.0 - font.string_width(&self_percentage) as f32,
                y,
                color,
            );
            font.draw_string_with_shadow(
                d,
                &total_percentage,
                center_x + radius - font.string_width(&total_percentage) as f32,
                y,
                color,
            );
        }
    }
}

/// Java uses floor(percentage / 4) + 1 pie-ray intervals for one result.
fn profiler_arc_steps(percentage: f64) -> usize {
    (percentage / 4.0).floor().clamp(0.0, 10_000.0) as usize + 1
}

/// One point on Java's horizontally wide, vertically compressed pie.
fn profiler_point(center_x: f32, center_y: f32, percentage: f64) -> [f32; 2] {
    // Java narrows the complete angle to f32 before calling MathHelper's lookup-table sin/cos.
    let angle = (percentage * std::f64::consts::PI * 2.0 / 100.0) as f32;
    [
        center_x + mc_math::sin(angle) * PROFILER_RADIUS as f32,
        center_y - mc_math::cos(angle) * PROFILER_VERTICAL_RADIUS,
    ]
}

/// Darken each RGB component independently, as Java's three right-shifts do.
fn profiler_half_color(color: u32) -> u32 {
    let red = ((color >> 16) & 0xff) >> 1;
    let green = ((color >> 8) & 0xff) >> 1;
    let blue = (color & 0xff) >> 1;
    red << 16 | green << 8 | blue
}

/// GL_TRIANGLE_STRIP's triangles in GL's vertex order: `(v[n], v[n+1],
/// v[n+2])` for even `n`, `(v[n+1], v[n], v[n+2])` for odd `n`, so every triangle keeps the
/// strip's winding and the back-face cull treats it as GL does.
fn profiler_strip_triangles(strip: &[[f32; 2]]) -> impl Iterator<Item = [[f32; 2]; 3]> + '_ {
    strip.windows(3).enumerate().map(|(n, vertices)| {
        if n % 2 == 0 {
            [vertices[0], vertices[1], vertices[2]]
        } else {
            [vertices[1], vertices[0], vertices[2]]
        }
    })
}

/// DecimalFormat("##0.00") followed by Java's literal `%`, used by every chart label.
fn profiler_percent(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.2}%")
    } else {
        "0.00%".to_owned()
    }
}

/// Java's `drawHorizontalLine(0, 239, y, colour)`: the right endpoint is inclusive.
fn horizontal_lagometer_line(d: &mut DrawList, y: i32, color: u32) {
    d.rect(0, y, LAGOMETER_WIDTH, y + 1, color);
}

/// Vanilla prints the enum's own `toString`, which is the lower-case name.
fn facing_name(facing: Facing) -> &'static str {
    facing.name()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;
    use crate::widget::ScaledResolution;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn info() -> DebugInfo {
        DebugInfo {
            position: (123.456_789, 64.0, -987.654_321),
            rotation: (450.0, -12.34),
            memory: (256 * 1024 * 1024, 512 * 1024 * 1024, 1024 * 1024 * 1024),
            ..DebugInfo::default()
        }
    }

    /// Hand-derived from the Java format strings, not from this implementation:
    /// `"XYZ: %.3f / %.5f / %.3f"`. Note the five decimals on Y.
    #[test]
    fn the_xyz_line_uses_three_five_three_decimals() {
        let o = DebugOverlay::new();
        let lines = o.left_column(&info(), None);
        let xyz = lines.iter().find(|l| l.starts_with("XYZ:")).expect("xyz");
        assert_eq!(xyz, "XYZ: 123.457 / 64.00000 / -987.654");
    }

    #[test]
    fn the_header_keeps_vanillas_literal_version_separate_from_the_launched_version() {
        let o = DebugOverlay::new();
        let mut i = info();
        i.version = "1.8.9-custom".into();
        i.brand = "vanilla".into();
        assert_eq!(
            o.left_column(&i, None)[0],
            "Minecraft 1.8.9 (1.8.9-custom/vanilla)"
        );
    }

    /// Java's right column has a runtime-version slot that is independent from the launcher's
    /// `--version` value, followed by host/GPU values in fixed line positions. Keeping each value
    /// deliberately distinctive makes an old `info.version`/adapter-name copy fail loudly.
    #[test]
    fn the_right_column_keeps_runtime_and_graphics_values_in_vanillas_slots() {
        let o = DebugOverlay::new();
        let i = DebugInfo {
            version: "1.8.9-custom".into(),
            java_version: "17.0.12".into(),
            java_bits: 32,
            cpu: "16x Test CPU".into(),
            display: (1920, 1080, "Test Vendor".into()),
            renderer: "Test Renderer".into(),
            api_version: "test-api 1.2".into(),
            ..info()
        };
        let lines = o.right_column(&i);
        assert_eq!(lines[0], "Java: 17.0.12 32bit");
        assert_ne!(lines[0], "Java: 1.8.9-custom 32bit");
        assert_eq!(lines[4], "CPU: 16x Test CPU");
        assert_eq!(lines[6], "Display: 1920x1080 (Test Vendor)");
        assert_eq!(lines[7], "Test Renderer");
        assert_eq!(lines[8], "test-api 1.2");
    }

    #[test]
    fn default_machine_values_still_emit_vanillas_three_graphics_lines() {
        let lines = DebugOverlay::new().right_column(&DebugInfo::default());
        assert!(lines[0].starts_with("Java: <unknown> "));
        assert_eq!(lines[4], "CPU: <unknown>");
        assert_eq!(lines[6], "Display: 854x480 (<unknown>)");
        assert_eq!(lines[7], "<unknown>");
        assert_eq!(lines[8], "<unknown>");
    }

    #[test]
    fn fixed_point_lines_round_halfway_values_away_from_zero_like_java() {
        assert_eq!(format_fixed_half_up(-3.8125, 3), "-3.813");
        assert_eq!(format_fixed_half_up(-0.015625, 5), "-0.01563");
        assert_eq!(format_fixed_half_up(11.25, 1), "11.3");
        assert_eq!(format_fixed_half_up(1.125, 2), "1.13");

        // Negative zero is observable in both Java Formatter and the overlay.
        assert_eq!(format_fixed_half_up(-0.0, 3), "-0.000");
        // The odd-floor control already rounds away from zero under half-even and must not be
        // advanced twice.
        assert_eq!(format_fixed_half_up(1.375, 2), "1.38");

        let o = DebugOverlay::new();
        let mut i = info();
        i.position = (-3.8125, -0.015625, 1.0);
        i.rotation = (11.25, -11.25);
        let lines = o.left_column(&i, None);
        assert!(lines
            .iter()
            .any(|line| line == "XYZ: -3.813 / -0.01563 / 1.000"));
        assert!(lines.iter().any(|line| line.ends_with("(11.3 / -11.3)")));
    }

    #[test]
    fn the_block_and_chunk_lines_use_floor_and_masking_not_truncation() {
        let o = DebugOverlay::new();
        let mut i = info();
        // A negative coordinate is where truncation and flooring differ.
        i.position = (-0.5, 70.25, -17.75);
        let lines = o.left_column(&i, None);
        let block = lines.iter().find(|l| l.starts_with("Block:")).unwrap();
        assert_eq!(block, "Block: -1 70 -18");
        let chunk = lines.iter().find(|l| l.starts_with("Chunk:")).unwrap();
        // -1 & 15 = 15, -1 >> 4 = -1; -18 & 15 = 14, -18 >> 4 = -2.
        assert_eq!(chunk, "Chunk: 15 6 14 in -1 4 -2");
    }

    #[test]
    fn the_facing_line_wraps_the_angles_to_plus_or_minus_one_eighty() {
        let o = DebugOverlay::new();
        let lines = o.left_column(&info(), None);
        let facing = lines.iter().find(|l| l.starts_with("Facing:")).unwrap();
        // 450° wraps to 90°, which faces west.
        assert!(facing.contains("(90.0 / -12.3)"), "{facing}");
        assert!(facing.contains("Towards negative X"), "{facing}");
    }

    /// F-P2-client.gui-11-4/11-5/11-7: the driver-supplied stats occupy the vanilla header
    /// slots unchanged, while the world-backed provider line remains its own sixth entry.
    /// Distinct values make accidental reformatting or the old `Overworld` fallback visible.
    #[test]
    fn the_stats_header_slots_keep_vanilla_values_and_provider_position() {
        let o = DebugOverlay::new();
        let i = DebugInfo {
            renders: "C: 7/4624 (s) D: 8, pC: 003, pU: 0, aB: 0".into(),
            entities: "E: 4/9, B: 2, I: 3".into(),
            dimension: "MultiplayerChunkCache: 12, 34".into(),
            ..info()
        };
        let lines = o.left_column(&i, None);
        assert_eq!(lines[2], "C: 7/4624 (s) D: 8, pC: 003, pU: 0, aB: 0");
        assert_eq!(lines[3], "E: 4/9, B: 2, I: 3");
        assert_eq!(lines[5], "MultiplayerChunkCache: 12, 34");
        assert_ne!(lines[2], "C: 7/99. D: 3, GPU: 1.0/2 MiB (50%), evicted 4");
        assert_ne!(lines[3], "E: 9");
        assert_ne!(lines[5], "Overworld");
    }

    #[test]
    fn entity_facing_preserves_javas_float_evaluation_before_flooring() {
        for (yaw, expected) in [
            (0.0, Facing::South),
            (45.0, Facing::West),
            (90.0, Facing::West),
            (180.0, Facing::North),
            (270.0, Facing::East),
            (-90.0, Facing::East),
            (450.0, Facing::West),
        ] {
            assert_eq!(entity_horizontal_facing(yaw), expected, "yaw {yaw}");
        }

        // This finite f32 lies beside a quadrant boundary after Java's float multiply/divide.
        // Widening first (the old implementation) lands on the other side of that boundary.
        let divergent_yaw = f32::from_bits(0x4bb4_0016);
        assert_eq!(divergent_yaw, 23_593_004.0);
        assert_eq!(entity_horizontal_facing(divergent_yaw), Facing::West);
        assert_eq!(
            Facing::from_angle(f64::from(divergent_yaw)),
            Facing::South,
            "a from_angle mutant must demonstrably disagree at this bit pattern"
        );

        let o = DebugOverlay::new();
        let mut i = info();
        i.rotation.0 = divergent_yaw;
        let facing = o
            .left_column(&i, None)
            .into_iter()
            .find(|line| line.starts_with("Facing:"))
            .expect("facing line");
        assert!(facing.starts_with("Facing: west "), "{facing}");
    }

    #[test]
    fn every_horizontal_facing_gets_its_own_sentence() {
        assert_eq!(facing_description(Facing::North), "Towards negative Z");
        assert_eq!(facing_description(Facing::South), "Towards positive Z");
        assert_eq!(facing_description(Facing::West), "Towards negative X");
        assert_eq!(facing_description(Facing::East), "Towards positive X");
        // Vanilla's switch has no Up/Down arm, so they keep the "Invalid" initialiser.
        assert_eq!(facing_description(Facing::Up), "Invalid");
        assert_eq!(facing_description(Facing::Down), "Invalid");
    }

    /// Hand-derived from `"Mem: % 2d%% %03d/%03dMB"`.
    #[test]
    fn the_memory_lines_are_space_and_zero_padded_exactly_like_java() {
        let o = DebugOverlay::new();
        let lines = o.right_column(&info());
        assert_eq!(lines[1], "Mem:  25% 256/1024MB");
        assert_eq!(lines[2], "Allocated:  50% 512MB");
        // A small number keeps the three-digit zero padding.
        let small = DebugInfo {
            memory: (5 * 1024 * 1024, 9 * 1024 * 1024, 100 * 1024 * 1024),
            ..DebugInfo::default()
        };
        let lines = o.right_column(&small);
        assert_eq!(lines[1], "Mem:  5% 005/100MB");
        assert_eq!(lines[2], "Allocated:  9% 009MB");
    }

    #[test]
    fn zero_max_memory_reports_zero_rather_than_dividing_by_it() {
        let o = DebugOverlay::new();
        let lines = o.right_column(&DebugInfo::default());
        assert_eq!(lines[1], "Mem:  0% 000/000MB");
    }

    #[test]
    fn bytes_to_mb_truncates_like_two_integer_divisions() {
        assert_eq!(bytes_to_mb(0), 0);
        assert_eq!(bytes_to_mb(1024 * 1024 - 1), 0);
        assert_eq!(bytes_to_mb(1024 * 1024), 1);
        assert_eq!(bytes_to_mb(3 * 1024 * 1024 + 500), 3);
    }

    #[test]
    fn reduced_debug_info_collapses_the_left_column_to_eight_lines() {
        let o = DebugOverlay::new();
        let mut i = info();
        i.reduced = true;
        let lines = o.left_column(&i, None);
        assert_eq!(lines.len(), 8);
        assert!(lines.last().unwrap().starts_with("Chunk-relative:"));
        assert!(
            !lines.iter().any(|l| l.starts_with("XYZ:")),
            "no coordinates"
        );
    }

    #[test]
    fn the_looking_at_and_difficulty_lines_only_appear_when_they_have_data() {
        let o = DebugOverlay::new();
        let mut i = info();
        assert!(!o
            .left_column(&i, None)
            .iter()
            .any(|l| l.starts_with("Looking at:")));
        i.looking_at = Some(BlockPos::new(1, 2, 3));
        i.difficulty = Some((1.75, 42));
        let lines = o.left_column(&i, None);
        assert!(lines.iter().any(|l| l == "Looking at: 1 2 3"));
        assert!(
            !lines.iter().any(|l| l.starts_with("Local Difficulty:")),
            "vanilla emits difficulty only for a loaded in-range position"
        );
    }

    /// The right column's registry-name + blockstate-property
    /// list, java: `GuiOverlayDebug.getDebugInfoRight` (`:185`-`:212`). Uses a real block with a
    /// real boolean property from `mc_data::legal_states()` rather than a hand-picked literal, so
    /// the test does not depend on which specific block happens to have one.
    /// Java appends the active group's full `ResourceLocation` after the loaded-world difficulty
    /// line and immediately before the raytrace's `Looking at` line. F4 only changes visibility,
    /// so the UI receives this independently tracked group name.
    #[test]
    fn active_shader_line_is_ordered_and_hidden_by_reduced_debug() {
        let o = DebugOverlay::new();
        let mut i = info();
        i.shader = Some("minecraft:shaders/post/notch.json".to_owned());
        i.looking_at = Some(BlockPos::new(1, 2, 3));
        let lines = o.left_column(&i, None);
        let shader = lines
            .iter()
            .position(|line| line == "Shader: minecraft:shaders/post/notch.json")
            .expect("active shader line");
        let looking = lines
            .iter()
            .position(|line| line == "Looking at: 1 2 3")
            .expect("looking-at line");
        assert_eq!(looking, shader + 1);

        i.shader = None;
        assert!(!o
            .left_column(&i, None)
            .iter()
            .any(|line| line.starts_with("Shader: ")));

        i.shader = Some("minecraft:shaders/post/notch.json".to_owned());
        i.reduced = true;
        assert!(!o
            .left_column(&i, None)
            .iter()
            .any(|line| line.starts_with("Shader: ")));
    }

    #[test]
    fn the_right_column_lists_the_looked_at_blocks_registry_name_and_properties() {
        let o = DebugOverlay::new();
        let mut i = info();
        assert!(
            !o.right_column(&i).iter().any(|l| l.starts_with("§")),
            "no block targeted yet: no registry name or property lines"
        );

        // A state with no properties: just the blank separator and the registry name.
        let stone = mc_data::BlockState::from_raw(1 << 4); // block id 1 (stone), meta 0
        i.looking_at_state = Some(mc_data::block::ActualBlockState::from_stored(stone));
        let lines = o.right_column(&i);
        assert!(lines.iter().any(|l| l == "minecraft:stone"));

        // A state with a real boolean property, found rather than guessed, to prove the
        // true/false colour-coding (`§a`/`§c`, vanilla's `EnumChatFormatting.GREEN`/`RED`).
        let (bool_state, key, value) = mc_data::legal_states()
            .find_map(|s| {
                mc_data::state_properties(s)
                    .iter()
                    .find(|(_, v)| *v == "true" || *v == "false")
                    .map(|&(k, v)| (s, k, v))
            })
            .expect("at least one vanilla block has a boolean property");
        i.looking_at_state = Some(mc_data::block::ActualBlockState::from_stored(bool_state));
        let lines = o.right_column(&i);
        let expected_colour = if value == "true" { "§a" } else { "§c" };
        assert!(
            lines
                .iter()
                .any(|l| *l == format!("{key}: {expected_colour}{value}")),
            "expected a coloured `{key}: {value}` line, got: {lines:?}"
        );
        let name = mc_data::block_info(mc_data::block_of(bool_state)).name;
        assert!(lines.iter().any(|l| l == name));

        // Gated on `reduced`, exactly like vanilla's `isReducedDebug()` early return.
        i.reduced = true;
        assert!(
            !o.right_column(&i).iter().any(|l| l == name),
            "reduced debug info must not show the targeted block"
        );
    }

    #[test]
    fn quartz_f3_uses_enum_to_string_without_changing_model_variant_names() {
        let overlay = DebugOverlay::new();
        let mut input = info();
        for (meta, model_value, debug_value) in [
            (0, "default", "default"),
            (1, "chiseled", "chiseled"),
            (2, "lines_y", "lines"),
            (3, "lines_x", "lines"),
            (4, "lines_z", "lines"),
        ] {
            let actual = mc_data::block::ActualBlockState::from_stored(mc_data::BlockState::new(
                mc_data::BlockId(155),
                meta,
            ));
            assert_eq!(
                actual.get("variant"),
                Some(model_value),
                "the blockstate/model key remains directional"
            );
            input.looking_at_state = Some(actual);
            let lines = overlay.right_column(&input);
            assert!(
                lines
                    .iter()
                    .any(|line| line == &format!("variant: {debug_value}")),
                "quartz metadata {meta} must display EnumType.toString(): {lines:?}"
            );
        }
    }

    #[test]
    fn light_blue_dye_f3_uses_enum_to_string_without_changing_model_keys() {
        let overlay = DebugOverlay::new();
        let mut input = info();
        for block_id in [35, 95, 159, 160, 171] {
            let actual = mc_data::block::ActualBlockState::from_stored(mc_data::BlockState::new(
                mc_data::BlockId(block_id),
                3,
            ));
            assert_eq!(
                actual.get("color"),
                Some("light_blue"),
                "the blockstate/model key remains serialized"
            );
            input.looking_at_state = Some(actual);
            let lines = overlay.right_column(&input);
            assert!(
                lines.iter().any(|line| line == "color: lightBlue"),
                "block {block_id} must display EnumDyeColor.toString(): {lines:?}"
            );
        }
    }

    #[test]
    fn biome_and_light_come_from_a_loaded_chunk_and_are_omitted_otherwise() {
        let o = DebugOverlay::new();
        let mut i = info();
        i.position = (8.0, 5.0, 8.0);
        i.difficulty = Some((1.125, 42));
        let mut world = World::new();
        // Nothing loaded yet.
        let lines = o.left_column(&i, Some(&world));
        assert_eq!(lines[5], "MultiplayerChunkCache: 0, 0");
        assert!(!lines.iter().any(|l| l.starts_with("Biome:")));
        assert!(!lines.iter().any(|l| l.starts_with("Light:")));
        assert!(!lines.iter().any(|l| l.starts_with("Local Difficulty:")));

        // One section of a chunk at (0, 0), with sky light 15 everywhere.
        let mut payload = vec![0u8; 4096 * 2 + 2048 + 2048];
        for b in payload[4096 * 2 + 2048..].iter_mut() {
            *b = 0xFF;
        }
        payload.extend(std::iter::repeat_n(1u8, 256)); // biome id 1 = Plains
        world
            .apply_chunk_data(mc_math::ChunkPos::new(0, 0), true, 0b1, &payload)
            .expect("chunk");
        let lines = o.left_column(&i, Some(&world));
        assert_eq!(lines[5], "MultiplayerChunkCache: 1, 1");
        let biome = lines
            .iter()
            .find(|l| l.starts_with("Biome:"))
            .expect("biome");
        assert_eq!(biome, "Biome: Plains");
        let light = lines
            .iter()
            .find(|l| l.starts_with("Light:"))
            .expect("light");
        assert_eq!(light, "Light: 15 (15 sky, 0 block)");
        assert!(lines.iter().any(|l| l == "Local Difficulty: 1.13 (Day 42)"));

        // The chunk exists, but vanilla's `World.isBlockLoaded(BlockPos)` rejects both void
        // positions before looking it up.
        for y in [-1.0, 256.0] {
            i.position.1 = y;
            let lines = o.left_column(&i, Some(&world));
            assert!(!lines.iter().any(|l| l.starts_with("Biome:")), "y={y}");
            assert!(!lines.iter().any(|l| l.starts_with("Light:")), "y={y}");
            assert!(
                !lines.iter().any(|l| l.starts_with("Local Difficulty:")),
                "y={y}"
            );
        }
    }

    #[test]
    fn the_line_plate_uses_vanillas_argb_literal() {
        assert_eq!(PLATE_COLOR, (-1_873_784_752_i32) as u32);
    }

    #[test]
    fn lagometer_colours_follow_the_green_yellow_red_java_ramp() {
        assert_eq!(frame_color(0, 0, 30, 60), 0xff00_ff00);
        assert_eq!(frame_color(15, 0, 30, 60), 0xff7f_ff00);
        assert_eq!(frame_color(30, 0, 30, 60), 0xffff_ff00);
        assert_eq!(frame_color(45, 0, 30, 60), 0xffff_7f00);
        assert_eq!(frame_color(60, 0, 30, 60), 0xffff_0000);
    }

    #[test]
    fn lagometer_data_preserves_the_frame_timer_wrap_order_and_cap_guide() {
        let mut timer = FrameTimer::new();
        for frame in 1..=241 {
            timer.add_frame(i64::from(frame) * 16_666_666);
        }
        let data = lagometer_data(&timer, 60);
        assert_eq!(data.bars.len(), 239);
        assert_eq!(
            data.bars.first().map(|bar| (bar.x, bar.height)),
            Some((0, 90))
        );
        assert_eq!(
            data.bars.last().map(|bar| (bar.x, bar.height)),
            Some((238, 7_230))
        );
        assert_eq!(data.frame_limit_y, Some(30));
        assert_eq!(lagometer_data(&timer, 121).frame_limit_y, None);
    }

    #[test]
    fn lagometer_draw_includes_zero_height_columns_and_the_vanilla_border_extent() {
        let mut timer = FrameTimer::new();
        timer.add_frame(0);
        let mut font = uniform_font();
        let mut draw = DrawList::new();
        let c = ctx();
        DebugOverlay::new().draw_lagometer(&mut draw, &mut font, &c, &timer, 120);

        // The zero-height sample is still a one-pixel `drawVerticalLine` at the screen bottom.
        assert!(draw
            .vertices()
            .iter()
            .any(|v| v.pos == [0.0, c.height as f32]));
        // Java's two vertical borders include the bottom endpoint (`height + 1`).
        assert!(draw
            .vertices()
            .iter()
            .any(|v| v.pos == [(LAGOMETER_WIDTH - 1) as f32, (c.height + 1) as f32]));
    }

    /// Java's String.hashCode() consumes UTF-16 code units, then masks the signed hash before
    /// adding the fixed colour base. A UTF-8-byte implementation gets the emoji control wrong.
    #[test]
    fn profiler_result_colors_use_javas_utf16_hash() {
        assert_eq!(profiler_color("tick"), 0x0066_44cc);
        assert_eq!(profiler_color("😀"), 0x004e_4c66);

        let utf8_hash = "😀".as_bytes().iter().fold(0_i32, |hash, byte| {
            hash.wrapping_mul(31).wrapping_add(i32::from(*byte))
        });
        let utf8_color = ((utf8_hash as u32) & 0x00aa_aaaa) + 0x0044_4444;
        assert_ne!(
            profiler_color("😀"),
            utf8_color,
            "the negative control must distinguish Java UTF-16 from Rust UTF-8"
        );
    }

    #[test]
    fn profiler_outline_darken_shifts_each_rgb_component_independently() {
        assert_eq!(profiler_half_color(0x0067_45cd), 0x0033_2266);
        assert_ne!(
            profiler_half_color(0x0067_45cd),
            0x0033_a2e6,
            "a packed integer shift would leak red's low bit into green"
        );
    }

    #[test]
    fn profiler_percent_keeps_vanillas_literal_suffix() {
        assert_eq!(profiler_percent(100.0), "100.00%");
        assert_eq!(profiler_percent(0.5), "0.50%");
    }

    #[test]
    fn profiler_chart_data_keeps_java_percentage_order_and_skips_invalid_rows() {
        let results = [
            ProfilerResult::root(),
            ProfilerResult::new("tick", 60.0, 30.0),
            ProfilerResult::new("render", 40.0, 70.0),
            ProfilerResult::new("zero", 0.0, 0.0),
            ProfilerResult::new("nan", f64::NAN, 0.0),
        ];
        let data = profiler_chart_data(&results).expect("root result");
        assert_eq!(data.root.name, "");
        assert_eq!(data.slices.len(), 2);
        assert_eq!(
            (
                data.slices[0].start_percentage,
                data.slices[0].end_percentage
            ),
            (0.0, 60.0)
        );
        assert_eq!(
            (
                data.slices[1].start_percentage,
                data.slices[1].end_percentage
            ),
            (60.0, 100.0)
        );
        assert_eq!(data.slices[0].result.total_use_percentage, 30.0);
        assert!(profiler_chart_data(&[]).is_none());
    }

    /// The first four vertices are the Java panel's four corners. The pie itself then has one
    /// triangle per fan interval, so a 50/50 fixture must produce both colours in the CPU mesh.
    #[test]
    fn profiler_chart_draws_vanilla_panel_pie_and_legend_without_a_gpu() {
        let mut overlay = DebugOverlay::new();
        overlay.on_f3(Modifiers::SHIFT);
        let results = [
            ProfilerResult::root(),
            ProfilerResult::new("tick", 50.0, 50.0),
            ProfilerResult::new("render", 50.0, 50.0),
        ];
        let mut font = uniform_font();
        let mut draw = DrawList::new();
        overlay.draw_profiler_chart(&mut draw, &mut font, &ctx(), &results);

        assert_eq!(draw.vertices()[0].pos, [508.0, 48.0]);
        assert_eq!(draw.vertices()[1].pos, [508.0, 480.0]);
        assert_eq!(draw.vertices()[2].pos, [860.0, 48.0]);
        assert_eq!(draw.vertices()[4].pos, [860.0, 480.0]);
        let tick = unpack_argb(0xff66_44cc);
        let render = unpack_argb(0xff4e_44c6);
        assert!(
            draw.vertices().iter().any(|v| v.color == tick),
            "the first slice colour must reach the CPU mesh"
        );
        assert!(
            draw.vertices().iter().any(|v| v.color == render),
            "the second slice colour must reach the CPU mesh"
        );
        assert!(
            draw.vertices().iter().any(|v| v.pos[0] >= 508.0),
            "the legend must be emitted beside the panel"
        );
    }

    /// Twice the signed area with y down: negative is counter-clockwise on screen once the GUI's
    /// y-flipping ortho has mapped it, which is GL's (and wgpu's) front face.
    fn signed_area_y_down(triangle: [[f32; 2]; 3]) -> f32 {
        let [a, b, c] = triangle;
        (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    }

    /// java: each slice's side is `begin(5, ...)`, a GL_TRIANGLE_STRIP of
    /// [point, point + 10] pairs, so a filled 10-pixel band in the half colour (it was drawn
    /// as 1-pixel lines). The strip goes out in GL's triangle order under the back-face cull, as
    /// vanilla's inherited GL_CULL_FACE: the triangles that survive are exactly those on the
    /// near (lower) half of the pie, and those on the far half, which would paint over the
    /// pie's top, are culled. One slice of 100% covers the whole rim.
    #[test]
    fn profiler_pie_rim_is_a_ten_pixel_strip_culled_to_its_near_half() {
        let mut overlay = DebugOverlay::new();
        overlay.on_f3(Modifiers::SHIFT);
        let results = [
            ProfilerResult::root(),
            ProfilerResult::new("tick", 100.0, 100.0),
        ];
        let mut font = uniform_font();
        let mut draw = DrawList::new();
        overlay.draw_profiler_chart(&mut draw, &mut font, &ctx(), &results);

        let rim = unpack_argb(0xff00_0000 | profiler_half_color(profiler_color("tick")));
        let mut triangles = Vec::new();
        for batch in draw.batches() {
            let vertices = &draw.vertices()[batch.range.clone()];
            for triangle in vertices.chunks_exact(3) {
                if triangle.iter().all(|vertex| vertex.color == rim) {
                    assert_eq!(batch.cull, UiCull::Back, "the rim is drawn under the cull");
                    triangles.push([triangle[0].pos, triangle[1].pos, triangle[2].pos]);
                }
            }
        }
        let steps = profiler_arc_steps(100.0);
        assert_eq!(
            triangles.len(),
            2 * steps,
            "one strip of {} vertices",
            2 * (steps + 1)
        );
        // `ctx()` is 854x480: the pie's centre is 320 pixels above the bottom.
        let center_y = 480.0 - 2.0 * PROFILER_RADIUS as f32;
        // Where the arc turns (90 and 270 degrees) a segment's two samples are as good as one
        // above the other: those triangles are slivers far below a pixel, left out here.
        let (mut front, mut back) = (0, 0);
        for triangle in triangles {
            let area = signed_area_y_down(triangle);
            let top = triangle.iter().map(|v| v[1]).fold(f32::MAX, f32::min);
            let bottom = triangle.iter().map(|v| v[1]).fold(f32::MIN, f32::max);
            assert!(
                bottom - top >= 10.0 - 1e-3,
                "each triangle spans the 10-pixel band"
            );
            if area < -1.0 {
                front += 1;
                assert!(
                    top > center_y,
                    "a kept triangle lies on the near half: {triangle:?}"
                );
            } else if area > 1.0 {
                back += 1;
                assert!(
                    bottom < center_y + 10.0,
                    "a culled triangle lies on the far half: {triangle:?}"
                );
            }
        }
        assert!(
            front >= steps - 2,
            "about half the strip is kept: {front} of {}",
            2 * steps
        );
        assert!(
            back >= steps - 2,
            "about half the strip is culled: {back} of {}",
            2 * steps
        );
    }

    /// GL's strip order keeps one winding for every triangle of a strip. Control: the same
    /// vertices taken as plain consecutive triples alternate their winding, so the cull would
    /// punch every second triangle out of the visible band.
    #[test]
    fn profiler_strip_triangles_keep_one_winding_in_gl_order() {
        let strip = [
            [0.0, 0.0],
            [0.0, 10.0],
            [5.0, 0.5],
            [5.0, 10.5],
            [10.0, 1.5],
            [10.0, 11.5],
        ];
        let areas: Vec<f32> = profiler_strip_triangles(&strip)
            .map(signed_area_y_down)
            .collect();
        assert_eq!(areas.len(), 4);
        assert!(areas.iter().all(|&area| area < 0.0), "{areas:?}");
        let plain: Vec<f32> = strip
            .windows(3)
            .map(|v| signed_area_y_down([v[0], v[1], v[2]]))
            .collect();
        assert!(plain.iter().any(|&area| area > 0.0), "{plain:?}");
    }

    #[test]
    fn profiler_chart_can_paint_the_root_while_the_profiler_warms_up() {
        let mut overlay = DebugOverlay::new();
        overlay.on_f3(Modifiers::SHIFT);
        let mut font = uniform_font();
        let mut draw = DrawList::new();
        overlay.draw_profiler_chart(&mut draw, &mut font, &ctx(), &[ProfilerResult::root()]);
        assert!(!draw.is_empty());
        assert_eq!(
            profiler_chart_data(&[ProfilerResult::root()])
                .expect("root")
                .slices
                .len(),
            0
        );
    }

    #[test]
    fn draw_routes_profiler_chart_only_to_shift_mode() {
        let results = vec![
            ProfilerResult::root(),
            ProfilerResult::new("tick", 100.0, 100.0),
        ];
        let panel_corner = [508.0, 48.0];
        let panel_color = unpack_argb(PROFILER_PANEL_COLOR);

        let mut normal = DebugOverlay::new();
        normal.visible = true;
        let mut normal_info = info();
        normal_info.profiler = results.clone();
        let mut normal_font = uniform_font();
        let mut normal_draw = DrawList::new();
        normal.draw(
            &mut normal_draw,
            &mut normal_font,
            &ctx(),
            &normal_info,
            None,
        );
        assert!(!normal_draw
            .vertices()
            .iter()
            .any(|vertex| vertex.pos == panel_corner && vertex.color == panel_color));

        let mut shift = DebugOverlay::new();
        shift.on_f3(Modifiers::SHIFT);
        let mut shift_info = info();
        shift_info.profiler = results;
        let mut shift_font = uniform_font();
        let mut shift_draw = DrawList::new();
        shift.draw(&mut shift_draw, &mut shift_font, &ctx(), &shift_info, None);
        assert!(shift_draw
            .vertices()
            .iter()
            .any(|vertex| vertex.pos == panel_corner && vertex.color == panel_color));
    }

    /// F-P2-client.gui-11-13: every F3 edge overwrites the two auxiliary debug modes with the
    /// modifier snapshot, rather than leaving a previous Alt/Shift mode latched.
    #[test]
    fn f3_modifier_edges_select_and_clear_auxiliary_debug_modes() {
        let mut overlay = DebugOverlay::new();
        assert!(!overlay.visible);
        assert!(!overlay.profiler_chart_visible());
        assert!(!overlay.lagometer_visible());

        overlay.on_f3(Modifiers::SHIFT);
        assert!(overlay.visible);
        assert!(overlay.profiler_chart_visible());
        assert!(!overlay.lagometer_visible());

        overlay.on_f3(Modifiers {
            ctrl: false,
            shift: false,
            alt: true,
        });
        assert!(!overlay.visible);
        assert!(!overlay.profiler_chart_visible());
        assert!(overlay.lagometer_visible());

        overlay.on_f3(Modifiers::NONE);
        assert!(overlay.visible);
        assert!(!overlay.profiler_chart_visible());
        assert!(!overlay.lagometer_visible());
    }

    #[test]
    fn profiler_number_keys_drill_down_and_backtrack_without_selecting_remainder() {
        let mut overlay = DebugOverlay::new();
        let results = [
            ProfilerResult::new("root", 100.0, 100.0),
            ProfilerResult::new("tick", 70.0, 70.0),
            ProfilerResult::new("render", 30.0, 30.0),
            ProfilerResult::new("unspecified", 1.0, 1.0),
        ];
        assert_eq!(overlay.profiler_path(), "root");
        overlay.update_profiler_name(1, &results);
        assert_eq!(overlay.profiler_path(), "root/tick");
        overlay.update_profiler_name(0, &results);
        assert_eq!(overlay.profiler_path(), "root");
        overlay.update_profiler_name(3, &results);
        assert_eq!(overlay.profiler_path(), "root");

        // A selected child can itself be backed out, while root's key-0 behavior remains a no-op
        // just like Java's `lastIndexOf(".") == -1` case.
        overlay.update_profiler_name(2, &results);
        assert_eq!(overlay.profiler_path(), "root/render");
        overlay.update_profiler_name(0, &results);
        assert_eq!(overlay.profiler_path(), "root");
        overlay.update_profiler_name(0, &results);
        assert_eq!(overlay.profiler_path(), "root");
    }

    /// The driver feeds seconds from the shell into this Java-shaped nanosecond history. Invalid
    /// values are defensive zero samples and must not poison the ring with a wrapped integer.
    #[test]
    fn frame_time_samples_are_nanoseconds_and_invalid_values_are_zero() {
        let mut overlay = DebugOverlay::new();
        overlay.record_frame_time(0.025);
        overlay.record_frame_time(f32::NAN);
        assert_eq!(overlay.frame_timer().index(), 2);
        assert_eq!(overlay.frame_timer().frames()[0], 25_000_000);
        assert_eq!(overlay.frame_timer().frames()[1], 0);
    }

    #[test]
    fn a_hidden_overlay_draws_nothing_and_a_visible_one_plates_every_line() {
        let mut o = DebugOverlay::new();
        let mut font = uniform_font();
        let mut d = DrawList::new();
        o.draw(&mut d, &mut font, &ctx(), &info(), None);
        assert!(d.is_empty());

        o.visible = true;
        o.draw(&mut d, &mut font, &ctx(), &info(), None);
        assert!(!d.is_empty());
        // The first plate starts at (1, 1) and is font_height tall.
        assert!(d.vertices().iter().any(|v| v.pos == [1.0, 1.0]));
        // Blank lines draw neither a plate nor text.
        let plates = d
            .batches()
            .iter()
            .filter(|b| b.texture == crate::widget::UiTexture::Solid)
            .map(|b| b.range.len() / 6)
            .sum::<usize>();
        let non_empty = o
            .left_column(&info(), None)
            .iter()
            .filter(|l| !l.is_empty())
            .count()
            + o.right_column(&info())
                .iter()
                .filter(|l| !l.is_empty())
                .count();
        assert_eq!(plates, non_empty);
    }

    #[test]
    fn the_right_column_is_flush_with_the_right_edge() {
        let mut o = DebugOverlay::new();
        o.visible = true;
        let mut font = uniform_font();
        let mut d = DrawList::new();
        let c = ctx();
        o.draw(&mut d, &mut font, &c, &info(), None);
        let max_x = d
            .vertices()
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::MIN, f32::max);
        // The plate ends at `width - 2 + 1`.
        assert_eq!(max_x, (c.width - 1) as f32);
    }
}
