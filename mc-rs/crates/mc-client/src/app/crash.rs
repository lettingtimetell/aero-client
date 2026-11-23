//! Crash reports and the panic handler.
//!
//! A port of `net/minecraft/crash/CrashReport.java` (395 lines) and
//! `CrashReportCategory.java` (308 lines), plus the `Minecraft.displayCrashReport` wrapper that
//! decides where the file goes.
//!
//! # What changes, and why
//!
//! Java's report is built around a `Throwable`: it holds the exception, prints its stack trace,
//! and prunes category stack traces against the exception's own. Rust has no exception object —
//! a crash arrives either as a panic (payload + [`std::panic::Location`] + a
//! [`std::backtrace::Backtrace`]) or as an [`anyhow::Error`] chain. So [`CrashReport`] stores the
//! *rendered* cause and the *captured* stack trace rather than a live throwable, and
//! `CrashReport.makeCategoryDepth`'s stack-trace-pruning arithmetic — which exists only to line a
//! category's `Thread.currentThread().getStackTrace()` up against the throwable's — has no
//! meaning here and is deliberately not ported. Everything that produces *output* is ported
//! exactly, because the output format is the part a human reads.
//!
//! # Layout, byte for byte with vanilla
//!
//! ```text
//! ---- Minecraft Crash Report ----
//! // <witty comment>
//!
//! Time: <timestamp>
//! Description: <description>
//!
//! <cause>
//!
//! A detailed walkthrough of the error, its code path and all known details is as follows:
//! --------------------------------------------------------------------------------------- (87)
//!
//! -- Head --
//! Stacktrace:
//! <TAB>at <frame>
//!
//! -- <category> --
//! Details:
//! <TAB><key>: <value>
//!
//! -- System Details --
//! Details:
//! <TAB>Minecraft Version: 1.8.9
//! ```

use std::fmt::Write as _;
use std::panic::PanicHookInfo;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use mc_data::{state_properties, BlockId, BlockState};
use mc_math::BlockPos;

/// One `key: value` line inside a category.
///
/// java: `CrashReportCategory$Entry`. The two special renderings are vanilla's: a missing value
/// prints `~~NULL~~` and an error value prints `~~ERROR~~ <Type>: <message>`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Entry {
    key: String,
    value: String,
}

impl Entry {
    /// java: `Entry(String, Object)` with a non-null, non-`Throwable` value.
    fn value(key: impl Into<String>, value: impl Into<String>) -> Entry {
        Entry {
            key: key.into(),
            value: value.into(),
        }
    }

    /// java: `Entry(String, null)`.
    fn null(key: impl Into<String>) -> Entry {
        Entry {
            key: key.into(),
            value: "~~NULL~~".to_owned(),
        }
    }

    /// java: `Entry(String, Throwable)` — `"~~ERROR~~ " + getSimpleName() + ": " + getMessage()`.
    fn error(key: impl Into<String>, kind: &str, message: &str) -> Entry {
        Entry {
            key: key.into(),
            value: format!("~~ERROR~~ {kind}: {message}"),
        }
    }
}

/// A named group of `key: value` details, optionally with a stack trace.
///
/// java: `CrashReportCategory`.
#[derive(Debug, Clone, Default)]
pub struct CrashReportCategory {
    name: String,
    children: Vec<Entry>,
    stack_trace: Vec<String>,
}

impl CrashReportCategory {
    /// A new, empty category.
    ///
    /// java: `CrashReportCategory(CrashReport, String)`. The back-pointer to the owning report is
    /// dropped: vanilla only uses it for the pruning arithmetic this port does not reproduce, and
    /// keeping it would be a parent pointer of exactly the kind the arena-index rule exists to avoid.
    #[must_use]
    pub fn new(name: impl Into<String>) -> CrashReportCategory {
        CrashReportCategory {
            name: name.into(),
            children: Vec::new(),
            stack_trace: Vec::new(),
        }
    }

    /// This category's name, as it appears in the `-- name --` header.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add a detail line. java: `addCrashSection`.
    pub fn add(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.children.push(Entry::value(key, value));
        self
    }

    /// Add a detail line whose value could not be produced. java: a `null` value.
    pub fn add_null(&mut self, key: impl Into<String>) -> &mut Self {
        self.children.push(Entry::null(key));
        self
    }

    /// Add a detail line for something that failed while being computed.
    ///
    /// java: `addCrashSectionCallable` catching, then `addCrashSectionThrowable`. Vanilla runs a
    /// `Callable` and converts a throw into this line; the Rust equivalent is a caller handing us
    /// the error it already has.
    pub fn add_error(&mut self, key: impl Into<String>, kind: &str, message: &str) -> &mut Self {
        self.children.push(Entry::error(key, kind, message));
        self
    }

    /// Add a detail line from a [`Result`], which is the direct analogue of
    /// `addCrashSectionCallable`: the value on success, an `~~ERROR~~` line on failure.
    pub fn add_result<T: std::fmt::Display, E: std::error::Error>(
        &mut self,
        key: impl Into<String>,
        value: Result<T, E>,
    ) -> &mut Self {
        match value {
            Ok(value) => self.add(key, value.to_string()),
            Err(error) => {
                // Rust has no `getSimpleName()`; the type name is the closest equivalent and is
                // what a reader actually wants to see.
                let kind = std::any::type_name::<E>()
                    .rsplit("::")
                    .next()
                    .unwrap_or("Error");
                let kind = kind.to_owned();
                self.add_error(key, &kind, &error.to_string())
            }
        }
    }

    /// Attach a stack trace, printed under a `Stacktrace:` heading.
    pub fn set_stack_trace(&mut self, frames: Vec<String>) -> &mut Self {
        self.stack_trace = frames;
        self
    }

    /// The frames attached to this category. java: `getStackTrace`.
    #[must_use]
    pub fn stack_trace(&self) -> &[String] {
        &self.stack_trace
    }

    /// Render this category into `builder`.
    ///
    /// java: `appendToStringBuilder` — note that `Details:` is emitted **unconditionally**, even
    /// with no children, so an empty category still prints two lines. Reproduced.
    pub fn append_to(&self, builder: &mut String) {
        builder.push_str("-- ");
        builder.push_str(&self.name);
        builder.push_str(" --\n");
        builder.push_str("Details:");

        for entry in &self.children {
            builder.push_str("\n\t");
            builder.push_str(&entry.key);
            builder.push_str(": ");
            builder.push_str(&entry.value);
        }

        if !self.stack_trace.is_empty() {
            builder.push_str("\nStacktrace:");
            for frame in &self.stack_trace {
                builder.push_str("\n\tat ");
                builder.push_str(frame);
            }
        }
    }
}

/// Vanilla's `World: (…), Chunk: (…), Region: (…)` block-location line.
///
/// java: `CrashReportCategory.getCoordinateInfo(BlockPos)` (`:28`). Every shift here is Java's,
/// including the arithmetic-shift behaviour on negatives that makes `-1 >> 4 == -1`.
#[must_use]
pub fn coordinate_info(x: i32, y: i32, z: i32) -> String {
    let mut out = String::new();
    let _ = write!(out, "World: ({x},{y},{z})");
    out.push_str(", ");

    {
        let chunk_x = x >> 4;
        let chunk_z = z >> 4;
        let in_chunk_x = x & 15;
        let section_y = y >> 4;
        let in_chunk_z = z & 15;
        let min_x = chunk_x << 4;
        let min_z = chunk_z << 4;
        let max_x = ((chunk_x + 1) << 4) - 1;
        let max_z = ((chunk_z + 1) << 4) - 1;
        let _ = write!(
            out,
            "Chunk: (at {in_chunk_x},{section_y},{in_chunk_z} in {chunk_x},{chunk_z}; \
             contains blocks {min_x},0,{min_z} to {max_x},255,{max_z})"
        );
    }
    out.push_str(", ");

    {
        let region_x = x >> 9;
        let region_z = z >> 9;
        let min_chunk_x = region_x << 5;
        let min_chunk_z = region_z << 5;
        let max_chunk_x = ((region_x + 1) << 5) - 1;
        let max_chunk_z = ((region_z + 1) << 5) - 1;
        let min_x = region_x << 9;
        let min_z = region_z << 9;
        let max_x = ((region_x + 1) << 9) - 1;
        let max_z = ((region_z + 1) << 9) - 1;
        let _ = write!(
            out,
            "Region: ({region_x},{region_z}; contains chunks {min_chunk_x},{min_chunk_z} to \
             {max_chunk_x},{max_chunk_z}, blocks {min_x},0,{min_z} to {max_x},255,{max_z})"
        );
    }

    out
}

/// Add the three legacy-block details emitted by vanilla's
/// `CrashReportCategory.addBlockInfo(category, pos, block, blockData)` overload.
///
/// The block identity is read from the same generated registry that drives the client, so its
/// numeric id, unlocalized name, and Java implementation class are the values vanilla would
/// obtain from `Block.getIdFromBlock`, `getUnlocalizedName`, and `getCanonicalName`.
///
/// This deliberately accepts an `i32` metadata value: vanilla accepts an `int`, and its negative
/// `Unknown?` branch is useful when a caller is reporting malformed input rather than a decoded
/// [`BlockState`].
pub fn add_block_info_with_metadata(
    category: &mut CrashReportCategory,
    pos: BlockPos,
    block: BlockId,
    block_data: i32,
) -> &mut CrashReportCategory {
    let info = block.info();
    category.add(
        "Block type",
        format!(
            "ID #{} ({} // {})",
            block.0, info.unlocalized, info.java_class
        ),
    );
    if block_data < 0 {
        category.add("Block data value", format!("Unknown? (Got {block_data})"));
    } else {
        // java: `String.format("%4s", Integer.toBinaryString(blockData)).replace(" ", "0")`.
        // Width four is a minimum, not a mask: metadata outside the normal 0..=15 range keeps
        // all of its bits, exactly as the Java helper does.
        let binary = format!("{block_data:b}");
        category.add(
            "Block data value",
            format!("{block_data} / 0x{block_data:X} / 0b{binary:0>4}"),
        );
    }
    category.add("Block location", coordinate_info(pos.x, pos.y, pos.z))
}

/// Add the state details emitted by vanilla's
/// `CrashReportCategory.addBlockInfo(category, pos, state)` overload.
///
/// Java's `BlockStateBase.toString()` writes the registry name and, when present, its property
/// map in name-sorted order. `mc_data` preserves that map for every 1.8.9 packed state, including
/// variants that are invalid on the wire; this builds the same visible string without depending
/// on a renderer-specific state type.
pub fn add_block_info(
    category: &mut CrashReportCategory,
    pos: BlockPos,
    state: BlockState,
) -> &mut CrashReportCategory {
    let info = state.block().info();
    let properties = state_properties(state);
    let mut rendered = info.name.to_owned();
    if !properties.is_empty() {
        rendered.push('[');
        for (index, (name, value)) in properties.iter().enumerate() {
            if index != 0 {
                rendered.push(',');
            }
            let _ = write!(rendered, "{name}={value}");
        }
        rendered.push(']');
    }
    category
        .add("Block", rendered)
        .add("Block location", coordinate_info(pos.x, pos.y, pos.z))
}

/// The `%.2f,%.2f,%.2f - <block info>` form.
///
/// java: `CrashReportCategory.getCoordinateInfo(double, double, double)` (`:23`), whose
/// `new BlockPos(x, y, z)` **floors** each coordinate.
#[must_use]
pub fn coordinate_info_precise(x: f64, y: f64, z: f64) -> String {
    let block = coordinate_info(
        mc_math::floor_f64(x),
        mc_math::floor_f64(y),
        mc_math::floor_f64(z),
    );
    format!(
        "{},{},{} - {block}",
        format_java_decimal(x, 2),
        format_java_decimal(y, 2),
        format_java_decimal(z, 2),
    )
}

/// Format a finite `f64` as Java's `String.format("%.<precision>f", value)` does.
///
/// `f64::to_string` is Rust's shortest round-tripping decimal representation, the same starting
/// point Java's formatter uses before applying decimal HALF_UP rounding. Rust's precision
/// formatter instead rounds the binary float directly, so values such as `0.125` used to render
/// as `0.12` here where Java renders `0.13`.
fn format_java_decimal(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-Infinity".to_owned()
        } else {
            "Infinity".to_owned()
        };
    }

    let negative = value.is_sign_negative();
    let shortest = value.abs().to_string();
    let (mantissa, exponent) = shortest
        .split_once(['e', 'E'])
        .map_or((shortest.as_str(), 0_i32), |(mantissa, exponent)| {
            (mantissa, exponent.parse().unwrap_or(0))
        });
    let decimal = mantissa.find('.').unwrap_or(mantissa.len());
    let digits: Vec<u8> = mantissa.bytes().filter(u8::is_ascii_digit).collect();
    let decimal = decimal as i32 + exponent;

    let digit_at = |index: i32| -> u8 {
        usize::try_from(index)
            .ok()
            .and_then(|index| digits.get(index).copied())
            .unwrap_or(b'0')
    };
    let whole_len = usize::try_from(decimal).unwrap_or(0);
    let mut rendered: Vec<u8> = if whole_len == 0 {
        vec![b'0']
    } else {
        (0..whole_len).map(|index| digit_at(index as i32)).collect()
    };
    let whole_len = rendered.len();
    rendered.extend((0..precision).map(|offset| digit_at(decimal + offset as i32)));

    // Java Formatter's decimal rounding mode is HALF_UP, i.e. ties increase the magnitude for
    // both signs. Increment the un-signed digit run, then prepend the original sign below.
    if digit_at(decimal + precision as i32) >= b'5' {
        let mut index = rendered.len();
        loop {
            if index == 0 {
                rendered.insert(0, b'1');
                break;
            }
            index -= 1;
            if rendered[index] == b'9' {
                rendered[index] = b'0';
            } else {
                rendered[index] += 1;
                break;
            }
        }
    }

    // A carry can have inserted one digit at the front, moving the decimal point with it.
    let whole_len = whole_len + (rendered.len() - (whole_len + precision));
    let mut out = String::with_capacity(rendered.len() + usize::from(negative) + 1);
    if negative {
        out.push('-');
    }
    for (index, digit) in rendered.into_iter().enumerate() {
        if index == whole_len {
            out.push('.');
        }
        out.push(char::from(digit));
    }
    if precision == 0 {
        debug_assert!(!out.ends_with('.'));
    }
    out
}

/// java: `CrashReport.getWittyComment`'s 34-entry array.
const WITTY_COMMENTS: [&str; 34] = [
    "Who set us up the TNT?",
    "Everything's going to plan. No, really, that was supposed to happen.",
    "Uh... Did I do that?",
    "Oops.",
    "Why did you do that?",
    "I feel sad now :(",
    "My bad.",
    "I'm sorry, Dave.",
    "I let you down. Sorry :(",
    "On the bright side, I bought you a teddy bear!",
    "Daisy, daisy...",
    "Oh - I know what I did wrong!",
    "Hey, that tickles! Hehehe!",
    "I blame Dinnerbone.",
    "You should try our sister game, Minceraft!",
    "Don't be sad. I'll do better next time, I promise!",
    "Don't be sad, have a hug! <3",
    "I just don't know what went wrong :(",
    "Shall we play a game?",
    "Quite honestly, I wouldn't worry myself about that.",
    "I bet Cylons wouldn't have this problem.",
    "Sorry :(",
    "Surprise! Haha. Well, this is awkward.",
    "Would you like a cupcake?",
    "Hi. I'm Minecraft, and I'm a crashaholic.",
    "Ooh. Shiny.",
    "This doesn't make any sense!",
    "Why is it breaking :(",
    "Don't do that.",
    "Ouch. That hurt :(",
    "You're mean.",
    "This is a token for 1 free hug. Redeem at your nearest Mojangsta: [~~HUG~~]",
    "There are four lights!",
    "But it works on my machine.",
];

/// One of vanilla's 34 witty comments, picked the way vanilla picks it.
///
/// java: `astring[(int)(System.nanoTime() % (long)astring.length)]`. Rust's `%` on a
/// non-negative `u128` cannot go negative, so vanilla's `try`/`catch` fallback
/// (`"Witty comment unavailable :("`) is unreachable and is not ported.
#[must_use]
pub fn witty_comment() -> &'static str {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    WITTY_COMMENTS[(nanos % WITTY_COMMENTS.len() as u128) as usize]
}

/// Split a Unix timestamp into `(year, month, day, hour, minute, second)` in **UTC**.
///
/// This remains the dependency-free fallback for platforms where the host clock formatter cannot
/// be queried. The date arithmetic is the standard `civil_from_days` algorithm, exact for the whole
/// proleptic Gregorian range. Vanilla's normal path uses the platform's local time via
/// `SimpleDateFormat`; `local_datetime` supplies that path below.
#[must_use]
pub fn civil_from_unix(seconds: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = seconds.div_euclid(86_400);
    let secs_of_day = seconds.rem_euclid(86_400);

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };

    (
        year,
        m as u32,
        d as u32,
        (secs_of_day / 3600) as u32,
        ((secs_of_day % 3600) / 60) as u32,
        (secs_of_day % 60) as u32,
    )
}

/// Java 8's default `DateFormat.SHORT` date/time, in the host's local time zone and locale.
///
/// Windows uses the current-culture short-date/short-time patterns that the Java 8 host reports,
/// with Java's two-digit year convention. Unix uses the host `date` formatter. If the platform
/// formatter is unavailable, the explicit UTC fallback keeps the report useful and truthful
/// rather than silently labelling UTC as local time.
#[must_use]
pub fn format_timestamp(seconds: i64) -> String {
    format_local_short_timestamp(seconds).unwrap_or_else(|| {
        let (y, mo, d, h, mi, s) = civil_from_unix(seconds);
        format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02} UTC")
    })
}

/// `crash-yyyy-MM-dd_HH.mm.ss-client.txt`.
///
/// java: `Minecraft.displayCrashReport` (`:763`) — `"crash-" + SimpleDateFormat(
/// "yyyy-MM-dd_HH.mm.ss") + "-client.txt"`. The pattern is vanilla's exactly and the timestamp
/// uses the host's local zone, with a UTC fallback only when the platform query fails.
#[must_use]
pub fn crash_file_name(seconds: i64) -> String {
    local_datetime(seconds)
        .unwrap_or_else(|| LocalDateTime::from_utc(seconds))
        .file_name()
}

/// `ScreenShotHelper`'s local filename timestamp (`yyyy-MM-dd_HH.mm.ss`).
///
/// Keep this beside the crash filename helper because both vanilla paths use the same
/// `SimpleDateFormat` local-zone conversion. The UTC civil arithmetic is only the explicit
/// fallback when the host formatter is unavailable; screenshots must not silently diverge from
/// crash reports on a machine with a non-UTC local timezone.
#[must_use]
pub fn screenshot_file_timestamp(seconds: i64) -> String {
    local_datetime(seconds)
        .unwrap_or_else(|| LocalDateTime::from_utc(seconds))
        .timestamp()
}

/// The local civil components needed by Java's two crash-report date formatters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LocalDateTime {
    year: i64,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
}

impl LocalDateTime {
    fn from_utc(seconds: i64) -> Self {
        let (year, month, day, hour, minute, second) = civil_from_unix(seconds);
        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
        }
    }

    fn file_name(self) -> String {
        format!(
            "crash-{year:04}-{month:02}-{day:02}_{hour:02}.{minute:02}.{second:02}-client.txt",
            year = self.year,
            month = self.month,
            day = self.day,
            hour = self.hour,
            minute = self.minute,
            second = self.second,
        )
    }

    fn timestamp(self) -> String {
        format!(
            "{year:04}-{month:02}-{day:02}_{hour:02}.{minute:02}.{second:02}",
            year = self.year,
            month = self.month,
            day = self.day,
            hour = self.hour,
            minute = self.minute,
            second = self.second,
        )
    }
}

/// Parse the invariant numeric line emitted by the platform clock query.
///
/// Keeping this parser separate gives the crash path a strict failure mode: malformed helper
/// output falls back to UTC instead of accidentally turning a diagnostic string into a filename.
fn parse_local_datetime(text: &str) -> Option<LocalDateTime> {
    let fields: Vec<_> = text.trim().split('|').collect();
    if fields.len() != 6 {
        return None;
    }
    let datetime = LocalDateTime {
        year: fields[0].parse().ok()?,
        month: fields[1].parse().ok()?,
        day: fields[2].parse().ok()?,
        hour: fields[3].parse().ok()?,
        minute: fields[4].parse().ok()?,
        second: fields[5].parse().ok()?,
    };
    let valid = (1..=12).contains(&datetime.month)
        && (1..=31).contains(&datetime.day)
        && datetime.hour < 24
        && datetime.minute < 60
        && datetime.second < 60;
    valid.then_some(datetime)
}

/// Read the host's local date/time components for a Unix timestamp.
///
/// The OS's own time-zone conversion, in-process: this was a
/// `powershell.exe` run per crash report. `mc-render/tests/crash_os_facts.rs` checks it against
/// that PowerShell query.
#[cfg(target_os = "windows")]
fn local_datetime(seconds: i64) -> Option<LocalDateTime> {
    let [year, month, day, hour, minute, second] = mc_render::w21_mb::local_time_fields(seconds)?;
    parse_local_datetime(&format!(
        "{year:04}|{month:02}|{day:02}|{hour:02}|{minute:02}|{second:02}"
    ))
}

/// Read the host's local date/time components for a Unix timestamp.
#[cfg(unix)]
fn local_datetime(seconds: i64) -> Option<LocalDateTime> {
    first_output_line(&date_output(seconds, "+%Y|%m|%d|%H|%M|%S")?)
        .and_then(|line| parse_local_datetime(&line))
}

/// Platforms without a standard command-line local clock use the explicit UTC fallback.
#[cfg(not(any(target_os = "windows", unix)))]
fn local_datetime(_seconds: i64) -> Option<LocalDateTime> {
    None
}

/// Ask Windows for Java 8's default short date/time shape in the current user culture.
///
/// The user locale's own patterns and formatters, in-process (it was
/// a `powershell.exe` run); checked against that query by `mc-render/tests/crash_os_facts.rs`.
#[cfg(target_os = "windows")]
fn format_local_short_timestamp(seconds: i64) -> Option<String> {
    mc_render::w21_mb::local_short_date_time(seconds).filter(|line| !line.is_empty())
}

/// Ask Unix's locale-aware `date` formatter for a short local date/time.
#[cfg(unix)]
fn format_local_short_timestamp(seconds: i64) -> Option<String> {
    let date = first_output_line(&date_output(seconds, "+%x")?)?;
    // `%p` is empty for 24-hour locales. `%I:%M` otherwise gives Java's short-time shape after
    // dropping the leading zero; unlike `%r`, it does not append seconds.
    let meridiem = first_output_line(&date_output(seconds, "+%p")?);
    let time_format = if meridiem.as_deref().is_some_and(|value| !value.is_empty()) {
        "+%I:%M %p"
    } else {
        "+%H:%M"
    };
    let mut time = first_output_line(&date_output(seconds, time_format)?)?;
    if time.starts_with('0') {
        time.remove(0);
    }
    Some(format!("{date} {time}"))
}

/// Platforms without a standard locale-aware formatter use the explicit UTC fallback.
#[cfg(not(any(target_os = "windows", unix)))]
fn format_local_short_timestamp(_seconds: i64) -> Option<String> {
    None
}

#[cfg(unix)]
fn date_output(seconds: i64, format: &str) -> Option<String> {
    let seconds_text = seconds.to_string();
    let at = format!("@{seconds}");
    let gnu = std::process::Command::new("date")
        .args(["-d", &at, format])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned());
    gnu.or_else(|| {
        std::process::Command::new("date")
            .args(["-r", &seconds_text, format])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
    })
}

#[cfg(unix)]
fn first_output_line(output: &str) -> Option<String> {
    output
        .lines()
        .next()
        .map(ToOwned::to_owned)
        .filter(|line| !line.is_empty())
}

/// A crash report: a description, a cause, and a list of categories.
///
/// java: `CrashReport`.
#[derive(Debug, Clone)]
pub struct CrashReport {
    description: String,
    cause: String,
    system_details: CrashReportCategory,
    sections: Vec<CrashReportCategory>,
    stacktrace: Vec<String>,
    file: Option<PathBuf>,
    timestamp: i64,
    witty: &'static str,
}

impl CrashReport {
    /// Build a report and populate the `System Details` category.
    ///
    /// java: `CrashReport(String, Throwable)` → `populateEnvironment()`.
    #[must_use]
    pub fn new(description: impl Into<String>, cause: impl Into<String>) -> CrashReport {
        let mut report = CrashReport {
            description: description.into(),
            cause: cause.into(),
            system_details: CrashReportCategory::new("System Details"),
            sections: Vec::new(),
            stacktrace: Vec::new(),
            file: None,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs() as i64),
            witty: witty_comment(),
        };
        report.populate_environment();
        report
    }

    /// Build a report from a panic, capturing a backtrace.
    ///
    /// java: the `catch (Throwable)` arms in `Minecraft.run` (`:453`), which is the only place a
    /// crash that was not deliberately reported turns into a `CrashReport`.
    #[must_use]
    pub fn from_panic(info: &PanicHookInfo<'_>) -> CrashReport {
        if let Some(block) = info
            .payload()
            .downcast_ref::<mc_render::terrain::BlockMeshingPanic>()
        {
            return Self::from_block_meshing_panic(info.location(), block);
        }
        let payload = panic_message(info);
        let location = info
            .location()
            .map_or_else(|| "<unknown location>".to_owned(), ToString::to_string);
        let stacktrace = capture_backtrace();
        let mut cause = format!("panic at {location}:\n\t{payload}");
        append_cause_stacktrace(&mut cause, &stacktrace);
        let mut report = CrashReport::new("Unexpected error", cause);
        report.stacktrace = stacktrace.clone();
        report
            .make_category("Panic")
            .set_stack_trace(stacktrace)
            .add("Location", location)
            .add("Payload", payload);
        report
    }

    /// Build the report produced by `BlockRendererDispatcher`'s per-block catch boundary.
    ///
    /// The renderer carries its packed state and world coordinate across the Rayon unwind in a
    /// [`mc_render::terrain::BlockMeshingPanic`].  The client owns crash-report presentation, so
    /// it recreates Java's `CrashReport.makeCrashReport(..., "Tesselating block in world")` and
    /// its legacy `(Block, int)` `addBlockInfo` overload here.
    fn from_block_meshing_panic(
        location: Option<&std::panic::Location<'_>>,
        block: &mc_render::terrain::BlockMeshingPanic,
    ) -> CrashReport {
        let location =
            location.map_or_else(|| "<unknown location>".to_owned(), ToString::to_string);
        let stacktrace = capture_backtrace();
        let mut cause = format!("panic at {location}:\n\t{}", block.cause());
        append_cause_stacktrace(&mut cause, &stacktrace);
        let mut report = CrashReport::new("Tesselating block in world", cause);
        report.stacktrace = stacktrace;
        add_block_info_with_metadata(
            report.make_category("Block being tesselated"),
            block.pos(),
            block.state().block(),
            i32::from(block.state().meta()),
        );
        report
    }

    /// Build a report from an [`anyhow::Error`], including its whole `source` chain.
    #[must_use]
    pub fn from_error(description: impl Into<String>, error: &anyhow::Error) -> CrashReport {
        let mut rendered = format!("{error}");
        for (depth, source) in error.chain().skip(1).enumerate() {
            let _ = write!(rendered, "\n\tCaused by ({}): {source}", depth + 1);
        }
        let stacktrace = capture_backtrace();
        append_cause_stacktrace(&mut rendered, &stacktrace);
        let mut report = CrashReport::new(description, rendered);
        report.stacktrace = stacktrace;
        report
    }

    /// java: `populateEnvironment` (`:54`). The Java-specific rows (`Java VM Version`,
    /// `JVM Flags`, `IntCache`) have no Rust equivalent and are replaced by the facts that
    /// actually matter for this client; `Minecraft Version`, `Operating System` and `Memory` are
    /// kept in vanilla's order and shape.
    fn populate_environment(&mut self) {
        let details = &mut self.system_details;
        details.add("Minecraft Version", "1.8.9");
        details.add("Protocol Version", "47");
        details.add("Operating System", operating_system_detail());
        details.add("Client Version", env!("CARGO_PKG_VERSION"));
        details.add("Client Brand", "mc-rs");
        details.add(
            "Memory",
            match process_memory_bytes() {
                Some(bytes) => format!("{bytes} bytes ({} MB) resident", bytes / 1024 / 1024),
                None => "(unavailable)".to_owned(),
            },
        );
        details.add(
            "CPUs",
            std::thread::available_parallelism().map_or_else(
                |_| "(unavailable)".to_owned(),
                |count| count.get().to_string(),
            ),
        );
        details.add("Backtrace captured", "yes (captured automatically)");
        details.add("Current Section", mc_core::profile::current_section());
    }

    /// Append graphics facts captured when the render adapter was selected.
    ///
    /// The values are already presentation-ready strings so this remains usable after shutdown
    /// and in CPU-only crash-report tests.
    pub fn add_adapter_facts(&mut self, facts: &mc_render::AdapterFacts) {
        self.system_details.add("Renderer", facts.renderer.clone());
        self.system_details.add("Vendor", facts.vendor.clone());
        self.system_details.add("Backend", facts.backend.clone());
        self.system_details.add("Driver", facts.driver.clone());
        self.system_details.add("Device", facts.device.clone());
    }

    /// The report's one-line description. java: `getDescription`.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The rendered cause. java: `getCauseStackTraceOrString`.
    #[must_use]
    pub fn cause(&self) -> &str {
        &self.cause
    }

    /// The `System Details` category, for a caller adding more rows. java: `getCategory`.
    pub fn category(&mut self) -> &mut CrashReportCategory {
        &mut self.system_details
    }

    /// Append a new category and return it for filling in. java: `makeCategory`.
    ///
    /// java's `makeCategoryDepth` additionally prunes stack traces against the throwable's own;
    /// see this module's header for why that is not ported.
    pub fn make_category(&mut self, name: impl Into<String>) -> &mut CrashReportCategory {
        self.sections.push(CrashReportCategory::new(name));
        self.sections
            .last_mut()
            .expect("just pushed a category, so there is a last one")
    }

    /// Set the `-- Head --` stack trace.
    pub fn set_stack_trace(&mut self, frames: Vec<String>) {
        self.stacktrace = frames;
    }

    /// java: `getSectionsInStringBuilder` (`:151`).
    ///
    /// The `-- Head --` block is emitted only when a trace exists; vanilla additionally borrows
    /// the first frame of the first category when the report has none of its own, which is
    /// reproduced.
    pub fn append_sections(&self, builder: &mut String) {
        let head: Vec<String> = if self.stacktrace.is_empty() {
            // java: `ArrayUtils.subarray(sections.get(0).getStackTrace(), 0, 1)`.
            self.sections
                .first()
                .and_then(|category| category.stack_trace.first())
                .cloned()
                .into_iter()
                .collect()
        } else {
            self.stacktrace.clone()
        };

        if !head.is_empty() {
            builder.push_str("-- Head --\n");
            builder.push_str("Stacktrace:\n");
            for frame in &head {
                builder.push('\t');
                builder.push_str("at ");
                builder.push_str(frame);
                builder.push('\n');
            }
            builder.push('\n');
        }

        for category in &self.sections {
            category.append_to(builder);
            builder.push_str("\n\n");
        }

        self.system_details.append_to(builder);
    }

    /// The whole report as a string. java: `getCompleteReport` (`:229`).
    #[must_use]
    pub fn complete_report(&self) -> String {
        let mut out = String::new();
        out.push_str("---- Minecraft Crash Report ----\n");
        out.push_str("// ");
        out.push_str(self.witty);
        out.push_str("\n\n");
        out.push_str("Time: ");
        out.push_str(&format_timestamp(self.timestamp));
        out.push('\n');
        out.push_str("Description: ");
        out.push_str(&self.description);
        out.push_str("\n\n");
        out.push_str(&self.cause);
        out.push_str(
            "\n\nA detailed walkthrough of the error, its code path and all known details is as follows:\n",
        );
        // java: `for (int i = 0; i < 87; ++i) stringbuilder.append("-");`
        for _ in 0..87 {
            out.push('-');
        }
        out.push_str("\n\n");
        self.append_sections(&mut out);
        out
    }

    /// Where the report was written, once it has been. java: `getFile`.
    #[must_use]
    pub fn file(&self) -> Option<&Path> {
        self.file.as_deref()
    }

    /// The default file name for this report, inside a `crash-reports` directory.
    #[must_use]
    pub fn suggested_file_name(&self) -> String {
        crash_file_name(self.timestamp)
    }

    /// Write the report, creating parent directories.
    ///
    /// java: `saveToFile` (`:266`) — returns `false` if this report has **already** been saved,
    /// which is what stops `displayCrashReport` writing it twice. Reproduced, including the
    /// swallowed I/O failure: a crash handler that panics is worse than a lost report.
    pub fn save_to_file(&mut self, to: &Path) -> bool {
        if self.file.is_some() {
            return false;
        }
        if let Some(parent) = to.parent() {
            if !parent.as_os_str().is_empty() && std::fs::create_dir_all(parent).is_err() {
                return false;
            }
        }
        // A crash report is meant to be shared: the user's home folder is written as the log
        // writes it, and a sign-in token
        // never appears, even in a panic message or an error chain that quoted one.
        let text = self.complete_report();
        match std::fs::write(to, mc_core::logwriter::redact_for_sharing(&text).as_bytes()) {
            Ok(()) => {
                self.file = Some(to.to_path_buf());
                true
            }
            Err(error) => {
                tracing::error!(path = %to.display(), %error, "could not save crash report");
                false
            }
        }
    }
}

/// The process exit code a crash produces. java: `Minecraft.displayCrashReport` (`:760`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashExit {
    /// The report was written. java: `System.exit(-1)`.
    Saved = -1,
    /// The report could not be written. java: `System.exit(-2)`.
    NotSaved = -2,
}

/// Print the report, save it under `<game_dir>/crash-reports/`, and say what exit code vanilla
/// would use.
///
/// java: `Minecraft.displayCrashReport`. This function deliberately **does not** call
/// `std::process::exit`: vanilla exits here because `shutdownMinecraftApplet` has already run in
/// the `finally` block, whereas here the caller still owns a GPU device, a tokio runtime and an
/// unsaved `options.txt`. Returning the code lets [`crate::app::shutdown`] run first, which is
/// the whole point of that task.
pub fn display_crash_report(report: &mut CrashReport, game_dir: &Path) -> CrashExit {
    let text = report.complete_report();
    // java: `Bootstrap.printToSYSOUT` writes to the *real* stdout, bypassing the log framework,
    // because a crash may well be the log framework's fault. It is the log players are asked to
    // send, so the user's home folder is written as the log writes it.
    println!("{}", mc_core::logwriter::redact_for_sharing(&text));

    if let Some(existing) = report.file() {
        println!(
            "#@!@# Game crashed! Crash report saved to: #@!@# {}",
            mc_core::logwriter::redact_home(&existing.display().to_string())
        );
        return CrashExit::Saved;
    }

    let path = game_dir
        .join("crash-reports")
        .join(report.suggested_file_name());
    if report.save_to_file(&path) {
        println!(
            "#@!@# Game crashed! Crash report saved to: #@!@# {}",
            mc_core::logwriter::redact_home(&path.display().to_string())
        );
        CrashExit::Saved
    } else {
        println!("#@?@# Game crashed! Crash report could not be saved. #@?@#");
        CrashExit::NotSaved
    }
}

/// The panic captured by [`install_panic_hook`], if one has happened.
static LAST_PANIC: OnceLock<Mutex<Option<CrashReport>>> = OnceLock::new();

fn panic_slot() -> &'static Mutex<Option<CrashReport>> {
    LAST_PANIC.get_or_init(|| Mutex::new(None))
}

/// Install a panic hook that turns any panic — on **any** thread — into a [`CrashReport`].
///
/// java: `Minecraft.crashed(CrashReport)` (`:751`) sets `hasCrashed`/`crashReporter` and the run
/// loop notices on its next iteration. This is the same shape: the hook records, the game loop
/// asks. A hook that tried to write the file itself would run on whichever thread panicked,
/// possibly while the frame holds the GPU, which is exactly the situation vanilla avoids by
/// deferring.
///
/// Calling this more than once is harmless; the second call replaces the first hook.
pub fn install_panic_hook() {
    let default = std::panic::take_hook();
    // Every path below ends in the previous hook, which prints the message to
    // stderr: the zip's launcher saves that as a log players send. A message holding a registered
    // secret (a sign-in token) is printed by `print_panic_scrubbed` instead.
    let previous = move |info: &PanicHookInfo<'_>| print_panic_scrubbed(&*default, info);
    std::panic::set_hook(Box::new(move |info| {
        // `BlockRendererDispatcher` catches the original renderer throwable, enriches it, then
        // throws `ReportedException`.  The terrain mesher follows that shape with a short
        // `catch_unwind`; do not let its first, information-poor payload win this first-panic
        // slot before the enriched `BlockMeshingPanic` is raised.
        if mc_render::terrain::block_meshing_panic_is_being_wrapped() {
            previous(info);
            return;
        }
        // A sound-prewarm decode panics inside its own boundary: that file is skipped and the
        // session goes on, so it is not a crash (the play path reports the file if it is used).
        if mc_audio::prewarm_panic_is_being_contained() {
            previous(info);
            return;
        }
        // A panic on the Spotify widget's media worker
        // (mc-media) is caught there too: the widget goes blank and the game goes on.
        if mc_media::panic_is_being_contained() {
            previous(info);
            return;
        }
        // Any other work that catches its own panic and carries on (one
        // chunk packet unpacked off-thread, one skin, a background preparation) runs inside
        // `mc_core::panic_scope::contain`; it is not the game's crash. So is a failed restart of
        // the audio output after a device change (`mc_audio::is_output_backend_panic`).
        if mc_core::panic_scope::is_contained() || mc_audio::is_output_backend_panic(info) {
            previous(info);
            return;
        }
        let report = CrashReport::from_panic(info);
        if let Ok(mut slot) = panic_slot().lock() {
            // Keep the *first* panic: a panic during unwinding of another is a consequence, and
            // the original is what a reader needs.
            if slot.is_none() {
                *slot = Some(report);
                CRASHED.store(true, Ordering::Release);
            }
        }
        // Lines queued for the background log writer go out before the panic message, so the
        // console keeps their order. Bounded, so a stuck writer cannot hold up the panic.
        mc_core::logwriter::flush(std::time::Duration::from_millis(250));
        previous(info);
    }));
}

/// Take the crash report the hook recorded, if any. java: the `hasCrashed` check at
/// `Minecraft.run:422`.
#[must_use]
pub fn take_recorded_crash() -> Option<CrashReport> {
    panic_slot().lock().ok().and_then(|mut slot| {
        CRASHED.store(false, Ordering::Release);
        slot.take()
    })
}

/// Whether a panic has been recorded but not yet taken. java: `Minecraft.hasCrashed`.
///
/// Read every frame by `Client::keep_running` (vanilla's run loop checks
/// `hasCrashed` every iteration), so it is one atomic load, never the slot's lock.
#[must_use]
pub fn has_crashed() -> bool {
    CRASHED.load(Ordering::Acquire)
}

/// Mirrors "the slot holds a report"; written only while the slot's lock is held.
static CRASHED: AtomicBool = AtomicBool::new(false);

/// Run `hook` (the default panic hook) for `info`, unless the panic message
/// holds a registered secret: then write what the default hook writes, the thread, the location
/// and the message, with the secret replaced. Panics are rare, so the message is rendered and
/// checked every time.
fn print_panic_scrubbed(
    hook: &(dyn Fn(&PanicHookInfo<'_>) + Send + Sync),
    info: &PanicHookInfo<'_>,
) {
    use std::io::Write as _;

    let message = panic_message(info);
    let std::borrow::Cow::Owned(clean) = mc_core::secrets::scrub(&message) else {
        hook(info);
        return;
    };
    let thread = std::thread::current();
    let location = info
        .location()
        .map_or_else(|| "<unknown location>".to_owned(), ToString::to_string);
    // As the default hook does: a broken stderr is ignored, never a second panic.
    let _ = writeln!(
        std::io::stderr(),
        "\nthread '{}' panicked at {location}:\n{clean}",
        thread.name().unwrap_or("<unnamed>")
    );
}

/// Render a panic payload the way `Throwable.getMessage()` would.
fn panic_message(info: &PanicHookInfo<'_>) -> String {
    let payload = info.payload();
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "Box<dyn Any>".to_owned()
    }
}

/// The current backtrace as one frame per line.
///
/// Vanilla's `Throwable.printStackTrace` always carries frames. `Backtrace::capture` honours
/// `RUST_BACKTRACE` and would make the normal report empty, so crash reports deliberately force a
/// capture regardless of that developer-facing environment setting.
fn capture_backtrace() -> Vec<String> {
    let backtrace = std::backtrace::Backtrace::force_capture();
    if backtrace.status() != std::backtrace::BacktraceStatus::Captured {
        return Vec::new();
    }
    backtrace
        .to_string()
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

/// Append the backtrace in the same `\tat frame` shape that `Throwable.printStackTrace` uses.
fn append_cause_stacktrace(cause: &mut String, frames: &[String]) {
    for frame in frames {
        let _ = write!(cause, "\n\tat {frame}");
    }
}

/// Java's `os.name + " (" + os.arch + ") version " + os.version` report row.
///
/// `std::env::consts::FAMILY` is a compile-time platform family (`"windows"` / `"unix"`), not
/// an OS version. On Windows, query the same OS facility that exposes the display name and kernel
/// version. The fallback deliberately says `unknown` rather than repeating the incorrect family.
fn operating_system_detail() -> String {
    let (name, version) = operating_system_name_and_version();
    format!("{name} ({}) version {version}", java_os_architecture())
}

fn java_os_architecture() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" => "amd64",
        "x86" => "x86",
        architecture => architecture,
    }
}

/// WMI's `Win32_OperatingSystem` `Caption` and `Version`, from the
/// OS calls behind them, in-process (this was a `powershell.exe` run of `Get-CimInstance` per
/// crash report); checked against that query by `mc-render/tests/crash_os_facts.rs`.
#[cfg(target_os = "windows")]
fn operating_system_name_and_version() -> (String, String) {
    match (
        mc_render::w21_mb::windows_caption(),
        mc_render::w21_mb::windows_version(),
    ) {
        (Some(name), Some(version)) => (name, version),
        _ => ("Windows".to_owned(), "unknown".to_owned()),
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn operating_system_name_and_version() -> (String, String) {
    let output = std::process::Command::new("uname")
        .args(["-s", "-r"])
        .output();
    let Some(output) = output.ok().filter(|output| output.status.success()) else {
        return (std::env::consts::OS.to_owned(), "unknown".to_owned());
    };
    let mut fields = String::from_utf8_lossy(&output.stdout).split_whitespace();
    let name = fields.next().unwrap_or(std::env::consts::OS).to_owned();
    let version = fields.next().unwrap_or("unknown").to_owned();
    (name, version)
}

#[cfg(target_os = "macos")]
fn operating_system_name_and_version() -> (String, String) {
    let output = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output();
    let version = output
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|version| !version.is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    ("Mac OS X".to_owned(), version)
}

#[cfg(not(any(target_os = "windows", unix)))]
fn operating_system_name_and_version() -> (String, String) {
    (std::env::consts::OS.to_owned(), "unknown".to_owned())
}

/// How stale the F3 overlay's memory line is allowed to be.
///
/// One second is vanilla's own refresh feel for that line and is three orders of magnitude longer
/// than the frame it must not stall.
pub const MEMORY_SAMPLE_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(1);

/// [`process_memory_bytes`], answered from a cache no older than `max_age`.
///
/// # Why this exists
///
/// [`process_memory_bytes`] **spawns a child process**. On Windows that is `tasklist`, and a
/// process spawn plus its output costs tens of milliseconds — measured at **55.8 ms per call** on
/// the machine this was found on. The F3 overlay called it once per frame, which made a single
/// line of debug text cost more than the entire renderer by two orders of magnitude, whether or
/// not the overlay was even on screen.
///
/// Callers that need a *fresh* figure — the crash report, and `tests/leak.rs`, which measures
/// resident memory across connect/disconnect cycles and would be wrong to read a stale one —
/// must keep calling [`process_memory_bytes`]. This is for the per-frame reader, and for it only.
///
/// # A stale answer beats a stalled frame
///
/// Only the **first** call samples on the calling thread — unless [`prime_process_memory_sample`]
/// already took that sample at startup, which is what the game binary does so the first F3 press
/// does not stall for `tasklist` (HITCH-2). Once there is a figure to show, an
/// expired one is served immediately and the refresh happens on a background thread, so no frame
/// ever waits for the OS — a 1-second cache that refreshes *synchronously* would still hitch for
/// 55 ms once a second, which is a visible stutter rather than a fixed bug.
#[must_use]
pub fn process_memory_bytes_cached(max_age: std::time::Duration) -> Option<u64> {
    use std::sync::atomic::Ordering;

    let cached = *MEMORY_SAMPLE_CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some((sampled, value)) = cached else {
        // Nothing to show yet: this one call pays, so the line is never blank. If a startup prime
        // is sampling right now this waits for *its* answer instead, which costs at most what
        // sampling here would have.
        return prime_process_memory_sample();
    };
    if sampled.elapsed() < max_age {
        return value;
    }
    if MEMORY_SAMPLE_REFRESHING
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        match memory_sampler() {
            Some(sampler) => sampler.unpark(),
            // No thread could be started, so nothing will clear the flag but this.
            None => MEMORY_SAMPLE_REFRESHING.store(false, Ordering::Release),
        }
    }
    value
}

/// Whether a background memory sample is requested or in flight, so a slow sampler cannot queue
/// one request per frame behind it.
static MEMORY_SAMPLE_REFRESHING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The sampler thread's job: take the requested sample, publish it, clear the request.
fn refresh_memory_sample() {
    use std::sync::atomic::Ordering;
    if !MEMORY_SAMPLE_REFRESHING.load(Ordering::Acquire) {
        return;
    }
    let fresh = process_memory_bytes();
    *MEMORY_SAMPLE_CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some((std::time::Instant::now(), fresh));
    MEMORY_SAMPLE_REFRESHING.store(false, Ordering::Release);
}

/// The long-lived thread behind [`process_memory_bytes_cached`]'s refreshes, started on first use.
pub(crate) fn memory_sampler() -> Option<&'static std::thread::Thread> {
    static SAMPLER: std::sync::OnceLock<Option<std::thread::Thread>> = std::sync::OnceLock::new();
    SAMPLER
        .get_or_init(|| spawn_resampler("mc-memory-sample", refresh_memory_sample))
        .as_ref()
}

/// Start a parked thread that runs `refresh` each time it is unparked (N2).
///
/// The F3 caches used to spawn a fresh OS thread for every 1-second refresh — from the frame
/// thread, twice a second while the overlay is open, each spawn tens to hundreds of microseconds
/// of thread creation inside a frame. Unparking a long-lived thread costs one wake-up. `refresh`
/// must check its own request flag: `park` may wake spuriously.
pub(crate) fn spawn_resampler(name: &str, refresh: fn()) -> Option<std::thread::Thread> {
    std::thread::Builder::new()
        .name(name.to_owned())
        .spawn(move || loop {
            std::thread::park();
            refresh();
        })
        .ok()
        .map(|handle| handle.thread().clone())
}

/// [`process_memory_bytes_cached`]'s last sample and when it was taken.
#[allow(clippy::type_complexity)]
static MEMORY_SAMPLE_CACHE: std::sync::Mutex<Option<(std::time::Instant, Option<u64>)>> =
    std::sync::Mutex::new(None);

/// Take [`process_memory_bytes_cached`]'s first sample now, on the calling thread, unless one is
/// already cached; returns the cached figure either way.
///
/// The cache lock is held **while** sampling, so a frame that asks meanwhile waits for this
/// sample — at most the time it would have spent taking its own — rather than starting a second
/// `tasklist`. The game binary calls this from a startup thread (`Client::new_deferred`), seconds
/// before anyone can press F3, so the first F3 frame finds the figure ready (HITCH-2). Callers
/// that need a *fresh* figure still use [`process_memory_bytes`].
pub fn prime_process_memory_sample() -> Option<u64> {
    let mut cache = MEMORY_SAMPLE_CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some((_, value)) = *cache {
        return value;
    }
    let value = process_memory_bytes();
    *cache = Some((std::time::Instant::now(), value));
    value
}

/// Resident set size of this process, in bytes.
///
/// java: `Runtime.getRuntime().totalMemory()/freeMemory()/maxMemory()`, which report the JVM heap
/// — a number Rust has no equivalent of, since there is no managed heap to ask. The honest
/// analogue is the OS's own view of the process, which is also what
/// [`crate::app::leak`](crate::app) measures across connect/disconnect cycles.
///
/// On Windows this is the working set from `GetProcessMemoryInfo`, read through
/// [`mc_render::w21_mb::process_memory`] (the no-`unsafe` rule keeps the FFI in `mc-render`): the figure
/// `tasklist` reports as "Mem Usage", which this used to start `tasklist.exe` for (55 ms a call).
///
/// Cheap on Windows now (microseconds), but per-frame paths still go through
/// [`process_memory_bytes_cached`]: other platforms read a file.
#[must_use]
pub fn process_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "windows")]
    {
        mc_render::w21_mb::process_memory().map(|memory| memory.working_set)
    }
    #[cfg(not(target_os = "windows"))]
    {
        // `VmRSS:\t   12345 kB`
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reason [`process_memory_bytes_cached`] exists, asserted rather than asserted-about:
    /// a run of cache hits must cost less than the single miss it replaces.
    ///
    /// The control is the uncached call in the same test. Without it this would pass on a machine
    /// where sampling memory is free, and prove nothing about the machine where it costs 55 ms.
    #[test]
    fn the_cached_memory_sample_is_reused_within_its_window() {
        use std::time::{Duration, Instant};

        let window = Duration::from_secs(600);
        // No OS answer at all (a container without `tasklist`, say) is not a failure; there is
        // simply nothing to cache.
        let Some(first) = process_memory_bytes_cached(window) else {
            return;
        };
        let start = Instant::now();
        for _ in 0..200 {
            assert_eq!(
                process_memory_bytes_cached(window),
                Some(first),
                "a sample inside its window must be the same sample"
            );
        }
        let hits = start.elapsed();

        let start = Instant::now();
        let _ = process_memory_bytes();
        let miss = start.elapsed();

        assert!(
            hits < miss.max(Duration::from_millis(1)),
            "200 cache hits took {hits:?}, one uncached sample took {miss:?}"
        );

        // An *expired* window must not stall the caller either: the stale figure is served and
        // the resample happens elsewhere. Without this the fix would still hitch once a second.
        let start = Instant::now();
        let expired = process_memory_bytes_cached(Duration::ZERO);
        let served = start.elapsed();
        assert!(
            expired.is_some(),
            "an expired window still has a last value"
        );
        assert!(
            served < miss.max(Duration::from_millis(1)),
            "serving an expired sample took {served:?}, as long as resampling ({miss:?})"
        );
    }

    /// java: `getCoordinateInfo(BlockPos)` at a hand-computed positive coordinate.
    #[test]
    fn coordinate_info_matches_the_java_format_string() {
        assert_eq!(
            coordinate_info(35, 70, -45),
            "World: (35,70,-45), \
             Chunk: (at 3,4,3 in 2,-3; contains blocks 32,0,-48 to 47,255,-33), \
             Region: (0,-1; contains chunks 0,-32 to 31,-1, blocks 0,0,-512 to 511,255,-1)"
        );
    }

    /// The shifts are Java's arithmetic shifts, so a negative coordinate floors rather than
    /// truncates: `-1 >> 4 == -1`, not `0`. Getting this wrong puts the "contains blocks" range
    /// one chunk out for every negative coordinate — half the world.
    #[test]
    fn negative_coordinates_use_arithmetic_shift_semantics() {
        let info = coordinate_info(-1, 0, -1);
        assert!(
            info.contains("Chunk: (at 15,0,15 in -1,-1; contains blocks -16,0,-16 to -1,255,-1)"),
            "{info}"
        );
        assert!(
            info.contains("Region: (-1,-1; contains chunks -32,-32 to -1,-1"),
            "{info}"
        );
    }

    /// java: `String.format("%.2f,%.2f,%.2f - %s", …)` with `new BlockPos(x, y, z)` flooring.
    /// java: `CrashReportCategory.addBlockInfo(category, pos, block, blockData)`. This is the
    /// overload used by `BlockRendererDispatcher`: its numeric renderings must retain the leading
    /// binary zeroes, uppercase hexadecimal, and the negative-metadata branch.
    #[test]
    fn legacy_block_info_matches_vanillas_three_detail_rows() {
        use mc_data::BlockId;
        use mc_math::BlockPos;

        let mut category = CrashReportCategory::new("Block being tesselated");
        add_block_info_with_metadata(&mut category, BlockPos::new(35, 70, -45), BlockId(1), 5);
        let mut out = String::new();
        category.append_to(&mut out);
        assert!(
            out.contains("\tBlock type: ID #1 (tile.stone // net.minecraft.block.BlockStone)"),
            "{out}"
        );
        assert!(
            out.contains("\tBlock data value: 5 / 0x5 / 0b0101"),
            "{out}"
        );
        assert!(
            out.contains("\tBlock location: World: (35,70,-45)"),
            "{out}"
        );

        // Negative control: Java's width four is a minimum rather than a four-bit mask, while
        // negative metadata takes its separate `Unknown?` branch.
        let mut malformed = CrashReportCategory::new("Malformed block");
        add_block_info_with_metadata(&mut malformed, BlockPos::ORIGIN, BlockId(1), 16);
        add_block_info_with_metadata(&mut malformed, BlockPos::ORIGIN, BlockId(1), -1);
        let mut malformed_out = String::new();
        malformed.append_to(&mut malformed_out);
        assert!(
            malformed_out.contains("\tBlock data value: 16 / 0x10 / 0b10000"),
            "{malformed_out}"
        );
        assert!(
            malformed_out.contains("\tBlock data value: Unknown? (Got -1)"),
            "{malformed_out}"
        );
    }

    /// java: `CrashReportCategory.addBlockInfo(category, pos, state)` and
    /// `BlockStateBase.toString()`. This is distinct from the legacy overload: vanilla writes
    /// one `Block:` row containing state properties, not type/data rows.
    #[test]
    fn state_block_info_matches_vanillas_property_rendering() {
        use mc_data::{BlockId, BlockState};
        use mc_math::BlockPos;

        let mut category = CrashReportCategory::new("Block model being tesselated");
        // stone's `variant=stone` is a real BlockState property, proving this is not just a
        // registry-name formatter.
        add_block_info(
            &mut category,
            BlockPos::new(-1, 64, 3),
            BlockState::new(BlockId(1), 0),
        );
        let mut out = String::new();
        category.append_to(&mut out);
        assert!(
            out.contains("\tBlock: minecraft:stone[variant=stone]"),
            "{out}"
        );
        assert!(out.contains("\tBlock location: World: (-1,64,3)"), "{out}");

        // Negative control: an air state has no properties, so Java's `toString()` leaves off
        // the brackets entirely. A blanket `name[variant]` renderer would fail this case.
        let mut air = CrashReportCategory::new("Air");
        add_block_info(&mut air, BlockPos::ORIGIN, BlockState::AIR);
        let mut air_out = String::new();
        air.append_to(&mut air_out);
        assert!(air_out.contains("\tBlock: minecraft:air"), "{air_out}");
        assert!(!air_out.contains("minecraft:air["), "{air_out}");
    }

    /// `BlockRendererDispatcher.renderBlock` catches a failed block renderer and reports the
    /// *stored* block plus metadata, not `BlockModelRenderer`'s property-list overload.  The
    /// mesher unit proves this payload is captured from the real per-block unwind; this client
    /// boundary proves the exact vanilla report shape survives that cross-crate handoff.
    #[test]
    fn a_meshing_panic_reports_the_actual_block_position_and_legacy_metadata() {
        use mc_data::{BlockId, BlockState};
        use mc_math::BlockPos;

        let pos = BlockPos::new(-12, 70, 33);
        let state = BlockState::new(BlockId(53), 3); // `oak_stairs`, north/up in 1.8.9.
        let panic = mc_render::terrain::BlockMeshingPanic::new(pos, state, "model exploded");
        let report = CrashReport::from_block_meshing_panic(None, &panic);
        let text = report.complete_report();

        assert!(
            text.contains("Description: Tesselating block in world"),
            "{text}"
        );
        assert!(text.contains("-- Block being tesselated --"), "{text}");
        assert!(
            text.contains(
                "\tBlock type: ID #53 (tile.stairsWood // net.minecraft.block.BlockStairs)"
            ),
            "{text}"
        );
        assert!(
            text.contains("\tBlock data value: 3 / 0x3 / 0b0011"),
            "{text}"
        );
        assert!(
            text.contains("\tBlock location: World: (-12,70,33)"),
            "{text}"
        );

        // Negative control: the dispatcher overload is intentionally *not* the model renderer's
        // `IBlockState` overload, so a state-property list would prove the wrong Java producer
        // was wired.
        assert!(!text.contains("\tBlock: minecraft:oak_stairs["), "{text}");
    }

    /// java: `String.format("%.2f,%.2f,%.2f - %s", ...)` with `new BlockPos(x, y, z)` flooring.
    #[test]
    fn precise_coordinate_info_floors_the_block_position() {
        let info = coordinate_info_precise(-0.5, 64.25, 3.75);
        assert!(
            info.starts_with("-0.50,64.25,3.75 - World: (-1,64,3),"),
            "{info}"
        );
    }

    /// Java's `Formatter` rounds its shortest decimal input HALF_UP. These three values are the
    /// divergent region: Rust's native `:.2` would render `2.67,0.12,-0.12` instead.
    #[test]
    fn precise_coordinate_info_uses_java_half_up_rounding() {
        let info = coordinate_info_precise(2.675, 0.125, -0.125);
        assert!(
            info.starts_with("2.68,0.13,-0.13 - World: (2,0,-1),"),
            "{info}"
        );

        // Negative control: ordinary non-tie values must still retain their expected digits, so
        // this is not merely a blanket round-away-from-zero implementation.
        assert_eq!(format_java_decimal(1.234, 2), "1.23");
        assert_eq!(format_java_decimal(9.999, 2), "10.00");
        assert_eq!(format_java_decimal(1e20, 2), "100000000000000000000.00");
        assert_eq!(format_java_decimal(1e-7, 2), "0.00");
        assert_eq!(format_java_decimal(-0.0, 2), "-0.00");
        assert_eq!(format_java_decimal(f64::NAN, 2), "NaN");
        assert_eq!(format_java_decimal(f64::NEG_INFINITY, 2), "-Infinity");
    }

    #[test]
    fn civil_from_unix_matches_known_timestamps() {
        assert_eq!(civil_from_unix(0), (1970, 1, 1, 0, 0, 0));
        // Hand-derived: 1_776_170_096 / 86400 = 20557 whole days, remainder 45_296 s =
        // 12:34:56. 2026-01-01 is day 20454 (56 years x 365 + 14 leap days), so day 20557 is
        // 103 days into 2026 — 31 + 28 + 31 = 90 takes us to April 1, and +13 to April 14.
        assert_eq!(civil_from_unix(1_776_170_096), (2026, 4, 14, 12, 34, 56));
        assert_eq!(civil_from_unix(1_755_261_296), (2025, 8, 15, 12, 34, 56));
        // A leap day, and one second before an epoch boundary.
        assert_eq!(civil_from_unix(1_709_164_800), (2024, 2, 29, 0, 0, 0));
        assert_eq!(civil_from_unix(-1), (1969, 12, 31, 23, 59, 59));
    }

    /// java: `"crash-" + SimpleDateFormat("yyyy-MM-dd_HH.mm.ss") + "-client.txt"`.
    #[test]
    fn crash_file_name_matches_vanillas_pattern() {
        let seconds = 1_755_261_296;
        let local = local_datetime(seconds).unwrap_or_else(|| LocalDateTime::from_utc(seconds));
        assert_eq!(crash_file_name(seconds), local.file_name());
        assert_eq!(screenshot_file_timestamp(seconds), local.timestamp());
        assert_eq!(
            screenshot_file_timestamp(seconds),
            crash_file_name(seconds)
                .strip_prefix("crash-")
                .and_then(|name| name.strip_suffix("-client.txt"))
                .expect("crash and screenshot timestamps share the same local shape")
        );
        assert!(crash_file_name(seconds).starts_with("crash-2025-"));
        assert!(crash_file_name(seconds).ends_with("-client.txt"));
    }

    #[test]
    fn local_datetime_parser_rejects_malformed_or_impossible_clock_values() {
        assert_eq!(
            parse_local_datetime("2025|08|15|08|34|56\r\n"),
            Some(LocalDateTime {
                year: 2025,
                month: 8,
                day: 15,
                hour: 8,
                minute: 34,
                second: 56,
            })
        );
        // Negative control: a helper that returns a success status but swaps fields must not
        // produce a plausible filename. The strict ranges catch hour/minute/second corruption.
        assert_eq!(parse_local_datetime("2025|08|15|24|34|56"), None);
        assert_eq!(parse_local_datetime("2025|13|15|08|34|56"), None);
        assert_eq!(parse_local_datetime("2025|08|15|08|34"), None);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn timestamp_uses_java8_short_local_format_on_windows() {
        let seconds = 1_755_261_296;
        let expected = format_local_short_timestamp(seconds)
            .expect("PowerShell must provide the host's local short date/time");
        assert_eq!(format_timestamp(seconds), expected);
        // The former implementation's UTC ISO output cannot pass as Java's DateFormat.SHORT.
        assert_ne!(
            expected, "2025-08-15 12:34:56 UTC",
            "host formatter must not collapse back to the old UTC shape"
        );
    }

    /// java: `CrashReportCategory.appendToStringBuilder` — `Details:` is unconditional and each
    /// entry is preceded by a newline and a **tab**.
    #[test]
    fn a_category_renders_exactly_like_vanilla() {
        let mut category = CrashReportCategory::new("Affected level");
        category
            .add("Level name", "MpServer")
            .add_null("All players")
            .add_error("Chunk stats", "AssetError", "missing pack")
            .set_stack_trace(vec!["mc_client::app::run".to_owned()]);

        let mut out = String::new();
        category.append_to(&mut out);
        assert_eq!(
            out,
            "-- Affected level --\n\
             Details:\n\
             \tLevel name: MpServer\n\
             \tAll players: ~~NULL~~\n\
             \tChunk stats: ~~ERROR~~ AssetError: missing pack\n\
             Stacktrace:\n\
             \tat mc_client::app::run"
        );
    }

    /// An empty category still prints its header and a bare `Details:` line, because vanilla
    /// appends `"Details:"` before it has looked at the children at all.
    #[test]
    fn an_empty_category_still_prints_the_details_header() {
        let mut out = String::new();
        CrashReportCategory::new("Empty").append_to(&mut out);
        assert_eq!(out, "-- Empty --\nDetails:");
    }

    #[test]
    fn add_result_renders_ok_and_err_differently() {
        let mut category = CrashReportCategory::new("Probe");
        let ok: Result<i32, std::io::Error> = Ok(7);
        let err: Result<i32, std::io::Error> = Err(std::io::Error::other("nope"));
        category.add_result("Good", ok).add_result("Bad", err);
        let mut out = String::new();
        category.append_to(&mut out);
        assert!(out.contains("\tGood: 7"), "{out}");
        assert!(out.contains("\tBad: ~~ERROR~~ Error: nope"), "{out}");
    }

    /// The whole-report skeleton: header, witty comment, the 87 dashes, the sections, and
    /// `System Details` last.
    #[test]
    fn the_complete_report_has_vanillas_skeleton() {
        let mut report = CrashReport::new("Ticking screen", "TestError: boom");
        report.set_stack_trace(vec!["mc_client::app::App::tick".to_owned()]);
        report
            .make_category("Affected screen")
            .add("Screen name", "MainMenu");

        let text = report.complete_report();
        let lines: Vec<&str> = text.lines().collect();

        assert_eq!(lines[0], "---- Minecraft Crash Report ----");
        assert!(lines[1].starts_with("// "), "{}", lines[1]);
        assert!(WITTY_COMMENTS.contains(&&lines[1][3..]), "{}", lines[1]);
        assert_eq!(lines[2], "");
        assert!(lines[3].starts_with("Time: "), "{}", lines[3]);
        assert_eq!(lines[4], "Description: Ticking screen");
        assert_eq!(lines[5], "");
        assert_eq!(lines[6], "TestError: boom");
        assert_eq!(lines[7], "");
        assert_eq!(
            lines[8],
            "A detailed walkthrough of the error, its code path and all known details is as follows:"
        );
        assert_eq!(lines[9], "-".repeat(87), "vanilla writes exactly 87 dashes");
        assert_eq!(lines[10], "");
        assert_eq!(lines[11], "-- Head --");
        assert_eq!(lines[12], "Stacktrace:");
        assert_eq!(lines[13], "\tat mc_client::app::App::tick");

        assert!(text.contains("-- Affected screen --\nDetails:\n\tScreen name: MainMenu"));
        // System Details is always last, and is the only category with no blank line after it.
        let details = text
            .find("-- System Details --")
            .expect("System Details must be present");
        assert!(
            text[details..].contains("Minecraft Version: 1.8.9"),
            "the environment must be populated"
        );
        assert!(
            text[details..].contains("\tCurrent Section: "),
            "the profiler position must be captured in System Details"
        );
        assert!(
            text.find("-- Affected screen --").unwrap() < details,
            "System Details is appended after every other category"
        );
    }

    /// java: `saveToFile` returns `false` on the second call and leaves the first file alone,
    /// which is what stops `displayCrashReport` overwriting a report the caller already placed.
    #[test]
    fn a_report_saves_once_and_refuses_to_save_twice() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first = dir.path().join("nested").join("crash-one.txt");
        let second = dir.path().join("crash-two.txt");

        let mut report = CrashReport::new("Ticking player", "TestError: boom");
        assert!(report.save_to_file(&first), "the first save must succeed");
        assert_eq!(report.file(), Some(first.as_path()));
        assert!(first.exists(), "parent directories must be created");
        let written = std::fs::read_to_string(&first).expect("read back");
        assert_eq!(written, report.complete_report());

        assert!(
            !report.save_to_file(&second),
            "a report that already has a file must refuse"
        );
        assert!(!second.exists());
        assert_eq!(report.file(), Some(first.as_path()));
    }

    /// The end-to-end handler: print, write under `crash-reports/`, and report `Saved`.
    #[test]
    fn display_crash_report_writes_under_the_game_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut report = CrashReport::new("Ticking entity", "TestError: boom");
        let exit = display_crash_report(&mut report, dir.path());
        assert_eq!(exit, CrashExit::Saved);

        let file = report.file().expect("a saved report has a file").to_owned();
        assert_eq!(file.parent().unwrap(), dir.path().join("crash-reports"));
        let name = file.file_name().unwrap().to_string_lossy().into_owned();
        assert!(
            name.starts_with("crash-") && name.ends_with("-client.txt"),
            "{name}"
        );
        assert!(std::fs::read_to_string(&file)
            .expect("read back")
            .starts_with("---- Minecraft Crash Report ----"));

        // A second call must not write a second file. java: the `getFile() != null` branch.
        let exit = display_crash_report(&mut report, dir.path());
        assert_eq!(exit, CrashExit::Saved);
        let entries: Vec<_> = std::fs::read_dir(dir.path().join("crash-reports"))
            .expect("read dir")
            .collect();
        assert_eq!(entries.len(), 1, "exactly one crash file must exist");
    }

    /// A saved report is meant to be shared, so the user's
    /// home folder in it is written as `%USERPROFILE%` (`~` off Windows), as in the log; the
    /// rest of each path stays.
    #[test]
    fn a_saved_report_does_not_name_the_home_folder() {
        let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))
        else {
            return;
        };
        let home = home.to_string_lossy().into_owned();
        if mc_core::logwriter::redact_home(&home) == home.as_str() {
            eprintln!("skip: this home folder is too short to redact");
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("crash.txt");
        let sep = std::path::MAIN_SEPARATOR;
        let mut report = CrashReport::new(
            "Ticking player",
            format!("TestError: cannot read {home}{sep}secret.txt"),
        );
        assert!(report.save_to_file(&file));
        let written = std::fs::read_to_string(&file).expect("read back");
        assert!(
            !written.to_lowercase().contains(&home.to_lowercase()),
            "{written}"
        );
        assert!(
            written.contains("secret.txt"),
            "the rest of the path stays: {written}"
        );
    }

    #[test]
    fn a_report_from_an_anyhow_chain_keeps_every_cause() {
        let root = std::io::Error::other("socket closed");
        let error = anyhow::Error::new(root)
            .context("reading the chunk")
            .context("Ticking world");
        let report = CrashReport::from_error("Ticking world", &error);
        let text = report.complete_report();
        assert!(text.contains("Ticking world"), "{text}");
        assert!(text.contains("Caused by (1): reading the chunk"), "{text}");
        assert!(text.contains("Caused by (2): socket closed"), "{text}");
    }

    /// The normal process environment leaves `RUST_BACKTRACE` unset, but a crash report must
    /// still contain the equivalent of Java's unconditional `Throwable.printStackTrace` output.
    #[test]
    fn crash_reports_capture_a_backtrace_without_rust_backtrace() {
        let frames = capture_backtrace();
        assert!(
            !frames.is_empty(),
            "force_capture must produce frames even without RUST_BACKTRACE"
        );

        let report = CrashReport::from_error("Ticking world", &anyhow::anyhow!("boom"));
        let text = report.complete_report();
        assert!(text.contains("-- Head --\nStacktrace:"), "{text}");
        assert!(
            text.contains("Backtrace captured: yes (captured automatically)"),
            "{text}"
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_os_probe_keeps_the_name_and_version_separate() {
        // The name and the version come from two OS calls now, with
        // no output line to split; `mc-render/tests/crash_os_facts.rs` checks both against the
        // WMI query they replace.
        let (name, version) = operating_system_name_and_version();
        assert!(name.starts_with("Microsoft Windows"), "{name}");
        let parts: Vec<&str> = version.split('.').collect();
        assert!(
            parts.len() == 3
                && parts
                    .iter()
                    .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit())),
            "major.minor.build: {version}"
        );

        let detail = operating_system_detail();
        assert!(detail.contains("(amd64) version "), "{detail}");
        assert!(
            !detail.ends_with("version windows"),
            "the platform family is not an OS version: {detail}"
        );
    }

    /// The hook must survive a panic on a **worker** thread, which is where a crash in the
    /// network task or a rayon mesher would land.
    #[test]
    fn the_panic_hook_records_a_panic_from_any_thread() {
        // Take anything an earlier test in this binary left behind.
        let _ = take_recorded_crash();
        install_panic_hook();
        assert!(!has_crashed());

        let joined = std::thread::spawn(|| {
            panic!("terrain mesher exploded");
        })
        .join();
        assert!(joined.is_err(), "the thread must have panicked");

        assert!(has_crashed());
        let report = take_recorded_crash().expect("the hook must have recorded the panic");
        let text = report.complete_report();
        assert!(text.contains("terrain mesher exploded"), "{text}");
        assert!(
            text.contains("terrain mesher exploded\n\tat "),
            "the panic cause itself must retain frames: {text}"
        );
        assert!(text.contains("-- Panic --"), "{text}");
        assert!(text.contains("-- Head --\nStacktrace:"), "{text}");
        assert!(
            text.contains("-- Panic --\nDetails:\n\tLocation:")
                && text.contains("\nStacktrace:\n\tat "),
            "the panic category must retain automatic frames: {text}"
        );
        assert!(text.contains("Description: Unexpected error"), "{text}");
        assert!(!has_crashed(), "taking the report clears it");

        // Restore something harmless for the rest of the binary.
        let _ = std::panic::take_hook();
    }

    #[test]
    fn process_memory_is_readable_and_plausible() {
        let bytes = process_memory_bytes().expect("the OS must report this process's memory");
        assert!(
            bytes > 1024 * 1024,
            "a running test process uses more than a megabyte, got {bytes}"
        );
        assert!(
            bytes < 64 * 1024 * 1024 * 1024,
            "implausibly large: {bytes}"
        );
    }
}
