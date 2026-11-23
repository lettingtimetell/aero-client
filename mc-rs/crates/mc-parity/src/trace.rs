//! A reusable simulation-trace format and comparator, **bit-exact by default**.
//!
//! # Why this exists
//!
//! S13 proved player physics against the real 1.8.9 bytecode by dumping raw IEEE-754 bit
//! patterns from a JVM and replaying the same inputs through `mc-physics` at tolerance `0.0`.
//! That technique is the strongest evidence in this workspace, and it was written once, for one
//! subsystem, in a bespoke binary format. This module is that technique with the physics taken
//! out of it: a named, per-step numeric trace plus a comparator, so the next subsystem that
//! gets a Java reference generator does not have to re-invent either half.
//!
//! [`crate::physics_trace`] is the worked example — it reads S13's committed fixtures into a
//! [`Trace`] and re-proves them through [`Trace::compare`].
//!
//! # Bit-exact by default
//!
//! [`Tolerance::default`] is [`Tolerance::Exact`], and every value kind compares by **bit
//! pattern**, not by `==`. That distinction is not pedantry:
//!
//! * `-0.0 == 0.0` is true, but a sign bit that flipped means a `floor` went the other way;
//! * `NaN == NaN` is false, so an `==` comparator silently reports every NaN as a mismatch and
//!   an "all steps matched" comparator would report none at all.
//!
//! A tolerance is something a caller opts *into*, per channel, with a comment saying why. S13's
//! experience is the argument: its 1e-9 contract was met, but the implementations turned out to
//! agree exactly, and the exact test is the one that would catch an algebraically-equal
//! rewrite that rounds differently.
//!
//! # Format (`MCTRACE1`, all integers big-endian)
//!
//! ```text
//!   8  magic "MCTRACE1"
//!   1  format version (1)
//!   2  name length (u16) + name bytes, UTF-8
//!   2  channel count (u16)
//!      per channel: 2-byte name length + name bytes + 1 kind tag
//!   4  step count (u32)
//!      per step, per channel, in channel order: the value's fixed-width big-endian bytes
//! ```
//!
//! Fixed-width and column-ordered means a Java generator is a `DataOutputStream` and eight
//! lines of loop, which is the point: the format has to be trivial to emit from the *other*
//! side or nobody will emit it.

use std::collections::BTreeMap;

use crate::ParityError;

/// The trace container's magic bytes.
pub const TRACE_MAGIC: &[u8; 8] = b"MCTRACE1";

/// The trace format version this build reads and writes.
pub const TRACE_VERSION: u8 = 1;

/// Ceilings applied when parsing a trace, for the same reason [`crate::CaptureLimits`] exists.
const MAX_CHANNELS: u16 = 4096;
const MAX_STEPS: u32 = 16_000_000;
const MAX_NAME_LEN: u16 = 1024;

/// What a channel holds. Every kind is compared by bit pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// 64-bit float, stored as its IEEE-754 bits.
    F64,
    /// 32-bit float, stored as its IEEE-754 bits.
    F32,
    /// 64-bit signed integer.
    I64,
    /// 32-bit signed integer.
    I32,
    /// One bit, stored in a byte. `0` is false; any other value is rejected on parse.
    Bool,
}

impl Kind {
    /// The wire tag.
    #[must_use]
    pub const fn tag(self) -> u8 {
        match self {
            Kind::F64 => 0,
            Kind::F32 => 1,
            Kind::I64 => 2,
            Kind::I32 => 3,
            Kind::Bool => 4,
        }
    }

    /// How many bytes one value of this kind occupies.
    #[must_use]
    pub const fn width(self) -> usize {
        match self {
            Kind::F64 | Kind::I64 => 8,
            Kind::F32 | Kind::I32 => 4,
            Kind::Bool => 1,
        }
    }

    /// Parse a wire tag.
    ///
    /// # Errors
    /// [`ParityError::InvalidTag`] for an undefined tag.
    pub fn from_tag(tag: u8) -> Result<Kind, ParityError> {
        match tag {
            0 => Ok(Kind::F64),
            1 => Ok(Kind::F32),
            2 => Ok(Kind::I64),
            3 => Ok(Kind::I32),
            4 => Ok(Kind::Bool),
            other => Err(ParityError::InvalidTag {
                what: "trace channel kind",
                value: other,
            }),
        }
    }
}

/// One sampled value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    /// A 64-bit float.
    F64(f64),
    /// A 32-bit float.
    F32(f32),
    /// A 64-bit signed integer.
    I64(i64),
    /// A 32-bit signed integer.
    I32(i32),
    /// A boolean.
    Bool(bool),
}

impl Value {
    /// This value's kind.
    #[must_use]
    pub const fn kind(self) -> Kind {
        match self {
            Value::F64(_) => Kind::F64,
            Value::F32(_) => Kind::F32,
            Value::I64(_) => Kind::I64,
            Value::I32(_) => Kind::I32,
            Value::Bool(_) => Kind::Bool,
        }
    }

    /// The value's raw bits, widened to `u64`. This is what [`Tolerance::Exact`] compares, and
    /// it is why `-0.0` and `0.0` are *different* here while `==` would call them equal.
    #[must_use]
    pub fn bits(self) -> u64 {
        match self {
            Value::F64(v) => v.to_bits(),
            Value::F32(v) => u64::from(v.to_bits()),
            Value::I64(v) => v as u64,
            Value::I32(v) => u64::from(v as u32),
            Value::Bool(v) => u64::from(v),
        }
    }

    /// The value as an `f64`, for tolerance comparisons and for reporting an error magnitude.
    /// Integers and booleans widen losslessly (an `i64` beyond 2⁵³ loses precision, but a
    /// channel that large is compared exactly in practice).
    #[must_use]
    pub fn as_f64(self) -> f64 {
        match self {
            Value::F64(v) => v,
            Value::F32(v) => f64::from(v),
            Value::I64(v) => v as f64,
            Value::I32(v) => f64::from(v),
            Value::Bool(v) => f64::from(u8::from(v)),
        }
    }

    fn write_into(self, out: &mut Vec<u8>) {
        match self {
            Value::F64(v) => out.extend_from_slice(&v.to_bits().to_be_bytes()),
            Value::F32(v) => out.extend_from_slice(&v.to_bits().to_be_bytes()),
            Value::I64(v) => out.extend_from_slice(&v.to_be_bytes()),
            Value::I32(v) => out.extend_from_slice(&v.to_be_bytes()),
            Value::Bool(v) => out.push(u8::from(v)),
        }
    }
}

/// How much difference a channel is allowed.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Tolerance {
    /// Bit patterns must be identical. The default, and the standard S13 set.
    #[default]
    Exact,
    /// `|a - b|` must not exceed this. Use only where the reference generator itself is
    /// lossy — and say so at the call site.
    Abs(f64),
    /// The two floats must be within this many representable steps of one another. Values of
    /// different sign, or a NaN against a non-NaN, never match under this rule.
    Ulps(u32),
}

impl Tolerance {
    /// Whether `actual` is acceptable against `reference`.
    #[must_use]
    pub fn accepts(self, reference: Value, actual: Value) -> bool {
        if reference.kind() != actual.kind() {
            return false;
        }
        match self {
            Tolerance::Exact => reference.bits() == actual.bits(),
            Tolerance::Abs(eps) => {
                if reference.bits() == actual.bits() {
                    return true;
                }
                let (r, a) = (reference.as_f64(), actual.as_f64());
                if r.is_nan() || a.is_nan() {
                    return false;
                }
                (r - a).abs() <= eps
            }
            Tolerance::Ulps(n) => {
                if reference.bits() == actual.bits() {
                    return true;
                }
                let (r, a) = (reference.as_f64(), actual.as_f64());
                if r.is_nan() || a.is_nan() || r.is_sign_negative() != a.is_sign_negative() {
                    return false;
                }
                let (rb, ab) = (r.to_bits(), a.to_bits());
                rb.abs_diff(ab) <= u64::from(n)
            }
        }
    }
}

/// Per-channel tolerances, with a fallback for channels not named.
#[derive(Debug, Clone, Default)]
pub struct Tolerances {
    default: Tolerance,
    per_channel: BTreeMap<String, Tolerance>,
}

impl Tolerances {
    /// Everything bit-exact.
    #[must_use]
    pub fn exact() -> Tolerances {
        Tolerances::default()
    }

    /// Everything at this tolerance unless overridden.
    #[must_use]
    pub fn uniform(t: Tolerance) -> Tolerances {
        Tolerances {
            default: t,
            per_channel: BTreeMap::new(),
        }
    }

    /// Relax one channel. Chainable.
    #[must_use]
    pub fn with(mut self, channel: impl Into<String>, t: Tolerance) -> Tolerances {
        self.per_channel.insert(channel.into(), t);
        self
    }

    /// The tolerance in force for a channel.
    #[must_use]
    pub fn for_channel(&self, channel: &str) -> Tolerance {
        self.per_channel
            .get(channel)
            .copied()
            .unwrap_or(self.default)
    }
}

/// A named channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Channel {
    /// The channel's name. Comparison is by name, not by position, so a generator that adds a
    /// column in the middle does not silently shift every later comparison.
    pub name: String,
    /// What it holds.
    pub kind: Kind,
}

/// A rectangular table of per-step samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    /// What this is a trace of — the scenario name, matched between reference and actual.
    pub name: String,
    channels: Vec<Channel>,
    /// `steps[step][channel]`, always rectangular.
    steps: Vec<Vec<Value>>,
}

impl Trace {
    /// An empty trace with a fixed channel list.
    #[must_use]
    pub fn new(name: impl Into<String>, channels: Vec<Channel>) -> Trace {
        Trace {
            name: name.into(),
            channels,
            steps: Vec::new(),
        }
    }

    /// The channel list, in column order.
    #[must_use]
    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }

    /// How many steps this trace holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether this trace holds no steps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// One step's values, in channel order.
    #[must_use]
    pub fn step(&self, index: usize) -> Option<&[Value]> {
        self.steps.get(index).map(Vec::as_slice)
    }

    /// One channel's values across every step.
    #[must_use]
    pub fn column(&self, channel: &str) -> Option<Vec<Value>> {
        let i = self.channels.iter().position(|c| c.name == channel)?;
        Some(
            self.steps
                .iter()
                .filter_map(|s| s.get(i).copied())
                .collect(),
        )
    }

    /// Append a step.
    ///
    /// # Errors
    /// [`ParityError::ShapeMismatch`] if the row's length or any value's kind disagrees with
    /// the channel list. Enforcing this on the way in is what lets every reader assume the
    /// table is rectangular and correctly typed.
    pub fn push(&mut self, values: Vec<Value>) -> Result<(), ParityError> {
        if values.len() != self.channels.len() {
            return Err(ParityError::ShapeMismatch(format!(
                "trace `{}` has {} channels but step {} has {} values",
                self.name,
                self.channels.len(),
                self.steps.len(),
                values.len()
            )));
        }
        for (channel, value) in self.channels.iter().zip(&values) {
            if channel.kind != value.kind() {
                return Err(ParityError::ShapeMismatch(format!(
                    "trace `{}` step {}: channel `{}` is {:?} but the value is {:?}",
                    self.name,
                    self.steps.len(),
                    channel.name,
                    channel.kind,
                    value.kind()
                )));
            }
        }
        self.steps.push(values);
        Ok(())
    }

    /// Compare `self` (the reference) with `actual`.
    ///
    /// Channels are matched **by name**. A channel present in one trace and not the other is
    /// reported rather than skipped, because a silently-dropped channel is how a comparator
    /// starts passing for the wrong reason.
    ///
    /// # Errors
    /// [`ParityError::ShapeMismatch`] if the step counts differ. That is a structural problem,
    /// not a divergence, and reporting it as "step 0 mismatched" would be misleading.
    pub fn compare(
        &self,
        actual: &Trace,
        tolerances: &Tolerances,
    ) -> Result<TraceDiff, ParityError> {
        if self.len() != actual.len() {
            return Err(ParityError::ShapeMismatch(format!(
                "trace `{}`: reference has {} steps, actual has {}",
                self.name,
                self.len(),
                actual.len()
            )));
        }

        let mut diff = TraceDiff {
            name: self.name.clone(),
            steps: self.len(),
            ..TraceDiff::default()
        };

        for channel in &self.channels {
            if !actual.channels.iter().any(|c| c.name == channel.name) {
                diff.missing_channels.push(channel.name.clone());
            }
        }
        for channel in &actual.channels {
            if !self.channels.iter().any(|c| c.name == channel.name) {
                diff.extra_channels.push(channel.name.clone());
            }
        }

        let pairs: Vec<(usize, usize, &Channel)> = self
            .channels
            .iter()
            .enumerate()
            .filter_map(|(ri, c)| {
                actual
                    .channels
                    .iter()
                    .position(|a| a.name == c.name)
                    .map(|ai| (ri, ai, c))
            })
            .collect();
        diff.compared_channels = pairs.len();

        for step in 0..self.len() {
            for &(ri, ai, channel) in &pairs {
                let (Some(r), Some(a)) = (
                    self.steps.get(step).and_then(|s| s.get(ri)).copied(),
                    actual.steps.get(step).and_then(|s| s.get(ai)).copied(),
                ) else {
                    continue;
                };
                let tolerance = tolerances.for_channel(&channel.name);
                let error = (r.as_f64() - a.as_f64()).abs();
                let worst = diff.worst.entry(channel.name.clone()).or_insert(0.0);
                if error > *worst || (error.is_nan() && !worst.is_nan()) {
                    *worst = error;
                }
                if !tolerance.accepts(r, a) {
                    if diff.first_divergence.is_none() {
                        diff.first_divergence = Some(Divergence {
                            step,
                            channel: channel.name.clone(),
                            reference: r,
                            actual: a,
                            tolerance,
                        });
                    }
                    diff.mismatches += 1;
                    *diff
                        .mismatches_by_channel
                        .entry(channel.name.clone())
                        .or_default() += 1;
                }
            }
        }

        Ok(diff)
    }

    /// Serialise to the `MCTRACE1` container.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(TRACE_MAGIC);
        out.push(TRACE_VERSION);
        let name = self.name.as_bytes();
        let name = &name[..name.len().min(MAX_NAME_LEN as usize)];
        out.extend_from_slice(&(name.len() as u16).to_be_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&(self.channels.len() as u16).to_be_bytes());
        for c in &self.channels {
            let cn = c.name.as_bytes();
            let cn = &cn[..cn.len().min(MAX_NAME_LEN as usize)];
            out.extend_from_slice(&(cn.len() as u16).to_be_bytes());
            out.extend_from_slice(cn);
            out.push(c.kind.tag());
        }
        out.extend_from_slice(&(self.steps.len() as u32).to_be_bytes());
        for step in &self.steps {
            for value in step {
                value.write_into(&mut out);
            }
        }
        out
    }

    /// Parse an `MCTRACE1` container.
    ///
    /// # Errors
    /// The same untrusted-input variants [`crate::Capture::parse`] uses. A trace generated by a
    /// Java program on someone else's machine is exactly as untrusted as a capture.
    pub fn decode(bytes: &[u8]) -> Result<Trace, ParityError> {
        let mut r = Reader::new(bytes);
        let magic = r.take(8, "trace magic")?;
        if magic != TRACE_MAGIC {
            return Err(ParityError::BadMagic {
                expected: "MCTRACE1",
                found: String::from_utf8_lossy(magic).into_owned(),
            });
        }
        let version = r.u8("format version")?;
        if version != TRACE_VERSION {
            return Err(ParityError::UnsupportedVersion {
                container: "trace",
                found: version,
                supported: TRACE_VERSION,
            });
        }
        let name = r.string("trace name")?;

        let channel_count = r.u16("channel count")?;
        if channel_count > MAX_CHANNELS {
            return Err(ParityError::LimitExceeded {
                what: "channel count",
                value: u64::from(channel_count),
                max: u64::from(MAX_CHANNELS),
            });
        }
        let mut channels = Vec::with_capacity(channel_count as usize);
        for _ in 0..channel_count {
            let name = r.string("channel name")?;
            let kind = Kind::from_tag(r.u8("channel kind")?)?;
            channels.push(Channel { name, kind });
        }

        let step_count = r.u32("step count")?;
        if step_count > MAX_STEPS {
            return Err(ParityError::LimitExceeded {
                what: "step count",
                value: u64::from(step_count),
                max: u64::from(MAX_STEPS),
            });
        }
        let row_width: usize = channels.iter().map(|c| c.kind.width()).sum();
        let needed = (step_count as usize).saturating_mul(row_width);
        if r.remaining() < needed {
            return Err(ParityError::Truncated {
                what: "trace steps",
                need: needed,
                have: r.remaining(),
            });
        }

        let mut steps = Vec::with_capacity(step_count as usize);
        for _ in 0..step_count {
            let mut row = Vec::with_capacity(channels.len());
            for c in &channels {
                row.push(match c.kind {
                    Kind::F64 => Value::F64(f64::from_bits(r.u64("f64 value")?)),
                    Kind::F32 => Value::F32(f32::from_bits(r.u32("f32 value")?)),
                    Kind::I64 => Value::I64(r.u64("i64 value")? as i64),
                    Kind::I32 => Value::I32(r.u32("i32 value")? as i32),
                    Kind::Bool => match r.u8("bool value")? {
                        0 => Value::Bool(false),
                        1 => Value::Bool(true),
                        other => {
                            return Err(ParityError::InvalidTag {
                                what: "trace boolean",
                                value: other,
                            })
                        }
                    },
                });
            }
            steps.push(row);
        }
        if r.remaining() != 0 {
            return Err(ParityError::TrailingBytes {
                what: "the trace steps",
                count: r.remaining(),
            });
        }

        Ok(Trace {
            name,
            channels,
            steps,
        })
    }
}

/// Where a comparison first went wrong.
#[derive(Debug, Clone, PartialEq)]
pub struct Divergence {
    /// The step index.
    pub step: usize,
    /// The channel name.
    pub channel: String,
    /// What the reference said.
    pub reference: Value,
    /// What the implementation produced.
    pub actual: Value,
    /// The tolerance that rejected it.
    pub tolerance: Tolerance,
}

/// The result of comparing two traces.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TraceDiff {
    /// The reference trace's name.
    pub name: String,
    /// How many steps were compared.
    pub steps: usize,
    /// How many channels were compared (present in both traces).
    pub compared_channels: usize,
    /// Channels in the reference that the actual trace does not have.
    pub missing_channels: Vec<String>,
    /// Channels in the actual trace that the reference does not have.
    pub extra_channels: Vec<String>,
    /// Total number of rejected samples.
    pub mismatches: usize,
    /// Rejected samples per channel.
    pub mismatches_by_channel: BTreeMap<String, usize>,
    /// The largest `|reference - actual|` seen per channel, whether or not it was rejected.
    /// A run that passes with a worst error of exactly `0.0` everywhere is *bit-exact*, which
    /// is a stronger statement than "passed", and worth printing.
    pub worst: BTreeMap<String, f64>,
    /// The first rejected sample, in step then channel order.
    pub first_divergence: Option<Divergence>,
}

impl TraceDiff {
    /// Whether every compared sample was accepted and no channel was missing or extra.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.mismatches == 0 && self.missing_channels.is_empty() && self.extra_channels.is_empty()
    }

    /// Whether every compared sample was *identical*, not merely within tolerance.
    #[must_use]
    pub fn is_bit_exact(&self) -> bool {
        self.is_clean() && self.worst.values().all(|w| *w == 0.0)
    }

    /// The largest error across every channel.
    #[must_use]
    pub fn worst_error(&self) -> f64 {
        self.worst.values().copied().fold(0.0, f64::max)
    }

    /// A one-line summary suitable for a test's stdout.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{}: {} steps × {} channels, {} mismatches, worst error {:e}{}",
            self.name,
            self.steps,
            self.compared_channels,
            self.mismatches,
            self.worst_error(),
            if self.is_bit_exact() {
                " (bit-exact)"
            } else {
                ""
            }
        )
    }
}

/// A checked reader, as in [`crate::capture`].
struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Reader<'a> {
        Reader { data, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.offset
    }

    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], ParityError> {
        if self.remaining() < n {
            return Err(ParityError::Truncated {
                what,
                need: n,
                have: self.remaining(),
            });
        }
        let out = &self.data[self.offset..self.offset + n];
        self.offset += n;
        Ok(out)
    }

    fn u8(&mut self, what: &'static str) -> Result<u8, ParityError> {
        Ok(self.take(1, what)?[0])
    }

    fn u16(&mut self, what: &'static str) -> Result<u16, ParityError> {
        let b = self.take(2, what)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, ParityError> {
        let b = self.take(4, what)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, ParityError> {
        let b = self.take(8, what)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_be_bytes(a))
    }

    fn string(&mut self, what: &'static str) -> Result<String, ParityError> {
        let len = self.u16(what)?;
        if len > MAX_NAME_LEN {
            return Err(ParityError::LimitExceeded {
                what: "trace name length",
                value: u64::from(len),
                max: u64::from(MAX_NAME_LEN),
            });
        }
        let bytes = self.take(len as usize, what)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| ParityError::InvalidUtf8("a trace name"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channels() -> Vec<Channel> {
        vec![
            Channel {
                name: "pos.x".into(),
                kind: Kind::F64,
            },
            Channel {
                name: "fall".into(),
                kind: Kind::F32,
            },
            Channel {
                name: "flags".into(),
                kind: Kind::I32,
            },
            Channel {
                name: "on_ground".into(),
                kind: Kind::Bool,
            },
        ]
    }

    fn trace(name: &str, xs: &[f64]) -> Trace {
        let mut t = Trace::new(name, channels());
        for (i, x) in xs.iter().enumerate() {
            t.push(vec![
                Value::F64(*x),
                Value::F32(i as f32),
                Value::I32(i as i32),
                Value::Bool(i % 2 == 0),
            ])
            .expect("well-shaped step");
        }
        t
    }

    #[test]
    fn a_trace_round_trips_through_the_container() {
        let t = trace("flat_walk", &[1.0, -0.0, f64::NAN, f64::INFINITY, 1e-300]);
        let back = Trace::decode(&t.encode()).expect("decode");
        // `PartialEq` on NaN is false, so compare the encodings — which is the bit-level
        // statement this format exists to make.
        assert_eq!(back.encode(), t.encode());
        assert_eq!(back.name, "flat_walk");
        assert_eq!(back.channels(), t.channels());
        assert_eq!(back.len(), 5);
    }

    #[test]
    fn identical_traces_compare_bit_exact() {
        let a = trace("s", &[1.0, 2.0, 3.0]);
        let b = trace("s", &[1.0, 2.0, 3.0]);
        let d = a.compare(&b, &Tolerances::exact()).expect("compare");
        assert!(d.is_clean(), "{d:?}");
        assert!(d.is_bit_exact(), "{d:?}");
        assert_eq!(d.compared_channels, 4);
        assert_eq!(d.steps, 3);
    }

    #[test]
    fn negative_zero_is_a_mismatch_under_exact_even_though_it_compares_equal() {
        let a = trace("s", &[0.0]);
        let b = trace("s", &[-0.0]);
        // The language calls these equal; the whole point of comparing by bits is that we do
        // not. Written through `std::cmp::PartialEq::eq` so clippy does not fold the constant.
        assert!(
            f64::eq(&0.0, &-0.0),
            "the language says these two are the same value"
        );
        let d = a.compare(&b, &Tolerances::exact()).expect("compare");
        assert!(
            !d.is_clean(),
            "a flipped sign bit means a floor went the other way; it must not pass"
        );
        assert_eq!(d.mismatches, 1);
        assert_eq!(d.first_divergence.as_ref().expect("divergence").step, 0);
    }

    #[test]
    fn nan_matches_itself_under_exact_even_though_it_compares_unequal() {
        let a = trace("s", &[f64::NAN]);
        let b = trace("s", &[f64::NAN]);
        let d = a.compare(&b, &Tolerances::exact()).expect("compare");
        assert!(
            d.is_clean(),
            "identical bit patterns must match, NaN included: {d:?}"
        );
    }

    #[test]
    fn a_nan_never_satisfies_a_numeric_tolerance() {
        let a = trace("s", &[f64::NAN]);
        let b = trace("s", &[1.0]);
        for t in [Tolerance::Abs(1e300), Tolerance::Ulps(u32::MAX)] {
            let d = a.compare(&b, &Tolerances::uniform(t)).expect("compare");
            assert!(!d.is_clean(), "{t:?} accepted a NaN against a number");
        }
    }

    #[test]
    fn a_tolerance_is_per_channel() {
        let mut a = Trace::new("s", channels());
        let mut b = Trace::new("s", channels());
        a.push(vec![
            Value::F64(1.0),
            Value::F32(1.0),
            Value::I32(0),
            Value::Bool(true),
        ])
        .expect("a");
        b.push(vec![
            Value::F64(1.000_1),
            Value::F32(1.5),
            Value::I32(0),
            Value::Bool(true),
        ])
        .expect("b");

        let loose = Tolerances::exact().with("pos.x", Tolerance::Abs(1e-3));
        let d = a.compare(&b, &loose).expect("compare");
        assert_eq!(d.mismatches, 1, "only `fall` should be rejected: {d:?}");
        assert_eq!(
            d.first_divergence.as_ref().expect("divergence").channel,
            "fall"
        );
    }

    #[test]
    fn ulps_rejects_a_sign_flip_however_small() {
        let a = trace("s", &[f64::MIN_POSITIVE]);
        let b = trace("s", &[-f64::MIN_POSITIVE]);
        let d = a
            .compare(&b, &Tolerances::uniform(Tolerance::Ulps(u32::MAX)))
            .expect("compare");
        assert!(!d.is_clean(), "opposite signs must never be within N ulps");
    }

    #[test]
    fn a_missing_channel_is_reported_not_skipped() {
        let a = trace("s", &[1.0]);
        let mut b = Trace::new(
            "s",
            vec![Channel {
                name: "pos.x".into(),
                kind: Kind::F64,
            }],
        );
        b.push(vec![Value::F64(1.0)]).expect("b");
        let d = a.compare(&b, &Tolerances::exact()).expect("compare");
        assert_eq!(d.mismatches, 0, "the one shared channel does match");
        assert!(!d.is_clean(), "but three channels went unchecked");
        assert_eq!(d.missing_channels, vec!["fall", "flags", "on_ground"]);
        assert_eq!(d.compared_channels, 1);
    }

    #[test]
    fn channels_are_matched_by_name_not_by_position() {
        let a = trace("s", &[1.0]);
        let mut reordered = Trace::new(
            "s",
            vec![
                Channel {
                    name: "on_ground".into(),
                    kind: Kind::Bool,
                },
                Channel {
                    name: "flags".into(),
                    kind: Kind::I32,
                },
                Channel {
                    name: "fall".into(),
                    kind: Kind::F32,
                },
                Channel {
                    name: "pos.x".into(),
                    kind: Kind::F64,
                },
            ],
        );
        reordered
            .push(vec![
                Value::Bool(true),
                Value::I32(0),
                Value::F32(0.0),
                Value::F64(1.0),
            ])
            .expect("row");
        let d = a
            .compare(&reordered, &Tolerances::exact())
            .expect("compare");
        assert!(d.is_bit_exact(), "{d:?}");
    }

    #[test]
    fn differing_step_counts_are_a_shape_error_not_a_divergence() {
        let a = trace("s", &[1.0, 2.0]);
        let b = trace("s", &[1.0]);
        let err = a
            .compare(&b, &Tolerances::exact())
            .expect_err("step count mismatch");
        assert!(matches!(err, ParityError::ShapeMismatch(_)), "{err:?}");
    }

    #[test]
    fn a_wrongly_typed_step_is_refused_on_the_way_in() {
        let mut t = Trace::new("s", channels());
        let err = t
            .push(vec![
                Value::F32(1.0),
                Value::F32(1.0),
                Value::I32(0),
                Value::Bool(true),
            ])
            .expect_err("kind mismatch");
        assert!(matches!(err, ParityError::ShapeMismatch(_)), "{err:?}");
        assert!(t.is_empty());
    }

    #[test]
    fn decode_rejects_every_truncation() {
        let bytes = trace("s", &[1.0, 2.0]).encode();
        for cut in 0..bytes.len() {
            assert!(
                Trace::decode(&bytes[..cut]).is_err(),
                "a {cut}-byte prefix decoded successfully"
            );
        }
    }

    #[test]
    fn decode_rejects_trailing_bytes_and_bad_tags() {
        let mut bytes = trace("s", &[1.0]).encode();
        bytes.push(0);
        assert!(matches!(
            Trace::decode(&bytes),
            Err(ParityError::TrailingBytes { .. })
        ));

        let mut bytes = trace("s", &[1.0]).encode();
        let last = bytes.len() - 1;
        bytes[last] = 2; // the trailing `Bool` byte
        assert!(matches!(
            Trace::decode(&bytes),
            Err(ParityError::InvalidTag { .. })
        ));
    }

    #[test]
    fn column_extracts_one_channel_across_steps() {
        let t = trace("s", &[1.0, 2.0, 3.0]);
        assert_eq!(
            t.column("pos.x"),
            Some(vec![Value::F64(1.0), Value::F64(2.0), Value::F64(3.0)])
        );
        assert_eq!(t.column("nope"), None);
    }
}
