//! Depth and allocation limits for parsing untrusted NBT.
//!
//! java: `NBTSizeTracker.java`
//!
//! Vanilla charges a per-tag overhead in *bits* against a byte budget (`read += bits / 8`),
//! which makes the budget an accounting figure rather than a real byte count — a compound
//! costs 384 "bytes" before any of its contents are read. We reproduce the same numbers so
//! that a payload vanilla accepts, we accept.
//!
//! Two deliberate departures, both strictly more conservative:
//!
//! * Vanilla computes the per-element charges (`8 * i`, `32 * i`, `16 * len`) in 32-bit
//!   `int` arithmetic, so a hostile length near `Integer.MAX_VALUE` **overflows to a small
//!   or negative number**, sails past the tracker, and then reaches `new byte[i]`. We widen
//!   to `u64` and saturate, so the length is charged honestly.
//! * The reader never sizes an allocation from a length field. Arrays are filled from the
//!   stream as bytes actually arrive, so a bogus 2-billion-element header costs nothing
//!   beyond the read that fails.

use crate::error::NbtError;

/// How much untrusted NBT we are willing to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NbtLimits {
    /// `NBTSizeTracker` budget, or `None` for vanilla's `NBTSizeTracker.INFINITE`.
    pub max_bytes: Option<u64>,
    /// Reject nesting deeper than this. Vanilla's constant is 512.
    pub max_depth: u32,
}

impl NbtLimits {
    /// Vanilla's depth cap. java: `NBTTagCompound.read`, `depth > 512`
    pub const VANILLA_MAX_DEPTH: u32 = 512;

    /// The limits vanilla applies to NBT arriving over the network: a 2 MiB budget.
    ///
    /// java: `PacketBuffer.readNBTTagCompoundFromBuffer`, `new NBTSizeTracker(2097152L)`
    ///
    /// This is the default, because network NBT is the only NBT an attacker controls.
    pub const PROTOCOL: NbtLimits = NbtLimits {
        max_bytes: Some(2 * 1024 * 1024),
        max_depth: Self::VANILLA_MAX_DEPTH,
    };

    /// No size budget — what vanilla uses for files on disk.
    ///
    /// java: `CompressedStreamTools.read`, `NBTSizeTracker.INFINITE`
    ///
    /// Still depth-limited, and the reader still refuses to pre-allocate from a length
    /// field, so this is safe to point at a file of unknown provenance.
    pub const UNLIMITED: NbtLimits = NbtLimits {
        max_bytes: None,
        max_depth: Self::VANILLA_MAX_DEPTH,
    };
}

impl Default for NbtLimits {
    fn default() -> Self {
        NbtLimits::PROTOCOL
    }
}

/// Running total of the [`NbtLimits::max_bytes`] budget.
///
/// java: `NBTSizeTracker.java`
#[derive(Debug, Clone)]
pub struct SizeTracker {
    max: Option<u64>,
    used: u64,
}

impl SizeTracker {
    /// A tracker enforcing `limits`.
    #[must_use]
    pub fn new(limits: NbtLimits) -> Self {
        Self {
            max: limits.max_bytes,
            used: 0,
        }
    }

    /// Accounting bytes consumed so far.
    #[must_use]
    pub fn used(&self) -> u64 {
        self.used
    }

    /// Charge `bits` bits (vanilla's unit) against the budget.
    ///
    /// java: `NBTSizeTracker.read(long bits)`
    pub fn charge_bits(&mut self, bits: u64) -> Result<(), NbtError> {
        self.charge_bytes(bits / 8)
    }

    /// Charge `bytes` bytes against the budget.
    pub fn charge_bytes(&mut self, bytes: u64) -> Result<(), NbtError> {
        self.used = self.used.saturating_add(bytes);
        match self.max {
            Some(max) if self.used > max => Err(NbtError::SizeLimit {
                limit: max,
                requested: self.used,
            }),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_are_divided_by_eight_like_vanilla() {
        let mut t = SizeTracker::new(NbtLimits {
            max_bytes: Some(100),
            max_depth: 512,
        });
        t.charge_bits(384).expect("384 bits = 48 bytes");
        assert_eq!(t.used(), 48);
        t.charge_bits(384).expect("96 bytes still fits");
        assert_eq!(t.used(), 96);
        assert!(matches!(
            t.charge_bits(384),
            Err(NbtError::SizeLimit { limit: 100, .. })
        ));
    }

    #[test]
    fn unlimited_never_trips() {
        let mut t = SizeTracker::new(NbtLimits::UNLIMITED);
        t.charge_bytes(u64::MAX).expect("no budget, no failure");
        t.charge_bytes(u64::MAX).expect("and it saturates");
        assert_eq!(t.used(), u64::MAX);
    }

    #[test]
    fn hostile_length_does_not_overflow_the_charge() {
        // Vanilla computes `(long)(8 * i)` in int arithmetic: 8 * 0x2000_0000 == 0, which
        // charges nothing and then allocates 512 MB. Our u64 charge rejects it.
        let mut t = SizeTracker::new(NbtLimits::PROTOCOL);
        let hostile = 0x2000_0000i32;
        assert_eq!(
            hostile.wrapping_mul(8),
            0,
            "the vanilla overflow, for the record"
        );
        assert!(t.charge_bits(8 * u64::try_from(hostile).unwrap()).is_err());
    }
}
