//! Gzip and zlib wrappers — the `CompressedStreamTools` equivalents.
//!
//! `level.dat`, `playerdata/*.dat` and friends are gzipped; region-file chunk payloads are
//! zlib-deflated. `servers.dat` is stored raw, so always sniff before assuming.
//!
//! java: `CompressedStreamTools.java`

use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use flate2::read::{GzDecoder, ZlibDecoder};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use flate2::GzBuilder;

use crate::error::NbtError;
use crate::limits::NbtLimits;
use crate::tag::Nbt;

/// Leading bytes of a gzip member: `1f 8b`.
pub const GZIP_MAGIC: [u8; 2] = [0x1F, 0x8B];

/// How a `.dat` file's NBT is framed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// Raw NBT, no container. `servers.dat` is stored this way.
    Raw,
    /// Gzip. `level.dat` and most single-file saves.
    Gzip,
    /// Raw zlib/deflate. Region-file chunk payloads.
    Zlib,
}

/// Guess how `bytes` is framed by looking at its first two bytes.
///
/// Unambiguous in practice: NBT always starts with a tag type in `0..=11`, gzip with
/// `1f 8b`, and a zlib header's first byte is `0x78` for every deflate window size the
/// vanilla writer uses.
#[must_use]
pub fn sniff(bytes: &[u8]) -> Framing {
    match bytes {
        [0x1F, 0x8B, ..] => Framing::Gzip,
        // zlib: CMF low nibble 8 = deflate, and (CMF<<8 | FLG) % 31 == 0.
        [cmf, flg, ..]
            if cmf & 0x0F == 0x08
                && (u16::from(*cmf) << 8 | u16::from(*flg)) % 31 == 0
                && *cmf > 11 =>
        {
            Framing::Zlib
        }
        _ => Framing::Raw,
    }
}

/// Read a gzipped root tag, using the unbounded file budget by default.
///
/// java: `CompressedStreamTools.readCompressed` passes `NBTSizeTracker.INFINITE`; use
/// [`read_gzip_with_limits`] when the compressed stream is untrusted.
///
/// java: `CompressedStreamTools.readCompressed`
pub fn read_gzip(r: &mut impl Read) -> Result<(String, Nbt), NbtError> {
    read_gzip_with_limits(r, NbtLimits::UNLIMITED)
}

/// [`read_gzip`] with explicit limits. Use [`NbtLimits::UNLIMITED`] for local files.
pub fn read_gzip_with_limits(
    r: &mut impl Read,
    limits: NbtLimits,
) -> Result<(String, Nbt), NbtError> {
    let mut decoder = GzDecoder::new(r);
    crate::read::read_with_limits(&mut decoder, limits)
}

/// Read a zlib-deflated root tag, using the unbounded file budget by default.
///
/// Region-file NBT is local compressed data, so its default mirrors the file-oriented vanilla
/// readers. Use [`read_zlib_with_limits`] when the compressed stream is untrusted.
pub fn read_zlib(r: &mut impl Read) -> Result<(String, Nbt), NbtError> {
    read_zlib_with_limits(r, NbtLimits::UNLIMITED)
}

/// [`read_zlib`] with explicit limits.
pub fn read_zlib_with_limits(
    r: &mut impl Read,
    limits: NbtLimits,
) -> Result<(String, Nbt), NbtError> {
    let mut decoder = ZlibDecoder::new(r);
    crate::read::read_with_limits(&mut decoder, limits)
}

/// Read a root tag from bytes framed any of the three ways, sniffing which.
///
/// This is what you want for a `.dat` file of unknown provenance — vanilla's own save
/// files are inconsistent about it.
pub fn read_any(bytes: &[u8], limits: NbtLimits) -> Result<(String, Nbt), NbtError> {
    match sniff(bytes) {
        Framing::Raw => crate::read::from_slice_with_limits(bytes, limits),
        Framing::Gzip => read_gzip_with_limits(&mut &bytes[..], limits),
        Framing::Zlib => read_zlib_with_limits(&mut &bytes[..], limits),
    }
}

/// Write a root tag, gzipped. java: `CompressedStreamTools.writeCompressed`
pub fn write_gzip(w: &mut impl Write, name: &str, v: &Nbt) -> Result<(), NbtError> {
    // java.util.zip.GZIPOutputStream writes the fixed RFC 1952 OS field as zero. flate2's
    // convenience constructor uses 0xFF (unknown), which is valid gzip but not vanilla's
    // header. The other fixed header fields (including MTIME=0) are set by GzBuilder's
    // defaults, matching Java's constructor.
    let mut encoder = GzBuilder::new()
        .mtime(0)
        .operating_system(0)
        .write(w, Compression::default());
    crate::write::write(&mut encoder, name, v)?;
    encoder.finish()?;
    Ok(())
}

/// Write a root tag, zlib-deflated.
pub fn write_zlib(w: &mut impl Write, name: &str, v: &Nbt) -> Result<(), NbtError> {
    let mut encoder = ZlibEncoder::new(w, Compression::default());
    crate::write::write(&mut encoder, name, v)?;
    encoder.finish()?;
    Ok(())
}

/// Write an uncompressed compound root through vanilla's sibling `_tmp` file.
///
/// This is the filesystem form used by `ServerList.saveServerList` for `servers.dat`. Like Java
/// 1.8.9's `CompressedStreamTools.safeWrite`, the complete replacement is written to
/// `<file>_tmp` (a stale one from an earlier crash is overwritten) and then renamed into place,
/// so a serialization failure never touches the destination. Two deliberate differences, both
/// closing a window in which a crash or power cut lost the player's server list: the temporary
/// is flushed to the disk before the rename, and the old destination is never deleted first
/// (vanilla deletes it because `File.renameTo` cannot replace a file on Windows; the rename here
/// replaces it in one step). See [`mc_core::durable::write_atomic_via`].
///
/// # Errors
///
/// Returns [`NbtError::Io`] if the temporary file cannot be written or flushed or the
/// replacement cannot be renamed into place. The old destination is untouched in every error
/// case.
///
/// java: `CompressedStreamTools.safeWrite`
pub fn safe_write(path: impl AsRef<Path>, value: &crate::tag::NbtCompound) -> Result<(), NbtError> {
    let target = absolute_path(path.as_ref())?;
    let temporary = suffixed_path(&target, "_tmp");

    let mut bytes = Vec::new();
    crate::write::write_compound(&mut bytes, "", value)?;
    mc_core::durable::write_atomic_via(&target, &temporary, &bytes)?;
    Ok(())
}

fn absolute_path(path: &Path) -> Result<PathBuf, NbtError> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn suffixed_path(path: &Path, suffix: &str) -> PathBuf {
    let mut value: OsString = path.as_os_str().to_owned();
    value.push(suffix);
    PathBuf::from(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tag::NbtCompound;

    fn sample() -> Nbt {
        let mut c = NbtCompound::new();
        c.insert("LevelName", Nbt::String("mcpworld".into()));
        c.insert("Time", Nbt::Long(1620));
        c.insert("blob", Nbt::ByteArray(vec![7; 4096]));
        Nbt::Compound(c)
    }

    #[test]
    fn gzip_round_trips() {
        let mut buf = Vec::new();
        write_gzip(&mut buf, "root", &sample()).expect("gzip write");
        assert_eq!(&buf[..2], &GZIP_MAGIC);
        assert_eq!(buf[4..8], [0, 0, 0, 0], "Java writes an epoch gzip MTIME");
        assert_eq!(buf[9], 0, "Java's fixed gzip OS field is FAT/transparent");
        assert!(buf.len() < 4096, "4 KiB of identical bytes should compress");
        let (name, value) =
            read_gzip_with_limits(&mut &buf[..], NbtLimits::UNLIMITED).expect("read");
        assert_eq!(name, "root");
        assert_eq!(value, sample());
    }

    #[test]
    fn zlib_round_trips() {
        let mut buf = Vec::new();
        write_zlib(&mut buf, "root", &sample()).expect("zlib write");
        let (name, value) =
            read_zlib_with_limits(&mut &buf[..], NbtLimits::UNLIMITED).expect("read");
        assert_eq!(name, "root");
        assert_eq!(value, sample());
    }

    #[test]
    fn local_compressed_reader_defaults_are_unbounded_like_vanilla() {
        // The protocol budget is 2 MiB of NBTSizeTracker accounting bytes. A local file may
        // exceed it; this payload is deliberately just over that boundary while remaining
        // cheap to compress and read. The explicit bounded APIs remain available for hostile
        // compressed input.
        let mut compound = NbtCompound::new();
        compound.insert("large", Nbt::ByteArray(vec![0xA5; 2_100_000]));
        let value = Nbt::Compound(compound);

        let mut gzip = Vec::new();
        write_gzip(&mut gzip, "", &value).expect("gzip write");
        let (_, decoded) = read_gzip(&mut &gzip[..]).expect("vanilla's infinite file budget");
        assert_eq!(decoded, value);

        let mut zlib = Vec::new();
        write_zlib(&mut zlib, "", &value).expect("zlib write");
        let (_, decoded) = read_zlib(&mut &zlib[..]).expect("local zlib's infinite file budget");
        assert_eq!(decoded, value);

        assert!(matches!(
            read_gzip_with_limits(&mut &gzip[..], NbtLimits::PROTOCOL),
            Err(NbtError::SizeLimit { .. })
        ));
    }

    #[test]
    fn sniffing_tells_the_three_framings_apart() {
        let raw = crate::write::to_vec("", &sample()).expect("raw");
        let mut gz = Vec::new();
        write_gzip(&mut gz, "", &sample()).expect("gzip");
        let mut zl = Vec::new();
        write_zlib(&mut zl, "", &sample()).expect("zlib");

        assert_eq!(sniff(&raw), Framing::Raw);
        assert_eq!(sniff(&gz), Framing::Gzip);
        assert_eq!(sniff(&zl), Framing::Zlib);

        for bytes in [&raw, &gz, &zl] {
            let (_, value) = read_any(bytes, NbtLimits::UNLIMITED).expect("read_any");
            assert_eq!(value, sample());
        }
    }

    #[test]
    fn a_raw_compound_is_never_mistaken_for_zlib() {
        // 0x0a is a TAG_Compound lead byte; its low nibble is 0x0a, not 8, so the zlib
        // arm cannot match. Check every legal tag type for the same reason.
        for id in 0u8..=11 {
            assert_eq!(sniff(&[id, 0x00]), Framing::Raw, "tag id {id}");
        }
    }

    #[test]
    fn safe_write_replaces_the_destination_and_consumes_the_sibling_temp_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("servers.dat");
        let temporary = suffixed_path(&path, "_tmp");
        std::fs::write(&path, b"old destination").expect("old destination");
        std::fs::write(&temporary, b"stale temporary").expect("stale temporary");

        let Nbt::Compound(compound) = sample() else {
            unreachable!()
        };
        safe_write(&path, &compound).expect("safe write");

        assert!(
            !temporary.exists(),
            "the rename must consume the temporary file"
        );
        let bytes = std::fs::read(&path).expect("replacement");
        let (name, value) = crate::read::from_slice_with_limits(&bytes, NbtLimits::UNLIMITED)
            .expect("replacement is raw NBT");
        assert_eq!(name, "");
        assert_eq!(value, Nbt::Compound(compound));
    }

    #[test]
    fn a_failed_temporary_serialization_does_not_truncate_the_destination() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("servers.dat");
        let original = b"the old server list survives";
        std::fs::write(&path, original).expect("old destination");

        let mut invalid = NbtCompound::new();
        invalid.insert(
            "mixed",
            Nbt::list(vec![Nbt::Int(1), Nbt::String("not an int".to_owned())]),
        );
        assert!(safe_write(&path, &invalid).is_err());

        assert_eq!(
            std::fs::read(&path).expect("preserved destination"),
            original,
            "writing directly to the destination would truncate it before this error"
        );
    }

    /// A write that fails after serialization (here the temporary cannot be
    /// created) leaves the old list in place. Vanilla's order deleted the destination before
    /// the rename, so a crash between the two lost the list.
    #[test]
    fn a_failed_temporary_write_keeps_the_destination_and_no_temporary() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("servers.dat");
        let original = b"the player's servers";
        std::fs::write(&path, original).expect("old destination");
        // A folder in the temporary's place makes creating the file fail.
        let temporary = suffixed_path(&path, "_tmp");
        std::fs::create_dir_all(&temporary).expect("blocker");

        let Nbt::Compound(compound) = sample() else {
            unreachable!()
        };
        assert!(safe_write(&path, &compound).is_err());
        assert_eq!(std::fs::read(&path).expect("kept"), original);

        std::fs::remove_dir_all(&temporary).expect("unblock");
        safe_write(&path, &compound).expect("second write");
        assert!(!temporary.exists());
        let bytes = std::fs::read(&path).expect("replacement");
        let (_, value) = crate::read::from_slice_with_limits(&bytes, NbtLimits::UNLIMITED)
            .expect("replacement is raw NBT");
        assert_eq!(value, Nbt::Compound(compound));
    }
}
