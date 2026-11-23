//! "Fast start": move an MP4's index (`moov`) in front of its data (`mdat`), so a player can
//! start before the whole file has arrived (chat apps' and browsers' inline players).
//!
//! Media Foundation writes the index last. Its own `MF_MPEG4SINK_MOOV_BEFORE_MDAT` was measured
//! broken here: from a URL, `Finalize` fails with "Access is denied"; with a read/write byte stream
//! it writes the index first but leaves the chunk offsets pointing where the data used to be, so
//! every frame decodes as garbage. This module does the rewrite instead: copy the boxes before
//! `mdat`, then `moov` with every chunk offset (`stco`/`co64`) moved by the index's size, then the
//! rest. It never edits the source; a failure leaves the original, valid file.

use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Seconds from 1904-01-01 (the MP4 clock's epoch) to 1970-01-01.
const MP4_EPOCH_TO_UNIX: u64 = 2_082_844_800;

/// One top-level box.
#[derive(Debug, Clone, Copy)]
struct TopBox {
    kind: [u8; 4],
    offset: u64,
    size: u64,
}

fn read_top_level(file: &mut File, length: u64) -> io::Result<Vec<TopBox>> {
    let mut boxes = Vec::new();
    let mut offset = 0_u64;
    while offset + 8 <= length {
        file.seek(SeekFrom::Start(offset))?;
        let mut header = [0_u8; 8];
        file.read_exact(&mut header)?;
        let short = u64::from(u32::from_be_bytes([
            header[0], header[1], header[2], header[3],
        ]));
        let kind = [header[4], header[5], header[6], header[7]];
        let size = match short {
            0 => length - offset,
            1 => {
                let mut large = [0_u8; 8];
                file.read_exact(&mut large)?;
                u64::from_be_bytes(large)
            }
            n => n,
        };
        if size < 8 || offset + size > length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad MP4 box size",
            ));
        }
        boxes.push(TopBox { kind, offset, size });
        offset += size;
    }
    Ok(boxes)
}

/// Add `shift` to every chunk offset inside a `moov` box (its full bytes, header included).
/// Returns `false` when a 32-bit offset would overflow (the file would need `co64`).
fn shift_chunk_offsets(data: &mut [u8], shift: u64) -> bool {
    shift_in(data, 8, shift)
}

/// Walk the boxes in `data[start..]`, recursing into the containers on the way to `stbl`.
fn shift_in(data: &mut [u8], start: usize, shift: u64) -> bool {
    let mut at = start;
    while at + 8 <= data.len() {
        let size =
            u32::from_be_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]) as usize;
        let kind = [data[at + 4], data[at + 5], data[at + 6], data[at + 7]];
        let (header, size) = match size {
            0 => (8, data.len() - at),
            1 => {
                if at + 16 > data.len() {
                    return false;
                }
                let mut large = [0_u8; 8];
                large.copy_from_slice(&data[at + 8..at + 16]);
                (16, u64::from_be_bytes(large) as usize)
            }
            n => (8, n),
        };
        if size < header || at + size > data.len() {
            return false;
        }
        let body = at + header;
        let end = at + size;
        match &kind {
            b"trak" | b"mdia" | b"minf" | b"stbl" => {
                if !shift_in(&mut data[..end], body, shift) {
                    return false;
                }
            }
            b"stco" => {
                // version + flags, entry count, then 32-bit offsets.
                if body + 8 > end {
                    return false;
                }
                let count = u32::from_be_bytes([
                    data[body + 4],
                    data[body + 5],
                    data[body + 6],
                    data[body + 7],
                ]) as usize;
                if body + 8 + count * 4 > end {
                    return false;
                }
                for i in 0..count {
                    let p = body + 8 + i * 4;
                    let old = u64::from(u32::from_be_bytes([
                        data[p],
                        data[p + 1],
                        data[p + 2],
                        data[p + 3],
                    ]));
                    let Ok(new) = u32::try_from(old + shift) else {
                        return false;
                    };
                    data[p..p + 4].copy_from_slice(&new.to_be_bytes());
                }
            }
            b"co64" => {
                if body + 8 > end {
                    return false;
                }
                let count = u32::from_be_bytes([
                    data[body + 4],
                    data[body + 5],
                    data[body + 6],
                    data[body + 7],
                ]) as usize;
                if body + 8 + count * 8 > end {
                    return false;
                }
                for i in 0..count {
                    let p = body + 8 + i * 8;
                    let mut raw = [0_u8; 8];
                    raw.copy_from_slice(&data[p..p + 8]);
                    let new = u64::from_be_bytes(raw) + shift;
                    data[p..p + 8].copy_from_slice(&new.to_be_bytes());
                }
            }
            _ => {}
        }
        at = end;
    }
    true
}

/// `time` on the MP4 clock: whole seconds since 1904-01-01 UTC.
fn mp4_seconds(time: SystemTime) -> u64 {
    MP4_EPOCH_TO_UNIX + time.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// The box at `data[at..]`: its kind, header length and whole size. `None` when malformed.
fn box_header(data: &[u8], at: usize) -> Option<([u8; 4], usize, usize)> {
    let short = u32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?);
    let kind: [u8; 4] = data.get(at + 4..at + 8)?.try_into().ok()?;
    let (header, size) = match short {
        0 => (8, data.len() - at),
        1 => {
            let large = u64::from_be_bytes(data.get(at + 8..at + 16)?.try_into().ok()?);
            (16, usize::try_from(large).ok()?)
        }
        n => (8, n as usize),
    };
    (size >= header && at + size <= data.len()).then_some((kind, header, size))
}

/// Set the creation and modification time of the movie (`mvhd`), every track (`tkhd`) and every
/// track's media (`mdhd`) inside a `moov` (its full bytes, header included). `false` (with some
/// boxes possibly stamped) on a malformed box or a time a version-0 box cannot hold.
fn stamp_times(data: &mut [u8], seconds: u64) -> bool {
    stamp_in(data, 8, seconds)
}

fn stamp_in(data: &mut [u8], start: usize, seconds: u64) -> bool {
    let mut at = start;
    while at + 8 <= data.len() {
        let Some((kind, header, size)) = box_header(data, at) else {
            return false;
        };
        let body = at + header;
        let end = at + size;
        match &kind {
            b"trak" | b"mdia" => {
                if !stamp_in(&mut data[..end], body, seconds) {
                    return false;
                }
            }
            b"mvhd" | b"tkhd" | b"mdhd" => {
                // Version and flags, then the creation and modification times: 32-bit in
                // version 0, 64-bit in version 1.
                let wide = data.get(body) == Some(&1);
                let field = if wide { 8 } else { 4 };
                if body + 4 + 2 * field > end {
                    return false;
                }
                for slot in 0..2 {
                    let p = body + 4 + slot * field;
                    if wide {
                        data[p..p + 8].copy_from_slice(&seconds.to_be_bytes());
                    } else {
                        let Ok(short) = u32::try_from(seconds) else {
                            return false;
                        };
                        data[p..p + 4].copy_from_slice(&short.to_be_bytes());
                    }
                }
            }
            _ => {}
        }
        at = end;
    }
    true
}

fn copy_range(src: &mut File, dst: &mut impl Write, offset: u64, length: u64) -> io::Result<()> {
    src.seek(SeekFrom::Start(offset))?;
    let copied = io::copy(&mut src.take(length), dst)?;
    if copied == length {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "MP4 shorter than its boxes",
        ))
    }
}

/// Write `src` to `dst` with `moov` moved in front of `mdat`.
///
/// Returns `Ok(false)` (and writes nothing) when there is nothing to do or it cannot be done
/// safely: `moov` already first, no `mdat`, several `mdat`s, or 32-bit offsets that would
/// overflow. The caller then keeps `src` as it is, which is a valid file either way.
///
/// # Errors
/// File system errors, or a malformed `src`.
pub fn move_index_to_front(src: &Path, dst: &Path) -> io::Result<bool> {
    move_index_to_front_stamped(src, dst, None)
}

/// [`move_index_to_front`], also writing `created` as the creation and modification time of the
/// movie, its tracks and their media (Windows Explorer's "Media created"). Media Foundation writes
/// the moment the file was finished there, so two encodes of the same frames would otherwise
/// differ in those bytes. `None` keeps Media Foundation's times; so does an `Ok(false)`.
///
/// # Errors
/// File system errors, or a malformed `src`.
pub fn move_index_to_front_stamped(
    src: &Path,
    dst: &Path,
    created: Option<SystemTime>,
) -> io::Result<bool> {
    let mut file = File::open(src)?;
    let length = file.metadata()?.len();
    let boxes = read_top_level(&mut file, length)?;
    let mdats: Vec<&TopBox> = boxes.iter().filter(|b| &b.kind == b"mdat").collect();
    let Some(moov) = boxes.iter().find(|b| &b.kind == b"moov").copied() else {
        return Ok(false);
    };
    let [mdat] = mdats.as_slice() else {
        return Ok(false);
    };
    if moov.offset < mdat.offset {
        return Ok(false);
    }
    let moov_size = usize::try_from(moov.size)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "index too large"))?;
    let mut index = vec![0_u8; moov_size];
    file.seek(SeekFrom::Start(moov.offset))?;
    file.read_exact(&mut index)?;
    if !shift_chunk_offsets(&mut index, moov.size) {
        return Ok(false);
    }
    if let Some(created) = created {
        // On a copy: a box that cannot be stamped leaves every time as Media Foundation wrote it.
        let mut stamped = index.clone();
        if stamp_times(&mut stamped, mp4_seconds(created)) {
            index = stamped;
        }
    }
    let mut out = BufWriter::with_capacity(1 << 20, File::create(dst)?);
    // Everything before the data (ftyp), then the index, then every other box in order.
    copy_range(&mut file, &mut out, 0, mdat.offset)?;
    out.write_all(&index)?;
    for b in boxes
        .iter()
        .filter(|b| b.offset >= mdat.offset && &b.kind != b"moov")
    {
        copy_range(&mut file, &mut out, b.offset, b.size)?;
    }
    out.into_inner()
        .map_err(io::IntoInnerError::into_error)?
        .sync_all()?;
    Ok(true)
}

/// Whether `path`'s `moov` comes before its `mdat` (for tests and checks).
///
/// # Errors
/// File system errors, or a malformed file.
pub fn index_is_first(path: &Path) -> io::Result<bool> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    let boxes = read_top_level(&mut file, length)?;
    let position = |kind: &[u8; 4]| boxes.iter().position(|b| &b.kind == kind);
    Ok(matches!((position(b"moov"), position(b"mdat")), (Some(m), Some(d)) if m < d))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    fn full(kind: &[u8; 4], entries: &[u8], count: u32) -> Vec<u8> {
        let mut body = vec![0_u8; 4];
        body.extend_from_slice(&count.to_be_bytes());
        body.extend_from_slice(entries);
        boxed(kind, &body)
    }

    /// ftyp, mdat (two chunks: "AAAA" at 24+8 and "BBBB" after it), moov with an stco pointing
    /// at both chunks and a co64 pointing at the second.
    fn sample_file() -> (Vec<u8>, u64, u64) {
        let ftyp = boxed(b"ftyp", b"isom\0\0\0\x01isomavc1");
        let mdat = boxed(b"mdat", b"AAAABBBB");
        let first = (ftyp.len() + 8) as u32;
        let second = first + 4;
        let mut stco_entries = first.to_be_bytes().to_vec();
        stco_entries.extend_from_slice(&second.to_be_bytes());
        let stco = full(b"stco", &stco_entries, 2);
        let co64 = full(b"co64", &u64::from(second).to_be_bytes(), 1);
        let stbl = boxed(b"stbl", &[stco, co64].concat());
        let minf = boxed(b"minf", &stbl);
        let mdia = boxed(b"mdia", &minf);
        let trak = boxed(b"trak", &mdia);
        let mvhd = full(b"mvhd", &[0; 8], 0);
        let moov = boxed(b"moov", &[mvhd, trak].concat());
        (
            [ftyp, mdat, moov].concat(),
            u64::from(first),
            u64::from(second),
        )
    }

    fn offsets_in(file: &[u8], table: &[u8; 4], wide: bool) -> Vec<u64> {
        let at = file.windows(4).position(|w| w == table).unwrap() + 4;
        let count = u32::from_be_bytes(file[at + 4..at + 8].try_into().unwrap()) as usize;
        (0..count)
            .map(|i| {
                if wide {
                    u64::from_be_bytes(file[at + 8 + i * 8..at + 16 + i * 8].try_into().unwrap())
                } else {
                    u64::from(u32::from_be_bytes(
                        file[at + 8 + i * 4..at + 12 + i * 4].try_into().unwrap(),
                    ))
                }
            })
            .collect()
    }

    #[test]
    fn index_moves_first_and_offsets_still_point_at_the_data() {
        let dir = std::env::temp_dir().join("mc-video-faststart-test");
        std::fs::create_dir_all(&dir).unwrap();
        let (bytes, first, second) = sample_file();
        let src = dir.join("in.mp4");
        let dst = dir.join("out.mp4");
        std::fs::write(&src, &bytes).unwrap();
        assert!(!index_is_first(&src).unwrap());
        assert!(move_index_to_front(&src, &dst).unwrap());
        assert!(index_is_first(&dst).unwrap());
        let out = std::fs::read(&dst).unwrap();
        assert_eq!(out.len(), bytes.len());
        let stco = offsets_in(&out, b"stco", false);
        let co64 = offsets_in(&out, b"co64", true);
        assert_ne!(stco, vec![first, second], "offsets moved");
        assert_eq!(&out[stco[0] as usize..stco[0] as usize + 4], b"AAAA");
        assert_eq!(&out[stco[1] as usize..stco[1] as usize + 4], b"BBBB");
        assert_eq!(&out[co64[0] as usize..co64[0] as usize + 4], b"BBBB");
        // Already fast-start: nothing to do.
        let again = dir.join("again.mp4");
        assert!(!move_index_to_front(&dst, &again).unwrap());
        assert!(!again.exists());
    }

    #[test]
    fn stamping_sets_every_creation_and_modification_time() {
        // mvhd and mdhd in version 0 (32-bit times), tkhd in version 1 (64-bit), old times 7..10.
        let mut v0 = vec![0_u8; 4];
        v0.extend_from_slice(&7_u32.to_be_bytes());
        v0.extend_from_slice(&8_u32.to_be_bytes());
        v0.extend_from_slice(&[0xAB; 8]);
        let mut v1 = vec![1_u8, 0, 0, 0];
        v1.extend_from_slice(&9_u64.to_be_bytes());
        v1.extend_from_slice(&10_u64.to_be_bytes());
        v1.extend_from_slice(&[0xCD; 8]);
        let mdia = boxed(b"mdia", &boxed(b"mdhd", &v0));
        let trak = boxed(b"trak", &[boxed(b"tkhd", &v1), mdia].concat());
        let mut moov = boxed(b"moov", &[boxed(b"mvhd", &v0), trak].concat());
        // 2026-11-12, about.
        let when = UNIX_EPOCH + std::time::Duration::from_secs(1_794_000_000);
        let seconds = mp4_seconds(when);
        assert_eq!(seconds, 1_794_000_000 + MP4_EPOCH_TO_UNIX);
        assert!(stamp_times(&mut moov, seconds));
        let body = |kind: &[u8; 4]| moov.windows(4).position(|w| w == kind).unwrap() + 4;
        let short = u32::try_from(seconds).unwrap().to_be_bytes();
        for kind in [b"mvhd", b"mdhd"] {
            let at = body(kind);
            assert_eq!(moov[at + 4..at + 8], short);
            assert_eq!(moov[at + 8..at + 12], short);
            assert_eq!(moov[at + 12..at + 20], [0xAB; 8], "the rest is untouched");
        }
        let at = body(b"tkhd");
        assert_eq!(moov[at + 4..at + 12], seconds.to_be_bytes());
        assert_eq!(moov[at + 12..at + 20], seconds.to_be_bytes());
        assert_eq!(moov[at + 20..at + 28], [0xCD; 8]);
    }

    #[test]
    fn two_encodes_finished_at_different_times_come_out_identical() {
        let dir = std::env::temp_dir().join("mc-video-faststart-stamp-test");
        std::fs::create_dir_all(&dir).unwrap();
        let (bytes, _, _) = sample_file();
        // The second file was finished 5 s later: only its movie header's time differs.
        let mvhd = bytes.windows(4).position(|w| w == b"mvhd").unwrap() + 4;
        let mut later = bytes.clone();
        later[mvhd + 4..mvhd + 8].copy_from_slice(&5_u32.to_be_bytes());
        let when = UNIX_EPOCH + std::time::Duration::from_secs(1_794_000_000);
        let encode = |name: &str, file: &[u8], created: Option<SystemTime>| {
            let src = dir.join(format!("{name}.mp4"));
            let dst = dir.join(format!("{name}-out.mp4"));
            std::fs::write(&src, file).unwrap();
            assert!(move_index_to_front_stamped(&src, &dst, created).unwrap());
            std::fs::read(&dst).unwrap()
        };
        assert_eq!(
            encode("a", &bytes, Some(when)),
            encode("b", &later, Some(when))
        );
        // Without a time, Media Foundation's stay and the files differ.
        assert_ne!(encode("c", &bytes, None), encode("d", &later, None));
    }
}
