//! Several audio tracks in one MP4.
//!
//! An export can carry one audio track per sound channel (game, microphone, desktop audio) after
//! the normal mixed one, for video editors. Players must still play the mix and only the mix:
//!
//! * the writer puts a copy of the mix after the channel tracks
//!   (`Mix, Game sound, Microphone, Desktop, Mix`): measured on Windows 11, Windows' playback
//!   stack (`MediaPlaybackItem`, what the Media Player app uses) and Media Foundation's source
//!   reader both play a file's *last* audio track, whatever the flags below say, while browsers
//!   (and the chat apps built on them), ffmpeg-based players and editors take the first or the
//!   default one;
//! * every audio track joins one alternate group (ISO/IEC 14496-12 `tkhd.alternate_group`), the
//!   way the languages of one film do, so a player picks one of them;
//! * only the first audio track (the mix) keeps `track_enabled`; the others are disabled. That is
//!   what ffmpeg writes for tracks that are not the default, and what players read that way;
//! * each audio track's handler name (`hdlr`) says what it holds ("Mix", "Game sound",
//!   "Microphone", "Desktop"), written over Media Foundation's "SoundHandler", so the index keeps
//!   its size and no chunk offset moves (a longer name is cut to fit).
//!
//! Media Foundation writes every track enabled and in no group, so [`label_audio_tracks`] edits
//! the finished file's `moov` in place.

use std::fs::OpenOptions;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// The largest index this reads into memory (a 2000 fps hour is far below it).
const MAX_MOOV: u64 = 512 << 20;

/// The `tkhd` flag that makes a track play by default.
const TRACK_ENABLED: u32 = 1;

/// The alternate group all audio tracks join.
const AUDIO_GROUP: u16 = 1;

/// Label the audio tracks of the MP4 at `path`, in file order, with `names` (missing names keep
/// what is there); only the first stays enabled, and all join one alternate group. `Ok(false)`
/// (nothing written) when the file has no `moov` or no audio track, or its boxes are malformed.
///
/// # Errors
/// File system errors.
pub fn label_audio_tracks(path: &Path, names: &[&str]) -> io::Result<bool> {
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    let length = file.metadata()?.len();
    let mut at = 0_u64;
    let mut moov = None;
    while at + 8 <= length {
        file.seek(SeekFrom::Start(at))?;
        let mut header = [0_u8; 16];
        let wanted = (length - at).min(16) as usize;
        file.read_exact(&mut header[..wanted])?;
        let short = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        let size = match short {
            0 => length - at,
            1 if wanted == 16 => u64::from_be_bytes([
                header[8], header[9], header[10], header[11], header[12], header[13], header[14],
                header[15],
            ]),
            1 => return Ok(false),
            n => u64::from(n),
        };
        if size < 8 || at + size > length {
            return Ok(false);
        }
        if &header[4..8] == b"moov" {
            moov = Some((at, size));
            break;
        }
        at += size;
    }
    let Some((offset, size)) = moov.filter(|&(_, size)| size <= MAX_MOOV) else {
        return Ok(false);
    };
    let mut data = vec![0_u8; size as usize];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut data)?;
    if !label_in_moov(&mut data, names) {
        return Ok(false);
    }
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(&data)?;
    file.sync_all()?;
    Ok(true)
}

/// The box at `data[at..end]`: its kind, header length and whole size. `None` when malformed.
fn box_at(data: &[u8], at: usize, end: usize) -> Option<([u8; 4], usize, usize)> {
    let short = u32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?);
    let kind: [u8; 4] = data.get(at + 4..at + 8)?.try_into().ok()?;
    let (header, size) = match short {
        0 => (8, end - at),
        1 => {
            let large = u64::from_be_bytes(data.get(at + 8..at + 16)?.try_into().ok()?);
            (16, usize::try_from(large).ok()?)
        }
        n => (8, n as usize),
    };
    (size >= header && at + size <= end).then_some((kind, header, size))
}

/// The body range of the first child box of kind `kind` within `data[start..end]`.
fn child(data: &[u8], start: usize, end: usize, kind: &[u8; 4]) -> Option<(usize, usize)> {
    let mut at = start;
    while at + 8 <= end {
        let (found, header, size) = box_at(data, at, end)?;
        if &found == kind {
            return Some((at + header, at + size));
        }
        at += size;
    }
    None
}

/// Edit a `moov` box (its full bytes, header included); `false` when it has no audio track or a
/// box is malformed (then nothing is changed).
pub fn label_in_moov(moov: &mut [u8], names: &[&str]) -> bool {
    let Some((kind, header, size)) = box_at(moov, 0, moov.len()) else {
        return false;
    };
    if &kind != b"moov" {
        return false;
    }
    // First find every audio track's boxes, then edit (nothing changes on a malformed index).
    struct Audio {
        tkhd: (usize, usize),
        hdlr: (usize, usize),
    }
    let mut audio = Vec::new();
    let mut at = header;
    while at + 8 <= size {
        let Some((kind, header, box_size)) = box_at(moov, at, size) else {
            return false;
        };
        if &kind == b"trak" {
            let (body, end) = (at + header, at + box_size);
            let found = (|| {
                let tkhd = child(moov, body, end, b"tkhd")?;
                let (mdia, mdia_end) = child(moov, body, end, b"mdia")?;
                let hdlr = child(moov, mdia, mdia_end, b"hdlr")?;
                Some((tkhd, hdlr))
            })();
            if let Some((tkhd, hdlr)) = found {
                // hdlr: version and flags, pre_defined, handler_type, 12 reserved, then the name.
                if moov.get(hdlr.0 + 8..hdlr.0 + 12) == Some(b"soun") {
                    let wide = moov.get(tkhd.0) == Some(&1);
                    let group_at = tkhd.0 + if wide { 46 } else { 34 };
                    if group_at + 2 > tkhd.1 || hdlr.0 + 24 > hdlr.1 {
                        return false;
                    }
                    audio.push(Audio { tkhd, hdlr });
                }
            }
        }
        at += box_size;
    }
    if audio.is_empty() {
        return false;
    }
    for (index, track) in audio.iter().enumerate() {
        let body = track.tkhd.0;
        let flags = u32::from_be_bytes([0, moov[body + 1], moov[body + 2], moov[body + 3]]);
        let flags = if index == 0 {
            flags | TRACK_ENABLED
        } else {
            flags & !TRACK_ENABLED
        };
        moov[body + 1..body + 4].copy_from_slice(&flags.to_be_bytes()[1..]);
        let group_at = body + if moov[body] == 1 { 46 } else { 34 };
        moov[group_at..group_at + 2].copy_from_slice(&AUDIO_GROUP.to_be_bytes());
        if let Some(name) = names.get(index) {
            let slot = &mut moov[track.hdlr.0 + 24..track.hdlr.1];
            // Keep the terminating zero: cut the name to the room Media Foundation left.
            let room = slot.len().saturating_sub(1);
            let bytes = name.as_bytes();
            let cut = bytes.len().min(room);
            slot.fill(0);
            slot[..cut].copy_from_slice(&bytes[..cut]);
        }
    }
    true
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

    /// A track as Media Foundation writes it: tkhd version `version` with flags 1 and group 0,
    /// an mdia with an hdlr of `handler` named "SoundHandler"/"VideoHandler".
    fn trak(version: u8, handler: &[u8; 4]) -> Vec<u8> {
        let fixed = if version == 1 { 92 } else { 80 };
        let mut tkhd = vec![0_u8; fixed];
        tkhd[0] = version;
        tkhd[3] = 1;
        let mut hdlr = vec![0_u8; 24];
        hdlr[8..12].copy_from_slice(handler);
        let name: &[u8] = if handler == b"soun" {
            b"SoundHandler\0"
        } else {
            b"VideoHandler\0"
        };
        hdlr.extend_from_slice(name);
        let mdia = boxed(
            b"mdia",
            &[boxed(b"mdhd", &[0; 24]), boxed(b"hdlr", &hdlr)].concat(),
        );
        boxed(b"trak", &[boxed(b"tkhd", &tkhd), mdia].concat())
    }

    fn tkhd_of(moov: &[u8], track: usize) -> (u32, u16) {
        let mut at = 8;
        let mut n = 0;
        loop {
            let (kind, header, size) = box_at(moov, at, moov.len()).expect("box");
            if &kind == b"trak" {
                if n == track {
                    let (body, _) = child(moov, at + header, at + size, b"tkhd").expect("tkhd");
                    let bytes = [0, moov[body + 1], moov[body + 2], moov[body + 3]];
                    let flags = u32::from_be_bytes(bytes);
                    let group_at = body + if moov[body] == 1 { 46 } else { 34 };
                    return (
                        flags,
                        u16::from_be_bytes([moov[group_at], moov[group_at + 1]]),
                    );
                }
                n += 1;
            }
            at += size;
        }
    }

    #[test]
    fn audio_tracks_get_one_group_names_and_only_the_mix_stays_enabled() {
        let mut moov = boxed(
            b"moov",
            &[
                boxed(b"mvhd", &[0; 100]),
                trak(0, b"vide"),
                trak(0, b"soun"),
                trak(1, b"soun"),
                trak(0, b"soun"),
            ]
            .concat(),
        );
        let before = moov.clone();
        assert!(label_in_moov(
            &mut moov,
            &["Mix", "Game sound", "A name far too long to fit"]
        ));
        assert_eq!(moov.len(), before.len(), "the index keeps its size");
        assert_eq!(tkhd_of(&moov, 0), (1, 0), "the video track is untouched");
        assert_eq!(tkhd_of(&moov, 1), (1, 1), "the mix: enabled, group 1");
        assert_eq!(
            tkhd_of(&moov, 2),
            (0, 1),
            "version 1 tkhd: disabled, group 1"
        );
        assert_eq!(tkhd_of(&moov, 3), (0, 1));
        let text = String::from_utf8_lossy(&moov).into_owned();
        assert!(text.contains("VideoHandler"));
        assert!(text.contains("Mix\0"));
        assert!(text.contains("Game sound\0"));
        assert!(
            text.contains("A name far t\0"),
            "cut to the 12 bytes there were"
        );
        assert_eq!(text.matches("SoundHandler").count(), 0);
    }

    /// Negative control: no audio track, or a damaged box, changes nothing.
    #[test]
    fn an_index_without_audio_or_with_a_damaged_box_is_left_alone() {
        let parts = [boxed(b"mvhd", &[0; 100]), trak(0, b"vide")];
        let mut video_only = boxed(b"moov", &parts.concat());
        let before = video_only.clone();
        assert!(!label_in_moov(&mut video_only, &["Mix"]));
        assert_eq!(video_only, before);
        // The second track's size runs past the index: nothing is edited, not even the first.
        let first = trak(0, b"soun");
        let mut damaged = boxed(b"moov", &[first.clone(), trak(0, b"soun")].concat());
        let second = 8 + first.len();
        damaged[second..second + 4].copy_from_slice(&0x7FFF_FFFF_u32.to_be_bytes());
        let before = damaged.clone();
        assert!(!label_in_moov(&mut damaged, &["Mix", "Game sound"]));
        assert_eq!(damaged, before);
    }
}
