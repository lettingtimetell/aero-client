//! Cutting a clip out of a finished recording.
//!
//! The clip key cuts from the in-memory session buffer instead ([`crate::Writer::save_clip`]);
//! both end in the same file shape: a state prefix at time 0, then the blocks from a world reset
//! on (copied still compressed, only their headers retimed), and the window to show in the
//! metadata.

use std::path::Path;

use crate::reader::{Reader, RecordRef};
use crate::writer::{write_assembled, Finished, Packer};
use crate::{Record, ReplayError, ReplayKind, ReplayTime, WriterOptions};

/// What [`cut_file`] decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipCut {
    /// The file written.
    pub finished: Finished,
    /// The block the clip's world starts at (in the source).
    pub base_block: usize,
    /// The source time that became 0 in the clip.
    pub base_time: ReplayTime,
    /// Records kept from before the base as the state prefix.
    pub prefix_records: usize,
}

/// Cut `window` (`start`, `end`, in `reader`'s clock) out of a recording into a new file `out`.
///
/// * The clip's world starts at the last world reset at or before `start` (the first block if
///   there is none).
/// * Records before it are kept, as a state prefix at time 0, when `keep_before` says so: the
///   client keeps what a world reset does not clear (the tab list, the scoreboard, the first
///   `JoinGame`). Every record of a block flagged as a prefix is offered to it too.
/// * Then the blocks from the reset to the one holding `end`, copied without recompression.
///
/// # Errors
///
/// The source is damaged before `end`, or `out` cannot be written (it must not exist).
pub fn cut_file(
    reader: &Reader,
    window: (ReplayTime, ReplayTime),
    out: &Path,
    mut keep_before: impl FnMut(&RecordRef<'_>) -> bool,
) -> Result<ClipCut, ReplayError> {
    let blocks = reader.blocks();
    let (start, end) = (window.0.min(window.1), window.0.max(window.1));
    let base_block = reader.keyframe_before(start).unwrap_or(0);
    let base_time = blocks.get(base_block).map_or(0, |b| b.first_time);
    // The last block with a record at or before `end`.
    let last_block = blocks
        .partition_point(|b| b.first_time <= end)
        .saturating_sub(1)
        .max(base_block);

    // The records before the base are exactly the first `before` records of the file.
    let before: u64 = blocks[..base_block.min(blocks.len())]
        .iter()
        .map(|b| u64::from(b.records))
        .sum();
    let mut prefix = Vec::new();
    let mut cursor = reader.cursor();
    for _ in 0..before {
        let Some(record) = cursor.next()? else {
            break;
        };
        if keep_before(&record) {
            prefix.push(Record {
                time: 0,
                stream: record.stream,
                payload: record.payload.to_vec(),
            });
        }
    }

    let mut copied = Vec::new();
    if !blocks.is_empty() {
        for index in base_block..=last_block.min(blocks.len() - 1) {
            copied.push(reader.block_bytes(index)?);
        }
    }
    let last_time = copied
        .iter()
        .map(|(header, _)| header.last_time)
        .max()
        .unwrap_or(base_time);

    let mut meta = reader.meta().clone();
    let clip_end = end.clamp(base_time, last_time.max(base_time)) - base_time;
    let clip_start = start.max(base_time).min(end.max(base_time)) - base_time;
    meta.kind = ReplayKind::Clip {
        start: clip_start.min(clip_end),
        end: clip_end,
    };
    meta.view_start = clip_start.min(clip_end);
    meta.duration = last_time - base_time;
    meta.world_resets = meta
        .world_resets
        .iter()
        .filter(|&&t| t >= base_time && t <= last_time)
        .map(|&t| t - base_time)
        .collect();
    meta.markers
        .retain(|m| m.time >= base_time && m.time <= last_time);
    for marker in &mut meta.markers {
        marker.time -= base_time;
    }

    let mut packer = Packer::new(WriterOptions::default().level);
    let prefix_records = prefix.len();
    let finished = write_assembled(
        out,
        &meta,
        &mut packer,
        &prefix,
        copied.iter().map(|(header, payload)| (*header, *payload)),
        base_time,
        &[],
    )?;
    Ok(ClipCut {
        finished,
        base_block,
        base_time,
        prefix_records,
    })
}
