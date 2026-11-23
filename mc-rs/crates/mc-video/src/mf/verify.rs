//! Decode a finished file with Windows' own Media Foundation decoders: the decoders the Media
//! Player and Films & TV apps use. The encode tests use it to show that exports (1000 and 2000 fps
//! ones included) play on Windows itself, not only in ffmpeg.

use windows::Win32::Media::MediaFoundation::{
    IMFSample, MFMediaType_Video, MFVideoFormat_NV12, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE,
    MF_SOURCE_READERF_ENDOFSTREAM, MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING,
    MF_SOURCE_READER_FIRST_VIDEO_STREAM,
};

use super::api::MfSession;
use crate::VideoError;

/// What decoding a file found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decoded {
    /// Video frames decoded.
    pub frames: u64,
    /// The last frame's timestamp, in 100 ns units.
    pub last_time_hns: i64,
}

/// Decode every video frame of `path` to NV12 with Windows' decoders (hardware when available).
///
/// # Errors
/// [`VideoError::Unavailable`] without Media Foundation, [`VideoError::MediaFoundation`] when the
/// file cannot be opened or a frame cannot be decoded.
pub fn decode_video(path: &std::path::Path) -> Result<Decoded, VideoError> {
    let session = MfSession::start()?;
    // Every Media Foundation object lives inside `decode` and is released before the session.
    decode(&session, path)
}

fn decode(session: &MfSession, path: &std::path::Path) -> Result<Decoded, VideoError> {
    let api = session.api();
    let mf = |context| move |e: windows::core::Error| VideoError::mf(context, &e);
    let attributes = api.attributes(1).map_err(mf("MFCreateAttributes"))?;
    // SAFETY: a live attribute store and a valid key.
    unsafe { attributes.SetUINT32(&MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1) }
        .map_err(mf("enabling video processing"))?;
    let reader = api
        .source_reader(&path.to_string_lossy(), &attributes)
        .map_err(mf("opening the file"))?;
    let stream = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;
    let nv12 = api.media_type().map_err(mf("MFCreateMediaType"))?;
    // SAFETY: live objects; the reader takes a reference to the media type.
    unsafe {
        nv12.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
            .map_err(mf("video type"))?;
        nv12.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)
            .map_err(mf("NV12 type"))?;
        reader
            .SetCurrentMediaType(stream, None, &nv12)
            .map_err(mf("asking for NV12"))?;
    }
    let mut decoded = Decoded {
        frames: 0,
        last_time_hns: 0,
    };
    loop {
        let mut flags = 0_u32;
        let mut time = 0_i64;
        let mut sample: Option<IMFSample> = None;
        // SAFETY: valid out-pointers; a synchronous reader returns one sample (or the end).
        unsafe {
            reader.ReadSample(
                stream,
                0,
                None,
                Some(&mut flags),
                Some(&mut time),
                Some(&mut sample),
            )
        }
        .map_err(mf("decoding a frame"))?;
        if sample.is_some() {
            decoded.frames += 1;
            decoded.last_time_hns = time;
        }
        if flags & (MF_SOURCE_READERF_ENDOFSTREAM.0 as u32) != 0 {
            break;
        }
    }
    Ok(decoded)
}
