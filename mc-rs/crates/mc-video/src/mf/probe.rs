//! Which encoders this PC has, for the export dialog ("H.265 (GPU)" or greyed out).

use std::sync::OnceLock;

use windows::core::GUID;
use windows::Win32::Media::MediaFoundation::{
    IMFActivate, MFMediaType_Video, MFT_ENUM_HARDWARE_URL_Attribute,
    MFT_ENUM_HARDWARE_VENDOR_ID_Attribute, MFT_FRIENDLY_NAME_Attribute, MFVideoFormat_H264,
    MFVideoFormat_HEVC, MFT_CATEGORY_VIDEO_ENCODER, MFT_ENUM_FLAG_ASYNCMFT, MFT_ENUM_FLAG_HARDWARE,
    MFT_ENUM_FLAG_LOCALMFT, MFT_ENUM_FLAG_SORTANDFILTER, MFT_ENUM_FLAG_SYNCMFT,
    MFT_REGISTER_TYPE_INFO,
};

use super::api::MfSession;
use crate::VideoCodec;

/// One encoder Media Foundation offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderEntry {
    /// What it encodes.
    pub codec: VideoCodec,
    /// Runs on the GPU.
    pub hardware: bool,
    /// Its friendly name.
    pub name: String,
}

static FOUND: OnceLock<Vec<EncoderEntry>> = OnceLock::new();

/// Whether Media Foundation can be started at all on this PC.
#[must_use]
pub fn media_foundation_available() -> bool {
    super::api::api().is_ok()
}

/// Every H.264 and H.265 encoder, in the order Media Foundation prefers them (hardware first).
/// Asked once per run on a short-lived thread of its own (so the caller's COM apartment does not
/// matter), then cached. Empty without Media Foundation.
#[must_use]
pub fn encoders() -> Vec<EncoderEntry> {
    FOUND
        .get_or_init(|| {
            // A panic in the probe means "no encoders" (the join below), not the game's crash.
            std::thread::Builder::new()
                .name("mc-video probe".to_owned())
                .spawn(|| mc_core::panic_scope::contain_for_join(enumerate))
                .ok()
                .and_then(|thread| thread.join().ok())
                .unwrap_or_default()
        })
        .clone()
}

/// Whether `codec` can be exported here, and whether on the GPU.
#[must_use]
pub fn codec_support(codec: VideoCodec) -> Option<bool> {
    let all = encoders();
    let mut entries = all.iter().filter(|e| e.codec == codec).peekable();
    entries.peek()?;
    Some(entries.any(|e| e.hardware))
}

fn enumerate() -> Vec<EncoderEntry> {
    let Ok(session) = MfSession::start() else {
        return Vec::new();
    };
    let mut found = Vec::new();
    // Which vendor each hardware encoder belongs to, for the log: a PC with two
    // graphics adapters lists encoders of both. (The list carries no adapter LUIDs: measured.)
    let mut vendors = Vec::new();
    for (codec, subtype) in [
        (VideoCodec::H264, MFVideoFormat_H264),
        (VideoCodec::H265, MFVideoFormat_HEVC),
    ] {
        let flags = (MFT_ENUM_FLAG_HARDWARE.0
            | MFT_ENUM_FLAG_SYNCMFT.0
            | MFT_ENUM_FLAG_ASYNCMFT.0
            | MFT_ENUM_FLAG_LOCALMFT.0
            | MFT_ENUM_FLAG_SORTANDFILTER.0) as u32;
        let output = MFT_REGISTER_TYPE_INFO {
            guidMajorType: MFMediaType_Video,
            guidSubtype: subtype,
        };
        let Ok(activates) =
            session
                .api()
                .enum_transforms(MFT_CATEGORY_VIDEO_ENCODER, flags, None, Some(output))
        else {
            continue;
        };
        for activate in activates {
            let entry = EncoderEntry {
                codec,
                hardware: has_attribute(&activate, &MFT_ENUM_HARDWARE_URL_Attribute),
                name: string_attribute(&activate, &MFT_FRIENDLY_NAME_Attribute)
                    .unwrap_or_else(|| "unnamed encoder".to_owned()),
            };
            if entry.hardware {
                let vendor = string_attribute(&activate, &MFT_ENUM_HARDWARE_VENDOR_ID_Attribute);
                vendors.push(format!(
                    "{} {}: {}",
                    codec.label(),
                    entry.name,
                    vendor.as_deref().unwrap_or("no vendor"),
                ));
            }
            found.push(entry);
        }
    }
    drop(session);
    tracing::info!(?found, ?vendors, "video encoders");
    found
}

fn has_attribute(activate: &IMFActivate, key: &GUID) -> bool {
    // SAFETY: a live attribute store and a valid key.
    unsafe { activate.GetItem(key, None) }.is_ok()
}

/// A string attribute of any attribute store (an activation object, an MFT's attributes).
pub(crate) fn string_attribute(
    store: &windows::Win32::Media::MediaFoundation::IMFAttributes,
    key: &GUID,
) -> Option<String> {
    // SAFETY: a live attribute store; the buffer is sized from GetStringLength plus the NUL.
    unsafe {
        let length = store.GetStringLength(key).ok()?;
        let mut buffer = vec![0_u16; length as usize + 1];
        store.GetString(key, &mut buffer, None).ok()?;
        Some(String::from_utf16_lossy(&buffer[..length as usize]))
    }
}
