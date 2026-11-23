//! The Media Foundation sink writer: NV12 frames and 16-bit PCM in, H.264/H.265 + AAC in MP4 out.
//!
//! One [`Mp4Writer`] lives on one thread (it owns a COM/Media Foundation session for it). The file
//! is written under `<name>.partial` and renamed when [`Mp4Writer::finish`] completes, so a
//! cancelled, failed or crashed encode never leaves a broken `.mp4`: dropping the writer without
//! finishing deletes the partial file.
//!
//! # High frame rates
//!
//! Frame times are exact (see [`FrameRate::frame_start_hns`]) and the media types carry the real
//! rate, so 1000 and 2000 fps files come out with that rate in their headers. What a codec level
//! allows is the encoder's business: the writer leaves `MF_MT_MPEG2_LEVEL` unset so each encoder
//! picks the level its own limits need (see the RD report for what was measured on NVIDIA and
//! Microsoft's encoders). One exception: H.264 rates over level 5.2's
//! macroblock rate are never offered to a hardware encoder, because AMD's accepts them and runs
//! outside the level it writes (NVIDIA's refuses them, so nothing changes there).
//!
//! # The encoder's adapter
//!
//! With [`EncoderConfig::adapter`] set, the hardware encoder runs on that graphics adapter, on a
//! Direct3D 11 device made for it (`device.rs`), instead of wherever Media Foundation's list puts
//! it.

use std::path::PathBuf;

use windows::core::{Interface, GUID};
use windows::Win32::Media::MediaFoundation::{
    eAVEncCommonRateControlMode_CBR, eAVEncCommonRateControlMode_PeakConstrainedVBR,
    eAVEncCommonRateControlMode_Quality, eAVEncH264VProfile_High, eAVEncH265VProfile_Main_420_8,
    CODECAPI_AVEncCommonBufferSize, CODECAPI_AVEncCommonMaxBitRate,
    CODECAPI_AVEncCommonMeanBitRate, CODECAPI_AVEncCommonQuality,
    CODECAPI_AVEncCommonQualityVsSpeed, CODECAPI_AVEncCommonRateControlMode,
    CODECAPI_AVEncMPVGOPSize, IMFAttributes, IMFMediaType, IMFSinkWriter, IMFSinkWriterEx,
    IMFTransform, MFAudioFormat_AAC, MFAudioFormat_PCM, MFMediaType_Audio, MFMediaType_Video,
    MFNominalRange_16_235, MFT_ENUM_HARDWARE_URL_Attribute, MFT_ENUM_HARDWARE_VENDOR_ID_Attribute,
    MFT_FRIENDLY_NAME_Attribute, MFTranscodeContainerType_MPEG4, MFVideoFormat_H264,
    MFVideoFormat_HEVC, MFVideoFormat_NV12, MFVideoInterlace_Progressive, MFVideoPrimaries_BT709,
    MFVideoTransFunc_709, MFVideoTransferMatrix_BT709, MFT_CATEGORY_VIDEO_ENCODER,
    MFT_ENUM_ADAPTER_LUID, MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, MF_MT_AAC_PAYLOAD_TYPE,
    MF_MT_ALL_SAMPLES_INDEPENDENT, MF_MT_AUDIO_AVG_BYTES_PER_SECOND, MF_MT_AUDIO_BITS_PER_SAMPLE,
    MF_MT_AUDIO_BLOCK_ALIGNMENT, MF_MT_AUDIO_NUM_CHANNELS, MF_MT_AUDIO_SAMPLES_PER_SECOND,
    MF_MT_AVG_BITRATE, MF_MT_DEFAULT_STRIDE, MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE,
    MF_MT_INTERLACE_MODE, MF_MT_MAJOR_TYPE, MF_MT_MPEG2_PROFILE, MF_MT_PIXEL_ASPECT_RATIO,
    MF_MT_SUBTYPE, MF_MT_TRANSFER_FUNCTION, MF_MT_VIDEO_NOMINAL_RANGE, MF_MT_VIDEO_PRIMARIES,
    MF_MT_YUV_MATRIX, MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, MF_SINK_WRITER_D3D_MANAGER,
    MF_TRANSCODE_CONTAINERTYPE,
};

use super::api::MfSession;
use super::device::EncoderDevice;
use super::probe::string_attribute;
use crate::config::HNS_PER_SECOND;
use crate::{
    EncodeSummary, EncoderConfig, EncoderInfo, EncoderPreference, FrameRate, RateControl,
    VideoCodec, VideoError,
};

/// Why opening failed: the video encoder refused the settings (worth retrying at a lower nominal
/// rate), or anything else.
enum Open {
    Encoder(VideoError),
    Other(VideoError),
}

impl From<VideoError> for Open {
    fn from(error: VideoError) -> Self {
        Self::Other(error)
    }
}

/// How many encoder settings to hand over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Params {
    /// Everything: rate control, buffer size, keyframe interval, quality versus speed.
    Full,
    /// Only the rate control mode, its bitrate or quality, and the keyframe interval.
    Essential,
}

/// The real rate, then the real rate divided by each divisor of 1000 in turn (down to about
/// 20 fps).
///
/// Why divisors of 1000: Media Foundation's MP4 sink gives the track a timescale of 1000 x the
/// *nominal* rate (measured: nominal 500 -> 1/500000, nominal 62.5 -> 1/62500), so a frame of
/// the real rate lasts `1000 / k` ticks at nominal `real / k`. Only a divisor of 1000 keeps every
/// frame the same exact length; plain halving reached k = 16 at 4K 1000 fps, where 1 ms is 62.5
/// ticks and frames alternated 63/62 (ffprobe then guessed 125 fps).
fn nominal_rates(real: FrameRate) -> Vec<FrameRate> {
    const DIVISORS: [u32; 15] = [1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125, 200, 250, 500];
    DIVISORS
        .iter()
        .filter_map(|&k| {
            real.den
                .checked_mul(k)
                .map(|den| FrameRate::new(real.num, den))
        })
        .filter(|rate| rate.as_f64() >= 20.0 || rate.den == real.den)
        .collect()
}

/// H.264 level 5.2's macroblock rate (ITU-T H.264 Table A-1, MaxMBPS): the
/// highest level Media Foundation's H.264 interface names (`eAVEncH264VLevel5_2`).
const H264_LEVEL_5_2_MAX_MBPS: u64 = 2_073_600;

/// `MC_W19_R2_LEVEL=0` (tests only, read once) offers every rate as before.
fn level_guard_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MC_W19_R2_LEVEL").map_or(true, |v| v.trim() != "0"))
}

/// Whether `nominal` frames a second at `width` x `height` stay within H.264
/// level 5.2's macroblock rate. Measured on this PC's encoders: NVIDIA's takes 1080p at 254 fps
/// and refuses 255, takes 720p at 576 and refuses 577 (exactly this limit); AMD's takes both and
/// writes level 5.2 into the stream all the same.
fn fits_h264_level_5_2(width: u32, height: u32, nominal: FrameRate) -> bool {
    let macroblocks = u128::from(width.div_ceil(16)) * u128::from(height.div_ceil(16));
    // macroblocks x num / den <= max, in integers.
    macroblocks * u128::from(nominal.num)
        <= u128::from(H264_LEVEL_5_2_MAX_MBPS) * u128::from(nominal.den.max(1))
}

/// A bitrate meant for the real rate, as the encoder must be told it at the nominal rate: it
/// spends `bitrate / nominal` bits per frame, and the frames really come `real` per second.
fn scaled(bitrate: u32, config: &EncoderConfig, nominal: FrameRate) -> u32 {
    let factor = nominal.as_f64() / config.frame_rate.as_f64();
    (f64::from(bitrate) * factor)
        .round()
        .clamp(1_000.0, f64::from(u32::MAX)) as u32
}

/// `MF_E_TOPO_CODEC_NOT_FOUND`/`MF_E_TRANSFORM_NOT_POSSIBLE_FOR_CURRENT_MEDIATYPE_COMBINATION` and
/// friends: what a missing codec looks like.
const MF_E_TOPO_CODEC_NOT_FOUND: u32 = 0xC00D_5212;
const MF_E_INVALIDMEDIATYPE: u32 = 0xC00D_36B4;

fn pack(high: u32, low: u32) -> u64 {
    (u64::from(high) << 32) | u64::from(low)
}

/// Shorthand for the attribute setters, each wrapped with what failed.
trait Attrs {
    fn u32(&self, key: &GUID, value: u32, context: &'static str) -> Result<(), VideoError>;
    fn u64(&self, key: &GUID, value: u64, context: &'static str) -> Result<(), VideoError>;
    fn guid(&self, key: &GUID, value: &GUID, context: &'static str) -> Result<(), VideoError>;
}

impl Attrs for IMFAttributes {
    fn u32(&self, key: &GUID, value: u32, context: &'static str) -> Result<(), VideoError> {
        // SAFETY: a live attribute store and a valid key.
        unsafe { self.SetUINT32(key, value) }.map_err(|e| VideoError::mf(context, &e))
    }
    fn u64(&self, key: &GUID, value: u64, context: &'static str) -> Result<(), VideoError> {
        // SAFETY: as above.
        unsafe { self.SetUINT64(key, value) }.map_err(|e| VideoError::mf(context, &e))
    }
    fn guid(&self, key: &GUID, value: &GUID, context: &'static str) -> Result<(), VideoError> {
        // SAFETY: as above.
        unsafe { self.SetGUID(key, value) }.map_err(|e| VideoError::mf(context, &e))
    }
}

/// What opening a writer gives: the writer, the video stream, the first audio stream, the extra
/// audio streams (the separate tracks) and the encoder's identity.
type Opened = (IMFSinkWriter, u32, Option<u32>, Vec<u32>, EncoderInfo);

/// An MP4 being written. See the module docs.
pub struct Mp4Writer {
    /// `None` once finished.
    writer: Option<IMFSinkWriter>,
    video_stream: u32,
    audio_stream: Option<u32>,
    config: EncoderConfig,
    info: EncoderInfo,
    frame_bytes: usize,
    frames: u64,
    audio_frames: u64,
    /// The extra audio tracks' streams and sample frames written to each.
    extra_audio: Vec<(u32, u64)>,
    partial: PathBuf,
    /// The hardware encoder's device on the render adapter, if any: after the
    /// writer (released first), before the session.
    _device: Option<EncoderDevice>,
    /// Declared last: dropped after every Media Foundation object above.
    session: MfSession,
}

impl Mp4Writer {
    /// Open the file and set up the encoders.
    ///
    /// # Errors
    /// [`VideoError::InvalidConfig`], [`VideoError::Unavailable`] (no Media Foundation, no encoder
    /// for the codec, or no hardware encoder with [`EncoderPreference::HardwareOnly`]),
    /// [`VideoError::MediaFoundation`] or [`VideoError::Io`].
    pub fn create(config: EncoderConfig) -> Result<Self, VideoError> {
        config.validate()?;
        let session = MfSession::start()?;
        let partial = config.partial_path();
        if let Some(parent) = partial.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        // A partial file from an earlier crash: never append to it.
        let _ = std::fs::remove_file(&partial);

        // The real rate first; if the encoder refuses it (a codec level's macroblock rate, or a
        // software encoder's 120 fps cap), the same rate halved until one is accepted. The
        // samples keep their exact real timestamps either way, which is what the MP4 stores; only
        // the encoder's rate control and the stream's timing hint see the nominal rate, and the
        // bitrate is scaled so the real bitrate is still the one asked for.
        //
        // Within one rate, the full set of encoder settings first and then only the essential ones
        // (rate control mode, bitrate or quality, keyframe interval): a hardware encoder that
        // refuses an optional setting makes Media Foundation fall back to the software encoder
        // silently (measured: NVENC at 720p60 refused a CBR buffer above the level's limit). While
        // the PC has a hardware encoder for the codec, a software pick is only kept when no
        // combination gets the hardware one.
        let want_hardware = config.encoder == EncoderPreference::Auto
            && super::probe::codec_support(config.codec) == Some(true);
        // The hardware encoder runs on the adapter the frames are rendered on.
        let mut device = encoder_device_for(&config);
        let mut rates = nominal_rates(config.frame_rate);
        // H.264 at the highest level Media Foundation's encoders take: rates
        // over level 5.2's macroblock rate are not offered to a hardware encoder at all.
        // NVIDIA's encoder refuses exactly those rates (so it ends on the same rate and settings
        // as before, without the refused tries); AMD's took them, running outside the level it
        // writes into the stream (1080p at 480 fps: the settings of a friend's export that lost
        // the GPU, though that cause is not proven). The file keeps the real rate either way.
        if config.codec == VideoCodec::H264
            && config.encoder != EncoderPreference::SoftwareOnly
            && level_guard_on()
        {
            let fitting: Vec<FrameRate> = rates
                .iter()
                .copied()
                .filter(|&rate| fits_h264_level_5_2(config.width, config.height, rate))
                .collect();
            if !fitting.is_empty() && fitting.len() < rates.len() {
                tracing::info!(
                    fps = config.frame_rate.as_f64(),
                    nominal_fps = fitting[0].as_f64(),
                    "H.264 level 5.2: the encoder is given a lower nominal rate"
                );
                rates = fitting;
            }
        }
        let all_attempts: Vec<(FrameRate, Params)> = rates
            .into_iter()
            .flat_map(|rate| [(rate, Params::Full), (rate, Params::Essential)])
            .collect();
        // First with the render adapter's device; when that gets no hardware
        // encoder (one that will not take the device), every combination again the way Media
        // Foundation chooses by itself (the path before this change) before the processor's encoder
        // or an error is settled for.
        loop {
            let mut attempts = all_attempts.clone();
            let mut software: Option<(FrameRate, Params)> = None;
            let mut first_error = None;
            let mut index = 0;
            while index < attempts.len() {
                let (nominal, params) = attempts[index];
                index += 1;
                let last = index == attempts.len();
                match Self::open(
                    &session,
                    &config,
                    &partial,
                    nominal,
                    params,
                    device.as_ref(),
                ) {
                    Ok((_, _, _, _, info)) if want_hardware && !info.hardware && !last => {
                        // Dropped unfinished: the partial file goes with it.
                        let _ = std::fs::remove_file(&partial);
                        tracing::debug!(
                            nominal_fps = nominal.as_f64(),
                            ?params,
                            "software encoder picked; trying for the hardware one"
                        );
                        software.get_or_insert((nominal, params));
                    }
                    Ok((_, _, _, _, info))
                        if want_hardware && !info.hardware && software.is_some() =>
                    {
                        // Out of combinations: use the first one that worked at all.
                        let _ = std::fs::remove_file(&partial);
                        if let Some(first) = software.take() {
                            attempts.push(first);
                        }
                    }
                    Ok((writer, _, _, _, info))
                        if want_hardware && !info.hardware && device.is_some() =>
                    {
                        // Not settled for yet: the second pass may find the
                        // hardware encoder without the device.
                        drop(writer);
                        let _ = std::fs::remove_file(&partial);
                        break;
                    }
                    Ok((writer, video_stream, audio_stream, extra_streams, mut info)) => {
                        info.nominal_rate = nominal;
                        tracing::info!(
                            encoder = %info.name,
                            hardware = info.hardware,
                            width = config.width,
                            height = config.height,
                            fps = config.frame_rate.as_f64(),
                            nominal_fps = nominal.as_f64(),
                            codec = config.codec.label(),
                            // The render adapter it was handed, if any.
                            adapter = device.as_ref().map_or("not set", |d| d.adapter.as_str()),
                            "video encoder ready",
                        );
                        return Ok(Self {
                            writer: Some(writer),
                            video_stream,
                            audio_stream,
                            frame_bytes: config.nv12_frame_bytes(),
                            config,
                            info,
                            frames: 0,
                            audio_frames: 0,
                            extra_audio: extra_streams.into_iter().map(|s| (s, 0)).collect(),
                            partial,
                            _device: device,
                            session,
                        });
                    }
                    Err(Open::Encoder(error)) => {
                        let _ = std::fs::remove_file(&partial);
                        tracing::debug!(%error, nominal_fps = nominal.as_f64(), ?params, "encoder refused these settings");
                        first_error.get_or_insert(error);
                        if last {
                            if let Some(first) = software.take() {
                                attempts.push(first);
                            }
                        }
                    }
                    Err(Open::Other(error)) => {
                        let _ = std::fs::remove_file(&partial);
                        // Perhaps the device: the second pass goes without.
                        if device.is_some() {
                            tracing::warn!(
                                %error,
                                "the encoder failed with the render adapter's device"
                            );
                            break;
                        }
                        return Err(error);
                    }
                }
            }
            if device.take().is_some() {
                tracing::warn!(
                    "no hardware encoder took the render adapter's device; Media Foundation \
                     chooses"
                );
                continue;
            }
            return Err(first_error.unwrap_or_else(|| {
                VideoError::Unavailable(format!("no {} encoder", config.codec.label()))
            }));
        }
    }

    fn open(
        session: &MfSession,
        config: &EncoderConfig,
        partial: &std::path::Path,
        nominal: FrameRate,
        params: Params,
        device: Option<&EncoderDevice>,
    ) -> Result<Opened, Open> {
        let api = session.api();
        let mf = |context| move |e: windows::core::Error| VideoError::mf(context, &e);

        let attributes = api.attributes(4).map_err(mf("MFCreateAttributes"))?;
        let hardware = config.encoder != EncoderPreference::SoftwareOnly;
        attributes.u32(
            &MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS,
            u32::from(hardware),
            "enabling hardware encoders",
        )?;
        // The render adapter's device: the sink writer then uses an encoder on
        // that adapter, working on this device instead of one it makes itself.
        if device.is_some() && device_switch() == DeviceSwitch::Refuse {
            return Err(Open::Encoder(VideoError::Unavailable(
                "every encoder refused the device (MC_W19_R2_ENCODER_DEVICE=refuse)".to_owned(),
            )));
        }
        if let Some(device) = device.filter(|_| hardware) {
            // SAFETY: a live attribute store and device manager (the store AddRefs it).
            unsafe { attributes.SetUnknown(&MF_SINK_WRITER_D3D_MANAGER, &device.manager) }
                .map_err(mf("handing the encoder its graphics device"))?;
        }
        attributes.guid(
            &MF_TRANSCODE_CONTAINERTYPE,
            &MFTranscodeContainerType_MPEG4,
            "choosing MP4",
        )?;
        // Fast start is done after `Finalize` by `crate::faststart`, not with
        // MF_MPEG4SINK_MOOV_BEFORE_MDAT, which was measured broken (see that module).
        let path = partial.to_string_lossy();
        let file = api
            .read_write_file(&path)
            .map_err(mf("creating the MP4 file"))?;
        let writer = api
            .sink_writer(&file, &attributes)
            .map_err(mf("creating the MP4 writer"))?;
        // The sink holds its own reference; the file closes when the writer is released.
        drop(file);

        // ---- video ----
        let (w, h) = (config.width, config.height);
        let rate = pack(nominal.num, nominal.den);
        let subtype = match config.codec {
            VideoCodec::H264 => MFVideoFormat_H264,
            VideoCodec::H265 => MFVideoFormat_HEVC,
        };
        let out = api.media_type().map_err(mf("MFCreateMediaType"))?;
        out.guid(&MF_MT_MAJOR_TYPE, &MFMediaType_Video, "video output type")?;
        out.guid(&MF_MT_SUBTYPE, &subtype, "video output subtype")?;
        out.u32(
            &MF_MT_AVG_BITRATE,
            scaled(nominal_bitrate(config), config, nominal),
            "video bitrate",
        )?;
        out.u32(
            &MF_MT_INTERLACE_MODE,
            MFVideoInterlace_Progressive.0 as u32,
            "progressive",
        )?;
        out.u64(&MF_MT_FRAME_SIZE, pack(w, h), "frame size")?;
        out.u64(&MF_MT_FRAME_RATE, rate, "frame rate")?;
        out.u64(&MF_MT_PIXEL_ASPECT_RATIO, pack(1, 1), "pixel aspect")?;
        let profile = match config.codec {
            VideoCodec::H264 => eAVEncH264VProfile_High.0 as u32,
            VideoCodec::H265 => eAVEncH265VProfile_Main_420_8.0 as u32,
        };
        out.u32(&MF_MT_MPEG2_PROFILE, profile, "profile")?;
        set_colour(&out)?;
        // SAFETY: a live writer and media type.
        let video_stream = unsafe { writer.AddStream(&out) }
            .map_err(|e| Open::Encoder(codec_error(config.codec, &e)))?;

        let input = api.media_type().map_err(mf("MFCreateMediaType"))?;
        input.guid(&MF_MT_MAJOR_TYPE, &MFMediaType_Video, "video input type")?;
        input.guid(&MF_MT_SUBTYPE, &MFVideoFormat_NV12, "NV12 input")?;
        input.u32(
            &MF_MT_INTERLACE_MODE,
            MFVideoInterlace_Progressive.0 as u32,
            "progressive",
        )?;
        input.u64(&MF_MT_FRAME_SIZE, pack(w, h), "frame size")?;
        input.u64(&MF_MT_FRAME_RATE, rate, "frame rate")?;
        input.u64(&MF_MT_PIXEL_ASPECT_RATIO, pack(1, 1), "pixel aspect")?;
        input.u32(&MF_MT_DEFAULT_STRIDE, w, "stride")?;
        input.u32(&MF_MT_ALL_SAMPLES_INDEPENDENT, 1, "independent samples")?;
        set_colour(&input)?;
        let params = encoding_parameters(session, config, nominal, params)?;
        // SAFETY: a live writer, stream index from AddStream, live attribute stores.
        unsafe { writer.SetInputMediaType(video_stream, &input, &params) }
            .map_err(|e| Open::Encoder(codec_error(config.codec, &e)))?;

        // ---- audio ----
        // The first track, then one more per `audio_tracks`
        // name, all in the same format.
        let mut audio_streams = Vec::new();
        // With extra tracks, one more at the end: a copy of the mix. Windows' playback stack (the
        // Media Player app) plays the *last* audio track of a file while browsers (chat apps),
        // ffmpeg and editors take the first (measured, `tests/tracks.rs`), so the mix is both.
        let copy = usize::from(!config.audio_tracks.is_empty());
        for _ in 0..1 + config.audio_tracks.len() + copy {
            let Some(audio) = config.audio else {
                break;
            };
            {
                let out = api.media_type().map_err(mf("MFCreateMediaType"))?;
                out.guid(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio, "audio output type")?;
                out.guid(&MF_MT_SUBTYPE, &MFAudioFormat_AAC, "AAC")?;
                out.u32(&MF_MT_AUDIO_BITS_PER_SAMPLE, 16, "audio bits")?;
                out.u32(
                    &MF_MT_AUDIO_SAMPLES_PER_SECOND,
                    audio.sample_rate,
                    "audio rate",
                )?;
                out.u32(
                    &MF_MT_AUDIO_NUM_CHANNELS,
                    u32::from(audio.channels),
                    "audio channels",
                )?;
                out.u32(
                    &MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
                    audio.bitrate.bytes_per_second(),
                    "AAC bitrate",
                )?;
                // Raw AAC (no ADTS headers): what MP4 stores. AAC-LC, level 2 (0x29).
                out.u32(&MF_MT_AAC_PAYLOAD_TYPE, 0, "AAC payload")?;
                out.u32(
                    &MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION,
                    0x29,
                    "AAC profile",
                )?;
                // SAFETY: a live writer and media type.
                let stream =
                    unsafe { writer.AddStream(&out) }.map_err(mf("adding the audio stream"))?;
                let input = api.media_type().map_err(mf("MFCreateMediaType"))?;
                let block = u32::from(audio.channels) * 2;
                input.guid(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio, "audio input type")?;
                input.guid(&MF_MT_SUBTYPE, &MFAudioFormat_PCM, "PCM")?;
                input.u32(&MF_MT_AUDIO_BITS_PER_SAMPLE, 16, "audio bits")?;
                input.u32(
                    &MF_MT_AUDIO_SAMPLES_PER_SECOND,
                    audio.sample_rate,
                    "audio rate",
                )?;
                input.u32(
                    &MF_MT_AUDIO_NUM_CHANNELS,
                    u32::from(audio.channels),
                    "audio channels",
                )?;
                input.u32(&MF_MT_AUDIO_BLOCK_ALIGNMENT, block, "audio block")?;
                input.u32(
                    &MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
                    block * audio.sample_rate,
                    "PCM rate",
                )?;
                // SAFETY: a live writer, stream index from AddStream, a live media type.
                unsafe { writer.SetInputMediaType(stream, &input, None) }
                    .map_err(mf("setting up the AAC encoder"))?;
                audio_streams.push(stream);
            }
        }
        let audio_stream = audio_streams.first().copied();
        // The extra tracks, then the mix's copy last (written by `write_audio` with the mix).
        let extra_streams = audio_streams
            .get(1..)
            .map_or_else(Vec::new, <[u32]>::to_vec);

        // SAFETY: a live, fully configured writer.
        unsafe { writer.BeginWriting() }
            .map_err(|e| Open::Encoder(codec_error(config.codec, &e)))?;
        let (hardware, name) = encoder_identity(&writer, video_stream);
        if config.encoder == EncoderPreference::HardwareOnly && !hardware {
            return Err(Open::Other(VideoError::Unavailable(format!(
                "no hardware {} encoder (Media Foundation picked {name})",
                config.codec.label()
            ))));
        }
        let info = EncoderInfo {
            codec: config.codec,
            hardware,
            name,
            audio: audio_stream.is_some(),
            nominal_rate: nominal,
        };
        Ok((writer, video_stream, audio_stream, extra_streams, info))
    }

    /// The encoder in use.
    #[must_use]
    pub fn info(&self) -> &EncoderInfo {
        &self.info
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Video frames written so far.
    #[must_use]
    pub fn frames_written(&self) -> u64 {
        self.frames
    }

    /// Audio sample frames written so far.
    #[must_use]
    pub fn audio_frames_written(&self) -> u64 {
        self.audio_frames
    }

    /// Video frames handed to Media Foundation that its encoder has not output
    /// yet (the sink writer's statistics), or `None` when it cannot say. The encoder thread paces
    /// a hardware encoder with it (`thread.rs`, `EncoderPace`).
    #[must_use]
    pub fn video_frames_in_encoder(&self) -> Option<u64> {
        use windows::Win32::Media::MediaFoundation::MF_SINK_WRITER_STATISTICS;
        let writer = self.writer.as_ref()?;
        let mut stats = MF_SINK_WRITER_STATISTICS {
            cb: std::mem::size_of::<MF_SINK_WRITER_STATISTICS>() as u32,
            ..Default::default()
        };
        // SAFETY: a live writer, a stream index from `AddStream`, and a statistics block whose
        // size is in `cb`.
        unsafe { writer.GetStatistics(self.video_stream, &mut stats) }.ok()?;
        Some(
            stats
                .qwNumSamplesReceived
                .saturating_sub(stats.qwNumSamplesEncoded),
        )
    }

    /// Append the next video frame: NV12, exactly [`EncoderConfig::nv12_frame_bytes`] long. Its
    /// time is its index over the frame rate.
    ///
    /// # Errors
    /// [`VideoError::FrameSize`] or [`VideoError::MediaFoundation`].
    pub fn write_frame(&mut self, nv12: &[u8]) -> Result<(), VideoError> {
        if nv12.len() != self.frame_bytes {
            return Err(VideoError::FrameSize {
                got: nv12.len(),
                expected: self.frame_bytes,
            });
        }
        let rate = self.config.frame_rate;
        let time = rate.frame_start_hns(self.frames);
        let duration = rate.frame_duration_hns(self.frames);
        self.write_sample(
            self.video_stream,
            nv12,
            time,
            duration,
            "writing a video frame",
        )?;
        self.frames += 1;
        Ok(())
    }

    /// Append audio: interleaved 16-bit samples, a whole number of frames for the configured
    /// channel count. Timed from the end of the audio written so far.
    ///
    /// # Errors
    /// [`VideoError::InvalidConfig`] without an audio track, [`VideoError::MediaFoundation`].
    pub fn write_audio(&mut self, pcm: &[i16]) -> Result<(), VideoError> {
        let (Some(stream), Some(audio)) = (self.audio_stream, self.config.audio) else {
            return Err(VideoError::InvalidConfig("no audio track".to_owned()));
        };
        let channels = usize::from(audio.channels);
        let frames = (pcm.len() / channels) as u64;
        if frames == 0 {
            return Ok(());
        }
        let rate = u64::from(audio.sample_rate);
        let start =
            (u128::from(self.audio_frames) * u128::from(HNS_PER_SECOND) / u128::from(rate)) as i64;
        let end = (u128::from(self.audio_frames + frames) * u128::from(HNS_PER_SECOND)
            / u128::from(rate)) as i64;
        let bytes: &[u8] = bytes_of_i16(&pcm[..frames as usize * channels]);
        self.write_sample(stream, bytes, start, end - start, "writing audio")?;
        self.audio_frames += frames;
        // With extra tracks, the mix again as the last track (see `open`).
        if let Some(copy) = self.extra_audio.len().checked_sub(1) {
            self.write_audio_track(copy, pcm)?;
        }
        Ok(())
    }

    /// How many audio tracks after the first an [`crate::AudioSource`] fills
    /// ([`EncoderConfig::audio_tracks`]); the mix's copy at the end is written by
    /// [`Self::write_audio`] itself.
    #[must_use]
    pub fn extra_audio_tracks(&self) -> usize {
        self.extra_audio.len().saturating_sub(1)
    }

    /// Append audio to extra track `track` (0 = the first after the mixed one), as
    /// [`Self::write_audio`] does for the first. Each track keeps its own timeline.
    ///
    /// # Errors
    /// [`VideoError::InvalidConfig`] for a track that does not exist,
    /// [`VideoError::MediaFoundation`].
    pub fn write_audio_track(&mut self, track: usize, pcm: &[i16]) -> Result<(), VideoError> {
        let (Some(&(stream, written)), Some(audio)) =
            (self.extra_audio.get(track), self.config.audio)
        else {
            return Err(VideoError::InvalidConfig("no such audio track".to_owned()));
        };
        let channels = usize::from(audio.channels);
        let frames = (pcm.len() / channels) as u64;
        if frames == 0 {
            return Ok(());
        }
        let rate = u128::from(audio.sample_rate);
        let start = (u128::from(written) * u128::from(HNS_PER_SECOND) / rate) as i64;
        let end = (u128::from(written + frames) * u128::from(HNS_PER_SECOND) / rate) as i64;
        let bytes: &[u8] = bytes_of_i16(&pcm[..frames as usize * channels]);
        self.write_sample(stream, bytes, start, end - start, "writing an audio track")?;
        self.extra_audio[track].1 += frames;
        Ok(())
    }

    fn write_sample(
        &self,
        stream: u32,
        bytes: &[u8],
        time: i64,
        duration: i64,
        context: &'static str,
    ) -> Result<(), VideoError> {
        let api = self.session.api();
        let writer = self.writer.as_ref().ok_or(VideoError::Cancelled)?;
        let mf = |e: windows::core::Error| VideoError::mf(context, &e);
        let length = u32::try_from(bytes.len())
            .map_err(|_| VideoError::InvalidConfig("sample too large".to_owned()))?;
        let buffer = api.memory_buffer(length).map_err(mf)?;
        // SAFETY: Lock hands out a pointer to at least `length` writable bytes (the buffer was
        // created with that maximum) until Unlock; nothing else touches the new buffer meanwhile.
        unsafe {
            let mut data = std::ptr::null_mut();
            buffer.Lock(&mut data, None, None).map_err(mf)?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len());
            buffer.Unlock().map_err(mf)?;
            buffer.SetCurrentLength(length).map_err(mf)?;
        }
        let sample = api.sample().map_err(mf)?;
        // SAFETY: live objects; WriteSample AddRefs what it keeps.
        unsafe {
            sample.AddBuffer(&buffer).map_err(mf)?;
            sample.SetSampleTime(time).map_err(mf)?;
            sample.SetSampleDuration(duration).map_err(mf)?;
            writer.WriteSample(stream, &sample).map_err(mf)?;
        }
        Ok(())
    }

    /// Finish the file: flush the encoders, write the index, rename `.partial` to the real name.
    ///
    /// # Errors
    /// [`VideoError::MediaFoundation`] or [`VideoError::Io`]; the partial file is deleted.
    pub fn finish(mut self) -> Result<EncodeSummary, VideoError> {
        let writer = self.writer.take().ok_or(VideoError::Cancelled)?;
        // SAFETY: a live writer that has begun writing.
        let finalized = unsafe { writer.Finalize() };
        drop(writer);
        if let Err(error) = finalized {
            let _ = std::fs::remove_file(&self.partial);
            return Err(VideoError::mf("finishing the file", &error));
        }
        let mut placed = false;
        if self.config.faststart {
            let mut staged = self.partial.clone().into_os_string();
            staged.push("2");
            let staged = PathBuf::from(staged);
            match crate::faststart::move_index_to_front_stamped(
                &self.partial,
                &staged,
                self.config.created,
            ) {
                Ok(true) => match std::fs::rename(&staged, &self.config.path) {
                    Ok(()) => {
                        let _ = std::fs::remove_file(&self.partial);
                        placed = true;
                    }
                    Err(error) => {
                        let _ = std::fs::remove_file(&staged);
                        tracing::warn!(%error, "fast-start copy could not be renamed; keeping the plain file");
                    }
                },
                Ok(false) => {}
                Err(error) => {
                    let _ = std::fs::remove_file(&staged);
                    tracing::warn!(%error, "fast-start rewrite failed; keeping the plain file");
                }
            }
        }
        if !placed {
            if let Err(error) = std::fs::rename(&self.partial, &self.config.path) {
                let _ = std::fs::remove_file(&self.partial);
                return Err(error.into());
            }
        }
        if !self.extra_audio.is_empty() {
            // Name the audio tracks and keep the mixed one the only one players pick.
            let mut names = vec!["Mix"];
            names.extend(self.config.audio_tracks.iter().map(String::as_str));
            names.push("Mix");
            match crate::tracks::label_audio_tracks(&self.config.path, &names) {
                Ok(true) => {}
                Ok(false) => tracing::warn!("audio tracks could not be labelled"),
                Err(error) => tracing::warn!(%error, "audio tracks could not be labelled"),
            }
        }
        let bytes = std::fs::metadata(&self.config.path)?.len();
        Ok(EncodeSummary {
            path: self.config.path.clone(),
            bytes,
            frames: self.frames,
            duration_hns: self.config.frame_rate.frame_start_hns(self.frames),
            audio_frames: self.audio_frames,
            info: self.info.clone(),
        })
    }
}

impl Drop for Mp4Writer {
    /// Dropping an unfinished writer abandons the file: the writer is released without
    /// `Finalize` and the partial file deleted.
    fn drop(&mut self) {
        if let Some(writer) = self.writer.take() {
            drop(writer);
            if let Err(error) = std::fs::remove_file(&self.partial) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    // The file name only (players share logs).
                    let file = self
                        .partial
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    tracing::warn!(%error, file, "could not delete a partial export");
                }
            }
        }
    }
}

/// `MC_W19_R2_ENCODER_DEVICE` (tests only, read once): `0` never hands the
/// encoder a device (the path before this change); `refuse` makes every encoder refuse it, which
/// tests the second pass of [`Mp4Writer::create`]. Unset for every player: on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceSwitch {
    On,
    Off,
    Refuse,
}

fn device_switch() -> DeviceSwitch {
    static SWITCH: std::sync::OnceLock<DeviceSwitch> = std::sync::OnceLock::new();
    *SWITCH.get_or_init(
        || match std::env::var("MC_W19_R2_ENCODER_DEVICE").as_deref() {
            Ok("0") => DeviceSwitch::Off,
            Ok("refuse") => DeviceSwitch::Refuse,
            _ => DeviceSwitch::On,
        },
    )
}

/// A device manager on the adapter the frames are rendered on, when the config
/// names one and a hardware encoder is allowed. Without one (no adapter given, none found, or
/// `MC_W19_R2_ENCODER_DEVICE=0`) Media Foundation chooses as before.
fn encoder_device_for(config: &EncoderConfig) -> Option<EncoderDevice> {
    let adapter = config.adapter.as_ref()?;
    if device_switch() == DeviceSwitch::Off || config.encoder == EncoderPreference::SoftwareOnly {
        return None;
    }
    match super::device::encoder_device(adapter) {
        Ok(device) => {
            tracing::info!(
                adapter = %device.adapter,
                luid = format_args!("{:016X}", device.luid),
                "hardware encoder device made on the render adapter"
            );
            Some(device)
        }
        Err(why) => {
            tracing::warn!(
                %why,
                adapter = %adapter.name,
                "no encoder device on the render adapter; Media Foundation chooses the encoder"
            );
            None
        }
    }
}

/// The bitrate written into the output type. Media Foundation's H.264 encoder requires one even in
/// quality mode, where it is only a hint.
fn nominal_bitrate(config: &EncoderConfig) -> u32 {
    match config.rate {
        RateControl::Bitrate { average } => average,
        RateControl::ConstantBitrate { bitrate } => bitrate,
        RateControl::Quality { max_bitrate, .. } => max_bitrate.unwrap_or_else(|| {
            // About 0.1 bit per pixel at 60 fps, growing more slowly than the frame rate.
            let pixels = f64::from(config.width) * f64::from(config.height);
            let fps = config.frame_rate.as_f64();
            (pixels * 0.1 * fps.powf(0.75) * 60f64.powf(0.25)).clamp(500_000.0, 400_000_000.0)
                as u32
        }),
    }
}

/// The CODECAPI settings, as the attribute store `SetInputMediaType` passes to the encoder.
fn encoding_parameters(
    session: &MfSession,
    config: &EncoderConfig,
    nominal: FrameRate,
    level: Params,
) -> Result<IMFAttributes, VideoError> {
    let full = level == Params::Full;
    let params = session
        .api()
        .attributes(8)
        .map_err(|e| VideoError::mf("MFCreateAttributes", &e))?;
    let bits = |bitrate: u32| scaled(bitrate, config, nominal);
    match config.rate {
        RateControl::Quality {
            quality,
            max_bitrate,
        } => {
            params.u32(
                &CODECAPI_AVEncCommonRateControlMode,
                eAVEncCommonRateControlMode_Quality.0 as u32,
                "quality mode",
            )?;
            params.u32(&CODECAPI_AVEncCommonQuality, u32::from(quality), "quality")?;
            if let Some(max) = max_bitrate.filter(|_| full) {
                params.u32(&CODECAPI_AVEncCommonMaxBitRate, bits(max), "max bitrate")?;
            }
        }
        RateControl::Bitrate { average } => {
            params.u32(
                &CODECAPI_AVEncCommonRateControlMode,
                eAVEncCommonRateControlMode_PeakConstrainedVBR.0 as u32,
                "VBR mode",
            )?;
            params.u32(&CODECAPI_AVEncCommonMeanBitRate, bits(average), "bitrate")?;
            if full {
                params.u32(
                    &CODECAPI_AVEncCommonMaxBitRate,
                    bits(average.saturating_add(average / 2)),
                    "peak bitrate",
                )?;
            }
        }
        RateControl::ConstantBitrate { bitrate } => {
            params.u32(
                &CODECAPI_AVEncCommonRateControlMode,
                eAVEncCommonRateControlMode_CBR.0 as u32,
                "CBR mode",
            )?;
            params.u32(&CODECAPI_AVEncCommonMeanBitRate, bits(bitrate), "bitrate")?;
            // A half-second buffer (in the encoder's nominal time; the value is in bytes):
            // tight enough that the size cannot drift. Measured: given in bits, NVENC refused it
            // at 720p60 (over the level's CPB limit) and Media Foundation quietly used the
            // software encoder instead.
            if full {
                params.u32(
                    &CODECAPI_AVEncCommonBufferSize,
                    bits(bitrate) / 16,
                    "buffer size",
                )?;
            }
        }
    }
    params.u32(
        &CODECAPI_AVEncMPVGOPSize,
        config.effective_keyframe_interval(),
        "keyframe interval",
    )?;
    if full {
        params.u32(&CODECAPI_AVEncCommonQualityVsSpeed, 70, "quality vs speed")?;
    }
    Ok(params)
}

/// BT.709 limited range on a video media type (see `nv12.rs`).
fn set_colour(media_type: &IMFMediaType) -> Result<(), VideoError> {
    media_type.u32(
        &MF_MT_VIDEO_NOMINAL_RANGE,
        MFNominalRange_16_235.0 as u32,
        "nominal range",
    )?;
    media_type.u32(
        &MF_MT_YUV_MATRIX,
        MFVideoTransferMatrix_BT709.0 as u32,
        "YUV matrix",
    )?;
    media_type.u32(
        &MF_MT_VIDEO_PRIMARIES,
        MFVideoPrimaries_BT709.0 as u32,
        "primaries",
    )?;
    media_type.u32(
        &MF_MT_TRANSFER_FUNCTION,
        MFVideoTransFunc_709.0 as u32,
        "transfer function",
    )
}

/// A failure setting up the video stream, with "no encoder for this codec" made readable.
fn codec_error(codec: VideoCodec, error: &windows::core::Error) -> VideoError {
    let code = error.code().0 as u32;
    if code == MF_E_TOPO_CODEC_NOT_FOUND || code == MF_E_INVALIDMEDIATYPE {
        VideoError::Unavailable(format!(
            "this PC has no {} encoder that accepts these settings ({})",
            codec.label(),
            error.message().trim()
        ))
    } else {
        VideoError::mf("setting up the video encoder", error)
    }
}

/// Which encoder the sink writer loaded for `stream`, and whether it is hardware.
fn encoder_identity(writer: &IMFSinkWriter, stream: u32) -> (bool, String) {
    let Ok(ex) = writer.cast::<IMFSinkWriterEx>() else {
        return (false, "unknown encoder".to_owned());
    };
    for index in 0..8 {
        let mut category = GUID::zeroed();
        let mut transform: Option<IMFTransform> = None;
        // SAFETY: valid out-pointers; the call fails past the last transform.
        if unsafe { ex.GetTransformForStream(stream, index, Some(&mut category), &mut transform) }
            .is_err()
        {
            break;
        }
        if category != MFT_CATEGORY_VIDEO_ENCODER {
            continue;
        }
        let Some(transform) = transform else { continue };
        // SAFETY: a live transform.
        let Ok(attributes) = (unsafe { transform.GetAttributes() }) else {
            return (false, "unknown encoder".to_owned());
        };
        // SAFETY: a live attribute store.
        let hardware =
            unsafe { attributes.GetItem(&MFT_ENUM_HARDWARE_URL_Attribute, None) }.is_ok();
        // Some hardware encoders carry no friendly name on the transform (a
        // friend's AMD log said only "hardware encoder"): their hardware URL names them instead.
        let name = string_attribute(&attributes, &MFT_FRIENDLY_NAME_Attribute)
            .or_else(|| string_attribute(&attributes, &MFT_ENUM_HARDWARE_URL_Attribute))
            .unwrap_or_else(|| {
                if hardware {
                    "hardware encoder".to_owned()
                } else {
                    "Microsoft software encoder".to_owned()
                }
            });
        // SAFETY: a live attribute store; absent on encoders that do not say.
        let luid = unsafe { attributes.GetUINT64(&MFT_ENUM_ADAPTER_LUID) }.ok();
        let vendor = string_attribute(&attributes, &MFT_ENUM_HARDWARE_VENDOR_ID_Attribute);
        tracing::debug!(
            name,
            hardware,
            luid = ?luid.map(|l| format!("{l:016X}")),
            ?vendor,
            "the sink writer's video encoder"
        );
        return (hardware, name);
    }
    (false, "unknown encoder".to_owned())
}

/// View 16-bit samples as their little-endian bytes (x86 and ARM Windows are little-endian).
fn bytes_of_i16(samples: &[i16]) -> &[u8] {
    // SAFETY: i16 has no padding and alignment 2 >= 1; the byte slice covers exactly the same
    // memory and lifetime.
    unsafe {
        std::slice::from_raw_parts(
            samples.as_ptr().cast::<u8>(),
            std::mem::size_of_val(samples),
        )
    }
}

#[cfg(test)]
mod w19_r2_tests {
    use super::*;

    /// The boundaries measured on NVIDIA's encoder (it refuses exactly what does not fit).
    #[test]
    fn level_5_2_is_the_measured_boundary() {
        let fits = |w, h, fps| fits_h264_level_5_2(w, h, FrameRate::whole(fps));
        assert!(fits(1920, 1080, 254));
        assert!(!fits(1920, 1080, 255));
        assert!(fits(1280, 720, 576));
        assert!(!fits(1280, 720, 577));
        assert!(fits(3840, 2160, 64));
        assert!(!fits(3840, 2160, 65));
        // Fractional rates compare exactly: 1080p's limit is 2073600 / 8160 = 254.12 fps.
        assert!(fits_h264_level_5_2(1920, 1080, FrameRate::new(2541, 10)));
        assert!(!fits_h264_level_5_2(1920, 1080, FrameRate::new(2542, 10)));
    }

    /// A friend's export (1080p at 480 fps) is offered to the encoder at 240 first.
    #[test]
    fn a_1080p_480_fps_export_starts_at_a_nominal_240() {
        let first_fitting = nominal_rates(FrameRate::whole(480))
            .into_iter()
            .find(|&rate| fits_h264_level_5_2(1920, 1080, rate));
        assert_eq!(first_fitting, Some(FrameRate::new(480, 2)));
        // 60 fps at any size up to 4K fits as before: nothing changes there.
        for (w, h) in [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)] {
            assert!(fits_h264_level_5_2(w, h, FrameRate::whole(60)), "{w}x{h}");
        }
    }
}
