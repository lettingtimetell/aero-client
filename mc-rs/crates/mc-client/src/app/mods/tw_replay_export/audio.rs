//! The export's soundtrack, mixed offline.
//!
//! The recorder keeps every sound the client started (event, the variant's `.ogg`, volume, pitch,
//! position, attenuation). Here each one is decoded with the client's own OGG decoder
//! (`mc_audio::ogg`), given the same gain and pitch the game gave it (`normalized_volume`,
//! `normalized_pitch`, the user's category sliders, the Sounds mod's per-event gain), and placed on
//! the export's timeline. Distance attenuation and panning are recomputed every 10 ms against the
//! recorded listener (the player's eyes, where the sound was heard), with vanilla's linear falloff
//! (`mc_audio::attenuate`) and the game's pan (`mc_audio::pan`). The microphone (the IMA
//! ADPCM frames) joins as one centred voice. `mc_video::mix` does the resampling, panning and
//! limiting; the encoder thread pulls the result as it encodes.

use std::collections::HashMap;
use std::sync::Arc;

use mc_audio::{attenuate, pan, AttenuationType, Listener};
use mc_math::DVec3;
use mc_replay::ReplayTime;
use mc_video::mix::{Clip, Limiter, OfflineMixer, Voice};
use mc_video::AudioSource;

/// The export's audio rate.
pub(super) const RATE: u32 = 48_000;

/// Output frames between listener samples (10 ms).
pub(super) const LISTENER_STEP: u32 = 480;

/// How far before the range a sound may start and still be heard in it (music, records).
pub(super) const LOOKBACK_MICROS: u64 = 300_000_000;

/// The same for ordinary effects (none of vanilla's lasts anywhere near this long).
pub(super) const EFFECT_LOOKBACK_MICROS: u64 = 30_000_000;

/// What the spatial callback needs for one voice.
#[derive(Debug, Clone, Copy)]
pub(super) struct Placement {
    /// World position; `None` plays centred at full volume.
    pub position: Option<DVec3>,
    /// Linear distance falloff (else none).
    pub linear: bool,
    /// `ISound.getVolume()`: louder sounds fade out further away.
    pub volume: f32,
}

/// One sound to place, as the client resolved it.
pub(super) struct PlacedSound {
    /// When it starts, in µs on the replay clock.
    pub time: ReplayTime,
    /// Decoded samples.
    pub clip: Arc<Clip>,
    /// Gain before distance (volume x variant x category x per-event gain), 0..=1.
    pub gain: f32,
    /// Playback rate, 0.5..=2.
    pub pitch: f32,
    /// Where and how it is heard.
    pub placement: Placement,
}

/// The peak limiter's ceiling: -2 dBFS. AAC at 128 kbit/s overshoots dense, loud mixes by about
/// 1.5 dB (measured: a -1 dBFS ceiling decoded to +0.5 dBFS peaks).
const CEILING: f32 = 0.79;

/// A centred voice channel (the microphone, desktop audio),
/// mixed on its own so it can also be written as a track of its own.
pub(super) struct VoiceChannel {
    /// The channel's material as one voice (gain = the channel's level).
    pub voice: Voice,
    /// Also write it as its own track.
    pub track: bool,
}

/// One centred channel while mixing.
struct VoiceMix {
    mixer: OfflineMixer,
    track: Option<Limiter>,
    buffer: Vec<f32>,
}

/// The mix, pulled by the encoder thread.
///
/// The sound is kept in channels until the end, the game's placed sounds in one mixer
/// and each centred voice (the microphone, desktop audio) in its own; the mix is their sum
/// through the peak limiter. With separate tracks, each channel also goes through a limiter of
/// its own into a track ([`AudioSource::fill_track`], game first, then the voices in order).
pub(super) struct MixSource {
    mixer: OfflineMixer,
    placements: Vec<Placement>,
    listeners: Vec<Listener>,
    limiter: Limiter,
    frame: i64,
    buffer: Vec<f32>,
    /// The game channel's level (1 leaves the game's own gains as they are).
    game_gain: f32,
    /// The game channel's own track, when written.
    game_track: Option<Limiter>,
    game_buffer: Vec<f32>,
    voices: Vec<VoiceMix>,
    /// This chunk of every track, in track order.
    tracks: Vec<Vec<i16>>,
}

impl MixSource {
    /// A mix of `sounds` from `start` on, heard by `listeners` (one per [`LISTENER_STEP`] from
    /// `start`; the last one holds after its end), plus centred `extra` voices (the microphone).
    /// The export itself uses [`MixSource::with_channels`].
    #[cfg(test)]
    pub(super) fn new(
        start: ReplayTime,
        sounds: Vec<PlacedSound>,
        listeners: Vec<Listener>,
        extra: Vec<Voice>,
    ) -> Self {
        let voices = extra
            .into_iter()
            .map(|voice| VoiceChannel {
                voice,
                track: false,
            })
            .collect();
        Self::with_channels(start, sounds, listeners, 1.0, false, voices)
    }

    /// [`MixSource::new`] with the channels kept apart: the game's sounds at
    /// `game_gain` (written as a track of their own too when `game_track`), and centred voice
    /// channels.
    pub(super) fn with_channels(
        start: ReplayTime,
        mut sounds: Vec<PlacedSound>,
        listeners: Vec<Listener>,
        game_gain: f32,
        game_track: bool,
        voices: Vec<VoiceChannel>,
    ) -> Self {
        sounds.sort_by_key(|sound| sound.time);
        let mut game = Vec::with_capacity(sounds.len());
        let mut placements = Vec::with_capacity(sounds.len());
        for sound in sounds {
            let offset = sound.time as i64 - start as i64;
            game.push(Voice {
                clip: sound.clip,
                start_frame: offset * i64::from(RATE) / 1_000_000,
                pitch: sound.pitch,
                gain: sound.gain,
                looping: false,
                end_frame: None,
            });
            placements.push(sound.placement);
        }
        let voices: Vec<VoiceMix> = voices
            .into_iter()
            .map(|channel| VoiceMix {
                mixer: OfflineMixer::new(RATE, vec![channel.voice]),
                track: channel.track.then(|| Limiter::new(CEILING)),
                buffer: Vec::new(),
            })
            .collect();
        let track_count =
            usize::from(game_track) + voices.iter().filter(|v| v.track.is_some()).count();
        Self {
            mixer: OfflineMixer::new(RATE, game),
            placements,
            listeners,
            limiter: Limiter::new(CEILING),
            frame: 0,
            buffer: Vec::new(),
            game_gain,
            game_track: game_track.then(|| Limiter::new(CEILING)),
            game_buffer: Vec::new(),
            voices,
            tracks: vec![Vec::new(); track_count],
        }
    }
}

impl AudioSource for MixSource {
    fn fill(&mut self, out: &mut [i16]) {
        self.buffer.resize(out.len(), 0.0);
        let placements = &self.placements;
        let listeners = &self.listeners;
        self.mixer
            .render(self.frame, &mut self.buffer, |voice, frame| {
                let placement = placements[voice];
                let Some(position) = placement.position else {
                    return (1.0, 0.0);
                };
                let index = (frame.max(0) as usize / LISTENER_STEP as usize)
                    .min(listeners.len().saturating_sub(1));
                let Some(listener) = listeners.get(index) else {
                    return (1.0, 0.0);
                };
                let kind = if placement.linear {
                    AttenuationType::Linear
                } else {
                    AttenuationType::None
                };
                (
                    attenuate(listener.distance_to(position), kind, placement.volume),
                    pan(listener, position),
                )
            });
        // The game channel's level, its track, then the voice channels on top. With
        // the game alone at 100% this is exactly the mix it always was.
        if self.game_gain != 1.0 {
            let gain = self.game_gain;
            self.buffer.iter_mut().for_each(|s| *s *= gain);
        }
        let mut track = 0;
        if let Some(limiter) = self.game_track.as_mut() {
            self.game_buffer.clone_from(&self.buffer);
            self.tracks[track].resize(out.len(), 0);
            limiter.process(&self.game_buffer, &mut self.tracks[track]);
            track += 1;
        }
        for voice in &mut self.voices {
            voice.buffer.resize(out.len(), 0.0);
            voice
                .mixer
                .render(self.frame, &mut voice.buffer, |_, _| (1.0, 0.0));
            for (mix, s) in self.buffer.iter_mut().zip(&voice.buffer) {
                *mix += s;
            }
            if let Some(limiter) = voice.track.as_mut() {
                self.tracks[track].resize(out.len(), 0);
                limiter.process(&voice.buffer, &mut self.tracks[track]);
                track += 1;
            }
        }
        self.limiter.process(&self.buffer, out);
        self.frame += (out.len() / 2) as i64;
    }

    fn fill_track(&mut self, track: usize, out: &mut [i16]) {
        match self.tracks.get(track) {
            Some(samples) if samples.len() == out.len() => out.copy_from_slice(samples),
            _ => out.fill(0),
        }
    }
}

/// The most interleaved samples one recorded sound decodes to for an export (see
/// [`ClipCache::get`]).
const MAX_CLIP_SAMPLES: usize = 1 << 28;

/// Decoded clips, by `.ogg` location, so a sound heard 300 times is decoded once.
#[derive(Default)]
pub(super) struct ClipCache {
    clips: HashMap<String, Option<Arc<Clip>>>,
}

impl ClipCache {
    /// The clip for `file`, decoding it on first use. `None` if it cannot be read or decoded.
    pub(super) fn get(
        &mut self,
        resources: &mc_assets::ResourceManager,
        file: &str,
    ) -> Option<Arc<Clip>> {
        self.clips
            .entry(file.to_owned())
            .or_insert_with(|| {
                let location = mc_assets::ResourceLocation::parse(file).ok()?;
                let bytes = resources.load(&location).ok()?;
                // Music and records are decoded whole here too (the
                // game streams them): vanilla's longest track is about 80 million samples, so
                // the export allows 2^28 (some 50 minutes of 44.1 kHz stereo), not the game's
                // limit for clips it decodes whole.
                let decoded =
                    mc_audio::ogg::decode_all_within(&bytes, file, MAX_CLIP_SAMPLES).ok()?;
                Some(Arc::new(Clip::from_i16(
                    decoded.sample_rate,
                    decoded.channels,
                    &decoded.samples,
                )))
            })
            .clone()
    }

    /// Distinct files decoded (or tried).
    pub(super) fn len(&self) -> usize {
        self.clips.len()
    }
}

/// The sample rates a recorded voice stream can have (the client records 24 kHz).
const VOICE_RATES: std::ops::RangeInclusive<u32> = 8_000..=192_000;
/// The longest voice track an export builds: 4 hours at the rate the client records (24 kHz),
/// 1.3 GiB of samples.
const MAX_VOICE_SAMPLES: u128 = 4 * 3600 * 24_000;

/// The microphone over `[start, end)` as one mono clip at its own rate (silence where nothing
/// was recorded), from `(time, decoded samples)` frames.
pub(super) fn mic_voice(
    start: ReplayTime,
    end: ReplayTime,
    sample_rate: u32,
    frames: &[(ReplayTime, Vec<i16>)],
    gain: f32,
) -> Option<Voice> {
    // The rate is the replay file's own (its metadata), so a shared file
    // chose it: a rate no recording has, or a track longer than any export keeps, is left out
    // instead of allocated (4 GHz over a minute asked for a terabyte and ended the process).
    if frames.is_empty() || !VOICE_RATES.contains(&sample_rate) || end <= start {
        return None;
    }
    let length = u128::from(end - start) * u128::from(sample_rate) / 1_000_000;
    if length > MAX_VOICE_SAMPLES {
        tracing::warn!(
            seconds = (end - start) / 1_000_000,
            "export: a voice track longer than 4 hours is left out"
        );
        return None;
    }
    let length = length as usize;
    let mut samples = vec![0.0_f32; length];
    for (time, pcm) in frames {
        if *time + 1_000_000 < start || *time >= end {
            continue;
        }
        let at = (*time as i128 - start as i128) * i128::from(sample_rate) / 1_000_000;
        for (i, &s) in pcm.iter().enumerate() {
            let index = at + i as i128;
            if (0..length as i128).contains(&index) {
                samples[index as usize] = f32::from(s) / 32768.0;
            }
        }
    }
    Some(Voice {
        clip: Arc::new(Clip {
            sample_rate,
            channels: 1,
            samples,
        }),
        start_frame: 0,
        pitch: 1.0,
        gain,
        looping: false,
        end_frame: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rate is the replay file's own: an impossible one, or a range
    /// longer than any export keeps, leaves the voice out instead of allocating it (the first
    /// case asked for a terabyte, which ends the process); a real track is built as before.
    #[test]
    fn a_voice_track_from_an_impossible_rate_or_range_is_left_out() {
        let frames = vec![(0u64, vec![1000i16; 480])];
        assert!(mic_voice(0, 60_000_000, 4_000_000_000, &frames, 1.0).is_none());
        assert!(mic_voice(0, 60_000_000, 0, &frames, 1.0).is_none());
        assert!(mic_voice(0, 60_000_000, 7_999, &frames, 1.0).is_none());
        let thirty_days = 30 * 24 * 3600 * 1_000_000;
        assert!(mic_voice(0, thirty_days, 24_000, &frames, 1.0).is_none());

        let voice = mic_voice(0, 60_000_000, 24_000, &frames, 1.0).expect("a real minute");
        assert_eq!(voice.clip.sample_rate, 24_000);
        assert_eq!(voice.clip.samples.len(), 60 * 24_000);
        assert!((voice.clip.samples[0] - 1000.0 / 32768.0).abs() < 1e-6);
        assert_eq!(voice.clip.samples[480], 0.0);
    }

    fn listener_at(x: f64) -> Listener {
        Listener {
            position: DVec3::new(x, 64.0, 0.0),
            at: mc_math::Vec3::new(0.0, 0.0, -1.0),
            up: mc_math::Vec3::new(0.0, 1.0, 0.0),
        }
    }

    fn click() -> Arc<Clip> {
        Arc::new(Clip {
            sample_rate: 48_000,
            channels: 1,
            samples: vec![0.5; 4800],
        })
    }

    #[test]
    fn a_sound_fades_with_the_listeners_distance_and_pans_to_its_side() {
        let sound = |x: f64, time| PlacedSound {
            time,
            clip: click(),
            gain: 1.0,
            pitch: 1.0,
            placement: Placement {
                position: Some(DVec3::new(x, 64.0, 0.0)),
                linear: true,
                volume: 1.0,
            },
        };
        // One at the listener, one 8 blocks to the right (half volume, panned right), one 20
        // blocks away (silent: vanilla's linear falloff ends at 16).
        let sounds = vec![sound(0.0, 0), sound(8.0, 200_000), sound(20.0, 400_000)];
        let mut mix = MixSource::new(0, sounds, vec![listener_at(0.0)], Vec::new());
        let mut out = vec![0_i16; 48_000 * 2 * 6 / 10];
        mix.fill(&mut out);
        let frame = |seconds: f64| {
            let f = (seconds * 48_000.0) as usize + 100;
            (out[f * 2], out[f * 2 + 1])
        };
        let (l0, r0) = frame(0.0);
        let (l1, r1) = frame(0.2);
        let (l2, r2) = frame(0.4);
        assert!(
            (i32::from(l0) - i32::from(r0)).abs() <= 1,
            "centred: {l0} {r0}"
        );
        assert!(r1 > l1, "panned right: {l1} {r1}");
        assert!(
            i32::from(r1).max(i32::from(l1)) < i32::from(l0),
            "quieter far away"
        );
        assert_eq!((l2, r2), (0, 0), "beyond 16 blocks: silent");
    }

    /// The channels stay apart until the mix. The mix is their sum with the game at its
    /// level (here 50%); with tracks on, each track is its channel alone (game first, then the
    /// voices in order), and a track that does not exist is silence.
    #[test]
    fn channels_add_up_to_the_mix_and_each_has_its_own_track() {
        let tone = |freq: f32| {
            Arc::new(Clip {
                sample_rate: 48_000,
                channels: 1,
                samples: (0..48_000)
                    .map(|i| (i as f32 / 48_000.0 * freq * std::f32::consts::TAU).sin() * 0.2)
                    .collect(),
            })
        };
        let voice = |clip| Voice {
            clip,
            start_frame: 0,
            pitch: 1.0,
            gain: 1.0,
            looping: false,
            end_frame: None,
        };
        let game = PlacedSound {
            time: 0,
            clip: tone(440.0),
            gain: 1.0,
            pitch: 1.0,
            placement: Placement {
                position: None,
                linear: false,
                volume: 1.0,
            },
        };
        let channels = vec![
            VoiceChannel {
                voice: voice(tone(660.0)),
                track: true,
            },
            VoiceChannel {
                voice: voice(tone(990.0)),
                track: true,
            },
        ];
        let mut mix =
            MixSource::with_channels(0, vec![game], vec![listener_at(0.0)], 0.5, true, channels);
        let mut out = vec![0_i16; 4800 * 2];
        mix.fill(&mut out);
        let mut tracks = vec![vec![0_i16; out.len()]; 3];
        for (index, track) in tracks.iter_mut().enumerate() {
            mix.fill_track(index, track);
        }
        let peak = |v: &[i16]| v.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        // Centred mono at 0.2: 0.2 of full scale on each side (the mixer's pan law keeps a
        // centred voice at full level); the game at half that.
        let full = (0.2 * 32767.0) as u16;
        assert!(
            peak(&tracks[0]).abs_diff(full / 2) < 200,
            "game at 50%: {}",
            peak(&tracks[0])
        );
        assert!(
            peak(&tracks[1]).abs_diff(full) < 300,
            "mic: {}",
            peak(&tracks[1])
        );
        assert!(
            peak(&tracks[2]).abs_diff(full) < 300,
            "desktop: {}",
            peak(&tracks[2])
        );
        for i in 0..out.len() {
            let sum: i32 = tracks.iter().map(|t| i32::from(t[i])).sum();
            assert!(
                (i32::from(out[i]) - sum).abs() <= 3,
                "the mix is the sum of its channels at sample {i}: {} vs {sum}",
                out[i]
            );
        }
        let mut missing = vec![1_i16; 8];
        mix.fill_track(3, &mut missing);
        assert!(missing.iter().all(|&s| s == 0), "no fourth track: silence");
        // Negative control: without tracks nothing is kept per channel.
        let mut plain =
            MixSource::with_channels(0, Vec::new(), vec![listener_at(0.0)], 1.0, false, {
                vec![VoiceChannel {
                    voice: voice(tone(660.0)),
                    track: false,
                }]
            });
        let mut out = vec![0_i16; 960];
        plain.fill(&mut out);
        assert!(peak(&out) > 1000, "the voice is in the mix");
        let mut track = vec![1_i16; 960];
        plain.fill_track(0, &mut track);
        assert!(track.iter().all(|&s| s == 0));
    }

    #[test]
    fn the_mic_lands_at_its_time() {
        let frames = vec![
            (1_000_000_u64, vec![1000_i16; 240]),
            (1_500_000, vec![-1000; 240]),
        ];
        let voice = mic_voice(500_000, 2_000_000, 24_000, &frames, 1.0).expect("voice");
        let clip = &voice.clip;
        // 0.5 s after the range start, at 24 kHz: sample 12000.
        assert!(clip.samples[12_000] > 0.0);
        assert_eq!(clip.samples[11_999], 0.0);
        assert!(clip.samples[24_000] < 0.0);
        assert_eq!(clip.samples.len(), 36_000);
    }
}
