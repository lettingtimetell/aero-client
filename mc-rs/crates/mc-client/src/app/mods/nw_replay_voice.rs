//! The replay viewer's sound channels.
//!
//! A replay keeps three sound channels apart: the game's own sounds (sound events, replayed
//! through the game's audio engine by the playback), the microphone and desktop audio (both
//! recorded as 24 kHz streams, `mc_replay::mic` / `mc_replay::desktop`). While a replay is
//! watched:
//!
//! * the microphone and desktop audio play through an output of their own
//!   ([`mc_replay::voice_out::VoiceOutput`]), fed each frame with the stretch of replay time the
//!   frame covered, so they follow play, pause and seeks; at another speed they play faster or
//!   slower like a tape (outside 0.25x-4x they are silent);
//! * every channel has a volume, the Replay mod's **Game sound volume**, **Microphone volume** and
//!   **Desktop audio volume** (shared with exports), and a mute in the viewer's Sound menu (for
//!   this viewing), on top of the game's master volume. The game channel's volume can lower its
//!   sounds, not raise them past the game's own sound settings;
//! * nothing opens while no replay with a recorded stream is watched or while an export runs (an
//!   export mixes the channels into the file instead), and `MC_NO_AUDIO` keeps it closed.
//!
//! Cost outside a replay: one `Option` test a frame.

use mc_audio::SoundCategory;
use mc_mods::mods::replay::ids;
use mc_replay::voice_out::VoiceOutput;
use mc_replay::ReplayTime;

use super::super::Client;

/// The recorded streams' rate.
const RATE: u64 = mc_replay::desktop::SAMPLE_RATE as u64;
/// A step larger than this between frames is a seek, not play.
const MAX_STEP: ReplayTime = 1_000_000;
/// Silence queued ahead of the first frame after a start or a gap, against frame jitter.
const PREBUFFER_SECONDS: f64 = 0.06;
/// More than this queued means the output fell behind: start over.
const MAX_QUEUED_SECONDS: f64 = 0.35;
/// How far before a stretch a record may be pushed and still sound in it (its lead).
const LOOKBACK: ReplayTime = 1_000_000;

/// The three channels, in the viewer's and the export's order.
pub(crate) const GAME: usize = 0;
pub(crate) const MIC: usize = 1;
pub(crate) const DESKTOP: usize = 2;

/// The field on [`Client`] (`nw_re_voice`).
#[derive(Debug, Default)]
pub(crate) struct VoiceState {
    out: Option<VoiceOutput>,
    /// The replay time fed up to.
    fed_to: Option<ReplayTime>,
    /// The viewer's mutes (this viewing): game, microphone, desktop audio.
    muted: [bool; 3],
    /// The Replay mod's channel volumes (0..=2), read when the settings change.
    volumes: [f32; 3],
    seen: Option<u64>,
    /// The resampler: the last input sample, and where the next output sample falls (in input
    /// samples after it).
    prev: f32,
    phase: f64,
    mono: Vec<f32>,
    pcm: Vec<i16>,
    stereo: Vec<f32>,
    /// Tests: an output with no device, used instead of the real one.
    #[cfg(test)]
    test_out: Option<VoiceOutput>,
}

/// One voice for [`VoiceState::render_stretch`]: its records and gain.
type VoiceInput<'a> = (&'a [(ReplayTime, &'a [u8])], f32);

impl VoiceState {
    /// The output (an output with no device in tests).
    fn open_output(&mut self) -> VoiceOutput {
        #[cfg(test)]
        if let Some(out) = self.test_out.take() {
            return out;
        }
        tracing::info!("replay voice output opening");
        VoiceOutput::start()
    }

    /// Render the 24 kHz samples `first..last` of the voices (`(records, gain)` each) into
    /// `self.stereo` at the device's `rate`, played at `speed` (tape-style: faster is higher).
    /// The resampler's position carries over, so consecutive stretches join without a seam.
    fn render_stretch(
        &mut self,
        first: u64,
        last: u64,
        speed: f64,
        rate: u32,
        voices: &[VoiceInput<'_>],
    ) {
        self.stereo.clear();
        let n = last.saturating_sub(first) as usize;
        if n == 0 || rate == 0 {
            return;
        }
        let mut mono = std::mem::take(&mut self.mono);
        mono.clear();
        mono.resize(n, 0.0);
        for &(records, gain) in voices {
            if gain > 0.0 {
                add_voice(records, first, &mut mono, gain, &mut self.pcm);
            }
        }
        // `ratio` input samples per output sample.
        let ratio = RATE as f64 * speed / f64::from(rate);
        while self.phase < n as f64 {
            let i = self.phase.floor() as usize;
            let frac = (self.phase - i as f64) as f32;
            let x0 = if i == 0 { self.prev } else { mono[i - 1] };
            let x1 = mono[i];
            let value = x0 + (x1 - x0) * frac;
            self.stereo.push(value);
            self.stereo.push(value);
            self.phase += ratio;
        }
        self.phase -= n as f64;
        self.prev = mono[n - 1];
        self.mono = mono;
    }

    fn reset(&mut self) {
        if self.out.take().is_some() {
            tracing::debug!("replay voice output closed");
        }
        self.fed_to = None;
        self.prev = 0.0;
        self.phase = 0.0;
    }
}

/// Add the recorded frames among `records` (`(push time, record)`, the microphone's layout) that
/// sound in `mono`'s stretch (24 kHz sample `first` on) at `gain`. Frames that do not reach the
/// stretch are skipped before decoding.
fn add_voice(
    records: &[(ReplayTime, &[u8])],
    first: u64,
    mono: &mut [f32],
    gain: f32,
    pcm: &mut Vec<i16>,
) {
    let end = first + mono.len() as u64;
    let frame = mc_replay::desktop::FRAME_SAMPLES as u64;
    for &(time, payload) in records {
        let Some(lead) = payload.get(..4) else {
            continue;
        };
        let lead = u32::from_le_bytes([lead[0], lead[1], lead[2], lead[3]]);
        let start = time.saturating_sub(u64::from(lead)) * RATE / 1_000_000;
        if start + frame <= first || start >= end {
            continue;
        }
        pcm.clear();
        if mc_replay::mic::decode_record(payload, pcm).is_err() {
            continue;
        }
        for (i, &sample) in pcm.iter().enumerate() {
            let index = start + i as u64;
            if index >= first && index < end {
                mono[(index - first) as usize] += f32::from(sample) / 32768.0 * gain;
            }
        }
    }
}

impl Client {
    /// Once a frame (`Client::frame`): play the recorded voices of the replay
    /// being watched. One `Option` test while none is.
    pub(crate) fn nw_re_voice_frame(&mut self) {
        if !self.replay_is_active() {
            if self.nw_re_voice.fed_to.is_some() || self.nw_re_voice.out.is_some() {
                self.nw_re_voice.reset();
                self.nw_re_voice.muted = [false; 3];
            }
            return;
        }
        let (has_mic, has_desktop) = self
            .replay_meta()
            .map_or((false, false), |meta| (meta.has_mic, meta.has_desktop));
        #[cfg(test)]
        let silent = self.nw_re_voice.test_out.is_none() && self.audio.is_silent();
        #[cfg(not(test))]
        let silent = self.audio.is_silent();
        if !(has_mic || has_desktop) || silent || self.export_running() {
            self.nw_re_voice.reset();
            return;
        }
        self.nw_re_voice_sync();
        let master = self
            .audio
            .categories
            .get(SoundCategory::Master)
            .clamp(0.0, 1.0);
        let gain = |channel: usize, present: bool, state: &VoiceState| {
            if present && !state.muted[channel] {
                state.volumes[channel] * master
            } else {
                0.0
            }
        };
        let mic_gain = gain(MIC, has_mic, &self.nw_re_voice);
        let desktop_gain = gain(DESKTOP, has_desktop, &self.nw_re_voice);
        let t = self.replay_time();
        let playing = self.replay_is_playing();
        let speed = f64::from(self.replay_speed());
        let mut state = std::mem::take(&mut self.nw_re_voice);
        let Some(from) = state.fed_to else {
            state.fed_to = Some(t);
            self.nw_re_voice = state;
            return;
        };
        let jumped = t < from || t - from > MAX_STEP;
        if !playing || jumped || !(0.25..=4.0).contains(&speed) {
            if jumped {
                if let Some(out) = state.out.as_ref() {
                    out.clear();
                }
                state.prev = 0.0;
                state.phase = 0.0;
            }
            state.fed_to = Some(t);
            self.nw_re_voice = state;
            return;
        }
        let first = from * RATE / 1_000_000;
        let last = t * RATE / 1_000_000;
        if last <= first {
            // Less than one sample of replay time: wait for more (fed_to stays).
            self.nw_re_voice = state;
            return;
        }
        if state.out.is_none() && mic_gain + desktop_gain > 0.0 {
            state.out = Some(state.open_output());
        }
        let Some(rate) = state.out.as_ref().map(VoiceOutput::rate).filter(|r| *r > 0) else {
            // Still opening (or muted with nothing open): skip this stretch.
            state.fed_to = Some(t);
            self.nw_re_voice = state;
            return;
        };
        let window = (from.saturating_sub(LOOKBACK), t + LOOKBACK);
        let mic = if mic_gain > 0.0 {
            self.replay_mic_in(window.0, window.1)
        } else {
            Vec::new()
        };
        let desktop = if desktop_gain > 0.0 {
            self.replay_desktop_in(window.0, window.1)
        } else {
            Vec::new()
        };
        state.render_stretch(
            first,
            last,
            speed,
            rate,
            &[(&mic, mic_gain), (&desktop, desktop_gain)],
        );
        if let Some(out) = state.out.as_ref() {
            let queued = out.queued_frames() as f64 / f64::from(rate);
            if queued > MAX_QUEUED_SECONDS {
                out.clear();
            }
            if out.queued_frames() == 0 {
                let silence = (PREBUFFER_SECONDS * f64::from(rate)) as usize * 2;
                out.push(&vec![0.0; silence]);
            }
            out.push(&state.stereo);
        }
        state.fed_to = Some(t);
        self.nw_re_voice = state;
    }

    /// The channel volumes from the Replay mod, when the settings changed.
    fn nw_re_voice_sync(&mut self) {
        let generation = self.mods.settings.generation();
        if self.nw_re_voice.seen == Some(generation) {
            return;
        }
        self.nw_re_voice.seen = Some(generation);
        let settings = &self.mods.settings;
        let volume = |id: &str| (settings.f32("replay", id) / 100.0).clamp(0.0, 2.0);
        self.nw_re_voice.volumes = [
            volume(ids::GAME_VOLUME),
            volume(ids::MIC_VOLUME),
            volume(ids::DESKTOP_VOLUME),
        ];
    }

    /// The gain for a replayed game sound in the viewer: the game sound
    /// volume (lowering only), 0 while muted. Read per replayed sound (the playback).
    pub(crate) fn nw_re_game_gain(&mut self) -> f32 {
        self.nw_re_voice_sync();
        if self.nw_re_voice.muted[GAME] {
            0.0
        } else {
            self.nw_re_voice.volumes[GAME].min(1.0)
        }
    }

    /// The viewer's Sound menu changed a channel: its mute (this viewing) and its volume (the
    /// Replay mod's option, saved).
    pub(crate) fn nw_re_set_channel(&mut self, channel: usize, muted: bool, volume: f32) {
        let Some(slot) = self.nw_re_voice.muted.get_mut(channel) else {
            return;
        };
        *slot = muted;
        let id = match channel {
            GAME => ids::GAME_VOLUME,
            MIC => ids::MIC_VOLUME,
            _ => ids::DESKTOP_VOLUME,
        };
        let percent = (volume * 100.0).clamp(0.0, 200.0).round();
        if (self.mods.settings.f32("replay", id) - percent).abs() > 0.1 {
            self.mods.settings.set_f32("replay", id, percent);
        }
    }

    /// The channels for the viewer's Sound menu when a replay opens: `(muted, volume 0..=2)` per
    /// channel. Mutes are for one viewing, so they start off here.
    pub(crate) fn nw_re_channels(&mut self) -> [(bool, f32); 3] {
        self.nw_re_voice.muted = [false; 3];
        self.nw_re_voice_sync();
        let state = &self.nw_re_voice;
        [GAME, MIC, DESKTOP].map(|c| (state.muted[c], state.volumes[c]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One record with `lead`, a constant `value` (the codec's second frame, settled on it).
    fn record(lead: u32, value: i16) -> Vec<u8> {
        let mut state = mc_replay::mic::AdpcmState::default();
        let mut frame = Vec::new();
        let samples = vec![value; mc_replay::desktop::FRAME_SAMPLES];
        mc_replay::mic::encode_frame(&mut state, &samples, &mut frame);
        mc_replay::mic::encode_frame(&mut state, &samples, &mut frame);
        let mut out = lead.to_le_bytes().to_vec();
        out.extend_from_slice(&frame);
        out
    }

    /// `(push time, record)` for a continuous `freq` Hz tone from 0 to `seconds`, lead 0.
    fn tone_records(freq: f64, seconds: f64) -> Vec<(ReplayTime, Vec<u8>)> {
        let mut state = mc_replay::mic::AdpcmState::default();
        let mut frame = Vec::new();
        let n = mc_replay::desktop::FRAME_SAMPLES;
        let frames = (seconds * 50.0) as u64;
        (0..frames)
            .map(|f| {
                let samples: Vec<i16> = (0..n as u64)
                    .map(|i| {
                        let t = (f * n as u64 + i) as f64 / RATE as f64;
                        ((t * freq * std::f64::consts::TAU).sin() * 9000.0) as i16
                    })
                    .collect();
                mc_replay::mic::encode_frame(&mut state, &samples, &mut frame);
                let mut record = 0_u32.to_le_bytes().to_vec();
                record.extend_from_slice(&frame);
                (f * 20_000, record)
            })
            .collect()
    }

    /// Power of `freq` relative to the total (Goertzel), over mono samples at `rate`.
    fn share(samples: &[f32], rate: f64, freq: f64) -> f64 {
        let coeff = 2.0 * (std::f64::consts::TAU * freq / rate).cos();
        let (mut s1, mut s2, mut total) = (0.0_f64, 0.0_f64, 0.0_f64);
        for &x in samples {
            let x = f64::from(x);
            total += x * x;
            let s0 = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - coeff * s1 * s2) / (total * samples.len() as f64 / 2.0).max(1e-12)
    }

    /// Stretches fed frame by frame (60 fps) join without a seam: the output is as long as the
    /// time played, at the tone's pitch, with no jump between samples larger than the tone's own
    /// slope. At 2x it lasts half as long, an octave up (tape-style).
    #[test]
    fn stretches_join_seamlessly_and_speed_works_like_a_tape() {
        let owned = tone_records(900.0, 1.2);
        let records: Vec<(ReplayTime, &[u8])> =
            owned.iter().map(|(t, r)| (*t, r.as_slice())).collect();
        for (speed, expect_freq) in [(1.0, 900.0), (2.0, 1800.0)] {
            let mut state = VoiceState::default();
            let mut out = Vec::new();
            // 1 s of replay time in 60 fps steps of wall time (so `speed` x as much replay time).
            let step = (1_000_000.0 / 60.0 * speed) as u64;
            let mut from = 0_u64;
            while from + step <= 1_000_000 {
                let to = from + step;
                state.render_stretch(
                    from * RATE / 1_000_000,
                    to * RATE / 1_000_000,
                    speed,
                    48_000,
                    &[(&records, 1.0)],
                );
                out.extend(state.stereo.iter().step_by(2).copied());
                from = to;
            }
            let played = from as f64 / 1e6 / speed;
            let expected = played * 48_000.0;
            assert!(
                (out.len() as f64 - expected).abs() < 4.0,
                "speed {speed}: {} samples for {expected}",
                out.len()
            );
            let body = &out[2_400..out.len() - 2_400];
            let peak = body.iter().fold(0.0_f32, |m, s| m.max(s.abs()));
            let slope = body
                .windows(2)
                .fold(0.0_f32, |m, w| m.max((w[1] - w[0]).abs()));
            // A gap would jump by about the whole amplitude (4-8x the tone's own steepest step);
            // the codec's noise stays well under 2x.
            let tone_slope = peak * (std::f32::consts::TAU * expect_freq as f32 / 48_000.0);
            assert!(peak > 0.2, "speed {speed}: the tone is there ({peak})");
            assert!(
                slope < tone_slope * 2.0,
                "speed {speed}: a seam ({slope} vs the tone's {tone_slope})"
            );
            let own = share(body, 48_000.0, expect_freq);
            let other = share(body, 48_000.0, if speed > 1.5 { 900.0 } else { 1800.0 });
            assert!(
                own > 0.5 && own > 20.0 * other,
                "speed {speed}: pitch {expect_freq} Hz ({own} vs {other})"
            );
        }
    }

    /// A frame lands where its time minus its lead says, at its gain; frames that miss the
    /// stretch are skipped, and the stretch's edges cut frames that straddle them.
    #[test]
    fn voices_land_at_their_time_at_their_gain() {
        // A frame pushed at 1.02 s with a 20 ms lead: it sounds from 1.000 s (24 kHz sample
        // 24 000) for 480 samples.
        let a = record(20_000, 8_000);
        // One far outside the stretch.
        let b = record(0, 8_000);
        let records = vec![(1_020_000, a.as_slice()), (5_000_000, b.as_slice())];
        // The stretch: 0.99 s .. 1.01 s (samples 23 760..24 240).
        let mut mono = vec![0.0_f32; 480];
        let mut pcm = Vec::new();
        add_voice(&records, 23_760, &mut mono, 0.5, &mut pcm);
        assert!(
            mono[..240].iter().all(|&s| s == 0.0),
            "silence before the frame"
        );
        let level = mono[300];
        let expected = 8_000.0 / 32768.0 * 0.5;
        assert!((level - expected).abs() < 0.02, "{level} vs {expected}");
        // Negative control: at gain 0 nothing is added.
        let mut quiet = vec![0.0_f32; 480];
        add_voice(&records, 23_760, &mut quiet, 0.0, &mut pcm);
        assert!(quiet.iter().all(|&s| s == 0.0));
    }
}
