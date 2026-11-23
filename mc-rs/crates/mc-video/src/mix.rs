//! The offline audio mixer for exports.
//!
//! Exports cannot record the speakers: they render faster or slower than real time. Instead the
//! client lists every sound that played (its decoded samples, when it started, its pitch and
//! gain) and this mixer renders the soundtrack sample by sample, asking the client for each
//! voice's distance gain and pan at every block, so a sound fades and pans as the export camera
//! moves, exactly as it did live.
//!
//! The resampler and the pan law are kira's (the game's own mixer): 4-point Hermite
//! interpolation, and constant-power panning normalised so a centred sound is unchanged. A
//! look-ahead peak limiter keeps loud moments (explosions, many hits at once) from clipping.

use std::sync::Arc;

/// Output frames between spatial updates (5.3 ms at 48 kHz). Gains ramp linearly inside a block.
pub const BLOCK: usize = 256;

/// A decoded sound, any rate, mono or stereo.
#[derive(Debug, Clone, PartialEq)]
pub struct Clip {
    /// Samples per second.
    pub sample_rate: u32,
    /// 1 or 2.
    pub channels: u8,
    /// Interleaved samples in -1..=1.
    pub samples: Vec<f32>,
}

impl Clip {
    /// From interleaved 16-bit PCM (what `mc_audio::ogg::decode_all` returns).
    #[must_use]
    pub fn from_i16(sample_rate: u32, channels: u8, samples: &[i16]) -> Self {
        Self {
            sample_rate,
            channels: channels.clamp(1, 2),
            samples: samples.iter().map(|&s| f32::from(s) / 32768.0).collect(),
        }
    }

    /// Sample frames (per channel).
    #[must_use]
    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.channels.max(1))
    }

    /// One frame as (left, right); mono is duplicated, outside the clip is silence.
    #[inline]
    fn frame(&self, index: i64, looping: bool) -> (f32, f32) {
        let frames = self.frames() as i64;
        if frames == 0 {
            return (0.0, 0.0);
        }
        let index = if looping {
            index.rem_euclid(frames)
        } else if index < 0 || index >= frames {
            return (0.0, 0.0);
        } else {
            index
        };
        let i = index as usize;
        if self.channels == 1 {
            let s = self.samples[i];
            (s, s)
        } else {
            (self.samples[i * 2], self.samples[i * 2 + 1])
        }
    }
}

/// One sound playing in the mix.
#[derive(Debug, Clone)]
pub struct Voice {
    /// What it plays.
    pub clip: Arc<Clip>,
    /// The output frame (at the mixer's rate) where it starts. Negative: it started before the
    /// exported range, and the range opens partway through it.
    pub start_frame: i64,
    /// Playback rate: 1 is normal, 2 an octave up and twice as fast (vanilla's 0.5..=2).
    pub pitch: f32,
    /// Gain before distance attenuation (sound volume x category volume).
    pub gain: f32,
    /// Whether it loops until `end_frame`.
    pub looping: bool,
    /// The output frame it is stopped at, if it was stopped early (or for a loop, at all).
    pub end_frame: Option<i64>,
}

impl Voice {
    /// Source frames advanced per output frame.
    fn step(&self, out_rate: u32) -> f64 {
        f64::from(self.clip.sample_rate) * f64::from(self.pitch.max(0.01)) / f64::from(out_rate)
    }

    /// The first output frame after the voice has finished.
    fn end(&self, out_rate: u32) -> i64 {
        let natural = if self.looping {
            i64::MAX
        } else {
            let length = (self.clip.frames() as f64 / self.step(out_rate)).ceil() as i64;
            self.start_frame.saturating_add(length + 2)
        };
        self.end_frame.map_or(natural, |end| end.min(natural))
    }
}

/// Kira's constant-power pan: -1 hard left, 1 hard right, 0 unchanged.
#[inline]
#[must_use]
pub fn pan_gains(pan: f32) -> (f32, f32) {
    if pan == 0.0 {
        return (1.0, 1.0);
    }
    let mix = (pan.clamp(-1.0, 1.0) + 1.0) * 0.5;
    (
        (1.0 - mix).sqrt() * std::f32::consts::SQRT_2,
        mix.sqrt() * std::f32::consts::SQRT_2,
    )
}

/// Kira's 4-point Hermite interpolation.
#[inline]
fn hermite(previous: f32, current: f32, next_1: f32, next_2: f32, fraction: f32) -> f32 {
    let c0 = current;
    let c1 = (next_1 - previous) * 0.5;
    let c2 = previous - current * 2.5 + next_1 * 2.0 - next_2 * 0.5;
    let c3 = (next_2 - previous) * 0.5 + (current - next_1) * 1.5;
    ((c3 * fraction + c2) * fraction + c1) * fraction + c0
}

/// Renders a list of voices into stereo float blocks.
#[derive(Debug, Clone)]
pub struct OfflineMixer {
    sample_rate: u32,
    voices: Vec<Voice>,
    /// `voices` sorted by start: the next one to become active.
    next: usize,
    /// Indices of voices that may still sound.
    active: Vec<usize>,
    /// Per-voice (gain, pan) at the start of the current block, for ramping.
    last: Vec<Option<(f32, f32)>>,
}

impl OfflineMixer {
    /// A mixer producing `sample_rate` Hz stereo.
    #[must_use]
    pub fn new(sample_rate: u32, mut voices: Vec<Voice>) -> Self {
        voices.sort_by_key(|v| v.start_frame);
        let count = voices.len();
        Self {
            sample_rate,
            voices,
            next: 0,
            active: Vec::new(),
            last: vec![None; count],
        }
    }

    /// The voices, sorted by start (indices passed to the spatial callback refer to this).
    #[must_use]
    pub fn voices(&self) -> &[Voice] {
        &self.voices
    }

    /// Output sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Mix `out.len() / 2` stereo frames starting at output frame `start` into `out` (overwritten,
    /// interleaved left/right). Calls must be consecutive (each starting where the last ended).
    ///
    /// `spatial(voice, frame)` returns the voice's distance gain (0..=1) and pan (-1..=1) at that
    /// output frame; it is called once per voice per [`BLOCK`] and the values ramp in between.
    pub fn render(
        &mut self,
        start: i64,
        out: &mut [f32],
        mut spatial: impl FnMut(usize, i64) -> (f32, f32),
    ) {
        out.fill(0.0);
        let frames = (out.len() / 2) as i64;
        let mut block_start = 0_i64;
        while block_start < frames {
            let block_len = (frames - block_start).min(BLOCK as i64);
            let at = start + block_start;
            let block_end = at + block_len;
            // Wake voices that start before this block ends.
            while self.next < self.voices.len() && self.voices[self.next].start_frame < block_end {
                self.active.push(self.next);
                self.next += 1;
            }
            let rate = self.sample_rate;
            let voices = &self.voices;
            let last = &mut self.last;
            self.active.retain(|&v| voices[v].end(rate) > at);
            for &v in &self.active {
                let voice = &voices[v];
                let end_frame = voice.end(rate);
                let from = at.max(voice.start_frame);
                let to = block_end.min(end_frame);
                if from >= to {
                    continue;
                }
                let begin = last[v].unwrap_or_else(|| spatial(v, from));
                let finish = spatial(v, block_end);
                last[v] = Some(finish);
                let step = voice.step(rate);
                let span = (block_end - at) as f32;
                for f in from..to {
                    let t = (f - at) as f32 / span;
                    let gain = voice.gain * (begin.0 + (finish.0 - begin.0) * t);
                    if gain == 0.0 {
                        continue;
                    }
                    let pan = begin.1 + (finish.1 - begin.1) * t;
                    let position = (f - voice.start_frame) as f64 * step;
                    let index = position.floor() as i64;
                    let fraction = (position - index as f64) as f32;
                    let clip = &voice.clip;
                    let p = clip.frame(index - 1, voice.looping);
                    let c = clip.frame(index, voice.looping);
                    let n1 = clip.frame(index + 1, voice.looping);
                    let n2 = clip.frame(index + 2, voice.looping);
                    let left = hermite(p.0, c.0, n1.0, n2.0, fraction);
                    let right = hermite(p.1, c.1, n1.1, n2.1, fraction);
                    let (pl, pr) = pan_gains(pan);
                    let o = ((f - start) * 2) as usize;
                    out[o] += left * pl * gain;
                    out[o + 1] += right * pr * gain;
                }
            }
            block_start += block_len;
        }
    }
}

/// A look-ahead peak limiter that turns float stereo into 16-bit PCM without clipping.
#[derive(Debug, Clone)]
pub struct Limiter {
    gain: f32,
    ceiling: f32,
    release_per_block: f32,
}

impl Limiter {
    /// A limiter that keeps peaks under `ceiling` (0..1 of full scale), releasing over about 80 ms
    /// at 48 kHz.
    #[must_use]
    pub fn new(ceiling: f32) -> Self {
        Self {
            gain: 1.0,
            ceiling,
            release_per_block: 0.03,
        }
    }

    /// Convert `input` (interleaved stereo float) into `out` (same length), limiting per
    /// [`BLOCK`]: the gain drops at once to fit a block's peak and recovers slowly afterwards.
    pub fn process(&mut self, input: &[f32], out: &mut [i16]) {
        assert_eq!(input.len(), out.len(), "limiter buffers differ in length");
        for (src, dst) in input.chunks(BLOCK * 2).zip(out.chunks_mut(BLOCK * 2)) {
            let peak = src.iter().fold(0.0_f32, |m, &s| m.max(s.abs()));
            let wanted = if peak > self.ceiling {
                self.ceiling / peak
            } else {
                1.0
            };
            let start = self.gain;
            let target = if wanted < start {
                wanted
            } else {
                (start + self.release_per_block).min(wanted)
            };
            let n = (src.len() / 2).max(1) as f32;
            for (i, (s, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
                // Attack lands at the block start (look-ahead); release ramps across it.
                let g = if target < start {
                    target
                } else {
                    start + (target - start) * ((i / 2) as f32 / n)
                };
                *d = (s * g * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
            }
            self.gain = target;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f32, seconds: f32, channels: u8) -> Clip {
        let frames = (rate as f32 * seconds) as usize;
        let mut samples = Vec::with_capacity(frames * usize::from(channels));
        for i in 0..frames {
            let s = (i as f32 / rate as f32 * hz * std::f32::consts::TAU).sin() * 0.5;
            for _ in 0..channels {
                samples.push(s);
            }
        }
        Clip {
            sample_rate: rate,
            channels,
            samples,
        }
    }

    fn zero_crossings(left: &[f32]) -> usize {
        left.windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count()
    }

    #[test]
    fn a_voice_plays_at_its_start_and_stops_at_its_end() {
        let clip = Arc::new(tone(48_000, 440.0, 0.1, 1));
        let voice = Voice {
            clip,
            start_frame: 1000,
            pitch: 1.0,
            gain: 1.0,
            looping: false,
            end_frame: None,
        };
        let mut mixer = OfflineMixer::new(48_000, vec![voice]);
        let mut out = vec![0.0_f32; 48_000 * 2];
        mixer.render(0, &mut out, |_, _| (1.0, 0.0));
        assert!(
            out[..1000 * 2].iter().all(|&s| s == 0.0),
            "silence before the start"
        );
        let sounding = &out[1000 * 2..(1000 + 4800) * 2];
        assert!(sounding.iter().any(|&s| s.abs() > 0.4));
        assert!(
            out[(1000 + 4810) * 2..].iter().all(|&s| s == 0.0),
            "silence after the end"
        );
    }

    #[test]
    fn pitch_two_plays_an_octave_up_in_half_the_time() {
        let clip = Arc::new(tone(44_100, 300.0, 0.5, 1));
        let make = |pitch| Voice {
            clip: clip.clone(),
            start_frame: 0,
            pitch,
            gain: 1.0,
            looping: false,
            end_frame: None,
        };
        let mut normal = OfflineMixer::new(48_000, vec![make(1.0)]);
        let mut fast = OfflineMixer::new(48_000, vec![make(2.0)]);
        let mut a = vec![0.0_f32; 12_000 * 2];
        let mut b = vec![0.0_f32; 12_000 * 2];
        normal.render(0, &mut a, |_, _| (1.0, 0.0));
        fast.render(0, &mut b, |_, _| (1.0, 0.0));
        let left = |v: &[f32]| v.iter().step_by(2).copied().collect::<Vec<_>>();
        let (za, zb) = (zero_crossings(&left(&a)), zero_crossings(&left(&b)));
        // 0.25 s: 300 Hz has ~150 crossings, 600 Hz ~300.
        assert!((145..=155).contains(&za), "{za}");
        assert!((295..=305).contains(&zb), "{zb}");
    }

    #[test]
    fn pan_law_matches_kira() {
        assert_eq!(pan_gains(0.0), (1.0, 1.0));
        let (l, r) = pan_gains(-1.0);
        assert!((l - std::f32::consts::SQRT_2).abs() < 1e-6 && r.abs() < 1e-6);
        let (l, r) = pan_gains(0.5);
        assert!((l * l + r * r - 2.0).abs() < 1e-5, "constant power");
        assert!(r > l);
    }

    #[test]
    fn spatial_gain_ramps_and_is_asked_per_block() {
        let clip = Arc::new(Clip {
            sample_rate: 48_000,
            channels: 1,
            samples: vec![0.5; 48_000],
        });
        let voice = Voice {
            clip,
            start_frame: 0,
            pitch: 1.0,
            gain: 1.0,
            looping: false,
            end_frame: None,
        };
        let mut mixer = OfflineMixer::new(48_000, vec![voice]);
        let mut out = vec![0.0_f32; BLOCK * 4 * 2];
        let mut calls = 0;
        // The gain falls from 1 to 0 over the four blocks.
        mixer.render(0, &mut out, |_, frame| {
            calls += 1;
            (1.0 - frame as f32 / (BLOCK * 4) as f32, 0.0)
        });
        assert_eq!(
            calls, 5,
            "one call to open the voice, then one per block end"
        );
        let first = out[2];
        let middle = out[BLOCK * 2 * 2];
        let last = out[out.len() - 2];
        assert!(first > middle && middle > last, "{first} {middle} {last}");
        assert!((middle - 0.25).abs() < 0.01, "{middle}");
    }

    #[test]
    fn rendering_in_pieces_equals_rendering_at_once() {
        let clip = Arc::new(tone(22_050, 523.0, 0.3, 2));
        let voices = vec![
            Voice {
                clip: clip.clone(),
                start_frame: -500,
                pitch: 1.3,
                gain: 0.7,
                looping: false,
                end_frame: None,
            },
            Voice {
                clip,
                start_frame: 3000,
                pitch: 0.8,
                gain: 0.5,
                looping: true,
                end_frame: Some(9000),
            },
        ];
        let spatial = |v: usize, f: i64| (0.5 + 0.4 * ((f as f32) / 7000.0).sin(), v as f32 * 0.3);
        let mut whole = OfflineMixer::new(48_000, voices.clone());
        let mut all = vec![0.0_f32; 10_000 * 2];
        whole.render(0, &mut all, spatial);
        let mut pieces = OfflineMixer::new(48_000, voices);
        let mut joined = Vec::new();
        let mut at = 0_i64;
        for len in [
            BLOCK as i64 * 3,
            BLOCK as i64 * 10,
            10_000 - BLOCK as i64 * 13,
        ] {
            let mut part = vec![0.0_f32; len as usize * 2];
            pieces.render(at, &mut part, spatial);
            joined.extend_from_slice(&part);
            at += len;
        }
        let worst = all
            .iter()
            .zip(&joined)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 1e-6, "{worst}");
    }

    #[test]
    fn a_looping_voice_stops_at_its_end_frame() {
        let clip = Arc::new(tone(22_050, 523.0, 0.05, 1));
        let voice = Voice {
            clip,
            start_frame: 3000,
            pitch: 0.8,
            gain: 0.5,
            looping: true,
            end_frame: Some(9000),
        };
        let mut mixer = OfflineMixer::new(48_000, vec![voice]);
        let mut out = vec![0.0_f32; 12_000 * 2];
        mixer.render(0, &mut out, |_, _| (1.0, 0.0));
        assert!(
            out[(8000 * 2)..(9000 * 2)].iter().any(|&s| s.abs() > 0.1),
            "still looping"
        );
        assert!(out[9000 * 2..].iter().all(|&s| s == 0.0), "stopped");
    }

    #[test]
    fn limiter_never_clips_and_leaves_quiet_audio_alone() {
        let mut limiter = Limiter::new(0.95);
        let loud: Vec<f32> = (0..4096)
            .map(|i| if i % 2 == 0 { 3.0 } else { -3.0 })
            .collect();
        let mut out = vec![0_i16; loud.len()];
        limiter.process(&loud, &mut out);
        assert!(
            out.iter().all(|&s| i32::from(s).abs() <= 31_200),
            "peak held under the ceiling"
        );
        let mut quiet_limiter = Limiter::new(0.95);
        let quiet: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin() * 0.3).collect();
        let mut q = vec![0_i16; quiet.len()];
        quiet_limiter.process(&quiet, &mut q);
        for (s, d) in quiet.iter().zip(&q) {
            assert!((s * 32767.0 - f32::from(*d)).abs() <= 1.0);
        }
    }
}
