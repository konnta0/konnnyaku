use std::collections::VecDeque;

use crate::audio::samples_for_ms;

/// Sample rate of mixed captions. Sherpa resamples this to 16 kHz.
pub const MIX_RATE: u32 = 48_000;

const SKEW_MS: u32 = 200;
const PENDING_MS: u32 = 2_000;

/// Which physical capture a chunk came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixPort {
    /// Default microphone.
    Input,
    /// App or system playback.
    Output,
}

/// Mixes microphone and playback into one PCM stream.
///
/// Each side keeps its own clock. When both are enabled, samples are summed
/// only while both have audio, and a side that runs more than 200 ms ahead is
/// skipped so a missing device cannot stall the other forever.
pub struct AudioMix {
    rate: u32,
    hear_input: bool,
    hear_output: bool,
    input: MixSide,
    output: MixSide,
}

impl AudioMix {
    /// Empty mixer. Nothing is heard until [`Self::set_enabled`].
    pub fn new(rate: u32) -> Self {
        Self {
            rate,
            hear_input: false,
            hear_output: false,
            input: MixSide::new(rate),
            output: MixSide::new(rate),
        }
    }

    /// Turn each side on or off. A side that turns off discards buffered audio.
    pub fn set_enabled(&mut self, input: bool, output: bool) {
        if self.hear_input && !input {
            self.input.clear();
        }
        if self.hear_output && !output {
            self.output.clear();
        }
        self.hear_input = input;
        self.hear_output = output;
    }

    /// Resample `samples` onto this mixer's rate and queue them.
    pub fn push(&mut self, port: MixPort, sample_rate: u32, samples: &[f32]) {
        if samples.is_empty() || sample_rate == 0 {
            return;
        }
        let enabled = match port {
            MixPort::Input => self.hear_input,
            MixPort::Output => self.hear_output,
        };
        if !enabled {
            return;
        }
        let side = match port {
            MixPort::Input => &mut self.input,
            MixPort::Output => &mut self.output,
        };
        side.push(sample_rate, samples);
        let cap = samples_for_ms(self.rate, PENDING_MS).max(1);
        if side.pending.len() > cap {
            let drop = side.pending.len() - cap;
            side.pending.drain(..drop);
        }
    }

    /// Take every sample that can be recognized now, at [`MIX_RATE`].
    pub fn drain(&mut self) -> Vec<f32> {
        match (self.hear_input, self.hear_output) {
            (false, false) => Vec::new(),
            (true, false) => self.input.take_all(),
            (false, true) => self.output.take_all(),
            (true, true) => self.drain_both(),
        }
    }

    fn drain_both(&mut self) -> Vec<f32> {
        let skew = samples_for_ms(self.rate, SKEW_MS).max(1);
        let in_len = self.input.pending.len();
        let out_len = self.output.pending.len();
        if in_len == 0 && out_len == 0 {
            return Vec::new();
        }
        if in_len == 0 && out_len > skew {
            return self.output.take(out_len - skew);
        }
        if out_len == 0 && in_len > skew {
            return self.input.take(in_len - skew);
        }
        if in_len > out_len + skew {
            let drop = in_len - (out_len + skew);
            self.input.pending.drain(..drop);
        } else if out_len > in_len + skew {
            let drop = out_len - (in_len + skew);
            self.output.pending.drain(..drop);
        }
        let n = self.input.pending.len().min(self.output.pending.len());
        mix_sum(&mut self.input.pending, &mut self.output.pending, n)
    }
}

struct MixSide {
    resampler: LinearResampler,
    pending: VecDeque<f32>,
}

impl MixSide {
    fn new(rate: u32) -> Self {
        Self {
            resampler: LinearResampler::new(rate),
            pending: VecDeque::new(),
        }
    }

    fn clear(&mut self) {
        self.pending.clear();
        self.resampler.reset();
    }

    fn push(&mut self, sample_rate: u32, samples: &[f32]) {
        self.resampler
            .resample(sample_rate, samples, &mut self.pending);
    }

    fn take_all(&mut self) -> Vec<f32> {
        self.pending.drain(..).collect()
    }

    fn take(&mut self, n: usize) -> Vec<f32> {
        let n = n.min(self.pending.len());
        self.pending.drain(..n).collect()
    }
}

fn mix_sum(left: &mut VecDeque<f32>, right: &mut VecDeque<f32>, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let a = left.pop_front().unwrap_or(0.0);
        let b = right.pop_front().unwrap_or(0.0);
        out.push((a + b).clamp(-1.0, 1.0));
    }
    out
}

struct LinearResampler {
    to: u32,
    from: u32,
    phase: f64,
    prev: f32,
}

impl LinearResampler {
    fn new(to: u32) -> Self {
        Self {
            to,
            from: to,
            phase: 0.0,
            prev: 0.0,
        }
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.prev = 0.0;
        self.from = self.to;
    }

    fn resample(&mut self, from: u32, input: &[f32], out: &mut VecDeque<f32>) {
        if input.is_empty() || from == 0 || self.to == 0 {
            return;
        }
        if self.from != from {
            self.from = from;
            self.phase = 0.0;
            self.prev = 0.0;
        }
        if self.from == self.to {
            out.extend(input.iter().copied());
            return;
        }
        let step = f64::from(self.from) / f64::from(self.to);
        let mut pos = self.phase;
        let n = input.len();
        while pos < n as f64 {
            let idx = pos.floor() as isize;
            let frac = (pos - idx as f64) as f32;
            let (left, right) = if idx < 0 {
                (self.prev, input[0])
            } else {
                let i = idx as usize;
                if i + 1 >= n {
                    break;
                }
                (input[i], input[i + 1])
            };
            out.push_back(left + (right - left) * frac);
            pos += step;
        }
        self.prev = input[n - 1];
        self.phase = pos - n as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_both_sides_and_clamps() {
        let mut mix = AudioMix::new(MIX_RATE);
        mix.set_enabled(true, true);
        mix.push(MixPort::Input, MIX_RATE, &[0.25, 0.25, 1.0]);
        mix.push(MixPort::Output, MIX_RATE, &[0.25, 0.50, 1.0]);
        assert_eq!(mix.drain(), vec![0.5, 0.75, 1.0]);
        assert!(mix.drain().is_empty());
    }

    #[test]
    fn disabled_side_is_ignored_and_not_replayed() {
        let mut mix = AudioMix::new(MIX_RATE);
        mix.set_enabled(true, true);
        mix.push(MixPort::Input, MIX_RATE, &[0.2, 0.2]);
        mix.push(MixPort::Output, MIX_RATE, &[0.3, 0.3]);
        mix.set_enabled(true, false);
        assert_eq!(mix.drain(), vec![0.2, 0.2]);
        mix.set_enabled(true, true);
        mix.push(MixPort::Output, MIX_RATE, &[0.4]);
        assert!(mix.drain().is_empty());
    }

    #[test]
    fn both_off_drops_audio() {
        let mut mix = AudioMix::new(MIX_RATE);
        mix.set_enabled(false, false);
        mix.push(MixPort::Input, MIX_RATE, &[1.0, 1.0]);
        mix.push(MixPort::Output, MIX_RATE, &[1.0, 1.0]);
        assert!(mix.drain().is_empty());
    }

    #[test]
    fn missing_side_does_not_stall_the_other() {
        let mut mix = AudioMix::new(MIX_RATE);
        mix.set_enabled(true, true);
        let skew = samples_for_ms(MIX_RATE, SKEW_MS);
        let audio = vec![0.5; skew + 32];
        mix.push(MixPort::Output, MIX_RATE, &audio);
        let drained = mix.drain();
        assert_eq!(drained.len(), 32);
        assert!(drained.iter().all(|sample| *sample == 0.5));
        assert!(mix.drain().is_empty());
    }

    #[test]
    fn resamples_a_constant_signal() {
        let mut mix = AudioMix::new(MIX_RATE);
        mix.set_enabled(true, false);
        let input = vec![0.25; 160];
        mix.push(MixPort::Input, 16_000, &input);
        let drained = mix.drain();
        assert!(drained.len() > 400 && drained.len() < 480);
        assert!(drained.iter().all(|sample| (*sample - 0.25).abs() < 0.001));
    }
}
