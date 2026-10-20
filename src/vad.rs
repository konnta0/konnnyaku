use std::path::Path;

use anyhow::{Result, anyhow};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

const VAD_SAMPLE_RATE: u32 = 16_000;
const WINDOW_SIZE: usize = 512;
const HANGOVER_MS: u32 = 400;
const BUFFER_SECONDS: f32 = 30.0;

/// Whether the current capture chunk should be treated as speech.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpeechDecision {
    /// Speech or hangover after speech.
    pub active: bool,
    /// Transitioned from speech to non-speech on this chunk.
    pub ended: bool,
}

/// Silero VAD gate that keeps Zipformer off music and BGM.
pub struct SpeechGate {
    vad: VoiceActivityDetector,
    remainder: Vec<f32>,
    hangover_left: usize,
    hangover_samples: usize,
    was_active: bool,
}

impl SpeechGate {
    /// Load Silero from `model_path` if the file exists.
    pub fn from_path(model_path: &Path) -> Result<Self> {
        if !model_path.is_file() {
            return Err(anyhow!(
                "missing VAD model at {}. Run scripts/download-asr.sh",
                model_path.display()
            ));
        }

        let mut config = VadModelConfig::default();
        config.silero_vad = SileroVadModelConfig {
            model: Some(path_string(model_path)),
            threshold: 0.5,
            min_silence_duration: 0.3,
            min_speech_duration: 0.15,
            window_size: i32::try_from(WINDOW_SIZE).expect("window fits i32"),
            max_speech_duration: 20.0,
        };
        config.sample_rate = i32::try_from(VAD_SAMPLE_RATE).expect("rate fits i32");
        config.num_threads = 1;
        config.provider = Some("cpu".to_string());
        config.debug = false;

        let vad = VoiceActivityDetector::create(&config, BUFFER_SECONDS)
            .ok_or_else(|| anyhow!("failed to create VoiceActivityDetector"))?;

        tracing::info!(path = %model_path.display(), "loaded Silero VAD");
        Ok(Self {
            vad,
            remainder: Vec::new(),
            hangover_left: 0,
            hangover_samples: samples_for_ms(VAD_SAMPLE_RATE, HANGOVER_MS),
            was_active: false,
        })
    }

    /// Feed a capture chunk (any sample rate) and return the speech decision.
    pub fn push(&mut self, sample_rate: u32, samples: &[f32]) -> SpeechDecision {
        let resampled = downsample_to_16k(sample_rate, samples);
        self.remainder.extend_from_slice(&resampled);

        while self.remainder.len() >= WINDOW_SIZE {
            let window: Vec<f32> = self.remainder.drain(..WINDOW_SIZE).collect();
            self.vad.accept_waveform(&window);
            if self.vad.detected() {
                self.hangover_left = self.hangover_samples;
            }
            drain_segments(&self.vad);
        }

        if !self.vad.detected() {
            self.hangover_left = self.hangover_left.saturating_sub(resampled.len());
        }

        let active = self.vad.detected() || self.hangover_left > 0;
        let ended = self.was_active && !active;
        self.was_active = active;
        SpeechDecision { active, ended }
    }

    /// Flush trailing speech at capture stop.
    pub fn flush(&mut self) -> SpeechDecision {
        if !self.remainder.is_empty() {
            if self.remainder.len() < WINDOW_SIZE {
                self.remainder.resize(WINDOW_SIZE, 0.0);
            }
            let rest = std::mem::take(&mut self.remainder);
            for window in rest.chunks(WINDOW_SIZE) {
                self.vad.accept_waveform(window);
            }
        }
        self.vad.flush();
        drain_segments(&self.vad);
        let ended = self.was_active;
        self.was_active = false;
        self.hangover_left = 0;
        SpeechDecision {
            active: false,
            ended,
        }
    }
}

fn drain_segments(vad: &VoiceActivityDetector) {
    while !vad.is_empty() {
        vad.pop();
    }
}

fn downsample_to_16k(sample_rate: u32, samples: &[f32]) -> Vec<f32> {
    if sample_rate == VAD_SAMPLE_RATE {
        return samples.to_vec();
    }
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    let ratio = f64::from(sample_rate) / f64::from(VAD_SAMPLE_RATE);
    let out_len = ((samples.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    let last = samples.len() - 1;
    for i in 0..out_len {
        let src = i as f64 * ratio;
        let i0 = (src.floor() as usize).min(last);
        let i1 = (i0 + 1).min(last);
        let frac = (src - i0 as f64) as f32;
        out.push(samples[i0].mul_add(1.0 - frac, samples[i1] * frac));
    }
    out
}

fn samples_for_ms(sample_rate: u32, ms: u32) -> usize {
    (u64::from(sample_rate) * u64::from(ms) / 1000) as usize
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn silence_is_not_speech() {
        let path = Path::new("models/asr/silero_vad.onnx");
        if !path.is_file() {
            return;
        }
        let mut gate = SpeechGate::from_path(path).expect("load vad");
        let silence = vec![0.0f32; 16_000];
        let decision = gate.push(16_000, &silence);
        assert!(!decision.active);
        assert!(!decision.ended);
    }
}
