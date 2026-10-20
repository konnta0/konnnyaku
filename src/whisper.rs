use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineWhisperModelConfig};

/// Offline Whisper used to rescore Zipformer finals before translation.
pub struct WhisperRescorer {
    recognizer: OfflineRecognizer,
}

impl WhisperRescorer {
    /// Load a sherpa-onnx Whisper directory (tiny.en by default).
    pub fn from_dir(model_dir: &Path, num_threads: i32) -> Result<Self> {
        let encoder = find_file(
            model_dir,
            &[
                "tiny.en-encoder.int8.onnx",
                "tiny.en-encoder.onnx",
                "encoder.int8.onnx",
            ],
        )?;
        let decoder = find_file(
            model_dir,
            &[
                "tiny.en-decoder.int8.onnx",
                "tiny.en-decoder.onnx",
                "decoder.int8.onnx",
            ],
        )?;
        let tokens = find_file(model_dir, &["tiny.en-tokens.txt", "tokens.txt"])?;

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.whisper = OfflineWhisperModelConfig {
            encoder: Some(path_string(&encoder)),
            decoder: Some(path_string(&decoder)),
            language: Some("en".to_string()),
            task: Some("transcribe".to_string()),
            tail_paddings: 50,
            enable_token_timestamps: false,
            enable_segment_timestamps: false,
        };
        config.model_config.tokens = Some(path_string(&tokens));
        config.model_config.provider = Some("cpu".to_string());
        config.model_config.num_threads = num_threads.max(1);
        config.model_config.debug = false;
        config.model_config.model_type = Some("whisper".to_string());

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| anyhow!("failed to create Whisper OfflineRecognizer"))?;
        tracing::info!(dir = %model_dir.display(), "loaded Whisper rescorer");
        Ok(Self { recognizer })
    }

    /// Transcribe one utterance. Returns `None` when audio is too short.
    pub fn transcribe(&self, sample_rate: u32, samples: &[f32]) -> Result<Option<String>> {
        if sample_rate == 0 || samples.len() < (sample_rate as usize / 4) {
            return Ok(None);
        }
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(
            i32::try_from(sample_rate).context("sample rate fits i32")?,
            samples,
        );
        self.recognizer.decode(&stream);
        let text = stream
            .get_result()
            .map(|result| result.text.trim().to_string())
            .filter(|text| !text.is_empty());
        Ok(text)
    }
}

fn find_file(dir: &Path, names: &[&str]) -> Result<PathBuf> {
    for name in names {
        let path = dir.join(name);
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(anyhow!(
        "missing Whisper files in {}. Run scripts/download-asr.sh",
        dir.display()
    ))
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn transcribes_bundled_tiny_en_wav() {
        let dir = Path::new("models/asr/whisper-tiny.en");
        if !dir.join("tiny.en-encoder.int8.onnx").is_file() {
            return;
        }
        let audio = crate::audio::load_wav_mono_f32(&dir.join("test_wavs/0.wav"))
            .expect("whisper test wav");
        let rescorer = WhisperRescorer::from_dir(dir, 1).expect("load whisper");
        let text = rescorer
            .transcribe(audio.sample_rate, &audio.samples)
            .expect("transcribe")
            .expect("non-empty");
        let upper = text.to_ascii_uppercase();
        assert!(
            upper.contains("NIGHTFALL") || upper.contains("YELLOW"),
            "{text}"
        );
    }
}
