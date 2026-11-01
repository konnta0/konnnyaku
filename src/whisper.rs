use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig,
    OfflineWhisperModelConfig,
};

/// Offline English model that rescores Zipformer finals before translation.
pub struct WhisperRescorer {
    recognizer: OfflineRecognizer,
}

impl WhisperRescorer {
    /// Load a sherpa-onnx Whisper directory, or a Parakeet directory that contains a joiner.
    pub fn from_dir(model_dir: &Path, num_threads: i32) -> Result<Self> {
        if find_onnx(model_dir, "joiner").is_ok() {
            return Self::parakeet(model_dir, num_threads);
        }
        Self::whisper(model_dir, num_threads)
    }

    fn whisper(model_dir: &Path, num_threads: i32) -> Result<Self> {
        let encoder = find_onnx(model_dir, "encoder")?;
        let decoder = find_onnx(model_dir, "decoder")?;
        let tokens = find_tokens(model_dir)?;
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

    fn parakeet(model_dir: &Path, num_threads: i32) -> Result<Self> {
        let encoder = find_onnx(model_dir, "encoder")?;
        let decoder = find_onnx(model_dir, "decoder")?;
        let joiner = find_onnx(model_dir, "joiner")?;
        let tokens = find_tokens(model_dir)?;
        let mut config = OfflineRecognizerConfig::default();
        config.model_config.transducer = OfflineTransducerModelConfig {
            encoder: Some(path_string(&encoder)),
            decoder: Some(path_string(&decoder)),
            joiner: Some(path_string(&joiner)),
        };
        config.model_config.tokens = Some(path_string(&tokens));
        config.model_config.provider = Some("cpu".to_string());
        config.model_config.num_threads = num_threads.max(1);
        config.model_config.debug = false;
        config.model_config.model_type = Some("nemo_transducer".to_string());
        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| anyhow!("failed to create Parakeet OfflineRecognizer"))?;
        tracing::info!(dir = %model_dir.display(), "loaded Parakeet rescorer");
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

fn find_onnx(dir: &Path, role: &str) -> Result<PathBuf> {
    let mut int8 = None;
    let mut full = None;
    let entries = std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))?;
    let dashed = format!("-{role}.");
    let bare = format!("{role}.");
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.ends_with(".onnx") {
            continue;
        }
        if !name.contains(&dashed) && !name.starts_with(&bare) {
            continue;
        }
        if name.contains(".int8.onnx") {
            int8 = Some(entry.path());
        } else if full.is_none() {
            full = Some(entry.path());
        }
    }
    int8.or(full).ok_or_else(|| {
        anyhow!(
            "missing {role} onnx in {}. Run scripts/download-asr.sh",
            dir.display()
        )
    })
}

fn find_tokens(dir: &Path) -> Result<PathBuf> {
    let mut preferred = None;
    let mut plain = None;
    let entries = std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name == "tokens.txt" {
            plain = Some(entry.path());
        } else if name.ends_with("-tokens.txt") {
            preferred = Some(entry.path());
        }
    }
    preferred.or(plain).ok_or_else(|| {
        anyhow!(
            "missing tokens.txt in {}. Run scripts/download-asr.sh",
            dir.display()
        )
    })
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::Instant;

    fn transcribe_wav(model_dir: &Path, wav: &Path) -> String {
        let audio = crate::audio::load_wav_mono_f32(wav).expect("wav");
        let rescorer = WhisperRescorer::from_dir(model_dir, 2).expect("load rescorer");
        let started = Instant::now();
        let text = rescorer
            .transcribe(audio.sample_rate, &audio.samples)
            .expect("transcribe")
            .expect("non-empty");
        eprintln!(
            "{} {} ms: {text}",
            model_dir.display(),
            started.elapsed().as_millis()
        );
        text
    }

    #[test]
    fn transcribes_bundled_tiny_en_wav() {
        let dir = Path::new("models/asr/whisper-tiny.en");
        if !dir.join("tiny.en-encoder.int8.onnx").is_file() {
            return;
        }
        let text = transcribe_wav(dir, &dir.join("test_wavs/0.wav"));
        let upper = text.to_ascii_uppercase();
        assert!(
            upper.contains("NIGHTFALL") || upper.contains("YELLOW"),
            "{text}"
        );
    }
}
