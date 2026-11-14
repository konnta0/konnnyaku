use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};

/// Offline Japanese recognizer. Haas sees a whole utterance, not a token stream.
pub struct HaasRecognizer {
    recognizer: OfflineRecognizer,
}

impl HaasRecognizer {
    /// Load the sherpa-onnx Haas ONNX directory (`encoder.quantized.onnx` plus `encoder.data`).
    pub fn from_dir(model_dir: &Path, num_threads: i32) -> Result<Self> {
        let encoder = find_role(model_dir, "encoder")?;
        let decoder = find_role(model_dir, "decoder")?;
        let joiner = find_role(model_dir, "joiner")?;
        let tokens = model_dir.join("tokens.txt");
        if !tokens.is_file() {
            anyhow::bail!("missing tokens.txt in {}", model_dir.display());
        }
        let mut config = OfflineRecognizerConfig::default();
        config.feat_config.feature_dim = 128;
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
            .ok_or_else(|| anyhow!("failed to create Haas OfflineRecognizer"))?;
        tracing::info!(dir = %model_dir.display(), "loaded Haas Japanese recognizer");
        Ok(Self { recognizer })
    }

    /// Transcribe one utterance. Returns `None` when the audio is too short or empty.
    pub fn transcribe(&self, sample_rate: u32, samples: &[f32]) -> Result<Option<String>> {
        if sample_rate == 0 || samples.len() < sample_rate as usize / 2 {
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

fn find_role(dir: &Path, role: &str) -> Result<PathBuf> {
    let entries = std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))?;
    let prefix = format!("{role}.");
    let dashed = format!("-{role}.");
    let mut found = None;
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".onnx") && (name.starts_with(&prefix) || name.contains(&dashed)) {
            found = Some(entry.path());
            break;
        }
    }
    found.ok_or_else(|| {
        anyhow!(
            "missing {role} onnx in {}. Run konnnyaku setup",
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

    #[test]
    fn loads_quantized_haas_when_requested() {
        if std::env::var("KONNNYAKU_HAAS").is_err() {
            return;
        }
        let dir = Path::new("models/asr/haas-ja");
        if !dir.join("encoder.data").is_file() {
            return;
        }
        let model = HaasRecognizer::from_dir(dir, 2).expect("load haas");
        let silence = vec![0.0_f32; 16_000];
        model.transcribe(16_000, &silence).expect("decode silence");
    }
}
