//! Spoken language identification for a finished slice of audio.
//!
//! The model is multilingual Whisper tiny. Only English and Japanese are acted on.
//! Anything else stays unlabeled so it is not forced through the wrong recognizer.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use sherpa_onnx::{
    SpokenLanguageIdentification, SpokenLanguageIdentificationConfig,
    SpokenLanguageIdentificationWhisperConfig,
};

/// Language of one audio clip.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Heard {
    /// English.
    En,
    /// Japanese.
    Ja,
    /// A language this app does not transcribe.
    Other,
}

/// Map a Whisper language id onto [`Heard`].
pub fn classify(lang: &str) -> Heard {
    let lang = lang.trim().to_ascii_lowercase();
    let primary = lang.split(['-', '_']).next().unwrap_or(lang.as_str());
    match primary {
        "en" => Heard::En,
        "ja" | "jp" => Heard::Ja,
        _ => Heard::Other,
    }
}

/// Lazy Whisper-tiny language identifier.
pub struct LanguageId {
    model: SpokenLanguageIdentification,
}

impl LanguageId {
    /// Load `tiny-encoder` and `tiny-decoder` from `model_dir`. Int8 is preferred.
    pub fn open(model_dir: &Path, num_threads: i32) -> Result<Self> {
        let encoder = find_onnx(model_dir, "encoder")?;
        let decoder = find_onnx(model_dir, "decoder")?;
        let config = SpokenLanguageIdentificationConfig {
            whisper: SpokenLanguageIdentificationWhisperConfig {
                encoder: Some(path_string(&encoder)),
                decoder: Some(path_string(&decoder)),
                tail_paddings: 0,
            },
            num_threads: num_threads.max(1),
            debug: false,
            provider: Some("cpu".to_string()),
        };
        let model = SpokenLanguageIdentification::create(&config)
            .ok_or_else(|| anyhow!("failed to create spoken language identifier"))?;
        tracing::info!(dir = %model_dir.display(), "loaded spoken language identifier");
        Ok(Self { model })
    }

    /// Identify `samples`. [`Heard::Other`] when the clip is under 0.8 seconds.
    pub fn identify(&self, sample_rate: u32, samples: &[f32]) -> Result<Heard> {
        if sample_rate == 0 || samples.len() < sample_rate as usize * 4 / 5 {
            return Ok(Heard::Other);
        }
        let stream = self.model.create_stream();
        stream.accept_waveform(
            i32::try_from(sample_rate).context("sample rate fits i32")?,
            samples,
        );
        let result = self
            .model
            .compute(&stream)
            .ok_or_else(|| anyhow!("language identification returned nothing"))?;
        Ok(classify(&result.lang))
    }
}

fn find_onnx(dir: &Path, role: &str) -> Result<std::path::PathBuf> {
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

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_synthesized_speech_when_requested() {
        if std::env::var("KONNNYAKU_LID").is_err() {
            return;
        }
        let dir = Path::new("models/asr/whisper-tiny");
        if !dir.join("tiny-encoder.int8.onnx").is_file() {
            return;
        }
        let model = LanguageId::open(dir, 2).expect("language id");
        let en = synthesize("Samantha", "The meeting starts at ten thirty.");
        let ja = synthesize("Kyoko", "会議は十時半に始まります。");
        assert_eq!(
            model.identify(en.sample_rate, &en.samples).expect("en"),
            Heard::En
        );
        assert_eq!(
            model.identify(ja.sample_rate, &ja.samples).expect("ja"),
            Heard::Ja
        );
    }

    fn synthesize(voice: &str, text: &str) -> crate::audio::AudioChunk {
        let dir = std::env::temp_dir().join(format!("konnnyaku-lid-{voice}"));
        std::fs::create_dir_all(&dir).expect("temp");
        let aiff = dir.join("speech.aiff");
        let wav = dir.join("speech.wav");
        let say = std::process::Command::new("say")
            .args(["-v", voice, "-o"])
            .arg(&aiff)
            .arg(text)
            .status()
            .expect("say");
        assert!(say.success());
        let convert = std::process::Command::new("afconvert")
            .args(["-f", "WAVE", "-d", "LEI16@16000", "-c", "1"])
            .arg(&aiff)
            .arg(&wav)
            .status()
            .expect("afconvert");
        assert!(convert.success());
        crate::audio::load_wav_mono_f32(&wav).expect("wav")
    }

    #[test]
    fn classifies_english_and_japanese_only() {
        assert_eq!(classify("en"), Heard::En);
        assert_eq!(classify("EN"), Heard::En);
        assert_eq!(classify("ja"), Heard::Ja);
        assert_eq!(classify("jp"), Heard::Ja);
        assert_eq!(classify("ja-JP"), Heard::Ja);
        assert_eq!(classify("zh"), Heard::Other);
        assert_eq!(classify(""), Heard::Other);
    }
}
