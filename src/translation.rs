use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use ort::environment::Environment;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::metrics::TranslationMetrics;

/// Translates finalized English utterances into Japanese.
pub trait Translator {
    /// Translate `text` from English to Japanese.
    fn translate(&mut self, text: &str) -> anyhow::Result<String>;
}

/// Generation hyper-parameters copied from the Hugging Face Marian config.
#[derive(Clone, Debug, Deserialize)]
pub struct GenerationConfig {
    /// Decoder start / pad id used as the first generated token.
    pub decoder_start_token_id: i64,
    /// End-of-sequence id.
    pub eos_token_id: i64,
    /// Padding id skipped in the decoded string.
    pub pad_token_id: i64,
    /// Maximum generated tokens, including the start token.
    pub max_length: usize,
}

/// Marian encoder-decoder loaded from an Optimum ONNX export.
pub struct OnnxMarianTranslator {
    encoder: Session,
    decoder: Session,
    source_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    generation: GenerationConfig,
    encoder_input_ids_name: String,
    encoder_mask_name: Option<String>,
    encoder_hidden_name: String,
    decoder_input_ids_name: String,
    decoder_encoder_hidden_name: String,
    decoder_encoder_mask_name: Option<String>,
    decoder_logits_name: String,
    metrics: TranslationMetrics,
}

impl OnnxMarianTranslator {
    /// Load encoder, decoder, tokenizers, and `generation.json` from an Optimum export directory.
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let paths = TranslationModelPaths::from_dir(model_dir)?;
        let env = ort::init()
            .with_name("konnnyaku-marian")
            .build()
            .map_err(ort_err)?;

        let encoder = session_from_file(&env, &paths.encoder)?;
        let decoder = session_from_file(&env, &paths.decoder)?;

        let source_tokenizer = Tokenizer::from_file(&paths.source_tokenizer)
            .map_err(|err| anyhow!("load source tokenizer: {err}"))?;
        let target_tokenizer = Tokenizer::from_file(&paths.target_tokenizer)
            .map_err(|err| anyhow!("load target tokenizer: {err}"))?;
        let generation: GenerationConfig = serde_json::from_str(
            &std::fs::read_to_string(&paths.generation)
                .with_context(|| format!("read {}", paths.generation.display()))?,
        )
        .context("parse generation.json")?;

        let encoder_input_ids_name = find_input_name(&encoder, &["input_ids"])?;
        let encoder_mask_name = find_optional_input_name(&encoder, &["attention_mask"]);
        let encoder_hidden_name = find_output_name(&encoder, &["last_hidden_state", "hidden"])?;
        let decoder_input_ids_name =
            find_input_name(&decoder, &["decoder_input_ids", "input_ids"])?;
        let decoder_encoder_hidden_name =
            find_input_name(&decoder, &["encoder_hidden_states", "encoder_outputs"])?;
        let decoder_encoder_mask_name =
            find_optional_input_name(&decoder, &["encoder_attention_mask"]);
        let decoder_logits_name = find_output_name(&decoder, &["logits"])?;

        tracing::info!(
            encoder = %paths.encoder.display(),
            decoder = %paths.decoder.display(),
            "loaded Marian ONNX translator"
        );

        Ok(Self {
            encoder,
            decoder,
            source_tokenizer,
            target_tokenizer,
            generation,
            encoder_input_ids_name,
            encoder_mask_name,
            encoder_hidden_name,
            decoder_input_ids_name,
            decoder_encoder_hidden_name,
            decoder_encoder_mask_name,
            decoder_logits_name,
            metrics: TranslationMetrics::new()?,
        })
    }

    /// Translation latency histogram.
    pub fn metrics(&self) -> &TranslationMetrics {
        &self.metrics
    }

    fn encode_source(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let encoding = self
            .source_tokenizer
            .encode(text, false)
            .map_err(|err| anyhow!("tokenize source: {err}"))?;
        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
        if ids.last().copied() != Some(self.generation.eos_token_id) {
            ids.push(self.generation.eos_token_id);
        }
        let mask = vec![1_i64; ids.len()];
        Ok((ids, mask))
    }

    fn run_encoder(
        &mut self,
        input_ids: &[i64],
        attention_mask: &[i64],
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let seq = i64::try_from(input_ids.len()).context("source length fits i64")?;
        let ids_tensor = Tensor::from_array(([1_i64, seq], input_ids.to_vec()))?;

        let outputs = if let Some(mask_name) = &self.encoder_mask_name {
            let mask_tensor = Tensor::from_array(([1_i64, seq], attention_mask.to_vec()))?;
            self.encoder
                .run(ort::inputs! {
                    self.encoder_input_ids_name.as_str() => ids_tensor,
                    mask_name.as_str() => mask_tensor
                })
                .map_err(ort_err)?
        } else {
            self.encoder
                .run(ort::inputs! {
                    self.encoder_input_ids_name.as_str() => ids_tensor
                })
                .map_err(ort_err)?
        };

        extract_f32(&outputs[self.encoder_hidden_name.as_str()])
    }

    fn greedy_decode(
        &mut self,
        encoder_hidden: &[f32],
        encoder_shape: &[usize],
    ) -> Result<Vec<u32>> {
        let mut tokens = vec![self.generation.decoder_start_token_id];
        let max_new = self.generation.max_length.clamp(8, 64);

        for _ in 0..max_new {
            let logits = self.decoder_forward(&tokens, encoder_hidden, encoder_shape)?;
            let next = argmax_last_token(&logits, self.generation.pad_token_id)?;
            if next == self.generation.eos_token_id {
                break;
            }
            tokens.push(next);
        }
        tracing::debug!(?tokens, "generated decoder ids");

        Ok(tokens
            .into_iter()
            .skip(1)
            .filter(|&id| id != self.generation.eos_token_id && id != self.generation.pad_token_id)
            .map(|id| u32::try_from(id).unwrap_or(0))
            .collect())
    }

    fn decoder_forward(
        &mut self,
        decoder_ids: &[i64],
        encoder_hidden: &[f32],
        encoder_shape: &[usize],
    ) -> Result<Logits> {
        let dec_seq = i64::try_from(decoder_ids.len()).context("decoder length fits i64")?;
        let decoder_ids_tensor = Tensor::from_array(([1_i64, dec_seq], decoder_ids.to_vec()))?;
        let hidden_dims: Vec<i64> = encoder_shape
            .iter()
            .map(|&dim| i64::try_from(dim).unwrap_or(0))
            .collect();
        let hidden_tensor = Tensor::from_array((hidden_dims, encoder_hidden.to_vec()))?;

        let outputs = if let Some(mask_name) = &self.decoder_encoder_mask_name {
            let enc_seq =
                i64::try_from(encoder_seq_len(encoder_shape)).context("encoder seq fits i64")?;
            let mask_tensor =
                Tensor::from_array(([1_i64, enc_seq], vec![1_i64; enc_seq as usize]))?;
            self.decoder
                .run(ort::inputs! {
                    self.decoder_input_ids_name.as_str() => decoder_ids_tensor,
                    self.decoder_encoder_hidden_name.as_str() => hidden_tensor,
                    mask_name.as_str() => mask_tensor
                })
                .map_err(ort_err)?
        } else {
            self.decoder
                .run(ort::inputs! {
                    self.decoder_input_ids_name.as_str() => decoder_ids_tensor,
                    self.decoder_encoder_hidden_name.as_str() => hidden_tensor
                })
                .map_err(ort_err)?
        };

        let (shape, data) = extract_f32(&outputs[self.decoder_logits_name.as_str()])?;
        Ok(Logits { shape, data })
    }
}

impl Translator for OnnxMarianTranslator {
    fn translate(&mut self, text: &str) -> anyhow::Result<String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(String::new());
        }
        let start = Instant::now();
        let (ids, mask) = self.encode_source(trimmed)?;
        let (shape, hidden) = self.run_encoder(&ids, &mask)?;
        let tokens = self.greedy_decode(&hidden, &shape)?;
        let japanese = self
            .target_tokenizer
            .decode(&tokens, true)
            .map_err(|err| anyhow!("decode target: {err}"))?;
        self.metrics.record(start.elapsed());
        Ok(cleanup_japanese(&japanese))
    }
}

struct TranslationModelPaths {
    encoder: PathBuf,
    decoder: PathBuf,
    source_tokenizer: PathBuf,
    target_tokenizer: PathBuf,
    generation: PathBuf,
}

impl TranslationModelPaths {
    fn from_dir(model_dir: &Path) -> Result<Self> {
        let paths = Self {
            encoder: model_dir.join("encoder_model.onnx"),
            decoder: model_dir.join("decoder_model.onnx"),
            source_tokenizer: model_dir.join("source_tokenizer.json"),
            target_tokenizer: model_dir.join("target_tokenizer.json"),
            generation: model_dir.join("generation.json"),
        };
        for (label, path) in [
            ("encoder_model.onnx", &paths.encoder),
            ("decoder_model.onnx", &paths.decoder),
            ("source_tokenizer.json", &paths.source_tokenizer),
            ("target_tokenizer.json", &paths.target_tokenizer),
            ("generation.json", &paths.generation),
        ] {
            if !path.is_file() {
                return Err(anyhow!(
                    "missing translation {label} at {}. Run scripts/export-marian-onnx.sh",
                    path.display()
                ));
            }
        }
        Ok(paths)
    }
}

struct Logits {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn session_from_file(env: &Environment, path: &Path) -> Result<Session> {
    let builder = Session::builder(env).map_err(ort_err)?;
    let builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?;
    let mut builder = builder.with_intra_threads(2).map_err(ort_err)?;
    builder
        .commit_from_file(path)
        .map_err(ort_err)
        .with_context(|| format!("load ONNX {}", path.display()))
}

fn ort_err(err: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{err}")
}

fn find_input_name(session: &Session, needles: &[&str]) -> Result<String> {
    find_optional_input_name(session, needles)
        .ok_or_else(|| anyhow!("no session input matching {needles:?}"))
}

fn find_optional_input_name(session: &Session, needles: &[&str]) -> Option<String> {
    session
        .inputs()
        .iter()
        .map(|input| input.name().to_string())
        .find(|name| {
            let lowered = name.to_ascii_lowercase();
            needles.iter().any(|needle| lowered.contains(needle))
        })
}

fn find_output_name(session: &Session, needles: &[&str]) -> Result<String> {
    session
        .outputs()
        .iter()
        .map(|output| output.name().to_string())
        .find(|name| {
            let lowered = name.to_ascii_lowercase();
            needles.iter().any(|needle| lowered.contains(needle))
        })
        .ok_or_else(|| anyhow!("no session output matching {needles:?}"))
}

fn extract_f32(value: &ort::value::DynValue) -> Result<(Vec<usize>, Vec<f32>)> {
    let array = value.try_extract_array::<f32>().map_err(ort_err)?;
    let shape = array.shape().to_vec();
    let data = array.iter().copied().collect();
    Ok((shape, data))
}

fn encoder_seq_len(shape: &[usize]) -> usize {
    match shape {
        [_batch, seq, _hidden] => *seq,
        [seq, _hidden] => *seq,
        other => other.get(1).copied().unwrap_or(1),
    }
}

fn argmax_last_token(logits: &Logits, banned_token: i64) -> Result<i64> {
    let vocab = *logits.shape.last().unwrap_or(&0);
    if vocab == 0 || logits.data.len() < vocab {
        return Err(anyhow!("logits tensor is empty: {:?}", logits.shape));
    }
    let start = logits.data.len() - vocab;
    let banned = usize::try_from(banned_token).ok();
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, value) in logits.data[start..].iter().enumerate() {
        if banned == Some(i) {
            continue;
        }
        if *value > best_val {
            best_val = *value;
            best_idx = i;
        }
    }
    i64::try_from(best_idx).context("vocab index fits i64")
}

fn cleanup_japanese(text: &str) -> String {
    text.replace(['▁', ' '], "").trim().to_string()
}
