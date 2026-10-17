use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig};

use crate::metrics::AsrMetrics;

const ENCODER_NAME: &str = "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx";
const DECODER_NAME: &str = "decoder-epoch-99-avg-1-chunk-16-left-128.onnx";
const JOINER_NAME: &str = "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx";
const TOKENS_NAME: &str = "tokens.txt";

/// Streaming ASR event. Only [`AsrEvent::Final`] should be translated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AsrEvent {
    /// Unstable hypothesis for the current utterance.
    Partial(String),
    /// Endpoint-confirmed transcript.
    Final(String),
}

/// File layout for the English Zipformer transducer.
#[derive(Clone, Debug)]
pub struct AsrModelPaths {
    /// Encoder ONNX path.
    pub encoder: PathBuf,
    /// Decoder ONNX path.
    pub decoder: PathBuf,
    /// Joiner ONNX path.
    pub joiner: PathBuf,
    /// SentencePiece / token table.
    pub tokens: PathBuf,
}

impl AsrModelPaths {
    /// Resolve the default 2023-06-26 English Zipformer files under `model_dir`.
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let paths = Self {
            encoder: model_dir.join(ENCODER_NAME),
            decoder: model_dir.join(DECODER_NAME),
            joiner: model_dir.join(JOINER_NAME),
            tokens: model_dir.join(TOKENS_NAME),
        };
        paths.validate()?;
        Ok(paths)
    }

    fn validate(&self) -> Result<()> {
        for (label, path) in [
            ("encoder", &self.encoder),
            ("decoder", &self.decoder),
            ("joiner", &self.joiner),
            ("tokens", &self.tokens),
        ] {
            if !path.is_file() {
                return Err(anyhow!(
                    "missing ASR {label} at {}. Run scripts/download-asr.sh",
                    path.display()
                ));
            }
        }
        Ok(())
    }
}

/// Streaming Zipformer recognizer that yields partial and final transcripts.
pub struct AsrEngine {
    recognizer: OnlineRecognizer,
    stream: sherpa_onnx::OnlineStream,
    metrics: AsrMetrics,
    last_text: String,
}

impl AsrEngine {
    /// Load the CPU Zipformer transducer.
    pub fn new(
        paths: &AsrModelPaths,
        num_threads: i32,
        rule2_min_trailing_silence: f32,
    ) -> Result<Self> {
        let mut config = OnlineRecognizerConfig::default();
        config.model_config.transducer.encoder = Some(path_string(&paths.encoder));
        config.model_config.transducer.decoder = Some(path_string(&paths.decoder));
        config.model_config.transducer.joiner = Some(path_string(&paths.joiner));
        config.model_config.tokens = Some(path_string(&paths.tokens));
        config.model_config.provider = Some("cpu".to_string());
        config.model_config.num_threads = num_threads.max(1);
        config.model_config.debug = false;
        config.enable_endpoint = true;
        config.decoding_method = Some("greedy_search".to_string());
        config.rule1_min_trailing_silence = 2.4;
        config.rule2_min_trailing_silence = rule2_min_trailing_silence;
        config.rule3_min_utterance_length = 20.0;

        let recognizer = OnlineRecognizer::create(&config)
            .ok_or_else(|| anyhow!("failed to create OnlineRecognizer"))?;
        let stream = recognizer.create_stream();
        Ok(Self {
            recognizer,
            stream,
            metrics: AsrMetrics::new(),
            last_text: String::new(),
        })
    }

    /// Feed a PCM chunk. Sherpa resamples to 16 kHz internally.
    pub fn push_samples(&mut self, sample_rate: u32, samples: &[f32]) -> Result<Vec<AsrEvent>> {
        self.metrics.add_audio_samples(samples.len(), sample_rate);
        self.stream.accept_waveform(
            i32::try_from(sample_rate).context("sample rate fits i32")?,
            samples,
        );
        Ok(self.poll())
    }

    /// Flush trailing audio after a file or capture stop.
    pub fn finish_input(&mut self) -> Result<Vec<AsrEvent>> {
        self.stream.input_finished();
        let mut events = self.poll();
        if !self.last_text.is_empty() {
            let text = std::mem::take(&mut self.last_text);
            events.push(AsrEvent::Final(text));
            self.recognizer.reset(&self.stream);
        }
        Ok(events)
    }

    /// Current ASR timing / drop counters.
    pub fn metrics(&self) -> &AsrMetrics {
        &self.metrics
    }

    /// Mutable metrics, used to fold capture drops into the summary.
    pub fn metrics_mut(&mut self) -> &mut AsrMetrics {
        &mut self.metrics
    }

    fn poll(&mut self) -> Vec<AsrEvent> {
        let mut events = Vec::new();
        while self.recognizer.is_ready(&self.stream) {
            self.metrics.measure_decode(|| {
                self.recognizer.decode(&self.stream);
            });

            if let Some(result) = self.recognizer.get_result(&self.stream) {
                let text = result.text.trim();
                if !text.is_empty() && text != self.last_text {
                    self.last_text = text.to_string();
                    events.push(AsrEvent::Partial(self.last_text.clone()));
                }
            }

            if self.recognizer.is_endpoint(&self.stream) {
                if !self.last_text.is_empty() {
                    let text = std::mem::take(&mut self.last_text);
                    events.push(AsrEvent::Final(text));
                }
                self.recognizer.reset(&self.stream);
            }
        }
        events
    }
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
