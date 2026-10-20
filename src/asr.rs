use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig};

use crate::metrics::AsrMetrics;
use crate::vad::{SpeechDecision, SpeechGate};

const ENCODER_NAME: &str = "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx";
const DECODER_NAME: &str = "decoder-epoch-99-avg-1-chunk-16-left-128.onnx";
const JOINER_NAME: &str = "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx";
const TOKENS_NAME: &str = "tokens.txt";
const MAX_UTTERANCE_SECS: u32 = 30;

/// Streaming ASR event. Only [`AsrEvent::Final`] should be translated.
#[derive(Debug)]
pub enum AsrEvent {
    /// Unstable hypothesis for the current utterance.
    Partial(String),
    /// Endpoint-confirmed transcript plus the speech audio used to produce it.
    Final(FinalUtterance),
}

/// One endpoint-confirmed utterance, with PCM for optional Whisper rescoring.
#[derive(Debug)]
pub struct FinalUtterance {
    /// Zipformer transcript at endpoint time.
    pub text: String,
    /// Speech-gated PCM for this utterance (capture sample rate).
    pub samples: Vec<f32>,
    /// Sample rate of [`Self::samples`].
    pub sample_rate: u32,
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
    last_emitted: String,
    skip_repeat: bool,
    utterance: Vec<f32>,
    utterance_rate: u32,
    vad: Option<SpeechGate>,
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
        config.decoding_method = Some("modified_beam_search".to_string());
        config.max_active_paths = 4;
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
            last_emitted: String::new(),
            skip_repeat: false,
            utterance: Vec::new(),
            utterance_rate: 0,
            vad: None,
        })
    }

    /// Gate Zipformer with Silero so music is not decoded as English.
    pub fn set_vad(&mut self, vad: SpeechGate) {
        self.vad = Some(vad);
    }

    /// Feed a PCM chunk. Sherpa resamples to 16 kHz internally.
    pub fn push_samples(&mut self, sample_rate: u32, samples: &[f32]) -> Result<Vec<AsrEvent>> {
        self.metrics.add_audio_samples(samples.len(), sample_rate);

        let decision = self
            .vad
            .as_mut()
            .map(|vad| vad.push(sample_rate, samples))
            .unwrap_or(SpeechDecision {
                active: true,
                ended: false,
            });

        if decision.active {
            self.append_utterance(sample_rate, samples);
            self.skip_repeat = false;
            self.stream.accept_waveform(
                i32::try_from(sample_rate).context("sample rate fits i32")?,
                samples,
            );
            return Ok(self.poll());
        }

        if decision.ended {
            return Ok(self.finalize_utterance());
        }

        Ok(Vec::new())
    }

    /// Flush trailing audio after a file or capture stop.
    pub fn finish_input(&mut self) -> Result<Vec<AsrEvent>> {
        if let Some(vad) = self.vad.as_mut() {
            let _ = vad.flush();
        }
        self.stream.input_finished();
        Ok(self.finalize_utterance())
    }

    /// Current ASR timing / drop counters.
    pub fn metrics(&self) -> &AsrMetrics {
        &self.metrics
    }

    /// Mutable metrics, used to fold capture drops into the summary.
    pub fn metrics_mut(&mut self) -> &mut AsrMetrics {
        &mut self.metrics
    }

    fn append_utterance(&mut self, sample_rate: u32, samples: &[f32]) {
        if self.utterance_rate != 0 && self.utterance_rate != sample_rate {
            self.utterance.clear();
        }
        self.utterance_rate = sample_rate;
        self.utterance.extend_from_slice(samples);
        let max_samples = (sample_rate as usize).saturating_mul(MAX_UTTERANCE_SECS as usize);
        if self.utterance.len() > max_samples {
            let drop = self.utterance.len() - max_samples;
            self.utterance.drain(..drop);
        }
    }

    fn finalize_utterance(&mut self) -> Vec<AsrEvent> {
        let mut events = self.poll();
        if !self.last_text.is_empty() {
            events.push(self.take_final());
            self.recognizer.reset(&self.stream);
            self.skip_repeat = true;
        } else {
            self.utterance.clear();
        }
        events
    }

    fn take_final(&mut self) -> AsrEvent {
        let text = std::mem::take(&mut self.last_text);
        self.last_emitted = text.clone();
        AsrEvent::Final(FinalUtterance {
            text,
            samples: std::mem::take(&mut self.utterance),
            sample_rate: self.utterance_rate,
        })
    }

    fn poll(&mut self) -> Vec<AsrEvent> {
        let mut events = Vec::new();
        while self.recognizer.is_ready(&self.stream) {
            self.metrics.measure_decode(|| {
                self.recognizer.decode(&self.stream);
            });

            if let Some(result) = self.recognizer.get_result(&self.stream) {
                let text = result.text.trim();
                let echo = self.skip_repeat && text == self.last_emitted;
                if !text.is_empty() && text != self.last_text && !echo {
                    self.last_text = text.to_string();
                    events.push(AsrEvent::Partial(self.last_text.clone()));
                }
            }

            if self.recognizer.is_endpoint(&self.stream) {
                if !self.last_text.is_empty() {
                    events.push(self.take_final());
                } else {
                    self.utterance.clear();
                }
                self.recognizer.reset(&self.stream);
                self.skip_repeat = true;
            }
        }
        events
    }
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
