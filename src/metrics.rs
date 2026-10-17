use std::fmt::{Display, Formatter, Result as FmtResult};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use hdrhistogram::Histogram;

/// ASR timing and capture-drop counters.
pub struct AsrMetrics {
    audio_seconds: f64,
    inference_seconds: f64,
    drops: u64,
}

impl AsrMetrics {
    /// Create empty ASR metrics.
    pub fn new() -> Self {
        Self {
            audio_seconds: 0.0,
            inference_seconds: 0.0,
            drops: 0,
        }
    }

    /// Record wall time spent in recognizer decode.
    pub fn measure_decode<F>(&mut self, decode: F)
    where
        F: FnOnce(),
    {
        let start = Instant::now();
        decode();
        self.inference_seconds += start.elapsed().as_secs_f64();
    }

    /// Add captured or file audio duration.
    pub fn add_audio_samples(&mut self, num_samples: usize, sample_rate: u32) {
        if sample_rate == 0 {
            return;
        }
        self.audio_seconds += num_samples as f64 / f64::from(sample_rate);
    }

    /// Record audio chunks dropped because the capture queue was full.
    pub fn add_drops(&mut self, drops: u64) {
        self.drops += drops;
    }

    /// Real-time factor: inference seconds / audio seconds.
    pub fn rtf(&self) -> f64 {
        if self.audio_seconds <= 0.0 {
            0.0
        } else {
            self.inference_seconds / self.audio_seconds
        }
    }

    /// Captured or decoded audio duration in seconds.
    pub fn audio_seconds(&self) -> f64 {
        self.audio_seconds
    }

    /// Time spent in decoder calls.
    pub fn inference_seconds(&self) -> f64 {
        self.inference_seconds
    }

    /// Number of dropped audio chunks.
    pub fn drops(&self) -> u64 {
        self.drops
    }
}

impl Default for AsrMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for AsrMetrics {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "audio={:.2}s inference={:.2}s rtf={:.3} drops={}",
            self.audio_seconds(),
            self.inference_seconds(),
            self.rtf(),
            self.drops()
        )
    }
}

/// Translation latency histogram in microseconds.
pub struct TranslationMetrics {
    histogram: Histogram<u64>,
}

impl TranslationMetrics {
    /// Create a histogram covering 1 microsecond to 60 seconds.
    pub fn new() -> Result<Self> {
        Ok(Self {
            histogram: Histogram::new_with_bounds(1, 60_000_000, 3)
                .context("create translation latency histogram")?,
        })
    }

    /// Record one translation duration.
    pub fn record(&mut self, duration: Duration) {
        let micros = duration.as_micros().max(1) as u64;
        let _ = self.histogram.record(micros.min(60_000_000));
    }

    /// True when at least one sample has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.histogram.is_empty()
    }

    /// Percentile in milliseconds.
    pub fn percentile_ms(&self, quantile: f64) -> f64 {
        if self.histogram.is_empty() {
            0.0
        } else {
            self.histogram.value_at_quantile(quantile) as f64 / 1000.0
        }
    }
}

impl Display for TranslationMetrics {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        if self.is_empty() {
            write!(f, "translations=0")
        } else {
            write!(
                f,
                "n={} p50={:.1}ms p95={:.1}ms p99={:.1}ms",
                self.histogram.len(),
                self.percentile_ms(0.50),
                self.percentile_ms(0.95),
                self.percentile_ms(0.99)
            )
        }
    }
}
