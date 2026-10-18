use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};
use hound::{SampleFormat as WavSampleFormat, WavReader};

#[cfg(target_os = "macos")]
#[path = "macos_tap.rs"]
mod macos_tap;

/// One captured or file-decoded PCM chunk, already mixed to mono `f32`.
pub struct AudioChunk {
    /// Sample rate of `samples`.
    pub sample_rate: u32,
    /// Mono PCM in `-1.0..=1.0`.
    pub samples: Vec<f32>,
}

/// Live capture source for the ASR / live pipelines.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CaptureSource {
    /// Default microphone / input device.
    Microphone,
    /// Default system output via a Core Audio tap (macOS only).
    Loopback,
    /// One application's output via a Core Audio process tap (macOS only).
    Application(String),
}

/// Running app that currently has Core Audio process objects.
pub struct TappableApp {
    /// Localized display name.
    pub name: String,
    /// Bundle identifier, such as `com.google.Chrome`.
    pub bundle_id: String,
    /// Matching HAL process objects (includes helpers).
    pub process_count: usize,
}

/// Live capture. Dropping this stops the stream.
pub struct AudioCapture {
    _stream: Stream,
    #[cfg(target_os = "macos")]
    _tap: Option<macos_tap::ProcessTap>,
    drops: Arc<AtomicU64>,
}

impl AudioCapture {
    /// Number of chunks dropped because the bounded queue was full.
    pub fn drops(&self) -> u64 {
        self.drops.load(Ordering::Relaxed)
    }
}

/// Start capture from `source` and return a bounded receiver of mono chunks.
pub fn start_capture(
    source: CaptureSource,
    queue_capacity: usize,
) -> Result<(AudioCapture, Receiver<AudioChunk>)> {
    match source {
        CaptureSource::Microphone => start_microphone(queue_capacity),
        CaptureSource::Loopback => start_loopback(queue_capacity),
        CaptureSource::Application(query) => start_application(&query, queue_capacity),
    }
}

/// List running apps that currently have tappable audio process objects.
pub fn list_tappable_apps() -> Result<Vec<TappableApp>> {
    list_tappable_apps_impl()
}

fn start_microphone(queue_capacity: usize) -> Result<(AudioCapture, Receiver<AudioChunk>)> {
    let host = cpal::default_host();
    let device = list_and_pick_default_input(&host)?;
    let supported = device
        .default_input_config()
        .context("query default input config")?;
    open_capture_stream(
        device,
        supported,
        queue_capacity,
        "microphone input format",
        "start microphone stream",
        #[cfg(target_os = "macos")]
        None,
    )
}

#[cfg(target_os = "macos")]
fn start_loopback(queue_capacity: usize) -> Result<(AudioCapture, Receiver<AudioChunk>)> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow!("no default output device"))?;

    if device.supports_input() {
        anyhow::bail!(
            "default output '{device}' also has inputs, so cpal would capture the device microphone instead of system audio. Set macOS default output to built-in speakers and retry"
        );
    }

    eprintln!("Capturing default system output: {device}");
    eprintln!(
        "This records every app on that output, not just the browser. If you hear playback but get silence, grant System Audio Recording (or Screen Recording) to Terminal/Cursor in System Settings → Privacy & Security."
    );

    let supported = device
        .default_output_config()
        .context("query default output config")?;
    open_capture_stream(
        device,
        supported,
        queue_capacity,
        "loopback output format",
        "start loopback stream",
        None,
    )
}

#[cfg(not(target_os = "macos"))]
fn start_loopback(_queue_capacity: usize) -> Result<(AudioCapture, Receiver<AudioChunk>)> {
    anyhow::bail!("--loopback is only supported on macOS")
}

#[cfg(target_os = "macos")]
fn start_application(
    query: &str,
    queue_capacity: usize,
) -> Result<(AudioCapture, Receiver<AudioChunk>)> {
    let tap = macos_tap::create_app_tap(query)?;
    let device = macos_tap::find_aggregate_input(&tap.uid)?;
    let supported = device
        .default_input_config()
        .context("query app-tap input config")?;
    open_capture_stream(
        device,
        supported,
        queue_capacity,
        "app tap input format",
        "start app tap stream",
        Some(tap),
    )
}

#[cfg(not(target_os = "macos"))]
fn start_application(
    _query: &str,
    _queue_capacity: usize,
) -> Result<(AudioCapture, Receiver<AudioChunk>)> {
    anyhow::bail!("--app is only supported on macOS")
}

#[cfg(target_os = "macos")]
fn list_tappable_apps_impl() -> Result<Vec<TappableApp>> {
    macos_tap::list_tappable_apps()
}

#[cfg(not(target_os = "macos"))]
fn list_tappable_apps_impl() -> Result<Vec<TappableApp>> {
    anyhow::bail!("listing apps is only supported on macOS")
}

fn open_capture_stream(
    device: Device,
    supported: cpal::SupportedStreamConfig,
    queue_capacity: usize,
    format_log: &'static str,
    play_context: &'static str,
    #[cfg(target_os = "macos")] tap: Option<macos_tap::ProcessTap>,
) -> Result<(AudioCapture, Receiver<AudioChunk>)> {
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();
    let sample_rate = config.sample_rate;
    let channels = usize::from(config.channels);

    tracing::info!(
        sample_format = ?sample_format,
        channels,
        sample_rate,
        "{format_log}"
    );

    let (tx, rx) = bounded::<AudioChunk>(queue_capacity);
    let drops = Arc::new(AtomicU64::new(0));
    let stream = build_input_stream(
        &device,
        config,
        sample_format,
        channels,
        sample_rate,
        tx,
        Arc::clone(&drops),
    )?;
    stream.play().context(play_context)?;

    Ok((
        AudioCapture {
            _stream: stream,
            #[cfg(target_os = "macos")]
            _tap: tap,
            drops,
        },
        rx,
    ))
}

/// Load a WAV file as mono `f32` PCM.
pub fn load_wav_mono_f32(path: &Path) -> Result<AudioChunk> {
    let mut reader =
        WavReader::open(path).with_context(|| format!("open wav {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err(anyhow!("wav has zero channels"));
    }

    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (WavSampleFormat::Float, 32) => collect_mono(reader.samples::<f32>(), channels, |s| s)?,
        (WavSampleFormat::Int, 16) => {
            collect_mono(reader.samples::<i16>(), channels, |s| s as f32 / 32768.0)?
        }
        (WavSampleFormat::Int, 32) => collect_mono(reader.samples::<i32>(), channels, |s| {
            s as f32 / 2147483648.0
        })?,
        (format, bits) => {
            return Err(anyhow!(
                "unsupported wav format {format:?} {bits}-bit in {}",
                path.display()
            ));
        }
    };

    Ok(AudioChunk {
        sample_rate: spec.sample_rate,
        samples,
    })
}

/// Split PCM into ~`chunk_ms` millisecond pieces for streaming ASR.
pub fn chunk_samples(samples: &[f32], sample_rate: u32, chunk_ms: u32) -> Vec<Vec<f32>> {
    let chunk_len = samples_for_ms(sample_rate, chunk_ms).max(1);
    samples.chunks(chunk_len).map(<[f32]>::to_vec).collect()
}

/// Sample count corresponding to `chunk_ms` at `sample_rate`.
pub fn samples_for_ms(sample_rate: u32, chunk_ms: u32) -> usize {
    (u64::from(sample_rate) * u64::from(chunk_ms) / 1000) as usize
}

fn list_and_pick_default_input(host: &cpal::Host) -> Result<Device> {
    let default_input = host.default_input_device();
    let default_label = default_input.as_ref().map(ToString::to_string);

    println!("Available input devices:");
    if let Ok(devices) = host.input_devices() {
        for device in devices {
            let label = device.to_string();
            let mark = if default_label.as_ref() == Some(&label) {
                "*"
            } else {
                " "
            };
            println!("{mark} {label}");
        }
    }

    let device = default_input.ok_or_else(|| anyhow!("no default input device"))?;
    println!("\nUsing default device: {device}");
    Ok(device)
}

fn build_input_stream(
    device: &Device,
    config: StreamConfig,
    sample_format: SampleFormat,
    channels: usize,
    sample_rate: u32,
    tx: Sender<AudioChunk>,
    drops: Arc<AtomicU64>,
) -> Result<Stream> {
    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _| {
                send_mono_chunk(
                    &tx,
                    &drops,
                    sample_rate,
                    frames_to_mono(data, channels, |s| s),
                );
            },
            log_stream_error,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _| {
                send_mono_chunk(
                    &tx,
                    &drops,
                    sample_rate,
                    frames_to_mono(data, channels, |s| f32::from(s) / 32768.0),
                );
            },
            log_stream_error,
            None,
        )?,
        SampleFormat::U16 => device.build_input_stream(
            config,
            move |data: &[u16], _| {
                send_mono_chunk(
                    &tx,
                    &drops,
                    sample_rate,
                    frames_to_mono(data, channels, |s| (f32::from(s) - 32768.0) / 32768.0),
                );
            },
            log_stream_error,
            None,
        )?,
        SampleFormat::I32 => device.build_input_stream(
            config,
            move |data: &[i32], _| {
                send_mono_chunk(
                    &tx,
                    &drops,
                    sample_rate,
                    frames_to_mono(data, channels, |s| s as f32 / 2_147_483_648.0),
                );
            },
            log_stream_error,
            None,
        )?,
        other => anyhow::bail!("unsupported sample format: {other:?}"),
    };

    Ok(stream)
}

fn log_stream_error(err: cpal::Error) {
    tracing::error!(?err, "audio stream error");
}

fn send_mono_chunk(
    tx: &Sender<AudioChunk>,
    drops: &AtomicU64,
    sample_rate: u32,
    samples: Vec<f32>,
) {
    if samples.is_empty() {
        return;
    }
    match tx.try_send(AudioChunk {
        sample_rate,
        samples,
    }) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
            drops.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn frames_to_mono<T, F>(data: &[T], channels: usize, mut convert: F) -> Vec<f32>
where
    T: Copy,
    F: FnMut(T) -> f32,
{
    if channels <= 1 {
        return data.iter().copied().map(convert).collect();
    }
    data.chunks(channels)
        .map(|frame| {
            let sum: f32 = frame.iter().copied().map(&mut convert).sum();
            sum / channels as f32
        })
        .collect()
}

fn collect_mono<S, E, F>(
    samples: impl Iterator<Item = Result<S, E>>,
    channels: usize,
    convert: F,
) -> Result<Vec<f32>>
where
    E: std::error::Error + Send + Sync + 'static,
    F: Fn(S) -> f32,
{
    let mut converted = Vec::new();
    for sample in samples {
        converted.push(convert(sample?));
    }
    if channels == 1 {
        return Ok(converted);
    }
    Ok(converted
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect())
}
