mod asr;
mod audio;
mod caption;
mod metrics;
mod text;
mod translation;
mod vad;
mod whisper;

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use crossbeam_channel::{RecvTimeoutError, bounded};
use tracing_subscriber::EnvFilter;

use crate::asr::{AsrEngine, AsrEvent, AsrModelPaths, FinalUtterance};
use crate::audio::{
    CaptureSource, chunk_samples, list_tappable_apps, load_wav_mono_f32, samples_for_ms,
    start_capture,
};
use crate::metrics::TranslationMetrics;
use crate::translation::{OnnxMarianTranslator, Translator};
use crate::vad::SpeechGate;
use crate::whisper::WhisperRescorer;

const DEFAULT_ASR_DIR: &str = "models/asr/zipformer-en";
const DEFAULT_MT_DIR: &str = "models/translation/marian-en-ja/onnx";
const AUDIO_QUEUE_CAPACITY: usize = 50;
const TRANSLATION_QUEUE_CAPACITY: usize = 8;
const CHUNK_MS: u32 = 200;

#[derive(Parser, Debug)]
#[command(
    name = "konnnyaku",
    about = "Fully local English speech to Japanese subtitles"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Streaming English ASR from a WAV file, microphone, system output, or one app.
    Asr(AsrArgs),
    /// Translate English text with the local Marian ONNX model.
    Translate(TranslateArgs),
    /// Live ASR with Japanese subtitles for finalized utterances.
    Live(LiveArgs),
    /// List running apps that currently have tappable audio output.
    Apps,
}

#[derive(clap::Args, Debug)]
struct AsrArgs {
    /// 16-bit or 32-bit WAV path. Mutually exclusive with --mic, --loopback, and --app.
    #[arg(long)]
    wav: Option<PathBuf>,
    /// Capture from the default microphone.
    #[arg(long, default_value_t = false)]
    mic: bool,
    /// Capture default system output (all apps on that device).
    #[arg(long, default_value_t = false)]
    loopback: bool,
    /// Capture one application's output (name or bundle ID, macOS).
    #[arg(long)]
    app: Option<String>,
    /// Directory containing encoder/decoder/joiner/tokens.
    #[arg(long, default_value = DEFAULT_ASR_DIR)]
    model_dir: PathBuf,
    /// Endpoint rule2 trailing silence in seconds.
    #[arg(long, default_value_t = 0.8)]
    rule2: f32,
    /// ONNX Runtime threads inside sherpa-onnx.
    #[arg(long, default_value_t = 2)]
    threads: i32,
}

#[derive(clap::Args, Debug)]
struct TranslateArgs {
    /// English sentence to translate.
    #[arg(long)]
    text: Option<String>,
    /// Text file with one English sentence per line.
    #[arg(long)]
    file: Option<PathBuf>,
    /// Directory containing encoder/decoder ONNX and tokenizers.
    #[arg(long, default_value = DEFAULT_MT_DIR)]
    model_dir: PathBuf,
}

#[derive(clap::Args, Debug)]
struct LiveArgs {
    /// Capture from the default microphone (the default when --loopback/--app is omitted).
    #[arg(long, default_value_t = false)]
    mic: bool,
    /// Capture default system output (all apps on that device).
    #[arg(long, default_value_t = false)]
    loopback: bool,
    /// Capture one application's output (name or bundle ID, macOS).
    #[arg(long)]
    app: Option<String>,
    /// Directory containing encoder/decoder/joiner/tokens.
    #[arg(long, default_value = DEFAULT_ASR_DIR)]
    asr_model_dir: PathBuf,
    /// Directory containing encoder/decoder ONNX and tokenizers.
    #[arg(long, default_value = DEFAULT_MT_DIR)]
    mt_model_dir: PathBuf,
    /// Endpoint rule2 trailing silence in seconds.
    #[arg(long, default_value_t = 0.8)]
    rule2: f32,
    /// ONNX Runtime threads inside sherpa-onnx.
    #[arg(long, default_value_t = 2)]
    threads: i32,
    /// Directory containing Whisper tiny.en encoder/decoder/tokens.
    #[arg(long, default_value = "models/asr/whisper-tiny.en")]
    whisper_dir: PathBuf,
    /// Silero VAD ONNX path.
    #[arg(long, default_value = "models/asr/silero_vad.onnx")]
    vad_model: PathBuf,
    /// Disable Whisper rescoring of Zipformer finals.
    #[arg(long, default_value_t = false)]
    no_whisper: bool,
    /// Disable Silero VAD music suppression.
    #[arg(long, default_value_t = false)]
    no_vad: bool,
}

struct TranslationJob {
    zipformer: String,
    samples: Vec<f32>,
    sample_rate: u32,
    utterance_ended: Instant,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("konnnyaku=info".parse()?))
        .with_writer(io::stderr)
        .init();

    match Cli::parse().command {
        Command::Asr(args) => run_asr(args),
        Command::Translate(args) => run_translate(args),
        Command::Live(args) => run_live(args),
        Command::Apps => run_apps(),
    }
}

fn run_asr(args: AsrArgs) -> Result<()> {
    let source = asr_capture_source(&args)?;
    let paths = AsrModelPaths::from_dir(&args.model_dir)?;
    let mut engine = AsrEngine::new(&paths, args.threads, args.rule2)?;

    if let Some(wav) = args.wav {
        run_asr_wav(&mut engine, wav)?;
    } else {
        run_asr_capture(&mut engine, source.expect("live capture source"))?;
    }
    eprintln!("ASR metrics: {}", engine.metrics());
    Ok(())
}

fn run_asr_wav(engine: &mut AsrEngine, wav: PathBuf) -> Result<()> {
    let audio = load_wav_mono_f32(&wav)?;
    tracing::info!(
        path = %wav.display(),
        sample_rate = audio.sample_rate,
        samples = audio.samples.len(),
        "loaded wav"
    );
    let mut display = TranscriptDisplay::new();
    for chunk in chunk_samples(&audio.samples, audio.sample_rate, CHUNK_MS) {
        display.handle(engine.push_samples(audio.sample_rate, &chunk)?);
    }
    display.handle(engine.finish_input()?);
    display.finish();
    Ok(())
}

fn run_apps() -> Result<()> {
    let apps = list_tappable_apps()?;
    if apps.is_empty() {
        println!("No running apps currently have Core Audio process objects.");
        println!("Start playback in the app, then run this command again.");
        return Ok(());
    }
    println!("Tappable apps (name / bundle ID / process objects):");
    for app in apps {
        println!(
            "  {:<28}  {:<40}  {}",
            app.name, app.bundle_id, app.process_count
        );
    }
    Ok(())
}

fn run_asr_capture(engine: &mut AsrEngine, source: CaptureSource) -> Result<()> {
    let running = install_ctrlc()?;
    let label = capture_label(&source);
    let (capture, rx) = start_capture(source, AUDIO_QUEUE_CAPACITY)?;
    let mut display = TranscriptDisplay::new();
    let mut buffer = Vec::new();
    let mut sample_rate = 0_u32;

    println!("Streaming {label} ASR. Press Ctrl+C to stop.");
    while running.load(Ordering::SeqCst) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => {
                sample_rate = chunk.sample_rate;
                buffer.extend_from_slice(&chunk.samples);
                let chunk_len = samples_for_ms(sample_rate, CHUNK_MS).max(1);
                while buffer.len() >= chunk_len {
                    let samples: Vec<f32> = buffer.drain(..chunk_len).collect();
                    display.handle(engine.push_samples(sample_rate, &samples)?);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    if sample_rate > 0 && !buffer.is_empty() {
        display.handle(engine.push_samples(sample_rate, &buffer)?);
    }
    display.handle(engine.finish_input()?);
    display.finish();
    engine.metrics_mut().add_drops(capture.drops());
    drop(capture);
    Ok(())
}

fn run_translate(args: TranslateArgs) -> Result<()> {
    if args.text.is_some() == args.file.is_some() {
        anyhow::bail!("specify exactly one of --text or --file");
    }
    let mut translator = OnnxMarianTranslator::from_dir(&args.model_dir)?;
    if let Some(text) = args.text {
        println!("{}", translator.translate(&text)?);
    } else {
        let file = args.file.expect("file set");
        let contents =
            std::fs::read_to_string(&file).with_context(|| format!("read {}", file.display()))?;
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let japanese = translator.translate(line)?;
            println!("EN: {line}");
            println!("JA: {japanese}");
            println!();
        }
    }
    eprintln!("translation metrics: {}", translator.metrics());
    Ok(())
}

fn run_live(args: LiveArgs) -> Result<()> {
    let source = live_capture_source(&args)?;
    let asr_paths = AsrModelPaths::from_dir(&args.asr_model_dir)?;
    let mut engine = AsrEngine::new(&asr_paths, args.threads, args.rule2)?;
    if !args.no_vad {
        match SpeechGate::from_path(&args.vad_model) {
            Ok(vad) => engine.set_vad(vad),
            Err(err) => tracing::warn!(
                %err,
                "Silero VAD unavailable; music will leak into Zipformer"
            ),
        }
    }
    let whisper = if args.no_whisper {
        None
    } else {
        match WhisperRescorer::from_dir(&args.whisper_dir, args.threads) {
            Ok(rescorer) => Some(rescorer),
            Err(err) => {
                tracing::warn!(
                    %err,
                    "Whisper rescorer unavailable; finals stay Zipformer-only"
                );
                None
            }
        }
    };
    let mut translator = OnnxMarianTranslator::from_dir(&args.mt_model_dir)?;

    let running = install_ctrlc()?;
    let label = capture_label(&source);
    let (capture, rx) = start_capture(source, AUDIO_QUEUE_CAPACITY)?;
    let (mt_tx, mt_rx) = bounded::<TranslationJob>(TRANSLATION_QUEUE_CAPACITY);
    let (done_tx, done_rx) = bounded::<()>(1);

    let mt_thread = std::thread::spawn(move || -> Result<TranslationMetrics> {
        let mut e2e = TranslationMetrics::new()?;
        while let Ok(job) = mt_rx.recv() {
            match caption_final(&whisper, &mut translator, job) {
                Ok(Some(metrics_elapsed)) => {
                    e2e.record(metrics_elapsed);
                }
                Ok(None) => {}
                Err(err) => tracing::error!(%err, "caption failed"),
            }
        }
        let _ = done_tx.send(());
        Ok(e2e)
    });

    let mut display = TranscriptDisplay::new();
    let mut buffer = Vec::new();
    let mut sample_rate = 0_u32;
    let mut translation_queue_drops = 0_u64;

    println!(
        "Live EN->JA from {label}. Partials are Zipformer; finals are VAD-gated, Whisper-rescored, then translated. Press Ctrl+C to stop."
    );
    while running.load(Ordering::SeqCst) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => {
                sample_rate = chunk.sample_rate;
                buffer.extend_from_slice(&chunk.samples);
                let chunk_len = samples_for_ms(sample_rate, CHUNK_MS).max(1);
                while buffer.len() >= chunk_len {
                    let samples: Vec<f32> = buffer.drain(..chunk_len).collect();
                    let events = engine.push_samples(sample_rate, &samples)?;
                    translation_queue_drops += enqueue_finals(&mt_tx, events, &mut display);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    if sample_rate > 0 && !buffer.is_empty() {
        let events = engine.push_samples(sample_rate, &buffer)?;
        translation_queue_drops += enqueue_finals(&mt_tx, events, &mut display);
    }
    let events = engine.finish_input()?;
    translation_queue_drops += enqueue_finals(&mt_tx, events, &mut display);
    display.finish();

    drop(mt_tx);
    let _ = done_rx.recv_timeout(Duration::from_secs(30));
    engine.metrics_mut().add_drops(capture.drops());
    drop(capture);

    eprintln!("ASR metrics: {}", engine.metrics());
    eprintln!("translation queue drops: {translation_queue_drops}");
    match mt_thread.join() {
        Ok(Ok(e2e)) => eprintln!("utterance-end to subtitle: {e2e}"),
        Ok(Err(err)) => tracing::error!(%err, "translation thread failed"),
        Err(_) => tracing::error!("translation thread panicked"),
    }
    Ok(())
}

fn asr_capture_source(args: &AsrArgs) -> Result<Option<CaptureSource>> {
    let selected = usize::from(args.wav.is_some())
        + usize::from(args.mic)
        + usize::from(args.loopback)
        + usize::from(args.app.is_some());
    if selected != 1 {
        anyhow::bail!("specify exactly one of --wav, --mic, --loopback, or --app");
    }
    if args.wav.is_some() {
        return Ok(None);
    }
    Ok(Some(selected_capture_source(
        args.loopback,
        args.app.as_deref(),
    )))
}

fn live_capture_source(args: &LiveArgs) -> Result<CaptureSource> {
    let selected =
        usize::from(args.mic) + usize::from(args.loopback) + usize::from(args.app.is_some());
    if selected > 1 {
        anyhow::bail!("specify at most one of --mic, --loopback, or --app");
    }
    Ok(selected_capture_source(args.loopback, args.app.as_deref()))
}

fn selected_capture_source(loopback: bool, app: Option<&str>) -> CaptureSource {
    if loopback {
        CaptureSource::Loopback
    } else if let Some(app) = app {
        CaptureSource::Application(app.to_string())
    } else {
        CaptureSource::Microphone
    }
}

fn capture_label(source: &CaptureSource) -> String {
    match source {
        CaptureSource::Microphone => "microphone".to_string(),
        CaptureSource::Loopback => "system output".to_string(),
        CaptureSource::Application(app) => format!("app {app}"),
    }
}

fn enqueue_finals(
    tx: &crossbeam_channel::Sender<TranslationJob>,
    events: Vec<AsrEvent>,
    display: &mut TranscriptDisplay,
) -> u64 {
    let mut drops = 0;
    for event in events {
        match event {
            AsrEvent::Partial(text) => display.partial(text),
            AsrEvent::Final(utterance) => {
                display.clear_partial();
                match tx.try_send(TranslationJob {
                    zipformer: utterance.text,
                    samples: utterance.samples,
                    sample_rate: utterance.sample_rate,
                    utterance_ended: Instant::now(),
                }) {
                    Ok(()) => {}
                    Err(_) => drops += 1,
                }
            }
        }
    }
    drops
}

fn caption_final(
    whisper: &Option<WhisperRescorer>,
    translator: &mut OnnxMarianTranslator,
    job: TranslationJob,
) -> Result<Option<Duration>> {
    let english = rescore_english(whisper, &job)?;
    let Some(english) = text::prepare_english(&english) else {
        tracing::debug!(zipformer = %job.zipformer, "skipped junk English");
        return Ok(None);
    };

    println!("EN: {english}");
    let _ = io::stdout().flush();

    let japanese = translator.translate(&english)?;
    let elapsed = job.utterance_ended.elapsed();
    if let Some(japanese) = text::keep_japanese(&japanese) {
        println!("JA: {japanese}");
        let _ = io::stdout().flush();
    } else {
        tracing::debug!(%english, "skipped junk Japanese");
    }
    Ok(Some(elapsed))
}

fn rescore_english(whisper: &Option<WhisperRescorer>, job: &TranslationJob) -> Result<String> {
    let Some(whisper) = whisper else {
        return Ok(job.zipformer.clone());
    };
    match whisper.transcribe(job.sample_rate, &job.samples)? {
        Some(text) => Ok(text),
        None => Ok(job.zipformer.clone()),
    }
}

fn install_ctrlc() -> Result<Arc<AtomicBool>> {
    let running = Arc::new(AtomicBool::new(true));
    let flag = Arc::clone(&running);
    ctrlc::set_handler(move || {
        flag.store(false, Ordering::SeqCst);
    })
    .context("install Ctrl+C handler")?;
    Ok(running)
}

struct TranscriptDisplay {
    last_partial: String,
}

impl TranscriptDisplay {
    fn new() -> Self {
        Self {
            last_partial: String::new(),
        }
    }

    fn handle(&mut self, events: Vec<AsrEvent>) {
        for event in events {
            match event {
                AsrEvent::Partial(text) => self.partial(text),
                AsrEvent::Final(FinalUtterance { text, .. }) => {
                    self.clear_partial();
                    println!("EN: {text}");
                    let _ = io::stdout().flush();
                }
            }
        }
    }

    fn partial(&mut self, text: String) {
        self.last_partial = text;
        print!("\rEN: {}", self.last_partial);
        let _ = io::stdout().flush();
    }

    fn clear_partial(&mut self) {
        if !self.last_partial.is_empty() {
            print!("\r");
            self.last_partial.clear();
        }
    }

    fn finish(&self) {
        if !self.last_partial.is_empty() {
            println!();
        }
    }
}
