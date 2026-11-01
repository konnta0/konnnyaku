mod asr;
mod audio;
mod metrics;
#[cfg(target_os = "macos")]
mod overlay;
mod summary;
mod text;
mod translation;
mod vad;
mod whisper;

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded, select};
use tracing_subscriber::EnvFilter;

use crate::asr::{AsrEngine, AsrEvent, AsrModelPaths, FinalUtterance};
use crate::audio::{
    AudioCapture, AudioChunk, CaptureSource, chunk_samples, list_tappable_apps, load_wav_mono_f32,
    samples_for_ms, start_capture,
};
use crate::metrics::TranslationMetrics;
use crate::summary::{MeetingDraft, SummarySession};
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
    /// Summarize microphone speech. This does not translate playback.
    Summary(SummaryArgs),
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
    /// English rescoring model. Whisper files, or a Parakeet directory with a joiner.
    #[arg(long, default_value = "models/asr/whisper-tiny.en")]
    whisper_dir: PathBuf,
    /// Silero VAD ONNX path.
    #[arg(long, default_value = "models/asr/silero_vad.onnx")]
    vad_model: PathBuf,
    /// Disable English rescoring of Zipformer finals.
    #[arg(long, default_value_t = false)]
    no_whisper: bool,
    /// Disable Silero VAD music suppression.
    #[arg(long, default_value_t = false)]
    no_vad: bool,
}

#[derive(clap::Args, Debug)]
struct SummaryArgs {
    /// 16-bit or 32-bit WAV path. Omit this to capture live audio.
    #[arg(long)]
    wav: Option<PathBuf>,
    /// Capture from the default microphone. This is the default.
    #[arg(long, default_value_t = false)]
    mic: bool,
    /// Capture default system output, such as a meeting on the speakers.
    #[arg(long, default_value_t = false)]
    loopback: bool,
    /// Capture one application's output (name or bundle ID, macOS).
    #[arg(long)]
    app: Option<String>,
    /// Print the meeting note as JSON on stdout. The file is written either way.
    #[arg(long, default_value_t = false)]
    json: bool,
    /// Keep the note in the terminal. On macOS the default is a floating window.
    #[arg(long, default_value_t = false)]
    cli: bool,
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
    /// English rescoring model. Whisper files, or a Parakeet directory with a joiner.
    #[arg(long, default_value = "models/asr/whisper-tiny.en")]
    whisper_dir: PathBuf,
    /// Silero VAD ONNX path.
    #[arg(long, default_value = "models/asr/silero_vad.onnx")]
    vad_model: PathBuf,
    /// Disable English rescoring of Zipformer finals.
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
        Command::Summary(args) => run_summary(args),
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
    let (mut capture, mut rx) = start_capture(source.clone(), AUDIO_QUEUE_CAPACITY)?;
    let mut display = TranscriptDisplay::new();
    let mut buffer = Vec::new();
    let mut sample_rate = 0_u32;
    let mut capture_drops = 0_u64;

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
            Err(RecvTimeoutError::Timeout) => {
                capture_drops += reconnect_capture(&mut capture, &mut rx, &source, false)?;
            }
            Err(RecvTimeoutError::Disconnected) => {
                capture_drops += reconnect_capture(&mut capture, &mut rx, &source, true)?;
            }
        }
    }

    if sample_rate > 0 && !buffer.is_empty() {
        display.handle(engine.push_samples(sample_rate, &buffer)?);
    }
    display.handle(engine.finish_input()?);
    display.finish();
    engine
        .metrics_mut()
        .add_drops(capture_drops + capture.drops());
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
    let (mut capture, mut rx) = start_capture(source.clone(), AUDIO_QUEUE_CAPACITY)?;
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
    let mut capture_drops = 0_u64;

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
            Err(RecvTimeoutError::Timeout) => {
                capture_drops += reconnect_capture(&mut capture, &mut rx, &source, false)?;
            }
            Err(RecvTimeoutError::Disconnected) => {
                capture_drops += reconnect_capture(&mut capture, &mut rx, &source, true)?;
            }
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
    engine
        .metrics_mut()
        .add_drops(capture_drops + capture.drops());
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

fn run_summary(args: SummaryArgs) -> Result<()> {
    #[cfg(target_os = "macos")]
    if args.wav.is_none() && !args.cli {
        return run_summary_hud(args);
    }
    run_summary_capture(args, None)
}

#[cfg(target_os = "macos")]
fn run_summary_hud(args: SummaryArgs) -> Result<()> {
    let running = install_ctrlc()?;
    let (tx, rx) = bounded::<String>(8);
    let (summarize_tx, summarize_rx) = bounded::<()>(1);
    let capture_running = Arc::clone(&running);
    let join = std::thread::Builder::new()
        .name("konnnyaku-summary".into())
        .spawn(move || {
            let result = run_summary_capture(
                args,
                Some(MeetingWindow {
                    running: Arc::clone(&capture_running),
                    notes: tx,
                    summarize: summarize_rx,
                }),
            );
            capture_running.store(false, Ordering::SeqCst);
            result
        })
        .context("spawn summary thread")?;
    let ui = crate::overlay::run_meeting_hud(Arc::clone(&running), rx, summarize_tx);
    running.store(false, Ordering::SeqCst);
    let capture_result = join
        .join()
        .map_err(|_| anyhow::anyhow!("summary thread panicked"))?;
    ui?;
    capture_result
}

struct MeetingWindow {
    running: Arc<AtomicBool>,
    notes: Sender<String>,
    summarize: Receiver<()>,
}

fn run_summary_capture(args: SummaryArgs, hud: Option<MeetingWindow>) -> Result<()> {
    let asr_paths = AsrModelPaths::from_dir(&args.asr_model_dir)?;
    let mut engine = AsrEngine::new(&asr_paths, args.threads, args.rule2)?;
    if !args.no_vad {
        match SpeechGate::from_path(&args.vad_model) {
            Ok(vad) => engine.set_vad(vad),
            Err(err) => tracing::warn!(
                %err,
                "Silero VAD unavailable; music will leak into the summary"
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
                    "English rescorer unavailable; summary stays on the Zipformer text"
                );
                None
            }
        }
    };
    let mut translator = OnnxMarianTranslator::from_dir(&args.mt_model_dir)?;
    let mut session = SummarySession::new();
    let mut cache = NoteCache::default();
    let source_label;

    if let Some(wav) = &args.wav {
        if args.mic || args.loopback || args.app.is_some() {
            anyhow::bail!("--wav cannot be combined with --mic, --loopback, or --app");
        }
        source_label = format!("wav {}", wav.display());
        let audio = load_wav_mono_f32(wav)?;
        tracing::info!(
            path = %wav.display(),
            sample_rate = audio.sample_rate,
            samples = audio.samples.len(),
            "loaded wav for summary"
        );
        for chunk in chunk_samples(&audio.samples, audio.sample_rate, CHUNK_MS) {
            note_events(
                &source_label,
                args.json,
                None,
                &mut session,
                &mut translator,
                &whisper,
                &mut cache,
                engine.push_samples(audio.sample_rate, &chunk)?,
            )?;
        }
        note_events(
            &source_label,
            args.json,
            None,
            &mut session,
            &mut translator,
            &whisper,
            &mut cache,
            engine.finish_input()?,
        )?;
    } else {
        let source = summary_capture_source(&args)?;
        source_label = capture_label(&source);
        let running = match &hud {
            Some(window) => Arc::clone(&window.running),
            None => install_ctrlc()?,
        };
        let (mut capture, mut rx) = start_capture(source.clone(), AUDIO_QUEUE_CAPACITY)?;
        if hud.is_some() {
            eprintln!(
                "Meeting note from {source_label}. Press 要約する in the window, or close it to stop."
            );
        } else {
            eprintln!("Meeting note from {source_label}. Press Ctrl+C to stop.");
        }
        let mut buffer = Vec::new();
        let mut sample_rate = 0_u32;
        while running.load(Ordering::SeqCst) {
            match wait_capture(&rx, hud.as_ref().map(|window| &window.summarize)) {
                CaptureWait::Chunk(chunk) => {
                    sample_rate = chunk.sample_rate;
                    buffer.extend_from_slice(&chunk.samples);
                    let chunk_len = samples_for_ms(sample_rate, CHUNK_MS).max(1);
                    while buffer.len() >= chunk_len {
                        let samples: Vec<f32> = buffer.drain(..chunk_len).collect();
                        note_events(
                            &source_label,
                            args.json,
                            hud.as_ref(),
                            &mut session,
                            &mut translator,
                            &whisper,
                            &mut cache,
                            engine.push_samples(sample_rate, &samples)?,
                        )?;
                    }
                }
                CaptureWait::Summarize => {
                    if let Err(err) = publish_note(
                        &source_label,
                        args.json,
                        hud.as_ref().map(|window| &window.notes),
                        &session.draft(),
                        &mut translator,
                        &mut cache,
                    ) {
                        tracing::error!(%err, "meeting summary failed");
                        if let Some(window) = hud.as_ref() {
                            send_summary_error(&window.notes, &session.draft());
                        }
                    }
                }
                CaptureWait::Timeout => {
                    let _ = reconnect_capture(&mut capture, &mut rx, &source, false)?;
                }
                CaptureWait::Disconnected => {
                    let _ = reconnect_capture(&mut capture, &mut rx, &source, true)?;
                }
            }
        }
        if sample_rate > 0 && !buffer.is_empty() {
            note_events(
                &source_label,
                args.json,
                hud.as_ref(),
                &mut session,
                &mut translator,
                &whisper,
                &mut cache,
                engine.push_samples(sample_rate, &buffer)?,
            )?;
        }
        note_events(
            &source_label,
            args.json,
            hud.as_ref(),
            &mut session,
            &mut translator,
            &whisper,
            &mut cache,
            engine.finish_input()?,
        )?;
        drop(capture);
    }

    if cache.wrote_sections.is_none() {
        if cache.wrote_file {
            println!("文字起こしだけを meetings/latest.json に書きました。");
        } else if !session.draft().transcript.is_empty() {
            println!("要約ボタンが押されなかったので meetings/latest.json は更新していません。");
        } else {
            println!("会議メモにできる発話がまだありません。");
        }
    }
    Ok(())
}

enum CaptureWait {
    Chunk(AudioChunk),
    Summarize,
    Timeout,
    Disconnected,
}

fn wait_capture(audio: &Receiver<AudioChunk>, summarize: Option<&Receiver<()>>) -> CaptureWait {
    let Some(summarize) = summarize else {
        return match audio.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => CaptureWait::Chunk(chunk),
            Err(RecvTimeoutError::Timeout) => CaptureWait::Timeout,
            Err(RecvTimeoutError::Disconnected) => CaptureWait::Disconnected,
        };
    };
    select! {
        recv(audio) -> msg => match msg {
            Ok(chunk) => CaptureWait::Chunk(chunk),
            Err(_) => CaptureWait::Disconnected,
        },
        recv(summarize) -> msg => {
            if msg.is_ok() {
                CaptureWait::Summarize
            } else {
                CaptureWait::Disconnected
            }
        }
        default(Duration::from_millis(100)) => CaptureWait::Timeout,
    }
}

#[derive(Default)]
struct NoteCache {
    japanese: HashMap<String, String>,
    wrote_file: bool,
    wrote_sections: Option<(Vec<String>, Vec<String>, Vec<String>)>,
}

fn summary_capture_source(args: &SummaryArgs) -> Result<CaptureSource> {
    let selected =
        usize::from(args.mic) + usize::from(args.loopback) + usize::from(args.app.is_some());
    if selected > 1 {
        anyhow::bail!("specify at most one of --mic, --loopback, or --app");
    }
    Ok(selected_capture_source(args.loopback, args.app.as_deref()))
}

fn note_events(
    source: &str,
    json: bool,
    hud: Option<&MeetingWindow>,
    session: &mut SummarySession,
    translator: &mut OnnxMarianTranslator,
    whisper: &Option<WhisperRescorer>,
    cache: &mut NoteCache,
    events: Vec<AsrEvent>,
) -> Result<()> {
    let mut changed = false;
    for event in events {
        let AsrEvent::Final(utterance) = event else {
            continue;
        };
        let english = rescore_english(
            whisper,
            &TranslationJob {
                zipformer: utterance.text,
                samples: utterance.samples,
                sample_rate: utterance.sample_rate,
                utterance_ended: Instant::now(),
            },
        )?;
        changed |= session.push(&english);
    }
    if !changed {
        return Ok(());
    }
    if let Some(window) = hud {
        send_hearing(&window.notes, &session.draft().transcript);
        return Ok(());
    }
    publish_note(source, json, None, &session.draft(), translator, cache)
}

fn publish_note(
    source: &str,
    json: bool,
    hud: Option<&Sender<String>>,
    draft: &MeetingDraft,
    translator: &mut OnnxMarianTranslator,
    cache: &mut NoteCache,
) -> Result<()> {
    let topics = translate_items(&draft.topics, translator, cache);
    let decisions = translate_items(&draft.decisions, translator, cache);
    let actions = translate_items(&draft.actions, translator, cache);
    let overview = topics
        .iter()
        .map(|item| item.japanese.as_str())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("");
    let note = MeetingNote {
        source: source.to_string(),
        transcript: draft.transcript.clone(),
        overview,
        topics,
        decisions,
        actions,
    };
    let dir = Path::new("meetings");
    std::fs::create_dir_all(dir).context("create meetings directory")?;
    let path = dir.join("latest.json");
    std::fs::write(&path, serde_json::to_string_pretty(&note)?)
        .with_context(|| format!("write {}", path.display()))?;

    let sections = (
        draft.topics.clone(),
        draft.decisions.clone(),
        draft.actions.clone(),
    );
    let sections_changed = cache.wrote_sections.as_ref() != Some(&sections);
    if let Some(tx) = hud {
        let view = HudNote {
            kind: "note",
            lines: &note.transcript,
            overview: &note.overview,
            topics: &note.topics,
            decisions: &note.decisions,
            actions: &note.actions,
            error: None,
        };
        match hud_json(&view) {
            Ok(payload) => {
                if tx.send_timeout(payload, Duration::from_secs(2)).is_err() {
                    tracing::debug!("meeting window missed the summary");
                }
            }
            Err(err) => tracing::warn!(%err, "meeting note json failed"),
        }
    } else if json {
        println!("{}", serde_json::to_string(&note)?);
        let _ = io::stdout().flush();
    } else if sections_changed && note.has_sections() {
        print_note(&note);
    }
    cache.wrote_file = true;
    if note.has_sections() {
        cache.wrote_sections = Some(sections);
    }
    Ok(())
}

fn translate_items(
    lines: &[String],
    translator: &mut OnnxMarianTranslator,
    cache: &mut NoteCache,
) -> Vec<MeetingItem> {
    lines
        .iter()
        .map(|english| MeetingItem {
            japanese: cached_japanese(english, translator, cache),
            english: english.clone(),
        })
        .collect()
}

fn cached_japanese(
    english: &str,
    translator: &mut OnnxMarianTranslator,
    cache: &mut NoteCache,
) -> String {
    if let Some(japanese) = cache.japanese.get(english) {
        return japanese.clone();
    }
    let japanese = match translator.translate(english) {
        Ok(japanese) => text::keep_japanese(&japanese).unwrap_or_default(),
        Err(err) => {
            tracing::warn!(%err, %english, "meeting line failed");
            String::new()
        }
    };
    cache.japanese.insert(english.to_string(), japanese.clone());
    japanese
}

fn print_note(note: &MeetingNote) {
    println!("会議メモ");
    if !note.overview.is_empty() {
        println!("概要");
        println!("{}", note.overview);
    }
    print_section("議題", &note.topics);
    print_section("決定", &note.decisions);
    print_section("次のアクション", &note.actions);
    println!("wrote meetings/latest.json");
    println!();
    let _ = io::stdout().flush();
}

fn print_section(title: &str, items: &[MeetingItem]) {
    let visible: Vec<&MeetingItem> = items
        .iter()
        .filter(|item| !item.japanese.is_empty())
        .collect();
    if visible.is_empty() {
        return;
    }
    println!("{title}");
    for item in visible {
        println!("- {}", item.japanese);
    }
}

#[derive(serde::Serialize)]
struct MeetingNote {
    source: String,
    transcript: Vec<String>,
    overview: String,
    topics: Vec<MeetingItem>,
    decisions: Vec<MeetingItem>,
    actions: Vec<MeetingItem>,
}

impl MeetingNote {
    fn has_sections(&self) -> bool {
        !self.topics.is_empty() || !self.decisions.is_empty() || !self.actions.is_empty()
    }
}

#[derive(serde::Serialize)]
struct MeetingItem {
    english: String,
    japanese: String,
}

#[derive(serde::Serialize)]
struct HudNote<'a> {
    kind: &'static str,
    lines: &'a [String],
    overview: &'a str,
    topics: &'a [MeetingItem],
    decisions: &'a [MeetingItem],
    actions: &'a [MeetingItem],
    error: Option<&'a str>,
}

#[derive(serde::Serialize)]
struct HudHearing<'a> {
    kind: &'static str,
    lines: &'a [String],
}

fn send_hearing(tx: &Sender<String>, lines: &[String]) {
    let view = HudHearing {
        kind: "hearing",
        lines,
    };
    if let Ok(payload) = hud_json(&view) {
        let _ = tx.try_send(payload);
    }
}

fn send_summary_error(tx: &Sender<String>, draft: &MeetingDraft) {
    let view = HudNote {
        kind: "note",
        lines: &draft.transcript,
        overview: "",
        topics: &[],
        decisions: &[],
        actions: &[],
        error: Some("要約に失敗しました"),
    };
    if let Ok(payload) = hud_json(&view) {
        let _ = tx.send_timeout(payload, Duration::from_secs(2));
    }
}

fn hud_json(value: &impl serde::Serialize) -> Result<String> {
    let json = serde_json::to_string(value)?;
    Ok(json
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029"))
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

fn reconnect_capture(
    capture: &mut AudioCapture,
    rx: &mut Receiver<AudioChunk>,
    source: &CaptureSource,
    force: bool,
) -> Result<u64> {
    if !force && !capture.is_invalidated() {
        return Ok(0);
    }
    println!("Audio stream dropped (device sample rate or route changed). Reconnecting...");
    let _ = io::stdout().flush();
    let drops = capture.drops();
    std::thread::sleep(Duration::from_millis(400));
    match start_capture(source.clone(), AUDIO_QUEUE_CAPACITY) {
        Ok((next, next_rx)) => {
            *capture = next;
            *rx = next_rx;
            println!("Capture reconnected.");
            let _ = io::stdout().flush();
            Ok(drops)
        }
        Err(err) => {
            tracing::warn!(%err, "capture reconnect failed");
            Ok(0)
        }
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
