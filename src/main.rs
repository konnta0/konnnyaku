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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
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
    /// Keep captions in the terminal. On macOS the default is a floating window.
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
    /// Accepted so older commands still parse. Summary always stays in the terminal.
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
    /// Caption row this final closes. Partials of the next utterance use a newer id.
    generation: u64,
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
    #[cfg(target_os = "macos")]
    if !args.cli {
        return run_live_hud(args);
    }
    run_live_session(args, None, None)
}

#[cfg(target_os = "macos")]
fn run_live_hud(args: LiveArgs) -> Result<()> {
    let running = install_ctrlc()?;
    let (tx, rx) = bounded::<String>(32);
    let capture_running = Arc::clone(&running);
    let join = std::thread::Builder::new()
        .name("konnnyaku-live".into())
        .spawn(move || {
            let result = run_live_session(args, Some(Arc::clone(&capture_running)), Some(tx));
            capture_running.store(false, Ordering::SeqCst);
            result
        })
        .context("spawn live thread")?;
    let ui = crate::overlay::run_caption_hud(Arc::clone(&running), rx);
    running.store(false, Ordering::SeqCst);
    let capture_result = join
        .join()
        .map_err(|_| anyhow::anyhow!("live thread panicked"))?;
    ui?;
    capture_result
}

fn run_live_session(
    args: LiveArgs,
    running: Option<Arc<AtomicBool>>,
    captions: Option<Sender<String>>,
) -> Result<()> {
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

    let running = match running {
        Some(running) => running,
        None => install_ctrlc()?,
    };
    let show_terminal = captions.is_none();
    let label = capture_label(&source);
    let (mut capture, mut rx) = start_capture(source.clone(), AUDIO_QUEUE_CAPACITY)?;
    let (mt_tx, mt_rx) = bounded::<TranslationJob>(TRANSLATION_QUEUE_CAPACITY);
    let (done_tx, done_rx) = bounded::<()>(1);
    let (caption_feed, caption_worker) = match captions.clone() {
        Some(ui) => {
            let (kick_tx, kick_rx) = bounded::<()>(1);
            let tail = Arc::new(CaptionTail::new());
            (
                Some(CaptionFeed {
                    ui: ui.clone(),
                    tail: Arc::clone(&tail),
                    kick: kick_tx,
                }),
                Some((ui, tail, kick_rx)),
            )
        }
        None => (None, None),
    };

    let mt_thread = std::thread::spawn(move || -> Result<TranslationMetrics> {
        let mut e2e = TranslationMetrics::new()?;
        if let Some((ui, tail, kick_rx)) = caption_worker {
            let mut translated = TailTranslation::default();
            loop {
                select! {
                    recv(mt_rx) -> job => {
                        match job {
                            Ok(job) => {
                                record_caption(&mut e2e, &whisper, &mut translator, job, Some(&ui));
                            }
                            Err(_) => break,
                        }
                    }
                    recv(kick_rx) -> msg => {
                        if msg.is_ok() {
                            translate_tail(&tail, &mut translator, &ui, &mut translated);
                        } else {
                            while let Ok(job) = mt_rx.recv() {
                                record_caption(
                                    &mut e2e,
                                    &whisper,
                                    &mut translator,
                                    job,
                                    Some(&ui),
                                );
                            }
                            break;
                        }
                    }
                }
            }
        } else {
            while let Ok(job) = mt_rx.recv() {
                record_caption(&mut e2e, &whisper, &mut translator, job, None);
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

    if show_terminal {
        println!(
            "Live EN->JA from {label}. Partials are Zipformer; finals are VAD-gated, Whisper-rescored, then translated. Press Ctrl+C to stop."
        );
    } else {
        eprintln!("Translation from {label}. Close the window or press Ctrl+C to stop.");
    }
    while running.load(Ordering::SeqCst) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => {
                sample_rate = chunk.sample_rate;
                buffer.extend_from_slice(&chunk.samples);
                let chunk_len = samples_for_ms(sample_rate, CHUNK_MS).max(1);
                while buffer.len() >= chunk_len {
                    let samples: Vec<f32> = buffer.drain(..chunk_len).collect();
                    let events = engine.push_samples(sample_rate, &samples)?;
                    translation_queue_drops +=
                        enqueue_finals(&mt_tx, events, &mut display, caption_feed.as_ref());
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
        translation_queue_drops +=
            enqueue_finals(&mt_tx, events, &mut display, caption_feed.as_ref());
    }
    let events = engine.finish_input()?;
    translation_queue_drops += enqueue_finals(&mt_tx, events, &mut display, caption_feed.as_ref());
    display.finish();

    drop(caption_feed);
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
    let _ = args.cli;
    run_summary_capture(args)
}

fn run_summary_capture(args: SummaryArgs) -> Result<()> {
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
            &mut session,
            &mut translator,
            &whisper,
            &mut cache,
            engine.finish_input()?,
        )?;
    } else {
        let source = summary_capture_source(&args)?;
        source_label = capture_label(&source);
        let running = install_ctrlc()?;
        let (mut capture, mut rx) = start_capture(source.clone(), AUDIO_QUEUE_CAPACITY)?;
        eprintln!("Meeting note from {source_label}. Press Ctrl+C to stop.");
        let mut buffer = Vec::new();
        let mut sample_rate = 0_u32;
        while running.load(Ordering::SeqCst) {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(chunk) => {
                    sample_rate = chunk.sample_rate;
                    buffer.extend_from_slice(&chunk.samples);
                    let chunk_len = samples_for_ms(sample_rate, CHUNK_MS).max(1);
                    while buffer.len() >= chunk_len {
                        let samples: Vec<f32> = buffer.drain(..chunk_len).collect();
                        note_events(
                            &source_label,
                            args.json,
                            &mut session,
                            &mut translator,
                            &whisper,
                            &mut cache,
                            engine.push_samples(sample_rate, &samples)?,
                        )?;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    let _ = reconnect_capture(&mut capture, &mut rx, &source, false)?;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    let _ = reconnect_capture(&mut capture, &mut rx, &source, true)?;
                }
            }
        }
        if sample_rate > 0 && !buffer.is_empty() {
            note_events(
                &source_label,
                args.json,
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
        } else {
            println!("会議メモにできる発話がまだありません。");
        }
    }
    Ok(())
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
                generation: 0,
            },
        )?;
        changed |= session.push(&english);
    }
    if !changed {
        return Ok(());
    }
    publish_note(source, json, &session.draft(), translator, cache)
}

fn publish_note(
    source: &str,
    json: bool,
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
    if json {
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

struct CaptionTail {
    rev: AtomicU64,
    text: Mutex<String>,
}

impl CaptionTail {
    fn new() -> Self {
        Self {
            rev: AtomicU64::new(0),
            text: Mutex::new(String::new()),
        }
    }

    fn publish(&self, text: String) -> u64 {
        if let Ok(mut slot) = self.text.lock() {
            *slot = text;
        }
        self.rev.load(Ordering::Acquire)
    }

    fn close_utterance(&self) -> u64 {
        let closed = self.rev.fetch_add(1, Ordering::Release);
        if let Ok(mut slot) = self.text.lock() {
            slot.clear();
        }
        closed
    }

    fn snapshot(&self) -> (u64, String) {
        let rev = self.rev.load(Ordering::Acquire);
        let text = self
            .text
            .lock()
            .map(|slot| slot.clone())
            .unwrap_or_default();
        (rev, text)
    }
}

struct CaptionFeed {
    ui: Sender<String>,
    tail: Arc<CaptionTail>,
    kick: Sender<()>,
}

fn enqueue_finals(
    tx: &crossbeam_channel::Sender<TranslationJob>,
    events: Vec<AsrEvent>,
    display: &mut TranscriptDisplay,
    feed: Option<&CaptionFeed>,
) -> u64 {
    let mut drops = 0;
    for event in events {
        match event {
            AsrEvent::Partial(text) => {
                if let Some(feed) = feed {
                    let rev = feed.tail.publish(text.clone());
                    send_caption(
                        &feed.ui,
                        &serde_json::json!({
                            "kind": "preview",
                            "rev": rev,
                            "english": text,
                            "single": false,
                        }),
                        false,
                    );
                    let _ = feed.kick.try_send(());
                } else {
                    display.partial(text);
                }
            }
            AsrEvent::Final(utterance) => {
                let generation = if let Some(feed) = feed {
                    feed.tail.close_utterance()
                } else {
                    display.clear_partial();
                    0
                };
                let job = TranslationJob {
                    zipformer: utterance.text,
                    samples: utterance.samples,
                    sample_rate: utterance.sample_rate,
                    utterance_ended: Instant::now(),
                    generation,
                };
                if let Err(err) = tx.try_send(job) {
                    drops += 1;
                    if let Some(feed) = feed {
                        let job = match err {
                            crossbeam_channel::TrySendError::Full(job)
                            | crossbeam_channel::TrySendError::Disconnected(job) => job,
                        };
                        send_caption(
                            &feed.ui,
                            &serde_json::json!({
                                "kind": "commit",
                                "rev": job.generation,
                                "english": job.zipformer,
                                "japanese": "",
                                "single": false,
                            }),
                            true,
                        );
                    }
                }
            }
        }
    }
    drops
}

fn record_caption(
    metrics: &mut TranslationMetrics,
    whisper: &Option<WhisperRescorer>,
    translator: &mut OnnxMarianTranslator,
    job: TranslationJob,
    captions: Option<&Sender<String>>,
) {
    match caption_final(whisper, translator, job, captions) {
        Ok(Some(elapsed)) => metrics.record(elapsed),
        Ok(None) => {}
        Err(err) => tracing::error!(%err, "caption failed"),
    }
}

#[derive(Default)]
struct TailTranslation {
    rev: u64,
    text: String,
    japanese: String,
}

fn translate_tail(
    tail: &CaptionTail,
    translator: &mut OnnxMarianTranslator,
    captions: &Sender<String>,
    translated: &mut TailTranslation,
) {
    loop {
        let (rev, text) = tail.snapshot();
        let text = text.trim().to_string();
        if text.is_empty() {
            *translated = TailTranslation::default();
            return;
        }
        if text == translated.text {
            if rev != translated.rev && !translated.japanese.is_empty() {
                send_caption(
                    captions,
                    &serde_json::json!({
                        "kind": "preview",
                        "rev": rev,
                        "japanese": translated.japanese,
                        "single": false,
                    }),
                    false,
                );
                translated.rev = rev;
            }
            return;
        }
        let japanese = match translator.translate(&text) {
            Ok(japanese) => text::keep_japanese(&japanese).unwrap_or_default(),
            Err(err) => {
                tracing::warn!(%err, "live caption translation failed");
                return;
            }
        };
        let (now, current) = tail.snapshot();
        if now != rev || current.trim() != text {
            continue;
        }
        translated.rev = rev;
        translated.text = text;
        translated.japanese = japanese;
        if translated.japanese.is_empty() {
            return;
        }
        send_caption(
            captions,
            &serde_json::json!({
                "kind": "preview",
                "rev": rev,
                "japanese": translated.japanese,
                "single": false,
            }),
            false,
        );
        return;
    }
}

fn caption_final(
    whisper: &Option<WhisperRescorer>,
    translator: &mut OnnxMarianTranslator,
    job: TranslationJob,
    captions: Option<&Sender<String>>,
) -> Result<Option<Duration>> {
    let english = rescore_english(whisper, &job)?;
    let Some(english) = text::prepare_english(&english) else {
        tracing::debug!(zipformer = %job.zipformer, "skipped junk English");
        if let Some(tx) = captions {
            send_caption(
                tx,
                &serde_json::json!({
                    "kind": "drop",
                    "rev": job.generation,
                }),
                true,
            );
        }
        return Ok(None);
    };

    let japanese = translator.translate(&english)?;
    let elapsed = job.utterance_ended.elapsed();
    let japanese = text::keep_japanese(&japanese).unwrap_or_default();
    if let Some(tx) = captions {
        send_caption(
            tx,
            &serde_json::json!({
                "kind": "commit",
                "rev": job.generation,
                "english": english,
                "japanese": japanese,
                "single": false,
            }),
            true,
        );
    } else {
        println!("EN: {english}");
        if !japanese.is_empty() {
            println!("JA: {japanese}");
        } else {
            tracing::debug!(%english, "skipped junk Japanese");
        }
        let _ = io::stdout().flush();
    }
    Ok(Some(elapsed))
}

fn send_caption(tx: &Sender<String>, value: &impl serde::Serialize, reliable: bool) {
    match hud_json(value) {
        Ok(payload) => {
            let sent = if reliable {
                tx.send_timeout(payload, Duration::from_millis(500)).is_ok()
            } else {
                tx.try_send(payload).is_ok()
            };
            if !sent {
                tracing::debug!("caption window missed an update");
            }
        }
        Err(err) => tracing::warn!(%err, "caption json failed"),
    }
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
