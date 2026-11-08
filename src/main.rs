mod asr;
mod audio;
mod haas;
mod metrics;
mod mix;
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
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
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
use crate::haas::HaasRecognizer;
use crate::metrics::TranslationMetrics;
use crate::mix::{AudioMix, MIX_RATE, MixPort};
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
    /// Capture the default microphone. Combine with --app or --loopback to start both.
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
    /// Spoken language. `en` or `ja`.
    #[arg(long, default_value = "en")]
    source: String,
    /// Caption language. `en` or `ja`.
    #[arg(long, default_value = "ja")]
    target: String,
    /// Haas Japanese ONNX directory.
    #[arg(long, default_value = "models/asr/haas-ja")]
    haas_dir: PathBuf,
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
    let langs = Arc::new(LangPair::parse(&args.source, &args.target)?);
    let hear = Arc::new(HearFlags::from_live(&args)?);
    #[cfg(target_os = "macos")]
    if !args.cli {
        return run_live_hud(args, langs, Arc::clone(&hear));
    }
    run_live_session(args, None, None, langs, hear)
}

#[cfg(target_os = "macos")]
fn run_live_hud(args: LiveArgs, langs: Arc<LangPair>, hear: Arc<HearFlags>) -> Result<()> {
    let running = install_ctrlc()?;
    let (tx, rx) = bounded::<String>(32);
    let capture_running = Arc::clone(&running);
    let ui_langs = Arc::clone(&langs);
    let session_hear = Arc::clone(&hear);
    let join = std::thread::Builder::new()
        .name("konnnyaku-live".into())
        .spawn(move || {
            let result = run_live_session(
                args,
                Some(Arc::clone(&capture_running)),
                Some(tx),
                langs,
                session_hear,
            );
            capture_running.store(false, Ordering::SeqCst);
            result
        })
        .context("spawn live thread")?;
    let ui = crate::overlay::run_caption_hud(Arc::clone(&running), rx, ui_langs, hear);
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
    langs: Arc<LangPair>,
    hear: Arc<HearFlags>,
) -> Result<()> {
    let output_source = live_output_source(&args)?;
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
    let label = hear_label(hear.input(), hear.output(), &output_source);
    let mut input_feed = LiveFeed::new(CaptureSource::Microphone);
    let mut output_feed = LiveFeed::new(output_source);
    let mut mix = AudioMix::new(MIX_RATE);
    let (mt_tx, mt_rx) = bounded::<TranslationJob>(TRANSLATION_QUEUE_CAPACITY);
    let (done_tx, done_rx) = bounded::<()>(1);
    let tail = Arc::new(CaptionTail::new());
    let ja_inbox = Arc::new(JaInbox::default());
    let (ja_kick_tx, ja_kick_rx) = bounded::<()>(1);
    let haas_dir = args.haas_dir.clone();
    let haas_threads = args.threads;
    let (en_kick_tx, en_kick_rx) = if captions.is_some() {
        let (kick_tx, kick_rx) = bounded::<()>(1);
        (Some(kick_tx), Some(kick_rx))
    } else {
        (None, None)
    };
    let route = LiveRoute {
        langs: Arc::clone(&langs),
        ui: captions.clone(),
        tail: Arc::clone(&tail),
        en_kick: en_kick_tx,
        ja_inbox: Arc::clone(&ja_inbox),
        ja_kick: ja_kick_tx,
    };
    let worker_langs = Arc::clone(&langs);
    let worker_ui = captions.clone();
    let worker_tail = Arc::clone(&tail);
    let worker_inbox = Arc::clone(&ja_inbox);

    let mt_thread = std::thread::spawn(move || -> Result<TranslationMetrics> {
        let mut e2e = TranslationMetrics::new()?;
        let mut translated = TailTranslation::default();
        let mut ja_model: Option<HaasRecognizer> = None;
        let mut ja_failed = false;
        if let Some(en_kick_rx) = en_kick_rx {
            let ui = worker_ui.expect("caption sender");
            loop {
                select! {
                    recv(mt_rx) -> job => {
                        match job {
                            Ok(job) => record_caption(
                                &mut e2e,
                                &whisper,
                                &mut translator,
                                job,
                                Some(&ui),
                                &worker_langs,
                            ),
                            Err(_) => break,
                        }
                    }
                    recv(en_kick_rx) -> msg => {
                        if msg.is_ok() {
                            translate_tail(
                                &worker_tail,
                                &mut translator,
                                &ui,
                                &mut translated,
                                &worker_langs,
                            );
                        } else {
                            drain_english(&mut e2e, &whisper, &mut translator, &mt_rx, Some(&ui), &worker_langs);
                            break;
                        }
                    }
                    recv(ja_kick_rx) -> msg => {
                        if msg.is_ok() {
                            transcribe_japanese(
                                &worker_inbox,
                                &mut ja_model,
                                &mut ja_failed,
                                &haas_dir,
                                haas_threads,
                                Some(&ui),
                                &worker_langs,
                            );
                        } else {
                            drain_english(&mut e2e, &whisper, &mut translator, &mt_rx, Some(&ui), &worker_langs);
                            break;
                        }
                    }
                }
            }
        } else {
            loop {
                select! {
                    recv(mt_rx) -> job => {
                        match job {
                            Ok(job) => record_caption(
                                &mut e2e,
                                &whisper,
                                &mut translator,
                                job,
                                None,
                                &worker_langs,
                            ),
                            Err(_) => break,
                        }
                    }
                    recv(ja_kick_rx) -> msg => {
                        if msg.is_ok() {
                            transcribe_japanese(
                                &worker_inbox,
                                &mut ja_model,
                                &mut ja_failed,
                                &haas_dir,
                                haas_threads,
                                worker_ui.as_ref(),
                                &worker_langs,
                            );
                        } else {
                            drain_english(&mut e2e, &whisper, &mut translator, &mt_rx, None, &worker_langs);
                            break;
                        }
                    }
                }
            }
        }
        let _ = done_tx.send(());
        Ok(e2e)
    });

    let mut display = TranscriptDisplay::new();
    let mut translation_queue_drops = 0_u64;

    if show_terminal {
        println!(
            "Live captions from {label}. Press Ctrl+C to stop. Source {}, target {}.",
            langs.source().label(),
            langs.target().label()
        );
    } else {
        eprintln!("Captions from {label}. Close the window or press Ctrl+C to stop.");
    }
    pump_live_audio(
        &running,
        &mut engine,
        &hear,
        &mut input_feed,
        &mut output_feed,
        &mut mix,
        &mt_tx,
        &mut display,
        &route,
        &captions,
        &mut translation_queue_drops,
    )?;
    let events = engine.finish_input()?;
    translation_queue_drops += enqueue_finals(&mt_tx, events, &mut display, &route);
    display.finish();

    drop(route);
    drop(mt_tx);
    let _ = done_rx.recv_timeout(Duration::from_secs(30));
    engine
        .metrics_mut()
        .add_drops(input_feed.total_drops() + output_feed.total_drops());
    drop(input_feed);
    drop(output_feed);

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

fn live_output_source(args: &LiveArgs) -> Result<CaptureSource> {
    if args.loopback && args.app.is_some() {
        anyhow::bail!("specify only one of --loopback or --app");
    }
    if args.loopback {
        Ok(CaptureSource::Loopback)
    } else if let Some(app) = &args.app {
        Ok(CaptureSource::Application(app.clone()))
    } else {
        Ok(CaptureSource::Loopback)
    }
}

fn hear_label(input: bool, output: bool, output_source: &CaptureSource) -> String {
    match (input, output) {
        (true, true) => format!("microphone and {}", capture_label(output_source)),
        (true, false) => "microphone".to_string(),
        (false, true) => capture_label(output_source),
        (false, false) => "no audio".to_string(),
    }
}

struct HearFlags {
    input: AtomicBool,
    output: AtomicBool,
}

impl HearFlags {
    fn from_live(args: &LiveArgs) -> Result<Self> {
        let output_selected = args.loopback || args.app.is_some();
        if args.loopback && args.app.is_some() {
            anyhow::bail!("specify only one of --loopback or --app");
        }
        let input = args.mic || !output_selected;
        Ok(Self {
            input: AtomicBool::new(input),
            output: AtomicBool::new(output_selected),
        })
    }

    fn input(&self) -> bool {
        self.input.load(Ordering::Relaxed)
    }

    fn output(&self) -> bool {
        self.output.load(Ordering::Relaxed)
    }

    fn set_input(&self, enabled: bool) {
        self.input.store(enabled, Ordering::Relaxed);
    }

    fn set_output(&self, enabled: bool) {
        self.output.store(enabled, Ordering::Relaxed);
    }
}

struct OpenFeed {
    capture: AudioCapture,
    rx: Receiver<AudioChunk>,
}

struct LiveFeed {
    source: CaptureSource,
    open: Option<OpenFeed>,
    drops: u64,
}

impl LiveFeed {
    fn new(source: CaptureSource) -> Self {
        Self {
            source,
            open: None,
            drops: 0,
        }
    }

    fn total_drops(&self) -> u64 {
        self.drops
            + self
                .open
                .as_ref()
                .map(|open| open.capture.drops())
                .unwrap_or(0)
    }

    fn ensure(&mut self, enabled: bool) -> Result<()> {
        if !enabled {
            self.close();
            return Ok(());
        }
        if self.open.is_some() {
            return Ok(());
        }
        self.start()
    }

    fn reopen(&mut self) -> Result<()> {
        self.close();
        std::thread::sleep(Duration::from_millis(400));
        self.start()
    }

    fn start(&mut self) -> Result<()> {
        let (capture, rx) = start_capture(self.source.clone(), AUDIO_QUEUE_CAPACITY)?;
        self.open = Some(OpenFeed { capture, rx });
        Ok(())
    }

    fn close(&mut self) {
        if let Some(open) = self.open.take() {
            self.drops += open.capture.drops();
        }
    }

    fn invalidated(&self) -> bool {
        self.open
            .as_ref()
            .is_some_and(|open| open.capture.is_invalidated())
    }
}

fn pump_live_audio(
    running: &AtomicBool,
    engine: &mut AsrEngine,
    hear: &HearFlags,
    input: &mut LiveFeed,
    output: &mut LiveFeed,
    mix: &mut AudioMix,
    mt_tx: &Sender<TranslationJob>,
    display: &mut TranscriptDisplay,
    route: &LiveRoute,
    ui: &Option<Sender<String>>,
    translation_drops: &mut u64,
) -> Result<()> {
    let mut active_in = hear.input();
    let mut active_out = hear.output();
    mix.set_enabled(active_in, active_out);
    let mut buffer = Vec::new();
    let mut last_ja_preview = 0_usize;
    let mut strict = true;
    let frame = samples_for_ms(MIX_RATE, CHUNK_MS).max(1);

    while running.load(Ordering::SeqCst) {
        let want_in = hear.input();
        let want_out = hear.output();
        if want_in != active_in || want_out != active_out {
            *translation_drops += enqueue_finals(mt_tx, engine.end_utterance()?, display, route);
            mix.set_enabled(want_in, want_out);
            active_in = want_in;
            active_out = want_out;
        }
        if let Err(err) = sync_feed(input, active_in, true, hear, ui, active_out, strict) {
            return Err(err);
        }
        active_in = hear.input();
        if let Err(err) = sync_feed(output, active_out, false, hear, ui, active_in, strict) {
            return Err(err);
        }
        active_out = hear.output();
        mix.set_enabled(active_in, active_out);

        let (input_dead, output_dead) = collect_feeds(input, output, mix);
        if active_in && (input_dead || input.invalidated()) {
            recover_feed(input, true, hear, ui, active_out, strict)?;
            active_in = hear.input();
        }
        if active_out && (output_dead || output.invalidated()) {
            recover_feed(output, false, hear, ui, active_in, strict)?;
            active_out = hear.output();
        }
        if !active_in {
            input.close();
        }
        if !active_out {
            output.close();
        }
        mix.set_enabled(active_in, active_out);
        strict = false;

        buffer.extend(mix.drain());
        let hearing_japanese = route.langs.source() == CaptionLang::Ja;
        engine.set_retain_audio_finals(hearing_japanese);
        while buffer.len() >= frame {
            let samples: Vec<f32> = buffer.drain(..frame).collect();
            let events = engine.push_samples(MIX_RATE, &samples)?;
            *translation_drops += enqueue_finals(mt_tx, events, display, route);
            preview_japanese(engine, route, &mut last_ja_preview);
        }
    }

    if !buffer.is_empty() {
        let events = engine.push_samples(MIX_RATE, &buffer)?;
        *translation_drops += enqueue_finals(mt_tx, events, display, route);
    }
    Ok(())
}

fn sync_feed(
    feed: &mut LiveFeed,
    enabled: bool,
    microphone: bool,
    hear: &HearFlags,
    ui: &Option<Sender<String>>,
    other_enabled: bool,
    strict: bool,
) -> Result<()> {
    match feed.ensure(enabled) {
        Ok(()) => Ok(()),
        Err(err) if strict && enabled => Err(err),
        Err(err) => {
            report_feed_failure(microphone, hear, ui, other_enabled, &err);
            Ok(())
        }
    }
}

fn recover_feed(
    feed: &mut LiveFeed,
    microphone: bool,
    hear: &HearFlags,
    ui: &Option<Sender<String>>,
    other_enabled: bool,
    strict: bool,
) -> Result<()> {
    match feed.reopen() {
        Ok(()) => Ok(()),
        Err(err) if strict => Err(err),
        Err(err) => {
            report_feed_failure(microphone, hear, ui, other_enabled, &err);
            Ok(())
        }
    }
}

fn report_feed_failure(
    microphone: bool,
    hear: &HearFlags,
    ui: &Option<Sender<String>>,
    other_enabled: bool,
    err: &anyhow::Error,
) {
    tracing::warn!(%err, microphone, "audio capture unavailable");
    if microphone {
        hear.set_input(false);
    } else {
        hear.set_output(false);
    }
    let which = if microphone { "マイク" } else { "出力" };
    let detail = short_error(err);
    let message = format!("{which}を開始できませんでした。{detail}");
    let (input, output) = if microphone {
        (false, other_enabled)
    } else {
        (other_enabled, false)
    };
    send_hear(ui, input, output, &message);
}

fn send_hear(ui: &Option<Sender<String>>, input: bool, output: bool, message: &str) {
    let Some(ui) = ui else {
        return;
    };
    send_caption(
        ui,
        &serde_json::json!({
            "kind": "hear",
            "input": input,
            "output": output,
            "message": message,
        }),
        true,
    );
}

fn short_error(err: &anyhow::Error) -> String {
    let text = err.to_string();
    let mut chars = text.chars();
    let short: String = chars.by_ref().take(160).collect();
    if chars.next().is_some() {
        format!("{short}…")
    } else {
        short
    }
}

fn collect_feeds(input: &LiveFeed, output: &LiveFeed, mix: &mut AudioMix) -> (bool, bool) {
    let input_rx = input.open.as_ref().map(|open| open.rx.clone());
    let output_rx = output.open.as_ref().map(|open| open.rx.clone());
    let timeout = Duration::from_millis(100);
    let mut input_dead = false;
    let mut output_dead = false;
    match (input_rx.as_ref(), output_rx.as_ref()) {
        (Some(input_rx), Some(output_rx)) => {
            select! {
                recv(input_rx) -> msg => {
                    input_dead = !push_chunk(mix, MixPort::Input, msg);
                }
                recv(output_rx) -> msg => {
                    output_dead = !push_chunk(mix, MixPort::Output, msg);
                }
                default(timeout) => {}
            }
            input_dead |= !drain_pending(mix, MixPort::Input, input_rx);
            output_dead |= !drain_pending(mix, MixPort::Output, output_rx);
        }
        (Some(rx), None) => {
            input_dead = !wait_feed(mix, MixPort::Input, rx, timeout);
        }
        (None, Some(rx)) => {
            output_dead = !wait_feed(mix, MixPort::Output, rx, timeout);
        }
        (None, None) => std::thread::sleep(timeout),
    }
    (input_dead, output_dead)
}

fn wait_feed(
    mix: &mut AudioMix,
    port: MixPort,
    rx: &Receiver<AudioChunk>,
    timeout: Duration,
) -> bool {
    match rx.recv_timeout(timeout) {
        Ok(chunk) => {
            mix.push(port, chunk.sample_rate, &chunk.samples);
            drain_pending(mix, port, rx)
        }
        Err(RecvTimeoutError::Timeout) => true,
        Err(RecvTimeoutError::Disconnected) => false,
    }
}

fn push_chunk(
    mix: &mut AudioMix,
    port: MixPort,
    msg: Result<AudioChunk, crossbeam_channel::RecvError>,
) -> bool {
    match msg {
        Ok(chunk) => {
            mix.push(port, chunk.sample_rate, &chunk.samples);
            true
        }
        Err(_) => false,
    }
}

fn drain_pending(mix: &mut AudioMix, port: MixPort, rx: &Receiver<AudioChunk>) -> bool {
    loop {
        match rx.try_recv() {
            Ok(chunk) => mix.push(port, chunk.sample_rate, &chunk.samples),
            Err(crossbeam_channel::TryRecvError::Empty) => return true,
            Err(crossbeam_channel::TryRecvError::Disconnected) => return false,
        }
    }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CaptionLang {
    En = 0,
    Ja = 1,
}

impl CaptionLang {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "en" | "english" => Ok(Self::En),
            "ja" | "japanese" => Ok(Self::Ja),
            other => anyhow::bail!("language must be en or ja, got {other}"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::Ja => "ja",
        }
    }

    fn code(self) -> u8 {
        self as u8
    }

    fn from_code(code: u8) -> Self {
        if code == Self::Ja.code() {
            Self::Ja
        } else {
            Self::En
        }
    }
}

struct LangPair {
    source: AtomicU8,
    target: AtomicU8,
}

impl LangPair {
    fn parse(source: &str, target: &str) -> Result<Self> {
        Ok(Self {
            source: AtomicU8::new(CaptionLang::parse(source)?.code()),
            target: AtomicU8::new(CaptionLang::parse(target)?.code()),
        })
    }

    fn source(&self) -> CaptionLang {
        CaptionLang::from_code(self.source.load(Ordering::Relaxed))
    }

    fn target(&self) -> CaptionLang {
        CaptionLang::from_code(self.target.load(Ordering::Relaxed))
    }

    fn set(&self, source: CaptionLang, target: CaptionLang) {
        self.source.store(source.code(), Ordering::Relaxed);
        self.target.store(target.code(), Ordering::Relaxed);
    }

    fn single(&self) -> bool {
        self.source() == self.target()
    }

    fn translates(&self) -> bool {
        self.source() == CaptionLang::En && self.target() == CaptionLang::Ja
    }
}

struct JaClip {
    samples: Vec<f32>,
    sample_rate: u32,
    generation: u64,
    commit: bool,
}

#[derive(Default)]
struct JaInbox {
    clip: Mutex<Option<JaClip>>,
}

struct LiveRoute {
    langs: Arc<LangPair>,
    ui: Option<Sender<String>>,
    tail: Arc<CaptionTail>,
    en_kick: Option<Sender<()>>,
    ja_inbox: Arc<JaInbox>,
    ja_kick: Sender<()>,
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

    fn current(&self) -> u64 {
        self.rev.load(Ordering::Acquire)
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

fn enqueue_finals(
    tx: &crossbeam_channel::Sender<TranslationJob>,
    events: Vec<AsrEvent>,
    display: &mut TranscriptDisplay,
    route: &LiveRoute,
) -> u64 {
    let mut drops = 0;
    for event in events {
        match event {
            AsrEvent::Partial(text) => {
                if route.langs.source() == CaptionLang::Ja {
                    continue;
                }
                if let Some(ui) = &route.ui {
                    let rev = route.tail.publish(text.clone());
                    if route.langs.translates() {
                        send_caption(
                            ui,
                            &serde_json::json!({
                                "kind": "preview",
                                "rev": rev,
                                "english": text,
                                "single": false,
                            }),
                            false,
                        );
                        if let Some(kick) = &route.en_kick {
                            let _ = kick.try_send(());
                        }
                    } else {
                        send_caption(
                            ui,
                            &serde_json::json!({
                                "kind": "preview",
                                "rev": rev,
                                "single": true,
                                "text": text,
                            }),
                            false,
                        );
                    }
                } else {
                    display.partial(text);
                }
            }
            AsrEvent::Final(utterance) => {
                if route.langs.source() == CaptionLang::Ja {
                    let generation = route.tail.close_utterance();
                    if utterance.samples.len() < utterance.sample_rate as usize / 2 {
                        if let Some(ui) = &route.ui {
                            send_caption(
                                ui,
                                &serde_json::json!({
                                    "kind": "drop",
                                    "rev": generation,
                                }),
                                true,
                            );
                        }
                        continue;
                    }
                    offer_japanese(
                        &route.ja_inbox,
                        &route.ja_kick,
                        JaClip {
                            samples: utterance.samples,
                            sample_rate: utterance.sample_rate,
                            generation,
                            commit: true,
                        },
                    );
                    continue;
                }
                let generation = if route.ui.is_some() {
                    route.tail.close_utterance()
                } else {
                    display.clear_partial();
                    0
                };
                if utterance.text.trim().is_empty() {
                    continue;
                }
                let job = TranslationJob {
                    zipformer: utterance.text,
                    samples: utterance.samples,
                    sample_rate: utterance.sample_rate,
                    utterance_ended: Instant::now(),
                    generation,
                };
                if let Err(err) = tx.try_send(job) {
                    drops += 1;
                    if let Some(ui) = &route.ui {
                        let job = match err {
                            crossbeam_channel::TrySendError::Full(job)
                            | crossbeam_channel::TrySendError::Disconnected(job) => job,
                        };
                        send_heard(
                            ui,
                            "commit",
                            job.generation,
                            &job.zipformer,
                            route.langs.single(),
                            true,
                        );
                    }
                }
            }
        }
    }
    drops
}

fn preview_japanese(engine: &AsrEngine, route: &LiveRoute, last_len: &mut usize) {
    if route.langs.source() != CaptionLang::Ja {
        *last_len = 0;
        return;
    }
    let len = engine.utterance_len();
    if len == 0 {
        *last_len = 0;
        return;
    }
    let (sample_rate, samples) = engine.snapshot_utterance();
    let step = sample_rate as usize;
    if step == 0 || samples.len() < step || samples.len() < last_len.saturating_add(step) {
        return;
    }
    *last_len = samples.len();
    offer_japanese(
        &route.ja_inbox,
        &route.ja_kick,
        JaClip {
            samples,
            sample_rate,
            generation: route.tail.current(),
            commit: false,
        },
    );
}

fn offer_japanese(inbox: &JaInbox, kick: &Sender<()>, clip: JaClip) {
    if let Ok(mut slot) = inbox.clip.lock() {
        *slot = Some(clip);
    }
    let _ = kick.try_send(());
}

fn drain_english(
    metrics: &mut TranslationMetrics,
    whisper: &Option<WhisperRescorer>,
    translator: &mut OnnxMarianTranslator,
    jobs: &crossbeam_channel::Receiver<TranslationJob>,
    captions: Option<&Sender<String>>,
    langs: &LangPair,
) {
    while let Ok(job) = jobs.recv() {
        record_caption(metrics, whisper, translator, job, captions, langs);
    }
}

fn record_caption(
    metrics: &mut TranslationMetrics,
    whisper: &Option<WhisperRescorer>,
    translator: &mut OnnxMarianTranslator,
    job: TranslationJob,
    captions: Option<&Sender<String>>,
    langs: &LangPair,
) {
    match caption_final(whisper, translator, job, captions, langs) {
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
    langs: &LangPair,
) {
    loop {
        if !langs.translates() {
            return;
        }
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
        if translated.japanese.is_empty() || !langs.translates() {
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

fn transcribe_japanese(
    inbox: &JaInbox,
    model: &mut Option<HaasRecognizer>,
    failed: &mut bool,
    model_dir: &std::path::Path,
    threads: i32,
    captions: Option<&Sender<String>>,
    langs: &LangPair,
) {
    loop {
        let Some(clip) = inbox.clip.lock().ok().and_then(|mut slot| slot.take()) else {
            return;
        };
        if *failed {
            continue;
        }
        if model.is_none() {
            match HaasRecognizer::from_dir(model_dir, threads) {
                Ok(recognizer) => *model = Some(recognizer),
                Err(err) => {
                    *failed = true;
                    tracing::warn!(%err, "Haas Japanese recognizer unavailable");
                    if let Some(tx) = captions {
                        send_caption(
                            tx,
                            &serde_json::json!({
                                "kind": "status",
                                "message": "日本語の聞き取りモデルがありません。scripts/download-asr.sh を実行してください。",
                            }),
                            true,
                        );
                    }
                    continue;
                }
            }
        }
        let Some(recognizer) = model.as_ref() else {
            return;
        };
        let heard = match recognizer.transcribe(clip.sample_rate, &clip.samples) {
            Ok(text) => text.and_then(|text| text::keep_japanese(&text)),
            Err(err) => {
                tracing::warn!(%err, "Haas transcription failed");
                None
            }
        };
        if inbox.clip.lock().ok().is_some_and(|slot| slot.is_some()) {
            continue;
        }
        let Some(heard) = heard else {
            if clip.commit {
                if let Some(tx) = captions {
                    send_caption(
                        tx,
                        &serde_json::json!({
                            "kind": "drop",
                            "rev": clip.generation,
                        }),
                        true,
                    );
                }
            }
            return;
        };
        let kind = if clip.commit { "commit" } else { "preview" };
        if let Some(tx) = captions {
            if langs.single() {
                send_caption(
                    tx,
                    &serde_json::json!({
                        "kind": kind,
                        "rev": clip.generation,
                        "single": true,
                        "text": heard,
                    }),
                    clip.commit,
                );
            } else {
                send_caption(
                    tx,
                    &serde_json::json!({
                        "kind": kind,
                        "rev": clip.generation,
                        "single": false,
                        "source": heard,
                    }),
                    clip.commit,
                );
            }
        } else {
            println!("{heard}");
            let _ = io::stdout().flush();
        }
        if clip.commit {
            return;
        }
    }
}

fn send_heard(tx: &Sender<String>, kind: &str, rev: u64, text: &str, single: bool, reliable: bool) {
    if single {
        send_caption(
            tx,
            &serde_json::json!({
                "kind": kind,
                "rev": rev,
                "single": true,
                "text": text,
            }),
            reliable,
        );
    } else {
        send_caption(
            tx,
            &serde_json::json!({
                "kind": kind,
                "rev": rev,
                "single": false,
                "english": text,
            }),
            reliable,
        );
    }
}

fn caption_final(
    whisper: &Option<WhisperRescorer>,
    translator: &mut OnnxMarianTranslator,
    job: TranslationJob,
    captions: Option<&Sender<String>>,
    langs: &LangPair,
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

    let japanese = if langs.translates() {
        let japanese = translator.translate(&english)?;
        text::keep_japanese(&japanese).unwrap_or_default()
    } else {
        String::new()
    };
    let elapsed = job.utterance_ended.elapsed();
    if let Some(tx) = captions {
        if langs.single() {
            send_caption(
                tx,
                &serde_json::json!({
                    "kind": "commit",
                    "rev": job.generation,
                    "single": true,
                    "text": english,
                }),
                true,
            );
        } else {
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
        }
    } else {
        println!("EN: {english}");
        if !japanese.is_empty() {
            println!("JA: {japanese}");
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
