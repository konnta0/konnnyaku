mod asr;
mod audio;
mod caption_log;
mod diar;
mod haas;
mod lid;
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
const DEFAULT_RESCORE_DIR: &str = "models/asr/parakeet-en";
const DEFAULT_LID_DIR: &str = "models/asr/whisper-tiny";
const DEFAULT_MT_DIR: &str = "models/translation/marian-en-ja/onnx";
const DEFAULT_MT_JA_EN_DIR: &str = "models/translation/marian-ja-en/onnx";
const AUDIO_QUEUE_CAPACITY: usize = 50;
const TRANSLATION_QUEUE_CAPACITY: usize = 8;
const CHUNK_MS: u32 = 200;
/// Continuous speech is closed on this cadence so translation and speaker labels
/// are not stuck on one growing line until a long pause.
const LIVE_TURN_SECS: u32 = 6;

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
    /// Print the translated captions saved by `live` in this directory.
    Captions,
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
    /// Language to read. Speech in the other language is translated into this one.
    #[arg(long, default_value = "ja")]
    primary: String,
    /// Haas Japanese ONNX directory.
    #[arg(long, default_value = "models/asr/haas-ja")]
    haas_dir: PathBuf,
    /// Pyannote segmentation and speaker-embedding ONNX directory.
    #[arg(long, default_value = "models/asr/diar")]
    diar_dir: PathBuf,
    /// Directory containing encoder/decoder/joiner/tokens.
    #[arg(long, default_value = DEFAULT_ASR_DIR)]
    asr_model_dir: PathBuf,
    /// English-to-Japanese Marian ONNX directory.
    #[arg(long, default_value = DEFAULT_MT_DIR)]
    mt_model_dir: PathBuf,
    /// Japanese-to-English Marian ONNX directory.
    #[arg(long, default_value = DEFAULT_MT_JA_EN_DIR)]
    mt_ja_en_dir: PathBuf,
    /// Multilingual Whisper tiny directory used to detect English or Japanese.
    #[arg(long, default_value = DEFAULT_LID_DIR)]
    lid_dir: PathBuf,
    /// Endpoint rule2 trailing silence in seconds.
    #[arg(long, default_value_t = 0.8)]
    rule2: f32,
    /// ONNX Runtime threads inside sherpa-onnx.
    #[arg(long, default_value_t = 2)]
    threads: i32,
    /// English recognizer. A Parakeet directory, or Whisper files without a joiner.
    #[arg(long, default_value = DEFAULT_RESCORE_DIR)]
    whisper_dir: PathBuf,
    /// Silero VAD ONNX path.
    #[arg(long, default_value = "models/asr/silero_vad.onnx")]
    vad_model: PathBuf,
    /// Disable the English recognizer and keep Zipformer text.
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
    #[arg(long, default_value = DEFAULT_RESCORE_DIR)]
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
        Command::Captions => print_saved_captions(),
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

fn print_saved_captions() -> Result<()> {
    let json = caption_log::read_json(Path::new("meetings/captions.json"))?;
    println!("{json}");
    Ok(())
}

fn run_live(args: LiveArgs) -> Result<()> {
    let langs = Arc::new(LangPair::parse(&args.primary)?);
    let hear = Arc::new(HearFlags::from_live(&args)?);
    let saved = Arc::new(caption_log::CaptionLog::create(
        Path::new("meetings/captions.json"),
        langs.primary().label(),
    )?);
    #[cfg(target_os = "macos")]
    if !args.cli {
        return run_live_hud(args, langs, Arc::clone(&hear), saved);
    }
    run_live_session(args, None, None, langs, hear, saved)
}

#[cfg(target_os = "macos")]
fn run_live_hud(
    args: LiveArgs,
    langs: Arc<LangPair>,
    hear: Arc<HearFlags>,
    saved: Arc<caption_log::CaptionLog>,
) -> Result<()> {
    let running = install_ctrlc()?;
    let (tx, rx) = bounded::<String>(32);
    let capture_running = Arc::clone(&running);
    let ui_langs = Arc::clone(&langs);
    let session_hear = Arc::clone(&hear);
    let session_saved = Arc::clone(&saved);
    let join = std::thread::Builder::new()
        .name("konnnyaku-live".into())
        .spawn(move || {
            let result = run_live_session(
                args,
                Some(Arc::clone(&capture_running)),
                Some(tx),
                langs,
                session_hear,
                session_saved,
            );
            capture_running.store(false, Ordering::SeqCst);
            result
        })
        .context("spawn live thread")?;
    let ui = crate::overlay::run_caption_hud(Arc::clone(&running), rx, ui_langs, hear, saved);
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
    saved: Arc<caption_log::CaptionLog>,
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
    engine.set_retain_audio_finals(true);
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
    let en_inbox = Arc::new(EnInbox::default());
    let haas_dir = args.haas_dir.clone();
    let lid_dir = args.lid_dir.clone();
    let mt_ja_en_dir = args.mt_ja_en_dir.clone();
    let haas_threads = args.threads;
    let diar_dir = args.diar_dir.clone();
    let (en_kick_tx, en_kick_rx) = if captions.is_some() {
        let (kick_tx, kick_rx) = bounded::<()>(1);
        (Some(kick_tx), Some(kick_rx))
    } else {
        (None, None)
    };
    let route = LiveRoute {
        ui: captions.clone(),
        tail: Arc::clone(&tail),
        en_kick: en_kick_tx,
        en_inbox: Arc::clone(&en_inbox),
    };
    let worker_langs = Arc::clone(&langs);
    let worker_ui = captions.clone();
    let worker_tail = Arc::clone(&tail);
    let worker_en_inbox = Arc::clone(&en_inbox);
    let worker_saved = Arc::clone(&saved);

    let mt_thread = std::thread::spawn(move || -> Result<TranslationMetrics> {
        let mut e2e = TranslationMetrics::new()?;
        let mut kit = RecognizeKit::new(haas_dir, lid_dir, mt_ja_en_dir, diar_dir, haas_threads);
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
                                &mut kit,
                                &worker_saved,
                            ),
                            Err(_) => break,
                        }
                    }
                    recv(en_kick_rx) -> msg => {
                        if msg.is_ok() {
                            live_preview(
                                &whisper,
                                &worker_en_inbox,
                                &worker_tail,
                                &ui,
                                &worker_langs,
                                &mut translator,
                                &mut kit,
                            );
                        } else {
                            drain_english(
                                &mut e2e,
                                &whisper,
                                &mut translator,
                                &mt_rx,
                                Some(&ui),
                                &worker_langs,
                                &mut kit,
                                &worker_saved,
                            );
                            break;
                        }
                    }
                }
            }
        } else {
            while let Ok(job) = mt_rx.recv() {
                record_caption(
                    &mut e2e,
                    &whisper,
                    &mut translator,
                    job,
                    None,
                    &worker_langs,
                    &mut kit,
                    &worker_saved,
                );
            }
        }
        let _ = done_tx.send(());
        Ok(e2e)
    });

    let mut display = TranscriptDisplay::new();
    let mut translation_queue_drops = 0_u64;

    if show_terminal {
        println!(
            "Live captions from {label}. Press Ctrl+C to stop. Display {}.",
            langs.primary().label()
        );
    } else {
        eprintln!("Captions from {label}. Close the window or press Ctrl+C to stop.");
    }
    eprintln!("Translated text is in meetings/captions.json. Read it with: konnnyaku captions");
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
    let mut last_preview = 0_usize;
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
        while buffer.len() >= frame {
            let samples: Vec<f32> = buffer.drain(..frame).collect();
            let events = engine.push_samples(MIX_RATE, &samples)?;
            *translation_drops += enqueue_finals(mt_tx, events, display, route);
            let split = engine.take_if_long(LIVE_TURN_SECS)?;
            *translation_drops += enqueue_finals(mt_tx, split, display, route);
            preview_audio(engine, route, &mut last_preview);
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
    primary: AtomicU8,
}

impl LangPair {
    fn parse(primary: &str) -> Result<Self> {
        Ok(Self {
            primary: AtomicU8::new(CaptionLang::parse(primary)?.code()),
        })
    }

    fn primary(&self) -> CaptionLang {
        CaptionLang::from_code(self.primary.load(Ordering::Relaxed))
    }

    fn set_primary(&self, primary: CaptionLang) {
        self.primary.store(primary.code(), Ordering::Relaxed);
    }
}

struct EnShot {
    samples: Vec<f32>,
    sample_rate: u32,
    generation: u64,
}

#[derive(Default)]
struct EnInbox {
    clip: Mutex<Option<EnShot>>,
}

struct LiveRoute {
    ui: Option<Sender<String>>,
    tail: Arc<CaptionTail>,
    en_kick: Option<Sender<()>>,
    en_inbox: Arc<EnInbox>,
}

struct CaptionTail {
    rev: AtomicU64,
}

impl CaptionTail {
    fn new() -> Self {
        Self {
            rev: AtomicU64::new(0),
        }
    }

    fn current(&self) -> u64 {
        self.rev.load(Ordering::Acquire)
    }

    fn close_utterance(&self) -> u64 {
        self.rev.fetch_add(1, Ordering::Release)
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
            // Zipformer text is English-only. Showing it would turn Japanese speech
            // into an English line and then a nonsense translation.
            AsrEvent::Partial(_) => {}
            AsrEvent::Final(utterance) => {
                let generation = if route.ui.is_some() {
                    route.tail.close_utterance()
                } else {
                    display.clear_partial();
                    0
                };
                let min_samples = (utterance.sample_rate as usize) / 2;
                if utterance.sample_rate == 0 || utterance.samples.len() < min_samples {
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
                let job = TranslationJob {
                    zipformer: utterance.text,
                    samples: utterance.samples,
                    sample_rate: utterance.sample_rate,
                    utterance_ended: Instant::now(),
                    generation,
                };
                if tx.try_send(job).is_err() {
                    drops += 1;
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
                }
            }
        }
    }
    drops
}

fn preview_audio(engine: &AsrEngine, route: &LiveRoute, last_len: &mut usize) {
    if route.en_kick.is_none() {
        return;
    }
    let len = engine.utterance_len();
    let rate = engine.utterance_rate();
    if len == 0 || rate == 0 || len < *last_len {
        *last_len = 0;
        return;
    }
    let step = rate as usize;
    if len < step || len < last_len.saturating_add(step) {
        return;
    }
    let (sample_rate, samples) = engine.snapshot_utterance();
    *last_len = samples.len();
    let generation = route.tail.current();
    if let Ok(mut slot) = route.en_inbox.clip.lock() {
        *slot = Some(EnShot {
            samples,
            sample_rate,
            generation,
        });
    }
    if let Some(kick) = &route.en_kick {
        let _ = kick.try_send(());
    }
}

struct SpeakerState {
    dir: PathBuf,
    threads: i32,
    model: Option<diar::Diarizer>,
    book: diar::SpeakerBook,
    failed: bool,
}

impl SpeakerState {
    fn new(dir: PathBuf, threads: i32) -> Self {
        Self {
            dir,
            threads,
            model: None,
            book: diar::SpeakerBook::default(),
            failed: false,
        }
    }

    fn segment(
        &mut self,
        captions: Option<&Sender<String>>,
        sample_rate: u32,
        samples: &[f32],
    ) -> Vec<diar::SpeechTurn> {
        if self.failed || sample_rate == 0 || samples.is_empty() {
            return Vec::new();
        }
        if self.model.is_none() {
            match diar::Diarizer::open(&self.dir, self.threads) {
                Ok(model) => self.model = Some(model),
                Err(err) => {
                    self.failed = true;
                    tracing::warn!(%err, "speaker diarization unavailable");
                    if let Some(tx) = captions {
                        send_caption(
                            tx,
                            &serde_json::json!({
                                "kind": "status",
                                "message": "話者分離のモデルがありません。scripts/download-asr.sh を実行してください。",
                            }),
                            true,
                        );
                    }
                    return Vec::new();
                }
            }
        }
        let Some(model) = self.model.as_ref() else {
            return Vec::new();
        };
        model.segment(sample_rate, samples)
    }

    fn label(&mut self, turn: &mut diar::SpeechTurn) -> u32 {
        turn.label(&mut self.book)
    }

    fn label_longest(&mut self, turns: &mut [diar::SpeechTurn]) -> u32 {
        let Some(index) = diar::longest_index(turns) else {
            return 0;
        };
        turns[index].label(&mut self.book)
    }
}

struct CaptionLine {
    speaker: u32,
    source: String,
    target: String,
}

struct RecognizeKit {
    haas_dir: PathBuf,
    threads: i32,
    japanese: Option<HaasRecognizer>,
    japanese_failed: bool,
    ja_en: JaEnModel,
    language: LanguageState,
    speakers: SpeakerState,
}

impl RecognizeKit {
    fn new(
        haas_dir: PathBuf,
        lid_dir: PathBuf,
        ja_en_dir: PathBuf,
        diar_dir: PathBuf,
        threads: i32,
    ) -> Self {
        Self {
            haas_dir,
            threads,
            japanese: None,
            japanese_failed: false,
            ja_en: JaEnModel::new(ja_en_dir),
            language: LanguageState::new(lid_dir, threads),
            speakers: SpeakerState::new(diar_dir, threads),
        }
    }

    fn japanese_model(&mut self, captions: Option<&Sender<String>>) -> Option<&HaasRecognizer> {
        if self.japanese_failed {
            return None;
        }
        if self.japanese.is_none() {
            match HaasRecognizer::from_dir(&self.haas_dir, self.threads) {
                Ok(model) => self.japanese = Some(model),
                Err(err) => {
                    self.japanese_failed = true;
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
                    return None;
                }
            }
        }
        self.japanese.as_ref()
    }
}

struct LanguageState {
    dir: PathBuf,
    threads: i32,
    model: Option<lid::LanguageId>,
    failed: bool,
}

impl LanguageState {
    fn new(dir: PathBuf, threads: i32) -> Self {
        Self {
            dir,
            threads,
            model: None,
            failed: false,
        }
    }

    fn detect(
        &mut self,
        captions: Option<&Sender<String>>,
        sample_rate: u32,
        samples: &[f32],
    ) -> Detected {
        if self.failed || sample_rate == 0 || samples.len() < sample_rate as usize * 4 / 5 {
            return Detected::Unknown;
        }
        if self.model.is_none() {
            match lid::LanguageId::open(&self.dir, self.threads) {
                Ok(model) => self.model = Some(model),
                Err(err) => {
                    self.failed = true;
                    tracing::warn!(%err, "language identification unavailable");
                    if let Some(tx) = captions {
                        send_caption(
                            tx,
                            &serde_json::json!({
                                "kind": "status",
                                "message": "言語判定のモデルがありません。scripts/download-asr.sh を実行してください。",
                            }),
                            true,
                        );
                    }
                    return Detected::Unknown;
                }
            }
        }
        let Some(model) = self.model.as_ref() else {
            return Detected::Unknown;
        };
        match model.identify(sample_rate, samples) {
            Ok(lid::Heard::En) => Detected::En,
            Ok(lid::Heard::Ja) => Detected::Ja,
            Ok(lid::Heard::Other) => Detected::Other,
            Err(err) => {
                tracing::warn!(%err, "language identification failed");
                Detected::Unknown
            }
        }
    }
}

enum Detected {
    En,
    Ja,
    Other,
    Unknown,
}

fn spoken_of(detected: Detected, primary: CaptionLang) -> Option<CaptionLang> {
    match detected {
        Detected::En => Some(CaptionLang::En),
        Detected::Ja => Some(CaptionLang::Ja),
        Detected::Unknown => Some(primary),
        Detected::Other => None,
    }
}

struct JaEnModel {
    dir: PathBuf,
    model: Option<OnnxMarianTranslator>,
    failed: bool,
}

impl JaEnModel {
    fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            model: None,
            failed: false,
        }
    }

    fn translate(&mut self, text: &str, captions: Option<&Sender<String>>) -> String {
        if self.failed || text.trim().is_empty() {
            return String::new();
        }
        if self.model.is_none() {
            match OnnxMarianTranslator::from_dir(&self.dir) {
                Ok(model) => self.model = Some(model),
                Err(err) => {
                    self.failed = true;
                    tracing::warn!(%err, "Japanese-to-English model unavailable");
                    if let Some(tx) = captions {
                        send_caption(
                            tx,
                            &serde_json::json!({
                                "kind": "status",
                                "message": "日本語から英語の翻訳モデルがありません。scripts/export-marian-onnx.sh で書き出してください。",
                            }),
                            true,
                        );
                    }
                    return String::new();
                }
            }
        }
        let Some(model) = self.model.as_mut() else {
            return String::new();
        };
        let mut combined = String::new();
        for piece in japanese_pieces(text) {
            match model.translate_into(&piece, false) {
                Ok(english) => {
                    let english = english.trim();
                    if english.is_empty() {
                        continue;
                    }
                    if !combined.is_empty() {
                        combined.push(' ');
                    }
                    combined.push_str(english);
                }
                Err(err) => {
                    tracing::warn!(%err, "Japanese-to-English translation failed");
                    break;
                }
            }
        }
        combined
    }
}

fn japanese_pieces(text: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '。' | '！' | '？' | '!' | '?') {
            let piece = current.trim().to_string();
            if !piece.is_empty() {
                pieces.push(piece);
            }
            current.clear();
        }
    }
    let rest = current.trim().to_string();
    if !rest.is_empty() {
        pieces.push(rest);
    }
    pieces
}

fn drain_english(
    metrics: &mut TranslationMetrics,
    whisper: &Option<WhisperRescorer>,
    translator: &mut OnnxMarianTranslator,
    jobs: &crossbeam_channel::Receiver<TranslationJob>,
    captions: Option<&Sender<String>>,
    langs: &LangPair,
    kit: &mut RecognizeKit,
    saved: &caption_log::CaptionLog,
) {
    while let Ok(job) = jobs.recv() {
        record_caption(
            metrics, whisper, translator, job, captions, langs, kit, saved,
        );
    }
}

fn record_caption(
    metrics: &mut TranslationMetrics,
    whisper: &Option<WhisperRescorer>,
    translator: &mut OnnxMarianTranslator,
    job: TranslationJob,
    captions: Option<&Sender<String>>,
    langs: &LangPair,
    kit: &mut RecognizeKit,
    saved: &caption_log::CaptionLog,
) {
    match caption_final(whisper, translator, job, captions, langs, kit, saved) {
        Ok(Some(elapsed)) => metrics.record(elapsed),
        Ok(None) => {}
        Err(err) => tracing::error!(%err, "caption failed"),
    }
}

fn live_preview(
    english: &Option<WhisperRescorer>,
    inbox: &EnInbox,
    tail: &CaptionTail,
    captions: &Sender<String>,
    langs: &LangPair,
    translator: &mut OnnxMarianTranslator,
    kit: &mut RecognizeKit,
) {
    let primary = langs.primary();
    for _ in 0..2 {
        let Some(shot) = inbox.clip.lock().ok().and_then(|mut slot| slot.take()) else {
            return;
        };
        if shot.generation != tail.current() {
            continue;
        }
        let detected = kit
            .language
            .detect(Some(captions), shot.sample_rate, &shot.samples);
        let Some(spoken) = spoken_of(detected, primary) else {
            return;
        };
        let Some(text) = transcribe_spoken(
            english,
            kit,
            spoken,
            shot.sample_rate,
            &shot.samples,
            None,
            Some(captions),
        ) else {
            return;
        };
        if inbox.clip.lock().ok().is_some_and(|slot| slot.is_some()) {
            continue;
        }
        if shot.generation != tail.current() {
            return;
        }
        let target = into_primary(translator, kit, spoken, primary, &text, Some(captions));
        send_lines(
            Some(captions),
            "preview",
            shot.generation,
            &[CaptionLine {
                speaker: 0,
                source: text,
                target,
            }],
            false,
        );
        return;
    }
}

fn caption_final(
    english: &Option<WhisperRescorer>,
    translator: &mut OnnxMarianTranslator,
    job: TranslationJob,
    captions: Option<&Sender<String>>,
    langs: &LangPair,
    kit: &mut RecognizeKit,
    saved: &caption_log::CaptionLog,
) -> Result<Option<Duration>> {
    let primary = langs.primary();
    let mut turns = kit
        .speakers
        .segment(captions, job.sample_rate, &job.samples);
    let mut pending = Vec::new();
    if diar::has_multiple(&turns) {
        for (index, turn) in turns.iter().enumerate() {
            let audio = diar::slice_turn(job.sample_rate, &job.samples, turn);
            let detected = kit.language.detect(captions, job.sample_rate, &audio);
            let Some(spoken) = spoken_of(detected, primary) else {
                continue;
            };
            let Some(text) = transcribe_spoken(
                english,
                kit,
                spoken,
                job.sample_rate,
                &audio,
                None,
                captions,
            ) else {
                continue;
            };
            let target = into_primary(translator, kit, spoken, primary, &text, captions);
            pending.push((index, text, target));
        }
    }
    let mut lines = Vec::new();
    for (index, text, target) in pending {
        let speaker = kit.speakers.label(&mut turns[index]);
        lines.push(CaptionLine {
            speaker,
            source: text,
            target,
        });
    }
    if lines.is_empty() {
        let detected = kit.language.detect(captions, job.sample_rate, &job.samples);
        if let Some(spoken) = spoken_of(detected, primary)
            && let Some(text) = transcribe_spoken(
                english,
                kit,
                spoken,
                job.sample_rate,
                &job.samples,
                Some(job.zipformer.as_str()),
                captions,
            )
        {
            let target = into_primary(translator, kit, spoken, primary, &text, captions);
            let speaker = kit.speakers.label_longest(&mut turns);
            lines.push(CaptionLine {
                speaker,
                source: text,
                target,
            });
        }
    }
    if lines.is_empty() {
        tracing::debug!(zipformer = %job.zipformer, "skipped utterance");
        drop_caption(captions, job.generation);
        return Ok(None);
    }
    let elapsed = job.utterance_ended.elapsed();
    send_lines(captions, "commit", job.generation, &lines, true);
    if let Err(err) = saved.record(
        primary.label(),
        lines
            .iter()
            .map(|line| (line.speaker, line.source.as_str(), line.target.as_str())),
    ) {
        tracing::warn!(%err, "could not save translated captions");
    }
    Ok(Some(elapsed))
}

fn transcribe_spoken(
    english: &Option<WhisperRescorer>,
    kit: &mut RecognizeKit,
    spoken: CaptionLang,
    sample_rate: u32,
    samples: &[f32],
    zipformer: Option<&str>,
    captions: Option<&Sender<String>>,
) -> Option<String> {
    match spoken {
        CaptionLang::Ja => {
            let recognizer = kit.japanese_model(captions)?;
            heard_japanese(recognizer, sample_rate, samples)
        }
        CaptionLang::En => {
            if let Some(model) = english {
                match model.transcribe(sample_rate, samples) {
                    Ok(Some(text)) => {
                        if let Some(text) = text::prepare_english(&text) {
                            return Some(text);
                        }
                    }
                    Ok(None) => {}
                    Err(err) => tracing::warn!(%err, "English recognition failed"),
                }
            }
            zipformer.and_then(text::prepare_english)
        }
    }
}

fn into_primary(
    translator: &mut OnnxMarianTranslator,
    kit: &mut RecognizeKit,
    spoken: CaptionLang,
    primary: CaptionLang,
    text: &str,
    captions: Option<&Sender<String>>,
) -> String {
    if spoken == primary || text.trim().is_empty() {
        return String::new();
    }
    match (spoken, primary) {
        (CaptionLang::En, CaptionLang::Ja) => {
            let mut combined = String::new();
            for piece in text::english_pieces(text, 16) {
                match translator.translate(&piece) {
                    Ok(japanese) => {
                        if let Some(part) = text::keep_japanese(&japanese) {
                            combined.push_str(&part);
                        }
                    }
                    Err(err) => {
                        tracing::warn!(%err, "English-to-Japanese translation failed");
                        break;
                    }
                }
            }
            combined
        }
        (CaptionLang::Ja, CaptionLang::En) => kit.ja_en.translate(text, captions),
        _ => String::new(),
    }
}

fn heard_japanese(
    recognizer: &HaasRecognizer,
    sample_rate: u32,
    samples: &[f32],
) -> Option<String> {
    match recognizer.transcribe(sample_rate, samples) {
        Ok(text) => text.and_then(|text| text::keep_japanese(&text)),
        Err(err) => {
            tracing::warn!(%err, "Haas transcription failed");
            None
        }
    }
}

fn drop_caption(captions: Option<&Sender<String>>, rev: u64) {
    if let Some(tx) = captions {
        send_caption(
            tx,
            &serde_json::json!({
                "kind": "drop",
                "rev": rev,
            }),
            true,
        );
    }
}

fn send_lines(
    captions: Option<&Sender<String>>,
    kind: &str,
    rev: u64,
    lines: &[CaptionLine],
    reliable: bool,
) {
    let Some(tx) = captions else {
        for line in lines {
            print_line(line);
        }
        let _ = io::stdout().flush();
        return;
    };
    if lines.is_empty() {
        drop_caption(Some(tx), rev);
        return;
    }
    let translated = lines.iter().any(|line| !line.target.is_empty());
    let labeled = lines.len() > 1 || lines.iter().any(|line| line.speaker > 0);
    if !labeled && !translated {
        send_caption(
            tx,
            &serde_json::json!({
                "kind": kind,
                "rev": rev,
                "single": true,
                "text": lines[0].source,
            }),
            reliable,
        );
        return;
    }
    if !labeled {
        send_caption(
            tx,
            &serde_json::json!({
                "kind": kind,
                "rev": rev,
                "single": false,
                "source": lines[0].source,
                "target": lines[0].target,
            }),
            reliable,
        );
        return;
    }
    let turns: Vec<_> = lines
        .iter()
        .map(|line| {
            serde_json::json!({
                "speaker": line.speaker,
                "source": line.source,
                "target": line.target,
            })
        })
        .collect();
    send_caption(
        tx,
        &serde_json::json!({
            "kind": kind,
            "rev": rev,
            "single": !translated,
            "turns": turns,
        }),
        reliable,
    );
}

fn print_line(line: &CaptionLine) {
    let who = if line.speaker > 0 {
        format!("話者{} ", line.speaker)
    } else {
        String::new()
    };
    println!("{who}{}", line.source);
    if !line.target.is_empty() {
        println!("{who}{}", line.target);
    }
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
