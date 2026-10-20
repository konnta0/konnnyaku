//! Live captions. Frozen spans are append-only for the open utterance.
//! The tail is the unmatched suffix of the latest hypothesis.
//! `step` is the only lifecycle transition.

use std::collections::{HashSet, VecDeque};
use std::fmt::{self, Display, Formatter};
use std::io::{self, Write};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossbeam_channel::{bounded, Receiver, Sender};

use crate::text;
use crate::translation::{OnnxMarianTranslator, Translator};
use crate::whisper::WhisperRescorer;

/// Boundary value. `run_live` builds this from `AsrEvent` and does not interpret it.
pub enum Heard {
    /// Full Zipformer hypothesis for the open utterance.
    Revision(String),
    /// VAD or sherpa endpoint, with the speech-gated PCM.
    Endpoint {
        /// Zipformer text at endpoint time.
        text: String,
        /// Speech-gated PCM for this utterance.
        samples: Vec<f32>,
        /// Sample rate of `samples`.
        sample_rate: u32,
    },
}

/// English that already passed [`text::prepare_english`].
#[derive(Clone, Debug, PartialEq, Eq)]
struct EnglishSource(String);

impl EnglishSource {
    fn prepare(raw: &str) -> Option<Self> {
        let stripped = strip_controls(raw);
        text::prepare_english(&stripped).map(Self)
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

/// Japanese that passed [`text::keep_japanese`] and contains no control characters.
#[derive(Clone, Debug, PartialEq, Eq)]
struct JapaneseCaption(String);

impl JapaneseCaption {
    fn keep(raw: &str) -> Option<Self> {
        let stripped = strip_controls(raw);
        text::keep_japanese(&stripped).map(Self)
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct UtteranceId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct JobId(u64);

enum Tail {
    Open { text: String, changed_at: Instant },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Listening,
    Sealed,
}

enum Screen {
    Blank,
    Tail,
    Frozen,
}

struct SealedUtterance {
    id: UtteranceId,
    english: EnglishSource,
}

struct CaptionDocument {
    utterance: UtteranceId,
    frozen: Vec<EnglishSource>,
    tail: Tail,
    screen: Screen,
    queue: WorkQueue,
    preview_gen: u32,
    last_preview: Option<EnglishSource>,
    phase: Phase,
    next_job: u64,
    previous: String,
    whisper: bool,
    correctable: Option<SealedUtterance>,
    applied: HashSet<JobId>,
    failures: u8,
    previews: u64,
    freezes: u64,
    corrections: u64,
    skipped_junk: u64,
}

struct WorkQueue {
    freezes: VecDeque<MarianJob>,
    preview: Option<MarianJob>,
    correct: Option<MarianJob>,
    busy: Option<MarianJob>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum MarianJob {
    Preview {
        id: JobId,
        utterance: UtteranceId,
        generation: u32,
        english: EnglishSource,
    },
    Freeze {
        id: JobId,
        english: EnglishSource,
    },
    Correct {
        id: JobId,
        utterance: UtteranceId,
        english: EnglishSource,
    },
}

impl MarianJob {
    fn id(&self) -> JobId {
        match self {
            Self::Preview { id, .. } | Self::Freeze { id, .. } | Self::Correct { id, .. } => *id,
        }
    }

    fn english(&self) -> &EnglishSource {
        match self {
            Self::Preview { english, .. }
            | Self::Freeze { english, .. }
            | Self::Correct { english, .. } => english,
        }
    }
}

#[derive(Debug)]
struct RescoreJob {
    utterance: UtteranceId,
    samples: Vec<f32>,
    sample_rate: u32,
    zipformer: EnglishSource,
}

enum Input {
    Revision(String),
    Endpoint {
        text: String,
        samples: Vec<f32>,
        sample_rate: u32,
    },
    Clock,
    Translated {
        job_id: JobId,
        japanese: Option<String>,
    },
    Failed {
        job_id: JobId,
    },
    Rescored {
        utterance: UtteranceId,
        text: Option<String>,
    },
}

#[derive(Debug, Default)]
struct Effects {
    translate: Option<MarianJob>,
    rescore: Option<RescoreJob>,
    frames: Vec<Frame>,
}

#[derive(Debug, PartialEq, Eq)]
enum Frame {
    ShowTail(JapaneseCaption),
    FreezeTail {
        lead_break: bool,
        english: EnglishSource,
        japanese: JapaneseCaption,
    },
    Fix {
        lead_break: bool,
        english: EnglishSource,
        japanese: JapaneseCaption,
    },
}

struct Gate;

struct Cut<'a> {
    left: &'a str,
    right: &'a str,
}

impl CaptionDocument {
    fn new(whisper: bool) -> Self {
        Self {
            utterance: UtteranceId(1),
            frozen: Vec::new(),
            tail: Tail::Open {
                text: String::new(),
                changed_at: Instant::now(),
            },
            screen: Screen::Blank,
            queue: WorkQueue {
                freezes: VecDeque::new(),
                preview: None,
                correct: None,
                busy: None,
            },
            preview_gen: 0,
            last_preview: None,
            phase: Phase::Listening,
            next_job: 1,
            previous: String::new(),
            whisper,
            correctable: None,
            applied: HashSet::new(),
            failures: 0,
            previews: 0,
            freezes: 0,
            corrections: 0,
            skipped_junk: 0,
        }
    }
}

impl Gate {
    const PREVIEW_HOLD: Duration = Duration::from_millis(400);
    const PREVIEW_NEW_WORDS: usize = 4;
    const CLAUSE_CUT_WORDS: usize = 12;
    const HARD_CUT_WORDS: usize = 18;
    const MAX_HYPOTHESIS_CHARS: usize = 2_000;

    fn preview(
        tail: &str,
        last: Option<&EnglishSource>,
        changed_at: Instant,
        now: Instant,
    ) -> Option<EnglishSource> {
        let english = EnglishSource::prepare(tail)?;
        if last.is_some_and(|prev| prev.as_str() == english.as_str()) {
            return None;
        }
        let grown = word_count(english.as_str())
            .saturating_sub(last.map(|prev| word_count(prev.as_str())).unwrap_or(0))
            >= Self::PREVIEW_NEW_WORDS;
        let held = now.saturating_duration_since(changed_at) >= Self::PREVIEW_HOLD;
        if grown || held {
            Some(english)
        } else {
            None
        }
    }

    /// Cut the stable prefix only. The caller passes words shared with the previous hypothesis.
    fn cut(stable: &str) -> Option<Cut<'_>> {
        if stable.is_empty() {
            return None;
        }
        if let Some(cut) = cut_punctuation(stable) {
            return Some(cut);
        }
        if let Some(cut) = cut_clause(stable) {
            return Some(cut);
        }
        cut_hard(stable)
    }
}

fn step(doc: &mut CaptionDocument, input: Input, now: Instant) -> Effects {
    match input {
        Input::Revision(text) => on_revision(doc, &text, now),
        Input::Endpoint {
            text,
            samples,
            sample_rate,
        } => on_endpoint(doc, &text, samples, sample_rate, now),
        Input::Clock => on_clock(doc, now),
        Input::Translated { job_id, japanese } => on_translated(doc, job_id, japanese.as_deref()),
        Input::Failed { job_id } => on_failed(doc, job_id),
        Input::Rescored { utterance, text } => on_rescored(doc, utterance, text.as_deref()),
    }
}

fn on_revision(doc: &mut CaptionDocument, text: &str, now: Instant) -> Effects {
    let hypothesis = cap_hypothesis(text);
    if doc.phase == Phase::Listening && hypothesis == doc.previous {
        return Effects::default();
    }
    if doc.phase == Phase::Sealed {
        doc.phase = Phase::Listening;
        doc.correctable = None;
        doc.last_preview = None;
        doc.previous.clear();
        doc.frozen.clear();
        doc.tail = Tail::Open {
            text: String::new(),
            changed_at: now,
        };
    }

    let mut stable = stable_region(&doc.previous, &hypothesis, &frozen_prefix(doc));
    loop {
        let Some(cut) = Gate::cut(&stable) else {
            break;
        };
        let Some(english) = EnglishSource::prepare(cut.left) else {
            break;
        };
        if cut.right.len() >= stable.len() {
            break;
        }
        let right = cut.right.to_string();
        enqueue_freeze(doc, english);
        if right.is_empty() {
            break;
        }
        stable = right;
    }

    let tail = hypothesis_tail(&frozen_prefix(doc), &hypothesis);
    if !tail.is_empty() && EnglishSource::prepare(&tail).is_none() {
        doc.skipped_junk += 1;
    }
    assign_tail(doc, tail, now);
    try_preview(doc, now);
    doc.previous = hypothesis;

    let mut effects = Effects::default();
    dispatch(doc, &mut effects);
    effects
}

fn on_clock(doc: &mut CaptionDocument, now: Instant) -> Effects {
    if doc.phase == Phase::Listening {
        try_preview(doc, now);
    }
    let mut effects = Effects::default();
    dispatch(doc, &mut effects);
    effects
}

fn on_endpoint(
    doc: &mut CaptionDocument,
    text: &str,
    samples: Vec<f32>,
    sample_rate: u32,
    now: Instant,
) -> Effects {
    if doc.phase == Phase::Sealed {
        return Effects::default();
    }
    let hypothesis = cap_hypothesis(text);
    let remainder = hypothesis_tail(&frozen_prefix(doc), &hypothesis);
    if let Some(english) = EnglishSource::prepare(&remainder) {
        enqueue_freeze(doc, english);
    } else if !remainder.is_empty() {
        doc.skipped_junk += 1;
    }
    let zipformer = EnglishSource::prepare(&frozen_prefix(doc));
    let mut effects = Effects::default();
    if doc.whisper {
        if let Some(zipformer) = zipformer {
            effects.rescore = Some(RescoreJob {
                utterance: doc.utterance,
                samples,
                sample_rate,
                zipformer: zipformer.clone(),
            });
            doc.correctable = Some(SealedUtterance {
                id: doc.utterance,
                english: zipformer,
            });
        }
    }
    doc.queue.preview = None;
    doc.utterance = UtteranceId(doc.utterance.0.wrapping_add(1));
    doc.frozen.clear();
    doc.previous.clear();
    doc.last_preview = None;
    doc.tail = Tail::Open {
        text: String::new(),
        changed_at: now,
    };
    doc.preview_gen = doc.preview_gen.wrapping_add(1);
    doc.phase = Phase::Sealed;
    dispatch(doc, &mut effects);
    effects
}

fn on_translated(doc: &mut CaptionDocument, job_id: JobId, japanese: Option<&str>) -> Effects {
    if doc.applied.contains(&job_id) {
        return Effects::default();
    }
    let Some(job) = doc.queue.busy.clone() else {
        return Effects::default();
    };
    if job.id() != job_id {
        return Effects::default();
    }
    doc.applied.insert(job_id);
    doc.queue.busy = None;
    doc.failures = 0;

    let mut effects = Effects::default();
    match job {
        MarianJob::Preview { generation, .. } => {
            if generation == doc.preview_gen {
                paint_preview(doc, japanese, &mut effects);
            }
        }
        MarianJob::Freeze { english, .. } => paint_freeze(doc, &english, japanese, &mut effects),
        MarianJob::Correct { english, .. } => paint_fix(doc, &english, japanese, &mut effects),
    }
    dispatch(doc, &mut effects);
    effects
}

fn on_failed(doc: &mut CaptionDocument, job_id: JobId) -> Effects {
    let Some(job) = doc.queue.busy.clone() else {
        return Effects::default();
    };
    if job.id() != job_id {
        return Effects::default();
    }
    if doc.failures == 0 {
        doc.failures = 1;
        return Effects {
            translate: Some(job),
            ..Effects::default()
        };
    }
    doc.queue.busy = None;
    doc.failures = 0;
    let mut effects = Effects::default();
    dispatch(doc, &mut effects);
    effects
}

fn on_rescored(doc: &mut CaptionDocument, utterance: UtteranceId, text: Option<&str>) -> Effects {
    let Some(sealed) = doc.correctable.as_ref() else {
        return Effects::default();
    };
    if sealed.id != utterance {
        return Effects::default();
    }
    let Some(raw) = text else {
        doc.correctable = None;
        return Effects::default();
    };
    let Some(english) = EnglishSource::prepare(raw) else {
        doc.skipped_junk += 1;
        doc.correctable = None;
        return Effects::default();
    };
    let same = sealed.english.as_str() == english.as_str();
    doc.correctable = None;
    if same {
        return Effects::default();
    }
    let job = MarianJob::Correct {
        id: alloc_job(doc),
        utterance,
        english,
    };
    doc.queue.correct = Some(job);
    let mut effects = Effects::default();
    dispatch(doc, &mut effects);
    effects
}

fn paint_preview(doc: &mut CaptionDocument, japanese: Option<&str>, effects: &mut Effects) {
    let Some(caption) = japanese.and_then(JapaneseCaption::keep) else {
        doc.skipped_junk += 1;
        return;
    };
    effects.frames.push(Frame::ShowTail(caption));
    doc.screen = Screen::Tail;
    doc.previews += 1;
}

fn paint_freeze(
    doc: &mut CaptionDocument,
    english: &EnglishSource,
    japanese: Option<&str>,
    effects: &mut Effects,
) {
    let Some(caption) = japanese.and_then(JapaneseCaption::keep) else {
        doc.skipped_junk += 1;
        return;
    };
    let lead_break = !matches!(doc.screen, Screen::Blank);
    effects.frames.push(Frame::FreezeTail {
        lead_break,
        english: english.clone(),
        japanese: caption,
    });
    doc.screen = Screen::Frozen;
    doc.freezes += 1;
}

fn paint_fix(
    doc: &mut CaptionDocument,
    english: &EnglishSource,
    japanese: Option<&str>,
    effects: &mut Effects,
) {
    let Some(caption) = japanese.and_then(JapaneseCaption::keep) else {
        doc.skipped_junk += 1;
        return;
    };
    let lead_break = !matches!(doc.screen, Screen::Blank);
    effects.frames.push(Frame::Fix {
        lead_break,
        english: english.clone(),
        japanese: caption,
    });
    doc.screen = Screen::Frozen;
    doc.corrections += 1;
}

fn try_preview(doc: &mut CaptionDocument, now: Instant) {
    let Tail::Open { text, changed_at } = &doc.tail;
    let text = text.clone();
    let changed_at = *changed_at;
    let last = doc.last_preview.clone();
    let Some(english) = Gate::preview(&text, last.as_ref(), changed_at, now) else {
        return;
    };
    enqueue_preview(doc, english);
}

fn enqueue_preview(doc: &mut CaptionDocument, english: EnglishSource) {
    let job = MarianJob::Preview {
        id: alloc_job(doc),
        utterance: doc.utterance,
        generation: doc.preview_gen,
        english: english.clone(),
    };
    doc.queue.preview = Some(job);
    doc.last_preview = Some(english);
}

fn enqueue_freeze(doc: &mut CaptionDocument, english: EnglishSource) {
    doc.frozen.push(english.clone());
    let id = alloc_job(doc);
    doc.queue
        .freezes
        .push_back(MarianJob::Freeze { id, english });
}

fn dispatch(doc: &mut CaptionDocument, effects: &mut Effects) {
    if doc.queue.busy.is_some() || effects.translate.is_some() {
        return;
    }
    let job = if let Some(job) = doc.queue.freezes.pop_front() {
        Some(job)
    } else if let Some(job) = doc.queue.preview.take() {
        Some(job)
    } else {
        doc.queue.correct.take()
    };
    let Some(job) = job else {
        return;
    };
    doc.failures = 0;
    effects.translate = Some(job.clone());
    doc.queue.busy = Some(job);
}

fn alloc_job(doc: &mut CaptionDocument) -> JobId {
    let id = JobId(doc.next_job);
    doc.next_job = doc.next_job.wrapping_add(1);
    id
}

fn assign_tail(doc: &mut CaptionDocument, text: String, now: Instant) {
    let same = matches!(&doc.tail, Tail::Open { text: existing, .. } if existing == &text);
    if same {
        return;
    }
    doc.preview_gen = doc.preview_gen.wrapping_add(1);
    doc.tail = Tail::Open {
        text,
        changed_at: now,
    };
}

fn cap_hypothesis(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= Gate::MAX_HYPOTHESIS_CHARS {
        trimmed.to_string()
    } else {
        trimmed.chars().take(Gate::MAX_HYPOTHESIS_CHARS).collect()
    }
}

fn frozen_prefix(doc: &CaptionDocument) -> String {
    doc.frozen
        .iter()
        .map(EnglishSource::as_str)
        .collect::<Vec<_>>()
        .join(" ")
}

fn stable_region(previous: &str, hypothesis: &str, frozen_prefix: &str) -> String {
    let previous_words: Vec<&str> = previous.split_whitespace().collect();
    let current: Vec<&str> = hypothesis.split_whitespace().collect();
    let frozen: Vec<&str> = frozen_prefix.split_whitespace().collect();
    let stable = common_prefix(&previous_words, &current);
    let matched = common_prefix(&frozen, &current);
    if stable <= matched {
        String::new()
    } else {
        current[matched..stable].join(" ")
    }
}

fn hypothesis_tail(frozen_prefix: &str, hypothesis: &str) -> String {
    let frozen: Vec<&str> = frozen_prefix.split_whitespace().collect();
    let current: Vec<&str> = hypothesis.split_whitespace().collect();
    let matched = common_prefix(&frozen, &current);
    current[matched..].join(" ")
}

fn common_prefix(left: &[&str], right: &[&str]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

fn strip_controls(text: &str) -> String {
    text.chars().filter(|ch| !ch.is_control()).collect()
}

fn cut_punctuation(text: &str) -> Option<Cut<'_>> {
    let bytes = text.as_bytes();
    for (index, byte) in bytes.iter().enumerate() {
        if !matches!(byte, b'.' | b'?' | b'!') {
            continue;
        }
        let after = index + 1;
        if after >= bytes.len() || !bytes[after].is_ascii_whitespace() {
            continue;
        }
        if let Some(cut) = accept_left(&text[..=index], &text[after..]) {
            return Some(cut);
        }
    }
    None
}

fn cut_clause(text: &str) -> Option<Cut<'_>> {
    const MARKERS: &[(&str, bool)] = &[
        (" and ", false),
        (" but ", false),
        (" so ", false),
        (" because ", false),
        (", ", true),
    ];
    let mut best_at = usize::MAX;
    let mut best: Option<Cut<'_>> = None;
    for (needle, keep_comma) in MARKERS {
        let mut search_from = 0;
        while let Some(relative) = text[search_from..].find(needle) {
            let index = search_from + relative;
            search_from = index + 1;
            let (left, right) = if *keep_comma {
                (&text[..index + 1], &text[index + needle.len()..])
            } else {
                (&text[..index], &text[index + 1..])
            };
            if word_count(left.trim()) < Gate::CLAUSE_CUT_WORDS {
                continue;
            }
            if index >= best_at {
                continue;
            }
            if let Some(cut) = accept_left(left, right) {
                best_at = index;
                best = Some(cut);
            }
        }
    }
    best
}

fn cut_hard(text: &str) -> Option<Cut<'_>> {
    let mut words = 0usize;
    let mut word_start = 0usize;
    for (index, ch) in text.char_indices() {
        if !ch.is_whitespace() {
            continue;
        }
        if index > word_start {
            words += 1;
            if words == Gate::HARD_CUT_WORDS {
                return accept_left(text[..index].trim(), text[index..].trim());
            }
        }
        word_start = index + ch.len_utf8();
    }
    if word_start < text.len() && !text[word_start..].trim().is_empty() {
        words += 1;
        if words >= Gate::HARD_CUT_WORDS {
            return accept_left(text.trim(), "");
        }
    }
    None
}

/// Models already loaded by `run_live`. The actor takes ownership.
pub struct CaptionModels {
    /// Marian EN→JA session.
    pub translator: OnnxMarianTranslator,
    /// Whisper tiny.en, when the rescorer loaded.
    pub whisper: Option<WhisperRescorer>,
}

/// Counts printed by `run_live` on stderr after shutdown.
pub struct CaptionReport {
    /// Open tails that were painted.
    pub previews: u64,
    /// Frozen rows that were painted.
    pub freezes: u64,
    /// Whisper `FIX` pairs that were painted.
    pub corrections: u64,
    /// Revisions dropped because the mailbox was full.
    pub dropped_revisions: u64,
    /// English or Japanese text the filters rejected.
    pub skipped_junk: u64,
}

impl Display for CaptionReport {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "previews={} freezes={} corrections={} dropped_revisions={} skipped_junk={}",
            self.previews,
            self.freezes,
            self.corrections,
            self.dropped_revisions,
            self.skipped_junk
        )
    }
}

/// Caption owner. The capture thread sends [`Heard`]. This type does not print.
pub struct LiveCaptions {
    tx: Sender<ActorMsg>,
    dropped: Arc<std::sync::atomic::AtomicU64>,
    join: Option<JoinHandle<Result<CaptionReport>>>,
}

impl LiveCaptions {
    /// Spawn the actor. It owns Marian, optional Whisper, the document, and stdout.
    pub fn start(models: CaptionModels) -> Result<Self> {
        let (tx, rx) = bounded(HEARD_CAPACITY);
        let dropped = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let join = std::thread::Builder::new()
            .name("konnnyaku-captions".into())
            .spawn(move || actor_main(models, rx))
            .context("spawn caption actor")?;
        Ok(Self {
            tx,
            dropped,
            join: Some(join),
        })
    }

    /// Enqueue one ASR observation. Revisions are lossy. Endpoints block.
    pub fn hear(&self, heard: Heard) -> Result<()> {
        use crossbeam_channel::TrySendError;
        use std::sync::atomic::Ordering;

        match heard {
            Heard::Revision(text) => {
                match self.tx.try_send(ActorMsg::Heard(Heard::Revision(text))) {
                    Ok(()) => Ok(()),
                    Err(TrySendError::Full(_)) => {
                        self.dropped.fetch_add(1, Ordering::Relaxed);
                        Ok(())
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        anyhow::bail!("caption actor stopped")
                    }
                }
            }
            endpoint => {
                self.tx
                    .send(ActorMsg::Heard(endpoint))
                    .context("caption actor stopped")?;
                Ok(())
            }
        }
    }

    /// Close the mailbox and join the actor.
    pub fn finish(mut self) -> Result<CaptionReport> {
        use std::sync::atomic::Ordering;

        let _ = self.tx.send(ActorMsg::Shutdown);
        let join = self.join.take().context("caption actor already joined")?;
        let mut report = join
            .join()
            .map_err(|_| anyhow::anyhow!("caption actor panicked"))??;
        report.dropped_revisions = self.dropped.load(Ordering::Relaxed);
        Ok(report)
    }
}

impl Drop for LiveCaptions {
    fn drop(&mut self) {
        let _ = self.tx.send(ActorMsg::Shutdown);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

const HEARD_CAPACITY: usize = 32;

enum ActorMsg {
    Heard(Heard),
    Shutdown,
}

struct WhisperOut {
    utterance: UtteranceId,
    text: Option<String>,
}

/// One queued PCM buffer. A newer endpoint replaces a job the worker has not taken.
struct WhisperSlot {
    state: Mutex<WhisperState>,
    notify: Condvar,
}

struct WhisperState {
    job: Option<RescoreJob>,
    stop: bool,
}

impl WhisperSlot {
    fn new() -> Self {
        Self {
            state: Mutex::new(WhisperState {
                job: None,
                stop: false,
            }),
            notify: Condvar::new(),
        }
    }

    fn submit(&self, job: RescoreJob) {
        let mut guard = lock_slot(&self.state);
        if guard.stop {
            return;
        }
        guard.job = Some(job);
        self.notify.notify_one();
    }

    fn recv(&self) -> Option<RescoreJob> {
        let mut guard = lock_slot(&self.state);
        loop {
            if guard.stop {
                return None;
            }
            if let Some(job) = guard.job.take() {
                return Some(job);
            }
            guard = self
                .notify
                .wait(guard)
                .unwrap_or_else(|err| err.into_inner());
        }
    }

    fn stop(&self) {
        let mut guard = lock_slot(&self.state);
        guard.stop = true;
        self.notify.notify_one();
    }
}

fn lock_slot(state: &Mutex<WhisperState>) -> std::sync::MutexGuard<'_, WhisperState> {
    state.lock().unwrap_or_else(|err| err.into_inner())
}

struct WhisperWorker {
    jobs: Arc<WhisperSlot>,
    results: Arc<ResultSlot>,
    join: JoinHandle<()>,
}

/// Latest unread Whisper texts. The worker never blocks on the actor.
struct ResultSlot {
    values: Mutex<VecDeque<WhisperOut>>,
}

impl ResultSlot {
    const CAPACITY: usize = 8;

    fn new() -> Self {
        Self {
            values: Mutex::new(VecDeque::new()),
        }
    }

    fn publish(&self, value: WhisperOut) {
        let mut values = self.values.lock().unwrap_or_else(|err| err.into_inner());
        if values.len() == Self::CAPACITY {
            values.pop_front();
        }
        values.push_back(value);
    }

    fn take(&self) -> Option<WhisperOut> {
        self.values
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .pop_front()
    }
}

fn actor_main(models: CaptionModels, heard_rx: Receiver<ActorMsg>) -> Result<CaptionReport> {
    let CaptionModels {
        mut translator,
        whisper,
    } = models;
    let mut doc = CaptionDocument::new(whisper.is_some());
    let whisper_worker = match whisper {
        Some(rescorer) => Some(spawn_whisper(rescorer)?),
        None => None,
    };
    let mut stdout = io::stdout().lock();
    loop {
        let stop = crossbeam_channel::select! {
            recv(heard_rx) -> msg => {
                match msg {
                    Ok(ActorMsg::Shutdown) | Err(_) => true,
                    Ok(ActorMsg::Heard(heard)) => {
                        let (inputs, stop) = coalesce(heard, &heard_rx);
                        for input in inputs {
                            drive(
                                &mut doc,
                                input,
                                &mut translator,
                                &mut stdout,
                                whisper_worker.as_ref().map(|worker| worker.jobs.as_ref()),
                            );
                        }
                        stop
                    }
                }
            }
            default(Duration::from_millis(100)) => {
                let jobs = whisper_worker.as_ref().map(|worker| worker.jobs.as_ref());
                drive(&mut doc, Input::Clock, &mut translator, &mut stdout, jobs);
                let _ = drive_rescore(
                    &mut doc,
                    &mut translator,
                    &mut stdout,
                    whisper_worker.as_ref(),
                );
                false
            }
        };
        if stop {
            break;
        }
    }
    if let Some(worker) = whisper_worker {
        worker.jobs.stop();
        let WhisperWorker { results, join, .. } = worker;
        let _ = join.join();
        while let Some(done) = results.take() {
            drive(
                &mut doc,
                Input::Rescored {
                    utterance: done.utterance,
                    text: done.text,
                },
                &mut translator,
                &mut stdout,
                None,
            );
        }
    }
    tracing::info!(metrics = %translator.metrics(), "caption actor stopped");
    Ok(doc.report())
}

fn spawn_whisper(rescorer: WhisperRescorer) -> Result<WhisperWorker> {
    let jobs = Arc::new(WhisperSlot::new());
    let results = Arc::new(ResultSlot::new());
    let jobs_for_thread = Arc::clone(&jobs);
    let results_for_thread = Arc::clone(&results);
    let join = std::thread::Builder::new()
        .name("konnnyaku-whisper".into())
        .spawn(move || whisper_loop(rescorer, jobs_for_thread, results_for_thread))
        .context("spawn whisper thread")?;
    Ok(WhisperWorker {
        jobs,
        results,
        join,
    })
}

fn drive_rescore(
    doc: &mut CaptionDocument,
    translator: &mut OnnxMarianTranslator,
    out: &mut impl Write,
    worker: Option<&WhisperWorker>,
) -> bool {
    let Some(worker) = worker else {
        return false;
    };
    let Some(done) = worker.results.take() else {
        return false;
    };
    drive(
        doc,
        Input::Rescored {
            utterance: done.utterance,
            text: done.text,
        },
        translator,
        out,
        Some(worker.jobs.as_ref()),
    );
    true
}

fn drive(
    doc: &mut CaptionDocument,
    input: Input,
    translator: &mut OnnxMarianTranslator,
    out: &mut impl Write,
    slot: Option<&WhisperSlot>,
) {
    let mut effects = step(doc, input, Instant::now());
    loop {
        if let Some(job) = effects.rescore.take() {
            if let Some(slot) = slot {
                tracing::debug!(
                    utterance = job.utterance.0,
                    zipformer = job.zipformer.as_str(),
                    "queue whisper rescore"
                );
                slot.submit(job);
            }
        }
        for frame in &effects.frames {
            if let Err(err) = paint(frame, out) {
                tracing::warn!(%err, "caption write failed");
            }
        }
        let Some(job) = effects.translate.take() else {
            break;
        };
        let job_id = job.id();
        let english = job.english().as_str().to_string();
        match translator.translate(&english) {
            Ok(japanese) => {
                effects = step(
                    doc,
                    Input::Translated {
                        job_id,
                        japanese: Some(japanese),
                    },
                    Instant::now(),
                );
            }
            Err(err) => {
                tracing::warn!(%err, "translation failed");
                effects = step(doc, Input::Failed { job_id }, Instant::now());
                let retried = effects
                    .translate
                    .as_ref()
                    .is_some_and(|job| job.id() == job_id);
                if !retried {
                    tracing::warn!(job = job_id.0, "dropped translation after one retry");
                }
            }
        }
    }
}

fn coalesce(first: Heard, rx: &Receiver<ActorMsg>) -> (Vec<Input>, bool) {
    let mut pending = vec![ActorMsg::Heard(first)];
    let mut stop = false;
    while let Ok(msg) = rx.try_recv() {
        match msg {
            ActorMsg::Shutdown => {
                stop = true;
                break;
            }
            other => pending.push(other),
        }
    }
    let mut inputs = Vec::new();
    let mut revision = None;
    for msg in pending {
        match msg {
            ActorMsg::Heard(Heard::Revision(text)) => revision = Some(text),
            ActorMsg::Heard(Heard::Endpoint {
                text,
                samples,
                sample_rate,
            }) => {
                if let Some(text) = revision.take() {
                    inputs.push(Input::Revision(text));
                }
                inputs.push(Input::Endpoint {
                    text,
                    samples,
                    sample_rate,
                });
            }
            ActorMsg::Shutdown => stop = true,
        }
    }
    if let Some(text) = revision {
        inputs.push(Input::Revision(text));
    }
    (inputs, stop)
}

fn whisper_loop(rescorer: WhisperRescorer, slot: Arc<WhisperSlot>, results: Arc<ResultSlot>) {
    while let Some(job) = slot.recv() {
        let text = match rescorer.transcribe(job.sample_rate, &job.samples) {
            Ok(text) => text,
            Err(err) => {
                tracing::warn!(%err, "whisper rescore failed");
                None
            }
        };
        results.publish(WhisperOut {
            utterance: job.utterance,
            text,
        });
    }
}

fn paint(frame: &Frame, out: &mut impl Write) -> io::Result<()> {
    match frame {
        Frame::ShowTail(japanese) => {
            write!(out, "\rJA: {}\x1b[K", japanese.as_str())?;
        }
        Frame::FreezeTail {
            lead_break,
            english,
            japanese,
        } => {
            if *lead_break {
                writeln!(out)?;
            }
            write!(out, "EN: {}\nJA: {}\n", english.as_str(), japanese.as_str())?;
        }
        Frame::Fix {
            lead_break,
            english,
            japanese,
        } => {
            if *lead_break {
                writeln!(out)?;
            }
            write!(
                out,
                "FIX\nEN: {}\nJA: {}\n",
                english.as_str(),
                japanese.as_str()
            )?;
        }
    }
    out.flush()
}

impl CaptionDocument {
    fn report(&self) -> CaptionReport {
        CaptionReport {
            previews: self.previews,
            freezes: self.freezes,
            corrections: self.corrections,
            dropped_revisions: 0,
            skipped_junk: self.skipped_junk,
        }
    }
}

fn accept_left<'a>(left: &'a str, right: &'a str) -> Option<Cut<'a>> {
    let left = left.trim();
    let right = right.trim();
    if left.is_empty() || EnglishSource::prepare(left).is_none() {
        return None;
    }
    Some(Cut { left, right })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete(
        doc: &mut CaptionDocument,
        effects: Effects,
        japanese: &str,
        now: Instant,
    ) -> Effects {
        let Some(job) = effects.translate else {
            return effects;
        };
        step(
            doc,
            Input::Translated {
                job_id: job.id(),
                japanese: Some(japanese.to_string()),
            },
            now,
        )
    }

    #[test]
    fn preview_after_400ms_hold() {
        let now = Instant::now();
        let mut doc = CaptionDocument::new(false);
        let opened = step(&mut doc, Input::Revision("update the server".into()), now);
        assert!(opened.translate.is_none());
        assert!(opened.frames.is_empty());

        let early = step(&mut doc, Input::Clock, now + Duration::from_millis(399));
        assert!(early.translate.is_none());

        let held = step(&mut doc, Input::Clock, now + Duration::from_millis(400));
        match &held.translate {
            Some(MarianJob::Preview { english, .. }) => {
                assert_eq!(english.as_str(), "update the server");
            }
            other => panic!("expected a preview, got {other:?}"),
        }

        let again = step(&mut doc, Input::Clock, now + Duration::from_secs(2));
        assert!(again.translate.is_none());
        assert!(again.frames.is_empty());
    }

    #[test]
    fn freeze_on_stable_prefix() {
        let now = Instant::now();
        let mut doc = CaptionDocument::new(false);
        let first = step(
            &mut doc,
            Input::Revision("We need to update the server before Friday. Then".into()),
            now,
        );
        assert!(
            matches!(first.translate, Some(MarianJob::Preview { .. })),
            "the first sight of a sentence is a preview"
        );
        let settled = complete(&mut doc, first, "プレビュー", now);
        assert!(matches!(settled.frames.first(), Some(Frame::ShowTail(_))));

        let second = step(
            &mut doc,
            Input::Revision(
                "We need to update the server before Friday. Then we roll it out tonight please"
                    .into(),
            ),
            now,
        );
        match &second.translate {
            Some(MarianJob::Freeze { english, .. }) => {
                assert_eq!(
                    english.as_str(),
                    "We need to update the server before Friday."
                );
            }
            other => panic!("expected a freeze of the shared sentence, got {other:?}"),
        }
    }

    #[test]
    fn stale_preview_paints_nothing() {
        let now = Instant::now();
        let mut doc = CaptionDocument::new(false);
        let preview = step(
            &mut doc,
            Input::Revision("we need to update the server tonight".into()),
            now,
        );
        let Some(MarianJob::Preview { id, .. }) = preview.translate else {
            panic!("expected a preview job");
        };
        let grown = step(
            &mut doc,
            Input::Revision("we need to update the server tonight before the window".into()),
            now,
        );
        assert!(grown.translate.is_none());
        let stale = step(
            &mut doc,
            Input::Translated {
                job_id: id,
                japanese: Some("古い訳".into()),
            },
            now,
        );
        assert!(stale.frames.is_empty());
    }

    #[test]
    fn junk_tail_is_skipped() {
        let now = Instant::now();
        let mut doc = CaptionDocument::new(false);
        let junk = step(&mut doc, Input::Revision("ha ha ha ha".into()), now);
        assert!(junk.translate.is_none());
        assert!(junk.frames.is_empty());

        let held = step(&mut doc, Input::Clock, now + Duration::from_millis(400));
        assert!(held.translate.is_none());

        let ended = step(
            &mut doc,
            Input::Endpoint {
                text: "ha ha ha ha".into(),
                samples: vec![0.0; 16],
                sample_rate: 16_000,
            },
            now,
        );
        assert!(ended.translate.is_none());
        assert!(ended.rescore.is_none());
        assert!(ended.frames.is_empty());
        assert!(doc.skipped_junk >= 1);
    }

    #[test]
    fn endpoint_freezes_remainder() {
        let now = Instant::now();
        let mut doc = CaptionDocument::new(false);
        let opened = step(&mut doc, Input::Revision("update the server".into()), now);
        assert!(opened.translate.is_none());

        let ended = step(
            &mut doc,
            Input::Endpoint {
                text: "update the server".into(),
                samples: vec![0.1; 4],
                sample_rate: 16_000,
            },
            now,
        );
        match &ended.translate {
            Some(MarianJob::Freeze { english, .. }) => {
                assert_eq!(english.as_str(), "update the server");
            }
            other => panic!("expected the short remainder to freeze, got {other:?}"),
        }
        assert!(ended.rescore.is_none());

        let again = step(
            &mut doc,
            Input::Endpoint {
                text: "update the server".into(),
                samples: Vec::new(),
                sample_rate: 16_000,
            },
            now,
        );
        assert!(again.translate.is_none());
        assert!(again.frames.is_empty());
        assert!(again.rescore.is_none());
    }

    #[test]
    fn clause_cut_on_stable_prefix() {
        let now = Instant::now();
        let mut doc = CaptionDocument::new(false);
        let shared =
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima and mike";
        let first = step(&mut doc, Input::Revision(shared.into()), now);
        let settled = complete(&mut doc, first, "節", now);
        assert!(settled.translate.is_none());

        let second = step(&mut doc, Input::Revision(format!("{shared} november")), now);
        match &second.translate {
            Some(MarianJob::Freeze { english, .. }) => {
                assert_eq!(
                    english.as_str(),
                    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
                );
            }
            other => panic!("expected a 12-word clause freeze, got {other:?}"),
        }
    }

    #[test]
    fn hard_cut_on_stable_prefix() {
        let now = Instant::now();
        let mut doc = CaptionDocument::new(false);
        let words = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
            "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
        ];
        let shared = words.join(" ");
        assert_eq!(word_count(&shared), 18);
        let first = step(&mut doc, Input::Revision(shared.clone()), now);
        let _ = complete(&mut doc, first, "硬", now);

        let second = step(&mut doc, Input::Revision(format!("{shared} sierra")), now);
        match &second.translate {
            Some(MarianJob::Freeze { english, .. }) => assert_eq!(english.as_str(), shared),
            other => panic!("expected an 18-word hard cut, got {other:?}"),
        }
    }

    #[test]
    fn same_revision_twice_is_empty() {
        let now = Instant::now();
        let mut doc = CaptionDocument::new(false);
        let first = step(
            &mut doc,
            Input::Revision("update the server now".into()),
            now,
        );
        assert!(matches!(first.translate, Some(MarianJob::Preview { .. })));
        let second = step(
            &mut doc,
            Input::Revision("update the server now".into()),
            now,
        );
        assert!(second.translate.is_none());
        assert!(second.frames.is_empty());
    }

    #[test]
    fn failed_translation_retries_once() {
        let now = Instant::now();
        let mut doc = CaptionDocument::new(false);
        let first = step(
            &mut doc,
            Input::Revision("update the server now".into()),
            now,
        );
        let id = first.translate.as_ref().expect("preview").id();
        let retry = step(&mut doc, Input::Failed { job_id: id }, now);
        assert_eq!(retry.translate.as_ref().map(MarianJob::id), Some(id));
        let dropped = step(&mut doc, Input::Failed { job_id: id }, now);
        assert!(dropped.translate.is_none());
        let late = step(
            &mut doc,
            Input::Translated {
                job_id: id,
                japanese: Some("訳".into()),
            },
            now,
        );
        assert!(late.frames.is_empty());
    }

    #[test]
    fn whisper_disagreement_appends_fix_after_freeze() {
        let now = Instant::now();
        let mut doc = CaptionDocument::new(true);
        let _ = step(&mut doc, Input::Revision("update the server".into()), now);
        let ended = step(
            &mut doc,
            Input::Endpoint {
                text: "update the server".into(),
                samples: vec![0.0; 8],
                sample_rate: 16_000,
            },
            now,
        );
        let (freeze_id, utterance) = match (&ended.translate, &ended.rescore) {
            (Some(job), Some(rescore)) => (job.id(), rescore.utterance),
            other => panic!("expected a freeze and a rescore, got {other:?}"),
        };
        let queued = step(
            &mut doc,
            Input::Rescored {
                utterance,
                text: Some("update the server tonight".into()),
            },
            now,
        );
        assert!(queued.translate.is_none());
        let frozen = step(
            &mut doc,
            Input::Translated {
                job_id: freeze_id,
                japanese: Some("サーバーを更新する".into()),
            },
            now,
        );
        assert!(matches!(
            frozen.frames.first(),
            Some(Frame::FreezeTail { .. })
        ));
        let correct_id = match &frozen.translate {
            Some(MarianJob::Correct { id, english, .. }) => {
                assert_eq!(english.as_str(), "update the server tonight");
                *id
            }
            other => panic!("expected the correction behind the freeze, got {other:?}"),
        };
        let fixed = step(
            &mut doc,
            Input::Translated {
                job_id: correct_id,
                japanese: Some("今夜サーバーを更新する".into()),
            },
            now,
        );
        assert!(matches!(fixed.frames.first(), Some(Frame::Fix { .. })));
    }

    #[test]
    fn whisper_for_older_utterance_is_empty() {
        let now = Instant::now();
        let mut doc = CaptionDocument::new(true);
        let _ = step(&mut doc, Input::Revision("update the server".into()), now);
        let ended = step(
            &mut doc,
            Input::Endpoint {
                text: "update the server".into(),
                samples: vec![0.0; 8],
                sample_rate: 16_000,
            },
            now,
        );
        let utterance = ended.rescore.expect("rescore").utterance;
        let _ = step(
            &mut doc,
            Input::Revision("ship the patch tonight".into()),
            now,
        );
        let late = step(
            &mut doc,
            Input::Rescored {
                utterance,
                text: Some("update the server tonight please".into()),
            },
            now,
        );
        assert!(late.translate.is_none());
        assert!(late.frames.is_empty());
    }

    #[test]
    fn paint_appends_freeze_and_fix_without_cursor_up() {
        let japanese = JapaneseCaption::keep("サーバー\r\nを更新").expect("japanese");
        assert_eq!(japanese.as_str(), "サーバーを更新");
        let english = EnglishSource::prepare("update the server").expect("english");
        let mut buf = Vec::new();
        paint(&Frame::ShowTail(japanese.clone()), &mut buf).expect("tail");
        paint(
            &Frame::FreezeTail {
                lead_break: true,
                english: english.clone(),
                japanese: japanese.clone(),
            },
            &mut buf,
        )
        .expect("freeze");
        paint(
            &Frame::Fix {
                lead_break: true,
                english: english.clone(),
                japanese: japanese.clone(),
            },
            &mut buf,
        )
        .expect("fix");
        let text = String::from_utf8(buf).expect("utf-8");
        assert!(text.starts_with("\rJA: サーバーを更新\x1b[K"));
        assert!(text.contains("\nEN: update the server\nJA: サーバーを更新\n"));
        assert!(text.contains("\nFIX\nEN: update the server\nJA: サーバーを更新\n"));
        assert!(!text.contains("\u{1b}[A"));

        let mut next = Vec::new();
        paint(
            &Frame::FreezeTail {
                lead_break: false,
                english: english.clone(),
                japanese: japanese.clone(),
            },
            &mut next,
        )
        .expect("first freeze");
        paint(&Frame::ShowTail(japanese.clone()), &mut next).expect("next tail");
        let next = String::from_utf8(next).expect("utf-8");
        assert!(
            next.starts_with("EN: update the server\nJA: サーバーを更新\n\rJA: サーバーを更新"),
            "a later tail must start after the frozen row, got {next:?}"
        );
    }
}
