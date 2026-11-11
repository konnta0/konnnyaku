//! Speaker labels for a finished utterance.
//!
//! Segmentation is pyannote 3.0. Identity is 3D-Speaker CAM++, trained on Chinese and
//! English (`speech_campplus_sv_zh_en_16k-common_advanced`, Apache-2.0). Local speaker
//! numbers reset on every call, so a session [`SpeakerBook`] keeps them stable by
//! cosine similarity. A pitch change inside one clip stays one person unless the
//! embeddings are clearly different.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, anyhow};
use sherpa_onnx::{
    OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig, SpeakerEmbeddingExtractor,
    SpeakerEmbeddingExtractorConfig,
};

/// Confident same-speaker match. A weaker score still reuses the nearest voice.
const MATCH_COSINE: f32 = 0.50;
/// Only a clearly different, long enough voice opens a new speaker id.
const NEW_SPEAKER_COSINE: f32 = 0.38;
const MIN_NEW_SECS: f32 = 1.5;
/// Ignore blips shorter than the recognizers will transcribe.
const MIN_TURN_SECS: f32 = 0.5;
/// A second voice below this length, or this share of the clip, stays one speaker.
const MIN_OTHER_SECS: f32 = 1.0;
const MIN_OTHER_SHARE: f32 = 0.30;
/// Join adjacent slices from the same person when the gap is only a breath.
const MERGE_GAP_SECS: f32 = 0.35;
const MAX_SPEAKERS: usize = 12;
const MAX_EMBEDDINGS: usize = 3;
/// File written by `scripts/download-asr.sh`. The Chinese-only ERes2Net file used
/// the same `embedding.onnx` name and treated a change in pitch as a new person.
const EMBEDDING_ID: &str = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced";

/// One slice of an utterance, in seconds at the original sample rate.
///
/// The embedding stays unused until [`SpeechTurn::label`] so a discarded line
/// does not become a new speaker.
#[derive(Clone, Debug)]
pub struct SpeechTurn {
    local: i32,
    embedding: Option<Vec<f32>>,
    pub start: f32,
    pub end: f32,
}

impl SpeechTurn {
    /// Remember this voice and return its session id. Calling it again returns 0.
    pub fn label(&mut self, book: &mut SpeakerBook) -> u32 {
        let seconds = self.duration();
        book.assign(self.embedding.take().unwrap_or_default(), seconds)
    }
}

impl SpeechTurn {
    fn duration(&self) -> f32 {
        (self.end - self.start).max(0.0)
    }
}

/// Embeddings enrolled for this process. Ids start at 1 and stay put for the session.
#[derive(Default)]
pub struct SpeakerBook {
    next: u32,
    speakers: Vec<Enrolled>,
}

struct Enrolled {
    id: u32,
    embeds: Vec<Vec<f32>>,
}

impl SpeakerBook {
    fn assign(&mut self, embedding: Vec<f32>, seconds: f32) -> u32 {
        let Some(embedding) = unit(embedding) else {
            return 0;
        };
        let mut best_id = 0;
        let mut best = f32::MIN;
        for speaker in &self.speakers {
            for known in &speaker.embeds {
                let score = dot(known, &embedding);
                if score > best {
                    best = score;
                    best_id = speaker.id;
                }
            }
        }
        if best >= MATCH_COSINE && best_id != 0 {
            if best >= 0.62
                && let Some(speaker) = self
                    .speakers
                    .iter_mut()
                    .find(|speaker| speaker.id == best_id)
                && speaker.embeds.len() < MAX_EMBEDDINGS
            {
                speaker.embeds.push(embedding);
            }
            return best_id;
        }
        let open = self.speakers.len() < MAX_SPEAKERS
            && seconds >= MIN_NEW_SECS
            && (self.speakers.is_empty() || best < NEW_SPEAKER_COSINE);
        if open {
            self.next += 1;
            let id = self.next;
            self.speakers.push(Enrolled {
                id,
                embeds: vec![embedding],
            });
            return id;
        }
        if best_id != 0 {
            return best_id;
        }
        self.speakers.first().map(|speaker| speaker.id).unwrap_or(0)
    }
}

/// True when segmentation found more than one local speaker.
pub fn has_multiple(turns: &[SpeechTurn]) -> bool {
    let mut first = None;
    for turn in turns {
        match first {
            None => first = Some(turn.local),
            Some(id) if id != turn.local => return true,
            Some(_) => {}
        }
    }
    false
}

/// Index of the longest slice, used when the line stays in one piece.
pub fn longest_index(turns: &[SpeechTurn]) -> Option<usize> {
    turns
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.duration()
                .partial_cmp(&right.duration())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
}

/// PCM for one turn. Empty when the range falls outside the buffer.
pub fn slice_turn(sample_rate: u32, samples: &[f32], turn: &SpeechTurn) -> Vec<f32> {
    if sample_rate == 0 || samples.is_empty() || turn.duration() <= 0.0 {
        return Vec::new();
    }
    let last = samples.len();
    let start = secs_to_index(turn.start, sample_rate, last);
    let end = secs_to_index(turn.end, sample_rate, last);
    if end <= start {
        Vec::new()
    } else {
        samples[start..end].to_vec()
    }
}

/// Lazy pyannote segmentation plus the embedding model used to name speakers.
pub struct Diarizer {
    diar: OfflineSpeakerDiarization,
    extractor: SpeakerEmbeddingExtractor,
}

impl Diarizer {
    /// Load `segmentation.onnx` and `embedding.onnx` from `model_dir`.
    pub fn open(model_dir: &Path, num_threads: i32) -> Result<Self> {
        let segmentation = model_dir.join("segmentation.onnx");
        let embedding = model_dir.join("embedding.onnx");
        if !segmentation.is_file() || !embedding.is_file() {
            anyhow::bail!(
                "missing segmentation.onnx or embedding.onnx in {}. Run scripts/download-asr.sh",
                model_dir.display()
            );
        }
        let installed =
            std::fs::read_to_string(model_dir.join("embedding.model")).unwrap_or_default();
        if installed.trim() != EMBEDDING_ID {
            anyhow::bail!(
                "speaker embedding in {} is not {EMBEDDING_ID}. Run scripts/download-asr.sh",
                model_dir.display()
            );
        }
        let threads = num_threads.max(1);
        let mut config = OfflineSpeakerDiarizationConfig::default();
        config.segmentation.pyannote.model = Some(path_string(&segmentation));
        config.segmentation.num_threads = threads;
        config.segmentation.debug = false;
        config.segmentation.provider = Some("cpu".to_string());
        config.embedding = SpeakerEmbeddingExtractorConfig {
            model: Some(path_string(&embedding)),
            num_threads: threads,
            debug: false,
            provider: Some("cpu".to_string()),
        };
        config.clustering.num_clusters = -1;
        config.clustering.threshold = 0.5;
        config.min_duration_on = MIN_TURN_SECS;
        config.min_duration_off = 0.2;
        let diar = OfflineSpeakerDiarization::create(&config)
            .ok_or_else(|| anyhow!("failed to create speaker diarizer"))?;
        let extractor = SpeakerEmbeddingExtractor::create(&config.embedding)
            .ok_or_else(|| anyhow!("failed to create speaker embedding extractor"))?;
        tracing::info!(dir = %model_dir.display(), "loaded speaker diarization");
        Ok(Self { diar, extractor })
    }

    /// Split `samples` into speaker turns. Empty when the clip is too short.
    ///
    /// Turns are not enrolled. Call [`SpeechTurn::label`] after the text is kept.
    pub fn segment(&self, sample_rate: u32, samples: &[f32]) -> Vec<SpeechTurn> {
        if sample_rate == 0 || samples.len() < sample_rate as usize / 2 {
            return Vec::new();
        }
        let model_rate = self.diar.sample_rate();
        if model_rate <= 0 {
            return Vec::new();
        }
        let audio = resample(samples, sample_rate, model_rate as u32);
        if audio.len() < model_rate as usize / 2 {
            return Vec::new();
        }
        let Some(result) = self.diar.process(&audio) else {
            return self.whole_clip(&audio, model_rate);
        };
        let segments = result.sort_by_start_time();
        if segments.is_empty() {
            return self.whole_clip(&audio, model_rate);
        }

        let mut grouped: HashMap<i32, (f32, Vec<(usize, usize)>)> = HashMap::new();
        for segment in &segments {
            let duration = segment.end - segment.start;
            if duration < MIN_TURN_SECS {
                continue;
            }
            let start = secs_to_index(segment.start, model_rate as u32, audio.len());
            let end = secs_to_index(segment.end, model_rate as u32, audio.len());
            if end <= start {
                continue;
            }
            let entry = grouped.entry(segment.speaker).or_default();
            entry.0 += duration;
            entry.1.push((start, end));
        }
        let totals: Vec<f32> = grouped.values().map(|(total, _)| *total).collect();
        if single_voice(&totals) {
            return self.whole_clip(&audio, model_rate);
        }

        let mut locals = Vec::new();
        for (local, (total, parts)) in &grouped {
            if *total < MIN_OTHER_SECS {
                continue;
            }
            let pooled = pool_parts(&audio, parts);
            if let Some(embedding) = self.embed_window(&pooled, model_rate) {
                locals.push((*local, embedding));
            }
        }
        if locals.len() <= 1 {
            return self.whole_clip(&audio, model_rate);
        }
        let units: Vec<Vec<f32>> = locals
            .iter()
            .filter_map(|(_, embedding)| unit(embedding.clone()))
            .collect();
        if units.len() != locals.len() {
            return self.whole_clip(&audio, model_rate);
        }
        if one_voice(&voice_groups(&units)) {
            tracing::debug!(
                speakers = locals.len(),
                "kept one speaker; the other voices in this clip were not clearly different"
            );
            return self.whole_clip(&audio, model_rate);
        }
        let embeddings: HashMap<i32, Vec<f32>> = locals.into_iter().collect();

        let mut turns: Vec<SpeechTurn> = Vec::new();
        for segment in segments {
            if segment.end - segment.start < MIN_TURN_SECS
                || !embeddings.contains_key(&segment.speaker)
            {
                continue;
            }
            if let Some(last) = turns.last_mut()
                && last.local == segment.speaker
                && segment.start - last.end < MERGE_GAP_SECS
            {
                last.end = segment.end;
                continue;
            }
            turns.push(SpeechTurn {
                local: segment.speaker,
                embedding: embeddings.get(&segment.speaker).cloned(),
                start: segment.start,
                end: segment.end,
            });
        }
        if turns.is_empty() {
            self.whole_clip(&audio, model_rate)
        } else {
            turns
        }
    }

    fn whole_clip(&self, audio: &[f32], sample_rate: i32) -> Vec<SpeechTurn> {
        let Some(embedding) = self.embed_window(audio, sample_rate) else {
            return Vec::new();
        };
        vec![SpeechTurn {
            local: 0,
            embedding: Some(embedding),
            start: 0.0,
            end: audio.len() as f32 / sample_rate as f32,
        }]
    }

    fn embed_window(&self, samples: &[f32], sample_rate: i32) -> Option<Vec<f32>> {
        if sample_rate <= 0 {
            return None;
        }
        let max = (sample_rate as usize).saturating_mul(6);
        if max == 0 || samples.is_empty() {
            return None;
        }
        if samples.len() <= max {
            return self.embed(samples, sample_rate);
        }
        let start = (samples.len() - max) / 2;
        self.embed(&samples[start..start + max], sample_rate)
    }

    fn embed(&self, samples: &[f32], sample_rate: i32) -> Option<Vec<f32>> {
        if sample_rate <= 0 || samples.len() < sample_rate as usize / 2 {
            return None;
        }
        let stream = self.extractor.create_stream()?;
        stream.accept_waveform(sample_rate, samples);
        stream.input_finished();
        if !self.extractor.is_ready(&stream) {
            return None;
        }
        self.extractor.compute(&stream)
    }
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn secs_to_index(secs: f32, sample_rate: u32, len: usize) -> usize {
    if sample_rate == 0 {
        return 0;
    }
    let index = (f64::from(secs) * f64::from(sample_rate)).round();
    if index <= 0.0 {
        0
    } else if index >= len as f64 {
        len
    } else {
        index as usize
    }
}

fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == 0 || to == 0 || samples.is_empty() {
        return Vec::new();
    }
    if from == to {
        return samples.to_vec();
    }
    if from.is_multiple_of(to) {
        let step = (from / to) as usize;
        let mut out = Vec::with_capacity(samples.len() / step);
        for chunk in samples.chunks_exact(step) {
            let sum: f32 = chunk.iter().sum();
            out.push(sum / step as f32);
        }
        return out;
    }
    let ratio = f64::from(from) / f64::from(to);
    let count = ((samples.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(count);
    let last = samples.len() - 1;
    for index in 0..count {
        let position = index as f64 * ratio;
        let left = (position.floor() as usize).min(last);
        let right = (left + 1).min(last);
        let frac = (position - left as f64) as f32;
        out.push(samples[left] + (samples[right] - samples[left]) * frac);
    }
    out
}

fn pool_parts(audio: &[f32], parts: &[(usize, usize)]) -> Vec<f32> {
    let mut pooled = Vec::new();
    for &(start, end) in parts {
        let end = end.min(audio.len());
        let start = start.min(end);
        pooled.extend_from_slice(&audio[start..end]);
    }
    pooled
}

/// Group indexes of unit embeddings. Voices at or above [`NEW_SPEAKER_COSINE`]
/// share a group, so a change in pitch does not become a second person.
fn voice_groups(embeddings: &[Vec<f32>]) -> Vec<usize> {
    let n = embeddings.len();
    let mut parent: Vec<usize> = (0..n).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            if dot(&embeddings[i], &embeddings[j]) >= NEW_SPEAKER_COSINE {
                let left = find(&mut parent, i);
                let right = find(&mut parent, j);
                if left != right {
                    parent[right] = left;
                }
            }
        }
    }
    let mut dense = HashMap::new();
    let mut groups = Vec::with_capacity(n);
    for i in 0..n {
        let root = find(&mut parent, i);
        let next = dense.len();
        groups.push(*dense.entry(root).or_insert(next));
    }
    groups
}

fn one_voice(groups: &[usize]) -> bool {
    groups
        .first()
        .is_none_or(|first| groups.iter().all(|group| group == first))
}

fn find(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

fn unit(embedding: Vec<f32>) -> Option<Vec<f32>> {
    let mut norm = 0.0;
    for value in &embedding {
        norm += value * value;
    }
    norm = norm.sqrt();
    if norm < 1.0e-6 {
        return None;
    }
    Some(embedding.into_iter().map(|value| value / norm).collect())
}

/// True when a second voice is too short to treat the clip as a conversation.
fn single_voice(totals: &[f32]) -> bool {
    let mut kept: Vec<f32> = totals
        .iter()
        .copied()
        .filter(|secs| *secs >= MIN_TURN_SECS)
        .collect();
    if kept.len() <= 1 {
        return true;
    }
    kept.sort_by(|left, right| right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal));
    let second = kept[1];
    let total: f32 = kept.iter().sum();
    second < MIN_OTHER_SECS || total <= 0.0 || second / total < MIN_OTHER_SHARE
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() {
        return f32::MIN;
    }
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimates_48k_to_16k_by_averaging() {
        let samples = vec![0.25; 9];
        let out = resample(&samples, 48_000, 16_000);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|sample| (*sample - 0.25).abs() < 1.0e-6));
    }

    #[test]
    fn book_reuses_a_voice_and_caps_growth() {
        let mut book = SpeakerBook::default();
        let first = book.assign(vec![1.0, 0.0, 0.0], 2.0);
        let again = book.assign(vec![0.9, 0.1, 0.0], 2.0);
        assert_eq!(first, 1);
        assert_eq!(again, first);
        let second = book.assign(vec![0.0, 1.0, 0.0], 2.0);
        assert_eq!(second, 2);
        assert_eq!(book.assign(vec![0.0, 0.0, 0.0], 2.0), 0);
    }

    #[test]
    fn book_stops_opening_speakers() {
        let mut book = SpeakerBook::default();
        for index in 0..MAX_SPEAKERS {
            let mut embedding = vec![0.0; MAX_SPEAKERS];
            embedding[index] = 1.0;
            assert_eq!(book.assign(embedding, 2.0), (index as u32) + 1);
        }
        let mut extra = vec![0.0; MAX_SPEAKERS];
        extra.push(1.0);
        let assigned = book.assign(extra, 2.0);
        assert!(assigned >= 1 && assigned <= MAX_SPEAKERS as u32);
        assert_eq!(book.speakers.len(), MAX_SPEAKERS);
    }

    #[test]
    fn short_different_voice_stays_with_the_current_speaker() {
        let mut book = SpeakerBook::default();
        let first = book.assign(vec![1.0, 0.0], 2.0);
        let brief = book.assign(vec![0.0, 1.0], 0.6);
        assert_eq!(brief, first);
        assert_eq!(book.speakers.len(), 1);
        assert!(single_voice(&[5.0, 0.4]));
        assert!(single_voice(&[4.0, 0.8]));
        assert!(!single_voice(&[3.0, 2.0]));
    }

    #[test]
    fn a_change_in_pitch_stays_one_voice() {
        let same = voice_groups(&[vec![1.0, 0.0], vec![0.8, 0.6]]);
        assert!(one_voice(&same), "{same:?}");
        let apart = voice_groups(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_ne!(apart[0], apart[1]);
    }

    #[test]
    fn labels_a_kept_slice_and_ignores_the_rest() {
        let mut book = SpeakerBook::default();
        let mut turns = vec![
            SpeechTurn {
                local: 0,
                embedding: Some(vec![1.0, 0.0]),
                start: 0.0,
                end: 1.0,
            },
            SpeechTurn {
                local: 1,
                embedding: Some(vec![0.0, 1.0]),
                start: 1.2,
                end: 3.0,
            },
        ];
        assert!(has_multiple(&turns));
        assert_eq!(longest_index(&turns), Some(1));
        assert_eq!(book.speakers.len(), 0);
        assert_eq!(turns[1].label(&mut book), 1);
        assert_eq!(book.speakers.len(), 1);
        let audio = vec![1.0; 16_000];
        assert_eq!(slice_turn(16_000, &audio, &turns[0]).len(), 16_000);
    }

    #[test]
    fn loads_and_separates_two_english_speakers_when_requested() {
        if std::env::var("KONNNYAKU_DIAR").is_err() {
            return;
        }
        let dir = Path::new("models/asr/diar");
        let model = Diarizer::open(dir, 2).expect("load diarizer");
        let wav = dir.join("two-speakers-en.wav");
        if !wav.is_file() {
            return;
        }
        let audio = crate::audio::load_wav_mono_f32(&wav).expect("wav");
        let mut book = SpeakerBook::default();
        let mut turns = model.segment(audio.sample_rate, &audio.samples);
        assert!(has_multiple(&turns), "expected two speakers, got {turns:?}");
        for turn in &mut turns {
            let _ = turn.label(&mut book);
        }
        let enrolled = book.speakers.len();
        assert!(enrolled >= 2, "enrolled {enrolled}");
        let mut again = model.segment(audio.sample_rate, &audio.samples);
        for turn in &mut again {
            let id = turn.label(&mut book);
            assert!(id >= 1 && id <= enrolled as u32, "new speaker {id}");
        }
        assert_eq!(book.speakers.len(), enrolled);
    }
}
