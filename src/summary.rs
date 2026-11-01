use crate::text;

const SECTION_LIMIT: usize = 5;
const MAX_LINES: usize = 80;
const MIN_CONTENT_WORDS: usize = 12;
const DUPLICATE_JACCARD: f32 = 0.55;

/// English record of one meeting. Japanese is filled in by the caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeetingDraft {
    pub transcript: Vec<String>,
    pub topics: Vec<String>,
    pub decisions: Vec<String>,
    pub actions: Vec<String>,
}

/// Lines heard so far, split into a meeting note.
#[derive(Debug, Default)]
pub struct SummarySession {
    lines: Vec<String>,
}

impl SummarySession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one finalized utterance. Returns true when the transcript grew.
    pub fn push(&mut self, raw: &str) -> bool {
        let Some(clean) = text::prepare_english(raw) else {
            return false;
        };
        if self.lines.last().is_some_and(|prev| prev == &clean) {
            return false;
        }
        self.lines.push(clean);
        if self.lines.len() > MAX_LINES {
            let extra = self.lines.len() - MAX_LINES;
            self.lines.drain(0..extra);
        }
        true
    }

    pub fn draft(&self) -> MeetingDraft {
        let mut transcript = Vec::new();
        let mut topics = Vec::new();
        let mut decisions = Vec::new();
        let mut actions = Vec::new();
        for line in &self.lines {
            if transcript
                .iter()
                .any(|prev: &String| jaccard(prev, line) >= DUPLICATE_JACCARD)
            {
                continue;
            }
            transcript.push(line.clone());
            if content_words(std::iter::once(line.as_str()))
                .next()
                .is_none()
            {
                continue;
            }
            match classify(line) {
                LineKind::Decision => decisions.push(line.clone()),
                LineKind::Action => actions.push(line.clone()),
                LineKind::Topic => topics.push(line.clone()),
            }
        }
        if content_words(transcript.iter().map(String::as_str)).count() < MIN_CONTENT_WORDS {
            topics.clear();
            decisions.clear();
            actions.clear();
        } else {
            topics = cap_section(topics);
            decisions = cap_section(decisions);
            actions = cap_section(actions);
        }
        MeetingDraft {
            transcript,
            topics,
            decisions,
            actions,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LineKind {
    Topic,
    Decision,
    Action,
}

fn classify(line: &str) -> LineKind {
    let lower = line.to_ascii_lowercase();
    if contains_any(
        &lower,
        &[
            "we decided",
            "decision is",
            "we agree",
            "we agreed",
            "let's go with",
            "lets go with",
            "approved",
        ],
    ) {
        return LineKind::Decision;
    }
    if contains_any(
        &lower,
        &[
            "let's ",
            "lets ",
            "we need to",
            "we should",
            "i will",
            "i'll",
            "please ",
            "follow up",
            "action item",
            "next step",
            "we will",
        ],
    ) {
        return LineKind::Action;
    }
    LineKind::Topic
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn cap_section(lines: Vec<String>) -> Vec<String> {
    if lines.len() <= SECTION_LIMIT {
        return lines;
    }
    let mut ranked: Vec<(usize, usize)> = lines
        .iter()
        .enumerate()
        .map(|(index, line)| (index, content_words(std::iter::once(line.as_str())).count()))
        .collect();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
    let mut chosen: Vec<usize> = ranked
        .into_iter()
        .take(SECTION_LIMIT)
        .map(|(index, _)| index)
        .collect();
    chosen.sort_unstable();
    chosen
        .into_iter()
        .map(|index| lines[index].clone())
        .collect()
}

fn jaccard(left: &str, right: &str) -> f32 {
    let left_tokens: Vec<String> = tokens(left).collect();
    let right_tokens: Vec<String> = tokens(right).collect();
    if left_tokens.is_empty() || right_tokens.is_empty() {
        return 0.0;
    }
    let mut shared = 0usize;
    for token in &left_tokens {
        if right_tokens.iter().any(|other| other == token) {
            shared += 1;
        }
    }
    let union = left_tokens.len() + right_tokens.len() - shared;
    shared as f32 / union as f32
}

fn content_words<'a>(lines: impl Iterator<Item = &'a str>) -> impl Iterator<Item = String> {
    lines.flat_map(tokens).filter(|token| !is_stopword(token))
}

fn tokens(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "a" | "an"
            | "the"
            | "and"
            | "or"
            | "but"
            | "to"
            | "of"
            | "in"
            | "on"
            | "for"
            | "with"
            | "at"
            | "from"
            | "by"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "it"
            | "this"
            | "that"
            | "we"
            | "you"
            | "i"
            | "they"
            | "he"
            | "she"
            | "do"
            | "did"
            | "yes"
            | "yeah"
            | "ok"
            | "okay"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_speech_keeps_a_transcript_without_sections() {
        let mut session = SummarySession::new();
        assert!(!session.push("yeah"));
        assert!(session.push("update the server"));
        let draft = session.draft();
        assert_eq!(draft.transcript, vec!["update the server".to_string()]);
        assert!(draft.topics.is_empty());
        assert!(draft.decisions.is_empty());
        assert!(draft.actions.is_empty());
    }

    #[test]
    fn splits_actions_decisions_and_topics() {
        let mut session = SummarySession::new();
        session.push("We need to update the server before Friday and ship the patch tonight");
        session.push("We need to update the server before Friday and ship the patch tonight");
        session.push("The migration copies the user table and the audit log into the new cluster");
        session.push("yeah yeah");
        session.push("We decided to keep the old API until the migration finishes next month");
        session.push("After that we tell support the window is Thursday night not Friday morning");
        let draft = session.draft();
        assert_eq!(
            draft.actions,
            vec!["We need to update the server before Friday and ship the patch tonight"]
        );
        assert_eq!(
            draft.decisions,
            vec!["We decided to keep the old API until the migration finishes next month"]
        );
        assert_eq!(
            draft.topics,
            vec![
                "The migration copies the user table and the audit log into the new cluster",
                "After that we tell support the window is Thursday night not Friday morning",
            ]
        );
    }

    #[test]
    fn drops_a_near_duplicate_action() {
        let mut session = SummarySession::new();
        session.push("We need to update the server before Friday and ship the patch tonight");
        session.push("We need to update the server before Friday and ship the patch");
        session.push("The migration copies the user table and the audit log into the new cluster");
        let draft = session.draft();
        assert_eq!(draft.actions.len(), 1);
        assert_eq!(draft.topics.len(), 1);
        assert!(draft.topics[0].contains("migration"));
    }
}
