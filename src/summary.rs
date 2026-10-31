use crate::text;

const POINT_LIMIT: usize = 3;
const MAX_LINES: usize = 80;
const MIN_CONTENT_WORDS: usize = 12;
const DUPLICATE_JACCARD: f32 = 0.55;

/// English lines heard from the microphone, reduced to a few points.
#[derive(Debug, Default)]
pub struct SummarySession {
    lines: Vec<String>,
    points: Vec<String>,
}

impl SummarySession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one finalized utterance. Returns true when the points changed.
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
        let next = select_points(&self.lines);
        if next == self.points {
            return false;
        }
        self.points = next;
        !self.points.is_empty()
    }

    pub fn points(&self) -> &[String] {
        &self.points
    }
}

fn select_points(lines: &[String]) -> Vec<String> {
    let mut kept: Vec<&str> = Vec::new();
    for line in lines {
        if kept
            .iter()
            .any(|prev| jaccard(prev, line) >= DUPLICATE_JACCARD)
        {
            continue;
        }
        kept.push(line.as_str());
    }
    kept.retain(|line| content_words(std::iter::once(*line)).next().is_some());
    if content_words(kept.iter().copied()).count() < MIN_CONTENT_WORDS {
        return Vec::new();
    }
    if kept.len() <= POINT_LIMIT {
        return kept.into_iter().map(str::to_string).collect();
    }
    let mut ranked: Vec<(usize, usize)> = kept
        .iter()
        .enumerate()
        .map(|(index, line)| (index, content_words(std::iter::once(*line)).count()))
        .collect();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
    let mut chosen: Vec<usize> = ranked
        .into_iter()
        .take(POINT_LIMIT)
        .map(|(index, _)| index)
        .collect();
    chosen.sort_unstable();
    chosen
        .into_iter()
        .map(|index| kept[index].to_string())
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
    fn short_or_repeated_speech_has_no_summary() {
        let mut session = SummarySession::new();
        assert!(!session.push("yeah"));
        assert!(session.points().is_empty());
        assert!(!session.push("update the server"));
        assert!(session.points().is_empty());
        assert!(!session.push("update the server"));
        assert!(session.points().is_empty());
    }

    #[test]
    fn keeps_distinct_points_in_time_order() {
        let mut session = SummarySession::new();
        assert!(
            !session.push("We need to update the server before Friday and ship the patch tonight")
        );
        assert!(
            !session.push("We need to update the server before Friday and ship the patch tonight")
        );
        assert!(session
            .push("The migration copies the user table and the audit log into the new cluster"));
        assert!(!session.push("yeah yeah"));
        assert!(session
            .push("After that we tell support the window is Thursday night not Friday morning"));
        assert_eq!(
            session.points(),
            &[
                "We need to update the server before Friday and ship the patch tonight",
                "The migration copies the user table and the audit log into the new cluster",
                "After that we tell support the window is Thursday night not Friday morning",
            ]
        );
    }

    #[test]
    fn drops_a_near_duplicate_and_keeps_a_later_topic() {
        let mut session = SummarySession::new();
        session.push("We need to update the server before Friday and ship the patch tonight");
        session.push("We need to update the server before Friday and ship the patch");
        session.push("The migration copies the user table and the audit log into the new cluster");
        let points = session.points();
        assert_eq!(points.len(), 2);
        assert!(points[0].contains("update the server"));
        assert!(points[1].contains("migration"));
    }
}
