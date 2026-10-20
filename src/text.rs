/// English and Japanese hygiene for live captions.
///
/// Zipformer and FuguMT both emit garbage on music and fragments. These helpers
/// drop that before it becomes a subtitle.

/// Drop unknown tokens, laughter, and too-short fragments.
pub fn prepare_english(text: &str) -> Option<String> {
    let cleaned = text.replace("<unk>", " ").replace("<UNK>", " ");
    let collapsed = collapse_ws(&cleaned);
    if collapsed.is_empty() || is_english_junk(&collapsed) {
        None
    } else {
        Some(collapsed)
    }
}

/// Strip SentencePiece artifacts and collapse character runs.
pub fn cleanup_japanese(text: &str) -> String {
    let stripped = text
        .replace(['▁', ' '], "")
        .replace("<unk>", "")
        .replace("<UNK>", "");
    collapse_char_runs(&stripped, 4).trim().to_string()
}

/// Keep a Japanese caption only when it is not laughter / `<unk>` residue.
pub fn keep_japanese(text: &str) -> Option<String> {
    let cleaned = cleanup_japanese(text);
    if cleaned.is_empty() || is_japanese_junk(&cleaned) {
        None
    } else {
        Some(cleaned)
    }
}

fn collapse_ws(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_english_junk(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    let letters: String = text.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    let letter_count = letters.chars().count();

    if words.len() < 2 && letter_count < 8 {
        return true;
    }

    if is_laughter(&letters) {
        return true;
    }

    if words.len() >= 4 {
        let first = words[0].to_ascii_lowercase();
        if words.iter().all(|word| word.eq_ignore_ascii_case(&first)) {
            return true;
        }
    }

    if letter_count >= 8 {
        if let Some(first) = letters.chars().next() {
            if letters.chars().all(|ch| ch.eq_ignore_ascii_case(&first)) {
                return true;
            }
        }
    }

    false
}

fn is_laughter(letters: &str) -> bool {
    let lower = letters.to_ascii_lowercase();
    if lower.chars().count() < 4 {
        return false;
    }
    lower.chars().all(|ch| ch == 'h' || ch == 'a')
}

fn is_japanese_junk(text: &str) -> bool {
    let significant: String = text
        .chars()
        .filter(|ch| !matches!(ch, '…' | '.' | '。' | 'ー' | '、' | '！' | '？' | '!' | '?'))
        .collect();
    if significant.is_empty() {
        return true;
    }
    let only_laughter = significant
        .chars()
        .all(|ch| matches!(ch, 'は' | 'あ' | 'ハ' | 'ア' | 'h' | 'a' | 'H' | 'A'));
    only_laughter && significant.chars().count() >= 2
}

fn collapse_char_runs(text: &str, max_run: usize) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev: Option<char> = None;
    let mut run = 0usize;
    for ch in text.chars() {
        if Some(ch) == prev {
            run += 1;
            if run <= max_run {
                out.push(ch);
            }
        } else {
            prev = Some(ch);
            run = 1;
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_unk_and_laughter() {
        assert_eq!(prepare_english("<unk>"), None);
        assert_eq!(prepare_english("ha ha ha ha"), None);
        assert_eq!(prepare_english("THE"), None);
    }

    #[test]
    fn keeps_real_sentences() {
        assert_eq!(
            prepare_english("We need to update the server."),
            Some("We need to update the server.".to_string())
        );
        assert_eq!(prepare_english("important"), Some("important".to_string()));
        assert_eq!(
            prepare_english("The unknown server failed."),
            Some("The unknown server failed.".to_string())
        );
    }

    #[test]
    fn drops_japanese_garbage() {
        assert_eq!(keep_japanese("ははは…"), None);
        assert_eq!(keep_japanese("<unk>"), None);
        assert_eq!(
            keep_japanese("サーバーを更新する必要がある。"),
            Some("サーバーを更新する必要がある。".to_string())
        );
    }
}
