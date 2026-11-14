//! Translated captions from the running live session, for a later command.
//!
//! `live` rewrites `meetings/captions.json` when a line is committed. `captions`
//! prints that file. The directory is gitignored. Nothing here calls a model
//! or the network.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};
use serde::Serialize;

const MAX_LINES: usize = 400;

#[derive(Clone, Serialize)]
struct Saved {
    primary: String,
    lines: Vec<SavedLine>,
}

#[derive(Clone, Serialize)]
struct SavedLine {
    #[serde(skip_serializing_if = "is_zero")]
    speaker: u32,
    primary: String,
    source: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    translation: String,
    text: String,
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

/// Caption file shared by the live session and the window's clear button.
pub struct CaptionLog {
    path: PathBuf,
    state: Mutex<Saved>,
}

impl CaptionLog {
    /// Start a session. Replaces any captions left by the previous process.
    pub fn create(path: &Path, primary: &str) -> Result<Self> {
        let primary = known_language(primary).unwrap_or("ja").to_string();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        let saved = Saved {
            primary,
            lines: Vec::new(),
        };
        write_atomic(path, &saved)?;
        Ok(Self {
            path: path.to_path_buf(),
            state: Mutex::new(saved),
        })
    }

    /// Remember committed lines. `translation` is empty when the speech is already
    /// in the display language, and `text` is the line the window shows.
    pub fn record<'a>(
        &self,
        primary: &str,
        lines: impl IntoIterator<Item = (u32, &'a str, &'a str)>,
    ) -> Result<()> {
        let Some(primary) = known_language(primary) else {
            return Ok(());
        };
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut added = false;
        for (speaker, source, translation) in lines {
            let source = source.trim();
            let translation = translation.trim();
            if source.is_empty() && translation.is_empty() {
                continue;
            }
            let text = if translation.is_empty() {
                source
            } else {
                translation
            };
            state.lines.push(SavedLine {
                speaker,
                primary: primary.to_string(),
                source: source.to_string(),
                translation: translation.to_string(),
                text: text.to_string(),
            });
            added = true;
        }
        if !added {
            return Ok(());
        }
        if state.lines.len() > MAX_LINES {
            let extra = state.lines.len() - MAX_LINES;
            state.lines.drain(0..extra);
        }
        write_atomic(&self.path, &state)
    }

    /// The window changed its display language. Existing lines keep the language
    /// they were shown in.
    pub fn set_primary(&self, primary: &str) {
        let Some(primary) = known_language(primary) else {
            return;
        };
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if state.primary == primary {
            return;
        }
        state.primary = primary.to_string();
        if let Err(err) = write_atomic(&self.path, &state) {
            tracing::warn!(%err, "could not save the caption language");
        }
    }

    /// Drop every saved line. Used when the caption window is cleared.
    pub fn clear(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if state.lines.is_empty() {
            return;
        }
        state.lines.clear();
        if let Err(err) = write_atomic(&self.path, &state) {
            tracing::warn!(%err, "could not clear saved captions");
        }
    }
}

/// Pretty JSON for an agent. Missing file means live has not been started here.
pub fn read_json(path: &Path) -> Result<String> {
    let meta = std::fs::symlink_metadata(path).map_err(|_| {
        anyhow!(
            "実行中の文字起こしがありません。konnnyaku live を起動してから、もう一度実行してください。"
        )
    })?;
    if meta.file_type().is_symlink() || !meta.file_type().is_file() {
        anyhow::bail!("文字起こしの読み取りに失敗しました。");
    }
    if meta.len() > 1_048_576 {
        anyhow::bail!("文字起こしが大きすぎます。");
    }
    let text = std::fs::read_to_string(path).context("文字起こしの読み取りに失敗しました。")?;
    let value: serde_json::Value =
        serde_json::from_str(&text).context("文字起こしの読み取りに失敗しました。")?;
    if !value.get("lines").is_some_and(serde_json::Value::is_array) {
        anyhow::bail!("文字起こしの読み取りに失敗しました。");
    }
    serde_json::to_string_pretty(&value).context("format captions")
}

fn known_language(value: &str) -> Option<&str> {
    match value {
        "en" | "ja" => Some(value),
        _ => None,
    }
}

fn write_atomic(path: &Path, saved: &Saved) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(saved).context("format captions")?;
    let tmp = path.with_extension("json.partial");
    {
        let mut options = OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&tmp)
            .with_context(|| format!("write {}", tmp.display()))?;
        file.write_all(&bytes)
            .with_context(|| format!("write {}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("replace {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(dir: &Path) -> CaptionLog {
        let _ = std::fs::remove_dir_all(dir);
        CaptionLog::create(&dir.join("captions.json"), "ja").expect("create")
    }

    #[test]
    fn committed_translation_is_what_an_agent_reads() {
        let dir = std::env::temp_dir().join(format!("konnnyaku-captions-{}", std::process::id()));
        let log = sample(&dir);
        log.record(
            "ja",
            [(0, "No, wait.", "いや、待って。"), (2, "説明がある。", "")],
        )
        .expect("record");
        let json = read_json(&dir.join("captions.json")).expect("read");
        let value: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(value["primary"], "ja");
        assert_eq!(value["lines"][0]["text"], "いや、待って。");
        assert_eq!(value["lines"][0]["translation"], "いや、待って。");
        assert!(value["lines"][0].get("speaker").is_none());
        assert_eq!(value["lines"][1]["text"], "説明がある。");
        assert_eq!(value["lines"][1]["speaker"], 2);
        assert!(value["lines"][1].get("translation").is_none());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.join("captions.json"))
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn clear_and_a_language_change_keep_old_lines_honest() {
        let dir =
            std::env::temp_dir().join(format!("konnnyaku-captions-clear-{}", std::process::id()));
        let log = sample(&dir);
        log.record("ja", [(0, "hello", "こんにちは。")])
            .expect("record");
        log.set_primary("en");
        let json = read_json(&dir.join("captions.json")).expect("read");
        let value: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(value["primary"], "en");
        assert_eq!(value["lines"][0]["primary"], "ja");
        assert_eq!(value["lines"][0]["text"], "こんにちは。");
        log.clear();
        let json = read_json(&dir.join("captions.json")).expect("read");
        let value: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(value["lines"].as_array().map(Vec::len), Some(0));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_file_keeps_only_the_newest_lines() {
        let dir =
            std::env::temp_dir().join(format!("konnnyaku-captions-cap-{}", std::process::id()));
        let log = sample(&dir);
        for index in 0..(MAX_LINES + 3) {
            let source = format!("line {index}");
            log.record("en", [(0, source.as_str(), "")])
                .expect("record");
        }
        let json = read_json(&dir.join("captions.json")).expect("read");
        let value: serde_json::Value = serde_json::from_str(&json).expect("json");
        let lines = value["lines"].as_array().expect("lines");
        assert_eq!(lines.len(), MAX_LINES);
        assert_eq!(lines[0]["text"], "line 3");
        assert_eq!(
            lines.last().unwrap()["text"],
            format!("line {}", MAX_LINES + 2)
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_missing_file_tells_the_agent_to_start_live() {
        let path = std::env::temp_dir().join(format!(
            "konnnyaku-captions-missing-{}-{}.json",
            std::process::id(),
            MAX_LINES
        ));
        let _ = std::fs::remove_file(&path);
        let err = read_json(&path).expect_err("missing");
        assert!(err.to_string().contains("live"));
    }
}
