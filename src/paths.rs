//! Where models and caption files live, independent of the working directory.
//!
//! `KONNNYAKU_HOME` wins. A checkout that already has `models/asr/zipformer-en`
//! keeps using the working directory, so development stays as it was. Otherwise
//! files go under `~/Library/Application Support/konnnyaku` on macOS.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub const ASR_DIR: &str = "asr/zipformer-en";
pub const RESCORE_DIR: &str = "asr/parakeet-en";
pub const LID_DIR: &str = "asr/whisper-tiny";
pub const HAAS_DIR: &str = "asr/haas-ja";
pub const DIAR_DIR: &str = "asr/diar";
pub const VAD_MODEL: &str = "asr/silero_vad.onnx";
pub const MT_DIR: &str = "translation/marian-en-ja/onnx";
pub const MT_JA_EN_DIR: &str = "translation/marian-ja-en/onnx";

const REPO_MARKER: &str = "models/asr/zipformer-en";

/// Base directory holding `models/` and `meetings/`.
pub fn home() -> &'static Path {
    static HOME: OnceLock<PathBuf> = OnceLock::new();
    HOME.get_or_init(|| {
        resolve(
            std::env::var_os("KONNNYAKU_HOME"),
            std::env::current_dir().ok(),
            std::env::var_os("HOME"),
        )
    })
}

/// A model file or directory under `models/`.
pub fn model(relative: &str) -> PathBuf {
    home().join("models").join(relative)
}

/// Captions saved by `live` and read by `captions`.
pub fn captions_file() -> PathBuf {
    home().join("meetings").join("captions.json")
}

/// Meeting note written by `summary`.
pub fn meetings_dir() -> PathBuf {
    home().join("meetings")
}

fn resolve(
    env_home: Option<std::ffi::OsString>,
    cwd: Option<PathBuf>,
    user_home: Option<std::ffi::OsString>,
) -> PathBuf {
    if let Some(value) = env_home.filter(|value| !value.is_empty()) {
        return PathBuf::from(value);
    }
    if let Some(cwd) = cwd
        && cwd.join(REPO_MARKER).is_dir()
    {
        return cwd;
    }
    let user_home = user_home
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    if cfg!(target_os = "macos") {
        user_home
            .join("Library")
            .join("Application Support")
            .join("konnnyaku")
    } else {
        user_home.join(".local").join("share").join("konnnyaku")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_home_wins_then_checkout_then_user_data() {
        let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        assert_eq!(
            resolve(
                Some("/tmp/kk".into()),
                Some(checkout.clone()),
                Some("/Users/a".into())
            ),
            PathBuf::from("/tmp/kk")
        );
        if checkout.join(REPO_MARKER).is_dir() {
            assert_eq!(
                resolve(None, Some(checkout.clone()), Some("/Users/a".into())),
                checkout
            );
        }
        let elsewhere = std::env::temp_dir();
        let data = resolve(Some("".into()), Some(elsewhere), Some("/Users/a".into()));
        if cfg!(target_os = "macos") {
            assert_eq!(
                data,
                PathBuf::from("/Users/a/Library/Application Support/konnnyaku")
            );
        } else {
            assert_eq!(data, PathBuf::from("/Users/a/.local/share/konnnyaku"));
        }
    }
}
