//! `konnnyaku setup`: fetch the models the app needs into `paths::home()`.
//!
//! Every file is pinned by SHA-256. Downloads go to a staging directory under
//! `models/.setup`, archives are unpacked there with only the listed members,
//! and a file is moved into place only after its hash matches. The system
//! `curl` and `tar` do the transfer, so nothing else has to be installed.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow};
use sha2::{Digest, Sha256};

use crate::paths;

/// Where a pinned file comes from.
enum Source {
    /// Fetch the URL directly.
    Url(&'static str),
    /// Member path inside the component's archive.
    Member(&'static str),
}

struct Pinned {
    name: &'static str,
    source: Source,
    sha256: &'static str,
}

struct Component {
    id: &'static str,
    label: &'static str,
    dir: &'static str,
    archive: Option<&'static str>,
    files: &'static [Pinned],
    /// Extra file written after the pinned files, such as the diar model stamp.
    note: Option<(&'static str, &'static str)>,
}

macro_rules! sherpa_asr {
    ($name:literal) => {
        concat!(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/",
            $name
        )
    };
}

macro_rules! haas {
    ($name:literal) => {
        concat!(
            "https://huggingface.co/oboroge0/Haas-ja-v1.0-onnx-quantized/resolve/7f5ab0a4b1bf39db78afd6691eeecbc4f9d42e39/",
            $name
        )
    };
}

/// Marian ONNX exported from FuguMT (CC-BY-SA-4.0) and published with this repo.
macro_rules! models_release {
    ($name:literal) => {
        concat!(
            "https://github.com/konnta0/konnnyaku/releases/download/models-v1/",
            $name
        )
    };
}

const COMPONENTS: &[Component] = &[
    Component {
        id: "zipformer-en",
        label: "English streaming ASR (Zipformer)",
        dir: paths::ASR_DIR,
        archive: Some(sherpa_asr!(
            "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
        )),
        files: &[
            Pinned {
                name: "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                source: Source::Member(
                    "sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                ),
                sha256: "5022b2eca5b19d1bc104fcf33e26bc32604b7df553cd2e1f62e31dc7b05e9c87",
            },
            Pinned {
                name: "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
                source: Source::Member(
                    "sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
                ),
                sha256: "85914f840b71110487ece7a30ebd0bd4b8fb5c2623215bba3aa5ab4c138fc591",
            },
            Pinned {
                name: "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                source: Source::Member(
                    "sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                ),
                sha256: "abd5e30f3f16fc510605c6029dba33f10e4386bd75c5bdc30cf94076864db10d",
            },
            Pinned {
                name: "tokens.txt",
                source: Source::Member("sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt"),
                sha256: "49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb",
            },
        ],
        note: None,
    },
    Component {
        id: "silero-vad",
        label: "Voice activity (Silero VAD)",
        dir: "asr",
        archive: None,
        files: &[Pinned {
            name: "silero_vad.onnx",
            source: Source::Url(sherpa_asr!("silero_vad.onnx")),
            sha256: "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        }],
        note: None,
    },
    Component {
        id: "parakeet-en",
        label: "English recognizer (Parakeet TDT 0.6B v2)",
        dir: paths::RESCORE_DIR,
        archive: Some(sherpa_asr!(
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
        )),
        files: &[
            Pinned {
                name: "encoder.int8.onnx",
                source: Source::Member(
                    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx",
                ),
                sha256: "92a10bd9c23936ea0566c29f963bc57c2a5df4261b6a8921c39abea66557ca8b",
            },
            Pinned {
                name: "decoder.int8.onnx",
                source: Source::Member(
                    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx",
                ),
                sha256: "4d097bae3bac5c96018d5293ba39f1b657296102b142affda40378f38b020f41",
            },
            Pinned {
                name: "joiner.int8.onnx",
                source: Source::Member(
                    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx",
                ),
                sha256: "77a29f3e5ac7d37074e0af465391b80f314015d5b97f0f90c6c7767f1d56f428",
            },
            Pinned {
                name: "tokens.txt",
                source: Source::Member("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt"),
                sha256: "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d",
            },
        ],
        note: None,
    },
    Component {
        id: "whisper-tiny",
        label: "Language detection (Whisper tiny)",
        dir: paths::LID_DIR,
        archive: Some(sherpa_asr!("sherpa-onnx-whisper-tiny.tar.bz2")),
        files: &[
            Pinned {
                name: "tiny-encoder.int8.onnx",
                source: Source::Member("sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx"),
                sha256: "d24fb083ae3b1041fc24e97971d60e280c9342201fbb67b0ab428a8b4a51a434",
            },
            Pinned {
                name: "tiny-decoder.int8.onnx",
                source: Source::Member("sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx"),
                sha256: "d2fece8dd42771f1df975c6c0445770d0c292bf7547c2cae04a6c0cc57540925",
            },
            Pinned {
                name: "tiny-tokens.txt",
                source: Source::Member("sherpa-onnx-whisper-tiny/tiny-tokens.txt"),
                sha256: "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126",
            },
        ],
        note: None,
    },
    Component {
        id: "haas-ja",
        label: "Japanese recognizer (Haas)",
        dir: paths::HAAS_DIR,
        archive: None,
        files: &[
            Pinned {
                name: "encoder.quantized.onnx",
                source: Source::Url(haas!("encoder.quantized.onnx")),
                sha256: "aba291d61d3d594bc6fb8c7f90acbd980e7bc7bf6b7caaf3965ddc25e6f42fac",
            },
            Pinned {
                name: "encoder.data",
                source: Source::Url(haas!("encoder.data")),
                sha256: "30befbf6aeb75383e6551f5e58de53efdb76972ddc7d0f76070c517899873586",
            },
            Pinned {
                name: "decoder.onnx",
                source: Source::Url(haas!("decoder.onnx")),
                sha256: "2947cb5512699cbba753fe45fe221428aadaa9856597debbb7aabe19cc253622",
            },
            Pinned {
                name: "joiner.onnx",
                source: Source::Url(haas!("joiner.onnx")),
                sha256: "8a68320e750890ff9e944a794b41ae1c42c6f439ac4c273503458a6b42adc12d",
            },
            Pinned {
                name: "tokens.txt",
                source: Source::Url(haas!("tokens.txt")),
                sha256: "ec78b13c5d9f156cf0d5fdbd6b5aec5785680f319c92630c11d5d11b14b83a84",
            },
        ],
        note: None,
    },
    Component {
        id: "diar-segmentation",
        label: "Speaker turns (pyannote segmentation 3.0)",
        dir: paths::DIAR_DIR,
        archive: Some(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        ),
        files: &[Pinned {
            name: "segmentation.onnx",
            source: Source::Member("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
            sha256: "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
        }],
        note: None,
    },
    Component {
        id: "diar-embedding",
        label: "Speaker voices (3D-Speaker CAM++ zh-en)",
        dir: paths::DIAR_DIR,
        archive: None,
        files: &[Pinned {
            name: "embedding.onnx",
            // The upstream release tag really is spelled "recongition".
            source: Source::Url(
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
            ),
            sha256: "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2",
        }],
        note: Some(("embedding.model", crate::diar::EMBEDDING_ID)),
    },
    Component {
        id: "marian-en-ja",
        label: "English to Japanese translation (FuguMT)",
        dir: paths::MT_DIR,
        archive: Some(models_release!("marian-en-ja-onnx.tar.gz")),
        files: &[
            Pinned {
                name: "encoder_model.onnx",
                source: Source::Member("encoder_model.onnx"),
                sha256: "f2cbbbbdac7bbb470e58d529e85bfafdd7cac965f1cf7ee423636f1b34dc52df",
            },
            Pinned {
                name: "encoder_model.onnx.data",
                source: Source::Member("encoder_model.onnx.data"),
                sha256: "9f915166245f5d9555c6dfdc5089ea49929b139fe07dee9d46ec7d8bfc022cc5",
            },
            Pinned {
                name: "decoder_model.onnx",
                source: Source::Member("decoder_model.onnx"),
                sha256: "1c460e3cd755d322397ed112d03c311b7c151aa8d93b0cf5067178372d130079",
            },
            Pinned {
                name: "decoder_model.onnx.data",
                source: Source::Member("decoder_model.onnx.data"),
                sha256: "2a923112837da2d9f4e1ca0911ec82c424f35a8b39ba6de45b19bac4d15d6d04",
            },
            Pinned {
                name: "source_tokenizer.json",
                source: Source::Member("source_tokenizer.json"),
                sha256: "8879c0ae800b9d426047f101ff3ee4ca9c85f9f48775d86d9a7848168b006dd3",
            },
            Pinned {
                name: "target_tokenizer.json",
                source: Source::Member("target_tokenizer.json"),
                sha256: "8879c0ae800b9d426047f101ff3ee4ca9c85f9f48775d86d9a7848168b006dd3",
            },
            Pinned {
                name: "generation.json",
                source: Source::Member("generation.json"),
                sha256: "2e48f11ef0c843e16ce35fcd946abbe29d1aa5e78ab34118000adaa27b50cfc0",
            },
        ],
        note: None,
    },
    Component {
        id: "marian-ja-en",
        label: "Japanese to English translation (FuguMT)",
        dir: paths::MT_JA_EN_DIR,
        archive: Some(models_release!("marian-ja-en-onnx.tar.gz")),
        files: &[
            Pinned {
                name: "encoder_model.onnx",
                source: Source::Member("encoder_model.onnx"),
                sha256: "ff4b3620d8978332bd4a2baeaa301f5a5a1d973f1ddb47790e50f56cf3f71951",
            },
            Pinned {
                name: "encoder_model.onnx.data",
                source: Source::Member("encoder_model.onnx.data"),
                sha256: "1b4ba811cecfd4c84b4b72f81726107943c58b26802b62755fe2e6366b4cbfbe",
            },
            Pinned {
                name: "decoder_model.onnx",
                source: Source::Member("decoder_model.onnx"),
                sha256: "ea82385ed86820af4b87c5d6c6a123a6ced93b8c1df1d6b1b841a3bf27b34e5c",
            },
            Pinned {
                name: "decoder_model.onnx.data",
                source: Source::Member("decoder_model.onnx.data"),
                sha256: "29b4296bb8877a8c322b3b0e60c8e6bee4a79e5cc4dcdb3afcb157d443899e34",
            },
            Pinned {
                name: "source_tokenizer.json",
                source: Source::Member("source_tokenizer.json"),
                sha256: "1b05f2db75342e1152358a817fa30552a99a34aa62bb5cc33181eae385430231",
            },
            Pinned {
                name: "target_tokenizer.json",
                source: Source::Member("target_tokenizer.json"),
                sha256: "1b05f2db75342e1152358a817fa30552a99a34aa62bb5cc33181eae385430231",
            },
            Pinned {
                name: "generation.json",
                source: Source::Member("generation.json"),
                sha256: "2e48f11ef0c843e16ce35fcd946abbe29d1aa5e78ab34118000adaa27b50cfc0",
            },
        ],
        note: None,
    },
];

/// Download and verify every missing model. `force` fetches again even when
/// the files on disk already match.
pub fn run(force: bool) -> Result<()> {
    let models = paths::home().join("models");
    std::fs::create_dir_all(&models).with_context(|| format!("create {}", models.display()))?;
    eprintln!("Models go to {}", models.display());
    for component in COMPONENTS {
        ensure(&models, component, force)
            .with_context(|| format!("{} ({})", component.label, component.id))?;
    }
    let staging = models.join(".setup");
    let _ = std::fs::remove_dir_all(&staging);
    eprintln!("All models are ready. Start captions with: konnnyaku live --app \"Google Chrome\"");
    Ok(())
}

fn ensure(models: &Path, component: &Component, force: bool) -> Result<()> {
    let target = models.join(component.dir);
    let stamp = target.join(format!(".setup-{}", component.id));
    let fingerprint = fingerprint(component);
    if !force && installed(&target, &stamp, &fingerprint, component)? {
        eprintln!("ok    {}", component.label);
        return Ok(());
    }
    eprintln!("fetch {}", component.label);
    let staging = models.join(".setup").join(component.id);
    if staging.exists() {
        std::fs::remove_dir_all(&staging)
            .with_context(|| format!("remove {}", staging.display()))?;
    }
    std::fs::create_dir_all(&staging).with_context(|| format!("create {}", staging.display()))?;

    let mut fetched = Vec::with_capacity(component.files.len());
    if let Some(archive_url) = component.archive {
        let archive = staging.join("archive");
        download(archive_url, &archive)?;
        let unpacked = staging.join("unpacked");
        std::fs::create_dir_all(&unpacked)?;
        let members: Vec<&str> = component
            .files
            .iter()
            .map(|file| match file.source {
                Source::Member(member) => Ok(member),
                Source::Url(_) => Err(anyhow!("{} mixes an archive and a URL", component.id)),
            })
            .collect::<Result<_>>()?;
        unpack(&archive, &unpacked, &members)?;
        let _ = std::fs::remove_file(&archive);
        for (file, member) in component.files.iter().zip(members) {
            fetched.push((file, unpacked.join(member)));
        }
    } else {
        for file in component.files {
            let Source::Url(url) = file.source else {
                anyhow::bail!(
                    "{} lists an archive member without an archive",
                    component.id
                );
            };
            let path = staging.join(file.name);
            download(url, &path)?;
            fetched.push((file, path));
        }
    }

    for (file, path) in &fetched {
        verify(path, file.sha256).with_context(|| format!("verify {}", file.name))?;
    }
    std::fs::create_dir_all(&target).with_context(|| format!("create {}", target.display()))?;
    for (file, path) in fetched {
        let dest = target.join(file.name);
        std::fs::rename(&path, &dest)
            .with_context(|| format!("move {} into {}", file.name, target.display()))?;
    }
    if let Some((name, contents)) = component.note {
        std::fs::write(target.join(name), format!("{contents}\n"))
            .with_context(|| format!("write {name}"))?;
    }
    std::fs::write(&stamp, &fingerprint).with_context(|| format!("write {}", stamp.display()))?;
    let _ = std::fs::remove_dir_all(&staging);
    Ok(())
}

/// Files already in place. A missing stamp, as in a checkout that used the old
/// download script, is accepted once the hashes match.
fn installed(
    target: &Path,
    stamp: &Path,
    fingerprint: &str,
    component: &Component,
) -> Result<bool> {
    let all_present = component
        .files
        .iter()
        .all(|file| is_plain_file(&target.join(file.name)));
    if !all_present {
        return Ok(false);
    }
    if let Some((name, contents)) = component.note {
        let current = std::fs::read_to_string(target.join(name)).unwrap_or_default();
        if current.trim() != contents {
            return Ok(false);
        }
    }
    if std::fs::read_to_string(stamp).is_ok_and(|saved| saved == fingerprint) {
        return Ok(true);
    }
    for file in component.files {
        if verify(&target.join(file.name), file.sha256).is_err() {
            return Ok(false);
        }
    }
    std::fs::write(stamp, fingerprint).with_context(|| format!("write {}", stamp.display()))?;
    Ok(true)
}

fn fingerprint(component: &Component) -> String {
    let mut text = String::new();
    for file in component.files {
        text.push_str(file.name);
        text.push(' ');
        text.push_str(file.sha256);
        text.push('\n');
    }
    text
}

fn is_plain_file(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_file())
}

fn download(url: &str, dest: &Path) -> Result<()> {
    if !url.starts_with("https://") {
        anyhow::bail!("refusing a non-https model URL");
    }
    let partial = dest.with_extension("partial");
    let status = Command::new(tool("curl"))
        .args([
            "--fail",
            "--location",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--retry",
            "3",
            "--progress-bar",
            "--output",
        ])
        .arg(&partial)
        .arg(url)
        .status()
        .context("run curl")?;
    if !status.success() {
        let _ = std::fs::remove_file(&partial);
        anyhow::bail!("download failed: {url}");
    }
    std::fs::rename(&partial, dest).with_context(|| format!("rename {}", dest.display()))?;
    Ok(())
}

fn unpack(archive: &Path, into: &Path, members: &[&str]) -> Result<()> {
    for member in members {
        if member.starts_with('/') || member.split('/').any(|part| part == "..") {
            anyhow::bail!("unsafe archive member {member}");
        }
    }
    let status = Command::new(tool("tar"))
        .arg("-xf")
        .arg(archive)
        .arg("-C")
        .arg(into)
        .args(members)
        .status()
        .context("run tar")?;
    if !status.success() {
        anyhow::bail!("could not unpack {}", archive.display());
    }
    Ok(())
}

fn verify(path: &Path, expected: &str) -> Result<()> {
    if !is_plain_file(path) {
        anyhow::bail!("{} is missing or not a regular file", path.display());
    }
    let actual = sha256_file(path)?;
    if actual != expected {
        anyhow::bail!(
            "checksum mismatch for {}: expected {expected}, got {actual}",
            path.display()
        );
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

fn tool(name: &str) -> PathBuf {
    let system = Path::new("/usr/bin").join(name);
    if system.is_file() {
        system
    } else {
        PathBuf::from(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pin_is_a_sha256_and_an_https_source() {
        for component in COMPONENTS {
            if let Some(url) = component.archive {
                assert!(url.starts_with("https://"), "{}", component.id);
            }
            for file in component.files {
                assert_eq!(file.sha256.len(), 64, "{}", file.name);
                assert!(file.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
                assert!(!file.name.contains('/'), "{}", file.name);
                match file.source {
                    Source::Url(url) => {
                        assert!(component.archive.is_none(), "{}", component.id);
                        assert!(url.starts_with("https://"), "{url}");
                    }
                    Source::Member(member) => {
                        assert!(component.archive.is_some(), "{}", component.id);
                        assert!(!member.starts_with('/') && !member.contains(".."));
                    }
                }
            }
        }
    }

    #[test]
    fn a_tampered_file_is_rejected() {
        let dir = std::env::temp_dir().join(format!("konnnyaku-setup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.onnx");
        std::fs::write(&path, b"abc").unwrap();
        let good = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(verify(&path, good).is_ok());
        std::fs::write(&path, b"abd").unwrap();
        assert!(verify(&path, good).is_err());
        assert!(unpack(&path, &dir, &["../escape"]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
