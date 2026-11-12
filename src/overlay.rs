use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossbeam_channel::Receiver;
use tao::dpi::{LogicalPosition, LogicalSize};
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tao::platform::run_return::EventLoopExtRunReturn;
use tao::window::WindowBuilder;
use window_vibrancy::{NSVisualEffectMaterial, NSVisualEffectState, apply_vibrancy};
use wry::WebViewBuilder;

enum UiEvent {
    Note(String),
    Drag,
}

pub fn run_caption_hud(
    running: Arc<AtomicBool>,
    lines: Receiver<String>,
    langs: Arc<crate::LangPair>,
    hear: Arc<crate::HearFlags>,
    saved: Arc<crate::caption_log::CaptionLog>,
) -> Result<()> {
    let mut event_loop = EventLoopBuilder::<UiEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();
    let feed_proxy = proxy.clone();
    std::thread::Builder::new()
        .name("konnnyaku-caption-feed".into())
        .spawn(move || {
            while let Ok(line) = lines.recv() {
                if feed_proxy.send_event(UiEvent::Note(line)).is_err() {
                    break;
                }
            }
        })
        .context("spawn caption feed")?;

    let window = WindowBuilder::new()
        .with_title("文字起こし")
        .with_inner_size(LogicalSize::new(740.0, 420.0))
        .with_min_inner_size(LogicalSize::new(520.0, 180.0))
        .with_position(LogicalPosition::new(540.0, 80.0))
        .with_decorations(false)
        .with_transparent(true)
        .with_always_on_top(true)
        .with_resizable(true)
        .build(&event_loop)
        .context("create caption window")?;

    apply_vibrancy(
        &window,
        NSVisualEffectMaterial::HudWindow,
        Some(NSVisualEffectState::Active),
        Some(20.0),
    )
    .context("apply caption window vibrancy")?;

    let close_flag = Arc::clone(&running);
    let drag_proxy = proxy;
    let ipc_langs = Arc::clone(&langs);
    let ipc_hear = Arc::clone(&hear);
    let ipc_saved = Arc::clone(&saved);
    let init = format!(
        "window.konnnyakuPrimary && window.konnnyakuPrimary('{}'); window.konnnyakuHear && window.konnnyakuHear({}, {})",
        langs.primary().label(),
        hear.input(),
        hear.output()
    );
    let webview = WebViewBuilder::new()
        .with_transparent(true)
        .with_html(include_str!("caption.html"))
        .with_accept_first_mouse(true)
        .with_ipc_handler(move |request| {
            let body = request.body();
            match body.as_str() {
                "close" => close_flag.store(false, Ordering::SeqCst),
                "clear" => ipc_saved.clear(),
                "drag" => {
                    let _ = drag_proxy.send_event(UiEvent::Drag);
                }
                "primary:en" => {
                    ipc_langs.set_primary(crate::CaptionLang::En);
                    ipc_saved.set_primary("en");
                }
                "primary:ja" => {
                    ipc_langs.set_primary(crate::CaptionLang::Ja);
                    ipc_saved.set_primary("ja");
                }
                "hear:in:0" => ipc_hear.set_input(false),
                "hear:in:1" => ipc_hear.set_input(true),
                "hear:out:0" => ipc_hear.set_output(false),
                "hear:out:1" => ipc_hear.set_output(true),
                other => {
                    if let Some(text) = clipboard_text(other) {
                        copy_to_pasteboard(text);
                    }
                }
            }
        })
        .build(&window)
        .context("create caption webview")?;
    if let Err(err) = webview.evaluate_script(&init) {
        tracing::debug!(%err, "caption language init failed");
    }

    let loop_running = Arc::clone(&running);
    event_loop.run_return(move |event, _, control_flow| {
        *control_flow = ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(200));
        if !loop_running.load(Ordering::SeqCst) {
            *control_flow = ControlFlow::Exit;
            return;
        }
        match event {
            Event::UserEvent(UiEvent::Note(line)) => {
                let script = format!("window.konnnyakuUpdate && window.konnnyakuUpdate({line})");
                if let Err(err) = webview.evaluate_script(&script) {
                    tracing::debug!(%err, "caption window script failed");
                }
            }
            Event::UserEvent(UiEvent::Drag) => {
                if let Err(err) = window.drag_window() {
                    tracing::debug!(?err, "drag caption window failed");
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                loop_running.store(false, Ordering::SeqCst);
                *control_flow = ControlFlow::Exit;
            }
            _ => {}
        }
    });
    running.store(false, Ordering::SeqCst);
    Ok(())
}

/// Caption text from a `copy:` IPC message. Anything else, including the
/// language and window commands, is ignored. The page is local, and the text
/// is only written to the pasteboard.
fn clipboard_text(body: &str) -> Option<&str> {
    let text = body.strip_prefix("copy:")?;
    if text.is_empty() || text.len() > 1_048_576 {
        None
    } else {
        Some(text)
    }
}

fn copy_to_pasteboard(text: &str) {
    let mut child = match Command::new("pbcopy")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            tracing::debug!(%err, "clipboard copy failed");
            return;
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if stdin.write_all(text.as_bytes()).is_err() {
            tracing::debug!("clipboard copy failed");
            let _ = child.wait();
            return;
        }
    }
    if child.wait().is_err() {
        tracing::debug!("clipboard copy failed");
    }
}

#[cfg(test)]
mod tests {
    use super::clipboard_text;

    #[test]
    fn copy_ipc_accepts_caption_text_only() {
        assert_eq!(clipboard_text("copy:hello"), Some("hello"));
        assert_eq!(clipboard_text("copy:日本語\n次"), Some("日本語\n次"));
        assert_eq!(clipboard_text("close"), None);
        assert_eq!(clipboard_text("pair:en:ja"), None);
        assert_eq!(clipboard_text("copy:"), None);
        let huge = format!("copy:{}", "a".repeat(1_048_577));
        assert_eq!(clipboard_text(&huge), None);
    }
}
