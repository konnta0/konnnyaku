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

pub fn run_caption_hud(running: Arc<AtomicBool>, lines: Receiver<String>) -> Result<()> {
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
    let webview = WebViewBuilder::new()
        .with_transparent(true)
        .with_html(include_str!("caption.html"))
        .with_accept_first_mouse(true)
        .with_ipc_handler(move |request| match request.body().as_str() {
            "close" => close_flag.store(false, Ordering::SeqCst),
            "drag" => {
                let _ = drag_proxy.send_event(UiEvent::Drag);
            }
            _ => {}
        })
        .build(&window)
        .context("create caption webview")?;

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
