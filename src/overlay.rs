use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender};
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

pub fn run_meeting_hud(
    running: Arc<AtomicBool>,
    notes: Receiver<String>,
    summarize: Sender<()>,
) -> Result<()> {
    let mut event_loop = EventLoopBuilder::<UiEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();
    let feed_proxy = proxy.clone();
    std::thread::Builder::new()
        .name("konnnyaku-overlay-feed".into())
        .spawn(move || {
            while let Ok(note) = notes.recv() {
                if feed_proxy.send_event(UiEvent::Note(note)).is_err() {
                    break;
                }
            }
        })
        .context("spawn overlay feed")?;

    let window = WindowBuilder::new()
        .with_title("会議メモ")
        .with_inner_size(LogicalSize::new(440.0, 520.0))
        .with_min_inner_size(LogicalSize::new(360.0, 280.0))
        .with_position(LogicalPosition::new(80.0, 80.0))
        .with_decorations(false)
        .with_transparent(true)
        .with_always_on_top(true)
        .with_resizable(true)
        .build(&event_loop)
        .context("create meeting window")?;

    apply_vibrancy(
        &window,
        NSVisualEffectMaterial::HudWindow,
        Some(NSVisualEffectState::Active),
        Some(20.0),
    )
    .context("apply meeting window vibrancy")?;

    let close_flag = Arc::clone(&running);
    let drag_proxy = proxy;
    let webview = WebViewBuilder::new()
        .with_transparent(true)
        .with_html(include_str!("overlay.html"))
        .with_accept_first_mouse(true)
        .with_ipc_handler(move |request| match request.body().as_str() {
            "close" => close_flag.store(false, Ordering::SeqCst),
            "drag" => {
                let _ = drag_proxy.send_event(UiEvent::Drag);
            }
            "summarize" => {
                let _ = summarize.try_send(());
            }
            _ => {}
        })
        .build(&window)
        .context("create meeting webview")?;

    let loop_running = Arc::clone(&running);
    event_loop.run_return(move |event, _, control_flow| {
        *control_flow = ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(200));
        if !loop_running.load(Ordering::SeqCst) {
            *control_flow = ControlFlow::Exit;
            return;
        }
        match event {
            Event::UserEvent(UiEvent::Note(note)) => {
                let script = format!("window.konnnyakuUpdate && window.konnnyakuUpdate({note})");
                if let Err(err) = webview.evaluate_script(&script) {
                    tracing::debug!(%err, "meeting window script failed");
                }
            }
            Event::UserEvent(UiEvent::Drag) => {
                if let Err(err) = window.drag_window() {
                    tracing::debug!(?err, "drag window failed");
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
