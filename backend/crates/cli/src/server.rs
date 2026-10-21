use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, Result, ensure};
use tokio::net::TcpListener;

use desynth_core::studio::Studio;

pub(super) async fn serve(
    root: PathBuf,
    python: PathBuf,
    port: u16,
    no_browser: bool,
) -> Result<()> {
    ensure!(port > 0, "Choose a port between 1 and 65535");
    ensure!(
        root.join("web/dist/index.html").is_file(),
        "The frontend has not been built. Run make web-build first."
    );
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
        .await
        .with_context(|| {
            format!("Cannot listen on 127.0.0.1:{port}; the port may already be in use")
        })?;
    let studio = Studio::new(root, python)?;
    let result = serve_session(listener, studio.clone(), port, no_browser).await;
    studio.close().await;
    result
}

async fn serve_session(
    listener: TcpListener,
    studio: Arc<Studio>,
    port: u16,
    no_browser: bool,
) -> Result<()> {
    studio.refresh()?;
    let url = format!("http://127.0.0.1:{port}/");
    println!("DeSynth: {url}");
    println!("Press Ctrl+C to stop.");
    if !no_browser && let Err(error) = webbrowser::open(&url) {
        tracing::warn!(%error, "Could not open the browser");
    }
    let shutdown = studio.clone();
    axum::serve(listener, desynth_server::router(studio, port))
        .with_graceful_shutdown(async move {
            if let Err(error) = tokio::signal::ctrl_c().await {
                tracing::error!(%error, "Could not listen for Ctrl+C");
            }
            shutdown.close().await;
        })
        .await?;
    Ok(())
}
