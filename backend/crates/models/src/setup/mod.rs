//! Environment commands and verified model downloads. Python is only invoked
//! to create/check the ML environment and run the inference worker.

mod backend;
mod download;
mod environment;
mod readiness;
mod repository;
mod selection;

pub use environment::environment_python;

use crate::catalog::Catalog;
use anyhow::{Context, Result, bail, ensure};
use download::download_models;
use environment::{create_environment, install};
use readiness::{doctor, prepare};
use selection::{choose_download_model, selected_model};
use std::path::Path;

pub async fn run(
    root: &Path,
    command: &str,
    venv: &Path,
    backend: &str,
    model: &str,
    lightning: &str,
) -> Result<()> {
    ensure!(
        ["auto", "cuda", "cpu"].contains(&backend),
        "Unknown dependency backend: {backend}"
    );
    ensure!(
        ["auto", "q4", "q8"].contains(&model),
        "Unknown model: {model}"
    );
    ensure!(
        ["auto", "4", "8"].contains(&lightning),
        "Unknown Lightning profile: {lightning}"
    );
    let backend = backend::resolve_backend(backend).await;
    ensure!(
        !(backend == "cuda" && cfg!(target_os = "macos")),
        "CUDA is unavailable on macOS. Use BACKEND=cpu."
    );
    let root = root
        .canonicalize()
        .context("Cannot find the project root")?;
    let venv = if venv.is_absolute() {
        venv.to_path_buf()
    } else {
        root.join(venv)
    };
    // Ask before installing a Ctrl+C handler or starting child processes. The
    // terminal's ordinary interrupt remains available while stdin is blocking.
    let model = if matches!(command, "models" | "setup")
        && selected_model(&root, &Catalog::scan(&root)?, model).is_none()
    {
        choose_download_model(model)?
    } else {
        model.to_owned()
    };
    match command {
        "venv" => create_environment(&root, &venv).await,
        "install" => install(&root, &venv, backend).await,
        "models" => download_models(&root, &model, lightning).await,
        "prepare" => prepare(&root, &venv).await,
        "doctor" => doctor(&root, &venv, backend, &model, lightning).await,
        "setup" => {
            install(&root, &venv, backend).await?;
            download_models(&root, &model, lightning).await?;
            prepare(&root, &venv).await?;
            doctor(&root, &venv, backend, &model, lightning).await
        }
        _ => bail!("Unknown setup command: {command}"),
    }
}
