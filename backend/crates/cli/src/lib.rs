mod args;
mod files;
mod image_commands;
mod processing;
mod server;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;

use args::{Cli, Command, CompareArgs, DiffArgs, RunArgs, SetupArgs};

pub async fn run() -> Result<()> {
    execute(Cli::parse()).await
}

async fn execute(cli: Cli) -> Result<()> {
    let default_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .context("Project root is missing")?
        .to_owned();
    let root = cli
        .root
        .unwrap_or(default_root)
        .canonicalize()
        .context("Project directory does not exist")?;
    let python = cli
        .python
        .unwrap_or_else(|| desynth_models::setup::environment_python(&root.join(".venv")));
    match cli.command {
        Command::Gui(args) => server::serve(root, python, args.port, args.no_browser).await,
        Command::Run(args) => processing::run(root, python, args).await,
        Command::Compare(args) => image_commands::compare(&root, args),
        Command::Diff(args) => image_commands::diff(&root, args),
        Command::Setup(args) => setup(&root, "setup", args).await,
        Command::Venv(args) => setup(&root, "venv", args).await,
        Command::Install(args) => setup(&root, "install", args).await,
        Command::Models(args) => setup(&root, "models", args).await,
        Command::Prepare(args) => setup(&root, "prepare", args).await,
        Command::Doctor(args) => setup(&root, "doctor", args).await,
    }
}

async fn setup(root: &Path, command: &str, args: SetupArgs) -> Result<()> {
    desynth_models::setup::run(
        root,
        command,
        &args.venv,
        &args.backend,
        &args.model,
        &args.lightning,
    )
    .await
}

pub(crate) fn rooted(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        root.join(path)
    }
}

pub(crate) fn write_image(
    path: &Path,
    image: &image::RgbaImage,
    metadata: Option<&serde_json::Value>,
) -> Result<()> {
    files::write(path, &desynth_images::encode_png(image, metadata)?)?;
    println!("  -> {}", path.display());
    Ok(())
}
