mod image_commands;
mod processing;
mod server;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "desynth",
    version,
    about = "Local image processing and comparison"
)]
pub struct Cli {
    #[arg(
        long,
        global = true,
        help = "Project directory (defaults to the source checkout)"
    )]
    root: Option<PathBuf>,
    #[arg(
        long,
        global = true,
        help = "Python executable for the inference worker"
    )]
    python: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Open the browser interface.
    Gui(GuiArgs),
    /// Process an image, saving every run under out/runs/.
    Run(RunArgs),
    /// Measure image similarity, optionally saving a labelled comparison.
    Compare(CompareArgs),
    /// Save a visible RGBA difference image.
    Diff(DiffArgs),
    /// Prepare the Python inference environment and selected weights.
    Setup(SetupArgs),
    /// Create the inference virtual environment.
    Venv(SetupArgs),
    /// Install inference dependencies.
    Install(SetupArgs),
    /// Select and download model weights.
    Models(SetupArgs),
    /// Cache model configuration and prompt embeddings.
    Prepare(SetupArgs),
    /// Check inference dependencies and hardware.
    Doctor(SetupArgs),
}

#[derive(Args)]
struct GuiArgs {
    #[arg(long, default_value_t = 8758)]
    port: u16,
    #[arg(long)]
    no_browser: bool,
}

#[derive(Args)]
pub(super) struct RunArgs {
    #[arg(default_value = "original.png")]
    input: PathBuf,
    #[arg(long, num_args = 1.., default_value = "0.25")]
    denoise: Vec<f64>,
    #[arg(long)]
    steps: Option<u32>,
    #[arg(long, default_value_t = 1)]
    passes: u32,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = 1.95)]
    restore_sigma: f64,
    #[arg(long, value_parser = ["gaussian", "edge"], default_value = "gaussian")]
    restore_mode: String,
    #[arg(long, default_value_t = 0.0)]
    unsharp: f64,
    #[arg(long)]
    no_restore: bool,
    #[arg(long)]
    keep_intermediate: bool,
    #[arg(long)]
    transformer: Option<PathBuf>,
    #[arg(long)]
    lightning: Option<PathBuf>,
    #[arg(long, default_value_t = 0.8)]
    lightning_scale: f64,
    #[arg(long, default_value = "auto")]
    device: String,
    #[arg(long, default_value = "auto")]
    offload: String,
    #[arg(long, default_value = "auto")]
    precision: String,
    #[arg(long)]
    preview: bool,
    #[arg(long, default_value = "out")]
    out: PathBuf,
}

#[derive(Args)]
pub(super) struct CompareArgs {
    original: PathBuf,
    result: PathBuf,
    #[arg(long)]
    visual: bool,
    #[arg(long)]
    visual_out: Option<PathBuf>,
}

#[derive(Args)]
pub(super) struct DiffArgs {
    original: PathBuf,
    result: PathBuf,
    #[arg(long, default_value = "out/diff.png")]
    out: PathBuf,
    #[arg(long, default_value_t = 8.0)]
    gain: f64,
    #[arg(long)]
    raw_out: Option<PathBuf>,
    #[arg(long)]
    preview_out: Option<PathBuf>,
}

#[derive(Args)]
struct SetupArgs {
    #[arg(long, default_value = ".venv")]
    venv: PathBuf,
    #[arg(long, value_parser = ["auto", "cuda", "cpu"], default_value = "auto")]
    backend: String,
    #[arg(long, value_parser = ["auto", "q4", "q8"], default_value = "auto")]
    model: String,
    #[arg(long, value_parser = ["auto", "4", "8"], default_value = "auto")]
    lightning: String,
}

pub async fn run() -> Result<()> {
    execute(Cli::parse()).await
}

async fn execute(cli: Cli) -> Result<()> {
    let default_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("Project root is missing")?
        .to_owned();
    let root = cli
        .root
        .unwrap_or(default_root)
        .canonicalize()
        .context("Project directory does not exist")?;
    let python = cli
        .python
        .unwrap_or_else(|| default_python(&root.join(".venv")));
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
    crate::setup::run(
        root,
        command,
        &args.venv,
        &args.backend,
        &args.model,
        &args.lightning,
    )
    .await
}

fn default_python(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts/python.exe")
    } else {
        venv.join("bin/python")
    }
}

pub(super) fn rooted(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        root.join(path)
    }
}

pub(super) fn write_image(
    path: &Path,
    image: &image::RgbaImage,
    metadata: Option<&serde_json::Value>,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, crate::images::encode_png(image, metadata)?)?;
    println!("  -> {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_accepts_a_denoise_sweep_and_existing_flags() {
        let parsed = Cli::try_parse_from([
            "desynth",
            "run",
            "original.png",
            "--denoise",
            "0.125",
            "0.25",
            "--steps",
            "32",
            "--lightning",
            "adapter.safetensors",
            "--restore-mode",
            "edge",
            "--keep-intermediate",
        ]);
        let cli = parsed.unwrap();
        let Command::Run(args) = cli.command else {
            panic!("run command expected")
        };
        assert_eq!(args.denoise, [0.125, 0.25]);
        assert_eq!(args.steps, Some(32));
        assert_eq!(args.restore_mode, "edge");
        assert!(args.keep_intermediate);
    }
}
