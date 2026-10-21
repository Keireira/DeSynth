use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "desynth",
    version,
    about = "Local image processing and comparison"
)]
pub(crate) struct Cli {
    #[arg(
        long,
        global = true,
        help = "Project directory (defaults to the source checkout)"
    )]
    pub(crate) root: Option<PathBuf>,
    #[arg(
        long,
        global = true,
        help = "Python executable for the inference worker"
    )]
    pub(crate) python: Option<PathBuf>,
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand)]
pub(crate) enum Command {
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
pub(crate) struct GuiArgs {
    #[arg(long, default_value_t = 8758)]
    pub(crate) port: u16,
    #[arg(long)]
    pub(crate) no_browser: bool,
}

#[derive(Args)]
pub(crate) struct RunArgs {
    #[arg(default_value = "original.png")]
    pub(crate) input: PathBuf,
    #[arg(long, num_args = 1.., default_value = "0.25")]
    pub(crate) denoise: Vec<f64>,
    #[arg(long)]
    pub(crate) steps: Option<u32>,
    #[arg(long, default_value_t = 1)]
    pub(crate) passes: u32,
    #[arg(long)]
    pub(crate) seed: Option<u64>,
    #[arg(long, default_value_t = 1.95)]
    pub(crate) restore_sigma: f64,
    #[arg(long, value_parser = ["gaussian", "edge"], default_value = "gaussian")]
    pub(crate) restore_mode: String,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) unsharp: f64,
    #[arg(long)]
    pub(crate) no_restore: bool,
    #[arg(long)]
    pub(crate) keep_intermediate: bool,
    #[arg(long)]
    pub(crate) transformer: Option<PathBuf>,
    #[arg(long)]
    pub(crate) lightning: Option<PathBuf>,
    #[arg(long, default_value_t = 0.8)]
    pub(crate) lightning_scale: f64,
    #[arg(long, default_value = "auto")]
    pub(crate) device: String,
    #[arg(long, default_value = "auto")]
    pub(crate) offload: String,
    #[arg(long, default_value = "auto")]
    pub(crate) precision: String,
    #[arg(long)]
    pub(crate) preview: bool,
    #[arg(long, default_value = "out")]
    pub(crate) out: PathBuf,
}

#[derive(Args)]
pub(crate) struct CompareArgs {
    pub(crate) original: PathBuf,
    pub(crate) result: PathBuf,
    #[arg(long)]
    pub(crate) visual: bool,
    #[arg(long)]
    pub(crate) visual_out: Option<PathBuf>,
}

#[derive(Args)]
pub(crate) struct DiffArgs {
    pub(crate) original: PathBuf,
    pub(crate) result: PathBuf,
    #[arg(long, default_value = "out/diff.png")]
    pub(crate) out: PathBuf,
    #[arg(long, default_value_t = 8.0)]
    pub(crate) gain: f64,
    #[arg(long)]
    pub(crate) raw_out: Option<PathBuf>,
    #[arg(long)]
    pub(crate) preview_out: Option<PathBuf>,
}

#[derive(Args)]
pub(crate) struct SetupArgs {
    #[arg(long, default_value = ".venv")]
    pub(crate) venv: PathBuf,
    #[arg(long, value_parser = ["auto", "cuda", "cpu"], default_value = "auto")]
    pub(crate) backend: String,
    #[arg(long, value_parser = ["auto", "q4", "q8"], default_value = "auto")]
    pub(crate) model: String,
    #[arg(long, value_parser = ["auto", "4", "8"], default_value = "auto")]
    pub(crate) lightning: String,
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
