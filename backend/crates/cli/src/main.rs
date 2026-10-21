#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "desynth_cli=info,desynth_core=info,desynth_server=info,desynth_models=info,desynth_inference=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    if let Err(error) = desynth_cli::run().await {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}
