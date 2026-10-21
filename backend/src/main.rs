#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "desynth=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    if let Err(error) = desynth::cli::run().await {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}
