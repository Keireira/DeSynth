use super::environment::require_environment;
use super::selection::{selected_lightning, selected_model};
use crate::catalog::{Catalog, MODELS_DIRECTORY, find_asset};
use anyhow::{Context, Result, ensure};
use desynth_inference::Inference;
use std::path::{Path, PathBuf};

const CACHE_FILES: [(&str, &[&str]); 2] = [
    (
        "Qwen/Qwen-Image",
        &[
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ],
    ),
    ("Qwen/Qwen-Image-2512", &["transformer/config.json"]),
];

pub(super) async fn prepare(root: &Path, directory: &Path) -> Result<()> {
    let python = require_environment(directory).await?;
    inference_command(root, python, "prepare").await?;
    println!("VAE and configs are cached; transformer/text encoder weights were not requested.");
    Ok(())
}

async fn inference_command(
    root: &Path,
    python: PathBuf,
    operation: &str,
) -> Result<serde_json::Value> {
    let worker = Inference::new(root.to_owned(), python);
    let result = tokio::select! {
        result = async {
            if operation == "prepare" { worker.prepare().await } else { worker.doctor().await }
        } => result.map_err(anyhow::Error::from),
        signal = tokio::signal::ctrl_c() => {
            signal.context("Cannot listen for Ctrl+C")?;
            Err(anyhow::anyhow!("Setup cancelled"))
        }
    };
    let shutdown = worker.shutdown().await;
    let result = result?;
    shutdown?;
    Ok(result)
}

fn hub_cache() -> Result<PathBuf> {
    if let Some(directory) =
        std::env::var_os("HF_HUB_CACHE").or_else(|| std::env::var_os("HUGGINGFACE_HUB_CACHE"))
    {
        return Ok(directory.into());
    }
    if let Some(directory) = std::env::var_os("HF_HOME") {
        return Ok(PathBuf::from(directory).join("hub"));
    }
    if let Some(directory) = std::env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(directory).join("huggingface/hub"));
    }
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .context("Cannot locate the Hugging Face cache directory")?;
    Ok(PathBuf::from(home).join(".cache/huggingface/hub"))
}

fn missing_cache_files(directory: &Path) -> Vec<String> {
    let mut missing = Vec::new();
    for (repository, files) in CACHE_FILES {
        let model = directory.join(format!("models--{}", repository.replace('/', "--")));
        let revision = std::fs::read_to_string(model.join("refs/main")).unwrap_or_default();
        let snapshot = model.join("snapshots").join(revision.trim());
        for filename in files {
            if !crate::catalog::nonempty_file(&snapshot.join(filename)) {
                missing.push(format!("{repository}/{filename}"));
            }
        }
    }
    missing
}

pub(super) async fn doctor(
    root: &Path,
    directory: &Path,
    backend: &str,
    model: &str,
    lightning: &str,
) -> Result<()> {
    let python = require_environment(directory).await?;
    println!(
        "ML Python: {}\nDependency backend: {backend}",
        python.display()
    );
    let probe = inference_command(root, python, "doctor").await?;
    let mut failures = Vec::new();
    for key in ["missing", "errors"] {
        if let Some(errors) = probe[key].as_array() {
            failures.extend(
                errors
                    .iter()
                    .filter_map(|error| error.as_str().map(str::to_owned)),
            );
        }
    }
    println!(
        "PyTorch {}; CUDA runtime: {}",
        probe["torch_version"].as_str().unwrap_or("unavailable"),
        probe["cuda_runtime"].as_str().unwrap_or("none")
    );
    println!("Device: {}", probe["device"].as_str().unwrap_or("unknown"));
    let cuda = probe["runtime"]["devices"]
        .as_array()
        .is_some_and(|devices| devices.iter().any(|device| device["type"] == "cuda"));
    if backend == "cuda" && !cuda {
        failures.push("CUDA is unavailable. Check the NVIDIA driver, or use BACKEND=cpu.".into());
    }
    if backend == "cpu" && !probe["cuda_runtime"].is_null() {
        failures.push("CUDA wheel installed for CPU backend. Run make install BACKEND=cpu.".into());
    }
    let catalog = Catalog::scan(root)?;
    if let Some(path) = selected_model(root, &catalog, model) {
        println!("GGUF: {}", path.display());
    } else {
        failures.push(format!(
            "Missing GGUF model ({model}). Run make models MODEL={model}."
        ));
    }
    if let Some(path) = selected_lightning(root, &catalog, lightning) {
        println!("Lightning: {}", path.display());
    } else {
        failures.push(format!(
            "Missing Lightning adapter ({lightning}). Run make models LIGHTNING={lightning}."
        ));
    }
    if find_asset(root, "embeds_cache.pt").is_none() {
        failures.push(format!(
            "Missing {MODELS_DIRECTORY}/embeds_cache.pt. Restore the included file."
        ));
    }
    for filename in missing_cache_files(&hub_cache()?) {
        failures.push(format!("Missing cached {filename}. Run make prepare."));
    }
    for failure in &failures {
        eprintln!("FAIL: {failure}");
    }
    ensure!(
        failures.is_empty(),
        "Readiness check failed. Run make setup to prepare the ML environment."
    );
    println!("Ready: dependencies, device, model files, and VAE/config cache are available.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_file(directory: &Path, repository: &str, revision: &str, filename: &str) {
        let model = directory.join(format!("models--{}", repository.replace('/', "--")));
        std::fs::create_dir_all(model.join("refs")).unwrap();
        std::fs::write(model.join("refs/main"), revision).unwrap();
        let path = model.join("snapshots").join(revision).join(filename);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"cached").unwrap();
    }

    fn cache_repositories(directory: &Path) {
        for (repository, files) in CACHE_FILES {
            let revision = if repository.ends_with("2512") {
                "transformer-commit"
            } else {
                "pipeline-commit"
            };
            for filename in files {
                cache_file(directory, repository, revision, filename);
            }
        }
    }

    #[test]
    fn cache_readiness_follows_each_repositories_main_revision() {
        let directory = tempfile::tempdir().unwrap();
        assert_eq!(missing_cache_files(directory.path()).len(), 6);
        cache_repositories(directory.path());
        assert!(missing_cache_files(directory.path()).is_empty());

        let transformer = directory.path().join("models--Qwen--Qwen-Image-2512");
        std::fs::write(transformer.join("refs/main"), "next-transformer-commit").unwrap();
        assert_eq!(
            missing_cache_files(directory.path()),
            ["Qwen/Qwen-Image-2512/transformer/config.json"]
        );
        std::fs::write(transformer.join("refs/main"), "transformer-commit\n").unwrap();
        let pipeline = directory.path().join("models--Qwen--Qwen-Image");
        std::fs::write(pipeline.join("refs/main"), "next-pipeline-commit").unwrap();
        let missing = missing_cache_files(directory.path());
        assert_eq!(missing.len(), 5);
        assert!(
            missing
                .iter()
                .all(|file| file.starts_with("Qwen/Qwen-Image/"))
        );
    }

    #[test]
    fn old_transformer_config_cannot_replace_the_2512_config() {
        let directory = tempfile::tempdir().unwrap();
        for filename in CACHE_FILES[0].1 {
            cache_file(directory.path(), "Qwen/Qwen-Image", "old-commit", filename);
        }
        cache_file(
            directory.path(),
            "Qwen/Qwen-Image",
            "old-commit",
            "transformer/config.json",
        );
        assert_eq!(
            missing_cache_files(directory.path()),
            ["Qwen/Qwen-Image-2512/transformer/config.json"]
        );
    }

    #[test]
    fn empty_files_and_unreferenced_snapshots_are_not_ready() {
        let directory = tempfile::tempdir().unwrap();
        cache_repositories(directory.path());
        let transformer = directory.path().join("models--Qwen--Qwen-Image-2512");
        let config = transformer.join("snapshots/transformer-commit/transformer/config.json");
        std::fs::write(&config, b"").unwrap();
        assert_eq!(
            missing_cache_files(directory.path()),
            ["Qwen/Qwen-Image-2512/transformer/config.json"]
        );
        std::fs::write(config, b"cached").unwrap();
        std::fs::remove_file(transformer.join("refs/main")).unwrap();
        assert_eq!(
            missing_cache_files(directory.path()),
            ["Qwen/Qwen-Image-2512/transformer/config.json"]
        );
    }
}
