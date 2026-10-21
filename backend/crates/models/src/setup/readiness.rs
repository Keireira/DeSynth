use super::environment::require_environment;
use super::selection::{selected_lightning, selected_model};
use crate::catalog::{Catalog, MODELS_DIRECTORY, find_asset};
use anyhow::{Context, Result, ensure};
use desynth_inference::Inference;
use std::path::{Path, PathBuf};

const CACHE_FILES: [&str; 6] = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "transformer/config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
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

fn missing_cache_files(directory: &Path) -> Vec<&'static str> {
    let model = directory.join("models--Qwen--Qwen-Image");
    let revision = std::fs::read_to_string(model.join("refs/main")).unwrap_or_default();
    let snapshot = model.join("snapshots").join(revision.trim());
    CACHE_FILES
        .into_iter()
        .filter(|filename| !crate::catalog::nonempty_file(&snapshot.join(filename)))
        .collect()
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

    #[test]
    fn cache_readiness_follows_the_main_revision() {
        let directory = tempfile::tempdir().unwrap();
        assert_eq!(missing_cache_files(directory.path()).len(), 6);
        let model = directory.path().join("models--Qwen--Qwen-Image");
        std::fs::create_dir_all(model.join("refs")).unwrap();
        std::fs::write(model.join("refs/main"), "commit").unwrap();
        for name in CACHE_FILES {
            let path = model.join("snapshots/commit").join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"cached").unwrap();
        }
        assert!(missing_cache_files(directory.path()).is_empty());
    }
}
