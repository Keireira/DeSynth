//! Environment orchestration and model downloads stay in Rust. Python is used
//! only for creating its ML environment, pip, and checking/loading that runtime.

use crate::catalog::{Catalog, MODELS_DIRECTORY, find_asset, models_directory};
use crate::inference::Inference;
use anyhow::{Context, Result, bail, ensure};
use reqwest::{Client, StatusCode, Url};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

const GGUF_REPOSITORY: &str = "Frederic75/Qwen-Image-2512-GGUF";
const LIGHTNING_REPOSITORY: &str = "lightx2v/Qwen-Image-2512-Lightning";
const CACHE_FILES: [&str; 6] = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "transformer/config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
];

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
    let backend = if backend == "auto" {
        if cfg!(target_os = "macos") {
            "cpu"
        } else {
            "cuda"
        }
    } else {
        backend
    };
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

pub fn environment_python(directory: &Path) -> PathBuf {
    directory.join(if cfg!(windows) {
        "Scripts/python.exe"
    } else {
        "bin/python"
    })
}

async fn require_environment(directory: &Path) -> Result<PathBuf> {
    let python = environment_python(directory);
    ensure!(
        directory.join("pyvenv.cfg").is_file() && python.is_file(),
        "No ML environment at {}. Run make install first.",
        directory.display()
    );
    let output = Command::new(&python)
        .arg("--version")
        .kill_on_drop(true)
        .output()
        .await
        .context("Cannot run the ML environment's Python")?;
    let version = String::from_utf8_lossy(&output.stdout);
    ensure!(
        output.status.success() && supported_python_version(&version),
        "The ML dependencies require Python 3.11–3.13; found {}.",
        version.trim()
    );
    Ok(python)
}

async fn bootstrap_python() -> Result<(PathBuf, Vec<String>)> {
    let mut candidates = Vec::new();
    if let Some(python) = std::env::var_os("DESYNTH_BOOTSTRAP_PYTHON") {
        candidates.push((PathBuf::from(python), Vec::new()));
    } else {
        if cfg!(windows) {
            for version in ["3.11", "3.12", "3.13"] {
                candidates.push((PathBuf::from("py"), vec![format!("-{version}")]));
            }
        }
        for program in [
            "python3.11",
            "python3.12",
            "python3.13",
            "python3",
            "python",
        ] {
            candidates.push((PathBuf::from(program), Vec::new()));
        }
    }
    for (program, arguments) in candidates {
        let output = Command::new(&program)
            .args(&arguments)
            .arg("--version")
            .kill_on_drop(true)
            .output()
            .await;
        if let Ok(output) = output {
            let version = String::from_utf8_lossy(&output.stdout);
            if output.status.success() && supported_python_version(&version) {
                return Ok((program, arguments));
            }
        }
    }
    bail!("Install Python 3.11–3.13, or set DESYNTH_BOOTSTRAP_PYTHON to its executable path.")
}

fn supported_python_version(version: &str) -> bool {
    let Some(version) = version.trim().strip_prefix("Python ") else {
        return false;
    };
    let mut parts = version.split('.');
    parts.next() == Some("3") && matches!(parts.next(), Some("11" | "12" | "13"))
}

async fn checked_command(root: &Path, program: &Path, arguments: &[String]) -> Result<()> {
    println!("> {} {}", program.display(), arguments.join(" "));
    let mut child = Command::new(program)
        .args(arguments)
        .current_dir(root)
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("Cannot start {}", program.display()))?;
    let status = tokio::select! {
        status = child.wait() => status?,
        signal = tokio::signal::ctrl_c() => {
            signal.context("Cannot listen for Ctrl+C")?;
            let _ = child.kill().await;
            let _ = child.wait().await;
            bail!("Setup cancelled");
        }
    };
    ensure!(
        status.success(),
        "{} failed with {status}",
        program.display()
    );
    Ok(())
}

async fn create_environment(root: &Path, directory: &Path) -> Result<()> {
    if directory.exists() {
        require_environment(directory).await?;
        println!("Using ML environment: {}", directory.display());
        return Ok(());
    }
    let (program, mut arguments) = bootstrap_python().await?;
    arguments.extend([
        "-m".into(),
        "venv".into(),
        directory.to_string_lossy().into_owned(),
    ]);
    checked_command(root, &program, &arguments).await?;
    require_environment(directory).await?;
    Ok(())
}

async fn install(root: &Path, directory: &Path, backend: &str) -> Result<()> {
    create_environment(root, directory).await?;
    let python = require_environment(directory).await?;
    let requirements = root.join(if backend == "cuda" {
        "inference/requirements/cuda.txt"
    } else {
        "inference/requirements/cpu.txt"
    });
    for arguments in [
        vec![
            "-m".into(),
            "pip".into(),
            "install".into(),
            "--upgrade".into(),
            "pip".into(),
        ],
        vec![
            "-m".into(),
            "pip".into(),
            "install".into(),
            "-r".into(),
            requirements.to_string_lossy().into_owned(),
        ],
        vec!["-m".into(), "pip".into(), "check".into()],
    ] {
        checked_command(root, &python, &arguments).await?;
    }
    Ok(())
}

fn selected_model(root: &Path, catalog: &Catalog, model: &str) -> Option<PathBuf> {
    if model == "auto" {
        return catalog.models.first().map(|model| root.join(&model.path));
    }
    find_asset(
        root,
        &format!(
            "Qwen-Image-2512-{}.gguf",
            if model == "q4" { "Q4_K_M" } else { "Q8_0" }
        ),
    )
}

fn selected_lightning(root: &Path, catalog: &Catalog, choice: &str) -> Option<PathBuf> {
    catalog
        .lightning_adapters
        .iter()
        .find(|adapter| choice == "auto" || choice.parse::<u32>() == Ok(adapter.steps))
        .map(|adapter| root.join(&adapter.path))
}

fn lightning_names(choice: &str) -> Vec<String> {
    let steps = match choice {
        "4" => vec![4],
        "8" => vec![8],
        _ => vec![8, 4],
    };
    let mut names = Vec::new();
    for count in steps {
        for precision in ["fp32", "bf16"] {
            names.push(format!(
                "Qwen-Image-2512-Lightning-{count}steps-V1.0-{precision}.safetensors"
            ));
        }
    }
    names
}

fn download_choice(value: &str) -> Option<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "1" | "q4" | "q4_k_m" => Some("q4"),
        "2" | "q8" | "q8_0" => Some("q8"),
        _ => None,
    }
}

fn choose_download_model(model: &str) -> Result<String> {
    if model != "auto" {
        return Ok(model.to_owned());
    }
    if !std::io::stdin().is_terminal() {
        return Ok("q4".into());
    }
    println!(
        "Choose a GGUF model to download:\n  1. Q4_K_M (~13 GB) - lower memory use [default]\n  2. Q8_0 (~21.8 GB) - higher memory use"
    );
    loop {
        print!("Choice [1]: ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 {
            return Ok("q4".into());
        }
        if let Some(choice) = download_choice(&line) {
            return Ok(choice.into());
        }
        println!("Enter 1 for Q4_K_M or 2 for Q8_0.");
    }
}

#[derive(Debug, Deserialize)]
struct Repository {
    sha: String,
    siblings: Vec<RepositoryFile>,
}

#[derive(Debug, Deserialize)]
struct RepositoryFile {
    rfilename: String,
    size: Option<u64>,
    lfs: Option<LfsFile>,
}

#[derive(Debug, Deserialize)]
struct LfsFile {
    sha256: String,
    size: u64,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct DownloadIdentity {
    repository: String,
    revision: String,
    filename: String,
    sha256: String,
    size: u64,
}

impl DownloadIdentity {
    fn same_content(&self, other: &Self) -> bool {
        self.repository == other.repository
            && self.filename == other.filename
            && self.sha256 == other.sha256
            && self.size == other.size
    }
}

async fn download_models(root: &Path, model: &str, lightning: &str) -> Result<()> {
    tokio::select! {
        result = download_models_inner(root, model, lightning) => result,
        signal = tokio::signal::ctrl_c() => {
            signal.context("Cannot listen for Ctrl+C")?;
            bail!("Model download cancelled; rerun make models to resume");
        }
    }
}

async fn download_models_inner(root: &Path, model: &str, lightning: &str) -> Result<()> {
    ensure!(
        find_asset(root, "embeds_cache.pt").is_some(),
        "Missing {MODELS_DIRECTORY}/embeds_cache.pt. Restore the included file."
    );
    let catalog = Catalog::scan(root)?;
    let existing_model = selected_model(root, &catalog, model);
    let selected = if existing_model.is_some() {
        model.to_owned()
    } else {
        choose_download_model(model)?
    };
    let names = vec![format!(
        "Qwen-Image-2512-{}.gguf",
        if selected == "q8" { "Q8_0" } else { "Q4_K_M" }
    )];
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(60))
        .user_agent("DeSynth/0.1")
        .build()?;
    for (repository, names, existing) in [
        (GGUF_REPOSITORY, names, existing_model),
        (
            LIGHTNING_REPOSITORY,
            lightning_names(lightning),
            selected_lightning(root, &catalog, lightning),
        ),
    ] {
        if let Some(existing) = existing {
            println!("Already present: {}", existing.display());
            continue;
        }
        let response = client
            .get(format!("https://huggingface.co/api/models/{repository}"))
            .query(&[("blobs", "true")])
            .timeout(Duration::from_secs(30))
            .send()
            .await?
            .error_for_status()?;
        let details: Repository = response
            .json()
            .await
            .context("Invalid Hugging Face repository response")?;
        let identity = select_repository_file(repository, &details, &names)?;
        let directory = models_directory(root);
        fs::create_dir_all(&directory).await?;
        download_file(&client, &directory, &identity).await?;
    }
    println!("Model files are ready.");
    Ok(())
}

fn select_repository_file(
    repository: &str,
    details: &Repository,
    names: &[String],
) -> Result<DownloadIdentity> {
    ensure!(
        details.sha.len() == 40 && details.sha.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "Invalid repository revision"
    );
    let file = names
        .iter()
        .find_map(|name| {
            details
                .siblings
                .iter()
                .find(|file| file.rfilename.eq_ignore_ascii_case(name))
        })
        .with_context(|| format!("Cannot find {} in {repository}", names[0]))?;
    let lfs = file
        .lfs
        .as_ref()
        .context("Model file is missing its SHA-256 metadata")?;
    ensure!(
        lfs.size > 0 && file.size.unwrap_or(lfs.size) == lfs.size,
        "Invalid model file size"
    );
    ensure!(
        lfs.sha256.len() == 64 && lfs.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "Invalid model checksum"
    );
    ensure!(
        Path::new(&file.rfilename).components().count() == 1
            && !file.rfilename.contains(['/', '\\']),
        "Invalid model filename"
    );
    Ok(DownloadIdentity {
        repository: repository.into(),
        revision: details.sha.clone(),
        filename: file.rfilename.clone(),
        sha256: lfs.sha256.to_ascii_lowercase(),
        size: lfs.size,
    })
}

async fn download_file(
    client: &Client,
    directory: &Path,
    identity: &DownloadIdentity,
) -> Result<()> {
    let mut url = Url::parse("https://huggingface.co/")?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| anyhow::anyhow!("Invalid Hugging Face base URL"))?;
        for segment in identity.repository.split('/') {
            segments.push(segment);
        }
        segments.extend(["resolve", &identity.revision, &identity.filename]);
    }
    url.query_pairs_mut().append_pair("download", "true");
    download_file_from_url(client, directory, identity, url).await
}

async fn download_file_from_url(
    client: &Client,
    directory: &Path,
    identity: &DownloadIdentity,
    url: Url,
) -> Result<()> {
    download_file_with_request(directory, identity, |offset| async move {
        let mut request = client.get(url).timeout(Duration::from_secs(6 * 60 * 60));
        if offset > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
        }
        Ok(request.send().await?.error_for_status()?)
    })
    .await
}

async fn download_file_with_request<Request, Reply>(
    directory: &Path,
    identity: &DownloadIdentity,
    request: Request,
) -> Result<()>
where
    Request: FnOnce(u64) -> Reply,
    Reply: std::future::Future<Output = Result<reqwest::Response>>,
{
    let destination = directory.join(&identity.filename);
    let partial = directory.join(format!(".{}.part", identity.filename));
    let metadata = directory.join(format!(".{}.part.json", identity.filename));
    let cached_identity = fs::read(&metadata)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice::<DownloadIdentity>(&bytes).ok());
    if !cached_identity
        .as_ref()
        .is_some_and(|cached| cached.same_content(identity))
    {
        if partial.exists() {
            fs::remove_file(&partial).await?;
        }
        fs::write(&metadata, serde_json::to_vec(identity)?).await?;
    }
    let mut offset = fs::metadata(&partial)
        .await
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    if offset > identity.size {
        fs::remove_file(&partial).await?;
        offset = 0;
    }
    let mut hasher = Sha256::new();
    if offset > 0 {
        let mut input = fs::File::open(&partial).await?;
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let count = input.read(&mut buffer).await?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
    }
    println!(
        "Downloading {}/{} (resume: {:.2} GiB)",
        identity.repository,
        identity.filename,
        offset as f64 / 1073741824.0
    );
    if offset < identity.size {
        let mut response = request(offset).await?;
        if offset > 0 && response.status() == StatusCode::OK {
            offset = 0;
            hasher = Sha256::new();
        } else if offset > 0 {
            ensure!(
                response.status() == StatusCode::PARTIAL_CONTENT,
                "Server did not honor the download range"
            );
            let range = response
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default();
            ensure!(
                valid_content_range(range, offset, identity.size),
                "Server returned an unexpected download range"
            );
        }
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(offset > 0)
            .truncate(offset == 0)
            .open(&partial)
            .await?;
        let mut reported = Instant::now();
        while let Some(chunk) = response.chunk().await? {
            ensure!(
                offset + chunk.len() as u64 <= identity.size,
                "Download exceeds the advertised model size"
            );
            file.write_all(&chunk).await?;
            hasher.update(&chunk);
            offset += chunk.len() as u64;
            if reported.elapsed() >= Duration::from_secs(2) {
                print!(
                    "\r{}: {:.1}%",
                    identity.filename,
                    offset as f64 * 100.0 / identity.size as f64
                );
                std::io::stdout().flush()?;
                reported = Instant::now();
            }
        }
        file.flush().await?;
        file.sync_all().await?;
        println!();
    }
    ensure!(
        offset == identity.size,
        "Incomplete download: got {offset} of {} bytes; rerun make models to resume",
        identity.size
    );
    let checksum = format!("{:x}", hasher.finalize());
    if checksum != identity.sha256 {
        fs::remove_file(&partial).await?;
        fs::remove_file(&metadata).await?;
        bail!(
            "SHA-256 mismatch for {}; rerun make models",
            identity.filename
        );
    }
    if destination.exists() {
        ensure!(
            fs::metadata(&destination).await?.len() == 0,
            "{} already exists; keeping the verified download at {}",
            destination.display(),
            partial.display()
        );
        fs::remove_file(&destination).await?;
    }
    fs::rename(&partial, &destination).await?;
    fs::remove_file(&metadata).await?;
    println!("Saved and verified: {}", destination.display());
    Ok(())
}

fn valid_content_range(value: &str, offset: u64, total: u64) -> bool {
    let Some(value) = value.strip_prefix("bytes ") else {
        return false;
    };
    let Some((range, size)) = value.split_once('/') else {
        return false;
    };
    let Some((start, end)) = range.split_once('-') else {
        return false;
    };
    start.parse::<u64>() == Ok(offset)
        && size.parse::<u64>() == Ok(total)
        && end
            .parse::<u64>()
            .is_ok_and(|end| end >= offset && end < total)
}

async fn prepare(root: &Path, directory: &Path) -> Result<()> {
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

async fn doctor(
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
    fn setup_selects_canonical_weights_and_keeps_legacy_installations_usable() {
        let root = tempfile::tempdir().unwrap();
        let canonical = models_directory(root.path());
        let legacy = root.path().join("ai");
        std::fs::create_dir_all(&canonical).unwrap();
        std::fs::create_dir_all(&legacy).unwrap();
        let model = "Qwen-Image-2512-Q4_K_M.gguf";
        let adapter = "Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors";
        for name in [model, adapter] {
            std::fs::write(canonical.join(name), b"canonical weight").unwrap();
            std::fs::write(legacy.join(name), b"legacy weight").unwrap();
        }
        let catalog = Catalog::scan(root.path()).unwrap();
        assert_eq!(
            selected_model(root.path(), &catalog, "auto"),
            Some(canonical.join(model))
        );
        assert_eq!(
            selected_model(root.path(), &catalog, "q4"),
            Some(canonical.join(model))
        );
        assert_eq!(
            selected_lightning(root.path(), &catalog, "8"),
            Some(canonical.join(adapter))
        );
        for name in [model, adapter] {
            std::fs::write(canonical.join(name), b"").unwrap();
        }
        let catalog = Catalog::scan(root.path()).unwrap();
        assert_eq!(
            selected_model(root.path(), &catalog, "q4"),
            Some(legacy.join(model))
        );
        assert_eq!(
            selected_lightning(root.path(), &catalog, "auto"),
            Some(legacy.join(adapter))
        );
    }

    #[test]
    fn interpreter_version_checks_keep_the_supported_minor_versions() {
        for version in ["Python 3.11.9", "Python 3.12.0rc1", "Python 3.13.7\n"] {
            assert!(supported_python_version(version));
        }
        for version in [
            "Python 2.7.18",
            "Python 3.10.9",
            "Python 3.14.0",
            "not installed",
            "",
        ] {
            assert!(!supported_python_version(version));
        }
    }

    fn download_identity(bytes: &[u8]) -> DownloadIdentity {
        DownloadIdentity {
            repository: "owner/repo".into(),
            revision: "a".repeat(40),
            filename: "model.gguf".into(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
            size: bytes.len() as u64,
        }
    }

    fn response(
        status: u16,
        body: reqwest::Body,
        length: u64,
        range: Option<&str>,
    ) -> reqwest::Response {
        let mut response = axum::http::Response::builder()
            .status(status)
            .header("content-length", length);
        if let Some(range) = range {
            response = response.header("content-range", range);
        }
        reqwest::Response::from(response.body(body).unwrap())
    }

    #[tokio::test]
    async fn interrupted_download_resumes_and_publishes_only_verified_bytes() {
        let root = tempfile::tempdir().unwrap();
        let directory = models_directory(root.path());
        fs::create_dir_all(&directory).await.unwrap();
        let identity = download_identity(b"abcdef");
        let interrupted = reqwest::Body::wrap_stream(futures_util::stream::iter([
            Ok::<_, std::io::Error>(b"abc".to_vec()),
            Err(std::io::Error::other(
                "Simulated interrupted model download",
            )),
        ]));
        let first =
            download_file_with_request(directory.as_path(), &identity, |offset| async move {
                assert_eq!(offset, 0);
                Ok(response(200, interrupted, 6, None))
            })
            .await;
        assert!(first.is_err());
        assert!(!directory.as_path().join("model.gguf").exists());
        assert_eq!(
            fs::read(directory.as_path().join(".model.gguf.part"))
                .await
                .unwrap(),
            b"abc"
        );
        download_file_with_request(directory.as_path(), &identity, |offset| async move {
            assert_eq!(offset, 3);
            Ok(response(
                206,
                reqwest::Body::from("def"),
                3,
                Some("bytes 3-5/6"),
            ))
        })
        .await
        .unwrap();
        assert_eq!(
            fs::read(directory.as_path().join("model.gguf"))
                .await
                .unwrap(),
            b"abcdef"
        );
        assert!(!directory.as_path().join(".model.gguf.part").exists());
        assert!(!directory.as_path().join(".model.gguf.part.json").exists());
    }

    #[tokio::test]
    async fn checksum_mismatch_never_becomes_a_discoverable_model() {
        let root = tempfile::tempdir().unwrap();
        let directory = models_directory(root.path());
        fs::create_dir_all(&directory).await.unwrap();
        let identity = download_identity(b"abc");
        let error =
            download_file_with_request(directory.as_path(), &identity, |offset| async move {
                assert_eq!(offset, 0);
                Ok(response(200, reqwest::Body::from("xyz"), 3, None))
            })
            .await
            .unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"), "{error:#}");
        assert!(!directory.as_path().join("model.gguf").exists());
        assert!(!directory.as_path().join(".model.gguf.part").exists());
    }

    #[tokio::test]
    async fn ignored_range_restarts_the_file_instead_of_appending_duplicate_bytes() {
        let root = tempfile::tempdir().unwrap();
        let directory = models_directory(root.path());
        fs::create_dir_all(&directory).await.unwrap();
        let identity = download_identity(b"abc");
        fs::write(directory.as_path().join(".model.gguf.part"), b"a")
            .await
            .unwrap();
        fs::write(
            directory.as_path().join(".model.gguf.part.json"),
            serde_json::to_vec(&identity).unwrap(),
        )
        .await
        .unwrap();
        download_file_with_request(directory.as_path(), &identity, |offset| async move {
            assert_eq!(offset, 1);
            Ok(response(200, reqwest::Body::from("abc"), 3, None))
        })
        .await
        .unwrap();
        assert_eq!(
            fs::read(directory.as_path().join("model.gguf"))
                .await
                .unwrap(),
            b"abc"
        );
    }

    #[test]
    fn model_download_defaults_to_lower_memory_choice() {
        for value in ["", "1", "q4", "Q4_K_M", " q4 "] {
            assert_eq!(download_choice(value), Some("q4"));
        }
        for value in ["2", "q8", "Q8_0"] {
            assert_eq!(download_choice(value), Some("q8"));
        }
        assert_eq!(download_choice("99"), None);
        let names = lightning_names("auto");
        assert!(names[0].contains("8steps-V1.0-fp32"));
        assert!(
            lightning_names("4")
                .iter()
                .all(|name| name.contains("4steps"))
        );
    }

    #[test]
    fn repository_metadata_preserves_upstream_case_and_checks_integrity_fields() {
        let details: Repository = serde_json::from_value(serde_json::json!({
            "sha": "a".repeat(40),
            "siblings": [{"rfilename":"Qwen-Image-2512-Q4_K_M.gguf", "size":42,
                          "lfs":{"sha256":"b".repeat(64),"size":42}}]
        }))
        .unwrap();
        let names = ["qwen-image-2512-q4_k_m.gguf".into()];
        let selected = select_repository_file("owner/repo", &details, &names).unwrap();
        assert_eq!(selected.filename, "Qwen-Image-2512-Q4_K_M.gguf");
        assert_eq!(selected.size, 42);
        assert_eq!(selected.sha256, "b".repeat(64));
        assert!(select_repository_file("owner/repo", &details, &["absent.gguf".into()]).is_err());
    }

    #[test]
    fn resumable_download_rejects_wrong_ranges() {
        assert!(valid_content_range("bytes 10-99/100", 10, 100));
        for value in [
            "bytes 0-99/100",
            "bytes 10-99/200",
            "bytes 10-100/100",
            "bytes 10-9/100",
            "bad",
        ] {
            assert!(!valid_content_range(value, 10, 100), "{value}");
        }
    }

    #[test]
    fn readme_changes_do_not_discard_a_verified_partial_model() {
        let first = download_identity(b"model bytes");
        let mut updated = download_identity(b"model bytes");
        updated.revision = "b".repeat(40);
        assert!(first.same_content(&updated));
        updated.sha256 = "c".repeat(64);
        assert!(!first.same_content(&updated));
    }

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
