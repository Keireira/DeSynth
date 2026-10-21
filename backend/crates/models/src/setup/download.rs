use super::repository::{DownloadIdentity, Repository, select_repository_file};
use super::request::send;
use super::selection::{
    choose_download_model, lightning_names, selected_lightning, selected_model,
};
use crate::catalog::{Catalog, MODELS_DIRECTORY, find_asset, models_directory};
use anyhow::{Context, Result, bail, ensure};
use reqwest::{Client, StatusCode, Url};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const GGUF_REPOSITORY: &str = "Frederic75/Qwen-Image-2512-GGUF";
const LIGHTNING_REPOSITORY: &str = "lightx2v/Qwen-Image-2512-Lightning";

pub(super) async fn download_models(root: &Path, model: &str, lightning: &str) -> Result<()> {
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
        .retry(reqwest::retry::never())
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
        let response = send(
            client
                .get(format!("https://huggingface.co/api/models/{repository}"))
                .query(&[("blobs", "true")])
                .timeout(Duration::from_secs(30)),
        )
        .await
        .with_context(|| format!("Cannot fetch the model list from {repository}"))?;
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
        send(request).await
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
    let lock_path = directory.join(format!(".{}.download.lock", identity.filename));
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("Cannot open download lock {}", lock_path.display()))?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => bail!(
            "Another download of {} is already running",
            identity.filename
        ),
        Err(std::fs::TryLockError::Error(error)) => {
            return Err(error).context("Cannot lock the model download");
        }
    }
    // Keep this inode and its lock until publication finishes or the future is
    // canceled; unlinking the lock file would let another writer bypass it.
    let _download_lock = lock;
    let destination = directory.join(&identity.filename);
    if crate::catalog::nonempty_file(&destination) {
        println!("Already present: {}", destination.display());
        return Ok(());
    }
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

#[cfg(test)]
mod tests;
