use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Deserialize)]
pub(super) struct Repository {
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
pub(super) struct DownloadIdentity {
    pub(super) repository: String,
    pub(super) revision: String,
    pub(super) filename: String,
    pub(super) sha256: String,
    pub(super) size: u64,
}

impl DownloadIdentity {
    pub(super) fn same_content(&self, other: &Self) -> bool {
        self.repository == other.repository
            && self.filename == other.filename
            && self.sha256 == other.sha256
            && self.size == other.size
    }
}

pub(super) fn select_repository_file(
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
