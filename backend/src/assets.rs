use crate::images;
use anyhow::{Context, Result, bail};
use image::RgbaImage;
use indexmap::IndexMap;
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    time::SystemTime,
};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize)]
pub struct Asset {
    pub id: String,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub url: String,
    pub kind: String,
}

#[derive(Clone)]
pub struct StoredAsset {
    pub asset: Asset,
    pub path: PathBuf,
    pub temporary_bytes: Option<u64>,
    pub metadata: Option<Value>,
}

/// A newly written file is removed if its request is canceled before committing it.
pub struct PreparedAsset(Option<StoredAsset>);

impl PreparedAsset {
    pub fn commit(mut self) -> StoredAsset {
        self.0
            .take()
            .expect("Prepared assets can only be committed once")
    }
}

impl Drop for PreparedAsset {
    fn drop(&mut self) {
        if let Some(entry) = &self.0 {
            let _ = std::fs::remove_file(&entry.path);
        }
    }
}

pub struct Assets {
    pub entries: IndexMap<String, StoredAsset>,
    pub clean_sources: IndexMap<String, String>,
    pub default_source: Option<String>,
    pub default_result: Option<String>,
    pub latest_source: Option<String>,
    pub directory: PathBuf,
}

impl Assets {
    pub fn new(root: &Path) -> Result<Self> {
        let directory = root.join("out/gui");
        std::fs::create_dir_all(directory.join("temporary"))?;
        let mut assets = Self {
            entries: IndexMap::new(),
            clean_sources: IndexMap::new(),
            default_source: None,
            default_result: None,
            latest_source: None,
            directory,
        };

        let original = std::fs::read_dir(root)?
            .filter_map(Result::ok)
            .find(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .eq_ignore_ascii_case("original.png")
            });
        if let Some(original) = original {
            match assets.register(original.path(), "source", None) {
                Ok(asset) => assets.default_source = Some(asset.id),
                Err(error) => tracing::warn!(%error, "Could not load reference image"),
            }
        }

        let source_size = assets
            .default_source
            .as_ref()
            .and_then(|id| assets.entries.get(id))
            .map(|entry| (entry.asset.width, entry.asset.height));
        let mut results = std::fs::read_dir(root.join("out"))?
            .filter_map(Result::ok)
            .filter(|entry| {
                let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                name.starts_with("original_desynth")
                    && name.ends_with(".png")
                    && !name.contains("_clean")
                    && !name.contains("_diff")
                    && !name.contains("comparison_")
            })
            .collect::<Vec<_>>();
        results.sort_by_key(|entry| {
            entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH)
        });
        for entry in results
            .into_iter()
            .rev()
            .take(16)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            if let Ok(asset) = assets.register(entry.path(), "result", None) {
                if Some((asset.width, asset.height)) == source_size {
                    assets.default_result = Some(asset.id);
                } else {
                    assets.entries.shift_remove(&asset.id);
                }
            }
        }
        Ok(assets)
    }

    pub fn register(&mut self, path: PathBuf, kind: &str, name: Option<&str>) -> Result<Asset> {
        let entry = Self::inspect(path, kind, name)?;
        Ok(self.insert(entry))
    }

    pub fn inspect(path: PathBuf, kind: &str, name: Option<&str>) -> Result<StoredAsset> {
        let bytes = std::fs::read(&path)?;
        let image = images::open(&bytes)?;
        let metadata = images::png_metadata(&bytes);
        let id = Uuid::new_v4().simple().to_string();
        let asset = Asset {
            id: id.clone(),
            name: name.map(str::to_owned).unwrap_or_else(|| {
                path.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            }),
            width: image.width(),
            height: image.height(),
            url: format!("/api/image/{id}"),
            kind: kind.to_owned(),
        };
        Ok(StoredAsset {
            asset,
            path,
            temporary_bytes: None,
            metadata,
        })
    }

    pub fn insert(&mut self, entry: StoredAsset) -> Asset {
        let asset = entry.asset.clone();
        self.entries.insert(asset.id.clone(), entry);
        asset
    }

    pub fn save(
        &mut self,
        image: &RgbaImage,
        kind: &str,
        name: &str,
        temporary: bool,
        metadata: Option<Value>,
    ) -> Result<Asset> {
        let prepared = Self::prepare(&self.directory, image, kind, name, temporary, metadata)?;
        Ok(self.insert(prepared.commit()))
    }

    pub fn prepare(
        directory: &Path,
        image: &RgbaImage,
        kind: &str,
        name: &str,
        temporary: bool,
        metadata: Option<Value>,
    ) -> Result<PreparedAsset> {
        let id = Uuid::new_v4().simple().to_string();
        let path = directory
            .join(if temporary { "temporary" } else { "" })
            .join(format!("{id}.png"));
        let bytes = images::encode_png(image, metadata.as_ref())?;
        if let Err(error) = std::fs::write(&path, &bytes) {
            let _ = std::fs::remove_file(&path);
            return Err(error.into());
        }
        let asset = Asset {
            id: id.clone(),
            name: name.to_owned(),
            width: image.width(),
            height: image.height(),
            url: format!("/api/image/{id}"),
            kind: kind.to_owned(),
        };
        Ok(PreparedAsset(Some(StoredAsset {
            asset,
            path,
            temporary_bytes: temporary.then_some(bytes.len() as u64),
            metadata,
        })))
    }

    pub fn get(&self, id: &str, kind: Option<&str>) -> Result<StoredAsset> {
        let entry = self
            .entries
            .get(id)
            .context("Image not found. Please upload it again.")?;
        if kind.is_some_and(|kind| entry.asset.kind != kind) {
            bail!("Unexpected image type.");
        }
        Ok(entry.clone())
    }

    pub fn prune(&mut self, protected: &HashSet<String>) {
        let mut keep = protected.clone();
        keep.extend(self.default_source.iter().cloned());
        keep.extend(self.default_result.iter().cloned());
        keep.extend(self.latest_source.iter().cloned());
        let mut temporary_count = self
            .entries
            .values()
            .filter(|entry| entry.temporary_bytes.is_some())
            .count();
        let mut temporary_bytes: u64 = self
            .entries
            .values()
            .filter_map(|entry| entry.temporary_bytes)
            .sum();
        let candidates = self.entries.keys().cloned().collect::<Vec<_>>();
        for id in candidates {
            if keep.contains(&id) {
                continue;
            }
            let Some(entry) = self.entries.get(&id) else {
                continue;
            };
            let excess_temp = entry.temporary_bytes.is_some()
                && (temporary_count > 48 || temporary_bytes > 512 * 1024 * 1024);
            if self.entries.len() <= 128 && !excess_temp {
                continue;
            }
            let Some(entry) = self.entries.shift_remove(&id) else {
                continue;
            };
            if let Some(bytes) = entry.temporary_bytes {
                if entry.path.parent() == Some(self.directory.join("temporary").as_path()) {
                    let _ = std::fs::remove_file(entry.path);
                }
                temporary_count -= 1;
                temporary_bytes = temporary_bytes.saturating_sub(bytes);
            }
            self.clean_sources.shift_remove(&id);
        }
    }

    pub fn cleanup(&mut self) {
        for entry in self.entries.values() {
            if entry.temporary_bytes.is_some()
                && entry.path.parent() == Some(self.directory.join("temporary").as_path())
            {
                let _ = std::fs::remove_file(&entry.path);
            }
        }
    }
}
