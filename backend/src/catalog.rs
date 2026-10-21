use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

pub const MODELS_DIRECTORY: &str = "inference/models";

pub fn models_directory(root: &Path) -> PathBuf {
    root.join(MODELS_DIRECTORY)
}

fn asset_directories(root: &Path) -> [PathBuf; 3] {
    [models_directory(root), root.join("ai"), root.to_path_buf()]
}

#[derive(Clone, Debug, Serialize)]
pub struct Model {
    pub name: String,
    pub path: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct LightningAdapter {
    pub name: String,
    pub path: String,
    pub steps: u32,
    pub label: String,
}

#[derive(Clone, Debug, Default)]
pub struct Catalog {
    pub models: Vec<Model>,
    pub lightning_adapters: Vec<LightningAdapter>,
}

impl Catalog {
    pub fn scan(root: &Path) -> io::Result<Self> {
        let mut models: Vec<_> = files(root, "gguf")?
            .into_iter()
            .map(|path| Model {
                name: filename(&path),
                path: relative_path(root, &path),
            })
            .collect();
        models.sort_by_key(|model| {
            let name = model.name.to_lowercase();
            let priority = if name.contains("q4_k_m") {
                0
            } else if name.contains("q2_k_m") {
                1
            } else {
                2
            };
            (priority, name)
        });
        let mut lightning_adapters = Vec::new();
        for path in files(root, "safetensors")? {
            let name = filename(&path);
            let lower = name.to_lowercase();
            if !lower.contains("qwen-image-2512-lightning-") {
                continue;
            }
            let Some(steps) = lightning_steps(&name) else {
                continue;
            };
            let precision = if lower.contains("fp32") {
                " · FP32"
            } else if lower.contains("bf16") {
                " · BF16"
            } else {
                ""
            };
            lightning_adapters.push(LightningAdapter {
                path: relative_path(root, &path),
                label: format!("{steps} steps{precision}"),
                name,
                steps,
            });
        }
        lightning_adapters.sort_by_key(|adapter| {
            let name = adapter.name.to_lowercase();
            (adapter.steps != 8, !name.contains("fp32"), name)
        });
        Ok(Self {
            models,
            lightning_adapters,
        })
    }

    pub fn default_lightning(&self) -> Option<&str> {
        self.lightning_adapters
            .first()
            .map(|adapter| adapter.path.as_str())
    }

    pub fn default_steps(&self) -> u32 {
        default_steps_for_lightning(self.default_lightning().unwrap_or_default())
    }

    pub fn automatic_model_paths(&self, root: &Path) -> Vec<String> {
        let available: Vec<_> = self
            .models
            .iter()
            .filter_map(|model| {
                let metadata = fs::metadata(root.join(&model.path)).ok()?;
                if !metadata.is_file() || metadata.len() == 0 {
                    return None;
                }
                Some((model.path.clone(), metadata.len()))
            })
            .collect();
        let Some((preferred, size)) = available.first() else {
            return Vec::new();
        };
        let mut smaller: Vec<_> = available
            .iter()
            .skip(1)
            .filter(|(_, candidate_size)| candidate_size < size)
            .cloned()
            .collect();
        smaller.sort_by_key(|(_, candidate_size)| std::cmp::Reverse(*candidate_size));
        let mut paths = vec![preferred.clone()];
        paths.extend(smaller.into_iter().map(|(path, _)| path));
        paths
    }
}

pub fn nonempty_file(path: &Path) -> bool {
    fs::metadata(path)
        .map(|metadata| metadata.is_file() && metadata.len() > 0)
        .unwrap_or(false)
}

pub fn find_asset(root: &Path, name: &str) -> Option<PathBuf> {
    for directory in asset_directories(root) {
        let exact = directory.join(name);
        if nonempty_file(&exact) {
            return Some(exact);
        }
        let Ok(entries) = fs::read_dir(&directory) else {
            continue;
        };
        let mut entries: Vec<_> = entries.filter_map(Result::ok).collect();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            if entry
                .file_name()
                .to_string_lossy()
                .eq_ignore_ascii_case(name)
                && nonempty_file(&entry.path())
            {
                return Some(entry.path());
            }
        }
    }
    None
}

pub fn embeddings_path(root: &Path) -> PathBuf {
    find_asset(root, "embeds_cache.pt")
        .unwrap_or_else(|| models_directory(root).join("embeds_cache.pt"))
}

pub fn lightning_steps(path: &str) -> Option<u32> {
    let name = Path::new(path)
        .file_name()?
        .to_string_lossy()
        .to_lowercase();
    for count in [4, 8] {
        let pattern = format!("-{count}steps");
        if let Some((_, suffix)) = name.split_once(&pattern)
            && (suffix.is_empty() || suffix.starts_with('-') || suffix.starts_with('.'))
        {
            return Some(count);
        }
    }
    None
}

pub fn default_steps_for_lightning(path: &str) -> u32 {
    if lightning_steps(path) == Some(8) {
        32
    } else {
        8
    }
}

pub fn weight_identity(path: &Path) -> io::Result<(PathBuf, u64, u128)> {
    let metadata = fs::metadata(path)?;
    let modified = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    Ok((path.canonicalize()?, metadata.len(), modified))
}

fn filename(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn files(root: &Path, extension: &str) -> io::Result<Vec<PathBuf>> {
    let mut found = BTreeMap::new();
    for directory in asset_directories(root) {
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        let mut entries = entries.collect::<io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let matches_extension = path
                .extension()
                .is_some_and(|suffix| suffix.to_string_lossy().eq_ignore_ascii_case(extension));
            if matches_extension && nonempty_file(&path) {
                found.entry(filename(&path).to_lowercase()).or_insert(path);
            }
        }
    }
    Ok(found.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Directory(PathBuf);

    impl Directory {
        fn new() -> Self {
            let root =
                std::env::temp_dir().join(format!("desynth-catalog-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(root.join("ai")).unwrap();
            fs::create_dir_all(models_directory(&root)).unwrap();
            Self(root)
        }

        fn file(&self, name: &str, size: usize) {
            fs::write(self.0.join(name), vec![0_u8; size]).unwrap();
        }
    }

    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn prefers_model_store_and_only_falls_back_to_smaller_models() {
        let root = Directory::new();
        root.file("inference/models/model-Q4_K_M.gguf", 100);
        root.file("MODEL-Q4_K_M.GGUF", 90);
        root.file("inference/models/model-Q2_K_M.gguf", 40);
        root.file("inference/models/model-Q8_0.gguf", 200);
        root.file("inference/models/empty.gguf", 0);
        let catalog = Catalog::scan(&root.0).unwrap();
        assert_eq!(catalog.models.len(), 3);
        assert_eq!(catalog.models[0].path, "inference/models/model-Q4_K_M.gguf");
        assert_eq!(
            catalog.automatic_model_paths(&root.0),
            [
                "inference/models/model-Q4_K_M.gguf",
                "inference/models/model-Q2_K_M.gguf"
            ]
        );
        fs::remove_file(root.0.join("inference/models/model-Q4_K_M.gguf")).unwrap();
        assert_eq!(
            catalog.automatic_model_paths(&root.0),
            ["inference/models/model-Q2_K_M.gguf"]
        );
    }

    #[test]
    fn prefers_eight_step_fp32_and_ignores_unrelated_weights() {
        let root = Directory::new();
        for name in [
            "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
            "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors",
            "Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors",
            "other-8steps.safetensors",
            "Qwen-Image-2512-Lightning-8stepsmore.safetensors",
        ] {
            root.file(&format!("inference/models/{name}"), 1);
        }
        let catalog = Catalog::scan(&root.0).unwrap();
        assert_eq!(catalog.lightning_adapters.len(), 3);
        assert_eq!(catalog.lightning_adapters[0].label, "8 steps · FP32");
        assert_eq!(catalog.default_steps(), 32);
        assert_eq!(default_steps_for_lightning("model-4steps.safetensors"), 8);
    }

    #[test]
    fn embeddings_prefer_model_store_and_support_both_legacy_locations() {
        let root = Directory::new();
        root.file("inference/models/EMBEDS_CACHE.PT", 1);
        root.file("ai/EMBEDS_CACHE.PT", 1);
        root.file("embeds_cache.pt", 1);
        assert_eq!(
            embeddings_path(&root.0).canonicalize().unwrap(),
            root.0
                .join("inference/models/EMBEDS_CACHE.PT")
                .canonicalize()
                .unwrap()
        );
        fs::remove_file(models_directory(&root.0).join("EMBEDS_CACHE.PT")).unwrap();
        assert_eq!(
            embeddings_path(&root.0).canonicalize().unwrap(),
            root.0.join("ai/EMBEDS_CACHE.PT").canonicalize().unwrap()
        );
        fs::remove_file(root.0.join("ai/EMBEDS_CACHE.PT")).unwrap();
        assert_eq!(
            embeddings_path(&root.0).canonicalize().unwrap(),
            root.0.join("embeds_cache.pt").canonicalize().unwrap()
        );
        fs::remove_file(root.0.join("embeds_cache.pt")).unwrap();
        assert_eq!(
            embeddings_path(&root.0),
            models_directory(&root.0).join("embeds_cache.pt")
        );
    }

    #[test]
    fn canonical_models_deduplicate_legacy_names_and_empty_files_allow_fallback() {
        let root = Directory::new();
        root.file("inference/models/MODEL-Q4_K_M.GGUF", 100);
        root.file("ai/model-q4_k_m.gguf", 90);
        root.file("Model-Q4_K_M.gguf", 80);
        let canonical = root.0.join("inference/models/MODEL-Q4_K_M.GGUF");
        let legacy = root.0.join("ai/model-q4_k_m.gguf");
        for (path, expected) in [
            (&canonical, "inference/models/MODEL-Q4_K_M.GGUF"),
            (&legacy, "ai/model-q4_k_m.gguf"),
        ] {
            let catalog = Catalog::scan(&root.0).unwrap();
            assert_eq!(catalog.models.len(), 1);
            assert_eq!(catalog.models[0].path, expected);
            assert_eq!(
                find_asset(&root.0, "model-q4_k_m.gguf")
                    .unwrap()
                    .canonicalize()
                    .unwrap(),
                path.canonicalize().unwrap()
            );
            fs::write(path, b"").unwrap();
        }
        let catalog = Catalog::scan(&root.0).unwrap();
        assert_eq!(catalog.models.len(), 1);
        assert_eq!(catalog.models[0].path, "Model-Q4_K_M.gguf");
    }
}
