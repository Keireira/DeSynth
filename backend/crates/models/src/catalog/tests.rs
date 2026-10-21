use super::*;

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("desynth-catalog-{}", uuid::Uuid::new_v4()));
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
