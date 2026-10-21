use crate::catalog::{Catalog, find_asset};
use anyhow::Result;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

pub(super) fn selected_model(root: &Path, catalog: &Catalog, model: &str) -> Option<PathBuf> {
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

pub(super) fn selected_lightning(root: &Path, catalog: &Catalog, choice: &str) -> Option<PathBuf> {
    catalog
        .lightning_adapters
        .iter()
        .find(|adapter| choice == "auto" || choice.parse::<u32>() == Ok(adapter.steps))
        .map(|adapter| root.join(&adapter.path))
}

pub(super) fn lightning_names(choice: &str) -> Vec<String> {
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

pub(super) fn choose_download_model(model: &str) -> Result<String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::models_directory;

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
}
