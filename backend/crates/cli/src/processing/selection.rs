use std::path::Path;

use anyhow::{Context, Result};
use desynth_models::catalog::{self, LightningAdapter, Model};

use crate::rooted;

pub(super) fn resolve_model(
    root: &Path,
    catalog: &mut catalog::Catalog,
    requested: Option<&Path>,
    adapter: bool,
) -> Result<Option<String>> {
    let Some(requested) = requested else {
        return Ok(None);
    };
    let requested_text = requested.to_string_lossy();
    let candidates: Vec<_> = if adapter {
        catalog
            .lightning_adapters
            .iter()
            .map(|item| (&item.name, &item.path))
            .collect()
    } else {
        catalog
            .models
            .iter()
            .map(|item| (&item.name, &item.path))
            .collect()
    };
    if let Some((_, path)) = candidates
        .iter()
        .find(|(name, path)| path.as_str() == requested_text || name.as_str() == requested_text)
    {
        return Ok(Some(path.to_string()));
    }
    let mut path = rooted(root, requested);
    let filename_only = requested.components().count() == 1;
    let legacy_directory = requested.parent().is_some_and(|parent| {
        parent == Path::new("ai")
            || (parent
                .file_name()
                .is_some_and(|name| name.eq_ignore_ascii_case("ai"))
                && parent
                    .parent()
                    .and_then(|base| base.canonicalize().ok())
                    .as_deref()
                    == Some(root))
    });
    if !path.is_file()
        && (filename_only || legacy_directory)
        && let Some(name) = requested.file_name().and_then(|name| name.to_str())
        && let Some(found) = catalog::find_asset(root, name)
    {
        if let Ok(relative) = found.strip_prefix(root) {
            let relative = relative.to_string_lossy().replace('\\', "/");
            if candidates.iter().any(|(_, path)| path.as_str() == relative) {
                return Ok(Some(relative));
            }
        }
        path = found;
    }
    let path = path
        .canonicalize()
        .with_context(|| format!("Model file not found: {}", requested.display()))?;
    let extension = if adapter { "safetensors" } else { "gguf" };
    anyhow::ensure!(
        path.extension()
            .is_some_and(|actual| actual.eq_ignore_ascii_case(extension))
            && path.metadata()?.len() > 0,
        "A nonempty .{extension} file is required"
    );
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let path = path.to_string_lossy().into_owned();
    if adapter {
        let steps = catalog::lightning_steps(&name).unwrap_or(4);
        catalog.lightning_adapters.push(LightningAdapter {
            name,
            path: path.clone(),
            steps,
            label: format!("{steps} steps"),
        });
    } else {
        catalog.models.push(Model {
            name,
            path: path.clone(),
        });
    }
    Ok(Some(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn moved_models_keep_working_with_filenames_and_legacy_ai_paths() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let root = directory.path().canonicalize()?;
        let models = catalog::models_directory(&root);
        std::fs::create_dir_all(&models)?;
        for (name, adapter) in [
            ("Qwen-Image-2512-Q4_K_M.gguf", false),
            (
                "Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors",
                true,
            ),
        ] {
            std::fs::write(models.join(name), [1])?;
            let expected = format!("{}/{name}", catalog::MODELS_DIRECTORY);
            for requested in [
                PathBuf::from(name),
                PathBuf::from("ai").join(name),
                root.join("ai").join(name),
            ] {
                let mut catalog = catalog::Catalog::scan(&root)?;
                assert_eq!(
                    resolve_model(&root, &mut catalog, Some(&requested), adapter)?,
                    Some(expected.clone())
                );
            }
            let mut catalog = catalog::Catalog::scan(&root)?;
            assert!(
                resolve_model(
                    &root,
                    &mut catalog,
                    Some(&root.join("missing").join(name)),
                    adapter
                )
                .is_err()
            );
        }
        Ok(())
    }
}
