use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

pub(super) fn is_reference(input: &Path, root: &Path) -> bool {
    match (
        input.canonicalize(),
        root.join("original.png").canonicalize(),
    ) {
        (Ok(input), Ok(reference)) => input == reference,
        _ => false,
    }
}

pub(super) fn model_filename_tag(stem: &str) -> String {
    let suffix = stem.rsplit('-').next().unwrap_or(stem);
    let mut characters = suffix.chars();
    let quant = characters
        .next()
        .is_some_and(|character| matches!(character, 'q' | 'Q' | 'i' | 'I'))
        && characters
            .next()
            .is_some_and(|character| character.is_ascii_digit());
    let selected = if quant { suffix } else { stem };
    let tag: String = selected
        .chars()
        .take(32)
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    if tag.is_empty() { "model".into() } else { tag }
}

pub(super) fn reference_stem(params: &Value) -> Result<String> {
    let sigma = params["restore_sigma"]
        .as_f64()
        .context("Restoration sigma is missing")?;
    let denoise = params["denoise"].as_f64().context("Denoise is missing")?;
    let mut stem = format!(
        "Original_desynth_s{}_d{denoise:.3}_p{}_r{sigma}",
        params["steps"], params["passes"]
    );
    if params["restore"] == false {
        stem.push_str("_norestore");
    } else if params["restore_mode"] == "edge" {
        stem.push_str("_edge");
    }
    if params["restore"] == true && params["unsharp"].as_f64().is_some_and(|value| value != 0.0) {
        stem.push_str(&format!("_u{}", params["unsharp"]));
    }
    Ok(stem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn only_root_original_publishes_reference_aliases() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let root = directory.path();
        std::fs::write(root.join("original.png"), [1])?;
        std::fs::create_dir(root.join("another"))?;
        std::fs::write(root.join("another/original.png"), [1])?;
        assert!(is_reference(&root.join("original.png"), root));
        assert!(!is_reference(&root.join("another/original.png"), root));
        let mut params = json!({"steps":32,"denoise":0.25,"passes":1,"restore_sigma":1.95,"restore":true,"restore_mode":"gaussian","unsharp":0.0});
        assert_eq!(
            reference_stem(&params)?,
            "Original_desynth_s32_d0.250_p1_r1.95"
        );
        params["restore_mode"] = json!("edge");
        assert_eq!(
            reference_stem(&params)?,
            "Original_desynth_s32_d0.250_p1_r1.95_edge"
        );
        Ok(())
    }
}
