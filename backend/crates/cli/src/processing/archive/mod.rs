use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use desynth_core::{parameters, studio::Studio};
use desynth_images as images;
use desynth_models::catalog;
use serde_json::{Value, json};

use crate::write_image;
use naming::{is_reference, model_filename_tag, reference_stem};

mod naming;
#[cfg(test)]
mod tests;

pub(super) struct RunArchive<'a> {
    pub(super) input: &'a Path,
    pub(super) directory: &'a Path,
    pub(super) stem: &'a str,
    pub(super) run_id: &'a str,
}

pub(super) fn archive_result(
    studio: &Studio,
    archive: &RunArchive<'_>,
    index: usize,
    params: &Value,
    job: &Value,
) -> Result<Value> {
    let RunArchive {
        input,
        directory,
        stem,
        run_id,
    } = archive;
    let (result, clean) = {
        let store = studio.store.lock();
        let result_id = job["result"]["id"]
            .as_str()
            .context("Job has no result image")?;
        let clean_id = job["clean"]["id"]
            .as_str()
            .context("Job has no clean image")?;
        (
            store.assets.get(result_id, Some("result"))?,
            store.assets.get(clean_id, Some("clean"))?,
        )
    };
    let model_name = Path::new(job["transformer"].as_str().unwrap_or("model"))
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let model_tag = model_filename_tag(&model_name);
    let adapter_name = result
        .metadata
        .as_ref()
        .and_then(|metadata| metadata["lightning"].as_str())
        .unwrap_or_default();
    let adapter_steps = catalog::lightning_steps(adapter_name)
        .map_or_else(|| "custom".to_owned(), |value| value.to_string());
    let adapter_precision = if adapter_name.to_lowercase().contains("fp32") {
        "fp32"
    } else if adapter_name.to_lowercase().contains("bf16") {
        "bf16"
    } else {
        ""
    };
    let mut tag = format!(
        "{model_tag}_l{adapter_steps}{adapter_precision}_s{}_d{}_p{}_seed{}_r{}{}_u{}",
        params["steps"],
        params["denoise"],
        params["passes"],
        job["seed"],
        params["restore_sigma"],
        params["restore_mode"].as_str().unwrap_or("gaussian"),
        params["unsharp"]
    );
    if params["restore"] == false {
        tag.push_str("_norestore");
    }
    let prefix = format!("{stem}_desynth_{tag}_{run_id}_{index:02}");
    let final_path = directory.join(format!("{prefix}.png"));
    let reference = if is_reference(input, &studio.root) {
        let path = studio
            .root
            .join("out")
            .join(format!("{}.png", reference_stem(params)?));
        let comparison = path.with_file_name(format!(
            "comparison_{}",
            path.file_name().unwrap_or_default().to_string_lossy()
        ));
        Some((path, comparison))
    } else {
        None
    };
    let intermediate = (params["restore"] == true && params["keep_intermediate"] == true)
        .then(|| directory.join(format!("{prefix}_clean.png")));
    let comparison_path = directory.join(format!("comparison_{prefix}.png"));
    let mut protected = vec![input.to_path_buf(), result.path.clone(), clean.path.clone()];
    let mut outputs = vec![final_path.as_path()];
    outputs.extend(intermediate.as_deref());
    if let Some((path, comparison)) = &reference {
        outputs.extend([
            comparison_path.as_path(),
            path.as_path(),
            comparison.as_path(),
        ]);
    }
    for output in outputs {
        crate::image_commands::check_output(
            output,
            &protected.iter().map(PathBuf::as_path).collect::<Vec<_>>(),
        )?;
        protected.push(output.to_path_buf());
    }
    crate::files::copy(&result.path, &final_path)?;
    println!("  -> {}", final_path.display());
    let original = images::open(&std::fs::read(input)?)?;
    let mut clean_path = None;
    if let Some(path) = intermediate {
        let clean_image = images::open(&std::fs::read(&clean.path)?)?;
        let clean_image = images::finish(
            &clean_image,
            &original,
            &images::RestorationParams {
                restore: false,
                ..Default::default()
            },
        )?;
        write_image(&path, &clean_image, clean.metadata.as_ref())?;
        clean_path = Some(path);
    }
    let denoise = params["denoise"].as_f64().context("Denoise is missing")?;
    let steps = params["steps"].as_u64().context("Step count is missing")? as u32;
    let mut record = json!({"kind":"result","path":final_path,"name":final_path.file_name().unwrap_or_default().to_string_lossy(),"denoise":denoise,"starting_sigma":denoise,"effective_steps":parameters::effective_img2img_steps(steps,denoise),"clean_path":clean_path,"transformer":job["transformer"],"lightning":job["lightning"],"runtime":job["runtime"],"seed":job["seed"]});
    if let Some((reference_path, reference_comparison)) = reference {
        let final_image = images::open(&std::fs::read(&result.path)?)?;
        let metrics = images::extended_metrics(&original, &final_image)?;
        let comparison = images::comparison_visual(&original, &final_image, &metrics)?;
        write_image(&comparison_path, &comparison, None)?;
        crate::files::copy(&final_path, &reference_path)?;
        crate::files::copy(&comparison_path, &reference_comparison)?;
        println!("  -> {}", reference_path.display());
        println!("  -> {}", reference_comparison.display());
        record["reference_path"] = json!(reference_path);
        record["comparison_path"] = json!(comparison_path);
        record["reference_comparison_path"] = json!(reference_comparison);
    }
    // The archive owns CLI outputs; discard the studio's private working copies.
    for asset in [&result, &clean] {
        anyhow::ensure!(
            asset.path.starts_with(studio.root.join("out/gui")),
            "Refusing to remove an image outside the working directory"
        );
        std::fs::remove_file(&asset.path)?;
        let mut store = studio.store.lock();
        store.assets.entries.shift_remove(&asset.asset.id);
        store.assets.clean_sources.shift_remove(&asset.asset.id);
    }
    Ok(record)
}
