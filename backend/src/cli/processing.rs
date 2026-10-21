use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use rand::Rng;
use serde_json::{Value, json};

use crate::{
    catalog::{self, LightningAdapter, Model},
    images, parameters,
    studio::Studio,
};

use super::{RunArgs, rooted, write_image};

struct RunArchive<'a> {
    input: &'a Path,
    directory: &'a Path,
    stem: &'a str,
    run_id: &'a str,
}

pub(super) async fn run(root: PathBuf, python: PathBuf, args: RunArgs) -> Result<()> {
    let studio = Studio::new(root, python)?;
    let result = tokio::select! {
        result = process_sweep(studio.clone(), args) => result,
        signal = tokio::signal::ctrl_c() => match signal {
            Ok(()) => Err(anyhow::anyhow!("Processing cancelled")),
            Err(error) => Err(error).context("Cannot listen for Ctrl+C"),
        },
    };
    studio.close().await;
    result
}

async fn process_sweep(studio: Arc<Studio>, args: RunArgs) -> Result<()> {
    let input = rooted(&studio.root, &args.input)
        .canonicalize()
        .context("Input image does not exist")?;
    let output = rooted(&studio.root, &args.out);
    let source = studio
        .store
        .lock()
        .assets
        .register(input.clone(), "source", None)?;
    let default_steps = args.lightning.as_deref().map_or_else(
        || studio.store.lock().catalog.default_steps(),
        |path| catalog::default_steps_for_lightning(&path.to_string_lossy()),
    );
    let steps = args.steps.unwrap_or(default_steps);
    let seed = args
        .seed
        .unwrap_or_else(|| rand::rng().random_range(0..(1_u64 << 31)));
    let mut params = json!({"steps":steps,"passes":args.passes,"seed":seed,"restore":!args.no_restore,"restore_sigma":args.restore_sigma,"restore_mode":args.restore_mode,"unsharp":args.unsharp,"keep_intermediate":args.keep_intermediate,"lightning_scale":args.lightning_scale,"device":args.device,"offload":args.offload,"precision":args.precision,"preview":args.preview});
    for denoise in &args.denoise {
        let mut candidate = params.clone();
        candidate["denoise"] = json!(denoise);
        parameters::parse_process(&candidate, steps).map_err(anyhow::Error::msg)?;
    }
    println!(
        "Input: {} ({} x {}); seed: {seed}",
        input.display(),
        source.width,
        source.height
    );
    println!(
        "Schedule: {steps} total steps; Lightning scale {}",
        args.lightning_scale
    );
    studio.refresh()?;
    wait_for_probe(&studio).await?;
    {
        let mut store = studio.store.lock();
        params["transformer"] = json!(resolve_model(
            &studio.root,
            &mut store.catalog,
            args.transformer.as_deref(),
            false
        )?);
        params["lightning"] = json!(resolve_model(
            &studio.root,
            &mut store.catalog,
            args.lightning.as_deref(),
            true
        )?);
    }
    let run_id = uuid::Uuid::new_v4().simple().to_string()[..12].to_owned();
    let directory = output.join("runs").join(&run_id);
    std::fs::create_dir_all(output.join("runs"))?;
    std::fs::create_dir(&directory)?;
    let stem: String = input
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .chars()
        .take(60)
        .collect();
    let manifest_path = directory.join(format!("{stem}_desynth_{run_id}.json"));
    let mut manifest = json!({"schema_version":1,"run_id":run_id,"sampling":"linear-sigma-v1","created_at_unix":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),"source":{"path":input,"name":source.name,"width":source.width,"height":source.height},"params":params,"outputs":[]});
    let archive = RunArchive {
        input: &input,
        directory: &directory,
        stem: &stem,
        run_id: &run_id,
    };
    for (index, denoise) in args.denoise.iter().enumerate() {
        let mut params = params.clone();
        params["denoise"] = json!(denoise);
        println!(
            "Denoise {denoise}: {} updates per pass",
            parameters::effective_img2img_steps(steps, *denoise)
        );
        let queued = studio.process(json!({"source_id":source.id,"params":params}))?;
        let id = queued["id"].as_str().context("Processing job has no ID")?;
        let job = wait_for_job(&studio, id).await?;
        manifest["runtime"] = job["runtime"].clone();
        manifest["working_size"] = job["preview_size"].clone();
        for field in ["transformer", "lightning"] {
            let selected = Path::new(
                job[field]
                    .as_str()
                    .context("Selected model path is missing")?,
            );
            manifest[field] = json!({
                "path":rooted(&studio.root, selected),
                "name":selected.file_name().unwrap_or_default().to_string_lossy(),
            });
        }
        manifest["lightning"]["scale"] = job["lightning_scale"].clone();
        let record = archive_result(&studio, &archive, index + 1, &params, &job)?;
        manifest["outputs"]
            .as_array_mut()
            .context("Run manifest has no outputs")?
            .push(record);
        super::files::write(&manifest_path, &serde_json::to_vec_pretty(&manifest)?)?;
    }
    println!("Run metadata: {}", manifest_path.display());
    Ok(())
}

async fn wait_for_probe(studio: &Arc<Studio>) -> Result<()> {
    for _ in 0..240 {
        let state = studio.state();
        if state["runtime"]["ready"] == true && state["runtime"]["refreshing"] != true {
            if state["ready"] != true {
                let missing = state["missing"]
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();
                bail!("Inference is unavailable ({missing}). Run make setup first.");
            }
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    bail!("Timed out while checking the inference worker")
}

async fn wait_for_job(studio: &Studio, id: &str) -> Result<Value> {
    let mut previous_stage = String::new();
    loop {
        let job = studio.job(id)?;
        match job["status"].as_str() {
            Some("done") => return Ok(job),
            Some("error") => bail!("{}", job["error"].as_str().unwrap_or("Processing failed")),
            _ => {}
        }
        if let Some(stage) = job["stage"]
            .as_str()
            .filter(|stage| *stage != previous_stage)
        {
            println!("  {stage}");
            previous_stage = stage.to_owned();
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn resolve_model(
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

fn archive_result(
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
        super::image_commands::check_output(
            output,
            &protected.iter().map(PathBuf::as_path).collect::<Vec<_>>(),
        )?;
        protected.push(output.to_path_buf());
    }
    super::files::copy(&result.path, &final_path)?;
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
        super::files::copy(&final_path, &reference_path)?;
        super::files::copy(&comparison_path, &reference_comparison)?;
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

fn is_reference(input: &Path, root: &Path) -> bool {
    match (
        input.canonicalize(),
        root.join("original.png").canonicalize(),
    ) {
        (Ok(input), Ok(reference)) => input == reference,
        _ => false,
    }
}

fn model_filename_tag(stem: &str) -> String {
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

fn reference_stem(params: &Value) -> Result<String> {
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

    #[tokio::test]
    async fn archived_results_keep_metadata_and_source_alpha_and_publish_only_final_reference_files()
    -> Result<()> {
        let directory = tempfile::tempdir()?;
        let root = directory.path();
        let original =
            image::RgbaImage::from_fn(8, 8, |x, y| image::Rgba([80, 90, 100, (x + y * 8) as u8]));
        let input = root.join("original.png");
        std::fs::write(&input, images::encode_png(&original, None)?)?;
        let studio = Studio::new(root.to_owned(), root.join("missing-python"))?;
        let metadata = json!({"seed":42,"lightning":"inference/models/Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors","params":{"steps":32}});
        let job = {
            let mut store = studio.store.lock();
            let clean = image::RgbaImage::from_pixel(4, 4, image::Rgba([120, 130, 140, 255]));
            let clean =
                store
                    .assets
                    .save(&clean, "clean", "clean.png", false, Some(metadata.clone()))?;
            let result = store.assets.save(
                &original,
                "result",
                "result.png",
                false,
                Some(metadata.clone()),
            )?;
            json!({"seed":42,"transformer":"inference/models/Qwen-Image-2512-Q4_K_M.gguf","lightning":metadata["lightning"],"runtime":{"device":"cuda:0","precision":"bfloat16","offload":"sequential"},"clean":clean,"result":result})
        };
        let archive_directory = root.join("out/runs/test-run");
        std::fs::create_dir_all(&archive_directory)?;
        let archive = RunArchive {
            input: &input,
            directory: &archive_directory,
            stem: "original",
            run_id: "test-run",
        };
        let params = json!({"steps":32,"denoise":0.25,"passes":1,"restore_sigma":1.95,"restore":true,"restore_mode":"edge","unsharp":0.0,"keep_intermediate":true});
        let record = archive_result(&studio, &archive, 1, &params, &job)?;
        assert_eq!(record["transformer"], job["transformer"]);
        assert_eq!(record["lightning"], job["lightning"]);
        assert_eq!(record["runtime"]["device"], "cuda:0");
        assert!(
            !studio
                .store
                .lock()
                .assets
                .entries
                .contains_key(job["result"]["id"].as_str().unwrap())
        );
        assert!(
            !studio
                .store
                .lock()
                .assets
                .entries
                .contains_key(job["clean"]["id"].as_str().unwrap())
        );
        let result_path = Path::new(record["path"].as_str().context("Archived result missing")?);
        assert_eq!(
            images::png_metadata(&std::fs::read(result_path)?),
            Some(metadata)
        );
        let clean = images::open(&std::fs::read(
            record["clean_path"]
                .as_str()
                .context("Archived clean missing")?,
        )?)?;
        assert_eq!(clean.dimensions(), original.dimensions());
        assert!(
            clean
                .pixels()
                .zip(original.pixels())
                .all(|(clean, source)| clean.0[0..3] == [120, 130, 140] && clean[3] == source[3])
        );
        assert!(
            root.join("out/Original_desynth_s32_d0.250_p1_r1.95_edge.png")
                .is_file()
        );
        assert!(
            root.join("out/comparison_Original_desynth_s32_d0.250_p1_r1.95_edge.png")
                .is_file()
        );
        assert_eq!(
            std::fs::read_dir(root.join("out"))?
                .filter_map(Result::ok)
                .filter(|entry| entry.path().is_file())
                .count(),
            2
        );
        studio.close().await;
        Ok(())
    }

    #[tokio::test]
    async fn reference_aliases_cannot_overwrite_original_and_long_model_names_are_bounded()
    -> Result<()> {
        let directory = tempfile::tempdir()?;
        let root = directory.path();
        let original = image::RgbaImage::from_pixel(8, 8, image::Rgba([80, 90, 100, 17]));
        let input = root.join("original.png");
        let original_bytes = images::encode_png(&original, None)?;
        std::fs::write(&input, &original_bytes)?;
        let studio = Studio::new(root.to_owned(), root.join("missing-python"))?;
        let job = {
            let mut store = studio.store.lock();
            let clean = store
                .assets
                .save(&original, "clean", "clean.png", false, None)?;
            let result = store
                .assets
                .save(&original, "result", "result.png", false, None)?;
            json!({"seed":42,"transformer":format!("custom-{}.gguf", "x".repeat(230)),"clean":clean,"result":result})
        };
        let archive_directory = root.join("out/runs/long-name");
        std::fs::create_dir_all(&archive_directory)?;
        let archive = RunArchive {
            input: &input,
            directory: &archive_directory,
            stem: "original",
            run_id: "long-name",
        };
        let params = json!({"steps":32,"denoise":0.25,"passes":1,"restore_sigma":1.95,"restore":true,"restore_mode":"edge","unsharp":0.0,"keep_intermediate":false});
        let alias = root.join("out/Original_desynth_s32_d0.250_p1_r1.95_edge.png");
        std::fs::hard_link(&input, &alias)?;
        let error = archive_result(&studio, &archive, 1, &params, &job).unwrap_err();
        assert!(error.to_string().contains("must not overwrite"));
        assert_eq!(std::fs::read(&input)?, original_bytes);
        assert_eq!(std::fs::read_dir(&archive_directory)?.count(), 0);
        std::fs::remove_file(alias)?;
        let record = archive_result(&studio, &archive, 1, &params, &job)?;
        let path = Path::new(record["path"].as_str().context("Archived result missing")?);
        assert!(path.is_file());
        assert!(path.file_name().unwrap().to_string_lossy().len() < 200);
        assert_eq!(std::fs::read(input)?, original_bytes);
        studio.close().await;
        Ok(())
    }
}
