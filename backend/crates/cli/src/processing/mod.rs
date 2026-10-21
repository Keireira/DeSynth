mod archive;
mod progress;
mod selection;

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use desynth_core::{parameters, studio::Studio};
use desynth_models::catalog;
use rand::Rng;
use serde_json::json;

use crate::{RunArgs, rooted};
use archive::{RunArchive, archive_result};
use progress::{wait_for_job, wait_for_probe};
use selection::resolve_model;

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
        crate::files::write(&manifest_path, &serde_json::to_vec_pretty(&manifest)?)?;
    }
    println!("Run metadata: {}", manifest_path.display());
    Ok(())
}
