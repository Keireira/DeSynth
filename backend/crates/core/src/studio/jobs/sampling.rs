use super::Studio;
use crate::{
    assets::Assets,
    error::{ServiceError, ServiceResult},
    parameters::ProcessParams,
};
use desynth_images as images;
use desynth_inference::SampleRequest;
use desynth_models::catalog;
use serde_json::json;
use std::path::PathBuf;
use tokio::sync::mpsc;

impl Studio {
    pub(super) async fn run_job(
        &self,
        id: &str,
        source_id: &str,
        params: ProcessParams,
        candidates: Vec<String>,
    ) -> ServiceResult<()> {
        self.set_progress(id, 2.0, "Preparing image");
        {
            self.store
                .lock()
                .jobs
                .get_mut(id)
                .ok_or_else(|| ServiceError::missing("Job not found."))?["status"] =
                json!("running");
        }
        let (source_asset, original) = self.image(source_id).await?;
        let working_path = self.root.join("out/gui").join(format!("{id}-input.png"));
        let clean_directory = if params.keep_intermediate {
            "out/gui"
        } else {
            "out/gui/temporary"
        };
        let clean_path = self
            .root
            .join(clean_directory)
            .join(format!("{id}-clean.png"));
        let _files = WorkingFiles {
            input: working_path.clone(),
            clean: clean_path.clone(),
        };
        let preparation_path = working_path.clone();
        let preview = params.preview;
        let (original, working_size) = self
            .cpu_task(move || {
                let working = images::prepare(&original, preview);
                std::fs::write(preparation_path, images::encode_png(&working, None)?)?;
                Ok((original, [working.width(), working.height()]))
            })
            .await?;
        let (progress, mut receiver) = mpsc::unbounded_channel();
        let selected_adapter = params
            .lightning
            .as_ref()
            .ok_or_else(|| ServiceError::bad("No Lightning adapter selected."))?;
        let mut chosen_model = None;
        let mut failure = None;
        for model in candidates {
            let request = SampleRequest {
                transformer: self.root.join(&model),
                lightning: self.root.join(selected_adapter),
                embeds: catalog::embeddings_path(&self.root),
                source: working_path.clone(),
                output: clean_path.clone(),
                denoise: params.denoise,
                steps: params.steps,
                passes: params.passes,
                seed: params.seed.unwrap_or(0),
                device: params.device.clone(),
                offload: params.offload.clone(),
                precision: params.precision.clone(),
                lightning_scale: params.lightning_scale,
            };
            let sampling = self.inference.sample(request, progress.clone());
            tokio::pin!(sampling);
            let result = loop {
                tokio::select! {
                    result = &mut sampling => break result,
                    Some(event) = receiver.recv() => self.set_progress(id, event.progress.clamp(3.0, 90.0), &event.stage),
                }
            };
            match result {
                Ok(_) => {
                    chosen_model = Some(model);
                    break;
                }
                Err(error) => {
                    let oom = error.oom;
                    failure = Some(error.message);
                    if !oom {
                        break;
                    }
                    self.set_progress(
                        id,
                        3.0,
                        "Trying a smaller model after running out of memory",
                    );
                }
            }
        }
        let _ = std::fs::remove_file(&working_path);
        let model = chosen_model.ok_or_else(|| {
            ServiceError::bad(failure.unwrap_or_else(|| "No model is available.".to_owned()))
        })?;
        self.set_progress(id, 93.0, "Restoring details");
        let restore = params.restoration.clone();
        let finish_path = clean_path.clone();
        let metadata = json!({
            "schema_version":1, "sampling":"linear-sigma-v1", "starting_sigma":params.denoise,
            "source":{"id":source_id,"path":source_asset.path,"name":source_asset.asset.name},
            "transformer":model,"lightning":selected_adapter,"params":params,"seed":params.seed,
            "runtime":{"device":params.device,"offload":params.offload,"precision":params.precision},
            "working_size":working_size,
        });
        let result_name = format!(
            "{}_desynth_seed{}.png",
            source_asset
                .asset
                .name
                .rsplit_once('.')
                .map_or(source_asset.asset.name.as_str(), |(stem, _)| stem),
            params.seed.unwrap_or(0)
        );
        let temporary_clean = !params.keep_intermediate;
        let directory = self.root.join("out/gui");
        let (prepared_clean, prepared_result, metrics) = self
            .cpu_task(move || {
                let clean = images::open(&std::fs::read(finish_path)?)?;
                let final_image = images::finish(&clean, &original, &restore)?;
                let metrics = images::metrics(&original, &final_image)?;
                let prepared_clean = Assets::prepare(
                    &directory,
                    &clean,
                    "clean",
                    "desynth_clean.png",
                    temporary_clean,
                    Some(metadata.clone()),
                )?;
                let prepared_result = Assets::prepare(
                    &directory,
                    &final_image,
                    "result",
                    &result_name,
                    false,
                    Some(metadata),
                )?;
                Ok((prepared_clean, prepared_result, metrics))
            })
            .await?;
        let mut store = self.store.lock();
        if *self.shutdown.borrow() {
            return Err(ServiceError::unavailable("The server is stopping."));
        }
        let clean_asset = store.assets.insert(prepared_clean.commit());
        store
            .assets
            .clean_sources
            .insert(clean_asset.id.clone(), source_id.to_owned());
        let result = store.assets.insert(prepared_result.commit());
        let elapsed = store
            .started
            .get(id)
            .map_or(0.0, |started| started.elapsed().as_secs_f64());
        let job = store
            .jobs
            .get_mut(id)
            .ok_or_else(|| ServiceError::missing("Job not found."))?;
        job["status"] = json!("done");
        job["stage"] = json!("Done");
        job["progress"] = json!(100);
        job["elapsed"] = json!(elapsed);
        job["result"] = json!(result);
        job["clean"] = json!(clean_asset);
        job["metrics"] = metrics;
        job["transformer"] = json!(model);
        job["preview_size"] = json!(working_size);
        Ok(())
    }
}

/// Remove incomplete worker output when a job fails or the server is stopped.
struct WorkingFiles {
    input: PathBuf,
    clean: PathBuf,
}

impl Drop for WorkingFiles {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.input);
        let _ = std::fs::remove_file(&self.clean);
    }
}
