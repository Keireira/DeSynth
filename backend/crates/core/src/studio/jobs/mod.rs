mod sampling;

use super::{Studio, prune};
use crate::{
    error::{ServiceError, ServiceResult},
    parameters,
};
use desynth_models::catalog;
use rand::Rng;
use serde_json::{Value, json};
use std::{sync::Arc, time::Instant};
use uuid::Uuid;

impl Studio {
    pub fn job(&self, id: &str) -> ServiceResult<Value> {
        let store = self.store.lock();
        let mut job = store
            .jobs
            .get(id)
            .cloned()
            .ok_or_else(|| ServiceError::missing("Job not found."))?;
        if matches!(job["status"].as_str(), Some("running" | "queued"))
            && let Some(started) = store.started.get(id)
        {
            job["elapsed"] = json!(started.elapsed().as_secs_f64());
        }
        Ok(job)
    }

    pub fn process(self: &Arc<Self>, payload: Value) -> ServiceResult<Value> {
        let mut tasks = self.tasks.lock();
        if *self.shutdown.borrow() {
            return Err(ServiceError::unavailable("The server is stopping."));
        }
        let source_id = payload
            .get("source_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ServiceError::bad("Open an original image first."))?
            .to_owned();
        let (id, params, candidates) = {
            let mut store = self.store.lock();
            store
                .assets
                .get(&source_id, Some("source"))
                .map_err(|error| ServiceError::bad(error.to_string()))?;
            if store.active.is_some() {
                return Err(ServiceError::busy(
                    "Processing is already running. Wait for the current job to finish.",
                ));
            }
            let raw = payload.get("params").cloned().unwrap_or_else(|| json!({}));
            let defaults = parameters::selected_default_steps(&raw, &store.catalog);
            let mut params =
                parameters::parse_process(&raw, defaults).map_err(ServiceError::bad)?;
            let runtime = parameters::resolve_runtime(&params, &store.probe["runtime"])
                .map_err(ServiceError::bad)?;
            if store.probe["ready"] != true
                || !catalog::nonempty_file(&catalog::embeddings_path(&self.root))
            {
                return Err(ServiceError::unavailable(
                    "Run make setup to enable processing.",
                ));
            }
            let candidates = parameters::select_models(&mut params, &store.catalog, &self.root)
                .map_err(ServiceError::bad)?;
            let requested_runtime = json!({
                "device": params.device,
                "offload": params.offload,
                "precision": params.precision,
            });
            params.device = runtime["device"].as_str().unwrap_or("cpu").to_owned();
            params.offload = runtime["offload"].as_str().unwrap_or("none").to_owned();
            params.precision = runtime["precision"]
                .as_str()
                .unwrap_or("float32")
                .to_owned();
            let seed = params
                .seed
                .unwrap_or_else(|| rand::rng().random_range(0..2_u64.pow(31)));
            params.seed = Some(seed);
            let id = Uuid::new_v4().simple().to_string();
            store.jobs.insert(
                id.clone(),
                json!({
                    "id":id, "source_id":source_id, "status":"queued", "stage":"Loading model",
                    "progress":0, "elapsed":0, "seed":seed, "runtime":runtime,
                    "requested_runtime":requested_runtime, "steps":params.steps,
                    "denoise":params.denoise, "starting_sigma":params.denoise,
                    "sampling":"linear-sigma-v1", "lightning":params.lightning,
                    "lightning_scale":params.lightning_scale,
                }),
            );
            store.started.insert(id.clone(), Instant::now());
            store.active = Some(id.clone());
            store.latest_job = Some(id.clone());
            while store.jobs.len() > 16 {
                if let Some((old_id, _)) = store.jobs.shift_remove_index(0) {
                    store.started.shift_remove(&old_id);
                }
            }
            (id, params, candidates)
        };
        let studio = self.clone();
        let job_id = id.clone();
        tasks.retain(|task| !task.is_finished());
        tasks.push(tokio::spawn(async move {
            let result = studio
                .run_job(&job_id, &source_id, params, candidates)
                .await;
            let mut store = studio.store.lock();
            if let Err(error) = result {
                tracing::error!(%error, job_id, "Processing failed");
                if let Some(job) = store.jobs.get_mut(&job_id) {
                    job["status"] = json!("error");
                    job["error"] = json!(error.to_string());
                    job["stage"] = json!("Failed");
                }
            }
            if store.active.as_deref() == Some(&job_id) {
                store.active = None;
            }
            prune(&mut store, []);
        }));
        self.job(&id)
    }

    pub fn set_progress(&self, id: &str, progress: f64, stage: &str) {
        if let Some(job) = self.store.lock().jobs.get_mut(id) {
            job["progress"] = json!(progress);
            job["stage"] = json!(stage);
        }
    }
}
