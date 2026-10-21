use crate::{
    assets::{Asset, Assets, StoredAsset},
    catalog::Catalog,
    error::{ServiceError, ServiceResult},
    images,
    inference::{Inference, SampleRequest},
    parameters::{self, ProcessParams},
    restoration::{RestoreClient, RestoreTicket},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use indexmap::IndexMap;
use parking_lot::Mutex;
use rand::Rng;
use serde_json::{Value, json};
use std::{collections::HashSet, path::PathBuf, sync::Arc, time::Instant};
use tokio::sync::{Notify, Semaphore, mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use uuid::Uuid;

pub struct Store {
    pub assets: Assets,
    pub catalog: Catalog,
    pub jobs: IndexMap<String, Value>,
    pub started: IndexMap<String, Instant>,
    pub active: Option<String>,
    pub latest_job: Option<String>,
    pub probe: Value,
    pub refreshing: bool,
    pub restore_clients: IndexMap<String, RestoreClient>,
    pub restore_pending: IndexMap<String, RestoreTicket>,
    pub restore_running: Option<(String, u64)>,
}

pub struct Studio {
    pub root: PathBuf,
    pub store: Mutex<Store>,
    pub inference: Inference,
    pub cpu: Arc<Semaphore>,
    pub restore_notify: Notify,
    pub shutdown: watch::Sender<bool>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
    closing: tokio::sync::Mutex<()>,
}

impl Studio {
    pub fn new(root: PathBuf, python: PathBuf) -> anyhow::Result<Arc<Self>> {
        let catalog = Catalog::scan(&root)?;
        let assets = Assets::new(&root)?;
        let (shutdown, _) = watch::channel(false);
        let studio = Arc::new(Self {
            inference: Inference::new(root.clone(), python),
            root,
            store: Mutex::new(Store {
                assets,
                catalog,
                jobs: IndexMap::new(),
                started: IndexMap::new(),
                active: None,
                latest_job: None,
                probe: default_probe(),
                refreshing: false,
                restore_clients: IndexMap::new(),
                restore_pending: IndexMap::new(),
                restore_running: None,
            }),
            cpu: Arc::new(Semaphore::new(2)),
            restore_notify: Notify::new(),
            shutdown,
            tasks: Mutex::new(Vec::new()),
            closing: tokio::sync::Mutex::new(()),
        });
        studio
            .tasks
            .lock()
            .push(tokio::spawn(studio.clone().restore_loop()));
        Ok(studio)
    }

    pub fn state(&self) -> Value {
        let store = self.store.lock();
        let mut missing = store
            .probe
            .get("missing")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_else(|| vec![json!("Python inference worker")]);
        if !crate::catalog::nonempty_file(&crate::catalog::embeddings_path(&self.root)) {
            missing.push(json!("embeds_cache.pt"));
        }
        if store.catalog.models.is_empty() {
            missing.push(json!("GGUF transformer"));
        }
        if store.catalog.lightning_adapters.is_empty() {
            missing.push(json!("Lightning adapter"));
        }
        let mut runtime = store
            .probe
            .get("runtime")
            .cloned()
            .unwrap_or_else(|| default_probe()["runtime"].clone());
        runtime["refreshing"] = json!(store.refreshing);
        json!({"ready":missing.is_empty() && store.probe["ready"] == true,"missing":missing,"device":store.probe["device"],"models":store.catalog.models,"lightning_adapters":store.catalog.lightning_adapters,"default_lightning":store.catalog.default_lightning(),"default_steps":store.catalog.default_steps(),"runtime":runtime,"assets":store.assets.entries.values().map(|entry| &entry.asset).collect::<Vec<_>>(),"default_source_id":store.assets.default_source,"default_result_id":store.assets.default_result,"active_job_id":store.active})
    }

    pub fn refresh(self: &Arc<Self>) -> ServiceResult<Value> {
        let mut tasks = self.tasks.lock();
        if *self.shutdown.borrow() {
            return Err(ServiceError::unavailable("The server is stopping."));
        }
        let catalog = Catalog::scan(&self.root)?;
        let should_probe = {
            let mut store = self.store.lock();
            store.catalog = catalog;
            if store.active.is_some() || store.refreshing {
                false
            } else {
                store.refreshing = true;
                true
            }
        };
        if should_probe {
            let studio = self.clone();
            tasks.retain(|task| !task.is_finished());
            tasks.push(tokio::spawn(async move {
                let probe = match studio.inference.probe().await {
                    Ok(value) => value,
                    Err(error) => {
                        let mut fallback = default_probe();
                        fallback["runtime"]["ready"] = json!(true);
                        fallback["runtime"]["error"] = json!(error.message);
                        fallback
                    }
                };
                let mut store = studio.store.lock();
                store.probe = probe;
                store.refreshing = false;
            }));
        }
        Ok(self.state())
    }

    pub async fn image(&self, id: &str) -> ServiceResult<(StoredAsset, image::RgbaImage)> {
        let asset = self.asset(id)?;
        let path = asset.path.clone();
        let image = self
            .cpu_task(move || Ok(images::open(&std::fs::read(path)?)?))
            .await?;
        Ok((asset, image))
    }

    pub(crate) fn asset(&self, id: &str) -> ServiceResult<StoredAsset> {
        self.store
            .lock()
            .assets
            .get(id, None)
            .map_err(|error| ServiceError::missing(error.to_string()))
    }

    /// Keep decoding, transforms and encoding under the same permit to bound image memory.
    pub async fn cpu_task<T, F>(&self, operation: F) -> ServiceResult<T>
    where
        T: Send + 'static,
        F: FnOnce() -> ServiceResult<T> + Send + 'static,
    {
        if *self.shutdown.borrow() {
            return Err(ServiceError::unavailable("The server is stopping."));
        }
        let permit = self
            .cpu
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| ServiceError::unavailable("The server is stopping."))?;
        let (response, receiver) = oneshot::channel();
        {
            let mut tasks = self.tasks.lock();
            if *self.shutdown.borrow() {
                return Err(ServiceError::unavailable("The server is stopping."));
            }
            let work = tokio::task::spawn_blocking(move || {
                let _permit = permit;
                operation()
            });
            tasks.retain(|task| !task.is_finished());
            tasks.push(tokio::spawn(async move {
                let result = work
                    .await
                    .map_err(|error| anyhow::anyhow!(error).into())
                    .and_then(|result| result);
                let _ = response.send(result);
            }));
        }
        receiver
            .await
            .map_err(|_| ServiceError::unavailable("The server is stopping."))?
    }

    pub async fn image_bytes(&self, id: &str, raw: bool) -> ServiceResult<Vec<u8>> {
        let asset = self.asset(id)?;
        self.cpu_task(move || {
            let image = images::open(&std::fs::read(asset.path)?)?;
            if raw {
                return Ok(image.into_raw());
            }
            Ok(images::encode_png(&image, asset.metadata.as_ref())?)
        })
        .await
    }

    pub async fn upload(&self, payload: Value) -> ServiceResult<Asset> {
        let role = match payload.get("role") {
            None => "source",
            Some(value) => value
                .as_str()
                .ok_or_else(|| ServiceError::bad("role must be source or result."))?,
        };
        if !matches!(role, "source" | "result") {
            return Err(ServiceError::bad("role must be source or result."));
        }
        let data = payload
            .get("data")
            .and_then(Value::as_str)
            .ok_or_else(|| ServiceError::bad("Image data is missing."))?;
        let encoded = if data.starts_with("data:") {
            data.split_once(";base64,")
                .ok_or_else(|| ServiceError::bad("A base64-encoded image is required."))?
                .1
        } else {
            data
        };
        if encoded.len() > 45 * 1024 * 1024 {
            return Err(ServiceError {
                status: 413,
                message: "Maximum file size: 32 MB.".to_owned(),
            });
        }
        let name = match payload.get("name") {
            None => "image.png",
            Some(value) => value
                .as_str()
                .ok_or_else(|| ServiceError::bad("name must be a string."))?,
        };
        let normalized = name.replace('\\', "/");
        let name = normalized.rsplit('/').next().unwrap_or("image.png");
        let name: String = name.chars().take(160).collect();
        let name = if name.is_empty() {
            "image.png".to_owned()
        } else {
            name
        };
        let encoded = encoded.to_owned();
        let role = role.to_owned();
        let directory = self.root.join("out/gui");
        drop(payload);
        let prepared = self
            .cpu_task(move || {
                let bytes = STANDARD
                    .decode(encoded)
                    .map_err(|_| ServiceError::bad("Invalid base64 image data."))?;
                if bytes.len() > 32 * 1024 * 1024 {
                    return Err(ServiceError {
                        status: 413,
                        message: "Maximum file size: 32 MB.".to_owned(),
                    });
                }
                let image = images::open(&bytes)
                    .map_err(|error| ServiceError::bad(format!("Cannot read image: {error}")))?;
                Ok(Assets::prepare(
                    &directory, &image, &role, &name, true, None,
                )?)
            })
            .await?;
        let mut store = self.store.lock();
        if *self.shutdown.borrow() {
            return Err(ServiceError::unavailable("The server is stopping."));
        }
        let asset = store.assets.insert(prepared.commit());
        if asset.kind == "source" {
            store.assets.latest_source = Some(asset.id.clone());
        }
        prune(&mut store, [asset.id.clone()]);
        Ok(asset)
    }

    pub async fn compare(&self, source: &str, result: &str) -> ServiceResult<Value> {
        let source = self.asset(source)?;
        let result = self.asset(result)?;
        self.cpu_task(move || {
            let source = images::open(&std::fs::read(source.path)?)?;
            let result = images::open(&std::fs::read(result.path)?)?;
            images::metrics(&source, &result).map_err(|error| ServiceError::bad(error.to_string()))
        })
        .await
    }

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
                || !crate::catalog::nonempty_file(&crate::catalog::embeddings_path(&self.root))
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

    async fn run_job(
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
                embeds: crate::catalog::embeddings_path(&self.root),
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

    pub fn set_progress(&self, id: &str, progress: f64, stage: &str) {
        if let Some(job) = self.store.lock().jobs.get_mut(id) {
            job["progress"] = json!(progress);
            job["stage"] = json!(stage);
        }
    }

    pub async fn close(&self) {
        let _closing = self.closing.lock().await;
        self.shutdown.send_replace(true);
        self.cpu.close();
        self.restore_notify.notify_waiters();
        let _ = self.inference.shutdown().await;
        let tasks = std::mem::take(&mut *self.tasks.lock());
        for task in tasks {
            if let Err(error) = task.await {
                tracing::error!(%error, "A server task failed while stopping");
            }
        }
        self.store.lock().assets.cleanup();
    }
}

pub fn prune(store: &mut Store, extra: impl IntoIterator<Item = String>) {
    let mut protected: HashSet<String> = extra.into_iter().collect();
    for id in [&store.active, &store.latest_job].into_iter().flatten() {
        if let Some(job) = store.jobs.get(id) {
            if let Some(source) = job["source_id"].as_str() {
                protected.insert(source.to_owned());
            }
            for role in ["clean", "result"] {
                if let Some(id) = job[role]["id"].as_str() {
                    protected.insert(id.to_owned());
                }
            }
        }
    }
    for client in store.restore_clients.values() {
        protected.extend(client.assets.iter().cloned());
    }
    store.assets.prune(&protected);
}

fn default_probe() -> Value {
    json!({"ready":false,"missing":["Python inference worker"],"device":"CPU","runtime":{"ready":false,"default_device":"cpu","devices":[{"id":"cpu","name":"CPU","type":"cpu","precisions":["float32"],"offload_modes":["none"]}]}})
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
