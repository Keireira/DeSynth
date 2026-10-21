use super::Studio;
use crate::error::{ServiceError, ServiceResult};
use desynth_models::catalog::{self, Catalog};
use serde_json::{Value, json};
use std::sync::Arc;

impl Studio {
    pub fn state(&self) -> Value {
        let store = self.store.lock();
        let mut missing = store
            .probe
            .get("missing")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_else(|| vec![json!("Python inference worker")]);
        if !catalog::nonempty_file(&catalog::embeddings_path(&self.root)) {
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
        json!({
            "ready": missing.is_empty() && store.probe["ready"] == true,
            "missing": missing,
            "device": store.probe["device"],
            "models": store.catalog.models,
            "lightning_adapters": store.catalog.lightning_adapters,
            "default_lightning": store.catalog.default_lightning(),
            "default_steps": store.catalog.default_steps(),
            "runtime": runtime,
            "assets": store.assets.entries.values().map(|entry| &entry.asset).collect::<Vec<_>>(),
            "default_source_id": store.assets.default_source,
            "default_result_id": store.assets.default_result,
            "active_job_id": store.active
        })
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
}

pub(super) fn default_probe() -> Value {
    json!({
        "ready": false,
        "missing": ["Python inference worker"],
        "device": "CPU",
        "runtime": {
            "ready": false,
            "default_device": "cpu",
            "devices": [{
                "id": "cpu",
                "name": "CPU",
                "type": "cpu",
                "precisions": ["float32"],
                "offload_modes": ["none"]
            }]
        }
    })
}
