use crate::{
    assets::{Assets, PreparedAsset},
    error::{ServiceError, ServiceResult},
    parameters,
    studio::{Store, Studio, prune},
};
use desynth_images as images;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::oneshot;

pub struct RestoreClient {
    pub revision: u64,
    pub assets: Vec<String>,
    pub result: Option<String>,
}
pub struct RestoreTicket {
    client: String,
    revision: u64,
    source: String,
    clean: String,
    params: parameters::RestoreParams,
    response: oneshot::Sender<ServiceResult<Value>>,
}

fn identity(payload: &Value) -> ServiceResult<(String, u64)> {
    let client = payload
        .get("client_id")
        .and_then(Value::as_str)
        .ok_or_else(|| ServiceError::bad("client_id is required."))?;
    if client.is_empty() || client.len() > 128 {
        return Err(ServiceError::bad("Invalid restore client."));
    }
    let revision = payload
        .get("revision")
        .and_then(Value::as_u64)
        .ok_or_else(|| ServiceError::bad("A nonnegative revision is required."))?;
    Ok((client.to_owned(), revision))
}

/// Advance a validated request under the same lock as queue registration.
fn advance_revision(store: &mut Store, client: &str, revision: u64) -> ServiceResult<u64> {
    if !store.restore_clients.contains_key(client) {
        if store.restore_clients.len() >= 8 {
            let old = store
                .restore_clients
                .keys()
                .find(|id| {
                    !store.restore_pending.contains_key(*id)
                        && store
                            .restore_running
                            .as_ref()
                            .is_none_or(|(running, _)| running != *id)
                })
                .cloned()
                .ok_or_else(|| ServiceError::busy("Too many restoration clients."))?;
            store.restore_clients.shift_remove(&old);
        }
        store.restore_clients.insert(
            client.to_owned(),
            RestoreClient {
                revision,
                assets: Vec::new(),
                result: None,
            },
        );
    }
    let current = store
        .restore_clients
        .get_mut(client)
        .ok_or_else(|| ServiceError::bad("Invalid restore client."))?;
    current.revision = current.revision.max(revision);
    let current_revision = current.revision;
    if store
        .restore_pending
        .get(client)
        .is_some_and(|ticket| ticket.revision < current_revision)
        && let Some(ticket) = store.restore_pending.shift_remove(client)
    {
        let _ = ticket.response.send(Ok(json!({"superseded":true})));
    }
    Ok(current_revision)
}

impl Studio {
    pub fn cancel_restore(&self, payload: Value) -> ServiceResult<Value> {
        let (client, revision) = identity(&payload)?;
        let mut store = self.store.lock();
        if *self.shutdown.borrow() {
            return Err(ServiceError::unavailable("The server is stopping."));
        }
        let revision = advance_revision(&mut store, &client, revision)?;
        Ok(json!({"revision":revision}))
    }

    pub async fn quick_restore(&self, payload: Value) -> ServiceResult<Value> {
        let (client, revision) = identity(&payload)?;
        let params = parameters::parse_restore(payload.get("params").unwrap_or(&json!({})))
            .map_err(ServiceError::bad)?;
        let source = payload
            .get("source_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ServiceError::bad("source_id is required."))?
            .to_owned();
        let clean = payload
            .get("clean_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ServiceError::bad("clean_id is required."))?
            .to_owned();
        let (response, receiver) = oneshot::channel();
        {
            let mut store = self.store.lock();
            if *self.shutdown.borrow() {
                return Err(ServiceError::unavailable("The server is stopping."));
            }
            store
                .assets
                .get(&source, Some("source"))
                .map_err(|error| ServiceError::bad(error.to_string()))?;
            store
                .assets
                .get(&clean, Some("clean"))
                .map_err(|error| ServiceError::bad(error.to_string()))?;
            if store.assets.clean_sources.get(&clean) != Some(&source) {
                return Err(ServiceError::bad(
                    "This clean result belongs to a different source image.",
                ));
            }
            let current_revision = advance_revision(&mut store, &client, revision)?;
            if revision < current_revision {
                return Ok(json!({"superseded":true}));
            }
            let current = store
                .restore_clients
                .get_mut(&client)
                .ok_or_else(|| ServiceError::bad("Invalid restore client."))?;
            current.assets = vec![source.clone(), clean.clone()];
            if store
                .restore_running
                .as_ref()
                .is_some_and(|running| running == &(client.clone(), revision))
            {
                return Ok(json!({"superseded":true}));
            }
            if let Some(previous) = store.restore_pending.insert(
                client.clone(),
                RestoreTicket {
                    client,
                    revision,
                    source,
                    clean,
                    params,
                    response,
                },
            ) {
                let _ = previous.response.send(Ok(json!({"superseded":true})));
            }
        }
        self.restore_notify.notify_one();
        tokio::time::timeout(std::time::Duration::from_secs(60), receiver)
            .await
            .map_err(|_| ServiceError::unavailable("Restoration timed out. Please try again."))?
            .map_err(|_| ServiceError::unavailable("The server is stopping."))?
    }

    pub async fn restore_loop(self: Arc<Self>) {
        let mut shutdown = self.shutdown.subscribe();
        loop {
            if *shutdown.borrow() {
                break;
            }
            let ticket = {
                let mut store = self.store.lock();
                let ticket = store
                    .restore_pending
                    .shift_remove_index(0)
                    .map(|(_, ticket)| ticket);
                if let Some(ticket) = &ticket {
                    store.restore_running = Some((ticket.client.clone(), ticket.revision));
                }
                ticket
            };
            let Some(ticket) = ticket else {
                tokio::select! { _ = self.restore_notify.notified() => {}, _ = shutdown.changed() => break }
                continue;
            };
            let result = self.calculate_restoration(&ticket).await;
            let mut store = self.store.lock();
            store.restore_running = None;
            let current = store
                .restore_clients
                .get(&ticket.client)
                .is_some_and(|client| client.revision == ticket.revision);
            let response = if !current || *shutdown.borrow() {
                Ok(json!({"superseded":true}))
            } else {
                result.map(|(prepared, metrics)| {
                    let result = store.assets.insert(prepared.commit());
                    if let Some(client) = store.restore_clients.get_mut(&ticket.client) {
                        client.result = Some(result.id.clone());
                    }
                    prune(&mut store, [result.id.clone()]);
                    json!({"result":result,"metrics":metrics,"revision":ticket.revision})
                })
            };
            let _ = ticket.response.send(response);
            if *shutdown.borrow() {
                break;
            }
        }
        let mut store = self.store.lock();
        for (_, ticket) in std::mem::take(&mut store.restore_pending) {
            let _ = ticket.response.send(Ok(json!({"superseded":true})));
        }
    }

    async fn calculate_restoration(
        &self,
        ticket: &RestoreTicket,
    ) -> ServiceResult<(PreparedAsset, Value)> {
        let source_asset = self.asset(&ticket.source)?;
        let clean_asset = self.asset(&ticket.clean)?;
        let params = ticket.params.clone();
        let mut metadata = clean_asset
            .metadata
            .clone()
            .unwrap_or_else(|| json!({"source":{"path":source_asset.path},"params":{}}));
        let directory = self.root.join("out/gui");
        self.cpu_task(move || {
            let source = source_asset.read_image()?;
            let clean = clean_asset.read_image()?;
            let image = images::finish(&clean, &source, &params)?;
            let metrics = images::metrics(&source, &image)?;
            let restoration = serde_json::to_value(&params).map_err(anyhow::Error::from)?;
            if let Some(record) = metadata.as_object_mut() {
                let saved_params = record.entry("params").or_insert_with(|| json!({}));
                if let (Some(saved), Some(restoration)) =
                    (saved_params.as_object_mut(), restoration.as_object())
                {
                    saved.extend(restoration.clone());
                }
            }
            let prepared = Assets::prepare(
                &directory,
                &image,
                "result",
                "desynth_restored.png",
                true,
                Some(metadata),
            )?;
            Ok((prepared, metrics))
        })
        .await
    }
}
