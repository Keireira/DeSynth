use super::{Store, Studio, inventory::default_probe};
use crate::{
    assets::Assets,
    error::{ServiceError, ServiceResult},
};
use desynth_inference::Inference;
use desynth_models::catalog::Catalog;
use indexmap::IndexMap;
use parking_lot::Mutex;
use std::{path::PathBuf, sync::Arc};
use tokio::sync::{Notify, Semaphore, oneshot, watch};

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
