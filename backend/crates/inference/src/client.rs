use crate::protocol::{InferenceError, InferenceProgress, SampleRequest, WorkerCommand};
use crate::supervisor::supervise;
use serde_json::{Value, json};
use std::path::PathBuf;
use tokio::sync::{Mutex, mpsc, oneshot, watch};
use tokio::task::JoinHandle;

pub struct Inference {
    commands: mpsc::Sender<WorkerCommand>,
    shutdown: watch::Sender<bool>,
    task: Mutex<Option<JoinHandle<()>>>,
}

impl Inference {
    /// Starts the supervisor, but defers Python startup until a command needs it.
    pub fn new(root: PathBuf, python: PathBuf) -> Self {
        let (commands, receiver) = mpsc::channel(8);
        let (shutdown, stop) = watch::channel(false);
        let task = tokio::spawn(supervise(root, python, receiver, stop));
        Self {
            commands,
            shutdown,
            task: Mutex::new(Some(task)),
        }
    }

    async fn request(
        &self,
        payload: Value,
        progress: Option<mpsc::UnboundedSender<InferenceProgress>>,
    ) -> Result<Value, InferenceError> {
        let (response, receiver) = oneshot::channel();
        self.commands
            .send(WorkerCommand {
                payload,
                progress,
                response,
            })
            .await
            .map_err(|_| InferenceError::transport("Inference worker is stopped"))?;
        receiver
            .await
            .map_err(|_| InferenceError::transport("Inference worker stopped before replying"))?
    }

    pub async fn probe(&self) -> Result<Value, InferenceError> {
        self.request(json!({"op": "probe"}), None).await
    }

    pub async fn prepare(&self) -> Result<Value, InferenceError> {
        self.request(json!({"op": "prepare"}), None).await
    }

    pub async fn doctor(&self) -> Result<Value, InferenceError> {
        self.request(json!({"op": "doctor"}), None).await
    }

    pub async fn sample(
        &self,
        request: SampleRequest,
        progress: mpsc::UnboundedSender<InferenceProgress>,
    ) -> Result<Value, InferenceError> {
        let mut payload = serde_json::to_value(request).map_err(|error| {
            InferenceError::transport(format!("Invalid inference request: {error}"))
        })?;
        payload["op"] = json!("sample");
        self.request(payload, Some(progress)).await
    }

    pub async fn release(&self) -> Result<(), InferenceError> {
        self.request(json!({"op": "release"}), None)
            .await
            .map(|_| ())
    }

    /// Interrupts active sampling and reaps the subprocess before returning.
    pub async fn shutdown(&self) -> Result<(), InferenceError> {
        self.shutdown.send_replace(true);
        // Keep ownership of the handle until cleanup finishes. Another caller
        // must wait too, and canceling this future must not detach the worker.
        let mut task = self.task.lock().await;
        let result = if let Some(active) = task.as_mut() {
            active.await.map_err(|error| {
                InferenceError::transport(format!("Inference supervisor failed: {error}"))
            })
        } else {
            Ok(())
        };
        task.take();
        result
    }
}

impl Drop for Inference {
    fn drop(&mut self) {
        self.shutdown.send_replace(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    fn delayed_shutdown() -> (Arc<Inference>, oneshot::Sender<()>) {
        let (commands, _receiver) = mpsc::channel(1);
        let (shutdown, mut stop) = watch::channel(false);
        let (finish, finished) = oneshot::channel();
        let task = tokio::spawn(async move {
            let _ = stop.changed().await;
            let _ = finished.await;
        });
        (
            Arc::new(Inference {
                commands,
                shutdown,
                task: Mutex::new(Some(task)),
            }),
            finish,
        )
    }

    #[tokio::test]
    async fn concurrent_shutdown_callers_wait_for_the_same_cleanup() {
        let (worker, finish) = delayed_shutdown();
        let first_owner = worker.clone();
        let first = tokio::spawn(async move { first_owner.shutdown().await });
        while !*worker.shutdown.borrow() {
            tokio::task::yield_now().await;
        }
        let second_owner = worker.clone();
        let mut second = tokio::spawn(async move { second_owner.shutdown().await });
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut second)
                .await
                .is_err()
        );
        finish.send(()).unwrap();
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn canceling_a_shutdown_wait_keeps_cleanup_joinable() {
        let (worker, finish) = delayed_shutdown();
        let first_owner = worker.clone();
        let first = tokio::spawn(async move { first_owner.shutdown().await });
        while !*worker.shutdown.borrow() {
            tokio::task::yield_now().await;
        }
        first.abort();
        let _ = first.await;
        let next_owner = worker.clone();
        let mut next = tokio::spawn(async move { next_owner.shutdown().await });
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut next)
                .await
                .is_err()
        );
        finish.send(()).unwrap();
        next.await.unwrap().unwrap();
        worker.shutdown().await.unwrap();
    }
}
