//! Serialized stdio access to the only Python process: PyTorch/Diffusers inference.

use serde::Serialize;
use serde_json::{Value, json};
use std::fmt;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, mpsc, oneshot, watch};
use tokio::task::JoinHandle;

const MAX_FRAME_BYTES: usize = 65536;

#[derive(Debug)]
pub struct InferenceError {
    pub message: String,
    pub oom: bool,
}

impl InferenceError {
    fn transport(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            oom: false,
        }
    }
}

impl fmt::Display for InferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for InferenceError {}

#[derive(Debug, Clone)]
pub struct InferenceProgress {
    pub progress: f64,
    pub stage: String,
}

#[derive(Debug, Serialize)]
pub struct SampleRequest {
    pub transformer: PathBuf,
    pub lightning: PathBuf,
    pub embeds: PathBuf,
    pub source: PathBuf,
    pub output: PathBuf,
    pub denoise: f64,
    pub steps: u32,
    pub passes: u32,
    pub seed: u64,
    pub device: String,
    pub offload: String,
    pub precision: String,
    pub lightning_scale: f64,
}

struct WorkerCommand {
    payload: Value,
    progress: Option<mpsc::UnboundedSender<InferenceProgress>>,
    response: oneshot::Sender<Result<Value, InferenceError>>,
}

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

struct Session {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Session {
    fn start(root: &PathBuf, python: &PathBuf) -> Result<Self, InferenceError> {
        let mut child = Command::new(python)
            .arg("-u")
            .arg(root.join("inference/worker.py"))
            .current_dir(root)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| {
                InferenceError::transport(format!(
                    "Cannot start Python inference worker ({}): {error}. Run make install.",
                    python.display()
                ))
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| InferenceError::transport("Inference worker stdin was not created"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| InferenceError::transport("Inference worker stdout was not created"))?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    async fn execute(&mut self, id: u64, command: &WorkerCommand) -> Result<Value, ExecutionError> {
        let mut payload = command.payload.clone();
        payload["id"] = json!(id);
        let mut bytes = serde_json::to_vec(&payload).map_err(ExecutionError::protocol)?;
        bytes.push(b'\n');
        if bytes.len() > MAX_FRAME_BYTES {
            return Err(ExecutionError {
                error: InferenceError::transport("Inference request exceeds 64 KiB"),
                restart: false,
            });
        }
        self.stdin
            .write_all(&bytes)
            .await
            .map_err(ExecutionError::protocol)?;
        self.stdin.flush().await.map_err(ExecutionError::protocol)?;
        loop {
            let event: Value = serde_json::from_slice(&self.read_frame().await?)
                .map_err(ExecutionError::protocol)?;
            if event["id"].as_u64() != Some(id) {
                return Err(ExecutionError::protocol(
                    "Inference response has an unexpected command id",
                ));
            }
            match event["event"].as_str() {
                Some("progress") => {
                    let progress = event["progress"]
                        .as_f64()
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| {
                            ExecutionError::protocol("Inference progress must be finite")
                        })?;
                    let stage = event["stage"].as_str().ok_or_else(|| {
                        ExecutionError::protocol("Inference progress is missing its stage")
                    })?;
                    if let Some(sender) = &command.progress {
                        // A disconnected UI/job receiver must not break the worker protocol.
                        let _ = sender.send(InferenceProgress {
                            progress,
                            stage: stage.to_owned(),
                        });
                    }
                }
                Some("result") => {
                    return event.get("result").cloned().ok_or_else(|| {
                        ExecutionError::protocol("Inference response is missing its result")
                    });
                }
                Some("error") => {
                    return Err(ExecutionError {
                        error: InferenceError {
                            message: event["error"]
                                .as_str()
                                .unwrap_or("Inference failed")
                                .to_owned(),
                            oom: event["oom"].as_bool().unwrap_or(false),
                        },
                        restart: false,
                    });
                }
                _ => return Err(ExecutionError::protocol("Unknown inference response event")),
            }
        }
    }

    async fn read_frame(&mut self) -> Result<Vec<u8>, ExecutionError> {
        let mut frame = Vec::new();
        loop {
            let buffered = self
                .stdout
                .fill_buf()
                .await
                .map_err(ExecutionError::protocol)?;
            if buffered.is_empty() {
                return Err(ExecutionError::protocol(
                    "Inference subprocess closed its output before a complete frame",
                ));
            }
            let end = buffered.iter().position(|byte| *byte == b'\n');
            let count = end.map_or(buffered.len(), |position| position + 1);
            if frame.len() + count > MAX_FRAME_BYTES {
                return Err(ExecutionError::protocol(
                    "Inference response exceeds 64 KiB",
                ));
            }
            frame.extend_from_slice(&buffered[..count]);
            self.stdout.consume(count);
            if end.is_some() {
                return Ok(frame);
            }
        }
    }

    async fn stop(mut self) {
        // kill().await also waits/reaps, including when a Torch call is blocked.
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
    }
}

struct ExecutionError {
    error: InferenceError,
    restart: bool,
}

impl ExecutionError {
    fn protocol(error: impl fmt::Display) -> Self {
        Self {
            error: InferenceError::transport(format!("Inference worker protocol failed: {error}")),
            restart: true,
        }
    }
}

async fn supervise(
    root: PathBuf,
    python: PathBuf,
    mut commands: mpsc::Receiver<WorkerCommand>,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut session: Option<Session> = None;
    let mut identifier = 0_u64;
    loop {
        let command = tokio::select! {
            biased;
            _ = shutdown.changed() => break,
            command = commands.recv() => match command { Some(command) => command, None => break },
        };
        if command.response.is_closed() {
            continue;
        }
        if session.is_none() && command.payload["op"] == "release" {
            let _ = command.response.send(Ok(json!({"released": true})));
            continue;
        }
        if session.is_none() {
            match Session::start(&root, &python) {
                Ok(started) => session = Some(started),
                Err(error) => {
                    let _ = command.response.send(Err(error));
                    continue;
                }
            }
        }
        identifier = identifier.wrapping_add(1);
        let Some(active) = session.as_mut() else {
            break;
        };
        let execution = tokio::select! {
            biased;
            _ = shutdown.changed() => None,
            result = active.execute(identifier, &command) => Some(result),
        };
        match execution {
            Some(Ok(result)) => {
                let _ = command.response.send(Ok(result));
            }
            Some(Err(failure)) => {
                if failure.restart
                    && let Some(failed) = session.take()
                {
                    failed.stop().await;
                }
                let _ = command.response.send(Err(failure.error));
            }
            None => {
                let _ = command.response.send(Err(InferenceError::transport(
                    "Inference interrupted by server shutdown",
                )));
                break;
            }
        }
    }
    if let Some(session) = session {
        session.stop().await;
    }
    commands.close();
    while let Some(command) = commands.recv().await {
        let _ = command.response.send(Err(InferenceError::transport(
            "Inference worker is stopped",
        )));
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
