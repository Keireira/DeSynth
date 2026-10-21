use serde::Serialize;
use serde_json::Value;
use std::fmt;
use std::path::PathBuf;
use tokio::sync::{mpsc, oneshot};

#[derive(Debug)]
pub struct InferenceError {
    pub message: String,
    pub oom: bool,
}

impl InferenceError {
    pub(crate) fn transport(message: impl Into<String>) -> Self {
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

pub(crate) struct WorkerCommand {
    pub(crate) payload: Value,
    pub(crate) progress: Option<mpsc::UnboundedSender<InferenceProgress>>,
    pub(crate) response: oneshot::Sender<Result<Value, InferenceError>>,
}

pub(crate) struct ExecutionError {
    pub(crate) error: InferenceError,
    pub(crate) restart: bool,
}

impl ExecutionError {
    pub(crate) fn protocol(error: impl fmt::Display) -> Self {
        Self {
            error: InferenceError::transport(format!("Inference worker protocol failed: {error}")),
            restart: true,
        }
    }
}
