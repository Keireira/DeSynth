use crate::protocol::{ExecutionError, InferenceError, InferenceProgress, WorkerCommand};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

const MAX_FRAME_BYTES: usize = 65536;

pub(crate) struct Session {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Session {
    pub(crate) fn start(root: &PathBuf, python: &PathBuf) -> Result<Self, InferenceError> {
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

    pub(crate) async fn execute(
        &mut self,
        id: u64,
        command: &WorkerCommand,
    ) -> Result<Value, ExecutionError> {
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

    pub(crate) async fn stop(mut self) {
        // kill().await also waits/reaps, including when a Torch call is blocked.
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
    }
}
