use crate::protocol::{InferenceError, WorkerCommand};
use crate::session::Session;
use serde_json::json;
use std::path::PathBuf;
use tokio::sync::{mpsc, watch};

pub(crate) async fn supervise(
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
