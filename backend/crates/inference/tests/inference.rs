use desynth_inference::Inference;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

fn fake_worker(script: &str) -> Option<(TempDir, PathBuf)> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project = root.parent()?.parent()?.parent()?;
    let python = std::env::var_os("DESYNTH_TEST_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            project.join(if cfg!(windows) {
                ".venv/Scripts/python.exe"
            } else {
                ".venv/bin/python"
            })
        });
    if !python.is_file() {
        // The Rust application and its image-only tests also run without Python.
        return None;
    }
    let fixture = tempfile::tempdir().unwrap();
    let worker_directory = fixture.path().join("inference");
    std::fs::create_dir_all(&worker_directory).unwrap();
    std::fs::write(worker_directory.join("worker.py"), script).unwrap();
    Some((fixture, python))
}

#[tokio::test]
async fn release_before_startup_does_not_require_python() {
    let fixture = tempfile::tempdir().unwrap();
    let worker = Inference::new(
        fixture.path().to_owned(),
        fixture.path().join("missing-python"),
    );
    worker.release().await.unwrap();
    let error = worker.probe().await.unwrap_err();
    assert!(!error.oom);
    assert!(
        error
            .message
            .contains("Cannot start Python inference worker")
    );
    worker.shutdown().await.unwrap();
    assert!(worker.probe().await.is_err());
}

#[tokio::test]
async fn inference_errors_preserve_oom_and_allow_the_next_command() {
    let Some((fixture, python)) = fake_worker(include_str!(
        "../../../../inference/tests/fixtures/bridge_errors.py"
    )) else {
        return;
    };
    let worker = Inference::new(fixture.path().to_owned(), python);
    let error = worker.probe().await.unwrap_err();
    assert!(error.oom);
    assert_eq!(error.message, "out of memory");
    assert_eq!(worker.probe().await.unwrap()["ready"], true);
    worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn broken_protocol_kills_the_session_before_restarting() {
    let Some((fixture, python)) = fake_worker(include_str!(
        "../../../../inference/tests/fixtures/bridge_restart.py"
    )) else {
        return;
    };
    let worker = Inference::new(fixture.path().to_owned(), python);
    let error = worker.probe().await.unwrap_err();
    assert!(error.message.contains("unexpected command id"));
    assert_eq!(worker.probe().await.unwrap()["ready"], true);
    worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn overlong_frame_without_newline_is_bounded_and_the_worker_can_restart() {
    let Some((fixture, python)) = fake_worker(include_str!(
        "../../../../inference/tests/fixtures/bridge_oversized.py"
    )) else {
        return;
    };
    let worker = Inference::new(fixture.path().to_owned(), python);
    let reply = tokio::time::timeout(Duration::from_secs(3), worker.probe()).await;
    if reply.is_err() {
        worker.shutdown().await.unwrap();
        panic!("An overlong frame must fail before a worker sends a newline");
    }
    let error = reply.unwrap().unwrap_err();
    assert!(error.message.contains("exceeds 64 KiB"));
    assert_eq!(worker.probe().await.unwrap()["ready"], true);
    worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn shutdown_interrupts_a_blocked_torch_command_and_rejects_queued_work() {
    let Some((fixture, python)) = fake_worker(include_str!(
        "../../../../inference/tests/fixtures/bridge_blocked.py"
    )) else {
        return;
    };
    let worker = Arc::new(Inference::new(fixture.path().to_owned(), python));
    let first_owner = worker.clone();
    let first = tokio::spawn(async move { first_owner.probe().await });
    tokio::time::timeout(Duration::from_secs(3), async {
        while !fixture.path().join("started").exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let queued_owner = worker.clone();
    let queued = tokio::spawn(async move { queued_owner.probe().await });
    tokio::time::timeout(Duration::from_secs(3), worker.shutdown())
        .await
        .unwrap()
        .unwrap();
    assert!(
        first
            .await
            .unwrap()
            .unwrap_err()
            .message
            .contains("shutdown")
    );
    assert!(queued.await.unwrap().is_err());
}
