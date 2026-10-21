use super::*;

fn download_identity(bytes: &[u8]) -> DownloadIdentity {
    DownloadIdentity {
        repository: "owner/repo".into(),
        revision: "a".repeat(40),
        filename: "model.gguf".into(),
        sha256: format!("{:x}", Sha256::digest(bytes)),
        size: bytes.len() as u64,
    }
}

fn response(
    status: u16,
    body: reqwest::Body,
    length: u64,
    range: Option<&str>,
) -> reqwest::Response {
    let mut response = axum::http::Response::builder()
        .status(status)
        .header("content-length", length);
    if let Some(range) = range {
        response = response.header("content-range", range);
    }
    reqwest::Response::from(response.body(body).unwrap())
}

#[tokio::test]
async fn concurrent_download_is_rejected_and_cancellation_releases_the_file_lock() {
    let root = tempfile::tempdir().unwrap();
    let directory = models_directory(root.path());
    fs::create_dir_all(&directory).await.unwrap();
    let owner_directory = directory.clone();
    let (started, pending) = tokio::sync::oneshot::channel();
    let (_hold, released) = tokio::sync::oneshot::channel::<()>();
    let first = tokio::spawn(async move {
        download_file_with_request(
            &owner_directory,
            &download_identity(b"abc"),
            |offset| async move {
                assert_eq!(offset, 0);
                started.send(()).unwrap();
                let _ = released.await;
                Ok(response(200, reqwest::Body::from("abc"), 3, None))
            },
        )
        .await
    });
    pending.await.unwrap();
    let metadata_path = directory.join(".model.gguf.part.json");
    let metadata = fs::read(&metadata_path).await.unwrap();
    let second = download_file_with_request(&directory, &download_identity(b"xyz"), |_| async {
        panic!("A competing writer must fail before sending a request")
    })
    .await
    .unwrap_err();
    assert!(second.to_string().contains("already running"));
    assert_eq!(fs::read(&metadata_path).await.unwrap(), metadata);
    first.abort();
    let _ = first.await;
    download_file_with_request(
        &directory,
        &download_identity(b"abc"),
        |offset| async move {
            assert_eq!(offset, 0);
            Ok(response(200, reqwest::Body::from("abc"), 3, None))
        },
    )
    .await
    .unwrap();
    assert_eq!(
        fs::read(directory.join("model.gguf")).await.unwrap(),
        b"abc"
    );
    download_file_with_request(&directory, &download_identity(b"abc"), |_| async {
        panic!("A completed competing download must be reused without another request")
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn interrupted_download_resumes_and_publishes_only_verified_bytes() {
    let root = tempfile::tempdir().unwrap();
    let directory = models_directory(root.path());
    fs::create_dir_all(&directory).await.unwrap();
    let identity = download_identity(b"abcdef");
    let interrupted = reqwest::Body::wrap_stream(futures_util::stream::iter([
        Ok::<_, std::io::Error>(b"abc".to_vec()),
        Err(std::io::Error::other(
            "Simulated interrupted model download",
        )),
    ]));
    let first = download_file_with_request(directory.as_path(), &identity, |offset| async move {
        assert_eq!(offset, 0);
        Ok(response(200, interrupted, 6, None))
    })
    .await;
    assert!(first.is_err());
    assert!(!directory.as_path().join("model.gguf").exists());
    assert_eq!(
        fs::read(directory.as_path().join(".model.gguf.part"))
            .await
            .unwrap(),
        b"abc"
    );
    download_file_with_request(directory.as_path(), &identity, |offset| async move {
        assert_eq!(offset, 3);
        Ok(response(
            206,
            reqwest::Body::from("def"),
            3,
            Some("bytes 3-5/6"),
        ))
    })
    .await
    .unwrap();
    assert_eq!(
        fs::read(directory.as_path().join("model.gguf"))
            .await
            .unwrap(),
        b"abcdef"
    );
    assert!(!directory.as_path().join(".model.gguf.part").exists());
    assert!(!directory.as_path().join(".model.gguf.part.json").exists());
}

#[tokio::test]
async fn checksum_mismatch_never_becomes_a_discoverable_model() {
    let root = tempfile::tempdir().unwrap();
    let directory = models_directory(root.path());
    fs::create_dir_all(&directory).await.unwrap();
    let identity = download_identity(b"abc");
    let error = download_file_with_request(directory.as_path(), &identity, |offset| async move {
        assert_eq!(offset, 0);
        Ok(response(200, reqwest::Body::from("xyz"), 3, None))
    })
    .await
    .unwrap_err();
    assert!(error.to_string().contains("SHA-256 mismatch"), "{error:#}");
    assert!(!directory.as_path().join("model.gguf").exists());
    assert!(!directory.as_path().join(".model.gguf.part").exists());
}

#[tokio::test]
async fn ignored_range_restarts_the_file_instead_of_appending_duplicate_bytes() {
    let root = tempfile::tempdir().unwrap();
    let directory = models_directory(root.path());
    fs::create_dir_all(&directory).await.unwrap();
    let identity = download_identity(b"abc");
    fs::write(directory.as_path().join(".model.gguf.part"), b"a")
        .await
        .unwrap();
    fs::write(
        directory.as_path().join(".model.gguf.part.json"),
        serde_json::to_vec(&identity).unwrap(),
    )
    .await
    .unwrap();
    download_file_with_request(directory.as_path(), &identity, |offset| async move {
        assert_eq!(offset, 1);
        Ok(response(200, reqwest::Body::from("abc"), 3, None))
    })
    .await
    .unwrap();
    assert_eq!(
        fs::read(directory.as_path().join("model.gguf"))
            .await
            .unwrap(),
        b"abc"
    );
}

#[test]
fn resumable_download_rejects_wrong_ranges() {
    assert!(valid_content_range("bytes 10-99/100", 10, 100));
    for value in [
        "bytes 0-99/100",
        "bytes 10-99/200",
        "bytes 10-100/100",
        "bytes 10-9/100",
        "bad",
    ] {
        assert!(!valid_content_range(value, 10, 100), "{value}");
    }
}

#[test]
fn readme_changes_do_not_discard_a_verified_partial_model() {
    let first = download_identity(b"model bytes");
    let mut updated = download_identity(b"model bytes");
    updated.revision = "b".repeat(40);
    assert!(first.same_content(&updated));
    updated.sha256 = "c".repeat(64);
    assert!(!first.same_content(&updated));
}
