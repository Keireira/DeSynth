use anyhow::{Context, Result, bail, ensure};
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{HeaderMap, Method, Request, StatusCode, header},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use desynth::{api, images, studio::Studio};
use image::{Rgba, RgbaImage};
use serde_json::{Value, json};
use std::{future::Future, path::PathBuf, sync::Arc, time::Duration};
use tempfile::TempDir;
use tower::ServiceExt;

const PORT: u16 = 38463;
const ADAPTER: &str = "inference/models/Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors";
const FOUR_STEP_ADAPTER: &str =
    "inference/models/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors";

struct Fixture {
    directory: TempDir,
    studio: Arc<Studio>,
    router: Router,
}

struct Reply {
    status: StatusCode,
    headers: HeaderMap,
    bytes: Vec<u8>,
}

impl Reply {
    fn json(&self) -> Result<Value> {
        serde_json::from_slice(&self.bytes).context("API response is not valid JSON")
    }

    fn expect(&self, status: StatusCode) -> Result<()> {
        ensure!(
            self.status == status,
            "Expected HTTP {status}, got {}: {}",
            self.status,
            String::from_utf8_lossy(&self.bytes)
        );
        Ok(())
    }
}

impl Fixture {
    fn new(python: Option<PathBuf>) -> Result<Self> {
        let directory = tempfile::tempdir()?;
        let root = directory.path();
        std::fs::create_dir_all(root.join("inference/models"))?;
        std::fs::create_dir_all(root.join("inference"))?;
        std::fs::write(
            root.join("inference/models/Qwen-Image-2512-Q4_K_M.gguf"),
            [0_u8; 16],
        )?;
        std::fs::write(
            root.join("inference/models/Qwen-Image-2512-Q2_K_M.gguf"),
            [0_u8; 8],
        )?;
        std::fs::write(
            root.join("inference/models/Qwen-Image-2512-Q8_0.gguf"),
            [0_u8; 32],
        )?;
        for name in [
            ADAPTER,
            FOUR_STEP_ADAPTER,
            "inference/models/embeds_cache.pt",
        ] {
            std::fs::write(root.join(name), b"test weights")?;
        }
        std::fs::write(
            root.join("inference/worker.py"),
            include_str!("../../inference/tests/fixtures/http_worker.py"),
        )?;
        let studio = Studio::new(
            root.to_owned(),
            python.unwrap_or_else(|| root.join("missing-python")),
        )?;
        studio.store.lock().probe = ready_probe();
        let router = api::router(studio.clone(), PORT);
        Ok(Self {
            directory,
            studio,
            router,
        })
    }

    fn root(&self) -> &std::path::Path {
        self.directory.path()
    }

    async fn raw(
        &self,
        method: Method,
        path: &str,
        body: Vec<u8>,
        extra_headers: &[(&str, &str)],
    ) -> Result<Reply> {
        let mut request = Request::builder()
            .method(method)
            .uri(path)
            .header(header::HOST, format!("127.0.0.1:{PORT}"));
        for (name, value) in extra_headers {
            request = request.header(*name, *value);
        }
        let response = self
            .router
            .clone()
            .oneshot(request.body(Body::from(body))?)
            .await?;
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = to_bytes(response.into_body(), usize::MAX).await?.to_vec();
        Ok(Reply {
            status,
            headers,
            bytes,
        })
    }

    async fn get(&self, path: &str) -> Result<Reply> {
        self.raw(Method::GET, path, Vec::new(), &[]).await
    }

    async fn post(&self, path: &str, payload: Value) -> Result<Reply> {
        self.raw(
            Method::POST,
            path,
            serde_json::to_vec(&payload)?,
            &[("content-type", "application/json")],
        )
        .await
    }

    async fn upload(&self, image: &RgbaImage, role: &str) -> Result<Value> {
        let data = STANDARD.encode(images::encode_png(image, None)?);
        let response = self
            .post("/api/upload", json!({"role":role,"name":"test.png","data":format!("data:image/png;base64,{data}")}))
            .await?;
        response.expect(StatusCode::OK)?;
        response.json()
    }

    async fn pixels(&self, asset: &Value) -> Result<Vec<u8>> {
        let response = self.get(&format!("/api/pixels/{}", id(asset)?)).await?;
        response.expect(StatusCode::OK)?;
        ensure!(response.headers[header::CONTENT_TYPE] == "application/octet-stream");
        Ok(response.bytes)
    }

    async fn wait_for_job(&self, job: &Value) -> Result<Value> {
        let path = format!("/api/job/{}", id(job)?);
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let response = self.get(&path).await?;
                response.expect(StatusCode::OK)?;
                let current = response.json()?;
                if matches!(current["status"].as_str(), Some("done" | "error"))
                    && self.studio.store.lock().active.is_none()
                {
                    return Ok(current);
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .context("Processing did not finish within five seconds")?
    }

    async fn process(&self, source: &Value, params: Value) -> Result<Value> {
        let response = self
            .post(
                "/api/process",
                json!({"source_id":id(source)?,"params":params}),
            )
            .await?;
        response.expect(StatusCode::ACCEPTED)?;
        self.wait_for_job(&response.json()?).await
    }

    fn clean(&self, source: &Value, image: &RgbaImage) -> Result<Value> {
        let mut store = self.studio.store.lock();
        let metadata = json!({"seed":42,"params":{"seed":42,"restore_mode":"gaussian"}});
        let asset = store
            .assets
            .save(image, "clean", "clean.png", false, Some(metadata))?;
        store
            .assets
            .clean_sources
            .insert(asset.id.clone(), id(source)?.to_owned());
        Ok(serde_json::to_value(asset)?)
    }

    fn sample_requests(&self) -> Result<Vec<Value>> {
        std::fs::read_to_string(self.root().join("sample-requests.jsonl"))?
            .lines()
            .map(|line| serde_json::from_str(line).map_err(Into::into))
            .collect()
    }
}

fn id(value: &Value) -> Result<&str> {
    value["id"]
        .as_str()
        .context("Asset or job has no string ID")
}

fn source_image() -> RgbaImage {
    RgbaImage::from_fn(35, 29, |x, y| {
        Rgba([
            (x * 7) as u8,
            (y * 9) as u8,
            ((x ^ y) * 13) as u8,
            ((x + y) * 11) as u8,
        ])
    })
}

fn ready_probe() -> Value {
    json!({"ready":true,"missing":[],"device":"CPU","runtime":{"ready":true,"default_device":"cpu","devices":[{"id":"cpu","name":"CPU","type":"cpu","precisions":["float32"],"offload_modes":["none"]}]}})
}

fn test_python() -> Option<PathBuf> {
    let project = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project = project.parent()?;
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
        return None;
    }
    let available = std::process::Command::new(&python)
        .arg(project.join("inference/tests/fixtures/http_worker.py"))
        .stdin(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    available.then_some(python)
}

// The scenario runs in an owned task so cleanup also happens after an assertion panics.
// All requests use Router::oneshot; no test leaves a TCP server listening.
async fn scenario<F, Fut>(needs_worker: bool, operation: F) -> Result<()>
where
    F: FnOnce(Arc<Fixture>) -> Fut,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    let python = if needs_worker {
        let Some(python) = test_python() else {
            eprintln!(
                "Skipping inference HTTP scenario: set DESYNTH_TEST_PYTHON to Python with Pillow, or run make install."
            );
            return Ok(());
        };
        Some(python)
    } else {
        None
    };
    let fixture = Arc::new(Fixture::new(python)?);
    let mut task = tokio::spawn(operation(fixture.clone()));
    let result = tokio::time::timeout(Duration::from_secs(20), &mut task).await;
    if result.is_err() {
        task.abort();
        let _ = task.await;
    }
    fixture.studio.close().await;
    match result {
        Ok(result) => result.context("HTTP scenario task failed")?,
        Err(_) => bail!("HTTP scenario exceeded twenty seconds"),
    }
}

#[tokio::test]
async fn rgba_upload_png_and_raw_pixels_preserve_hidden_colors_and_alpha() -> Result<()> {
    scenario(false, |fixture| async move {
        let source = source_image();
        let asset = fixture.upload(&source, "source").await?;
        ensure!(asset["width"] == 35 && asset["height"] == 29 && asset["kind"] == "source");
        ensure!(fixture.pixels(&asset).await? == *source.as_raw());
        let png = fixture
            .get(asset["url"].as_str().context("No asset URL")?)
            .await?;
        png.expect(StatusCode::OK)?;
        ensure!(png.headers[header::CONTENT_TYPE] == "image/png");
        ensure!(images::open(&png.bytes)? == source);
        ensure!(png.headers[header::X_CONTENT_TYPE_OPTIONS] == "nosniff");

        let metadata = json!({"seed":42,"params":{"restore_mode":"edge"}});
        let recorded_path = fixture.root().join("out/recorded.png");
        std::fs::write(
            &recorded_path,
            images::encode_png(&source, Some(&metadata))?,
        )?;
        let recorded =
            fixture
                .studio
                .store
                .lock()
                .assets
                .register(recorded_path, "result", None)?;
        let exported = fixture.get(&recorded.url).await?;
        ensure!(
            images::png_metadata(&exported.bytes) == Some(metadata),
            "Re-exporting a registered PNG discarded its provenance"
        );

        let mut alpha_changed = source.clone();
        alpha_changed.get_pixel_mut(0, 0)[3] = 19;
        let result = fixture.upload(&alpha_changed, "result").await?;
        let scores = fixture
            .post(
                "/api/compare",
                json!({"source_id":id(&asset)?,"result_id":id(&result)?}),
            )
            .await?;
        scores.expect(StatusCode::OK)?;
        let scores = scores.json()?;
        ensure!(scores["identical"] == false && scores["mae"] == 0.0);
        ensure!(scores["mean_alpha"].as_f64().context("No alpha metric")? > 0.0);

        let mut hidden_changed = source.clone();
        hidden_changed.get_pixel_mut(0, 0)[0] = 233;
        let result = fixture.upload(&hidden_changed, "result").await?;
        ensure!(fixture.pixels(&result).await?[0] == 233);
        let scores = fixture
            .post(
                "/api/compare",
                json!({"source_id":id(&asset)?,"result_id":id(&result)?}),
            )
            .await?
            .json()?;
        ensure!(
            scores["mean_alpha"] == 0.0 && scores["mae"].as_f64().context("No RGB metric")? > 0.0
        );
        let different_size = fixture.upload(&RgbaImage::new(3, 2), "result").await?;
        fixture
            .post(
                "/api/compare",
                json!({"source_id":id(&asset)?,"result_id":id(&different_size)?}),
            )
            .await?
            .expect(StatusCode::BAD_REQUEST)?;
        Ok(())
    })
    .await
}

#[tokio::test]
async fn json_image_validation_and_unknown_assets_return_api_errors_without_jobs() -> Result<()> {
    scenario(false, |fixture| async move {
        for path in [
            "/api/image/missing",
            "/api/pixels/missing",
            "/api/job/missing",
        ] {
            let response = fixture.get(path).await?;
            response.expect(StatusCode::NOT_FOUND)?;
            ensure!(response.json()?["error"].is_string());
        }
        for payload in [
            json!({"data":"not base64!"}),
            json!({"data":"AA=="}),
            json!({"data":"AA==","role":"clean"}),
            json!([]),
        ] {
            fixture
                .post("/api/upload", payload)
                .await?
                .expect(StatusCode::BAD_REQUEST)?;
        }
        fixture
            .raw(Method::POST, "/api/upload", b"{}".to_vec(), &[])
            .await?
            .expect(StatusCode::UNSUPPORTED_MEDIA_TYPE)?;
        fixture
            .raw(
                Method::POST,
                "/api/upload",
                b"{bad".to_vec(),
                &[("content-type", "application/json")],
            )
            .await?
            .expect(StatusCode::BAD_REQUEST)?;
        let source = fixture.upload(&source_image(), "source").await?;
        for params in [
            json!({"steps":true}),
            json!({"steps":8.5}),
            json!({"passes":9}),
            json!({"seed":-1}),
            json!({"restore":"true"}),
            json!({"device":"cuda:0"}),
            json!({"precision":"float16"}),
            json!({"lightning":"../outside.safetensors"}),
            json!({"transformer":"../outside.gguf"}),
        ] {
            let response = fixture
                .post(
                    "/api/process",
                    json!({"source_id":id(&source)?,"params":params}),
                )
                .await?;
            response.expect(StatusCode::BAD_REQUEST)?;
            ensure!(response.json()?["error"].is_string());
        }
        ensure!(fixture.studio.store.lock().jobs.is_empty());
        ensure!(!fixture.root().join("sample-requests.jsonl").exists());
        std::fs::remove_file(fixture.root().join("inference/models/embeds_cache.pt"))?;
        fixture
            .post(
                "/api/process",
                json!({"source_id":id(&source)?,"params":{}}),
            )
            .await?
            .expect(StatusCode::SERVICE_UNAVAILABLE)?;
        ensure!(fixture.studio.store.lock().jobs.is_empty());
        std::fs::write(fixture.root().join("inference/models/embeds_cache.pt"), b"")?;
        let state = fixture.get("/api/state").await?.json()?;
        ensure!(state["ready"] == false);
        fixture
            .post(
                "/api/process",
                json!({"source_id":id(&source)?,"params":{}}),
            )
            .await?
            .expect(StatusCode::SERVICE_UNAVAILABLE)?;
        ensure!(fixture.studio.store.lock().jobs.is_empty());
        Ok(())
    })
    .await
}

#[tokio::test]
async fn request_policy_rejects_foreign_hosts_origins_and_cross_site_mutations() -> Result<()> {
    scenario(false, |fixture| async move {
        for host in [
            "evil.invalid:38463",
            "127.0.0.1:1",
            "localhost",
            "localhost@evil.invalid:38463",
        ] {
            let request = Request::builder()
                .uri("/api/state")
                .header(header::HOST, host)
                .body(Body::empty())?;
            let response = fixture.router.clone().oneshot(request).await?;
            ensure!(
                response.status() == StatusCode::FORBIDDEN,
                "Accepted Host {host}"
            );
        }
        for headers in [
            vec![("origin", "https://evil.invalid")],
            vec![("sec-fetch-site", "cross-site")],
        ] {
            let response = fixture
                .raw(Method::POST, "/api/upload", b"{}".to_vec(), &headers)
                .await?;
            response.expect(StatusCode::FORBIDDEN)?;
            ensure!(response.headers[header::CACHE_CONTROL] == "no-store");
            ensure!(response.headers[header::X_FRAME_OPTIONS] == "DENY");
        }
        fixture
            .raw(
                Method::GET,
                "/api/state",
                Vec::new(),
                &[("origin", "http://127.0.0.1:38463")],
            )
            .await?
            .expect(StatusCode::OK)?;
        ensure!(fixture.studio.store.lock().assets.entries.is_empty());
        Ok(())
    })
    .await
}

#[tokio::test]
async fn built_static_assets_have_correct_mime_and_cannot_escape_the_build() -> Result<()> {
    scenario(false, |fixture| async move {
        fixture
            .get("/")
            .await?
            .expect(StatusCode::SERVICE_UNAVAILABLE)?;
        let build = fixture.root().join("web/dist");
        std::fs::create_dir_all(build.join("static/js"))?;
        std::fs::create_dir_all(build.join("static/css"))?;
        for (name, bytes, mime) in [
            ("index.html", "<title>DeSynth</title>", "text/html"),
            ("static/js/main.hash.js", "export {};", "javascript"),
            (
                "static/js/worker.hash.js",
                "self.onmessage = () => {};",
                "javascript",
            ),
            (
                "static/css/main.hash.css",
                "body { margin: 0; }",
                "text/css",
            ),
        ] {
            std::fs::write(build.join(name), bytes)?;
            let reply = fixture.get(&format!("/{name}")).await?;
            reply.expect(StatusCode::OK)?;
            ensure!(reply.bytes == bytes.as_bytes());
            ensure!(reply.headers[header::CONTENT_TYPE].to_str()?.contains(mime));
            ensure!(reply.headers[header::X_CONTENT_TYPE_OPTIONS] == "nosniff");
        }
        std::fs::write(fixture.root().join("web/private.json"), b"secret")?;
        std::fs::write(build.join(".hidden.js"), b"secret")?;
        std::fs::write(build.join("source.py"), b"secret")?;
        for path in [
            "/../private.json",
            "/%2e%2e/private.json",
            "/%2e%2e%5cprivate.json",
            "/F:%5cprivate.json",
            "/.hidden.js",
            "/source.py",
            "/missing.js",
            "/api/image/../../private.json",
        ] {
            fixture.get(path).await?.expect(StatusCode::NOT_FOUND)?;
        }
        fixture
            .post("/index.html", json!({}))
            .await?
            .expect(StatusCode::NOT_FOUND)?;
        Ok(())
    })
    .await
}

#[tokio::test]
async fn oversized_request_bodies_and_pixel_bombs_are_rejected_before_processing() -> Result<()> {
    scenario(false, |fixture| async move {
        let reply = fixture
            .raw(
                Method::POST,
                "/api/upload",
                vec![b' '; 48 * 1024 * 1024 + 1],
                &[("content-type", "application/json")],
            )
            .await?;
        reply.expect(StatusCode::PAYLOAD_TOO_LARGE)?;
        ensure!(reply.json()?["error"].is_string());
        let mut bomb = images::encode_png(&RgbaImage::new(1, 1), None)?;
        bomb[16..20].copy_from_slice(&6001_u32.to_be_bytes());
        bomb[20..24].copy_from_slice(&6000_u32.to_be_bytes());
        let mut crc = u32::MAX;
        for byte in &bomb[12..29] {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                crc = (crc >> 1) ^ (0xedb8_8320 & 0_u32.wrapping_sub(crc & 1));
            }
        }
        bomb[29..33].copy_from_slice(&(!crc).to_be_bytes());
        let reply = fixture
            .post("/api/upload", json!({"data":STANDARD.encode(bomb)}))
            .await?;
        reply.expect(StatusCode::BAD_REQUEST)?;
        ensure!(
            reply.json()?["error"]
                .as_str()
                .context("No pixel limit error")?
                .contains("36 million")
        );
        ensure!(fixture.studio.store.lock().assets.entries.is_empty());
        Ok(())
    })
    .await
}

#[tokio::test]
async fn process_refresh_polling_and_png_metadata_keep_source_alpha() -> Result<()> {
    scenario(true, |fixture| async move {
        let response = fixture.post("/api/refresh",json!({})).await?;
        response.expect(StatusCode::OK)?;
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let state = fixture.get("/api/state").await?.json()?;
                if state["ready"] == true && state["runtime"]["refreshing"] == false { break Ok::<_,anyhow::Error>(()); }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }).await??;
        let state = fixture.get("/api/state").await?.json()?;
        ensure!(state["default_steps"] == 32 && state["default_lightning"] == ADAPTER);
        let source_image = source_image();
        let source = fixture.upload(&source_image,"source").await?;
        let payload = json!({"source_id":id(&source)?,"params":{"seed":42,"restore_mode":"edge","keep_intermediate":true}});
        let queued = fixture.post("/api/process",payload.clone()).await?;
        queued.expect(StatusCode::ACCEPTED)?;
        fixture.post("/api/process",payload).await?.expect(StatusCode::CONFLICT)?;
        let done = fixture.wait_for_job(&queued.json()?).await?;
        ensure!(done["status"] == "done", "Job failed: {done}");
        ensure!(done["progress"] == 100 && done["seed"] == 42);
        ensure!(done["runtime"]["device"] == "cpu" && done["runtime"]["offload"] == "none");
        ensure!(done["requested_runtime"] == json!({"device":"auto","offload":"auto","precision":"auto"}));
        ensure!(done["steps"] == 32 && done["starting_sigma"] == 0.25);
        let result = &done["result"];
        let bytes = fixture.pixels(result).await?;
        ensure!(bytes.len() == source_image.as_raw().len());
        for (left,right) in bytes.chunks_exact(4).zip(source_image.pixels()) { ensure!(left[3] == right[3]); }
        ensure!(bytes != *source_image.as_raw(), "The deterministic sampler must actually change pixels");
        let png = fixture.get(result["url"].as_str().context("No result URL")?).await?;
        let provenance = images::png_metadata(&png.bytes).context("Export PNG lost generation provenance")?;
        ensure!(provenance["params"]["steps"] == 32 && provenance["params"]["restore_mode"] == "edge");
        ensure!(provenance["seed"] == 42 && provenance["source"]["id"] == source["id"]);
        let clean_path = fixture.studio.store.lock().assets.get(id(&done["clean"])?,None)?.path;
        ensure!(images::png_metadata(&std::fs::read(clean_path)?) == Some(provenance.clone()), "The saved clean image lost generation metadata");
        let scores = fixture.post("/api/compare",json!({"source_id":id(&source)?,"result_id":id(result)?})).await?.json()?;
        ensure!(scores == done["metrics"], "Displayed metrics differ from exported image metrics");
        ensure!(fixture.sample_requests()?[0]["steps"] == 32);
        let four = fixture.process(&source,json!({"lightning":FOUR_STEP_ADAPTER,"restore":false,"seed":43})).await?;
        ensure!(four["status"] == "done");
        let requests = fixture.sample_requests()?;
        ensure!(requests[1]["steps"] == 8);
        ensure!(requests[1]["precision"] == "float32");
        Ok(())
    }).await
}

#[tokio::test]
async fn automatic_oom_falls_back_only_to_smaller_models_and_other_errors_do_not_retry()
-> Result<()> {
    scenario(true, |fixture| async move {
        let source = fixture.upload(&source_image(), "source").await?;
        std::fs::write(fixture.root().join("oom-preferred"), b"yes")?;
        let done = fixture.process(&source, json!({"seed":42})).await?;
        ensure!(done["status"] == "done", "{done}");
        ensure!(done["transformer"] == "inference/models/Qwen-Image-2512-Q2_K_M.gguf");
        let requests = fixture.sample_requests()?;
        ensure!(requests.len() == 2);
        ensure!(requests.iter().all(|request| {
            !request["transformer"]
                .as_str()
                .unwrap_or_default()
                .contains("Q8_0")
        }));
        std::fs::remove_file(fixture.root().join("oom-preferred"))?;
        std::fs::write(fixture.root().join("fail-sample"), b"yes")?;
        let failed = fixture.process(&source, json!({"seed":42})).await?;
        ensure!(
            failed["status"] == "error"
                && failed["error"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("Intentional sampler failure")
        );
        ensure!(fixture.sample_requests()?.len() == 3);
        ensure!(fixture.studio.store.lock().active.is_none());
        ensure!(
            std::fs::read_dir(fixture.root().join("out/gui"))?.all(|entry| {
                let name = entry.unwrap().file_name().to_string_lossy().into_owned();
                !name.ends_with("-input.png") && !name.ends_with("-clean.png")
            }),
            "A failed job left uncommitted working files"
        );
        std::fs::remove_file(fixture.root().join("fail-sample"))?;
        let retry = fixture.process(&source, json!({"seed":42})).await?;
        ensure!(
            retry["status"] == "done",
            "A failed job blocked the next run: {retry}"
        );
        Ok(())
    })
    .await
}

#[tokio::test]
async fn live_restoration_uses_the_clean_source_pair_and_updates_png_metadata() -> Result<()> {
    scenario(false, |fixture| async move {
        let original = source_image();
        let source = fixture.upload(&original,"source").await?;
        let smooth = RgbaImage::from_pixel(32,16,Rgba([80,90,100,255]));
        let clean = fixture.clean(&source,&smooth)?;
        let client = "settings-preview";
        let mut outputs = Vec::new();
        for (revision,mode) in [(1,"gaussian"),(2,"edge")] {
            let params = json!({"restore_mode":mode,"restore_sigma":1.95,"unsharp":0.2});
            let reply = fixture.post("/api/restore",json!({"client_id":client,"revision":revision,"source_id":id(&source)?,"clean_id":id(&clean)?,"params":params})).await?;
            reply.expect(StatusCode::OK)?;
            let reply = reply.json()?;
            ensure!(reply["revision"] == revision && reply["superseded"] != true);
            let pixels = fixture.pixels(&reply["result"]).await?;
            for (left,right) in pixels.chunks_exact(4).zip(original.pixels()) { ensure!(left[3] == right[3]); }
            let png = fixture.get(reply["result"]["url"].as_str().context("Missing result URL")?).await?;
            let metadata = images::png_metadata(&png.bytes).context("Restoration discarded PNG metadata")?;
            ensure!(metadata["params"]["restore_mode"] == mode && metadata["seed"] == 42);
            outputs.push(pixels);
        }
        ensure!(outputs[0] != outputs[1], "Changing Gaussian to Edge did not update the preview");
        let another = fixture.upload(&original,"source").await?;
        fixture.post("/api/restore",json!({"client_id":client,"revision":3,"source_id":id(&another)?,"clean_id":id(&clean)?,"params":{}})).await?.expect(StatusCode::BAD_REQUEST)?;
        Ok(())
    }).await
}

#[tokio::test]
async fn newest_restore_revision_supersedes_both_pending_and_running_results() -> Result<()> {
    scenario(false, |fixture| async move {
        let source = fixture.upload(&source_image(),"source").await?;
        let clean = fixture.clean(&source,&RgbaImage::from_pixel(32,16,Rgba([50,60,70,255])))?;
        let identity = json!({"client_id":"interactive","source_id":id(&source)?,"clean_id":id(&clean)?,"params":{}});
        let permit = fixture.studio.cpu.clone().acquire_many_owned(2).await?;
        let first = {
            let fixture = fixture.clone();
            let mut payload = identity.clone();
            payload["revision"] = json!(1);
            tokio::spawn(async move { fixture.post("/api/restore",payload).await })
        };
        wait_until(|| fixture.studio.store.lock().restore_running.as_ref().is_some_and(|(_,revision)| *revision == 1)).await?;
        fixture.post("/api/restore/cancel",json!({"client_id":"interactive","revision":2})).await?.expect(StatusCode::OK)?;
        let pending = {
            let fixture = fixture.clone();
            let mut payload = identity.clone();
            payload["revision"] = json!(3);
            tokio::spawn(async move { fixture.post("/api/restore",payload).await })
        };
        wait_until(|| fixture.studio.store.lock().restore_pending.contains_key("interactive")).await?;
        fixture.post("/api/restore/cancel",json!({"client_id":"interactive","revision":4})).await?.expect(StatusCode::OK)?;
        ensure!(pending.await??.json()?["superseded"] == true);
        drop(permit);
        ensure!(first.await??.json()?["superseded"] == true);
        let mut payload = identity;
        payload["revision"] = json!(5);
        let latest = fixture.post("/api/restore",payload.clone()).await?;
        latest.expect(StatusCode::OK)?;
        ensure!(latest.json()?["revision"] == 5);
        payload["revision"] = json!(4);
        ensure!(fixture.post("/api/restore",payload).await?.json()?["superseded"] == true);
        Ok(())
    }).await
}

async fn wait_until(predicate: impl Fn() -> bool) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(3), async {
        while !predicate() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .context("The expected restoration state was never reached")
}

#[tokio::test]
async fn queued_image_reads_survive_cache_eviction() -> Result<()> {
    scenario(false, |fixture| async move {
        let image = source_image();
        let asset = fixture.upload(&image, "result").await?;
        let image_url = asset["url"].as_str().context("Missing image URL")?;
        let pixels_url = format!("/api/pixels/{}", id(&asset)?);
        let permit = fixture.studio.cpu.clone().acquire_many_owned(2).await?;
        let mut image_read = Box::pin(fixture.get(image_url));
        let mut pixels_read = Box::pin(fixture.get(&pixels_url));
        ensure!(futures_util::poll!(&mut image_read).is_pending());
        ensure!(futures_util::poll!(&mut pixels_read).is_pending());

        {
            let mut store = fixture.studio.store.lock();
            for _ in 0..50 {
                store
                    .assets
                    .save(&image, "result", "preview.png", true, None)?;
            }
            desynth::studio::prune(&mut store, []);
        }
        drop(permit);

        let png = image_read.await?;
        png.expect(StatusCode::OK)?;
        ensure!(images::open(&png.bytes)? == image);
        let pixels = pixels_read.await?;
        pixels.expect(StatusCode::OK)?;
        ensure!(pixels.bytes == *image.as_raw());

        {
            let mut store = fixture.studio.store.lock();
            store
                .assets
                .save(&image, "result", "next.png", true, None)?;
            desynth::studio::prune(&mut store, []);
        }
        fixture
            .get(&pixels_url)
            .await?
            .expect(StatusCode::NOT_FOUND)?;
        Ok(())
    })
    .await
}

#[tokio::test]
async fn another_tabs_previews_do_not_evict_the_displayed_result() -> Result<()> {
    scenario(false, |fixture| async move {
        let image = source_image();
        let source = fixture.upload(&image, "source").await?;
        let clean = fixture.clean(&source, &image)?;
        let request = json!({"client_id":"first-tab","revision":1,"source_id":id(&source)?,"clean_id":id(&clean)?});
        let first = fixture.post("/api/restore", request.clone()).await?.json()?;
        let result = &first["result"];
        let expected = fixture.pixels(result).await?;

        for revision in 1..55 {
            let mut next = request.clone();
            next["client_id"] = json!("second-tab");
            next["revision"] = json!(revision);
            fixture.post("/api/restore", next).await?.expect(StatusCode::OK)?;
        }
        ensure!(fixture.pixels(result).await? == expected);
        Ok(())
    })
    .await
}

#[tokio::test]
async fn invalid_restore_requests_do_not_cancel_a_valid_preview() -> Result<()> {
    scenario(false, |fixture| async move {
        let image = source_image();
        let source = fixture.upload(&image, "source").await?;
        let clean = fixture.clean(&source, &image)?;
        let permit = fixture.studio.cpu.clone().acquire_many_owned(2).await?;
        let payload = json!({"client_id":"preview","revision":1,"source_id":id(&source)?,"clean_id":id(&clean)?});
        let running = {
            let fixture = fixture.clone();
            let payload = payload.clone();
            tokio::spawn(async move { fixture.post("/api/restore", payload).await })
        };
        wait_until(|| fixture.studio.store.lock().restore_running.is_some()).await?;

        let mut invalid = payload;
        invalid["revision"] = json!(2);
        invalid["source_id"] = json!("missing-source");
        fixture.post("/api/restore", invalid).await?.expect(StatusCode::BAD_REQUEST)?;
        drop(permit);

        let reply = running.await??;
        reply.expect(StatusCode::OK)?;
        ensure!(reply.json()?["result"].is_object(), "Invalid input canceled a valid preview");
        Ok(())
    })
    .await
}

#[tokio::test]
async fn asset_inventory_and_temporary_preview_files_are_bounded_and_cleaned_on_close() -> Result<()>
{
    scenario(false, |fixture| async move {
        let pixel = RgbaImage::from_pixel(1,1,Rgba([130,80,50,17]));
        let first = fixture.upload(&pixel,"source").await?;
        for _ in 0..135 { fixture.upload(&pixel,"source").await?; }
        fixture.get(&format!("/api/pixels/{}",id(&first)?)).await?.expect(StatusCode::NOT_FOUND)?;
        ensure!(fixture.studio.store.lock().assets.entries.len() <= 128);
        let source = fixture.upload(&source_image(),"source").await?;
        let clean = fixture.clean(&source,&RgbaImage::from_pixel(32,16,Rgba([80,90,100,255])))?;
        for revision in 0..55 {
            fixture.post("/api/restore",json!({"client_id":"preview-cache","revision":revision,"source_id":id(&source)?,"clean_id":id(&clean)?,"params":{}})).await?.expect(StatusCode::OK)?;
        }
        let directory = fixture.root().join("out/gui/temporary");
        let temporary = std::fs::read_dir(&directory)?.collect::<std::io::Result<Vec<_>>>()?;
        ensure!(temporary.len() <= 48, "Temporary PNG files exceeded the cache bound: {}",temporary.len());
        ensure!(fixture.studio.store.lock().assets.entries.len() <= 128);
        fixture.studio.close().await;
        ensure!(std::fs::read_dir(directory)?.next().is_none(), "Closing the studio left preview PNGs behind");
        let clean_path = fixture.studio.store.lock().assets.get(id(&clean)?,None)?.path;
        ensure!(clean_path.is_file(), "Closing the studio removed a persistent model output");
        Ok(())
    }).await
}

#[tokio::test]
async fn closing_before_background_tasks_start_releases_the_studio() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let studio = Studio::new(
        directory.path().to_owned(),
        directory.path().join("missing-python"),
    )?;
    let weak = Arc::downgrade(&studio);
    studio.close().await;
    drop(studio);
    ensure!(
        weak.upgrade().is_none(),
        "The restoration loop retained a closed studio"
    );
    Ok(())
}

#[tokio::test]
async fn cpu_operations_are_bounded_and_shutdown_waits_for_running_work() -> Result<()> {
    scenario(false, |fixture| async move {
        use std::sync::{
            Condvar, Mutex,
            atomic::{AtomicUsize, Ordering},
        };
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        struct ReleaseGate(Arc<(Mutex<bool>, Condvar)>);
        impl Drop for ReleaseGate {
            fn drop(&mut self) {
                let (lock, condition) = &*self.0;
                *lock.lock().unwrap() = true;
                condition.notify_all();
            }
        }
        let _release = ReleaseGate(gate.clone());
        let active = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));
        let mut operations = Vec::new();
        for _ in 0..3 {
            let studio = fixture.studio.clone();
            let gate = gate.clone();
            let active = active.clone();
            let completed = completed.clone();
            operations.push(tokio::spawn(async move {
                studio
                    .cpu_task(move || {
                        active.fetch_add(1, Ordering::SeqCst);
                        let (lock, condition) = &*gate;
                        let mut released = lock.lock().unwrap();
                        while !*released {
                            released = condition.wait(released).unwrap();
                        }
                        active.fetch_sub(1, Ordering::SeqCst);
                        completed.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                    .await
            }));
        }
        wait_until(|| active.load(Ordering::SeqCst) == 2).await?;
        let studio = fixture.studio.clone();
        let closing = tokio::spawn(async move {
            studio.close().await;
        });
        wait_until(|| *fixture.studio.shutdown.borrow()).await?;
        let waited_for_work = !closing.is_finished();
        let (lock, condition) = &*gate;
        *lock.lock().unwrap() = true;
        condition.notify_all();
        closing.await?;
        ensure!(
            waited_for_work,
            "close returned before running CPU work completed"
        );
        ensure!(
            completed.load(Ordering::SeqCst) == 2,
            "A queued operation started after shutdown"
        );
        let mut rejected = 0;
        for operation in operations {
            if let Err(error) = operation.await? {
                ensure!(error.status == 503);
                rejected += 1;
            }
        }
        ensure!(rejected == 1);
        Ok(())
    })
    .await
}

#[tokio::test]
async fn queued_uploads_cannot_commit_temporary_files_after_shutdown() -> Result<()> {
    scenario(false, |fixture| async move {
        let permit = fixture.studio.cpu.clone().acquire_many_owned(2).await?;
        let queued = {
            let fixture = fixture.clone();
            let data = STANDARD.encode(images::encode_png(&source_image(), None)?);
            tokio::spawn(async move { fixture.post("/api/upload", json!({"data":data})).await })
        };
        tokio::task::yield_now().await;
        fixture.studio.close().await;
        drop(permit);
        queued.await??.expect(StatusCode::SERVICE_UNAVAILABLE)?;
        ensure!(
            std::fs::read_dir(fixture.root().join("out/gui/temporary"))?
                .next()
                .is_none()
        );
        ensure!(fixture.studio.store.lock().assets.entries.is_empty());
        fixture
            .post("/api/refresh", json!({}))
            .await?
            .expect(StatusCode::SERVICE_UNAVAILABLE)?;
        Ok(())
    })
    .await
}
