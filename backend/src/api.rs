use crate::{error::ServiceError, studio::Studio};
use axum::{
    Json, Router,
    body::Body,
    extract::{DefaultBodyLimit, Path, Request, State, rejection::JsonRejection},
    http::{HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc};

#[derive(Clone)]
pub struct ApiState {
    pub studio: Arc<Studio>,
    pub port: u16,
}

impl IntoResponse for ServiceError {
    fn into_response(self) -> Response {
        (
            StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(json!({"error":self.message})),
        )
            .into_response()
    }
}

pub fn router(studio: Arc<Studio>, port: u16) -> Router {
    let state = ApiState { studio, port };
    Router::new()
        .route("/health", get(|| async { Json(json!({"ok":true})) }))
        .route("/api/state", get(session))
        .route("/api/refresh", post(refresh))
        .route("/api/upload", post(upload))
        .route("/api/image/{id}", get(image))
        .route("/api/pixels/{id}", get(pixels))
        .route("/api/compare", post(compare))
        .route("/api/process", post(process))
        .route("/api/job/{id}", get(job))
        .route("/api/restore", post(restore))
        .route("/api/restore/cancel", post(cancel_restore))
        .fallback(static_file)
        .layer(DefaultBodyLimit::max(48 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(state.clone(), local_policy))
        .with_state(state)
}

async fn local_policy(State(state): State<ApiState>, request: Request, next: Next) -> Response {
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    let authority = host.parse::<axum::http::uri::Authority>().ok();
    let valid = authority.is_some_and(|value| {
        matches!(value.host(), "localhost" | "127.0.0.1" | "[::1]" | "::1")
            && value.port_u16() == Some(state.port)
    }) && !host.contains('@');
    let origin = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|value| value.to_str().ok());
    let foreign = origin.is_some_and(|origin| origin != format!("http://{host}"));
    let cross_site = request.method() != axum::http::Method::GET
        && request
            .headers()
            .get("sec-fetch-site")
            .is_some_and(|value| value == "cross-site");
    let mut response = if *state.studio.shutdown.borrow() {
        ServiceError::unavailable("The server is stopping.").into_response()
    } else if !valid || foreign || cross_site {
        ServiceError {
            status: 403,
            message: "Only requests from the local server address are allowed.".to_owned(),
        }
        .into_response()
    } else {
        next.run(request).await
    };
    for (name, value) in [
        (header::CACHE_CONTROL, "no-store"),
        (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        (header::X_FRAME_OPTIONS, "DENY"),
    ] {
        response
            .headers_mut()
            .insert(name, HeaderValue::from_static(value));
    }
    response
}

fn payload(body: Result<Json<Value>, JsonRejection>) -> Result<Value, ServiceError> {
    let Json(value) = body.map_err(|error| ServiceError {
        status: if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
            413
        } else if error.status() == StatusCode::UNSUPPORTED_MEDIA_TYPE {
            415
        } else {
            400
        },
        message: "A valid JSON request is required (maximum 48 MB).".to_owned(),
    })?;
    if !value.is_object() {
        return Err(ServiceError::bad("The request must be a JSON object."));
    }
    Ok(value)
}

fn required<'a>(body: &'a Value, field: &str) -> Result<&'a str, ServiceError> {
    body.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| ServiceError::bad(format!("{field} is required.")))
}
async fn session(State(state): State<ApiState>) -> Json<Value> {
    Json(state.studio.state())
}
async fn refresh(
    State(state): State<ApiState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, ServiceError> {
    payload(body)?;
    Ok(Json(state.studio.refresh()?))
}
async fn upload(
    State(state): State<ApiState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<crate::assets::Asset>, ServiceError> {
    Ok(Json(state.studio.upload(payload(body)?).await?))
}
async fn image(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Response, ServiceError> {
    let bytes = state.studio.image_bytes(&id, false).await?;
    Ok(([(header::CONTENT_TYPE, "image/png")], bytes).into_response())
}
async fn pixels(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Response, ServiceError> {
    let bytes = state.studio.image_bytes(&id, true).await?;
    Ok(([(header::CONTENT_TYPE, "application/octet-stream")], bytes).into_response())
}
async fn compare(
    State(state): State<ApiState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, ServiceError> {
    let body = payload(body)?;
    Ok(Json(
        state
            .studio
            .compare(required(&body, "source_id")?, required(&body, "result_id")?)
            .await?,
    ))
}
async fn process(
    State(state): State<ApiState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), ServiceError> {
    Ok((
        StatusCode::ACCEPTED,
        Json(state.studio.process(payload(body)?)?),
    ))
}
async fn job(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ServiceError> {
    Ok(Json(state.studio.job(&id)?))
}
async fn restore(
    State(state): State<ApiState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, ServiceError> {
    Ok(Json(state.studio.quick_restore(payload(body)?).await?))
}
async fn cancel_restore(
    State(state): State<ApiState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, ServiceError> {
    Ok(Json(state.studio.cancel_restore(payload(body)?)?))
}

async fn static_file(
    State(state): State<ApiState>,
    request: Request,
) -> Result<Response, ServiceError> {
    if request.method() != axum::http::Method::GET {
        return Err(ServiceError::missing("Not found."));
    }
    if request.uri().path() == "/favicon.ico" {
        return Ok(StatusCode::NO_CONTENT.into_response());
    }
    let decoded = percent_encoding::percent_decode_str(request.uri().path())
        .decode_utf8()
        .map_err(|_| ServiceError::missing("Not found."))?;
    let relative = decoded.trim_start_matches('/');
    let relative = if relative.is_empty() {
        "index.html"
    } else {
        relative
    };
    if relative.contains(['\\', '\0', ':']) || relative.split('/').any(|part| part.starts_with('.'))
    {
        return Err(ServiceError::missing("Not found."));
    }
    let root = state.studio.root.join("web/dist");
    if !root.join("index.html").is_file() {
        return Err(ServiceError::unavailable(
            "The editor has not been built. Run make web-build, then reload.",
        ));
    }
    let filename: PathBuf = root
        .join(relative)
        .canonicalize()
        .map_err(|_| ServiceError::missing("Not found."))?;
    if !filename.starts_with(root.canonicalize()?) || !filename.is_file() {
        return Err(ServiceError::missing("Not found."));
    }
    if !matches!(
        filename.extension().and_then(|value| value.to_str()),
        Some(
            "html"
                | "js"
                | "css"
                | "json"
                | "png"
                | "jpg"
                | "jpeg"
                | "svg"
                | "ico"
                | "webp"
                | "woff"
                | "woff2"
                | "ttf"
                | "txt"
        )
    ) {
        return Err(ServiceError::missing("Not found."));
    }
    let mime = mime_guess::from_path(&filename).first_or_octet_stream();
    let data = tokio::fs::read(filename).await?;
    let mut response = Response::new(Body::from(data));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(mime.as_ref()).map_err(|error| anyhow::anyhow!(error))?,
    );
    Ok(response)
}
