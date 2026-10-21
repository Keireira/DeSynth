use crate::{ApiState, error::ApiError};
use axum::{
    body::Body,
    extract::{Request, State},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use desynth_core::error::ServiceError;
use std::path::PathBuf;

pub(crate) async fn static_file(
    State(state): State<ApiState>,
    request: Request,
) -> Result<Response, ApiError> {
    if request.method() != axum::http::Method::GET {
        return Err(ServiceError::missing("Not found.").into());
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
        return Err(ServiceError::missing("Not found.").into());
    }
    let root = state.studio.root.join("web/dist");
    if !root.join("index.html").is_file() {
        return Err(ServiceError::unavailable(
            "The editor has not been built. Run make web-build, then reload.",
        )
        .into());
    }
    let filename: PathBuf = root
        .join(relative)
        .canonicalize()
        .map_err(|_| ServiceError::missing("Not found."))?;
    if !filename.starts_with(root.canonicalize()?) || !filename.is_file() {
        return Err(ServiceError::missing("Not found.").into());
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
        return Err(ServiceError::missing("Not found.").into());
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
