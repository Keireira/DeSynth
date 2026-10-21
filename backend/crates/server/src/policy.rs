use crate::{ApiState, error::ApiError};
use axum::{
    extract::{Request, State},
    http::{HeaderValue, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use desynth_core::error::ServiceError;

pub(crate) async fn local_policy(
    State(state): State<ApiState>,
    request: Request,
    next: Next,
) -> Response {
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
        ApiError::from(ServiceError::unavailable("The server is stopping.")).into_response()
    } else if !valid || foreign || cross_site {
        ApiError::from(ServiceError {
            status: 403,
            message: "Only requests from the local server address are allowed.".to_owned(),
        })
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
