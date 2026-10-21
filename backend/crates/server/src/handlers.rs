use crate::{ApiState, error::ApiError};
use axum::{
    Json,
    extract::{Path, State, rejection::JsonRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use desynth_core::{assets::Asset, error::ServiceError};
use serde_json::Value;

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

pub(crate) async fn session(State(state): State<ApiState>) -> Json<Value> {
    Json(state.studio.state())
}

pub(crate) async fn refresh(
    State(state): State<ApiState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    payload(body)?;
    Ok(Json(state.studio.refresh()?))
}

pub(crate) async fn upload(
    State(state): State<ApiState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Asset>, ApiError> {
    Ok(Json(state.studio.upload(payload(body)?).await?))
}

pub(crate) async fn image(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let bytes = state.studio.image_bytes(&id, false).await?;
    Ok(([(header::CONTENT_TYPE, "image/png")], bytes).into_response())
}

pub(crate) async fn pixels(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let bytes = state.studio.image_bytes(&id, true).await?;
    Ok(([(header::CONTENT_TYPE, "application/octet-stream")], bytes).into_response())
}

pub(crate) async fn compare(
    State(state): State<ApiState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let body = payload(body)?;
    Ok(Json(
        state
            .studio
            .compare(required(&body, "source_id")?, required(&body, "result_id")?)
            .await?,
    ))
}

pub(crate) async fn process(
    State(state): State<ApiState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    Ok((
        StatusCode::ACCEPTED,
        Json(state.studio.process(payload(body)?)?),
    ))
}

pub(crate) async fn job(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(state.studio.job(&id)?))
}

pub(crate) async fn restore(
    State(state): State<ApiState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(state.studio.quick_restore(payload(body)?).await?))
}

pub(crate) async fn cancel_restore(
    State(state): State<ApiState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(state.studio.cancel_restore(payload(body)?)?))
}
