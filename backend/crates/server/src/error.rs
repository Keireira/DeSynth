use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use desynth_core::error::ServiceError;
use serde_json::json;

pub(crate) struct ApiError(pub(crate) ServiceError);

impl From<ServiceError> for ApiError {
    fn from(error: ServiceError) -> Self {
        Self(error)
    }
}

impl From<std::io::Error> for ApiError {
    fn from(error: std::io::Error) -> Self {
        ServiceError::from(error).into()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        ServiceError::from(error).into()
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            StatusCode::from_u16(self.0.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(json!({"error":self.0.message})),
        )
            .into_response()
    }
}
