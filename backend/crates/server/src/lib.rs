mod error;
mod handlers;
mod policy;
mod static_files;

use axum::{
    Json, Router,
    extract::DefaultBodyLimit,
    middleware,
    routing::{get, post},
};
use desynth_core::studio::Studio;
use handlers::{
    cancel_restore, compare, image, job, pixels, process, refresh, restore, session, upload,
};
use policy::local_policy;
use serde_json::json;
use static_files::static_file;
use std::sync::Arc;

#[derive(Clone)]
pub struct ApiState {
    pub studio: Arc<Studio>,
    pub port: u16,
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
