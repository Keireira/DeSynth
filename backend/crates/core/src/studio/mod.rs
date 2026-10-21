mod cache;
mod image_operations;
mod inventory;
mod jobs;
mod lifecycle;

pub use cache::prune;

use crate::{
    assets::Assets,
    restoration::{RestoreClient, RestoreTicket},
};
use desynth_inference::Inference;
use desynth_models::catalog::Catalog;
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::Value;
use std::{path::PathBuf, sync::Arc, time::Instant};
use tokio::{
    sync::{Notify, Semaphore, watch},
    task::JoinHandle,
};

pub struct Store {
    pub assets: Assets,
    pub catalog: Catalog,
    pub jobs: IndexMap<String, Value>,
    pub started: IndexMap<String, Instant>,
    pub active: Option<String>,
    pub latest_job: Option<String>,
    pub probe: Value,
    pub refreshing: bool,
    pub restore_clients: IndexMap<String, RestoreClient>,
    pub restore_pending: IndexMap<String, RestoreTicket>,
    pub restore_running: Option<(String, u64)>,
}

pub struct Studio {
    pub root: PathBuf,
    pub store: Mutex<Store>,
    pub inference: Inference,
    pub cpu: Arc<Semaphore>,
    pub restore_notify: Notify,
    pub shutdown: watch::Sender<bool>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
    closing: tokio::sync::Mutex<()>,
}
