//! Supervised stdio access to the PyTorch/Diffusers inference process.

mod client;
mod protocol;
mod session;
mod supervisor;

pub use client::Inference;
pub use protocol::{InferenceError, InferenceProgress, SampleRequest};
