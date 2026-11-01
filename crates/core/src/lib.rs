//! OpenModel-Manager core library.
//!
//! Clean-room, open-source implementation of a local LLM / image / video / agent
//! manager. Everything here is UI-agnostic so it can be driven by the Tauri
//! desktop app, the CLI, or tests. There is no telemetry and no built-in proxy:
//! the only remote hosts contacted are the ones in [`config::Settings`].

pub mod agent;
pub mod catalog;
pub mod config;
pub mod diffusion;
pub mod download;
pub mod engine;
pub mod gguf;
pub mod hardware;
pub mod hf;
pub mod library;
pub mod server;
pub mod tiering;
pub mod tools;

pub use anyhow::{Error, Result};

/// Progress event shared by downloads, installs and generation jobs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Progress {
    pub job_id: String,
    pub stage: String,
    pub done: u64,
    pub total: Option<u64>,
    /// Bytes per second over the last few seconds.
    pub speed_bps: Option<u64>,
    /// 0-based index of the file being transferred, for multi-file jobs.
    pub file_index: usize,
    pub file_count: usize,
    pub message: Option<String>,
}

impl Progress {
    pub fn new(job_id: &str, stage: &str) -> Progress {
        Progress {
            job_id: job_id.into(),
            stage: stage.into(),
            done: 0,
            total: None,
            speed_bps: None,
            file_index: 0,
            file_count: 1,
            message: None,
        }
    }
}

pub type ProgressFn = std::sync::Arc<dyn Fn(Progress) + Send + Sync>;

pub fn noop_progress() -> ProgressFn {
    std::sync::Arc::new(|_| {})
}

/// Build the shared HTTP client (rustls, sane timeouts, honest user agent).
pub fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(concat!("OpenModel-Manager/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(20))
        .build()
        .expect("reqwest client")
}
