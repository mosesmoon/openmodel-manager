//! llama-server process management: start/stop, log capture, load progress,
//! health checks and automatic compatibility retries.

use crate::tiering::Plan;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::broadcast;

const MAX_LOG_LINES: usize = 3000;
pub const MAX_SAFE_LEVEL: u8 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadStage {
    Preparing,
    ReadingModel,
    AllocatingCache,
    WarmingUp,
    Ready,
}

impl LoadStage {
    pub fn label(&self, lang: &str) -> &'static str {
        let zh = lang.starts_with("zh");
        match self {
            LoadStage::Preparing => if zh { "準備中" } else { "Preparing" },
            LoadStage::ReadingModel => if zh { "讀取模型" } else { "Reading model" },
            LoadStage::AllocatingCache => if zh { "配置快取" } else { "Allocating cache" },
            LoadStage::WarmingUp => if zh { "預熱" } else { "Warming up" },
            LoadStage::Ready => if zh { "完成" } else { "Ready" },
        }
    }
}

/// Failure causes recognised in llama-server output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Problem {
    OutOfMemory,
    UnknownArch,
    BadFile,
    CudaError,
    PortInUse,
}

impl Problem {
    /// Problems a compatibility retry cannot fix.
    pub fn is_permanent(&self) -> bool {
        matches!(self, Problem::UnknownArch | Problem::BadFile | Problem::PortInUse)
    }
}

/// Incremental parser for llama-server's log output.
#[derive(Debug, Clone, Serialize)]
pub struct LogParser {
    pub stage: LoadStage,
    /// (offloaded, total) from `load_tensors: offloaded N/M layers to GPU`.
    pub gpu_layers: Option<(u64, u64)>,
    pub problems: Vec<Problem>,
    pub ready: bool,
}

impl Default for LogParser {
    fn default() -> Self {
        LogParser { stage: LoadStage::Preparing, gpu_layers: None, problems: vec![], ready: false }
    }
}

impl LogParser {
    fn advance(&mut self, s: LoadStage) {
        if (s as u8) > (self.stage as u8) {
            self.stage = s;
        }
    }

    pub fn feed(&mut self, line: &str) {
        let l = line.to_lowercase();
        if l.contains("llama_model_loader") || l.contains("load_tensors") || l.contains("clip_model_loader") {
            self.advance(LoadStage::ReadingModel);
        }
        if let Some(i) = l.find("offloaded ") {
            let rest = &l[i + 10..];
            if let Some((a, b)) = rest.split_once('/') {
                let b: String = b.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let (Ok(a), Ok(b)) = (a.trim().parse(), b.parse()) {
                    self.gpu_layers = Some((a, b));
                }
            }
        }
        if l.contains("llama_kv_cache") || l.contains("kv cache") || l.contains("llama_context:") {
            self.advance(LoadStage::AllocatingCache);
        }
        if l.contains("warming up") || l.contains("warmup") {
            self.advance(LoadStage::WarmingUp);
        }
        if l.contains("server is listening") || l.contains("model loaded") || l.contains("all slots are idle") {
            self.advance(LoadStage::Ready);
            self.ready = true;
        }
        let mut add = |p: Problem| {
            if !self.problems.contains(&p) {
                self.problems.push(p);
            }
        };
        if l.contains("out of memory") || l.contains("failed to allocate") || l.contains("unable to allocate") || l.contains("cudamalloc failed") || l.contains("erroroutofdevicememory") {
            add(Problem::OutOfMemory);
        }
        if l.contains("unknown model architecture") {
            add(Problem::UnknownArch);
        } else if l.contains("failed to load model") || l.contains("error loading model") || l.contains("invalid magic") {
            add(Problem::BadFile);
        }
        if l.contains("cuda error") {
            add(Problem::CudaError);
        }
        if l.contains("couldn't bind") || l.contains("address already in use") || l.contains("failed to bind") {
            add(Problem::PortInUse);
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LoadProgress {
    pub stage: LoadStage,
    pub fraction: f64,
    pub read_bytes: u64,
    pub total_bytes: u64,
    pub speed_bps: u64,
    pub elapsed_secs: f64,
}

/// Map parser stage + elapsed time to an overall fraction. Reading the model
/// is estimated from the planned disk speed since llama.cpp prints no bytes.
pub fn estimate_progress(stage: LoadStage, elapsed: Duration, total_bytes: u64, est_load_secs: f64) -> LoadProgress {
    let read_frac = (elapsed.as_secs_f64() / est_load_secs.max(1.0)).min(0.97);
    let fraction = match stage {
        LoadStage::Preparing => 0.02,
        LoadStage::ReadingModel => 0.05 + 0.75 * read_frac,
        LoadStage::AllocatingCache => 0.85,
        LoadStage::WarmingUp => 0.93,
        LoadStage::Ready => 1.0,
    };
    let read_bytes = match stage {
        LoadStage::Preparing => 0,
        LoadStage::ReadingModel => (total_bytes as f64 * read_frac) as u64,
        _ => total_bytes,
    };
    LoadProgress {
        stage,
        fraction,
        read_bytes,
        total_bytes,
        speed_bps: (total_bytes as f64 / est_load_secs.max(1.0)) as u64,
        elapsed_secs: elapsed.as_secs_f64(),
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerEvent {
    Status { state: ServerState, port: u16 },
    Log { line: String },
    Note { message: String },
    Progress(LoadProgress),
    Crashed { problems: Vec<Problem>, exit_code: Option<i32>, safe_level: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerState {
    Stopped,
    Starting,
    Ready,
    Crashed,
}

#[derive(Debug, Clone, Serialize)]
pub struct CrashInfo {
    pub problems: Vec<Problem>,
    pub exit_code: Option<i32>,
    pub tail: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ServerStatus {
    pub state: ServerState,
    pub port: u16,
    pub model: Option<PathBuf>,
    pub safe_level: u8,
    pub gpu_layers: Option<(u64, u64)>,
    pub last_crash: Option<CrashInfo>,
    /// "llama", "sd" or "strata".
    pub engine: String,
    /// Key clients must send (Strata in LAN mode).
    #[serde(skip)]
    pub api_key: Option<String>,
}

/// How to stop a server that runs detached from the process we started
/// (Strata's launcher returns once its server is up).
#[derive(Clone)]
struct StopCommand {
    program: PathBuf,
    args: Vec<String>,
    env: Vec<(String, String)>,
}

struct Inner {
    child: Option<Child>,
    state: ServerState,
    port: u16,
    model: Option<PathBuf>,
    safe_level: u8,
    logs: VecDeque<String>,
    parser: LogParser,
    last_crash: Option<CrashInfo>,
    engine: String,
    api_key: Option<String>,
    stop_cmd: Option<StopCommand>,
}

/// One managed llama-server (or sd-server) process.
#[derive(Clone)]
pub struct ManagedServer {
    inner: Arc<Mutex<Inner>>,
    events: broadcast::Sender<ServerEvent>,
}

impl Default for ManagedServer {
    fn default() -> Self {
        ManagedServer::new()
    }
}

/// First port at or above `start` that can be bound on `host`.
pub fn find_free_port(host: &str, start: u16) -> Option<u16> {
    (start..start.saturating_add(200)).find(|p| std::net::TcpListener::bind((host, *p)).is_ok())
}

/// Replace (or add) a flag's value in an argument list.
pub fn set_arg(args: &mut Vec<String>, flag: &str, value: &str) {
    if let Some(i) = args.iter().position(|a| a == flag) {
        if i + 1 < args.len() {
            args[i + 1] = value.to_string();
            return;
        }
    }
    args.push(flag.into());
    args.push(value.into());
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).map(String::as_str)
}

impl ManagedServer {
    pub fn new() -> ManagedServer {
        let (tx, _) = broadcast::channel(1024);
        ManagedServer {
            inner: Arc::new(Mutex::new(Inner {
                child: None,
                state: ServerState::Stopped,
                port: 0,
                model: None,
                safe_level: 0,
                logs: VecDeque::new(),
                parser: LogParser::default(),
                last_crash: None,
                engine: String::new(),
                api_key: None,
                stop_cmd: None,
            })),
            events: tx,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ServerEvent> {
        self.events.subscribe()
    }

    fn emit(&self, e: ServerEvent) {
        let _ = self.events.send(e);
    }

    pub fn status(&self) -> ServerStatus {
        let i = self.inner.lock().unwrap();
        ServerStatus {
            state: i.state,
            port: i.port,
            model: i.model.clone(),
            safe_level: i.safe_level,
            gpu_layers: i.parser.gpu_layers,
            last_crash: i.last_crash.clone(),
            engine: i.engine.clone(),
            api_key: i.api_key.clone(),
        }
    }

    pub fn logs(&self, last: usize) -> Vec<String> {
        let i = self.inner.lock().unwrap();
        i.logs.iter().skip(i.logs.len().saturating_sub(last)).cloned().collect()
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.inner.lock().unwrap().port)
    }

    /// Spawn the process. Returns once it is running (not necessarily ready).
    pub fn spawn(&self, binary: &Path, mut args: Vec<String>, env: &[(String, String)], safe_level: u8) -> Result<u16> {
        self.stop_now();
        let host = arg_value(&args, "--host").or_else(|| arg_value(&args, "--listen-ip")).unwrap_or("127.0.0.1").to_string();
        let port_flag = if args.iter().any(|a| a == "--listen-port") { "--listen-port" } else { "--port" };
        let wanted: u16 = arg_value(&args, port_flag).and_then(|p| p.parse().ok()).unwrap_or(8080);
        let bind_host = if host == "0.0.0.0" { "0.0.0.0" } else { "127.0.0.1" };
        let port = find_free_port(bind_host, wanted).context("no free port")?;
        set_arg(&mut args, port_flag, &port.to_string());
        let model = arg_value(&args, "-m").or_else(|| arg_value(&args, "--diffusion-model")).map(PathBuf::from);
        let engine = if args.iter().any(|a| a == "--diffusion-model") { "sd" } else { "llama" };
        self.launch(binary, &args, env, port, model, safe_level, engine)?;
        Ok(port)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(&self, binary: &Path, args: &[String], env: &[(String, String)], port: u16, model: Option<PathBuf>, safe_level: u8, engine: &str) -> Result<()> {
        let mut cmd = Command::new(binary);
        cmd.args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
        if let Some(dir) = binary.parent() {
            cmd.current_dir(dir); // CUDA runtime DLLs live next to the binary on Windows
        }
        for (k, v) in env {
            cmd.env(k, v);
        }
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        let mut child = cmd.spawn().with_context(|| format!("start {}", binary.display()))?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        {
            let mut i = self.inner.lock().unwrap();
            i.child = Some(child);
            i.state = ServerState::Starting;
            i.port = port;
            i.model = model;
            i.safe_level = safe_level;
            i.logs.clear();
            i.parser = LogParser::default();
            i.engine = engine.into();
            i.api_key = None;
            i.stop_cmd = None;
        }
        if let Some(o) = stdout {
            self.pump(o);
        }
        if let Some(e) = stderr {
            self.pump(e);
        }
        self.emit(ServerEvent::Status { state: ServerState::Starting, port });
        Ok(())
    }

    fn pump<R: tokio::io::AsyncRead + Unpin + Send + 'static>(&self, r: R) {
        let me = self.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(r).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                {
                    let mut i = me.inner.lock().unwrap();
                    i.parser.feed(&line);
                    i.logs.push_back(line.clone());
                    while i.logs.len() > MAX_LOG_LINES {
                        i.logs.pop_front();
                    }
                }
                me.emit(ServerEvent::Log { line });
            }
        });
    }

    /// Wait until `/health` answers 200, the process exits, or `timeout` passes.
    pub async fn wait_ready(&self, health_path: &str, timeout: Duration, total_bytes: u64, est_load_secs: f64) -> Result<()> {
        let http = reqwest::Client::builder().timeout(Duration::from_secs(3)).build()?;
        let url = format!("{}{}", self.base_url(), health_path);
        let t0 = Instant::now();
        loop {
            let exited = {
                let mut i = self.inner.lock().unwrap();
                match i.child.as_mut().map(|c| c.try_wait()) {
                    Some(Ok(Some(st))) => Some(st.code()),
                    None => Some(None),
                    _ => None,
                }
            };
            if let Some(code) = exited {
                // Give the log pumps a moment to drain the last lines.
                tokio::time::sleep(Duration::from_millis(300)).await;
                let crash = {
                    let mut i = self.inner.lock().unwrap();
                    i.child = None;
                    i.state = ServerState::Crashed;
                    let c = CrashInfo { problems: i.parser.problems.clone(), exit_code: code, tail: i.logs.iter().rev().take(40).rev().cloned().collect() };
                    i.last_crash = Some(c.clone());
                    c
                };
                let lvl = self.inner.lock().unwrap().safe_level;
                self.emit(ServerEvent::Crashed { problems: crash.problems.clone(), exit_code: code, safe_level: lvl });
                bail!("server exited during startup (code {:?}, problems {:?})", code, crash.problems);
            }
            if let Ok(r) = http.get(&url).send().await {
                if r.status().is_success() {
                    let port = {
                        let mut i = self.inner.lock().unwrap();
                        i.state = ServerState::Ready;
                        i.parser.advance(LoadStage::Ready);
                        i.port
                    };
                    self.emit(ServerEvent::Progress(estimate_progress(LoadStage::Ready, t0.elapsed(), total_bytes, est_load_secs)));
                    self.emit(ServerEvent::Status { state: ServerState::Ready, port });
                    return Ok(());
                }
            }
            let stage = self.inner.lock().unwrap().parser.stage;
            self.emit(ServerEvent::Progress(estimate_progress(stage, t0.elapsed(), total_bytes, est_load_secs)));
            if t0.elapsed() > timeout {
                self.stop().await;
                bail!("server did not become ready within {}s", timeout.as_secs());
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    fn stop_now(&self) {
        let mut i = self.inner.lock().unwrap();
        if let Some(mut c) = i.child.take() {
            let _ = c.start_kill();
        }
        i.state = ServerState::Stopped;
    }

    pub async fn stop(&self) {
        let (child, stop_cmd) = {
            let mut i = self.inner.lock().unwrap();
            (i.child.take(), i.stop_cmd.take())
        };
        if let Some(mut c) = child {
            // Strata's launcher is a .cmd that starts Python children; take the whole tree.
            #[cfg(windows)]
            if let Some(pid) = c.id() {
                let mut tk = Command::new("taskkill");
                tk.args(["/PID", &pid.to_string(), "/T", "/F"]).stdout(Stdio::null()).stderr(Stdio::null()).creation_flags(0x0800_0000);
                let _ = tk.status().await;
            }
            let _ = c.kill().await;
        }
        if let Some(sc) = stop_cmd {
            let mut cmd = Command::new(&sc.program);
            cmd.args(&sc.args).envs(sc.env.iter().map(|(k, v)| (k, v))).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
            #[cfg(windows)]
            cmd.creation_flags(0x0800_0000);
            let _ = tokio::time::timeout(Duration::from_secs(60), cmd.status()).await;
        }
        let port = {
            let mut i = self.inner.lock().unwrap();
            i.state = ServerState::Stopped;
            i.port
        };
        self.emit(ServerEvent::Status { state: ServerState::Stopped, port });
    }

    /// Start a server through Strata's launcher. The launcher prints its
    /// progress (including a first-time engine download), starts the server
    /// detached and returns; the server is ready once `/health` answers.
    pub async fn start_strata(&self, launcher: &Path, args: Vec<String>, env: Vec<(String, String)>, port: u16, model: &Path, api_key: Option<String>) -> Result<()> {
        self.stop().await;
        self.launch(launcher, &args, &env, port, Some(model.to_path_buf()), 0, "strata")?;
        {
            let mut i = self.inner.lock().unwrap();
            i.api_key = api_key;
            i.stop_cmd = Some(StopCommand { program: launcher.to_path_buf(), args: crate::strata::stop_args(), env });
        }
        let http = reqwest::Client::builder().timeout(Duration::from_secs(3)).build()?;
        let url = format!("http://127.0.0.1:{port}/health");
        let t0 = Instant::now();
        let mut launcher_done: Option<Instant> = None;
        loop {
            if launcher_done.is_none() {
                let exited = {
                    let mut i = self.inner.lock().unwrap();
                    match i.child.as_mut().map(|c| c.try_wait()) {
                        Some(Ok(Some(st))) => Some(st),
                        _ => None,
                    }
                };
                if let Some(st) = exited {
                    tokio::time::sleep(Duration::from_millis(300)).await;
                    if !st.success() {
                        self.fail_start(st.code());
                        bail!("Strata could not start the model (exit code {:?}); see the log", st.code());
                    }
                    launcher_done = Some(Instant::now());
                }
            }
            if let Ok(r) = http.get(&url).send().await {
                if r.status().is_success() {
                    self.inner.lock().unwrap().state = ServerState::Ready;
                    self.emit(ServerEvent::Status { state: ServerState::Ready, port });
                    return Ok(());
                }
            }
            // The launcher exits 0 without serving when Strata has no engine
            // for this PC; allow a grace period for a slow server.
            if launcher_done.map(|d| d.elapsed() > Duration::from_secs(120)).unwrap_or(false) {
                self.fail_start(Some(0));
                bail!("Strata finished but its server is not answering on port {port}; see the log");
            }
            let stage = self.inner.lock().unwrap().parser.stage;
            self.emit(ServerEvent::Progress(estimate_progress(stage, t0.elapsed(), 0, 600.0)));
            tokio::time::sleep(Duration::from_millis(1000)).await;
        }
    }

    fn fail_start(&self, code: Option<i32>) {
        let crash = {
            let mut i = self.inner.lock().unwrap();
            i.child = None;
            i.stop_cmd = None;
            i.state = ServerState::Crashed;
            let c = CrashInfo { problems: i.parser.problems.clone(), exit_code: code, tail: i.logs.iter().rev().take(40).rev().cloned().collect() };
            i.last_crash = Some(c.clone());
            c
        };
        self.emit(ServerEvent::Crashed { problems: crash.problems, exit_code: code, safe_level: 0 });
    }

    /// Start llama-server, retrying with higher compatibility levels when it
    /// crashes during load. `make_plan(level)` builds the plan for a level.
    pub async fn start_llama(&self, binary: &Path, total_bytes: u64, make_plan: impl Fn(u8) -> Plan, lang: &str) -> Result<Plan> {
        let zh = lang.starts_with("zh");
        let mut level = 0u8;
        if self.inner.lock().unwrap().stop_cmd.is_some() {
            self.stop().await; // a detached Strata server holds the GPU
        }
        loop {
            let plan = make_plan(level);
            self.spawn(binary, plan.args.clone(), &plan.env, level)?;
            let timeout = Duration::from_secs_f64((plan.est_load_secs * 6.0).max(300.0));
            match self.wait_ready("/health", timeout, total_bytes, plan.est_load_secs).await {
                Ok(()) => {
                    if level > 0 {
                        self.emit(ServerEvent::Note {
                            message: if zh { format!("已用相容模式 {level} 啟動") } else { format!("Started in compatibility mode {level}") },
                        });
                    }
                    return Ok(plan);
                }
                Err(e) => {
                    let problems = self.status().last_crash.map(|c| c.problems).unwrap_or_default();
                    if problems.iter().any(Problem::is_permanent) || level >= MAX_SAFE_LEVEL {
                        return Err(e);
                    }
                    level += 1;
                    self.emit(ServerEvent::Note {
                        message: if zh { format!("啟動失敗，改用相容模式 {level} 重試") } else { format!("Start failed; retrying in compatibility mode {level}") },
                    });
                }
            }
        }
    }
}

/// On Windows the NVIDIA driver silently spills into shared system memory
/// when VRAM is full, which makes generation crawl. Flag it when usage is
/// within 3 % of the total.
pub fn vram_spill_suspected(used: u64, total: u64) -> bool {
    total > 0 && used as f64 >= total as f64 * 0.97
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_load_sequence() {
        let mut p = LogParser::default();
        p.feed("llama_model_loader: loaded meta data with 40 key-value pairs");
        assert_eq!(p.stage, LoadStage::ReadingModel);
        p.feed("load_tensors: offloaded 33/49 layers to GPU");
        assert_eq!(p.gpu_layers, Some((33, 49)));
        p.feed("llama_kv_cache:      CUDA0 KV buffer size =  1024.00 MiB");
        assert_eq!(p.stage, LoadStage::AllocatingCache);
        p.feed("common_init_from_params: warming up the model with an empty run");
        assert_eq!(p.stage, LoadStage::WarmingUp);
        p.feed("llama_model_loader: late line");
        assert_eq!(p.stage, LoadStage::WarmingUp, "stage never goes backwards");
        p.feed("main: server is listening on http://127.0.0.1:8080 - starting the main loop");
        assert!(p.ready);
        assert!(p.problems.is_empty());
    }

    #[test]
    fn recognises_failures() {
        let mut p = LogParser::default();
        p.feed("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000.00 MiB on device 0: cudaMalloc failed: out of memory");
        p.feed("llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'foo'");
        assert_eq!(p.problems, vec![Problem::OutOfMemory, Problem::UnknownArch]);
        assert!(p.problems.iter().any(Problem::is_permanent));
    }

    #[test]
    fn progress_estimate() {
        let p = estimate_progress(LoadStage::ReadingModel, Duration::from_secs(5), 10_000, 10.0);
        assert!((p.fraction - 0.425).abs() < 1e-9);
        assert_eq!(p.read_bytes, 5000);
        assert_eq!(estimate_progress(LoadStage::Ready, Duration::ZERO, 1, 1.0).fraction, 1.0);
    }

    #[test]
    fn port_and_args() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let busy = l.local_addr().unwrap().port();
        assert_ne!(find_free_port("127.0.0.1", busy), Some(busy));
        let mut a: Vec<String> = ["--port", "8080"].map(String::from).to_vec();
        set_arg(&mut a, "--port", "8081");
        set_arg(&mut a, "--host", "0.0.0.0");
        assert_eq!(a, ["--port", "8081", "--host", "0.0.0.0"]);
        assert!(vram_spill_suspected(99, 100));
    }

    /// Runs a fake "server" that prints a crash line and exits, and checks that
    /// the crash is reported with the parsed problem.
    #[tokio::test]
    async fn reports_crash_from_process() {
        let (bin, args): (PathBuf, Vec<String>) = if cfg!(windows) {
            ("cmd".into(), ["/C", "echo cudaMalloc failed: out of memory 1>&2 & exit 3"].map(String::from).to_vec())
        } else {
            ("sh".into(), ["-c", "echo 'cudaMalloc failed: out of memory' >&2; exit 3"].map(String::from).to_vec())
        };
        let s = ManagedServer::new();
        let mut rx = s.subscribe();
        s.spawn(&which::which(&bin).unwrap(), args, &[], 0).unwrap();
        let err = s.wait_ready("/health", Duration::from_secs(20), 0, 1.0).await.unwrap_err();
        assert!(format!("{err}").contains("exited"));
        let st = s.status();
        assert_eq!(st.state, ServerState::Crashed);
        assert_eq!(st.last_crash.unwrap().problems, vec![Problem::OutOfMemory]);
        let mut saw_crash = false;
        while let Ok(e) = rx.try_recv() {
            saw_crash |= matches!(e, ServerEvent::Crashed { .. });
        }
        assert!(saw_crash);
    }
}
