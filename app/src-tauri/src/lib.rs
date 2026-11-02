//! Tauri shell around `openmodel-core`.
//!
//! Commands are thin wrappers; long work (downloads, installs, video) runs as
//! a job that reports through the `progress` and `job-done` events. Server
//! output arrives as `server` events, chat streaming as `chat` events and tool
//! approval requests as `approval` events. Approval itself is decided in the
//! core's `ApprovalGate`; the UI can only answer a request that is pending.

use openmodel_core::{
    agent::{self, AgentToolbox},
    catalog::{Catalog, CatalogModel},
    config::{data_dir, Settings},
    diffusion::{self, ImageRequest, MediaModel, SdClient, VideoRequest},
    download::{self, DownloadItem},
    engine::{self, EngineKind, InstalledEngine},
    gguf,
    hardware::{self, Backend, HardwareReport},
    hf::{HfClient, HfModel, RepoListing},
    library::{self, LocalModel},
    llm::{self, ChatEvent, Endpoint},
    server::{find_free_port, ManagedServer, ServerStatus},
    pyengine::{self, PyInstall, PyKind},
    strata::{self, StrataInstall},
    tiering::{self, Machine, ModelShape, Plan, PlanOptions, Tier},
    tools::{ApprovalGate, ChatToolbox, Decision, CHAT_MAX_ROUNDS},
    Progress, ProgressFn,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, State};
use tokio_util::sync::CancellationToken;

type Res<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    format!("{e:#}")
}

fn errs(e: anyhow::Error) -> String {
    format!("{e:#}")
}

struct Session {
    gate: Arc<ApprovalGate>,
    cancel: Mutex<CancellationToken>,
}

pub struct AppState {
    settings: RwLock<Settings>,
    /// Detected hardware before the user's GPU override is applied.
    hw: RwLock<Option<HardwareReport>>,
    llm: ManagedServer,
    sd: ManagedServer,
    /// (media model id, quant) loaded in sd-server.
    sd_model: Mutex<Option<(String, String)>>,
    jobs: Mutex<HashMap<String, CancellationToken>>,
    sessions: Mutex<HashMap<String, Arc<Session>>>,
    http: reqwest::Client,
}

impl AppState {
    fn settings(&self) -> Settings {
        self.settings.read().unwrap().clone()
    }

    async fn hardware(&self, refresh: bool) -> HardwareReport {
        let cached = self.hw.read().unwrap().clone();
        let raw = match cached {
            Some(h) if !refresh => h,
            _ => {
                let dir = self.settings().models_dir;
                let h = tokio::task::spawn_blocking(move || hardware::detect(&dir)).await.expect("hardware detection");
                *self.hw.write().unwrap() = Some(h.clone());
                h
            }
        };
        let mut h = raw;
        h.apply_override(self.settings().gpu_override.as_ref());
        h
    }

    async fn machine(&self) -> Machine {
        let hw = self.hardware(false).await;
        Machine::from_report(&hw, &self.settings())
    }

    fn session(&self, app: &AppHandle, id: &str) -> Arc<Session> {
        let mut m = self.sessions.lock().unwrap();
        m.entry(id.to_string())
            .or_insert_with(|| {
                let app = app.clone();
                let sid = id.to_string();
                let gate = ApprovalGate::new(Arc::new(move |req| {
                    let _ = app.emit("approval", json!({"session": sid, "request": req}));
                }));
                Arc::new(Session { gate, cancel: Mutex::new(CancellationToken::new()) })
            })
            .clone()
    }
}

// ---------------------------------------------------------------- jobs

#[derive(Clone, Serialize)]
struct JobDone {
    job_id: String,
    ok: bool,
    error: Option<String>,
    outputs: Vec<String>,
}

fn progress_emitter(app: &AppHandle) -> ProgressFn {
    let app = app.clone();
    Arc::new(move |p: Progress| {
        let _ = app.emit("progress", p);
    })
}

/// Run `work` in the background as a cancellable job; returns its id.
fn spawn_job<F, Fut>(app: &AppHandle, state: &AppState, prefix: &str, work: F) -> String
where
    F: FnOnce(CancellationToken, ProgressFn, String) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = anyhow::Result<Vec<String>>> + Send + 'static,
{
    let id = format!("{prefix}-{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
    let token = CancellationToken::new();
    state.jobs.lock().unwrap().insert(id.clone(), token.clone());
    let app = app.clone();
    let jid = id.clone();
    tauri::async_runtime::spawn(async move {
        let r = work(token, progress_emitter(&app), jid.clone()).await;
        app.state::<AppState>().jobs.lock().unwrap().remove(&jid);
        let done = match r {
            Ok(outputs) => JobDone { job_id: jid, ok: true, error: None, outputs },
            Err(e) => JobDone { job_id: jid, ok: false, error: Some(format!("{e:#}")), outputs: vec![] },
        };
        let _ = app.emit("job-done", done);
    });
    id
}

#[tauri::command]
fn cancel_job(state: State<AppState>, id: String) -> bool {
    match state.jobs.lock().unwrap().get(&id) {
        Some(t) => {
            t.cancel();
            true
        }
        None => false,
    }
}

// ---------------------------------------------------------------- settings / hardware

#[tauri::command]
fn get_settings(state: State<AppState>) -> Settings {
    state.settings()
}

#[tauri::command]
fn save_settings(state: State<AppState>, settings: Settings) -> Res<()> {
    settings.save().map_err(errs)?;
    *state.settings.write().unwrap() = settings;
    Ok(())
}

#[tauri::command]
fn app_paths() -> Value {
    json!({"data_dir": data_dir(), "version": env!("CARGO_PKG_VERSION")})
}

#[tauri::command]
async fn get_hardware(state: State<'_, AppState>, refresh: bool) -> Res<HardwareReport> {
    Ok(state.hardware(refresh).await)
}

#[derive(Serialize)]
struct VramUse {
    name: String,
    used: u64,
    total: u64,
    spill_suspected: bool,
}

#[tauri::command]
async fn live_vram() -> Vec<VramUse> {
    tokio::task::spawn_blocking(|| {
        hardware::live_vram()
            .into_iter()
            .map(|(name, used, total)| VramUse { spill_suspected: openmodel_core::server::vram_spill_suspected(used, total), name, used, total })
            .collect()
    })
    .await
    .unwrap_or_default()
}

// ---------------------------------------------------------------- catalog / planning

#[derive(Serialize)]
struct QuantFit {
    quant: String,
    size_bytes: u64,
    tier: Tier,
    tier_label: String,
    est_tps: f64,
    ctx: u64,
    gpu_bytes: u64,
    ram_bytes: u64,
    disk_bytes: u64,
    installed: bool,
}

#[derive(Serialize)]
struct CatalogEntry {
    #[serde(flatten)]
    model: CatalogModel,
    description: String,
    fits: Vec<QuantFit>,
    recommended: Option<String>,
    /// Strata's official pack for this model, when it has one.
    strata: Option<&'static strata::OfficialPack>,
}

fn rank(t: Tier) -> u8 {
    match t {
        Tier::Fast => 3,
        Tier::Ok => 2,
        Tier::Slow => 1,
        Tier::Crawl => 0,
    }
}

#[tauri::command]
async fn catalog(state: State<'_, AppState>) -> Res<Vec<CatalogEntry>> {
    let s = state.settings();
    let machine = state.machine().await;
    let local = library::list(&s.models_dir, &data_dir());
    let cat = Catalog::load(&data_dir());
    let opts = PlanOptions { lang: s.language.clone(), ..Default::default() };
    Ok(cat
        .models
        .into_iter()
        .map(|m| {
            let dir = library::repo_dir(&s.models_dir, &m.repo);
            let fits: Vec<QuantFit> = m
                .quants
                .iter()
                .map(|q| {
                    let p = tiering::plan(&m.shape(q), &machine, &opts);
                    let installed = local.iter().any(|l| l.path.starts_with(&dir) && l.quant.as_deref().is_some_and(|lq| lq.eq_ignore_ascii_case(q)));
                    QuantFit { quant: q.clone(), size_bytes: m.size_bytes(q), tier: p.tier, tier_label: p.tier_label, est_tps: p.est_tps, ctx: p.ctx, gpu_bytes: p.gpu_bytes, ram_bytes: p.ram_bytes, disk_bytes: p.disk_bytes, installed }
                })
                .collect();
            // Best tier first, then the largest (highest quality) quant within it.
            let recommended = fits.iter().max_by_key(|f| (rank(f.tier), f.size_bytes)).map(|f| f.quant.clone());
            CatalogEntry { description: m.description(&s.language).to_string(), strata: strata::official_pack(&m.id, &m.repo), model: m, fits, recommended }
        })
        .collect())
}

fn shape_for_file(path: &Path, vision: bool) -> anyhow::Result<(ModelShape, gguf::ModelSummary, Option<PathBuf>)> {
    let g = gguf::open(path)?;
    let sum = g.summary();
    if sum.is_mmproj {
        anyhow::bail!("{} is a vision projector, not a model", path.display());
    }
    if sum.is_diffusion {
        anyhow::bail!("{} is an image model; use the Image page", path.display());
    }
    let mmproj = if vision { path.parent().and_then(library::find_mmproj) } else { None };
    let mm_bytes = mmproj.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0);
    Ok((ModelShape::from_summary(&sum, mm_bytes), sum, mmproj))
}

#[derive(Deserialize)]
struct RunOptions {
    path: PathBuf,
    ctx: Option<u64>,
    custom_args: Option<String>,
    vision: bool,
    /// "strata" runs the model through Strata; anything else uses llama.cpp.
    #[serde(default)]
    engine: Option<String>,
}

fn base_options(s: &Settings, o: &RunOptions, mmproj: Option<PathBuf>) -> PlanOptions {
    let key = o.path.file_name().unwrap_or_default().to_string_lossy().to_string();
    PlanOptions {
        ctx: o.ctx,
        model_path: o.path.display().to_string(),
        mmproj_path: mmproj.map(|p| p.display().to_string()),
        host: s.listen_host().into(),
        port: s.server_port,
        custom_args: o.custom_args.clone().unwrap_or_else(|| s.custom_args.get(&key).cloned().unwrap_or_default()),
        lang: s.language.clone(),
        ..Default::default()
    }
}

#[derive(Serialize)]
struct PlanPreview {
    plan: Plan,
    summary: gguf::ModelSummary,
    mmproj: Option<PathBuf>,
    saved_custom_args: String,
    /// The model belongs to a family Strata runs.
    strata_supported: bool,
}

#[tauri::command]
async fn plan_preview(state: State<'_, AppState>, opts: RunOptions) -> Res<PlanPreview> {
    let s = state.settings();
    let (shape, mut summary, mmproj) = shape_for_file(&opts.path, opts.vision).map_err(errs)?;
    let base = base_options(&s, &opts, mmproj.clone());
    let plan = tiering::plan(&shape, &state.machine().await, &base);
    // Per-layer byte lists are large and only needed by the planner.
    summary.layer_bytes.clear();
    summary.layer_expert_bytes.clear();
    let key = opts.path.file_name().unwrap_or_default().to_string_lossy().to_string();
    let strata_supported = strata::supports(&opts.path.display().to_string());
    Ok(PlanPreview { plan, summary, mmproj, saved_custom_args: s.custom_args.get(&key).cloned().unwrap_or_default(), strata_supported })
}

// ---------------------------------------------------------------- library / Hugging Face

#[derive(Serialize)]
struct LibraryEntry {
    #[serde(flatten)]
    model: LocalModel,
    /// Belongs to a family Strata runs.
    strata: bool,
}

#[tauri::command]
fn library_list(state: State<AppState>) -> Vec<LibraryEntry> {
    library::list(&state.settings().models_dir, &data_dir())
        .into_iter()
        .map(|m| LibraryEntry { strata: strata::supports(&m.path.display().to_string()), model: m })
        .collect()
}

#[tauri::command]
fn library_import(path: PathBuf) -> Res<LocalModel> {
    library::import_path(&data_dir(), &path).map_err(errs)
}

#[tauri::command]
fn library_remove(state: State<AppState>, path: PathBuf) -> Res<()> {
    let s = state.settings();
    let m = library::list(&s.models_dir, &data_dir()).into_iter().find(|m| m.path == path).ok_or("model not found")?;
    library::remove(&s.models_dir, &data_dir(), &m).map_err(errs)
}

#[tauri::command]
async fn hf_search(state: State<'_, AppState>, query: String) -> Res<Vec<HfModel>> {
    HfClient::from_settings(&state.settings()).search(&query).await.map_err(errs)
}

#[tauri::command]
async fn hf_listing(state: State<'_, AppState>, repo: String) -> Res<RepoListing> {
    HfClient::from_settings(&state.settings()).repo_listing(&repo).await.map_err(errs)
}

#[tauri::command]
fn hf_download(app: AppHandle, state: State<AppState>, repo: String, quant: String, mmproj: bool) -> String {
    let s = state.settings();
    let http = state.http.clone();
    spawn_job(&app, &state, "dl", move |cancel, progress, jid| async move {
        let hf = HfClient::from_settings(&s);
        let l = hf.repo_listing(&repo).await?;
        let g = l.quants.iter().find(|g| g.quant.eq_ignore_ascii_case(&quant)).ok_or_else(|| anyhow::anyhow!("{repo} has no {quant}"))?;
        let dir = library::repo_dir(&s.models_dir, &repo);
        let mut files = g.files.clone();
        if mmproj {
            files.extend(openmodel_core::hf::best_mmproj(&l.mmproj).cloned());
        }
        let items: Vec<DownloadItem> =
            files.iter().map(|f| DownloadItem { url: hf.resolve_url(&repo, &f.path), dest: dir.join(f.path.rsplit('/').next().unwrap_or(&f.path)), size: Some(f.size) }).collect();
        let headers: Vec<(String, String)> = hf.auth_header().into_iter().collect();
        download::download_all(&http, &items, &headers, &cancel, &jid, &progress).await?;
        Ok(items.into_iter().map(|i| i.dest.display().to_string()).collect())
    })
}

// ---------------------------------------------------------------- engines

#[derive(Serialize)]
struct Engines {
    llama: Vec<InstalledEngine>,
    sd: Vec<InstalledEngine>,
    recommended: Vec<Backend>,
    strata: Option<StrataInstall>,
    /// Strata ships a package for this system.
    strata_available: bool,
    strata_data_dir: PathBuf,
}

#[tauri::command]
async fn engines(state: State<'_, AppState>) -> Res<Engines> {
    let s = state.settings();
    let hw = state.hardware(false).await;
    Ok(Engines {
        llama: engine::list_installed(&s, EngineKind::Llama),
        sd: engine::list_installed(&s, EngineKind::Sd),
        recommended: hw.recommended_backends,
        strata: strata::installed(&s),
        strata_available: strata::platform().is_some(),
        strata_data_dir: strata::data_dir(&s),
    })
}

#[derive(Serialize)]
struct PyEngineInfo {
    kind: PyKind,
    name: &'static str,
    package: &'static str,
    /// Why it cannot run here ("macos-arm64" / "linux"), None when it can.
    unsupported: Option<&'static str>,
    installed: Option<PyInstall>,
}

#[tauri::command]
fn py_engines(state: State<AppState>) -> Vec<PyEngineInfo> {
    let s = state.settings();
    PyKind::ALL
        .iter()
        .map(|k| PyEngineInfo {
            kind: *k,
            name: k.name(),
            package: k.package(),
            unsupported: k.unsupported_reason(std::env::consts::OS, std::env::consts::ARCH),
            installed: pyengine::installed(&s, *k),
        })
        .collect()
}

/// Install the latest release, or upgrade the installed one.
#[tauri::command]
fn py_engine_install(app: AppHandle, state: State<AppState>, kind: PyKind) -> Res<String> {
    if !kind.supported_here() {
        return Err(format!("{} does not run on this system", kind.name()));
    }
    let s = state.settings();
    Ok(spawn_job(&app, &state, "engine", move |cancel, progress, jid| async move {
        let i = pyengine::install(&s, kind, &cancel, &jid, &progress).await?;
        Ok(vec![i.version])
    }))
}

#[tauri::command]
async fn py_engine_latest(kind: PyKind) -> Res<String> {
    pyengine::latest_version(kind).await.map_err(errs)
}

#[tauri::command]
async fn py_engine_remove(state: State<'_, AppState>, kind: PyKind) -> Res<()> {
    let i = pyengine::installed(&state.settings(), kind).ok_or("not installed")?;
    let tag = match kind {
        PyKind::Vllm => "vllm",
        PyKind::Mlx => "mlx",
    };
    if state.llm.status().engine == tag {
        state.llm.stop().await;
    }
    pyengine::remove(&i).map_err(errs)
}

/// Serve a Hugging Face model (id or local folder) with vLLM or MLX.
#[tauri::command]
async fn py_engine_start(state: State<'_, AppState>, kind: PyKind, model: String, ctx: Option<u64>) -> Res<u16> {
    let s = state.settings();
    let i = pyengine::installed(&s, kind).ok_or("engine not installed (Settings → Engines)")?;
    let model = model.trim().to_string();
    if model.is_empty() {
        return Err("enter a Hugging Face model id or a folder".into());
    }
    let host = s.listen_host();
    let args = pyengine::serve_args(kind, &pyengine::ServeOptions { model: &model, host, port: s.server_port, ctx });
    let tag = match kind {
        PyKind::Vllm => "vllm",
        PyKind::Mlx => "mlx",
    };
    state.llm.start_python(tag, &i.python, args, &pyengine::serve_env(&s), host, s.server_port, &model).await.map_err(errs)
}

/// Open a folder in the system file manager. Only directories: the webview
/// has no general open-path permission, so it cannot launch programs.
#[tauri::command]
fn open_folder(app: AppHandle, path: PathBuf) -> Res<()> {
    use tauri_plugin_opener::OpenerExt;
    std::fs::create_dir_all(&path).map_err(err)?;
    let dir = std::fs::canonicalize(&path).map_err(err)?;
    if !dir.is_dir() {
        return Err(format!("{} is not a folder", path.display()));
    }
    app.opener().open_path(dir.display().to_string(), None::<&str>).map_err(err)
}

#[tauri::command]
fn strata_install(app: AppHandle, state: State<AppState>) -> String {
    let s = state.settings();
    spawn_job(&app, &state, "engine", move |cancel, progress, jid| async move {
        let i = strata::install(&s, &cancel, &jid, &progress).await?;
        Ok(vec![i.launcher.display().to_string()])
    })
}

#[tauri::command]
async fn strata_remove(state: State<'_, AppState>) -> Res<()> {
    let s = state.settings();
    let i = strata::installed(&s).ok_or("Strata is not installed")?;
    if state.llm.status().engine == "strata" {
        state.llm.stop().await;
    }
    strata::remove(&i).map_err(errs)
}

#[tauri::command]
async fn engine_install(app: AppHandle, state: State<'_, AppState>, kind: EngineKind, backend: Option<Backend>) -> Res<String> {
    let s = state.settings();
    let backend = match backend.or(s.backend) {
        Some(b) => b,
        None => state.hardware(false).await.recommended_backends.first().copied().unwrap_or(Backend::Cpu),
    };
    Ok(spawn_job(&app, &state, "engine", move |cancel, progress, jid| async move {
        let e = engine::install(&s, kind, backend, &cancel, &jid, &progress).await?;
        Ok(vec![e.binary.display().to_string()])
    }))
}

#[tauri::command]
fn engine_import(state: State<AppState>, kind: EngineKind, path: PathBuf) -> Res<InstalledEngine> {
    let s = state.settings();
    if path.is_dir() { engine::import_dir(&s, kind, &path, None) } else { engine::import_archive(&s, kind, &path) }.map_err(errs)
}

#[tauri::command]
fn engine_remove(state: State<AppState>, kind: EngineKind, binary: PathBuf) -> Res<()> {
    let e = engine::list_installed(&state.settings(), kind).into_iter().find(|e| e.binary == binary).ok_or("engine not found")?;
    engine::remove(&e).map_err(errs)
}

/// The engine to run: the preferred backend if one is set, else the newest install.
fn pick_engine(s: &Settings, kind: EngineKind) -> Res<InstalledEngine> {
    let all = engine::list_installed(s, kind);
    let preferred = s.backend.and_then(|b| all.iter().find(|e| e.backend == b).cloned());
    preferred.or_else(|| all.into_iter().next()).ok_or_else(|| match kind {
        EngineKind::Llama => "no llama.cpp engine installed (Settings → Engines)".to_string(),
        EngineKind::Sd => "no stable-diffusion.cpp engine installed (Settings → Engines)".to_string(),
    })
}

// ---------------------------------------------------------------- servers

fn server<'a>(state: &'a AppState, which: &str) -> &'a ManagedServer {
    if which == "sd" {
        &state.sd
    } else {
        &state.llm
    }
}

#[tauri::command]
fn server_status(state: State<AppState>, which: String) -> ServerStatus {
    server(&state, &which).status()
}

#[tauri::command]
fn server_logs(state: State<AppState>, which: String, last: usize) -> Vec<String> {
    server(&state, &which).logs(last)
}

#[tauri::command]
async fn server_stop(state: State<'_, AppState>, which: String) -> Res<()> {
    server(&state, &which).stop().await;
    if which == "sd" {
        *state.sd_model.lock().unwrap() = None;
    }
    Ok(())
}

#[tauri::command]
async fn llm_start(state: State<'_, AppState>, opts: RunOptions) -> Res<Option<Plan>> {
    if opts.engine.as_deref() == Some("strata") {
        let label = opts.path.clone();
        return strata_start(&state, strata::Source::Path(&opts.path), &label, opts.ctx, opts.vision).await.map(|_| None);
    }
    let mut s = state.settings();
    let eng = pick_engine(&s, EngineKind::Llama)?;
    if let Some(c) = &opts.custom_args {
        let key = opts.path.file_name().unwrap_or_default().to_string_lossy().to_string();
        if c.trim().is_empty() {
            s.custom_args.remove(&key);
        } else {
            s.custom_args.insert(key, c.clone());
        }
        s.save().map_err(errs)?;
        *state.settings.write().unwrap() = s.clone();
    }
    let (shape, _, mmproj) = shape_for_file(&opts.path, opts.vision).map_err(errs)?;
    let machine = state.machine().await;
    let base = base_options(&s, &opts, mmproj);
    state.llm.start_llama(&eng.binary, shape.total_bytes, |lvl| tiering::plan(&shape, &machine, &PlanOptions { safe_level: lvl, ..base.clone() }), &s.language).await.map(Some).map_err(errs)
}

/// Serve the model through Strata's launcher. Strata chooses its own GPU /
/// RAM placement, so no plan is made here.
async fn strata_start(state: &AppState, source: strata::Source<'_>, label: &Path, ctx: Option<u64>, vision: bool) -> Res<()> {
    let s = state.settings();
    let inst = strata::installed(&s).ok_or("Strata is not installed (Settings → Engines)")?;
    // Free whatever holds the port before choosing one.
    state.llm.stop().await;
    let port = find_free_port(s.listen_host(), s.server_port).ok_or("no free port")?;
    let key = s.lan_access.then(|| uuid::Uuid::new_v4().simple().to_string());
    let mut env = strata::env(&s);
    if let Some(k) = &key {
        env.push((strata::KEY_ENV.into(), k.clone()));
    }
    let args = strata::launch_args(&strata::LaunchOptions { source, port, ctx, vision, lan: s.lan_access });
    std::fs::create_dir_all(strata::data_dir(&s)).map_err(err)?;
    state.llm.start_strata(&inst.launcher, args, env, port, label, key).await.map_err(errs)
}

#[derive(Serialize)]
struct ManagedEntry {
    #[serde(flatten)]
    model: &'static strata::ManagedModel,
    description: &'static str,
    /// Where Strata puts the files.
    dir: PathBuf,
}

#[tauri::command]
fn strata_managed(state: State<AppState>) -> Vec<ManagedEntry> {
    let s = state.settings();
    let zh = s.language.starts_with("zh");
    strata::MANAGED.iter().map(|m| ManagedEntry { model: m, description: if zh { m.desc_zh } else { m.desc_en }, dir: strata::managed_dir(&s) }).collect()
}

fn managed_set(id: &str, set: &str) -> Res<(&'static strata::ManagedModel, &'static strata::ManagedSet)> {
    let m = strata::find_managed(id).ok_or("unknown model")?;
    let st = m.sets.iter().find(|x| x.model == set).ok_or("unknown model set")?;
    Ok((m, st))
}

#[tauri::command]
async fn strata_plan(state: State<'_, AppState>, id: String, set: String) -> Res<String> {
    let (m, st) = managed_set(&id, &set)?;
    let s = state.settings();
    let inst = strata::installed(&s).ok_or("Strata is not installed (Settings → Engines)")?;
    strata::plan_managed(&s, &inst.launcher, m.family, st.model).await.map_err(errs)
}

/// Download (when missing) and serve a Strata-managed model. The first start
/// can take hours; the launcher's output shows on the Run page.
#[tauri::command]
async fn strata_start_managed(state: State<'_, AppState>, id: String, set: String, ctx: Option<u64>) -> Res<()> {
    let (m, st) = managed_set(&id, &set)?;
    let dir = strata::managed_dir(&state.settings());
    std::fs::create_dir_all(&dir).map_err(err)?;
    let label = PathBuf::from(format!("{} {}", m.name, st.label));
    strata_start(&state, strata::Source::Managed { family: m.family, model: st.model, data_dir: &dir }, &label, ctx, true).await
}

#[derive(Serialize)]
struct MediaEntry {
    #[serde(flatten)]
    model: &'static MediaModel,
    installed: Vec<String>,
}

#[tauri::command]
fn media_models(state: State<AppState>) -> Vec<MediaEntry> {
    let dir = state.settings().models_dir;
    diffusion::MODELS.iter().map(|m| MediaEntry { model: m, installed: m.quants.iter().filter(|q| diffusion::is_installed(&dir, m, q)).map(|q| q.to_string()).collect() }).collect()
}

#[tauri::command]
fn media_download(app: AppHandle, state: State<AppState>, id: String, quant: String) -> Res<String> {
    let m = diffusion::find_model(&id).ok_or("unknown model")?;
    let s = state.settings();
    let http = state.http.clone();
    Ok(spawn_job(&app, &state, "dl", move |cancel, progress, jid| async move {
        let hf = HfClient::from_settings(&s);
        let items: Vec<DownloadItem> = diffusion::download_plan(&hf, &s.models_dir, m, &quant).into_iter().map(|(url, dest)| DownloadItem { url, dest, size: None }).collect();
        let headers: Vec<(String, String)> = hf.auth_header().into_iter().collect();
        download::download_all(&http, &items, &headers, &cancel, &jid, &progress).await?;
        Ok(vec![diffusion::model_dir(&s.models_dir, m).display().to_string()])
    }))
}

#[tauri::command]
async fn sd_start(state: State<'_, AppState>, id: String, quant: String) -> Res<()> {
    let m = diffusion::find_model(&id).ok_or("unknown model")?;
    let s = state.settings();
    if !diffusion::is_installed(&s.models_dir, m, &quant) {
        return Err("model files are not downloaded yet".into());
    }
    let eng = pick_engine(&s, EngineKind::Sd)?;
    let args = diffusion::server_args(&s.models_dir, m, &quant, s.listen_host(), s.image_port);
    let total: u64 = m.components.iter().filter_map(|c| std::fs::metadata(diffusion::component_path(&s.models_dir, m, c, &quant)).ok()).map(|md| md.len()).sum();
    let est = (total as f64 / 1.5e9).max(5.0);
    *state.sd_model.lock().unwrap() = None;
    state.sd.spawn(&eng.binary, args, &[], 0).map_err(errs)?;
    state.sd.wait_ready("/sdcpp/v1/capabilities", Duration::from_secs(900), total, est).await.map_err(errs)?;
    *state.sd_model.lock().unwrap() = Some((id, quant));
    Ok(())
}

fn loaded_media(state: &AppState) -> Res<&'static MediaModel> {
    let id = state.sd_model.lock().unwrap().clone().map(|x| x.0).ok_or("no image / video model is running")?;
    diffusion::find_model(&id).ok_or_else(|| "unknown model".into())
}

#[derive(Serialize)]
struct MediaOutput {
    path: String,
    data_url: String,
}

#[tauri::command]
async fn image_generate(state: State<'_, AppState>, req: ImageRequest) -> Res<Vec<MediaOutput>> {
    let m = loaded_media(&state)?;
    let out_dir = state.settings().outputs_dir.join("images");
    let imgs = SdClient::new(&state.sd.base_url()).generate_image(m, &req).await.map_err(errs)?;
    imgs.into_iter()
        .map(|b64| {
            let p = diffusion::save_output(&out_dir, &b64, "png").map_err(errs)?;
            let data_url = if b64.starts_with("data:") { b64 } else { format!("data:image/png;base64,{b64}") };
            Ok(MediaOutput { path: p.display().to_string(), data_url })
        })
        .collect()
}

#[tauri::command]
fn video_generate(app: AppHandle, state: State<AppState>, req: VideoRequest) -> Res<String> {
    let m = loaded_media(&state)?;
    let out_dir = state.settings().outputs_dir.join("videos");
    let base = state.sd.base_url();
    Ok(spawn_job(&app, &state, "video", move |cancel, progress, jid| async move {
        let c = SdClient::new(&base);
        let id = c.start_video(m, &req).await?;
        let vids = c
            .wait_video(&id, &cancel, |f| {
                progress(Progress { done: f.map(|f| (f * 1000.0) as u64).unwrap_or(0), total: f.map(|_| 1000), ..Progress::new(&jid, "generating") });
            })
            .await?;
        vids.iter().map(|v| diffusion::save_output(&out_dir, v, "webm").map(|p| p.display().to_string())).collect()
    }))
}

/// Read a generated file back as a data URL for display.
#[tauri::command]
fn read_media(state: State<AppState>, path: PathBuf) -> Res<String> {
    use base64::Engine;
    let out = state.settings().outputs_dir;
    let canon = path.canonicalize().map_err(err)?;
    if !canon.starts_with(out.canonicalize().map_err(err)?) {
        return Err("only files in the outputs folder can be read".into());
    }
    let mime = match canon.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webm" => "video/webm",
        "mp4" => "video/mp4",
        _ => "application/octet-stream",
    };
    let data = std::fs::read(&canon).map_err(err)?;
    Ok(format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(data)))
}

// ---------------------------------------------------------------- chat / agent

fn resolve_endpoint(state: &AppState, ep: Option<Endpoint>) -> Res<Endpoint> {
    match ep {
        Some(e) => Ok(e),
        None => {
            let st = state.llm.status();
            if st.state != openmodel_core::server::ServerState::Ready {
                return Err("no local model is running (Run page)".into());
            }
            // vLLM and MLX answer to the name of the model they serve.
            let model = if st.engine == "vllm" || st.engine == "mlx" { st.model.map(|m| m.display().to_string()).unwrap_or_default() } else { String::new() };
            Ok(Endpoint { base_url: format!("{}/v1", state.llm.base_url()), api_key: st.api_key, model, ..Default::default() })
        }
    }
}

fn chat_emitter(app: &AppHandle, session: &str) -> impl Fn(ChatEvent) + Send + Sync + 'static {
    let app = app.clone();
    let sid = session.to_string();
    move |e: ChatEvent| {
        let _ = app.emit("chat", json!({"session": sid, "event": e}));
    }
}

fn fresh_token(sess: &Session) -> CancellationToken {
    let t = CancellationToken::new();
    *sess.cancel.lock().unwrap() = t.clone();
    t
}

#[tauri::command]
async fn chat_send(app: AppHandle, state: State<'_, AppState>, session: String, endpoint: Option<Endpoint>, mut messages: Vec<Value>, tools: bool) -> Res<Vec<Value>> {
    let ep = resolve_endpoint(&state, endpoint)?;
    let sess = state.session(&app, &session);
    let cancel = fresh_token(&sess);
    let tb = ChatToolbox { http: state.http.clone(), gate: sess.gate.clone(), cwd: dirs_home() };
    let schemas = if tools { ChatToolbox::schemas() } else { vec![] };
    let on = chat_emitter(&app, &session);
    let tb = &tb;
    llm::run_with_tools(&state.http, &ep, &mut messages, &schemas, CHAT_MAX_ROUNDS, &cancel, &on, |name, args| async move { tb.execute(&name, &args).await })
        .await
        .map_err(errs)?;
    Ok(messages)
}

fn dirs_home() -> PathBuf {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

#[tauri::command]
async fn agent_send(app: AppHandle, state: State<'_, AppState>, session: String, endpoint: Option<Endpoint>, mut messages: Vec<Value>, workspace: PathBuf) -> Res<Vec<Value>> {
    let ep = resolve_endpoint(&state, endpoint)?;
    let sess = state.session(&app, &session);
    let cancel = fresh_token(&sess);
    let tb = AgentToolbox::new(&workspace, sess.gate.clone()).map_err(errs)?;
    if messages.first().and_then(|m| m.get("role")).and_then(Value::as_str) != Some("system") {
        let sys = format!("{}\n\nWorkspace: {}\nOS: {}", agent::SYSTEM_PROMPT, tb.ws.root().display(), std::env::consts::OS);
        messages.insert(0, json!({"role": "system", "content": sys}));
    }
    let on = chat_emitter(&app, &session);
    let tb = &tb;
    let app2 = app.clone();
    let sid = session.clone();
    let app2 = &app2;
    let sid = &sid;
    llm::run_with_tools(&state.http, &ep, &mut messages, &AgentToolbox::schemas(), agent::MAX_ROUNDS, &cancel, &on, |name, args| async move {
        let r = tb.execute(&name, &args).await;
        if name == "update_plan" {
            let steps = tb.plan.lock().unwrap().clone();
            let _ = app2.emit("chat", json!({"session": sid, "event": {"type": "plan", "steps": steps}}));
        }
        r
    })
    .await
    .map_err(errs)?;
    Ok(messages)
}

#[tauri::command]
fn chat_stop(app: AppHandle, state: State<AppState>, session: String) {
    let sess = state.session(&app, &session);
    sess.cancel.lock().unwrap().cancel();
    sess.gate.deny_all();
}

#[tauri::command]
fn approval_resolve(app: AppHandle, state: State<AppState>, session: String, id: String, decision: Decision) -> bool {
    state.session(&app, &session).gate.resolve(&id, decision)
}

#[tauri::command]
async fn endpoint_models(state: State<'_, AppState>, endpoint: Endpoint) -> Res<Vec<String>> {
    llm::list_models(&state.http, &endpoint).await.map_err(errs)
}

// ---------------------------------------------------------------- app

fn forward_server_events(app: &AppHandle, which: &'static str, srv: &ManagedServer) {
    let mut rx = srv.subscribe();
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(e) => {
                    let _ = app.emit("server", json!({"which": which, "event": e}));
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        }
    });
}

pub fn run() {
    let state = AppState {
        settings: RwLock::new(Settings::load()),
        hw: RwLock::new(None),
        llm: ManagedServer::new(),
        sd: ManagedServer::new(),
        sd_model: Mutex::new(None),
        jobs: Mutex::new(HashMap::new()),
        sessions: Mutex::new(HashMap::new()),
        http: openmodel_core::http_client(),
    };
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(state)
        .setup(|app| {
            let h = app.handle().clone();
            let st = app.state::<AppState>();
            forward_server_events(&h, "llm", &st.llm);
            forward_server_events(&h, "sd", &st.sd);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            cancel_job,
            get_settings,
            save_settings,
            app_paths,
            get_hardware,
            live_vram,
            catalog,
            plan_preview,
            library_list,
            library_import,
            library_remove,
            hf_search,
            hf_listing,
            hf_download,
            engines,
            engine_install,
            engine_import,
            engine_remove,
            open_folder,
            py_engines,
            py_engine_install,
            py_engine_latest,
            py_engine_remove,
            py_engine_start,
            strata_install,
            strata_remove,
            strata_managed,
            strata_plan,
            strata_start_managed,
            server_status,
            server_logs,
            server_stop,
            llm_start,
            media_models,
            media_download,
            sd_start,
            image_generate,
            video_generate,
            read_media,
            chat_send,
            agent_send,
            chat_stop,
            approval_resolve,
            endpoint_models,
        ])
        .build(tauri::generate_context!())
        .expect("failed to build the app");
    app.run(|app, event| {
        if let tauri::RunEvent::Exit = event {
            // Never leave model servers running after the window closes.
            let st = app.state::<AppState>();
            tauri::async_runtime::block_on(async {
                st.llm.stop().await;
                st.sd.stop().await;
            });
        }
    });
}
