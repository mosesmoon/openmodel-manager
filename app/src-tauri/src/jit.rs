//! The OpenAI API clients connect to.
//!
//! This gateway always answers on the server port (and on the chosen network
//! cards); model servers run behind it on 127.0.0.1 and the next free ports.
//! It also answers the routes agents probe that llama-server lacks, such as
//! `GET /v1/models/{id}`.
//!
//! With "keep the API on" (JIT), `GET /v1/models` lists every downloaded
//! model and a call naming one that is not running loads it first. A second
//! model stays loaded next to the first only when the GPU has room for it;
//! otherwise the loaded models are stopped before the new one starts.
//! Without JIT, calls go to whatever model runs.

use super::*;
use axum::body::{Body, Bytes};
use axum::extract::State as AxState;
use axum::http::{header, HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use std::net::IpAddr;
use tokio::task::JoinHandle;

pub struct Jit {
    restarting: tokio::sync::Mutex<()>,
    listeners: Mutex<Vec<JoinHandle<()>>>,
    status: Mutex<JitStatus>,
    /// One model load at a time.
    loading: tokio::sync::Mutex<()>,
    /// Models loaded beside the main one (`AppState::llm`), oldest first.
    extra: Mutex<Vec<Extra>>,
}

struct Extra {
    path: PathBuf,
    server: ManagedServer,
}

#[derive(Clone, Default, Serialize)]
pub struct JitStatus {
    running: bool,
    port: u16,
    urls: Vec<String>,
    /// Why an address is not open (yet).
    error: Option<String>,
}

#[derive(Serialize)]
pub struct JitLoaded {
    name: String,
    port: u16,
}

impl Default for Jit {
    fn default() -> Self {
        Jit { restarting: tokio::sync::Mutex::new(()), listeners: Mutex::new(vec![]), status: Mutex::new(JitStatus::default()), loading: tokio::sync::Mutex::new(()), extra: Mutex::new(vec![]) }
    }
}

/// Addresses model servers open themselves: none, the gateway does it.
pub fn server_exposure(_s: &Settings) -> Vec<IpAddr> {
    vec![]
}

/// (Re)start the gateway from the current settings.
pub async fn restart(app: AppHandle) {
    let state = app.state::<AppState>();
    let _one = state.jit.restarting.lock().await;
    let old: Vec<JoinHandle<()>> = state.jit.listeners.lock().unwrap().drain(..).collect();
    for h in old {
        h.abort();
        let _ = h.await;
    }
    let s = state.settings();
    if !s.jit_api {
        stop_extra(&state).await;
    }
    *state.jit.status.lock().unwrap() = JitStatus { port: s.server_port, ..Default::default() };
    let router = Router::new().route("/health", get(|| async { "ok" })).route("/v1", get(models))
        .route("/v1/models", get(models))
        .route("/v1/models/{id}", get(model))
        .route("/props", get(props))
        .route("/v1/props", get(props))
        .fallback(call).with_state(app.clone());
    let ips: Vec<IpAddr> = std::iter::once(IpAddr::from([127, 0, 0, 1])).chain(s.exposed_ips()).collect();
    let port = s.server_port;
    let handles = ips
        .into_iter()
        .map(|ip| {
            let r = router.clone();
            let app = app.clone();
            // A model started before the gateway may still hold the port:
            // keep trying until it is stopped.
            tokio::spawn(async move {
                let url = format!("http://{ip}:{port}/v1");
                loop {
                    match tokio::net::TcpListener::bind((ip, port)).await {
                        Ok(l) => {
                            set_open(&app, &url, None);
                            let _ = axum::serve(l, r).await;
                            return;
                        }
                        Err(e) => set_open(&app, &url, Some(format!("{ip}:{port}: {e}"))),
                    }
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
            })
        })
        .collect();
    *state.jit.listeners.lock().unwrap() = handles;
}

fn set_open(app: &AppHandle, url: &str, error: Option<String>) {
    let state = app.state::<AppState>();
    let mut st = state.jit.status.lock().unwrap();
    match error {
        None => {
            if !st.urls.iter().any(|u| u == url) {
                st.urls.push(url.to_string());
            }
            st.running = true;
            st.error = None;
        }
        Some(e) => st.error = Some(e),
    }
}

pub fn status(state: &AppState) -> JitStatus {
    state.jit.status.lock().unwrap().clone()
}

/// Models loaded beside the main one.
pub fn loaded(state: &AppState) -> Vec<JitLoaded> {
    state
        .jit
        .extra
        .lock()
        .unwrap()
        .iter()
        .map(|x| JitLoaded { name: model_name(&x.path), port: x.server.status().port })
        .collect()
}

pub async fn stop_extra(state: &AppState) {
    let all: Vec<Extra> = state.jit.extra.lock().unwrap().drain(..).collect();
    for x in all {
        x.server.stop().await;
    }
}

/// The id a model goes by in `/v1/models`: its file name without the
/// `.gguf` or split suffix (as the Models page shows it).
fn model_name(p: &Path) -> String {
    let n = p.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_else(|| p.display().to_string());
    match gguf::split_parts(&n) {
        Some((stem, _, _)) => stem.to_string(),
        None => n.trim_end_matches(".gguf").trim_end_matches(".GGUF").to_string(),
    }
}

fn oai_error(code: StatusCode, message: impl Into<String>) -> Response {
    (code, Json(json!({"error": {"message": message.into(), "type": "invalid_request_error"}}))).into_response()
}

fn authorized(s: &Settings, headers: &HeaderMap) -> bool {
    let Some(key) = s.server_key() else { return true };
    let sent = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).map(str::trim);
    sent == Some(key.as_str())
}

/// `GET /v1/models`, also answered on `GET /v1`.
async fn models(AxState(app): AxState<AppHandle>, headers: HeaderMap) -> Response {
    let state = app.state::<AppState>();
    if !authorized(&state.settings(), &headers) {
        return oai_error(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    Json(json!({"object": "list", "data": model_ids(&state).into_iter().map(model_obj).collect::<Vec<_>>()})).into_response()
}

fn model_obj(id: String) -> Value {
    json!({"id": id, "object": "model", "created": 0, "owned_by": "openmodel"})
}

/// With JIT every downloaded model, otherwise the ones running.
fn model_ids(state: &AppState) -> Vec<String> {
    let s = state.settings();
    let mut ids: Vec<String> = if s.jit_api { library::list(&s.models_dir, &data_dir()).into_iter().map(|m| m.name).collect() } else { vec![] };
    for x in state.jit.extra.lock().unwrap().iter() {
        let name = model_name(&x.path);
        if !ids.contains(&name) {
            ids.push(name);
        }
    }
    let st = state.llm.status();
    if let (ServerState::Ready, Some(m)) = (st.state, st.model.as_ref()) {
        let name = model_name(m);
        if !ids.contains(&name) {
            ids.push(name);
        }
    }
    ids
}

/// `GET /v1/models/{id}`: agents check the model they were given exists.
async fn model(AxState(app): AxState<AppHandle>, axum::extract::Path(id): axum::extract::Path<String>, headers: HeaderMap) -> Response {
    let state = app.state::<AppState>();
    if !authorized(&state.settings(), &headers) {
        return oai_error(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    match model_ids(&state).into_iter().find(|m| m.eq_ignore_ascii_case(id.trim())) {
        Some(m) => Json(model_obj(m)).into_response(),
        None => oai_error(StatusCode::NOT_FOUND, format!("model \"{id}\" not found (see GET /v1/models)")),
    }
}

/// llama.cpp's `/props` (context size and so on) of the running model.
async fn props(AxState(app): AxState<AppHandle>, method: Method, uri: Uri, headers: HeaderMap) -> Response {
    let state = app.state::<AppState>();
    if !authorized(&state.settings(), &headers) {
        return oai_error(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    let st = state.llm.status();
    if st.state != ServerState::Ready {
        return oai_error(StatusCode::NOT_FOUND, "no model is running");
    }
    forward(&state, method, &uri, &headers, Bytes::new(), st.port).await
}

/// Port of a running server for `wanted` (a model name or path), if any.
fn running_port(state: &AppState, wanted: &str) -> Option<u16> {
    let st = state.llm.status();
    if st.state == ServerState::Ready && st.model.as_deref().is_some_and(|m| matches(m, wanted)) {
        return Some(st.port);
    }
    state.jit.extra.lock().unwrap().iter().find(|x| matches(&x.path, wanted) && x.server.status().state == ServerState::Ready).map(|x| x.server.status().port)
}

fn matches(path: &Path, wanted: &str) -> bool {
    let w = wanted.trim();
    w.eq_ignore_ascii_case(&model_name(path)) || Path::new(w) == path || path.file_name().is_some_and(|f| f.to_string_lossy().eq_ignore_ascii_case(w))
}

async fn call(AxState(app): AxState<AppHandle>, method: Method, uri: Uri, headers: HeaderMap, body: Bytes) -> Response {
    let state = app.state::<AppState>();
    let s = state.settings();
    if !uri.path().starts_with("/v1/") {
        return oai_error(StatusCode::NOT_FOUND, "not found");
    }
    if !authorized(&s, &headers) {
        return oai_error(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    let wanted = serde_json::from_slice::<Value>(&body).ok().and_then(|v| v.get("model").and_then(Value::as_str).map(String::from)).unwrap_or_default();
    let port = if !s.jit_api || wanted.trim().is_empty() {
        // No model named: whatever runs now.
        let st = state.llm.status();
        match st.state {
            ServerState::Ready => st.port,
            _ if s.jit_api => return oai_error(StatusCode::BAD_REQUEST, "no model is loaded; name one in \"model\" (see GET /v1/models)"),
            _ => return oai_error(StatusCode::SERVICE_UNAVAILABLE, "no model is running; start one on the Run page, or turn on \"Keep the API on (JIT)\""),
        }
    } else if let Some(p) = running_port(&state, &wanted) {
        p
    } else {
        match load(&state, &wanted).await {
            Ok(p) => p,
            Err(e) => return oai_error(StatusCode::BAD_REQUEST, e),
        }
    };
    forward(&state, method, &uri, &headers, body, port).await
}

/// Load `wanted`, keeping what is loaded when the GPU has room for both.
async fn load(state: &AppState, wanted: &str) -> Res<u16> {
    let _one = state.jit.loading.lock().await;
    if let Some(p) = running_port(state, wanted) {
        return Ok(p); // another request loaded it meanwhile
    }
    let s = state.settings();
    let model = library::list(&s.models_dir, &data_dir())
        .into_iter()
        .find(|m| matches(&m.path, wanted))
        .ok_or_else(|| format!("model \"{wanted}\" is not downloaded (see GET /v1/models)"))?;
    let opts = RunOptions { path: model.path.clone(), ctx: None, custom_args: None, vision: true, engine: None };
    let machine = state.machine().await;

    let main_busy = matches!(state.llm.status().state, ServerState::Ready | ServerState::Starting);
    if !main_busy {
        stop_extra(state).await;
        launch_llama(state, &state.llm, &opts, machine).await?;
        return Ok(state.llm.status().port);
    }
    // Something is loaded: keep it only if the new model fits in the free VRAM.
    let (shape, _, mmproj) = shape_for_file(&opts.path, opts.vision).map_err(errs)?;
    let need = tiering::plan(&shape, &machine, &PlanOptions { safe_level: 0, ..base_options(&s, &opts, mmproj) }).gpu_bytes;
    let free = tokio::task::spawn_blocking(hardware::live_vram).await.unwrap_or_default().first().map(|(_, used, total)| total.saturating_sub(*used)).unwrap_or(0);
    if need > 0 && free >= need + machine.vram_reserve {
        let server = ManagedServer::new();
        let mut room = machine.clone();
        room.vram_bytes = free;
        launch_llama(state, &server, &opts, room).await?;
        let port = server.status().port;
        state.jit.extra.lock().unwrap().push(Extra { path: model.path, server });
        return Ok(port);
    }
    stop_extra(state).await;
    state.llm.stop().await;
    launch_llama(state, &state.llm, &opts, machine).await?;
    Ok(state.llm.status().port)
}

async fn forward(state: &AppState, method: Method, uri: &Uri, headers: &HeaderMap, body: Bytes, port: u16) -> Response {
    let url = format!("http://127.0.0.1:{port}{}", uri.path_and_query().map(|p| p.as_str()).unwrap_or("/"));
    let mut req = state.http.request(method, url).body(body);
    for (k, v) in headers {
        if k != header::HOST && k != header::CONTENT_LENGTH && k != header::CONNECTION {
            req = req.header(k, v);
        }
    }
    match req.send().await {
        Ok(r) => {
            let mut out = Response::builder().status(r.status().as_u16());
            for (k, v) in r.headers() {
                if k != header::CONTENT_LENGTH && k != header::TRANSFER_ENCODING && k != header::CONNECTION {
                    out = out.header(k, v);
                }
            }
            out.body(Body::from_stream(r.bytes_stream())).unwrap_or_else(|_| oai_error(StatusCode::BAD_GATEWAY, "bad response"))
        }
        Err(e) => oai_error(StatusCode::BAD_GATEWAY, format!("model server: {e}")),
    }
}
