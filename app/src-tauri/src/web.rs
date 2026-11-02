//! Web management page: the desktop UI served to browsers on other devices.
//!
//! The page is the same frontend the window shows (read from the app's
//! bundled assets). Its `invoke` calls arrive as `POST /api/invoke/<command>`
//! and run the same command functions; the events the window listens to are
//! streamed over `GET /api/events`. Everything under `/api` needs a session
//! cookie from `POST /api/login` with the password set in Settings.

use super::*;
use axum::extract::{Path as UrlPath, State as AxState};
use axum::http::{header, HeaderMap, StatusCode, Uri};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use std::collections::HashSet;
use std::convert::Infallible;
use std::net::IpAddr;
use tauri::Listener;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;

/// Events the window listens to; the web page gets the same ones.
const EVENTS: &[&str] = &["server", "progress", "job-done", "chat", "approval"];
const COOKIE: &str = "openmodel_session";

pub struct Web {
    /// One restart at a time, so two saves can't fight over the port.
    restarting: tokio::sync::Mutex<()>,
    listeners: Mutex<Vec<JoinHandle<()>>>,
    status: Mutex<WebStatus>,
    sessions: Mutex<HashSet<String>>,
    events: broadcast::Sender<(&'static str, String)>,
}

#[derive(Clone, Default, Serialize)]
pub struct WebStatus {
    running: bool,
    /// Addresses the page answers on.
    urls: Vec<String>,
    /// Why it is not running, or which address could not be opened.
    error: Option<String>,
}

impl Default for Web {
    fn default() -> Self {
        Web { restarting: tokio::sync::Mutex::new(()), listeners: Mutex::new(vec![]), status: Mutex::new(WebStatus::default()), sessions: Mutex::new(HashSet::new()), events: broadcast::channel(1024).0 }
    }
}

/// Copy the events the commands emit to the web page's stream.
pub fn relay_events(app: &AppHandle) {
    for &name in EVENTS {
        let tx = app.state::<AppState>().web.events.clone();
        app.listen_any(name, move |e| {
            let _ = tx.send((name, e.payload().to_string()));
        });
    }
}

/// (Re)start the page from the current settings; stops it when it is off.
pub async fn restart(app: AppHandle) {
    let state = app.state::<AppState>();
    let _one = state.web.restarting.lock().await;
    let old: Vec<JoinHandle<()>> = state.web.listeners.lock().unwrap().drain(..).collect();
    // Wait until the old sockets are closed, or binding the same port again fails.
    for h in old {
        h.abort();
        let _ = h.await;
    }
    let s = state.settings();
    let mut st = WebStatus::default();
    if !s.web_ui {
        *state.web.status.lock().unwrap() = st;
        return;
    }
    if s.web_password.is_none() {
        st.error = Some("set a password first".into());
        *state.web.status.lock().unwrap() = st;
        return;
    }
    let router = Router::new()
        .route("/api/session", get(session))
        .route("/api/login", post(login))
        .route("/api/logout", post(logout))
        .route("/api/invoke/{cmd}", post(invoke))
        .route("/api/events", get(events))
        .fallback(asset)
        .with_state(app.clone());
    let ips = std::iter::once(IpAddr::from([127, 0, 0, 1])).chain(s.web_ips());
    let mut errors = vec![];
    let mut handles = vec![];
    for ip in ips {
        match tokio::net::TcpListener::bind((ip, s.web_port)).await {
            Ok(l) => {
                st.urls.push(format!("http://{ip}:{}/", s.web_port));
                let r = router.clone();
                handles.push(tokio::spawn(async move {
                    let _ = axum::serve(l, r).await;
                }));
            }
            Err(e) => errors.push(format!("{ip}:{}: {e}", s.web_port)),
        }
    }
    st.running = !handles.is_empty();
    st.error = (!errors.is_empty()).then(|| errors.join("; "));
    *state.web.listeners.lock().unwrap() = handles;
    *state.web.status.lock().unwrap() = st;
}

pub fn status(state: &AppState) -> WebStatus {
    state.web.status.lock().unwrap().clone()
}

/// Set the password; signs every browser out.
pub fn set_password(state: &AppState, password: &str) -> Res<()> {
    if password.chars().count() < 6 {
        return Err("the password needs at least 6 characters".into());
    }
    let mut s = state.settings();
    s.web_password = Some(openmodel_core::config::hash_password(password));
    s.save().map_err(errs)?;
    *state.settings.write().unwrap() = s;
    state.web.sessions.lock().unwrap().clear();
    Ok(())
}

fn signed_in(app: &AppHandle, headers: &HeaderMap) -> bool {
    let token = headers.get_all(header::COOKIE).iter().filter_map(|v| v.to_str().ok()).flat_map(|v| v.split(';')).find_map(|c| c.trim().strip_prefix(COOKIE).and_then(|r| r.strip_prefix('=')));
    token.is_some_and(|t| app.state::<AppState>().web.sessions.lock().unwrap().contains(t))
}

fn error(code: StatusCode, msg: &str) -> Response {
    (code, Json(json!({ "error": msg }))).into_response()
}

async fn session(AxState(app): AxState<AppHandle>, headers: HeaderMap) -> StatusCode {
    if signed_in(&app, &headers) {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::UNAUTHORIZED
    }
}

#[derive(Deserialize)]
struct Login {
    password: String,
}

async fn login(AxState(app): AxState<AppHandle>, Json(l): Json<Login>) -> Response {
    let state = app.state::<AppState>();
    let ok = state.settings().web_password.is_some_and(|h| openmodel_core::config::verify_password(&h, &l.password));
    if !ok {
        tokio::time::sleep(Duration::from_secs(1)).await; // slow down guessing
        return error(StatusCode::UNAUTHORIZED, "wrong password");
    }
    let token = uuid::Uuid::new_v4().simple().to_string();
    state.web.sessions.lock().unwrap().insert(token.clone());
    ([(header::SET_COOKIE, format!("{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict"))], StatusCode::NO_CONTENT).into_response()
}

async fn logout(AxState(app): AxState<AppHandle>, headers: HeaderMap) -> Response {
    if let Some(t) = headers.get(header::COOKIE).and_then(|v| v.to_str().ok()).and_then(|v| v.split(';').find_map(|c| c.trim().strip_prefix(COOKIE).and_then(|r| r.strip_prefix('=')))) {
        app.state::<AppState>().web.sessions.lock().unwrap().remove(t);
    }
    ([(header::SET_COOKIE, format!("{COOKIE}=; Path=/; Max-Age=0"))], StatusCode::NO_CONTENT).into_response()
}

async fn invoke(AxState(app): AxState<AppHandle>, headers: HeaderMap, UrlPath(cmd): UrlPath<String>, Json(args): Json<Value>) -> Response {
    if !signed_in(&app, &headers) {
        return error(StatusCode::UNAUTHORIZED, "sign in first");
    }
    match dispatch(&app, &cmd, args).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, &e),
    }
}

async fn events(AxState(app): AxState<AppHandle>, headers: HeaderMap) -> Response {
    if !signed_in(&app, &headers) {
        return error(StatusCode::UNAUTHORIZED, "sign in first");
    }
    let rx = app.state::<AppState>().web.events.subscribe();
    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        loop {
            match rx.recv().await {
                Ok((name, payload)) => return Some((Ok::<_, Infallible>(Event::default().event(name).data(payload)), rx)),
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

/// The frontend's files, from the assets bundled into the app.
async fn asset(AxState(app): AxState<AppHandle>, uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let resolver = app.asset_resolver();
    // Unknown paths without an extension are the single-page app itself.
    let a = resolver.get(if path.is_empty() { "index.html".into() } else { path.to_string() }).or_else(|| (!path.contains('.')).then(|| resolver.get("index.html".into())).flatten());
    match a {
        Some(a) => ([(header::CONTENT_TYPE, a.mime_type().to_string())], a.bytes().to_vec()).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

fn arg<T: serde::de::DeserializeOwned>(args: &Value, name: &str) -> Res<T> {
    serde_json::from_value(args.get(name).cloned().unwrap_or(Value::Null)).map_err(|e| format!("{name}: {e}"))
}

fn out<T: Serialize>(v: T) -> Res<Value> {
    serde_json::to_value(v).map_err(err)
}

/// Run a command the way the window's `invoke` would.
async fn dispatch(app: &AppHandle, cmd: &str, a: Value) -> Res<Value> {
    let h = || app.clone();
    let st = || app.state::<AppState>();
    match cmd {
        "cancel_job" => out(cancel_job(st(), arg(&a, "id")?)),
        "get_settings" => out(get_settings(st())),
        "save_settings" => out(save_settings(h(), st(), arg(&a, "settings")?)?),
        "app_paths" => out(app_paths()),
        "get_hardware" => out(get_hardware(st(), arg(&a, "refresh")?).await?),
        "live_vram" => out(live_vram().await),
        "catalog" => out(catalog(st()).await?),
        "plan_preview" => out(plan_preview(st(), arg(&a, "opts")?).await?),
        "library_list" => out(library_list(st())),
        "library_import" => out(library_import(arg(&a, "path")?)?),
        "library_remove" => out(library_remove(st(), arg(&a, "path")?)?),
        "hf_search" => out(hf_search(st(), arg(&a, "query")?).await?),
        "hf_listing" => out(hf_listing(st(), arg(&a, "repo")?).await?),
        "hf_download" => out(hf_download(h(), st(), arg(&a, "repo")?, arg(&a, "quant")?, arg(&a, "mmproj")?)),
        "engines" => out(engines(st()).await?),
        "engine_install" => out(engine_install(h(), st(), arg(&a, "kind")?, arg(&a, "backend")?).await?),
        "engine_import" => out(engine_import(st(), arg(&a, "kind")?, arg(&a, "path")?)?),
        "engine_remove" => out(engine_remove(st(), arg(&a, "kind")?, arg(&a, "binary")?)?),
        "lan_addresses" => out(lan_addresses()),
        "py_engines" => out(py_engines(st()).await?),
        "py_engine_install" => out(py_engine_install(h(), st(), arg(&a, "kind")?, arg(&a, "variant")?).await?),
        "py_engine_latest" => out(py_engine_latest(st(), arg(&a, "kind")?, arg(&a, "variant")?).await?),
        "py_engine_remove" => out(py_engine_remove(st(), arg(&a, "kind")?).await?),
        "py_engine_start" => out(py_engine_start(st(), arg(&a, "kind")?, arg(&a, "model")?, arg(&a, "ctx")?).await?),
        "strata_install" => out(strata_install(h(), st())),
        "strata_remove" => out(strata_remove(st()).await?),
        "strata_managed" => out(strata_managed(st())),
        "strata_plan" => out(strata_plan(st(), arg(&a, "id")?, arg(&a, "set")?).await?),
        "strata_start_managed" => out(strata_start_managed(st(), arg(&a, "id")?, arg(&a, "set")?, arg(&a, "ctx")?).await?),
        "server_status" => out(server_status(st(), arg(&a, "which")?)),
        "server_logs" => out(server_logs(st(), arg(&a, "which")?, arg(&a, "last")?)),
        "server_stop" => out(server_stop(st(), arg(&a, "which")?).await?),
        "llm_start" => out(llm_start(st(), arg(&a, "opts")?).await?),
        "media_models" => out(media_models(st())),
        "media_download" => out(media_download(h(), st(), arg(&a, "id")?, arg(&a, "quant")?)?),
        "sd_start" => out(sd_start(st(), arg(&a, "id")?, arg(&a, "quant")?).await?),
        "image_generate" => out(image_generate(st(), arg(&a, "req")?).await?),
        "video_generate" => out(video_generate(h(), st(), arg(&a, "req")?)?),
        "read_media" => out(read_media(st(), arg(&a, "path")?)?),
        "chat_send" => out(chat_send(h(), st(), arg(&a, "session")?, arg(&a, "endpoint")?, arg(&a, "messages")?, arg(&a, "tools")?).await?),
        "agent_send" => out(agent_send(h(), st(), arg(&a, "session")?, arg(&a, "endpoint")?, arg(&a, "messages")?, arg(&a, "workspace")?).await?),
        "chat_stop" => {
            chat_stop(h(), st(), arg(&a, "session")?);
            Ok(Value::Null)
        }
        "approval_resolve" => out(approval_resolve(h(), st(), arg(&a, "session")?, arg(&a, "id")?, arg(&a, "decision")?)),
        "endpoint_models" => out(endpoint_models(st(), arg(&a, "endpoint")?).await?),
        "web_ui_status" => out(web_ui_status(st())),
        "web_ui_set_password" => out(web_ui_set_password(st(), arg(&a, "password")?)?),
        // Opens a window on the computer itself, not in the browser.
        "open_folder" => Err("not available in the web page".into()),
        _ => Err(format!("unknown command {cmd}")),
    }
}
