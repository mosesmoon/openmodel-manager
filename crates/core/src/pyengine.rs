//! Python-based engines: vLLM (Linux) and MLX (`mlx-lm`, macOS on Apple
//! Silicon). Both are installed from PyPI into their own virtual environment
//! with `uv` (Astral's single-binary installer, fetched from its GitHub
//! release and checked against the published SHA-256). `uv` also fetches the
//! Python interpreter, so the user needs no system Python. Everything lives
//! under `<engines>/python`.

use crate::config::Settings;
use crate::download::{download_all, DownloadItem};
use crate::engine::{extract_archive, find_binary, recent_releases};
use crate::{Progress, ProgressFn};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::io::AsyncBufReadExt;
use tokio_util::sync::CancellationToken;

const UV_REPO: &str = "astral-sh/uv";
/// vLLM's official ROCm wheel index (x86_64, Python 3.12; the ROCm release
/// it targets is in each wheel's local version, e.g. `+rocm100`).
pub const VLLM_ROCM_INDEX: &str = "https://wheels.vllm.ai/rocm/";
const PYTHON: &str = "3.12";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum PyKind {
    Vllm,
    Mlx,
}

impl PyKind {
    pub const ALL: [PyKind; 2] = [PyKind::Vllm, PyKind::Mlx];

    pub fn package(&self) -> &'static str {
        match self {
            PyKind::Vllm => "vllm",
            PyKind::Mlx => "mlx-lm",
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            PyKind::Vllm => "vLLM",
            PyKind::Mlx => "MLX (mlx-lm)",
        }
    }
    fn dir_name(&self) -> &'static str {
        match self {
            PyKind::Vllm => "vllm",
            PyKind::Mlx => "mlx",
        }
    }

    /// Why this engine cannot run on `os`/`arch`, or None when it can.
    pub fn unsupported_reason(&self, os: &str, arch: &str) -> Option<&'static str> {
        match self {
            PyKind::Mlx if !(os == "macos" && arch == "aarch64") => Some("macos-arm64"),
            PyKind::Vllm if !(os == "linux" && (arch == "x86_64" || arch == "aarch64")) => Some("linux"),
            _ => None,
        }
    }
    pub fn supported_here(&self) -> bool {
        self.unsupported_reason(std::env::consts::OS, std::env::consts::ARCH).is_none()
    }
}

/// GPU build of an engine. Only vLLM has more than one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Variant {
    /// PyPI's default build (CUDA for vLLM, Metal for MLX).
    #[default]
    Default,
    /// vLLM for AMD GPUs, from vLLM's ROCm wheel index.
    Rocm,
}

/// What `pyengine.json` records for an installed engine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PyInstall {
    pub kind: PyKind,
    #[serde(default)]
    pub variant: Variant,
    pub version: String,
    /// The virtual environment's Python.
    pub python: PathBuf,
    pub dir: PathBuf,
}

fn root(s: &Settings) -> PathBuf {
    s.engines_dir.join("python")
}

fn venv_python(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

pub fn installed(s: &Settings, kind: PyKind) -> Option<PyInstall> {
    let m = root(s).join(kind.dir_name()).join("pyengine.json");
    let i: PyInstall = serde_json::from_slice(&std::fs::read(m).ok()?).ok()?;
    i.python.exists().then_some(i)
}

pub fn remove(i: &PyInstall) -> Result<()> {
    std::fs::remove_dir_all(&i.dir)?;
    Ok(())
}

/// `uv`'s release asset for this machine.
pub fn uv_asset() -> Option<&'static str> {
    Some(match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => "uv-x86_64-pc-windows-msvc.zip",
        ("linux", "x86_64") => "uv-x86_64-unknown-linux-gnu.tar.gz",
        ("linux", "aarch64") => "uv-aarch64-unknown-linux-gnu.tar.gz",
        ("macos", "aarch64") => "uv-aarch64-apple-darwin.tar.gz",
        ("macos", "x86_64") => "uv-x86_64-apple-darwin.tar.gz",
        _ => return None,
    })
}

fn sha256_file(p: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h)?;
    Ok(format!("{:x}", h.finalize()))
}

/// The `uv` binary, downloading the latest release on first use.
pub async fn ensure_uv(s: &Settings, cancel: &CancellationToken, job_id: &str, progress: &ProgressFn) -> Result<PathBuf> {
    let dir = root(s).join("uv");
    if let Some(b) = find_binary(&dir, "uv") {
        return Ok(b);
    }
    let asset_name = uv_asset().context("uv has no build for this system")?;
    let http = crate::http_client();
    let releases = recent_releases(&http, &s.github_api, UV_REPO).await?;
    let rel = releases.iter().find(|r| !r.draft && r.assets.iter().any(|a| a.name == asset_name)).context("no uv release for this system")?;
    let asset = rel.assets.iter().find(|a| a.name == asset_name).unwrap();
    let sum = rel.assets.iter().find(|a| a.name == format!("{asset_name}.sha256")).context("uv release has no checksum file")?;
    let want = http.get(s.github_download_url(&sum.url)).send().await?.error_for_status()?.text().await?;
    let want = want.split_whitespace().next().unwrap_or_default().to_lowercase();
    let dl = s.engines_dir.join("_downloads").join(asset_name);
    download_all(&http, &[DownloadItem { url: s.github_download_url(&asset.url), dest: dl.clone(), size: Some(asset.size).filter(|n| *n > 0) }], &[], cancel, job_id, progress).await?;
    let got = sha256_file(&dl)?;
    if got != want {
        let _ = std::fs::remove_file(&dl);
        bail!("{asset_name} does not match its published checksum");
    }
    extract_archive(&dl, &dir)?;
    let _ = std::fs::remove_file(&dl);
    let uv = find_binary(&dir, "uv").context("uv not found in its archive")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&uv, std::fs::Permissions::from_mode(0o755))?;
    }
    Ok(uv)
}

/// Environment for `uv`: keep its Python builds and cache under our folder.
fn uv_env(s: &Settings) -> Vec<(String, String)> {
    let r = root(s);
    vec![
        ("UV_PYTHON_INSTALL_DIR".into(), r.join("pythons").display().to_string()),
        ("UV_CACHE_DIR".into(), r.join("cache").display().to_string()),
        ("UV_NO_PROGRESS".into(), "1".into()),
    ]
}

/// Run a command, reporting each output line as the job's stage text.
async fn run_logged(mut cmd: tokio::process::Command, cancel: &CancellationToken, job_id: &str, progress: &ProgressFn, what: &str) -> Result<String> {
    cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).kill_on_drop(true);
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);
    let mut child = cmd.spawn().with_context(|| format!("start {what}"))?;
    let out = child.stdout.take().unwrap();
    let err = child.stderr.take().unwrap();
    let mut lines = tokio_stream_lines(out, err);
    let mut tail: Vec<String> = vec![];
    loop {
        tokio::select! {
            l = lines.recv() => match l {
                Some(line) => {
                    progress(Progress { stage: format!("{what}: {}", crate::tools::truncate(&line, 160)), ..Progress::new(job_id, "") });
                    tail.push(line);
                    if tail.len() > 40 { tail.remove(0); }
                }
                None => break,
            },
            _ = cancel.cancelled() => {
                let _ = child.kill().await;
                bail!(crate::download::Cancelled);
            }
        }
    }
    let st = child.wait().await?;
    if !st.success() {
        bail!("{what} failed (exit {:?}):\n{}", st.code(), tail.join("\n"));
    }
    Ok(tail.join("\n"))
}

/// Merge two output pipes into one channel of lines.
fn tokio_stream_lines(a: tokio::process::ChildStdout, b: tokio::process::ChildStderr) -> tokio::sync::mpsc::Receiver<String> {
    let (tx, rx) = tokio::sync::mpsc::channel(256);
    let tx2 = tx.clone();
    tokio::spawn(async move {
        let mut l = tokio::io::BufReader::new(a).lines();
        while let Ok(Some(x)) = l.next_line().await {
            let _ = tx.send(x).await;
        }
    });
    tokio::spawn(async move {
        let mut l = tokio::io::BufReader::new(b).lines();
        while let Ok(Some(x)) = l.next_line().await {
            let _ = tx2.send(x).await;
        }
    });
    rx
}

fn uv_cmd(s: &Settings, uv: &Path) -> tokio::process::Command {
    let mut c = tokio::process::Command::new(uv);
    c.envs(uv_env(s));
    c
}

/// Version of `package` installed in the venv (from `uv pip show`).
pub fn parse_pip_show(out: &str) -> Option<String> {
    out.lines().find_map(|l| l.strip_prefix("Version:").map(|v| v.trim().to_string()))
}

/// `uv pip install` arguments for the latest release of `kind`/`variant`.
pub fn install_args(kind: PyKind, variant: Variant, python: &Path) -> Vec<String> {
    let mut a: Vec<String> = vec!["pip".into(), "install".into(), "--upgrade".into(), "--python".into(), python.display().to_string()];
    if kind == PyKind::Vllm && variant == Variant::Rocm {
        // uv prefers an extra index over PyPI, so vLLM and its ROCm torch come from here.
        a.extend(["--extra-index-url".into(), VLLM_ROCM_INDEX.into()]);
    }
    a.push(kind.package().into());
    a
}

/// Install the latest release of the engine, or upgrade an existing one in
/// place. Refuses on systems the engine does not support.
pub async fn install(s: &Settings, kind: PyKind, variant: Variant, cancel: &CancellationToken, job_id: &str, progress: &ProgressFn) -> Result<PyInstall> {
    let variant = if kind == PyKind::Vllm { variant } else { Variant::Default };
    if !kind.supported_here() {
        bail!("{} does not run on this system", kind.name());
    }
    let uv = ensure_uv(s, cancel, job_id, progress).await?;
    let dir = root(s).join(kind.dir_name());
    let venv = dir.join("venv");
    let python = venv_python(&venv);
    if !python.exists() {
        let mut c = uv_cmd(s, &uv);
        c.args(["venv", "--python", PYTHON, "--seed"]).arg(&venv);
        run_logged(c, cancel, job_id, progress, "uv venv").await?;
    }
    if installed(s, kind).is_some_and(|i| i.variant != variant) {
        // Switching CUDA <-> ROCm: start from a clean environment.
        std::fs::remove_dir_all(&venv)?;
        let mut c = uv_cmd(s, &uv);
        c.args(["venv", "--python", PYTHON, "--seed"]).arg(&venv);
        run_logged(c, cancel, job_id, progress, "uv venv").await?;
    }
    let mut c = uv_cmd(s, &uv);
    c.args(install_args(kind, variant, &python));
    run_logged(c, cancel, job_id, progress, kind.package()).await?;
    let mut c = uv_cmd(s, &uv);
    c.args(["pip", "show", "--python"]).arg(&python).arg(kind.package());
    let shown = run_logged(c, cancel, job_id, progress, "uv pip show").await?;
    let version = parse_pip_show(&shown).unwrap_or_else(|| "unknown".into());
    let i = PyInstall { kind, variant, version, python, dir: dir.clone() };
    std::fs::write(dir.join("pyengine.json"), serde_json::to_vec_pretty(&i)?)?;
    Ok(i)
}

/// Newest release: from vLLM's ROCm index for the ROCm build (it runs ahead
/// of PyPI), else from PyPI.
pub async fn latest_version(kind: PyKind, variant: Variant) -> Result<String> {
    if kind == PyKind::Vllm && variant == Variant::Rocm {
        let page = crate::http_client().get(format!("{VLLM_ROCM_INDEX}vllm/")).send().await?.error_for_status()?.text().await?;
        return newest_rocm_release(&page).context("no vLLM release on the ROCm index");
    }
    let v: serde_json::Value = crate::http_client().get(format!("https://pypi.org/pypi/{}/json", kind.package())).send().await?.error_for_status()?.json().await?;
    v.pointer("/info/version").and_then(|x| x.as_str()).map(String::from).context("PyPI answer has no version")
}

/// Newest stable release named in a wheel index page (`vllm-0.33.0+rocm100-...whl`);
/// release candidates and dev builds are skipped.
pub fn newest_rocm_release(page: &str) -> Option<String> {
    let re = regex::Regex::new(r"vllm-(\d+\.\d+\.\d+)(?:\+|%2B)(rocm\d+)-").unwrap();
    re.captures_iter(page).map(|c| format!("{}+{}", &c[1], &c[2])).max_by(|a, b| if is_newer(a, b) { std::cmp::Ordering::Greater } else if is_newer(b, a) { std::cmp::Ordering::Less } else { std::cmp::Ordering::Equal })
}

/// True when `latest` is a newer release than `current` (numeric compare of
/// dotted parts; pre-release suffixes count as older).
pub fn is_newer(latest: &str, current: &str) -> bool {
    let parts = |v: &str| -> Vec<u64> { v.split(['.', '+', '-']).map(|p| p.chars().take_while(char::is_ascii_digit).collect::<String>().parse().unwrap_or(0)).collect() };
    parts(latest) > parts(current)
}

pub struct ServeOptions<'a> {
    /// Hugging Face model id or a local folder.
    pub model: &'a str,
    pub host: &'a str,
    pub port: u16,
    pub ctx: Option<u64>,
}

/// Command line (after the venv's Python) that serves an OpenAI-compatible API.
pub fn serve_args(kind: PyKind, o: &ServeOptions) -> Vec<String> {
    let mut a: Vec<String> = match kind {
        PyKind::Vllm => vec!["-m".into(), "vllm.entrypoints.openai.api_server".into(), "--model".into(), o.model.into()],
        PyKind::Mlx => vec!["-m".into(), "mlx_lm.server".into(), "--model".into(), o.model.into()],
    };
    a.extend(["--host".into(), o.host.into(), "--port".into(), o.port.to_string()]);
    if let (PyKind::Vllm, Some(c)) = (kind, o.ctx) {
        a.extend(["--max-model-len".into(), c.to_string()]);
    }
    a
}

/// Environment for serving: model downloads go to the models folder and use
/// the configured Hugging Face mirror and token.
pub fn serve_env(s: &Settings) -> Vec<(String, String)> {
    let mut e = vec![("HF_HOME".to_string(), s.models_dir.join("hf").display().to_string()), ("HF_ENDPOINT".to_string(), s.hf_endpoint.clone())];
    if let Some(t) = s.hf_token.as_deref().filter(|t| !t.is_empty()) {
        e.push(("HF_TOKEN".into(), t.into()));
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_gating() {
        assert_eq!(PyKind::Mlx.unsupported_reason("macos", "aarch64"), None);
        assert!(PyKind::Mlx.unsupported_reason("macos", "x86_64").is_some());
        assert!(PyKind::Mlx.unsupported_reason("windows", "x86_64").is_some());
        assert_eq!(PyKind::Vllm.unsupported_reason("linux", "x86_64"), None);
        assert!(PyKind::Vllm.unsupported_reason("windows", "x86_64").is_some());
        assert!(PyKind::Vllm.unsupported_reason("macos", "aarch64").is_some());
    }

    #[test]
    fn versions() {
        assert!(is_newer("0.32.1", "0.32.0"));
        assert!(is_newer("0.33.0", "0.32.9"));
        assert!(!is_newer("0.32.0", "0.32.0"));
        assert!(!is_newer("0.31.5", "0.32.0"));
        assert_eq!(parse_pip_show("Name: vllm\nVersion: 0.32.0\nLocation: x"), Some("0.32.0".into()));
        let page = r#"<a href="vllm-0.32.1+rocm100-cp312-cp312-manylinux_2_34_x86_64.whl">x</a>
<a href="vllm-0.33.0%2Brocm100-cp312-cp312-manylinux_2_34_x86_64.whl">y</a>
<a href="vllm-0.33.0rc3.dev3+g4ba787ec9.rocm100-cp312-cp312-manylinux_2_34_x86_64.whl">z</a>"#;
        assert_eq!(newest_rocm_release(page).as_deref(), Some("0.33.0+rocm100"));
        assert!(!is_newer("0.33.0+rocm100", "0.33.0+rocm100"));
    }

    #[test]
    fn rocm_uses_vllm_wheel_index() {
        let py = Path::new("/v/bin/python");
        let rocm = install_args(PyKind::Vllm, Variant::Rocm, py);
        assert!(rocm.windows(2).any(|w| w == ["--extra-index-url", VLLM_ROCM_INDEX]));
        assert_eq!(rocm.last().unwrap(), "vllm");
        assert!(!install_args(PyKind::Vllm, Variant::Default, py).contains(&"--extra-index-url".to_string()));
        assert!(!install_args(PyKind::Mlx, Variant::Rocm, py).contains(&"--extra-index-url".to_string()));
    }

    #[test]
    fn serve_command_lines() {
        let o = ServeOptions { model: "Qwen/Qwen3-8B", host: "127.0.0.1", port: 8000, ctx: Some(32768) };
        let v = serve_args(PyKind::Vllm, &o);
        assert_eq!(&v[..4], ["-m", "vllm.entrypoints.openai.api_server", "--model", "Qwen/Qwen3-8B"]);
        assert!(v.windows(2).any(|w| w == ["--max-model-len", "32768"]));
        let m = serve_args(PyKind::Mlx, &o);
        assert_eq!(&m[..2], ["-m", "mlx_lm.server"]);
        assert!(!m.contains(&"--max-model-len".to_string()));
        assert!(m.windows(2).any(|w| w == ["--port", "8000"]));
    }

    /// Downloads uv (~15 MB) and creates a Python venv; the engines themselves
    /// only install on their own systems.
    /// `cargo test -p openmodel-core pyengine::tests::live_uv -- --ignored`
    #[tokio::test]
    #[ignore]
    async fn live_uv() {
        let d = tempfile::tempdir().unwrap();
        let s = Settings { engines_dir: d.path().join("engines"), ..Settings::default() };
        let uv = ensure_uv(&s, &CancellationToken::new(), "t", &crate::noop_progress()).await.unwrap();
        let venv = d.path().join("venv");
        let mut c = uv_cmd(&s, &uv);
        c.args(["venv", "--python", PYTHON, "--seed"]).arg(&venv);
        run_logged(c, &CancellationToken::new(), "t", &crate::noop_progress(), "uv venv").await.unwrap();
        assert!(venv_python(&venv).exists());
        let v = latest_version(PyKind::Vllm, Variant::Default).await.unwrap();
        assert!(v.split('.').count() >= 2, "{v}");
        let r = latest_version(PyKind::Vllm, Variant::Rocm).await.unwrap();
        assert!(r.contains("+rocm"), "{r}");
    }
}
