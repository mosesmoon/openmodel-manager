//! Inference engines: llama.cpp (`llama-server`) and stable-diffusion.cpp
//! (`sd-server`). Installs official GitHub release builds, or imports a zip /
//! folder the user already has.

use crate::config::Settings;
use crate::download::{download_all, DownloadItem};
use crate::hardware::Backend;
use crate::ProgressFn;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    /// llama.cpp
    Llama,
    /// stable-diffusion.cpp
    Sd,
}

impl EngineKind {
    pub fn repo(&self) -> &'static str {
        match self {
            EngineKind::Llama => "ggml-org/llama.cpp",
            EngineKind::Sd => "leejet/stable-diffusion.cpp",
        }
    }
    pub fn binary(&self) -> &'static str {
        match self {
            EngineKind::Llama => "llama-server",
            EngineKind::Sd => "sd-server",
        }
    }
    pub fn dir_name(&self) -> &'static str {
        match self {
            EngineKind::Llama => "llama",
            EngineKind::Sd => "sd",
        }
    }
}

pub fn exe_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base.to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Asset {
    pub name: String,
    #[serde(rename = "browser_download_url")]
    pub url: String,
    #[serde(default)]
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Release {
    pub tag_name: String,
    pub assets: Vec<Asset>,
}

/// What `engine.json` records for an installed or imported engine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InstalledEngine {
    pub kind: EngineKind,
    pub version: String,
    pub backend: Backend,
    /// Full path to llama-server / sd-server.
    pub binary: PathBuf,
    /// Folder that holds this engine's manifest.
    pub dir: PathBuf,
    /// "github" or "import".
    pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target<'a> {
    pub os: &'a str,
    pub arch: &'a str,
}

impl Target<'static> {
    pub fn current() -> Target<'static> {
        Target { os: std::env::consts::OS, arch: std::env::consts::ARCH }
    }
}

fn name_os(n: &str) -> Option<&'static str> {
    if n.contains("-win-") || n.contains("windows") || n.contains("-win64") {
        Some("windows")
    } else if n.contains("ubuntu") || n.contains("linux") {
        Some("linux")
    } else if n.contains("macos") || n.contains("darwin") {
        Some("macos")
    } else {
        None
    }
}

fn name_arch(n: &str) -> &'static str {
    if n.contains("arm64") || n.contains("aarch64") {
        "aarch64"
    } else {
        "x86_64"
    }
}

/// Backend a release asset was built for, judged from its name.
pub fn name_backend(n: &str) -> Option<Backend> {
    let n = n.to_lowercase();
    if n.contains("opencl") || n.contains("kompute") || n.contains("openvino") || n.contains("xcframework") || n.contains("musa") {
        return None;
    }
    Some(if n.contains("cuda") || n.contains("-cu1") {
        Backend::Cuda
    } else if n.contains("vulkan") {
        Backend::Vulkan
    } else if n.contains("rocm") || n.contains("hip") {
        Backend::Rocm
    } else if n.contains("sycl") {
        Backend::Sycl
    } else if name_os(&n) == Some("macos") {
        Backend::Metal
    } else {
        Backend::Cpu
    })
}

/// CUDA (major, minor) in an asset name (`cuda-12.4`, `cuda12`, `cu12`).
pub fn cuda_version(n: &str) -> Option<(u32, u32)> {
    let re = regex::Regex::new(r"(?i)(?:cuda-?|cu)(\d{2})(?:\.(\d+))?").unwrap();
    let c = re.captures(n)?;
    Some((c[1].parse().ok()?, c.get(2).and_then(|m| m.as_str().parse().ok()).unwrap_or(0)))
}

fn is_archive(n: &str) -> bool {
    n.ends_with(".zip") || n.ends_with(".tar.gz") || n.ends_with(".tgz")
}

/// Choose the main asset (and the CUDA runtime archive on Windows) for this
/// machine. `max_cuda` is the highest CUDA major the driver supports.
pub fn pick_assets(assets: &[Asset], t: Target, backend: Backend, max_cuda: Option<u32>) -> Option<(Asset, Option<Asset>)> {
    let mut best: Option<(i64, &Asset)> = None;
    for a in assets {
        let n = a.name.to_lowercase();
        if !is_archive(&n) || n.starts_with("cudart") || name_os(&n) != Some(t.os) || name_arch(&n) != t.arch {
            continue;
        }
        if name_backend(&n) != Some(backend) {
            continue;
        }
        let mut score = 100i64;
        if backend == Backend::Cuda {
            let (maj, min) = cuda_version(&n).unwrap_or((12, 0));
            let limit = max_cuda.unwrap_or(12);
            if maj > limit {
                continue;
            }
            score += (maj * 100 + min) as i64;
        }
        if backend == Backend::Cpu {
            // stable-diffusion.cpp ships avx / avx2 / avx512 / noavx builds.
            if n.contains("avx512") || n.contains("noavx") {
                score -= 20;
            } else if n.contains("avx2") {
                score += 10;
            }
        }
        if best.map(|(s, _)| score > s).unwrap_or(true) {
            best = Some((score, a));
        }
    }
    let (_, main) = best?;
    let mut runtime = None;
    if backend == Backend::Cuda && t.os == "windows" {
        let want = cuda_version(&main.name).map(|v| v.0);
        runtime = assets
            .iter()
            .filter(|a| {
                let n = a.name.to_lowercase();
                n.starts_with("cudart") && is_archive(&n) && name_arch(&n) == t.arch && cuda_version(&n).map(|v| v.0) == want
            })
            .max_by_key(|a| cuda_version(&a.name))
            .cloned();
    }
    Some((main.clone(), runtime))
}

/// Highest CUDA major version supported by the installed NVIDIA driver.
pub fn driver_cuda_major() -> Option<u32> {
    let out = std::process::Command::new("nvidia-smi").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let re = regex::Regex::new(r"CUDA Version:\s*(\d+)").unwrap();
    re.captures(&s)?[1].parse().ok()
}

pub async fn latest_release(http: &reqwest::Client, github_api: &str, repo: &str) -> Result<Release> {
    let url = format!("{}/repos/{}/releases/latest", github_api.trim_end_matches('/'), repo);
    let r = http.get(&url).header("Accept", "application/vnd.github+json").send().await?;
    if r.status().as_u16() == 403 {
        bail!("GitHub API rate limit reached; try again later");
    }
    if !r.status().is_success() {
        bail!("GitHub release lookup for {repo} failed: HTTP {}", r.status());
    }
    Ok(r.json().await?)
}

/// Extract a .zip or .tar.gz into `dest`. Entries escaping `dest` are rejected.
pub fn extract_archive(archive: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    let name = archive.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_lowercase();
    let f = std::fs::File::open(archive).with_context(|| format!("open {}", archive.display()))?;
    if name.ends_with(".zip") {
        let mut z = zip::ZipArchive::new(f)?;
        for i in 0..z.len() {
            let mut e = z.by_index(i)?;
            let Some(rel) = e.enclosed_name() else { bail!("unsafe path in archive: {}", e.name()) };
            let out = dest.join(rel);
            if e.is_dir() {
                std::fs::create_dir_all(&out)?;
                continue;
            }
            if let Some(p) = out.parent() {
                std::fs::create_dir_all(p)?;
            }
            let mut w = std::fs::File::create(&out)?;
            std::io::copy(&mut e, &mut w)?;
            #[cfg(unix)]
            if let Some(mode) = e.unix_mode() {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&out, std::fs::Permissions::from_mode(mode))?;
            }
        }
    } else if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        let mut t = tar::Archive::new(flate2::read::GzDecoder::new(f));
        for e in t.entries()? {
            let mut e = e?;
            // unpack_in refuses paths that escape `dest`.
            if !e.unpack_in(dest)? {
                bail!("unsafe path in archive: {}", e.path()?.display());
            }
        }
    } else {
        bail!("unsupported archive type: {name}");
    }
    Ok(())
}

/// Find `name` (plus .exe on Windows) anywhere under `dir`, shallowest first.
pub fn find_binary(dir: &Path, base: &str) -> Option<PathBuf> {
    let want = exe_name(base);
    let mut queue = std::collections::VecDeque::from([(dir.to_path_buf(), 0)]);
    while let Some((d, depth)) = queue.pop_front() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        let mut subdirs = Vec::new();
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                subdirs.push(p);
            } else if p.file_name().and_then(|n| n.to_str()).map(|n| n.eq_ignore_ascii_case(&want)).unwrap_or(false) {
                return Some(p);
            }
        }
        if depth < 6 {
            subdirs.sort();
            queue.extend(subdirs.into_iter().map(|p| (p, depth + 1)));
        }
    }
    None
}

#[cfg(unix)]
fn make_executable(dir: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let Some(bin) = find_binary(dir, "llama-server").or_else(|| find_binary(dir, "sd-server")) else { return };
    if let Some(parent) = bin.parent() {
        for e in std::fs::read_dir(parent).into_iter().flatten().flatten() {
            let p = e.path();
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            if p.is_file() && (n.starts_with("llama-") || n.starts_with("sd-") || n == "sd" || n == "rpc-server") {
                let _ = std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755));
            }
        }
    }
}

fn kind_dir(s: &Settings, kind: EngineKind) -> PathBuf {
    s.engines_dir.join(kind.dir_name())
}

fn write_manifest(e: &InstalledEngine) -> Result<()> {
    std::fs::create_dir_all(&e.dir)?;
    std::fs::write(e.dir.join("engine.json"), serde_json::to_vec_pretty(e)?)?;
    Ok(())
}

/// Download and install the latest release for `backend`.
pub async fn install(s: &Settings, kind: EngineKind, backend: Backend, cancel: &CancellationToken, progress: &ProgressFn) -> Result<InstalledEngine> {
    let http = crate::http_client();
    let rel = latest_release(&http, &s.github_api, kind.repo()).await?;
    let max_cuda = if backend == Backend::Cuda { driver_cuda_major() } else { None };
    let Some((main, runtime)) = pick_assets(&rel.assets, Target::current(), backend, max_cuda) else {
        bail!("{} {} has no {} build for {}/{}", kind.repo(), rel.tag_name, backend.as_str(), std::env::consts::OS, std::env::consts::ARCH);
    };
    let dl_dir = s.engines_dir.join("_downloads");
    let mut items = vec![DownloadItem { url: s.github_download_url(&main.url), dest: dl_dir.join(&main.name), size: Some(main.size).filter(|n| *n > 0) }];
    if let Some(r) = &runtime {
        items.push(DownloadItem { url: s.github_download_url(&r.url), dest: dl_dir.join(&r.name), size: Some(r.size).filter(|n| *n > 0) });
    }
    download_all(&http, &items, &[], cancel, &format!("engine-{}", kind.dir_name()), progress).await?;

    let dir = kind_dir(s, kind).join(format!("{}-{}", rel.tag_name, backend.as_str()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    for it in &items {
        extract_archive(&it.dest, &dir)?;
    }
    #[cfg(unix)]
    make_executable(&dir);
    let binary = find_binary(&dir, kind.binary()).with_context(|| format!("{} not found in {}", kind.binary(), main.name))?;
    let e = InstalledEngine { kind, version: rel.tag_name, backend, binary, dir, source: "github".into() };
    write_manifest(&e)?;
    for it in &items {
        let _ = std::fs::remove_file(&it.dest);
    }
    Ok(e)
}

/// Unpack a release zip / tar.gz the user downloaded themselves.
pub fn import_archive(s: &Settings, kind: EngineKind, archive: &Path) -> Result<InstalledEngine> {
    let name = archive.file_name().and_then(|n| n.to_str()).unwrap_or("engine").to_string();
    let backend = name_backend(&name).unwrap_or(Backend::Cpu);
    let stem = name.trim_end_matches(".zip").trim_end_matches(".tar.gz").trim_end_matches(".tgz");
    let dir = kind_dir(s, kind).join(format!("import-{stem}"));
    extract_archive(archive, &dir)?;
    #[cfg(unix)]
    make_executable(&dir);
    let binary = find_binary(&dir, kind.binary()).with_context(|| format!("{} not found in {name}", kind.binary()))?;
    let version = regex::Regex::new(r"b\d{4,}|master-[0-9a-f]+").unwrap().find(&name).map(|m| m.as_str().to_string()).unwrap_or_else(|| "imported".into());
    let e = InstalledEngine { kind, version, backend, binary, dir, source: "import".into() };
    write_manifest(&e)?;
    Ok(e)
}

/// Register an existing folder (e.g. a self-built llama.cpp) without copying it.
pub fn import_dir(s: &Settings, kind: EngineKind, folder: &Path, backend: Option<Backend>) -> Result<InstalledEngine> {
    let binary = find_binary(folder, kind.binary()).with_context(|| format!("{} not found under {}", kind.binary(), folder.display()))?;
    let id = uuid::Uuid::new_v4().simple().to_string();
    let dir = kind_dir(s, kind).join(format!("external-{}", &id[..8]));
    let backend = backend.or_else(|| name_backend(&folder.to_string_lossy())).unwrap_or(Backend::Cpu);
    let e = InstalledEngine { kind, version: "external".into(), backend, binary, dir, source: "import".into() };
    write_manifest(&e)?;
    Ok(e)
}

/// All installed engines of `kind`, newest first; missing binaries are skipped.
pub fn list_installed(s: &Settings, kind: EngineKind) -> Vec<InstalledEngine> {
    let mut out: Vec<(std::time::SystemTime, InstalledEngine)> = std::fs::read_dir(kind_dir(s, kind))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let m = e.path().join("engine.json");
            let t = std::fs::metadata(&m).and_then(|x| x.modified()).ok()?;
            let eng: InstalledEngine = serde_json::from_slice(&std::fs::read(&m).ok()?).ok()?;
            eng.binary.exists().then_some((t, eng))
        })
        .collect();
    out.sort_by_key(|e| std::cmp::Reverse(e.0));
    out.into_iter().map(|(_, e)| e).collect()
}

pub fn remove(e: &InstalledEngine) -> Result<()> {
    // Only delete folders we own; external imports just lose their manifest.
    if e.source == "import" && e.version == "external" {
        std::fs::remove_file(e.dir.join("engine.json"))?;
        let _ = std::fs::remove_dir(&e.dir);
    } else {
        std::fs::remove_dir_all(&e.dir)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLAMA: &[&str] = &[
        "llama-b11669-bin-win-cuda-12.4-x64.zip",
        "llama-b11669-bin-win-cuda-13.4-x64.zip",
        "llama-b11669-bin-win-vulkan-x64.zip",
        "llama-b11669-bin-win-cpu-x64.zip",
        "llama-b11669-bin-win-cpu-arm64.zip",
        "llama-b11669-bin-win-rocm-10.0-x64.zip",
        "llama-b11669-bin-win-opencl-adreno-arm64.zip",
        "llama-b11669-bin-ubuntu-x64.tar.gz",
        "llama-b11669-bin-ubuntu-vulkan-x64.tar.gz",
        "llama-b11669-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "llama-b11669-bin-ubuntu-rocm-10.0-x64.tar.gz",
        "llama-b11669-bin-macos-arm64.tar.gz",
        "llama-b11669-bin-macos-x64.tar.gz",
        "llama-b11669-xcframework.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-13.4-x64.zip",
    ];

    fn assets(names: &[&str]) -> Vec<Asset> {
        names.iter().map(|n| Asset { name: n.to_string(), url: format!("https://example/{n}"), size: 1 }).collect()
    }

    fn pick(os: &str, arch: &str, b: Backend, cuda: Option<u32>) -> Option<(String, Option<String>)> {
        pick_assets(&assets(LLAMA), Target { os, arch }, b, cuda).map(|(m, r)| (m.name, r.map(|r| r.name)))
    }

    #[test]
    fn windows_choices() {
        assert_eq!(
            pick("windows", "x86_64", Backend::Cuda, None),
            Some(("llama-b11669-bin-win-cuda-12.4-x64.zip".into(), Some("cudart-llama-bin-win-cuda-12.4-x64.zip".into())))
        );
        assert_eq!(
            pick("windows", "x86_64", Backend::Cuda, Some(13)),
            Some(("llama-b11669-bin-win-cuda-13.4-x64.zip".into(), Some("cudart-llama-bin-win-cuda-13.4-x64.zip".into())))
        );
        assert_eq!(pick("windows", "x86_64", Backend::Vulkan, None).unwrap().0, "llama-b11669-bin-win-vulkan-x64.zip");
        assert_eq!(pick("windows", "x86_64", Backend::Rocm, None).unwrap().0, "llama-b11669-bin-win-rocm-10.0-x64.zip");
        assert_eq!(pick("windows", "x86_64", Backend::Cpu, None).unwrap().0, "llama-b11669-bin-win-cpu-x64.zip");
        assert_eq!(pick("windows", "aarch64", Backend::Cpu, None).unwrap().0, "llama-b11669-bin-win-cpu-arm64.zip");
    }

    #[test]
    fn linux_and_mac_choices() {
        assert_eq!(pick("linux", "x86_64", Backend::Vulkan, None).unwrap().0, "llama-b11669-bin-ubuntu-vulkan-x64.tar.gz");
        assert_eq!(pick("linux", "x86_64", Backend::Rocm, None).unwrap().0, "llama-b11669-bin-ubuntu-rocm-10.0-x64.tar.gz");
        assert_eq!(pick("linux", "x86_64", Backend::Cpu, None).unwrap().0, "llama-b11669-bin-ubuntu-x64.tar.gz");
        assert_eq!(pick("linux", "x86_64", Backend::Cuda, None), Some(("llama-b11669-bin-ubuntu-cuda-12.8-x64.tar.gz".into(), None)));
        assert_eq!(pick("macos", "aarch64", Backend::Metal, None).unwrap().0, "llama-b11669-bin-macos-arm64.tar.gz");
        assert_eq!(pick("linux", "aarch64", Backend::Vulkan, None), None);
    }

    #[test]
    fn sd_names() {
        let sd = assets(&[
            "sd-master-1a2b3c4-bin-win-avx2-x64.zip",
            "sd-master-1a2b3c4-bin-win-avx512-x64.zip",
            "sd-master-1a2b3c4-bin-win-cuda12-x64.zip",
            "sd-master-1a2b3c4-bin-win-vulkan-x64.zip",
            "sd-master-1a2b3c4-bin-Linux-Ubuntu-24.04-x86_64.zip",
            "sd-master-1a2b3c4-bin-Darwin-macOS-15.6-arm64.zip",
            "cudart-sd-bin-win-cu12-x64.zip",
        ]);
        let w = |b| pick_assets(&sd, Target { os: "windows", arch: "x86_64" }, b, None).map(|(m, r)| (m.name, r.map(|r| r.name)));
        assert_eq!(w(Backend::Cuda), Some(("sd-master-1a2b3c4-bin-win-cuda12-x64.zip".into(), Some("cudart-sd-bin-win-cu12-x64.zip".into()))));
        assert_eq!(w(Backend::Cpu).unwrap().0, "sd-master-1a2b3c4-bin-win-avx2-x64.zip");
        let m = pick_assets(&sd, Target { os: "macos", arch: "aarch64" }, Backend::Metal, None).unwrap();
        assert!(m.0.name.contains("Darwin"));
        let l = pick_assets(&sd, Target { os: "linux", arch: "x86_64" }, Backend::Cpu, None).unwrap();
        assert!(l.0.name.contains("Linux"));
    }

    #[test]
    fn import_zip_and_list() {
        use std::io::Write;
        let tmp = tempfile::tempdir().unwrap();
        let s = Settings { engines_dir: tmp.path().join("engines"), ..Settings::default() };
        let zpath = tmp.path().join("llama-b9999-bin-win-vulkan-x64.zip");
        {
            let mut z = zip::ZipWriter::new(std::fs::File::create(&zpath).unwrap());
            let o = zip::write::SimpleFileOptions::default();
            z.start_file(format!("build/bin/{}", exe_name("llama-server")), o).unwrap();
            z.write_all(b"fake").unwrap();
            z.start_file("build/bin/ggml.dll", o).unwrap();
            z.finish().unwrap();
        }
        let e = import_archive(&s, EngineKind::Llama, &zpath).unwrap();
        assert_eq!(e.version, "b9999");
        assert_eq!(e.backend, Backend::Vulkan);
        assert!(e.binary.ends_with(format!("build/bin/{}", exe_name("llama-server"))));
        let list = list_installed(&s, EngineKind::Llama);
        assert_eq!(list, vec![e.clone()]);
        let ext = import_dir(&s, EngineKind::Llama, &e.dir, None).unwrap();
        assert_eq!(list_installed(&s, EngineKind::Llama).len(), 2);
        remove(&ext).unwrap();
        assert!(e.binary.exists(), "removing an external import keeps the files");
        assert_eq!(list_installed(&s, EngineKind::Llama).len(), 1);
    }

    #[test]
    fn rejects_zip_slip() {
        use std::io::Write;
        let tmp = tempfile::tempdir().unwrap();
        let zpath = tmp.path().join("evil.zip");
        {
            let mut z = zip::ZipWriter::new(std::fs::File::create(&zpath).unwrap());
            z.start_file("../escape.txt", zip::write::SimpleFileOptions::default()).unwrap();
            z.write_all(b"x").unwrap();
            z.finish().unwrap();
        }
        assert!(extract_archive(&zpath, &tmp.path().join("out")).is_err());
        assert!(!tmp.path().join("escape.txt").exists());
    }
}
