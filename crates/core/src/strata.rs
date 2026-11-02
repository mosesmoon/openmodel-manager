//! Strata (github.com/Niko1221/Strata): a third-party engine tuned for a few
//! large MoE models. We install its released Terminal package and drive it
//! through its own `strata launch` command; Strata then fetches and verifies
//! the engine for the GPU itself. Nothing of Strata's code is copied here.

use crate::config::Settings;
use crate::download::{download_all, DownloadItem};
use crate::engine::{extract_archive, recent_releases};
use crate::ProgressFn;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

pub const REPO: &str = "Niko1221/Strata";
/// Environment variable that carries the API key in LAN mode.
pub const KEY_ENV: &str = "OPENMODEL_STRATA_KEY";

/// What `strata.json` records for an installed Terminal package.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StrataInstall {
    pub version: String,
    /// `bin/strata.cmd` on Windows, `bin/strata` elsewhere.
    pub launcher: PathBuf,
    pub dir: PathBuf,
}

/// One line of the release's `install-manifest.txt`:
/// `<platform> <version> <file> <sha256> <size>`.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestEntry {
    pub version: String,
    pub file: String,
    pub sha256: String,
    pub size: u64,
}

/// Strata's name for this machine, or None where it ships no Terminal.
pub fn platform() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => Some("windows-x86_64"),
        ("linux", "x86_64") => Some("linux-x86_64"),
        ("linux", "aarch64") => Some("linux-aarch64"),
        ("macos", "aarch64") => Some("macos-arm64"),
        _ => None,
    }
}

pub fn parse_manifest(text: &str, platform: &str) -> Option<ManifestEntry> {
    text.lines().find_map(|l| {
        let f: Vec<&str> = l.split_whitespace().collect();
        (f.len() >= 5 && f[0] == platform).then(|| ManifestEntry { version: f[1].into(), file: f[2].into(), sha256: f[3].to_lowercase(), size: f[4].parse().unwrap_or(0) })
    })
}

/// Hugging Face packs Strata's docs list as supported (docs/MODELS.md).
/// Its DeepSeek sets need extra shared files that only Strata's own setup
/// fetches, so they are not here.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct OfficialPack {
    /// Normalised family name, as in `normalise`.
    pub family: &'static str,
    pub repo: &'static str,
    pub quants: &'static [&'static str],
}

pub const OFFICIAL: &[OfficialPack] = &[
    OfficialPack { family: "qwen3.8-flash-next", repo: "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", quants: &["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"] },
    OfficialPack { family: "qwen3.8-flash-next", repo: "unsloth/Qwen3.8-Flash-Next-GGUF", quants: &["UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"] },
    // Strata runs Qwen3.6 only from the edition that carries the MTP draft layer.
    OfficialPack { family: "qwen3.6-35b-a3b", repo: "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", quants: &["UD-Q4_K_XL", "UD-Q6_K_XL", "Q8_0"] },
];

fn normalise(name: &str) -> String {
    name.to_lowercase().replace(['_', ' '], "-")
}

/// The official pack for a catalog model: the one from the same repo, else
/// the family's pack from another repo (Qwen3.6 needs its MTP edition).
pub fn official_pack(id: &str, repo: &str) -> Option<&'static OfficialPack> {
    let n = normalise(id);
    OFFICIAL.iter().find(|p| p.repo.eq_ignore_ascii_case(repo)).or_else(|| OFFICIAL.iter().find(|p| n.contains(p.family)))
}

/// The model families Strata runs, judged from a file or folder path.
pub fn supports(name: &str) -> bool {
    let n = normalise(name);
    if n.contains("deepseekv4.1-flash") || n.contains("deepseek-v4.1-flash") || n.contains("qwen3.8-flash-next") {
        return true;
    }
    n.contains("qwen3.6-35b-a3b") && n.contains("mtp")
}

fn root(s: &Settings) -> PathBuf {
    s.engines_dir.join("strata")
}

/// Strata's own data folder (profiles, engine packages, presets), kept apart
/// from any Strata app the user installed separately.
pub fn data_dir(s: &Settings) -> PathBuf {
    root(s).join("data")
}

pub fn env(s: &Settings) -> Vec<(String, String)> {
    vec![("STRATA_DATA_DIR".into(), data_dir(s).display().to_string())]
}

fn launcher_name() -> &'static str {
    if cfg!(windows) {
        "strata.cmd"
    } else {
        "strata"
    }
}

/// `bin/<launcher>` in `dir` or in one folder below it.
pub fn find_launcher(dir: &Path) -> Option<PathBuf> {
    let direct = dir.join("bin").join(launcher_name());
    if direct.is_file() {
        return Some(direct);
    }
    std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path().join("bin").join(launcher_name())).find(|p| p.is_file())
}

/// The newest installed Terminal package.
pub fn installed(s: &Settings) -> Option<StrataInstall> {
    let mut all: Vec<(std::time::SystemTime, StrataInstall)> = std::fs::read_dir(root(s).join("terminal"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let m = e.path().join("strata.json");
            let t = std::fs::metadata(&m).and_then(|x| x.modified()).ok()?;
            let i: StrataInstall = serde_json::from_slice(&std::fs::read(&m).ok()?).ok()?;
            i.launcher.exists().then_some((t, i))
        })
        .collect();
    all.sort_by_key(|e| std::cmp::Reverse(e.0));
    all.into_iter().next().map(|(_, i)| i)
}

pub fn remove(i: &StrataInstall) -> Result<()> {
    std::fs::remove_dir_all(&i.dir)?;
    Ok(())
}

fn sha256_file(p: &Path) -> Result<String> {
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h)?;
    Ok(format!("{:x}", h.finalize()))
}

/// A command for the launcher with Strata's environment.
pub fn command(s: &Settings, launcher: &Path) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(launcher);
    cmd.envs(env(s)).stdin(std::process::Stdio::null());
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    cmd
}

/// Download the Terminal package of the newest release, check its SHA-256
/// against the release manifest and its signed feed with `strata update
/// --check`, then unpack it under the engines folder.
pub async fn install(s: &Settings, cancel: &CancellationToken, job_id: &str, progress: &ProgressFn) -> Result<StrataInstall> {
    let platform = platform().context("Strata has no package for this system")?;
    let http = crate::http_client();
    let releases = recent_releases(&http, &s.github_api, REPO).await?;
    let (rel, manifest) = releases
        .iter()
        .filter(|r| !r.draft)
        .find_map(|r| r.assets.iter().find(|a| a.name == "install-manifest.txt").map(|m| (r, m)))
        .context("no recent Strata release has an install manifest")?;
    let text = http.get(s.github_download_url(&manifest.url)).send().await?.error_for_status()?.text().await?;
    let entry = parse_manifest(&text, platform).with_context(|| format!("Strata {} has no package for {platform}", rel.tag_name))?;
    let asset = rel.assets.iter().find(|a| a.name == entry.file).with_context(|| format!("{} is missing from Strata {}", entry.file, rel.tag_name))?;

    let dl = s.engines_dir.join("_downloads").join(&entry.file);
    let items = [DownloadItem { url: s.github_download_url(&asset.url), dest: dl.clone(), size: Some(entry.size).filter(|n| *n > 0) }];
    download_all(&http, &items, &[], cancel, job_id, progress).await?;
    let got = sha256_file(&dl)?;
    if got != entry.sha256 {
        let _ = std::fs::remove_file(&dl);
        bail!("{} does not match the release checksum (got {got}, wanted {})", entry.file, entry.sha256);
    }

    let dir = root(s).join("terminal").join(&entry.version);
    let incoming = dir.with_extension("incoming");
    if incoming.exists() {
        std::fs::remove_dir_all(&incoming)?;
    }
    extract_archive(&dl, &incoming)?;
    #[cfg(unix)]
    if let Some(l) = find_launcher(&incoming) {
        use std::os::unix::fs::PermissionsExt;
        for e in std::fs::read_dir(l.parent().unwrap()).into_iter().flatten().flatten() {
            let _ = std::fs::set_permissions(e.path(), std::fs::Permissions::from_mode(0o755));
        }
    }
    let launcher = find_launcher(&incoming).context("the archive does not look like a Strata Terminal")?;

    std::fs::create_dir_all(data_dir(s))?;
    let out = command(s, &launcher).args(["update", "--check", "--no-engine"]).output().await.context("run strata update --check")?;
    if !out.status.success() {
        let _ = std::fs::remove_dir_all(&incoming);
        bail!("Strata's signed feed could not be verified, so nothing was installed:\n{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    }

    if dir.exists() {
        std::fs::remove_dir_all(&dir).context("an older copy of this version is in use; stop Strata first")?;
    }
    std::fs::rename(&incoming, &dir)?;
    let launcher = find_launcher(&dir).context("launcher vanished after unpacking")?;
    let i = StrataInstall { version: entry.version, launcher, dir: dir.clone() };
    std::fs::write(dir.join("strata.json"), serde_json::to_vec_pretty(&i)?)?;
    let _ = std::fs::remove_file(&dl);
    Ok(i)
}

pub struct LaunchOptions<'a> {
    pub model: &'a Path,
    pub port: u16,
    pub ctx: Option<u64>,
    pub vision: bool,
    pub lan: bool,
}

/// `strata launch` arguments that serve `model` without screens or questions.
pub fn launch_args(o: &LaunchOptions) -> Vec<String> {
    // Strata takes "a folder of shards or one .gguf"; hand it the folder for a split model.
    let is_shard = o.model.file_name().and_then(|n| n.to_str()).map(|n| regex::Regex::new(r"-\d{5}-of-\d{5}\.gguf$").unwrap().is_match(n)).unwrap_or(false);
    let path = if is_shard { o.model.parent().unwrap_or(o.model) } else { o.model };
    let mut a: Vec<String> = ["launch", "--model-path"].map(String::from).to_vec();
    a.push(path.display().to_string());
    a.extend(["--port".into(), o.port.to_string(), "--yes".into(), "--no-tui".into(), "--restart".into()]);
    if let Some(c) = o.ctx {
        a.extend(["--context".into(), c.to_string()]);
    }
    if !o.vision {
        a.extend(["--vision".into(), "off".into()]);
    }
    if o.lan {
        a.extend(["--access".into(), "lan".into(), "--api-key-env".into(), KEY_ENV.into()]);
    } else {
        a.extend(["--access".into(), "local".into()]);
    }
    a
}

pub fn stop_args() -> Vec<String> {
    vec!["launch".into(), "--stop".into()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_line() {
        let m = "linux-x86_64 0.1.55 strata-control-0.1.55-linux-x86_64.tar.gz 50783D2F 99498096\nwindows-x86_64 0.1.55 strata-control-0.1.55-windows-x86_64.zip bdca3a42 73121142\n";
        assert_eq!(
            parse_manifest(m, "windows-x86_64"),
            Some(ManifestEntry { version: "0.1.55".into(), file: "strata-control-0.1.55-windows-x86_64.zip".into(), sha256: "bdca3a42".into(), size: 73121142 })
        );
        assert_eq!(parse_manifest(m, "linux-x86_64").unwrap().sha256, "50783d2f");
        assert!(parse_manifest(m, "macos-arm64").is_none());
    }

    #[test]
    fn supported_families() {
        assert!(supports("Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00003.gguf"));
        assert!(supports("/m/unsloth__Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"));
        assert!(!supports("/m/unsloth__Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"), "Qwen3.6 needs the MTP edition");
        assert!(supports("DeepSeekV4.1-Flash-1017-IST-DAS-2.00bpw"));
        assert!(!supports("gpt-oss-20b-Q4_K_M.gguf"));
    }

    #[test]
    fn official_packs_by_catalog_id() {
        assert_eq!(official_pack("qwen3.8-flash-next", "unsloth/Qwen3.8-Flash-Next-GGUF").unwrap().quants.len(), 5);
        assert_eq!(official_pack("qwen3.8-flash-next-ista", "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF").unwrap().quants[0], "Q2_0");
        assert_eq!(official_pack("qwen3.6-35b-a3b", "unsloth/Qwen3.6-35B-A3B-GGUF").unwrap().repo, "unsloth/Qwen3.6-35B-A3B-MTP-GGUF");
        assert!(official_pack("qwen3.6-27b", "unsloth/Qwen3.6-27B-GGUF").is_none());
    }

    #[test]
    fn args_for_split_model_use_folder() {
        let m = Path::new("/m/Qwen3.8-Flash-Next/Q2_0-00001-of-00004.gguf");
        let a = launch_args(&LaunchOptions { model: m, port: 8081, ctx: Some(65536), vision: false, lan: true });
        let i = a.iter().position(|x| x == "--model-path").unwrap();
        assert_eq!(Path::new(&a[i + 1]), Path::new("/m/Qwen3.8-Flash-Next"));
        for w in [["--port", "8081"], ["--context", "65536"], ["--vision", "off"], ["--access", "lan"], ["--api-key-env", KEY_ENV]] {
            assert!(a.windows(2).any(|x| x[0] == w[0] && x[1] == w[1]), "{w:?} in {a:?}");
        }
        let single = launch_args(&LaunchOptions { model: Path::new("/m/a.gguf"), port: 8080, ctx: None, vision: true, lan: false });
        assert!(single.contains(&"/m/a.gguf".to_string()) && single.contains(&"local".to_string()) && !single.contains(&"--vision".to_string()));
    }

    /// Downloads the real Terminal package (~75 MB) and runs its signature
    /// check: `cargo test -p openmodel-core strata::tests::live_install -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn live_install() {
        let d = tempfile::tempdir().unwrap();
        let s = Settings { engines_dir: d.path().join("engines"), ..Settings::default() };
        let i = install(&s, &CancellationToken::new(), "test", &crate::noop_progress()).await.unwrap();
        assert!(i.launcher.is_file());
        assert_eq!(installed(&s), Some(i.clone()));
        let out = command(&s, &i.launcher).args(["launch", "--help"]).output().await.unwrap();
        let help = String::from_utf8_lossy(&out.stdout);
        for flag in ["--model-path", "--no-tui", "--restart", "--access", "--api-key-env", "--vision", "--context"] {
            assert!(help.contains(flag), "strata launch --help lacks {flag}");
        }
    }

    #[test]
    fn launcher_lookup() {
        let d = tempfile::tempdir().unwrap();
        let inner = d.path().join("strata-control").join("bin");
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(inner.join(launcher_name()), "").unwrap();
        assert_eq!(find_launcher(d.path()), Some(inner.join(launcher_name())));
    }
}
