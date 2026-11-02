//! Paths and persisted settings.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Manual GPU override for machines where detection is wrong.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuOverride {
    pub vendor: crate::hardware::GpuVendor,
    pub name: String,
    pub vram_gb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Where models (GGUF, diffusion weights) are stored.
    pub models_dir: PathBuf,
    /// Where inference engines are unpacked.
    pub engines_dir: PathBuf,
    /// Where generated images and videos are written.
    pub outputs_dir: PathBuf,
    /// Hugging Face endpoint (e.g. https://hf-mirror.com for mirrors).
    pub hf_endpoint: String,
    /// Optional HF token for gated repos.
    pub hf_token: Option<String>,
    /// GitHub API base used for engine releases.
    pub github_api: String,
    /// Optional user-chosen prefix for GitHub downloads (e.g. a public mirror).
    /// Empty means direct download. Never set by default.
    pub github_mirror_prefix: String,
    /// RAM (bytes) always left to the OS when planning offload.
    pub ram_reserve_bytes: u64,
    /// VRAM (bytes) left free on the GPU when planning offload.
    pub vram_reserve_bytes: u64,
    /// UI language: "zh-TW" or "en".
    pub language: String,
    /// Language of the MSI that installed this copy (Windows LCID), once it
    /// has been applied to `language`; a reinstall in another language
    /// switches the UI again.
    pub installer_language: Option<String>,
    /// Port used by llama-server (moves up if busy).
    pub server_port: u16,
    /// Port used by sd-server (moves up if busy).
    pub image_port: u16,
    /// Open the model servers to other computers (see `lan_addresses`).
    pub lan_access: bool,
    /// Addresses to open the servers on when `lan_access` is set; empty
    /// means every LAN address this computer has.
    #[serde(default)]
    pub lan_addresses: Vec<String>,
    /// Keep the OpenAI API answering on `server_port` while no model runs,
    /// and load the model a request names on demand.
    pub jit_api: bool,
    /// Key clients of the local LLM server must send; blank means none.
    #[serde(default)]
    pub server_api_key: Option<String>,
    /// Web management page: the same UI served to browsers on other devices.
    pub web_ui: bool,
    pub web_port: u16,
    /// Open the web page to other computers, on `web_addresses` (empty: all).
    pub web_lan: bool,
    pub web_addresses: Vec<String>,
    /// Salted hash of the web page password (see [`hash_password`]).
    pub web_password: Option<String>,
    /// Preferred backend; None = pick automatically.
    pub backend: Option<crate::hardware::Backend>,
    pub gpu_override: Option<GpuOverride>,
    /// Extra llama-server args per model key (file name).
    pub custom_args: HashMap<String, String>,
    /// Remote OpenAI-compatible endpoints the user added (name -> endpoint).
    /// API keys are stored here in plain text; the UI says so.
    pub endpoints: Vec<NamedEndpoint>,
    /// Folder the Agent page works in.
    pub agent_workspace: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NamedEndpoint {
    pub name: String,
    /// Preset the entry was made from ("anthropic", "xai", "custom", ...).
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(flatten)]
    pub endpoint: crate::llm::Endpoint,
}

impl Default for Settings {
    fn default() -> Self {
        let base = data_dir();
        Settings {
            models_dir: base.join("models"),
            engines_dir: base.join("engines"),
            outputs_dir: base.join("outputs"),
            hf_endpoint: "https://huggingface.co".into(),
            hf_token: None,
            github_api: "https://api.github.com".into(),
            github_mirror_prefix: String::new(),
            ram_reserve_bytes: 4 * GIB,
            vram_reserve_bytes: GIB / 2,
            language: first_run_language().into(),
            installer_language: None,
            server_port: 8080,
            image_port: 7860,
            lan_access: false,
            server_api_key: None,
            jit_api: false,
            lan_addresses: vec![],
            web_ui: false,
            web_port: 8765,
            web_lan: false,
            web_addresses: vec![],
            web_password: None,
            backend: None,
            gpu_override: None,
            custom_args: HashMap::new(),
            endpoints: vec![],
            agent_workspace: None,
        }
    }
}

/// The chosen addresses this computer still has, or all of its LAN
/// addresses when none are chosen.
fn pick_addresses(open: bool, chosen: &[String]) -> Vec<std::net::IpAddr> {
    if !open {
        return vec![];
    }
    let have: Vec<String> = crate::net::lan_addresses().into_iter().map(|a| a.ip).collect();
    let pick: Vec<&String> = if chosen.is_empty() { have.iter().collect() } else { chosen.iter().filter(|a| have.contains(a)).collect() };
    pick.into_iter().filter_map(|a| a.parse().ok()).collect()
}

const PW_ROUNDS: u32 = 100_000;

/// `salt$hex`: SHA-256 iterated over a random salt and the password.
pub fn hash_password(password: &str) -> String {
    let salt = uuid::Uuid::new_v4().simple().to_string();
    format!("{salt}${}", pw_digest(&salt, password))
}

pub fn verify_password(stored: &str, password: &str) -> bool {
    let Some((salt, want)) = stored.split_once('$') else { return false };
    let got = pw_digest(salt, password);
    // Compare every byte so the time taken does not depend on where they differ.
    got.len() == want.len() && got.bytes().zip(want.bytes()).fold(0u8, |d, (a, b)| d | (a ^ b)) == 0
}

fn pw_digest(salt: &str, password: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::digest(format!("{salt}:{password}").as_bytes());
    for _ in 1..PW_ROUNDS {
        h = Sha256::digest(h);
    }
    h.iter().map(|b| format!("{b:02x}")).collect()
}

pub const GIB: u64 = 1024 * 1024 * 1024;
pub const MIB: u64 = 1024 * 1024;

/// Data directory. Precedence: `OPENMODEL_HOME`, portable mode (a
/// `portable.txt` next to the executable puts data in `OpenModelData/`),
/// then the platform data dir.
pub fn data_dir() -> PathBuf {
    if let Ok(p) = std::env::var("OPENMODEL_HOME") {
        return PathBuf::from(p);
    }
    if let Some(d) = portable_dir() {
        return d;
    }
    directories::ProjectDirs::from("org", "openmodel", "OpenModelManager")
        .map(|d| d.data_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".openmodel"))
}

/// `OpenModelData/` next to the executable when a `portable.txt` sits there.
pub fn portable_dir() -> Option<PathBuf> {
    let dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    dir.join("portable.txt").exists().then(|| dir.join("OpenModelData"))
}

/// UI language before the user picks one: English for the portable build,
/// otherwise Traditional Chinese (an MSI overrides both, see `load`).
fn first_run_language() -> &'static str {
    if portable_dir().is_some() { "en" } else { "zh-TW" }
}

/// LCID the MSI wrote to `install.ini` next to the executable
/// (packaging/windows/language.wxs), e.g. "1028" for zh-TW.
fn installer_language() -> Option<String> {
    let ini = std::env::current_exe().ok()?.parent()?.join("install.ini");
    parse_installer_language(&std::fs::read(ini).ok()?)
}

fn parse_installer_language(raw: &[u8]) -> Option<String> {
    // Windows writes INI files as UTF-16 when they start with a BOM.
    let text = if raw.starts_with(&[0xFF, 0xFE]) {
        String::from_utf16_lossy(&raw[2..].as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect::<Vec<_>>())
    } else {
        String::from_utf8_lossy(raw).into_owned()
    };
    text.lines()
        .filter_map(|l| l.split_once('='))
        .find(|(k, _)| k.trim().eq_ignore_ascii_case("language"))
        .map(|(_, v)| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// UI language for a Windows LCID: Chinese (Taiwan, Hong Kong, Macao) gets
/// Traditional Chinese, everything else English.
fn language_for_lcid(lcid: &str) -> &'static str {
    match lcid {
        "1028" | "3076" | "5124" => "zh-TW",
        _ => "en",
    }
}

fn settings_path() -> PathBuf {
    data_dir().join("settings.json")
}

impl Settings {
    /// The local server's API key, or `None` when left blank.
    pub fn server_key(&self) -> Option<String> {
        self.server_api_key.as_deref().map(str::trim).filter(|k| !k.is_empty()).map(String::from)
    }

    pub fn load() -> Settings {
        let mut s: Settings = std::fs::read(settings_path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        if let Some(lcid) = installer_language() {
            if s.installer_language.as_deref() != Some(lcid.as_str()) {
                s.language = language_for_lcid(&lcid).into();
                s.installer_language = Some(lcid);
            }
        }
        s.resolved_in(&data_dir())
    }

    pub fn save(&self) -> Result<()> {
        let p = settings_path();
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(p, serde_json::to_vec_pretty(&self.relative_to(&data_dir()))?)?;
        Ok(())
    }

    fn dirs_mut(&mut self) -> [&mut PathBuf; 3] {
        [&mut self.models_dir, &mut self.engines_dir, &mut self.outputs_dir]
    }

    /// Folders inside `base` stored relative to it, so a portable folder
    /// keeps working after it is moved to another drive or PC.
    pub fn relative_to(&self, base: &Path) -> Settings {
        let mut s = self.clone();
        for d in s.dirs_mut() {
            if let Ok(rel) = d.strip_prefix(base) {
                *d = if rel.as_os_str().is_empty() { PathBuf::from(".") } else { rel.to_path_buf() };
            }
        }
        s
    }

    /// Relative folders (as `relative_to` writes them) joined back onto `base`.
    pub fn resolved_in(mut self, base: &Path) -> Settings {
        for d in self.dirs_mut() {
            if d.is_relative() {
                *d = base.join(&*d);
            }
        }
        self
    }

    /// Create the models / engines / outputs folders.
    pub fn ensure_dirs(&self) -> Result<()> {
        for d in [&self.models_dir, &self.engines_dir, &self.outputs_dir] {
            std::fs::create_dir_all(d)?;
        }
        Ok(())
    }

    /// Addresses the servers are opened on besides 127.0.0.1 (the servers
    /// themselves bind 127.0.0.1; see [`crate::net`]). Chosen addresses that
    /// this computer no longer has are skipped.
    pub fn exposed_ips(&self) -> Vec<std::net::IpAddr> {
        pick_addresses(self.lan_access, &self.lan_addresses)
    }

    /// Addresses the web management page is opened on besides 127.0.0.1.
    pub fn web_ips(&self) -> Vec<std::net::IpAddr> {
        pick_addresses(self.web_lan, &self.web_addresses)
    }

    /// Apply the optional mirror prefix to a GitHub download URL.
    pub fn github_download_url(&self, url: &str) -> String {
        let p = self.github_mirror_prefix.trim();
        if p.is_empty() {
            url.to_string()
        } else if p.ends_with('/') {
            format!("{p}{url}")
        } else {
            format!("{p}/{url}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installer_language_from_ini() {
        assert_eq!(parse_installer_language(b"[install]
language=1028
").as_deref(), Some("1028"));
        let utf16: Vec<u8> = [0xFF, 0xFE].into_iter().chain("[install]
Language = 1033
".encode_utf16().flat_map(u16::to_le_bytes)).collect();
        assert_eq!(parse_installer_language(&utf16).as_deref(), Some("1033"));
        assert_eq!(parse_installer_language(b"[install]
"), None);
        assert_eq!(language_for_lcid("1028"), "zh-TW");
        assert_eq!(language_for_lcid("1033"), "en");
    }

    #[test]
    fn folders_inside_the_data_dir_are_saved_relative() {
        let base = std::env::temp_dir().join("OpenModelData");
        // A folder outside the data dir, absolute on every platform.
        let elsewhere = if cfg!(windows) { PathBuf::from("D:/elsewhere/engines") } else { PathBuf::from("/elsewhere/engines") };
        let s = Settings { models_dir: base.join("models"), engines_dir: elsewhere.clone(), outputs_dir: base.join("outputs"), ..Settings::default() };
        let rel = s.relative_to(&base);
        assert_eq!(rel.models_dir, PathBuf::from("models"));
        assert_eq!(rel.engines_dir, elsewhere);
        // Moved to another place: the relative folders follow.
        let moved = std::env::temp_dir().join("usb").join("OpenModelData");
        let back = rel.resolved_in(&moved);
        assert_eq!(back.models_dir, moved.join("models"));
        assert_eq!(back.outputs_dir, moved.join("outputs"));
        assert_eq!(back.engines_dir, elsewhere);
    }

    #[test]
    fn mirror_prefix() {
        let mut s = Settings::default();
        let u = "https://github.com/a/b/releases/download/x/y.zip";
        assert_eq!(s.github_download_url(u), u);
        s.github_mirror_prefix = "https://mirror.example".into();
        assert_eq!(s.github_download_url(u), format!("https://mirror.example/{u}"));
    }

    #[test]
    fn old_settings_files_still_load() {
        let s: Settings = serde_json::from_str(r#"{"server_port": 9000}"#).unwrap();
        assert_eq!(s.server_port, 9000);
        assert_eq!(s.image_port, 7860);
    }

    #[test]
    fn passwords() {
        let h = super::hash_password("correct horse");
        assert!(super::verify_password(&h, "correct horse"));
        assert!(!super::verify_password(&h, "correct horsE"));
        assert!(!super::verify_password("garbage", "x"));
        assert_ne!(h, super::hash_password("correct horse")); // salted
    }
}
