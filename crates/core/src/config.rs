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
    /// Port used by llama-server (moves up if busy).
    pub server_port: u16,
    /// Port used by sd-server (moves up if busy).
    pub image_port: u16,
    /// Listen on 0.0.0.0 instead of 127.0.0.1.
    pub lan_access: bool,
    /// Addresses to open the servers on when `lan_access` is set; empty
    /// means every LAN address this computer has.
    #[serde(default)]
    pub lan_addresses: Vec<String>,
    /// Key clients of the local LLM server must send; blank means none.
    #[serde(default)]
    pub server_api_key: Option<String>,
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
            language: "zh-TW".into(),
            server_port: 8080,
            image_port: 7860,
            lan_access: false,
            server_api_key: None,
            lan_addresses: vec![],
            backend: None,
            gpu_override: None,
            custom_args: HashMap::new(),
            endpoints: vec![],
            agent_workspace: None,
        }
    }
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

fn settings_path() -> PathBuf {
    data_dir().join("settings.json")
}

impl Settings {
    /// The local server's API key, or `None` when left blank.
    pub fn server_key(&self) -> Option<String> {
        self.server_api_key.as_deref().map(str::trim).filter(|k| !k.is_empty()).map(String::from)
    }

    pub fn load() -> Settings {
        let s: Settings = std::fs::read(settings_path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
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
        if !self.lan_access {
            return vec![];
        }
        let have: Vec<String> = crate::net::lan_addresses().into_iter().map(|a| a.ip).collect();
        let pick: Vec<&String> = if self.lan_addresses.is_empty() { have.iter().collect() } else { self.lan_addresses.iter().filter(|a| have.contains(a)).collect() };
        pick.into_iter().filter_map(|a| a.parse().ok()).collect()
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
    fn folders_inside_the_data_dir_are_saved_relative() {
        let base = std::env::temp_dir().join("OpenModelData");
        let s = Settings { models_dir: base.join("models"), engines_dir: PathBuf::from("D:/elsewhere/engines"), outputs_dir: base.join("outputs"), ..Settings::default() };
        let rel = s.relative_to(&base);
        assert_eq!(rel.models_dir, PathBuf::from("models"));
        assert_eq!(rel.engines_dir, PathBuf::from("D:/elsewhere/engines"));
        // Moved to another place: the relative folders follow.
        let moved = std::env::temp_dir().join("usb").join("OpenModelData");
        let back = rel.resolved_in(&moved);
        assert_eq!(back.models_dir, moved.join("models"));
        assert_eq!(back.outputs_dir, moved.join("outputs"));
        assert_eq!(back.engines_dir, PathBuf::from("D:/elsewhere/engines"));
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
}
