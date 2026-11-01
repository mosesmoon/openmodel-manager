//! Hugging Face Hub: search GGUF repos, list files, group them by quant.
//!
//! Talks only to `Settings::hf_endpoint` (huggingface.co or a mirror the user
//! chose). The token, if any, is sent only to that endpoint.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct HfClient {
    http: reqwest::Client,
    endpoint: String,
    token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModel {
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,
    #[serde(default)]
    pub gated: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantGroup {
    pub quant: String,
    pub files: Vec<RepoFile>,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RepoListing {
    pub quants: Vec<QuantGroup>,
    /// Vision projector files (`mmproj-*.gguf`) found in the repo.
    pub mmproj: Vec<RepoFile>,
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    size: u64,
}

impl HfClient {
    pub fn new(http: reqwest::Client, endpoint: &str, token: Option<String>) -> HfClient {
        HfClient { http, endpoint: endpoint.trim_end_matches('/').to_string(), token: token.filter(|t| !t.trim().is_empty()) }
    }

    pub fn from_settings(s: &crate::config::Settings) -> HfClient {
        HfClient::new(crate::http_client(), &s.hf_endpoint, s.hf_token.clone())
    }

    fn get(&self, url: &str) -> reqwest::RequestBuilder {
        let r = self.http.get(url);
        match &self.token {
            Some(t) => r.bearer_auth(t),
            None => r,
        }
    }

    /// Headers to use when downloading files from this endpoint.
    pub fn auth_header(&self) -> Option<(String, String)> {
        self.token.as_ref().map(|t| ("Authorization".to_string(), format!("Bearer {t}")))
    }

    pub async fn search(&self, query: &str) -> Result<Vec<HfModel>> {
        let url = format!("{}/api/models", self.endpoint);
        let r = self
            .get(&url)
            .query(&[("search", query), ("filter", "gguf"), ("sort", "downloads"), ("direction", "-1"), ("limit", "40")])
            .send()
            .await?;
        if !r.status().is_success() {
            bail!("Hugging Face search failed: HTTP {}", r.status());
        }
        Ok(r.json().await?)
    }

    /// All files in the repo's main branch, following pagination.
    pub async fn list_files(&self, repo: &str) -> Result<Vec<RepoFile>> {
        let mut url = format!("{}/api/models/{}/tree/main?recursive=true&expand=false", self.endpoint, repo);
        let mut out = Vec::new();
        for _ in 0..50 {
            let r = self.get(&url).send().await?;
            match r.status().as_u16() {
                200 => {}
                401 | 403 => bail!("{repo} is gated or private: set a Hugging Face token and accept the license on the model page"),
                404 => bail!("repository {repo} not found"),
                s => bail!("listing {repo} failed: HTTP {s}"),
            }
            let next = next_link(r.headers().get("link").and_then(|v| v.to_str().ok()));
            let entries: Vec<TreeEntry> = r.json().await?;
            out.extend(entries.into_iter().filter(|e| e.kind == "file").map(|e| RepoFile {
                size: e.lfs.map(|l| l.size).unwrap_or(e.size),
                path: e.path,
            }));
            match next {
                Some(n) => url = n,
                None => break,
            }
        }
        Ok(out)
    }

    pub async fn repo_listing(&self, repo: &str) -> Result<RepoListing> {
        Ok(group_files(&self.list_files(repo).await?))
    }

    pub fn resolve_url(&self, repo: &str, path: &str) -> String {
        let enc: Vec<String> = path.split('/').map(encode_segment).collect();
        format!("{}/{}/resolve/main/{}", self.endpoint, repo, enc.join("/"))
    }
}

fn encode_segment(s: &str) -> String {
    let mut o = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => o.push(b as char),
            _ => o.push_str(&format!("%{b:02X}")),
        }
    }
    o
}

fn next_link(h: Option<&str>) -> Option<String> {
    h?.split(',').find(|p| p.contains("rel=\"next\"")).and_then(|p| {
        let s = p.find('<')? + 1;
        let e = p.find('>')?;
        Some(p[s..e].to_string())
    })
}

/// Quant label in a GGUF file name (`UD-Q4_K_XL`, `IQ3_XXS`, `Q8_0`, `BF16`...).
pub fn quant_of(file_name: &str) -> Option<String> {
    let re = regex::Regex::new(r"(?i)(?:^|[-_.])((?:UD-)?(?:IQ\d_(?:XXS|XS|S|M|NL)|Q\d_K(?:_(?:XL|S|M|L))?|Q\d_[01]|TQ\d_0|MXFP4(?:_MOE)?|BF16|F16|F32))(?:[-_.]|$)").unwrap();
    let stem = file_name.rsplit('/').next().unwrap_or(file_name);
    let mut found = None;
    let mut start = 0;
    // Take the last match so names like "Q8_0-model-Q4_K_M" pick the real quant.
    while let Some(c) = re.captures_at(stem, start) {
        let m = c.get(1).unwrap();
        found = Some(m.as_str().to_uppercase());
        start = m.end();
    }
    found
}

pub fn is_mmproj(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    name.ends_with(".gguf") && (name.starts_with("mmproj") || name.contains(".mmproj") || name.contains("-mmproj"))
}

pub fn group_files(files: &[RepoFile]) -> RepoListing {
    let mut listing = RepoListing::default();
    let mut groups: Vec<QuantGroup> = Vec::new();
    for f in files.iter().filter(|f| f.path.to_lowercase().ends_with(".gguf")) {
        if is_mmproj(&f.path) {
            listing.mmproj.push(f.clone());
            continue;
        }
        let q = quant_of(&f.path).unwrap_or_else(|| "OTHER".into());
        match groups.iter_mut().find(|g| g.quant == q) {
            Some(g) => {
                g.total_bytes += f.size;
                g.files.push(f.clone());
            }
            None => groups.push(QuantGroup { quant: q, files: vec![f.clone()], total_bytes: f.size }),
        }
    }
    for g in &mut groups {
        g.files.sort_by(|a, b| a.path.cmp(&b.path));
    }
    groups.sort_by_key(|g| g.total_bytes);
    listing.quants = groups;
    listing
}

/// Pick the mmproj that best matches the main model (prefers F16, then BF16, then Q8_0).
pub fn best_mmproj(mmproj: &[RepoFile]) -> Option<&RepoFile> {
    let rank = |p: &str| {
        let u = p.to_uppercase();
        if u.contains("F16") && !u.contains("BF16") {
            0
        } else if u.contains("BF16") {
            1
        } else if u.contains("Q8_0") {
            2
        } else {
            3
        }
    };
    mmproj.iter().min_by_key(|f| rank(&f.path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(p: &str, s: u64) -> RepoFile {
        RepoFile { path: p.into(), size: s }
    }

    #[test]
    fn quant_names() {
        assert_eq!(quant_of("Qwen3.6-27B-UD-Q4_K_XL.gguf").as_deref(), Some("UD-Q4_K_XL"));
        assert_eq!(quant_of("gpt-oss-20b-F16.gguf").as_deref(), Some("F16"));
        assert_eq!(quant_of("flux1-schnell-q4_k.gguf").as_deref(), Some("Q4_K"));
        assert_eq!(quant_of("UD-Q2_K_XL/Big-UD-Q2_K_XL-00001-of-00003.gguf").as_deref(), Some("UD-Q2_K_XL"));
        assert_eq!(quant_of("model-IQ3_XXS.gguf").as_deref(), Some("IQ3_XXS"));
        assert_eq!(quant_of("Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(quant_of("weird.gguf"), None);
    }

    #[test]
    fn groups_splits_and_mmproj() {
        let files = vec![
            f("README.md", 10),
            f("mmproj-F16.gguf", 900),
            f("mmproj-BF16.gguf", 900),
            f("M-Q4_K_M.gguf", 4000),
            f("Q8_0/M-Q8_0-00002-of-00002.gguf", 4000),
            f("Q8_0/M-Q8_0-00001-of-00002.gguf", 4500),
        ];
        let l = group_files(&files);
        assert_eq!(l.mmproj.len(), 2);
        assert_eq!(best_mmproj(&l.mmproj).unwrap().path, "mmproj-F16.gguf");
        assert_eq!(l.quants.len(), 2);
        assert_eq!(l.quants[0].quant, "Q4_K_M");
        let q8 = &l.quants[1];
        assert_eq!(q8.total_bytes, 8500);
        assert!(q8.files[0].path.contains("00001"));
    }

    #[test]
    fn urls() {
        let c = HfClient::new(reqwest::Client::new(), "https://hf-mirror.com/", None);
        assert_eq!(c.resolve_url("a/b", "dir/x y.gguf"), "https://hf-mirror.com/a/b/resolve/main/dir/x%20y.gguf");
        assert!(c.auth_header().is_none());
        assert_eq!(
            next_link(Some("<https://huggingface.co/api/models/a/b/tree/main?cursor=xyz>; rel=\"next\"")).as_deref(),
            Some("https://huggingface.co/api/models/a/b/tree/main?cursor=xyz")
        );
    }
}
