//! Local model library: GGUF files under `models_dir` plus models the user
//! registered from elsewhere (kept in place, never copied).

use crate::gguf;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModel {
    /// Display name (file name without the split suffix).
    pub name: String,
    /// First (or only) file; pass this to llama-server `-m`.
    pub path: PathBuf,
    pub files: Vec<PathBuf>,
    pub size_bytes: u64,
    pub quant: Option<String>,
    /// Vision projector next to the model, if any.
    pub mmproj: Option<PathBuf>,
    /// Registered from outside `models_dir`.
    pub external: bool,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Registry {
    external: Vec<PathBuf>,
}

fn registry_path(data_dir: &Path) -> PathBuf {
    data_dir.join("library.json")
}

fn load_registry(data_dir: &Path) -> Registry {
    std::fs::read(registry_path(data_dir)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

fn save_registry(data_dir: &Path, r: &Registry) -> Result<()> {
    std::fs::create_dir_all(data_dir)?;
    std::fs::write(registry_path(data_dir), serde_json::to_vec_pretty(r)?)?;
    Ok(())
}

fn is_gguf(p: &Path) -> bool {
    p.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false)
}

/// Find the best mmproj file in a folder.
pub fn find_mmproj(dir: &Path) -> Option<PathBuf> {
    let mut c: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_gguf(p) && crate::hf::is_mmproj(&p.file_name().unwrap_or_default().to_string_lossy()))
        .collect();
    let rank = |p: &PathBuf| {
        let n = p.to_string_lossy().to_uppercase();
        if n.contains("BF16") {
            1
        } else if n.contains("F16") {
            0
        } else {
            2
        }
    };
    c.sort_by_key(rank);
    c.into_iter().next()
}

/// Describe the model at `path` (any part of a split model works).
pub fn describe(path: &Path, external: bool) -> Option<LocalModel> {
    let name = path.file_name()?.to_string_lossy().to_string();
    if crate::hf::is_mmproj(&name) {
        return None;
    }
    let (display, files) = match gguf::split_parts(&name) {
        Some((stem, _, count)) => {
            let dir = path.parent()?;
            let files: Vec<PathBuf> = (1..=count).map(|i| dir.join(format!("{stem}-{i:05}-of-{count:05}.gguf"))).collect();
            (stem, files)
        }
        None => (name.trim_end_matches(".gguf").trim_end_matches(".GGUF").to_string(), vec![path.to_path_buf()]),
    };
    let size_bytes = files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum();
    Some(LocalModel {
        quant: crate::hf::quant_of(&name),
        mmproj: path.parent().and_then(find_mmproj),
        path: files[0].clone(),
        files,
        size_bytes,
        name: display,
        external,
    })
}

/// Scan `models_dir` (diffusion weights excluded) and registered external models.
pub fn list(models_dir: &Path, data_dir: &Path) -> Vec<LocalModel> {
    let mut out: Vec<LocalModel> = Vec::new();
    let mut stack = vec![models_dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.file_name().map(|n| n != "diffusion").unwrap_or(true) {
                    stack.push(p);
                }
                continue;
            }
            if !is_gguf(&p) {
                continue;
            }
            let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
            if matches!(gguf::split_parts(&name), Some((_, i, _)) if i != 1) {
                continue;
            }
            out.extend(describe(&p, false));
        }
    }
    for p in load_registry(data_dir).external {
        if p.exists() && !out.iter().any(|m| m.path == p) {
            out.extend(describe(&p, true));
        }
    }
    out.sort_by_key(|a| a.name.to_lowercase());
    out
}

/// Register a GGUF that lives elsewhere. It is validated by reading its header.
pub fn import_path(data_dir: &Path, path: &Path) -> Result<LocalModel> {
    let path = std::fs::canonicalize(path).with_context(|| format!("{} not found", path.display()))?;
    if !is_gguf(&path) {
        bail!("{} is not a .gguf file", path.display());
    }
    let g = gguf::open(&path)?;
    if g.summary().is_mmproj {
        bail!("{} is a vision projector (mmproj), not a model", path.display());
    }
    let first = describe(&path, true).context("cannot describe model")?.path;
    let mut r = load_registry(data_dir);
    if !r.external.contains(&first) {
        r.external.push(first.clone());
        save_registry(data_dir, &r)?;
    }
    describe(&first, true).context("cannot describe model")
}

/// Delete a downloaded model (all split parts). External models are only
/// unregistered; their files are left alone.
pub fn remove(models_dir: &Path, data_dir: &Path, m: &LocalModel) -> Result<()> {
    let mut r = load_registry(data_dir);
    if m.external || !m.path.starts_with(models_dir) {
        r.external.retain(|p| p != &m.path);
        return save_registry(data_dir, &r);
    }
    for f in &m.files {
        if f.exists() {
            std::fs::remove_file(f)?;
        }
        let part = crate::download::part_path(f);
        if part.exists() {
            std::fs::remove_file(part)?;
        }
    }
    // Drop the folder when nothing but an mmproj is left in it.
    if let Some(dir) = m.path.parent().filter(|d| *d != models_dir) {
        let left: Vec<_> = std::fs::read_dir(dir).into_iter().flatten().flatten().filter(|e| !crate::hf::is_mmproj(&e.file_name().to_string_lossy())).collect();
        if left.is_empty() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
    Ok(())
}

/// Folder a repo's files are downloaded into.
pub fn repo_dir(models_dir: &Path, repo: &str) -> PathBuf {
    models_dir.join(repo.replace('/', "__"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::testutil::{build, V};

    #[test]
    fn scans_groups_imports_and_removes() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("models");
        let data = tmp.path().join("data");
        let repo = repo_dir(&models, "org/Model-GGUF");
        std::fs::create_dir_all(&repo).unwrap();
        let llm = build(&[("general.architecture", V::Str("llama"))], &[("blk.0.attn_q.weight", &[32, 32], 1)]);
        std::fs::write(repo.join("M-Q8_0-00001-of-00002.gguf"), &llm).unwrap();
        std::fs::write(repo.join("M-Q8_0-00002-of-00002.gguf"), &llm).unwrap();
        std::fs::write(repo.join("mmproj-F16.gguf"), build(&[("general.architecture", V::Str("clip"))], &[])).unwrap();
        std::fs::create_dir_all(models.join("diffusion/x")).unwrap();
        std::fs::write(models.join("diffusion/x/flux.gguf"), &llm).unwrap();

        let ext_dir = tmp.path().join("elsewhere");
        std::fs::create_dir_all(&ext_dir).unwrap();
        std::fs::write(ext_dir.join("Other-Q4_K_M.gguf"), &llm).unwrap();
        import_path(&data, &ext_dir.join("Other-Q4_K_M.gguf")).unwrap();
        assert!(import_path(&data, &repo.join("mmproj-F16.gguf")).is_err());

        let l = list(&models, &data);
        assert_eq!(l.len(), 2, "{l:#?}");
        let m = l.iter().find(|m| m.name == "M-Q8_0").unwrap();
        assert_eq!(m.files.len(), 2);
        assert_eq!(m.quant.as_deref(), Some("Q8_0"));
        assert!(m.mmproj.as_ref().unwrap().ends_with("mmproj-F16.gguf"));
        let ext = l.iter().find(|m| m.external).unwrap();

        remove(&models, &data, ext).unwrap();
        assert!(ext_dir.join("Other-Q4_K_M.gguf").exists(), "external files are never deleted");
        remove(&models, &data, m).unwrap();
        assert!(!repo.exists());
        assert!(list(&models, &data).is_empty());
    }
}
