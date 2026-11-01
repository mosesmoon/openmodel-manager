//! Coding agent tools scoped to a workspace folder.
//!
//! Policy (enforced here, not in the UI):
//! - read / write / edit / list / search inside the workspace: automatic
//! - any path outside the workspace: asks the user
//! - `run_command` and `delete_path`: always ask
//!
//! Paths are resolved through symlinks and `..` before the inside/outside
//! decision, so `../`, absolute paths and links cannot slip out silently.

use crate::tools::{self, function, ApprovalGate};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const MAX_ROUNDS: usize = 150;

pub const SYSTEM_PROMPT: &str = "You are a careful software agent working inside the user's project folder.
Rules:
1. Start by calling update_plan with a short step list, and keep it updated as you go.
2. Look before you change: list_files / read_file / search_files before editing.
3. Prefer edit_file for small changes; old_text must match exactly once.
4. After changing code, run the project's tests or build with run_command and fix what fails.
5. Never delete or overwrite things you were not asked to touch.
6. Finish with a short summary of what changed and how it was verified.";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanItem {
    pub step: String,
    /// pending / in_progress / done
    pub status: String,
}

pub struct Workspace {
    root: PathBuf,
}

/// Remove `.` and resolve `..` lexically.
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Canonicalize the deepest existing ancestor (resolving symlinks) and
/// append the not-yet-existing remainder.
fn resolve_real(p: &Path) -> PathBuf {
    let mut existing = p.to_path_buf();
    let mut rest = Vec::new();
    while !existing.exists() {
        match existing.file_name() {
            Some(n) => {
                rest.push(n.to_os_string());
                existing.pop();
            }
            None => break,
        }
    }
    let mut base = std::fs::canonicalize(&existing).unwrap_or(existing);
    for n in rest.into_iter().rev() {
        base.push(n);
    }
    base
}

impl Workspace {
    pub fn new(root: &Path) -> Result<Workspace> {
        let root = std::fs::canonicalize(root).with_context(|| format!("workspace {} does not exist", root.display()))?;
        if !root.is_dir() {
            bail!("{} is not a folder", root.display());
        }
        Ok(Workspace { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Absolute real path and whether it is inside the workspace.
    pub fn resolve(&self, p: &str) -> (PathBuf, bool) {
        let raw = Path::new(p.trim());
        let joined = if raw.is_absolute() { raw.to_path_buf() } else { self.root.join(raw) };
        let real = resolve_real(&normalize(&joined));
        let inside = real.starts_with(&self.root);
        (real, inside)
    }
}

pub struct AgentToolbox {
    pub ws: Workspace,
    pub gate: Arc<ApprovalGate>,
    pub http: reqwest::Client,
    pub plan: Mutex<Vec<PlanItem>>,
}

fn s(v: &Value, k: &str) -> Result<String> {
    v.get(k).and_then(Value::as_str).map(String::from).with_context(|| format!("missing string argument '{k}'"))
}

fn n(v: &Value, k: &str) -> Option<u64> {
    v.get(k).and_then(|x| x.as_u64().or_else(|| x.as_str().and_then(|s| s.parse().ok())))
}

impl AgentToolbox {
    pub fn new(root: &Path, gate: Arc<ApprovalGate>) -> Result<AgentToolbox> {
        Ok(AgentToolbox { ws: Workspace::new(root)?, gate, http: crate::http_client(), plan: Mutex::new(vec![]) })
    }

    pub fn schemas() -> Vec<Value> {
        let path = json!({"type": "string", "description": "Path relative to the workspace (or absolute)"});
        vec![
            function("update_plan", "Replace the visible step list.", json!({"steps": {"type": "array", "items": {"type": "object", "properties": {"step": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "done"]}}, "required": ["step", "status"]}}}), &["steps"]),
            function("list_files", "List files and folders (skips node_modules and .git).", json!({"path": path, "depth": {"type": "integer", "minimum": 1, "maximum": 6}}), &[]),
            function("read_file", "Read a text file with line numbers.", json!({"path": path, "offset": {"type": "integer", "description": "1-based first line"}, "limit": {"type": "integer"}}), &["path"]),
            function("write_file", "Create or overwrite a file.", json!({"path": path, "content": {"type": "string"}}), &["path", "content"]),
            function("edit_file", "Replace old_text with new_text. old_text must occur exactly once unless replace_all is true.", json!({"path": path, "old_text": {"type": "string"}, "new_text": {"type": "string"}, "replace_all": {"type": "boolean"}}), &["path", "old_text", "new_text"]),
            function("make_dir", "Create a folder (and parents).", json!({"path": path}), &["path"]),
            function("delete_path", "Delete a file or folder (asks the user).", json!({"path": path}), &["path"]),
            function("search_files", "Regex search in file contents.", json!({"pattern": {"type": "string"}, "path": path, "glob": {"type": "string", "description": "e.g. *.rs"}}), &["pattern"]),
            function("run_command", "Run a shell command in the workspace (asks the user).", json!({"command": {"type": "string"}, "timeout": {"type": "integer", "description": "seconds, default 120, max 600"}}), &["command"]),
            function("web_search", "Search the web.", json!({"query": {"type": "string"}}), &["query"]),
            function("web_fetch", "Fetch a public web page as text.", json!({"url": {"type": "string"}}), &["url"]),
        ]
    }

    /// Resolve a path and ask the user when it is outside the workspace.
    async fn path_for(&self, args: &Value, key: &str, write: bool) -> Result<PathBuf> {
        let raw = args.get(key).and_then(Value::as_str).unwrap_or(".");
        let (p, inside) = self.ws.resolve(raw);
        if !inside {
            let (k, verb) = if write { ("outside_write", "Write outside the workspace") } else { ("outside_read", "Read outside the workspace") };
            self.gate.check(k, if write { "write_file" } else { "read_file" }, format!("{verb}: {}", p.display())).await?;
        }
        Ok(p)
    }

    pub async fn execute(&self, name: &str, args: &Value) -> Result<String> {
        match name {
            "update_plan" => {
                let steps: Vec<PlanItem> = serde_json::from_value(args.get("steps").cloned().unwrap_or(json!([])))?;
                let n = steps.len();
                *self.plan.lock().unwrap() = steps;
                Ok(format!("plan updated ({n} steps)"))
            }
            "list_files" => {
                let p = self.path_for(args, "path", false).await?;
                tools::list_dir(&p, n(args, "depth").unwrap_or(2) as usize)
            }
            "read_file" => {
                let p = self.path_for(args, "path", false).await?;
                tools::read_file(&p, n(args, "offset").unwrap_or(1) as usize, n(args, "limit").unwrap_or(0) as usize)
            }
            "write_file" => {
                let p = self.path_for(args, "path", true).await?;
                tools::write_file(&p, &s(args, "content")?)
            }
            "edit_file" => {
                let p = self.path_for(args, "path", true).await?;
                tools::edit_file(&p, &s(args, "old_text")?, &s(args, "new_text")?, args.get("replace_all").and_then(Value::as_bool).unwrap_or(false))
            }
            "make_dir" => {
                let p = self.path_for(args, "path", true).await?;
                std::fs::create_dir_all(&p)?;
                Ok(format!("created {}", p.display()))
            }
            "delete_path" => {
                let (p, inside) = self.ws.resolve(&s(args, "path")?);
                if p == self.ws.root {
                    bail!("refusing to delete the workspace itself");
                }
                let where_ = if inside { "" } else { " (OUTSIDE the workspace)" };
                self.gate.check(&format!("delete:{}", p.display()), name, format!("Delete {}{where_}", p.display())).await?;
                if p.is_dir() {
                    std::fs::remove_dir_all(&p)?;
                } else {
                    std::fs::remove_file(&p)?;
                }
                Ok(format!("deleted {}", p.display()))
            }
            "search_files" => {
                let p = self.path_for(args, "path", false).await?;
                tools::search_files(&p, &s(args, "pattern")?, args.get("glob").and_then(Value::as_str))
            }
            "run_command" => {
                let cmd = s(args, "command")?;
                self.gate.check("run_command", name, format!("Run in {}: {cmd}", self.ws.root.display())).await?;
                tools::run_command(&cmd, &self.ws.root, n(args, "timeout").unwrap_or(tools::DEFAULT_TIMEOUT)).await
            }
            "web_search" => Ok(serde_json::to_string_pretty(&tools::web_search(&self.http, &s(args, "query")?).await?)?),
            "web_fetch" => tools::web_fetch(&s(args, "url")?).await,
            other => bail!("unknown tool {other}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{ApprovalRequest, Decision, Denied};

    fn setup() -> (tempfile::TempDir, tempfile::TempDir, Arc<AgentToolbox>, Arc<Mutex<Vec<ApprovalRequest>>>) {
        let ws = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let seen: Arc<Mutex<Vec<ApprovalRequest>>> = Arc::default();
        let s2 = seen.clone();
        let gate = ApprovalGate::new(Arc::new(move |r| s2.lock().unwrap().push(r)));
        let tb = Arc::new(AgentToolbox::new(ws.path(), gate).unwrap());
        (ws, outside, tb, seen)
    }

    #[test]
    fn resolves_escapes() {
        let (ws, outside, tb, _) = setup();
        assert!(tb.ws.resolve("src/main.rs").1);
        assert!(tb.ws.resolve("./a/../b.txt").1);
        assert!(!tb.ws.resolve("../x").1);
        assert!(!tb.ws.resolve("a/../../x").1);
        assert!(!tb.ws.resolve(&outside.path().join("f").display().to_string()).1);
        assert!(tb.ws.resolve(&ws.path().join("new/dir/f").display().to_string()).1);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_out_is_outside() {
        let (ws, outside, tb, _) = setup();
        std::os::unix::fs::symlink(outside.path(), ws.path().join("link")).unwrap();
        assert!(!tb.ws.resolve("link/secret.txt").1);
    }

    #[tokio::test]
    async fn inside_is_automatic_outside_asks() {
        let (ws, outside, tb, seen) = setup();
        tb.execute("write_file", &json!({"path": "src/a.txt", "content": "x"})).await.unwrap();
        tb.execute("edit_file", &json!({"path": "src/a.txt", "old_text": "x", "new_text": "y"})).await.unwrap();
        assert_eq!(std::fs::read_to_string(ws.path().join("src/a.txt")).unwrap(), "y");
        assert!(seen.lock().unwrap().is_empty());

        let target = outside.path().join("evil.txt").display().to_string();
        let t = tb.clone();
        let tgt = target.clone();
        let h = tokio::spawn(async move { t.execute("write_file", &json!({"path": tgt, "content": "x"})).await });
        while seen.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
        let req = seen.lock().unwrap()[0].clone();
        assert!(req.summary.contains("outside"));
        tb.gate.resolve(&req.id, Decision::Deny);
        assert!(h.await.unwrap().unwrap_err().is::<Denied>());
        assert!(!outside.path().join("evil.txt").exists());
    }

    #[tokio::test]
    async fn commands_and_deletes_always_ask() {
        let (ws, _o, tb, seen) = setup();
        std::fs::write(ws.path().join("f.txt"), "1").unwrap();
        let t = tb.clone();
        let h = tokio::spawn(async move { t.execute("delete_path", &json!({"path": "f.txt"})).await });
        while seen.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
        let id = seen.lock().unwrap()[0].id.clone();
        tb.gate.resolve(&id, Decision::Once);
        h.await.unwrap().unwrap();
        assert!(!ws.path().join("f.txt").exists());

        let t = tb.clone();
        let h = tokio::spawn(async move { t.execute("run_command", &json!({"command": "echo hi"})).await });
        while seen.lock().unwrap().len() < 2 {
            tokio::task::yield_now().await;
        }
        let id = seen.lock().unwrap()[1].id.clone();
        tb.gate.resolve(&id, Decision::Once);
        assert!(h.await.unwrap().unwrap().contains("hi"));
        assert!(tb.execute("delete_path", &json!({"path": "."})).await.is_err());
    }

    #[tokio::test]
    async fn plan_updates() {
        let (_ws, _o, tb, _) = setup();
        tb.execute("update_plan", &json!({"steps": [{"step": "read code", "status": "in_progress"}]})).await.unwrap();
        assert_eq!(tb.plan.lock().unwrap()[0].step, "read code");
        assert_eq!(AgentToolbox::schemas().len(), 11);
    }
}
