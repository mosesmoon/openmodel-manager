//! Tools exposed to models (chat page and Agent page) and the approval gate.
//!
//! Approval is enforced here, in the backend: a tool that needs consent
//! awaits [`ApprovalGate::check`] before touching anything. The UI only
//! displays requests and calls [`ApprovalGate::resolve`]; it cannot run a
//! guarded tool directly.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::oneshot;

// ---------------------------------------------------------------- approvals

#[derive(Debug, Clone, Serialize)]
pub struct ApprovalRequest {
    pub id: String,
    pub tool: String,
    /// Human-readable description of exactly what will happen.
    pub summary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// Allow this one call.
    Once,
    /// Allow this kind of call for the rest of the conversation.
    Always,
    Deny,
}

#[derive(Debug, thiserror::Error)]
#[error("the user declined: {0}")]
pub struct Denied(pub String);

type Notifier = Arc<dyn Fn(ApprovalRequest) + Send + Sync>;

/// One gate per conversation / agent session.
pub struct ApprovalGate {
    pending: Mutex<HashMap<String, oneshot::Sender<Decision>>>,
    always: Mutex<HashSet<String>>,
    notify: Notifier,
}

impl ApprovalGate {
    pub fn new(notify: Notifier) -> Arc<ApprovalGate> {
        Arc::new(ApprovalGate { pending: Mutex::new(HashMap::new()), always: Mutex::new(HashSet::new()), notify })
    }

    /// Wait for the user. `key` groups calls for "always allow" (e.g. "run_command").
    pub async fn check(&self, key: &str, tool: &str, summary: String) -> Result<()> {
        if self.always.lock().unwrap().contains(key) {
            return Ok(());
        }
        let id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id.clone(), tx);
        (self.notify)(ApprovalRequest { id: id.clone(), tool: tool.into(), summary: summary.clone() });
        let d = rx.await.unwrap_or(Decision::Deny);
        self.pending.lock().unwrap().remove(&id);
        match d {
            Decision::Once => Ok(()),
            Decision::Always => {
                self.always.lock().unwrap().insert(key.to_string());
                Ok(())
            }
            Decision::Deny => Err(Denied(summary).into()),
        }
    }

    /// Answer a pending request. Returns false for unknown / stale ids.
    pub fn resolve(&self, id: &str, d: Decision) -> bool {
        match self.pending.lock().unwrap().remove(id) {
            Some(tx) => tx.send(d).is_ok(),
            None => false,
        }
    }

    /// Deny everything still waiting (conversation closed / stopped).
    pub fn deny_all(&self) {
        for (_, tx) in self.pending.lock().unwrap().drain() {
            let _ = tx.send(Decision::Deny);
        }
    }

    pub fn pending_ids(&self) -> Vec<String> {
        self.pending.lock().unwrap().keys().cloned().collect()
    }
}

// ---------------------------------------------------------------- calculator / time

/// Evaluate an arithmetic expression without any code execution.
pub fn calculator(expr: &str) -> Result<String> {
    let re = regex::Regex::new(r"[A-Za-z_][A-Za-z0-9_:]*|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?").unwrap();
    let mut out = String::new();
    let mut last = 0;
    for m in re.find_iter(expr) {
        out.push_str(&expr[last..m.start()]);
        let t = m.as_str();
        if t.chars().next().unwrap().is_ascii_digit() {
            // Force float maths so 1/3 is 0.333 rather than integer 0.
            if t.contains('.') || t.contains('e') || t.contains('E') {
                out.push_str(t);
            } else {
                out.push_str(t);
                out.push_str(".0");
            }
        } else {
            let mapped = match t {
                "sqrt" | "sin" | "cos" | "tan" | "ln" | "log10" | "log2" | "exp" | "abs" | "asin" | "acos" | "atan" | "cbrt" => format!("math::{t}"),
                "pi" | "PI" => "3.141592653589793".into(),
                "e" | "E" => "2.718281828459045".into(),
                other => other.into(),
            };
            out.push_str(&mapped);
        }
        last = m.end();
    }
    out.push_str(&expr[last..]);
    let v = evalexpr::eval(&out).map_err(|e| anyhow::anyhow!("cannot evaluate '{expr}': {e}"))?;
    Ok(match v {
        evalexpr::Value::Float(f) if f.fract() == 0.0 && f.abs() < 1e15 => format!("{}", f as i64),
        evalexpr::Value::Float(f) => format!("{}", (f * 1e12).round() / 1e12),
        other => other.to_string(),
    })
}

pub fn current_time() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S %:z (%A)").to_string()
}

// ---------------------------------------------------------------- web

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

fn xml_unescape(s: &str) -> String {
    s.replace("<![CDATA[", "").replace("]]>", "").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'").replace("&apos;", "'").replace("&amp;", "&")
}

pub fn parse_rss(xml: &str) -> Vec<SearchResult> {
    let item = regex::Regex::new(r"(?s)<item>(.*?)</item>").unwrap();
    let field = |body: &str, tag: &str| {
        regex::Regex::new(&format!(r"(?s)<{tag}>(.*?)</{tag}>")).unwrap().captures(body).map(|c| xml_unescape(c[1].trim())).unwrap_or_default()
    };
    item.captures_iter(xml)
        .map(|c| SearchResult { title: field(&c[1], "title"), url: field(&c[1], "link"), snippet: strip_html(&field(&c[1], "description")) })
        .filter(|r| r.url.starts_with("http"))
        .collect()
}

/// Web search through Bing's public RSS endpoint (no API key, no account).
pub async fn web_search(http: &reqwest::Client, q: &str) -> Result<Vec<SearchResult>> {
    let mut last_err = None;
    for host in ["https://www.bing.com", "https://cn.bing.com"] {
        let r = http.get(format!("{host}/search")).query(&[("format", "rss"), ("count", "10"), ("q", q)]).timeout(Duration::from_secs(15)).send().await;
        match r {
            Ok(r) if r.status().is_success() => {
                let res = parse_rss(&r.text().await?);
                if !res.is_empty() {
                    return Ok(res);
                }
            }
            Ok(r) => last_err = Some(anyhow::anyhow!("HTTP {}", r.status())),
            Err(e) => last_err = Some(e.into()),
        }
    }
    match last_err {
        Some(e) => Err(e.context("web search failed")),
        None => Ok(vec![]),
    }
}

pub fn strip_html(html: &str) -> String {
    let drop = regex::Regex::new(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</(script|style|noscript|svg|head)>").unwrap();
    let block = regex::Regex::new(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr)[^>]*>").unwrap();
    let tag = regex::Regex::new(r"(?s)<[^>]+>").unwrap();
    let s = drop.replace_all(html, " ");
    let s = block.replace_all(&s, "\n");
    let s = tag.replace_all(&s, " ");
    let s = xml_unescape(&s).replace("&nbsp;", " ");
    let ws = regex::Regex::new(r"[ \t\r\f]+").unwrap();
    let s = ws.replace_all(&s, " ");
    let lines: Vec<&str> = s.lines().map(str::trim).collect();
    let nl = regex::Regex::new(r"\n{3,}").unwrap();
    nl.replace_all(lines.join("\n").trim(), "\n\n").to_string()
}

fn is_private(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v) => v.is_private() || v.is_loopback() || v.is_link_local() || v.is_unspecified() || v.is_broadcast() || v.octets()[0] == 100 && (v.octets()[1] & 0xc0) == 64,
        IpAddr::V6(v) => v.is_loopback() || v.is_unspecified() || (v.segments()[0] & 0xfe00) == 0xfc00 || (v.segments()[0] & 0xffc0) == 0xfe80 || v.to_ipv4_mapped().map(|m| is_private(IpAddr::V4(m))).unwrap_or(false),
    }
}

/// Refuse URLs that point at this machine or the local network, so a model
/// cannot be steered into probing routers, admin panels or local servers.
pub async fn check_public_url(url: &reqwest::Url) -> Result<()> {
    if !matches!(url.scheme(), "http" | "https") {
        bail!("only http/https URLs can be fetched");
    }
    let host = url.host_str().context("URL has no host")?;
    let port = url.port_or_known_default().unwrap_or(80);
    let addrs: Vec<_> = tokio::net::lookup_host((host.trim_matches(['[', ']']), port)).await.with_context(|| format!("cannot resolve {host}"))?.collect();
    if addrs.is_empty() || addrs.iter().any(|a| is_private(a.ip())) {
        bail!("{host} points to a local or private address; refusing to fetch it");
    }
    Ok(())
}

const FETCH_LIMIT: usize = 20_000;

/// Fetch a public web page as plain text (redirects are checked hop by hop).
pub async fn web_fetch(url: &str) -> Result<String> {
    let http = reqwest::Client::builder()
        .user_agent(concat!("OpenModel-Manager/", env!("CARGO_PKG_VERSION")))
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(25))
        .build()?;
    let mut u = reqwest::Url::parse(url).context("invalid URL")?;
    for _ in 0..6 {
        check_public_url(&u).await?;
        let r = http.get(u.clone()).send().await?;
        if r.status().is_redirection() {
            let loc = r.headers().get("location").and_then(|v| v.to_str().ok()).context("redirect without location")?;
            u = u.join(loc)?;
            continue;
        }
        if !r.status().is_success() {
            bail!("HTTP {}", r.status());
        }
        let ct = r.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_lowercase();
        let body = r.text().await?;
        let text = if ct.contains("html") || body.trim_start().starts_with('<') { strip_html(&body) } else { body };
        return Ok(truncate(&text, FETCH_LIMIT));
    }
    bail!("too many redirects")
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n…[truncated {} bytes]", &s[..end], s.len() - end)
}

// ---------------------------------------------------------------- files / commands

const SKIP_DIRS: &[&str] = &["node_modules", ".git", "target", "__pycache__", ".venv"];
const MAX_READ: u64 = 10 * 1024 * 1024;

pub fn read_file(path: &Path, offset: usize, limit: usize) -> Result<String> {
    let meta = std::fs::metadata(path).with_context(|| format!("{} not found", path.display()))?;
    if meta.len() > MAX_READ {
        bail!("{} is larger than 10 MB", path.display());
    }
    let bytes = std::fs::read(path)?;
    if bytes.iter().take(8000).any(|b| *b == 0) {
        bail!("{} looks like a binary file", path.display());
    }
    let text = String::from_utf8_lossy(&bytes);
    let start = offset.max(1);
    let limit = if limit == 0 { 2000 } else { limit };
    let lines: Vec<&str> = text.lines().collect();
    let mut out: String = lines.iter().enumerate().skip(start - 1).take(limit).map(|(i, l)| format!("{:>6}\t{}\n", i + 1, l)).collect();
    if start - 1 + limit < lines.len() {
        out.push_str(&format!("… {} more lines; read again with offset={}\n", lines.len() - (start - 1 + limit), start + limit));
    }
    Ok(out)
}

pub fn write_file(path: &Path, content: &str) -> Result<String> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    std::fs::write(path, content)?;
    Ok(format!("wrote {} bytes to {}", content.len(), path.display()))
}

pub fn edit_file(path: &Path, old: &str, new: &str, replace_all: bool) -> Result<String> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    if old.is_empty() {
        bail!("old_text must not be empty");
    }
    let n = text.matches(old).count();
    match (n, replace_all) {
        (0, _) => bail!("old_text not found in {}", path.display()),
        (n, false) if n > 1 => bail!("old_text appears {n} times in {}; add more context or set replace_all", path.display()),
        _ => {}
    }
    let out = if replace_all { text.replace(old, new) } else { text.replacen(old, new, 1) };
    std::fs::write(path, out)?;
    Ok(format!("replaced {} occurrence(s) in {}", if replace_all { n } else { 1 }, path.display()))
}

pub fn list_dir(root: &Path, depth: usize) -> Result<String> {
    let depth = depth.clamp(1, 6);
    let mut out = String::new();
    let mut count = 0;
    fn walk(dir: &Path, base: &Path, d: usize, max: usize, out: &mut String, count: &mut usize) -> Result<()> {
        let mut entries: Vec<_> = std::fs::read_dir(dir)?.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            if *count >= 2000 {
                return Ok(());
            }
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            let rel = p.strip_prefix(base).unwrap_or(&p).display().to_string().replace('\\', "/");
            if p.is_dir() {
                out.push_str(&format!("{rel}/\n"));
                *count += 1;
                if d < max && !SKIP_DIRS.contains(&name.as_str()) {
                    walk(&p, base, d + 1, max, out, count)?;
                }
            } else {
                let sz = e.metadata().map(|m| m.len()).unwrap_or(0);
                out.push_str(&format!("{rel} ({sz} B)\n"));
                *count += 1;
            }
        }
        Ok(())
    }
    walk(root, root, 1, depth, &mut out, &mut count)?;
    if count >= 2000 {
        out.push_str("… listing truncated at 2000 entries\n");
    }
    Ok(out)
}

pub fn search_files(root: &Path, pattern: &str, glob: Option<&str>) -> Result<String> {
    let re = regex::Regex::new(pattern).context("invalid regex")?;
    let g = glob.map(|g| globset::Glob::new(g).map(|g| g.compile_matcher())).transpose()?;
    let mut hits = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            if p.is_dir() {
                if !SKIP_DIRS.contains(&name.as_str()) {
                    stack.push(p);
                }
                continue;
            }
            if let Some(g) = &g {
                if !g.is_match(&name) && !g.is_match(p.strip_prefix(root).unwrap_or(&p)) {
                    continue;
                }
            }
            if e.metadata().map(|m| m.len() > 2 * 1024 * 1024).unwrap_or(true) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else { continue };
            for (i, l) in text.lines().enumerate() {
                if re.is_match(l) {
                    let rel = p.strip_prefix(root).unwrap_or(&p).display().to_string().replace('\\', "/");
                    hits.push(format!("{rel}:{}: {}", i + 1, truncate(l.trim(), 200)));
                    if hits.len() >= 200 {
                        hits.push("… stopped at 200 matches".into());
                        return Ok(hits.join("\n"));
                    }
                }
            }
        }
    }
    Ok(if hits.is_empty() { "no matches".into() } else { hits.join("\n") })
}

pub const DEFAULT_TIMEOUT: u64 = 120;
pub const MAX_TIMEOUT: u64 = 600;

/// Run a shell command with a timeout; returns exit code and combined output.
pub async fn run_command(cmd: &str, cwd: &Path, timeout_secs: u64) -> Result<String> {
    let t = timeout_secs.clamp(1, MAX_TIMEOUT);
    let mut c = if cfg!(windows) {
        let mut c = tokio::process::Command::new("cmd");
        c.arg("/C").arg(cmd);
        c
    } else {
        let mut c = tokio::process::Command::new("sh");
        c.arg("-c").arg(cmd);
        c
    };
    c.current_dir(cwd).stdin(std::process::Stdio::null()).kill_on_drop(true);
    #[cfg(windows)]
    c.creation_flags(0x0800_0000);
    let fut = c.output();
    let out = match tokio::time::timeout(Duration::from_secs(t), fut).await {
        Ok(r) => r?,
        Err(_) => bail!("command timed out after {t}s"),
    };
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    let err = String::from_utf8_lossy(&out.stderr);
    if !err.trim().is_empty() {
        s.push_str("\n[stderr]\n");
        s.push_str(&err);
    }
    Ok(format!("exit code: {}\n{}", out.status.code().unwrap_or(-1), truncate(&s, 30_000)))
}

// ---------------------------------------------------------------- chat toolbox

/// Tools for the chat page (OpenAI function-calling format).
pub struct ChatToolbox {
    pub http: reqwest::Client,
    pub gate: Arc<ApprovalGate>,
    /// Base for relative paths.
    pub cwd: PathBuf,
}

pub const CHAT_MAX_ROUNDS: usize = 12;

fn s(v: &Value, k: &str) -> Result<String> {
    v.get(k).and_then(Value::as_str).map(String::from).with_context(|| format!("missing string argument '{k}'"))
}

fn u(v: &Value, k: &str) -> Option<u64> {
    v.get(k).and_then(|x| x.as_u64().or_else(|| x.as_str().and_then(|s| s.parse().ok())))
}

pub fn function(name: &str, desc: &str, props: Value, required: &[&str]) -> Value {
    json!({"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": required}}})
}

impl ChatToolbox {
    pub fn schemas() -> Vec<Value> {
        vec![
            function("get_current_time", "Current local date and time.", json!({}), &[]),
            function("calculator", "Evaluate an arithmetic expression, e.g. (3+4)*sqrt(2)/7.", json!({"expression": {"type": "string"}}), &["expression"]),
            function("web_search", "Search the web. Returns titles, URLs and snippets.", json!({"query": {"type": "string"}}), &["query"]),
            function("web_fetch", "Fetch a public web page and return its text.", json!({"url": {"type": "string"}}), &["url"]),
            function("read_file", "Read a text file from the user's computer (asks the user first).", json!({"path": {"type": "string"}}), &["path"]),
            function("list_dir", "List a folder on the user's computer (asks the user first).", json!({"path": {"type": "string"}}), &["path"]),
            function("write_file", "Create or overwrite a file (asks the user first).", json!({"path": {"type": "string"}, "content": {"type": "string"}}), &["path", "content"]),
            function("run_command", "Run a shell command (asks the user first).", json!({"command": {"type": "string"}, "timeout": {"type": "integer"}}), &["command"]),
        ]
    }

    fn path(&self, p: &str) -> PathBuf {
        let p = Path::new(p);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.cwd.join(p)
        }
    }

    pub async fn execute(&self, name: &str, args: &Value) -> Result<String> {
        match name {
            "get_current_time" => Ok(current_time()),
            "calculator" => calculator(&s(args, "expression")?),
            "web_search" => Ok(serde_json::to_string_pretty(&web_search(&self.http, &s(args, "query")?).await?)?),
            "web_fetch" => web_fetch(&s(args, "url")?).await,
            "read_file" => {
                let p = self.path(&s(args, "path")?);
                self.gate.check("read_file", name, format!("Read file {}", p.display())).await?;
                read_file(&p, 1, 2000)
            }
            "list_dir" => {
                let p = self.path(&s(args, "path")?);
                self.gate.check("list_dir", name, format!("List folder {}", p.display())).await?;
                list_dir(&p, 1)
            }
            "write_file" => {
                let p = self.path(&s(args, "path")?);
                let c = s(args, "content")?;
                self.gate.check("write_file", name, format!("Write {} bytes to {}", c.len(), p.display())).await?;
                write_file(&p, &c)
            }
            "run_command" => {
                let cmd = s(args, "command")?;
                self.gate.check("run_command", name, format!("Run: {cmd}  (in {})", self.cwd.display())).await?;
                run_command(&cmd, &self.cwd, u(args, "timeout").unwrap_or(DEFAULT_TIMEOUT)).await
            }
            other => bail!("unknown tool {other}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calc() {
        assert_eq!(calculator("1/3*3").unwrap(), "1");
        assert_eq!(calculator("(3+4)*2").unwrap(), "14");
        assert_eq!(calculator("sqrt(16) + 2^3").unwrap(), "12");
        assert_eq!(calculator("1/4").unwrap(), "0.25");
        assert!(calculator("std::process::exit(1)").is_err());
    }

    #[test]
    fn rss_and_html() {
        let x = r#"<rss><channel><item><title>Rust &amp; you</title><link>https://rust-lang.org/</link><description>&lt;b&gt;Fast&lt;/b&gt; language</description></item><item><title>bad</title><link>javascript:x</link></item></channel></rss>"#;
        let r = parse_rss(x);
        assert_eq!(r, vec![SearchResult { title: "Rust & you".into(), url: "https://rust-lang.org/".into(), snippet: "Fast language".into() }]);
        let t = strip_html("<html><head><title>x</title></head><body><script>evil()</script><p>Hello&nbsp;<b>world</b></p><p>Second</p></body></html>");
        assert_eq!(t, "Hello world\nSecond");
    }

    #[tokio::test]
    async fn refuses_private_targets() {
        for u in ["http://127.0.0.1:8080/", "http://localhost/", "http://192.168.1.1/", "http://[::1]/", "file:///etc/passwd", "http://169.254.169.254/latest/meta-data"] {
            assert!(web_fetch(u).await.is_err(), "{u} should be refused");
        }
    }

    #[test]
    fn file_helpers() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("a.txt");
        write_file(&f, "one\ntwo\ntwo\nthree\n").unwrap();
        assert!(read_file(&f, 2, 1).unwrap().starts_with("     2\ttwo\n"));
        assert!(edit_file(&f, "two", "2", false).unwrap_err().to_string().contains("2 times"));
        edit_file(&f, "two", "2", true).unwrap();
        edit_file(&f, "three", "3", false).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "one\n2\n2\n3\n");
        std::fs::create_dir_all(d.path().join("node_modules/x")).unwrap();
        std::fs::write(d.path().join("node_modules/x/y.js"), "2").unwrap();
        let l = list_dir(d.path(), 3).unwrap();
        assert!(l.contains("a.txt") && l.contains("node_modules/") && !l.contains("y.js"));
        let s = search_files(d.path(), r"^\d$", None).unwrap();
        assert!(s.contains("a.txt:2: 2") && !s.contains("y.js"));
        std::fs::write(d.path().join("bin"), [0u8, 1, 2]).unwrap();
        assert!(read_file(&d.path().join("bin"), 1, 10).is_err());
    }

    #[tokio::test]
    async fn command_runs_and_times_out() {
        let d = tempfile::tempdir().unwrap();
        let out = run_command("echo hello", d.path(), 30).await.unwrap();
        assert!(out.starts_with("exit code: 0") && out.contains("hello"));
        let slow = if cfg!(windows) { "ping -n 6 127.0.0.1 >NUL" } else { "sleep 5" };
        assert!(run_command(slow, d.path(), 1).await.unwrap_err().to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn gate_requires_user_decision() {
        let seen: Arc<Mutex<Vec<ApprovalRequest>>> = Arc::default();
        let s2 = seen.clone();
        let gate = ApprovalGate::new(Arc::new(move |r| s2.lock().unwrap().push(r)));
        let d = tempfile::tempdir().unwrap();
        let tb = Arc::new(ChatToolbox { http: reqwest::Client::new(), gate: gate.clone(), cwd: d.path().to_path_buf() });

        // Denied: nothing is written.
        let t = tb.clone();
        let h = tokio::spawn(async move { t.execute("write_file", &json!({"path": "x.txt", "content": "hi"})).await });
        while seen.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
        let id = seen.lock().unwrap()[0].id.clone();
        assert!(gate.resolve(&id, Decision::Deny));
        assert!(h.await.unwrap().unwrap_err().is::<Denied>());
        assert!(!d.path().join("x.txt").exists());
        assert!(!gate.resolve(&id, Decision::Once), "stale ids are rejected");

        // "Always" skips the prompt next time.
        let t = tb.clone();
        let h = tokio::spawn(async move { t.execute("write_file", &json!({"path": "x.txt", "content": "hi"})).await });
        while seen.lock().unwrap().len() < 2 {
            tokio::task::yield_now().await;
        }
        let id = seen.lock().unwrap()[1].id.clone();
        gate.resolve(&id, Decision::Always);
        h.await.unwrap().unwrap();
        tb.execute("write_file", &json!({"path": "y.txt", "content": "hi"})).await.unwrap();
        assert_eq!(seen.lock().unwrap().len(), 2);
        assert!(d.path().join("y.txt").exists());

        // Tools without side effects need no approval.
        assert_eq!(tb.execute("calculator", &json!({"expression": "2*21"})).await.unwrap(), "42");
    }
}
