//! Saved conversations: one JSON file per chat under `<base>/chats/`.
//!
//! The record never holds API keys or endpoint URLs, only the endpoint name;
//! `history` and `items` are opaque JSON owned by the frontend.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Newest file format this build can read.
pub const FORMAT_VERSION: u32 = 1;
/// A single record above this size is refused.
pub const MAX_RECORD_BYTES: usize = 64 * 1024 * 1024;
/// The list shows at most this many conversations.
pub const MAX_LIST: usize = 200;
const TITLE_CHARS: usize = 40;
const FALLBACK_TITLE: &str = "Untitled";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatMode {
    Chat,
    Agent,
}

fn format_version() -> u32 {
    FORMAT_VERSION
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRecord {
    #[serde(default = "format_version")]
    pub v: u32,
    pub id: String,
    pub mode: ChatMode,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub created: u64,
    #[serde(default)]
    pub updated: u64,
    /// Endpoint name ("local" or a provider name), never its URL or key.
    #[serde(default)]
    pub source: String,
    /// Working folder of an agent conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<String>,
    /// Messages sent to the model, replayed as-is when the chat continues.
    #[serde(default)]
    pub history: Vec<Value>,
    /// What the screen shows.
    #[serde(default)]
    pub items: Vec<Value>,
}

/// The fields the list needs; parsing only these keeps listing cheap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMeta {
    pub id: String,
    pub title: String,
    pub mode: ChatMode,
    pub updated: u64,
    pub source: String,
}

#[derive(Deserialize)]
struct VersionOnly {
    v: u32,
}

fn chats_dir(base: &Path) -> PathBuf {
    base.join("chats")
}

/// Only a canonical UUID may reach a path, which rules out traversal.
fn file_for(base: &Path, id: &str) -> Result<PathBuf> {
    let ok = id.len() == 36 && uuid::Uuid::parse_str(id).is_ok();
    if !ok {
        bail!("invalid chat id");
    }
    Ok(chats_dir(base).join(format!("{}.json", id.to_ascii_lowercase())))
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Text of the first user message; content is a string or an array of parts.
fn first_user_text(history: &[Value]) -> Option<String> {
    let msg = history.iter().find(|m| m.get("role").and_then(Value::as_str) == Some("user"))?;
    match msg.get("content")? {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => parts.iter().find_map(|p| {
            (p.get("type").and_then(Value::as_str) == Some("text")).then(|| p.get("text").and_then(Value::as_str)).flatten().map(str::to_string)
        }),
        _ => None,
    }
}

fn derive_title(history: &[Value]) -> String {
    let text = first_user_text(history).unwrap_or_default();
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.is_empty() {
        FALLBACK_TITLE.into()
    } else {
        flat.chars().take(TITLE_CHARS).collect()
    }
}

/// Newest conversations first, optionally only one mode. Unreadable files are skipped.
pub fn list(base: &Path, mode: Option<ChatMode>) -> Vec<ChatMeta> {
    let Ok(rd) = std::fs::read_dir(chats_dir(base)) else { return vec![] };
    let mut out: Vec<ChatMeta> = rd
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| {
            let meta: ChatMeta = serde_json::from_slice(&std::fs::read(e.path()).ok()?).ok()?;
            // The file name is the id; a mismatch means a stray or edited file.
            let stem = e.path().file_stem()?.to_str()?.to_string();
            (stem == meta.id.to_ascii_lowercase() && file_for(base, &meta.id).is_ok()).then_some(meta)
        })
        .filter(|m| mode.is_none_or(|w| w == m.mode))
        .collect();
    out.sort_by(|a, b| b.updated.cmp(&a.updated).then_with(|| a.id.cmp(&b.id)));
    out.truncate(MAX_LIST);
    out
}

pub fn load(base: &Path, id: &str) -> Result<ChatRecord> {
    let path = file_for(base, id)?;
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let ver: VersionOnly = serde_json::from_slice(&bytes).context("not a chat record")?;
    if ver.v > FORMAT_VERSION {
        bail!("this chat was saved by a newer version (format {})", ver.v);
    }
    serde_json::from_slice(&bytes).context("not a chat record")
}

/// Create or overwrite. `created` comes from the existing file, `updated` is
/// set here. With `enabled` false nothing is written.
pub fn save(base: &Path, enabled: bool, mut rec: ChatRecord) -> Result<()> {
    let path = file_for(base, &rec.id)?;
    if !enabled {
        return Ok(());
    }
    let t = now();
    let previous = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice::<ChatRecord>(&b).ok());
    rec.v = FORMAT_VERSION;
    rec.id = rec.id.to_ascii_lowercase();
    rec.created = previous.map(|p| p.created).filter(|c| *c > 0).unwrap_or(if rec.created > 0 { rec.created } else { t });
    rec.updated = t;
    if rec.title.trim().is_empty() {
        rec.title = derive_title(&rec.history);
    }
    let bytes = serde_json::to_vec(&rec)?;
    if bytes.len() > MAX_RECORD_BYTES {
        bail!("chat is too large to save (over 64 MB)");
    }
    let dir = chats_dir(base);
    std::fs::create_dir_all(&dir)?;
    // Write beside the target, then rename, so a crash leaves no half file.
    let tmp = dir.join(format!("{}.json.tmp", rec.id));
    std::fs::write(&tmp, &bytes)?;
    if let Err(e) = std::fs::rename(&tmp, &path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}

pub fn delete(base: &Path, id: &str) -> Result<()> {
    let path = file_for(base, id)?;
    match std::fs::remove_file(&path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
        _ => Ok(()),
    }
}

/// Remove every record (and leftover temp files), keeping the folder.
pub fn delete_all(base: &Path) -> Result<()> {
    let Ok(rd) = std::fs::read_dir(chats_dir(base)) else { return Ok(()) };
    for e in rd.flatten() {
        let p = e.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.ends_with(".json") || name.ends_with(".json.tmp") {
            std::fs::remove_file(&p)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Tmp(PathBuf);
    impl Tmp {
        fn new() -> Tmp {
            let p = std::env::temp_dir().join(format!("openmodel-chatlog-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&p).unwrap();
            Tmp(p)
        }
    }
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn rec(mode: ChatMode, history: Vec<Value>) -> ChatRecord {
        ChatRecord {
            v: 1,
            id: uuid::Uuid::new_v4().to_string(),
            mode,
            title: "t".into(),
            created: 0,
            updated: 0,
            source: "local".into(),
            workspace: None,
            history,
            items: vec![json!({"kind": "user", "text": "hi"})],
        }
    }

    /// Rewrite created/updated on disk to fixed values.
    fn backdate(base: &Path, id: &str, created: u64, updated: u64) {
        let p = file_for(base, id).unwrap();
        let mut r: ChatRecord = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        r.created = created;
        r.updated = updated;
        std::fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
    }

    #[test]
    fn round_trip() {
        let t = Tmp::new();
        let mut r = rec(ChatMode::Agent, vec![json!({"role": "user", "content": "hello"}), json!({"role": "assistant", "content": [{"type": "text", "text": "x"}], "_anthropic": [1, 2]})]);
        r.workspace = Some("F:\\w".into());
        save(&t.0, true, r.clone()).unwrap();
        let got = load(&t.0, &r.id).unwrap();
        assert!(got.created > 0 && got.updated > 0 && got.v == 1);
        // Everything else is identical once the backend-set times are copied.
        let mut want = r;
        want.created = got.created;
        want.updated = got.updated;
        assert_eq!(got, want);
    }

    #[test]
    fn bad_ids_rejected() {
        let t = Tmp::new();
        for id in ["../x", "a/b", "", "..\\x", "not-a-uuid", "{67e55044-10b1-426f-9247-bb680e5fe0c8}", "67e55044-10b1-426f-9247-bb680e5fe0c8/.."] {
            assert!(load(&t.0, id).is_err(), "load {id:?}");
            assert!(delete(&t.0, id).is_err(), "delete {id:?}");
            let mut r = rec(ChatMode::Chat, vec![]);
            r.id = id.into();
            assert!(save(&t.0, true, r.clone()).is_err(), "save {id:?}");
            assert!(save(&t.0, false, r).is_err(), "save(off) {id:?}");
        }
        assert!(!t.0.join("chats").exists());
    }

    #[test]
    fn corrupt_file_skipped() {
        let t = Tmp::new();
        let r = rec(ChatMode::Chat, vec![]);
        save(&t.0, true, r.clone()).unwrap();
        let dir = t.0.join("chats");
        std::fs::write(dir.join(format!("{}.json", uuid::Uuid::new_v4())), b"{ not json").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        let l = list(&t.0, None);
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].id, r.id);
    }

    #[test]
    fn created_kept_updated_changes() {
        let t = Tmp::new();
        let r = rec(ChatMode::Chat, vec![]);
        save(&t.0, true, r.clone()).unwrap();
        backdate(&t.0, &r.id, 100, 100);
        let mut again = r.clone();
        again.created = 5; // ignored: the file wins
        save(&t.0, true, again).unwrap();
        let got = load(&t.0, &r.id).unwrap();
        assert_eq!(got.created, 100);
        assert!(got.updated > 100);
    }

    #[test]
    fn list_order_and_filter() {
        let t = Tmp::new();
        let (a, b, c) = (rec(ChatMode::Chat, vec![]), rec(ChatMode::Agent, vec![]), rec(ChatMode::Chat, vec![]));
        for r in [&a, &b, &c] {
            save(&t.0, true, r.clone()).unwrap();
        }
        backdate(&t.0, &a.id, 1, 10);
        backdate(&t.0, &b.id, 1, 30);
        backdate(&t.0, &c.id, 1, 20);
        let ids = |m| list(&t.0, m).into_iter().map(|x| x.id).collect::<Vec<_>>();
        assert_eq!(ids(None), [b.id.clone(), c.id.clone(), a.id.clone()]);
        assert_eq!(ids(Some(ChatMode::Chat)), [c.id.clone(), a.id.clone()]);
        assert_eq!(ids(Some(ChatMode::Agent)), [b.id]);
    }

    #[test]
    fn delete_one_and_all() {
        let t = Tmp::new();
        let (a, b) = (rec(ChatMode::Chat, vec![]), rec(ChatMode::Agent, vec![]));
        save(&t.0, true, a.clone()).unwrap();
        save(&t.0, true, b).unwrap();
        delete(&t.0, &a.id).unwrap();
        assert!(load(&t.0, &a.id).is_err());
        delete(&t.0, &a.id).unwrap(); // already gone
        assert_eq!(list(&t.0, None).len(), 1);
        delete_all(&t.0).unwrap();
        assert!(list(&t.0, None).is_empty());
        delete_all(&t.0.join("nothing")).unwrap();
    }

    #[test]
    fn newer_version_rejected() {
        let t = Tmp::new();
        let r = rec(ChatMode::Chat, vec![]);
        save(&t.0, true, r.clone()).unwrap();
        let p = file_for(&t.0, &r.id).unwrap();
        let mut v: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        v["v"] = json!(FORMAT_VERSION + 1);
        v["something_new"] = json!({"shape": "unknown"});
        std::fs::write(&p, serde_json::to_vec(&v).unwrap()).unwrap();
        assert!(load(&t.0, &r.id).unwrap_err().to_string().contains("newer"));
    }

    #[test]
    fn title_derivation() {
        let t = Tmp::new();
        let long = "a".repeat(100);
        let mut r = rec(ChatMode::Chat, vec![json!({"role": "system", "content": "sys"}), json!({"role": "user", "content": format!("  {long}\nmore")})]);
        r.title = " ".into();
        save(&t.0, true, r.clone()).unwrap();
        assert_eq!(load(&t.0, &r.id).unwrap().title, "a".repeat(40));

        let mut p = rec(ChatMode::Chat, vec![json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}}, {"type": "text", "text": "看看這張圖"}]})]);
        p.title = String::new();
        save(&t.0, true, p.clone()).unwrap();
        assert_eq!(load(&t.0, &p.id).unwrap().title, "看看這張圖");

        let mut n = rec(ChatMode::Chat, vec![]);
        n.title = String::new();
        save(&t.0, true, n.clone()).unwrap();
        assert_eq!(load(&t.0, &n.id).unwrap().title, FALLBACK_TITLE);
    }

    #[test]
    fn save_disabled_writes_nothing() {
        let t = Tmp::new();
        save(&t.0, false, rec(ChatMode::Chat, vec![])).unwrap();
        assert!(!t.0.join("chats").exists());
    }
}
