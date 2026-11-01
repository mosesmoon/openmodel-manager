//! Resumable, cancellable multi-file HTTP downloads.
//!
//! Data goes to `<dest>.part` and is renamed when complete, so an interrupted
//! download resumes with `Range: bytes=N-` next time.

use crate::{Progress, ProgressFn};
use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

pub use tokio_util::sync::CancellationToken as Cancel;

#[derive(Debug, Clone)]
pub struct DownloadItem {
    pub url: String,
    pub dest: PathBuf,
    /// Expected size if known (enables skip-if-complete and progress totals).
    pub size: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
#[error("download cancelled")]
pub struct Cancelled;

const MAX_RETRIES: u32 = 6;

pub fn part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

/// Download every item in order. Progress `done/total` is aggregated over all files.
pub async fn download_all(
    http: &reqwest::Client,
    items: &[DownloadItem],
    headers: &[(String, String)],
    cancel: &CancellationToken,
    job_id: &str,
    progress: &ProgressFn,
) -> Result<()> {
    let total: Option<u64> = items.iter().map(|i| i.size).sum();
    let mut before = 0u64;
    for (idx, item) in items.iter().enumerate() {
        let base = Progress { file_index: idx, file_count: items.len(), total, ..Progress::new(job_id, "downloading") };
        let got = download_one(http, item, headers, cancel, &|done, speed| {
            progress(Progress { done: before + done, speed_bps: speed, ..base.clone() });
        })
        .await?;
        before += got;
    }
    progress(Progress { done: before, total: Some(before), file_index: items.len().saturating_sub(1), file_count: items.len(), ..Progress::new(job_id, "done") });
    Ok(())
}

/// Download one file with resume and retries. Returns its final size.
pub async fn download_one(
    http: &reqwest::Client,
    item: &DownloadItem,
    headers: &[(String, String)],
    cancel: &CancellationToken,
    on_progress: &(dyn Fn(u64, Option<u64>) + Send + Sync),
) -> Result<u64> {
    if let (Ok(m), Some(sz)) = (std::fs::metadata(&item.dest), item.size) {
        if m.len() == sz {
            on_progress(sz, None);
            return Ok(sz);
        }
    }
    if let Some(d) = item.dest.parent() {
        tokio::fs::create_dir_all(d).await?;
    }
    let part = part_path(&item.dest);
    let mut attempt = 0;
    loop {
        if cancel.is_cancelled() {
            return Err(Cancelled.into());
        }
        match try_download(http, item, &part, headers, cancel, on_progress).await {
            Ok(n) => {
                tokio::fs::rename(&part, &item.dest).await.with_context(|| format!("rename {}", part.display()))?;
                return Ok(n);
            }
            Err(e) if e.is::<Cancelled>() => return Err(e),
            Err(e) if e.downcast_ref::<Fatal>().is_some() => return Err(e),
            Err(e) => {
                attempt += 1;
                if attempt > MAX_RETRIES {
                    return Err(e.context(format!("giving up on {} after {MAX_RETRIES} retries", item.url)));
                }
                tracing::warn!("download error (attempt {attempt}): {e:#}");
                let wait = Duration::from_secs(1 << attempt.min(5));
                tokio::select! {
                    _ = cancel.cancelled() => return Err(Cancelled.into()),
                    _ = tokio::time::sleep(wait) => {}
                }
            }
        }
    }
}

/// Errors that retrying cannot fix (404, auth).
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct Fatal(String);

async fn try_download(
    http: &reqwest::Client,
    item: &DownloadItem,
    part: &Path,
    headers: &[(String, String)],
    cancel: &CancellationToken,
    on_progress: &(dyn Fn(u64, Option<u64>) + Send + Sync),
) -> Result<u64> {
    let have = tokio::fs::metadata(part).await.map(|m| m.len()).unwrap_or(0);
    if let Some(sz) = item.size {
        if have == sz && sz > 0 {
            return Ok(sz);
        }
    }
    let mut req = http.get(&item.url);
    for (k, v) in headers {
        req = req.header(k, v);
    }
    if have > 0 {
        req = req.header("Range", format!("bytes={have}-"));
    }
    let resp = req.send().await?;
    let status = resp.status().as_u16();
    let (mut file, mut done) = match status {
        206 => (tokio::fs::OpenOptions::new().append(true).open(part).await?, have),
        200 => (tokio::fs::File::create(part).await?, 0),
        416 if have > 0 => return Ok(have), // already complete
        401 | 403 => return Err(Fatal(format!("HTTP {status}: access denied (gated model? check the token)")).into()),
        404 => return Err(Fatal(format!("HTTP 404: {} not found", item.url)).into()),
        s => bail!("HTTP {s} for {}", item.url),
    };
    let expected = resp.content_length().map(|l| l + done).or(item.size);
    let mut stream = resp.bytes_stream();
    let mut window = (Instant::now(), done);
    let mut speed = None;
    let mut last_emit = Instant::now() - Duration::from_secs(1);
    loop {
        let chunk = tokio::select! {
            _ = cancel.cancelled() => {
                file.flush().await?;
                return Err(Cancelled.into());
            }
            c = stream.next() => c,
        };
        let Some(chunk) = chunk else { break };
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        done += chunk.len() as u64;
        let el = window.0.elapsed();
        if el >= Duration::from_secs(2) {
            speed = Some(((done - window.1) as f64 / el.as_secs_f64()) as u64);
            window = (Instant::now(), done);
        }
        if last_emit.elapsed() >= Duration::from_millis(250) {
            on_progress(done, speed);
            last_emit = Instant::now();
        }
    }
    file.flush().await?;
    drop(file);
    if let Some(e) = expected {
        if done < e {
            bail!("connection closed early ({done} of {e} bytes)");
        }
    }
    on_progress(done, speed);
    Ok(done)
}

#[cfg(test)]
pub(crate) mod testserver {
    //! Tiny HTTP/1.1 file server with Range support, for tests.
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Serves `body` at any path. Returns the base URL.
    pub async fn serve(body: Vec<u8>, throttle: bool) -> String {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        let body = Arc::new(body);
        tokio::spawn(async move {
            loop {
                let Ok((mut s, _)) = l.accept().await else { return };
                let body = body.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let mut req = Vec::new();
                    while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                        let n = s.read(&mut buf).await.unwrap_or(0);
                        if n == 0 {
                            return;
                        }
                        req.extend_from_slice(&buf[..n]);
                    }
                    let req = String::from_utf8_lossy(&req).to_lowercase();
                    if req.starts_with("get /missing") {
                        let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n").await;
                        return;
                    }
                    let start: usize = req
                        .lines()
                        .find_map(|l| l.strip_prefix("range: bytes="))
                        .and_then(|r| r.trim_end_matches('-').trim_end_matches(|c: char| !c.is_ascii_digit()).split('-').next()?.parse().ok())
                        .unwrap_or(0);
                    let part = &body[start.min(body.len())..];
                    let head = if start > 0 {
                        format!("HTTP/1.1 206 Partial Content\r\ncontent-length: {}\r\ncontent-range: bytes {}-{}/{}\r\nconnection: close\r\n\r\n", part.len(), start, body.len() - 1, body.len())
                    } else {
                        format!("HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", part.len())
                    };
                    let _ = s.write_all(head.as_bytes()).await;
                    for c in part.chunks(8192) {
                        if s.write_all(c).await.is_err() {
                            return;
                        }
                        if throttle {
                            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                        }
                    }
                });
            }
        });
        format!("http://{addr}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn data(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    #[tokio::test]
    async fn downloads_and_resumes() {
        let body = data(300_000);
        let base = testserver::serve(body.clone(), false).await;
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("a.bin");
        // Simulate an interrupted earlier attempt.
        std::fs::write(part_path(&dest), &body[..120_000]).unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s2 = seen.clone();
        let p: ProgressFn = Arc::new(move |p| s2.lock().unwrap().push((p.stage.clone(), p.done)));
        let items = [DownloadItem { url: format!("{base}/a.bin"), dest: dest.clone(), size: Some(body.len() as u64) }];
        download_all(&reqwest::Client::new(), &items, &[], &CancellationToken::new(), "j", &p).await.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(!part_path(&dest).exists());
        let seen = seen.lock().unwrap();
        assert_eq!(seen.last().unwrap(), &("done".to_string(), body.len() as u64));
    }

    #[tokio::test]
    async fn cancel_keeps_partial_file() {
        let body = data(2_000_000);
        let base = testserver::serve(body, true).await;
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("b.bin");
        let cancel = CancellationToken::new();
        let c2 = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            c2.cancel();
        });
        let item = DownloadItem { url: format!("{base}/b.bin"), dest: dest.clone(), size: None };
        let err = download_one(&reqwest::Client::new(), &item, &[], &cancel, &|_, _| {}).await.unwrap_err();
        assert!(err.is::<Cancelled>());
        assert!(!dest.exists());
        assert!(part_path(&dest).exists());
    }

    #[tokio::test]
    async fn not_found_is_not_retried() {
        let base = testserver::serve(vec![1, 2, 3], false).await;
        let dir = tempfile::tempdir().unwrap();
        let item = DownloadItem { url: format!("{base}/missing"), dest: dir.path().join("x"), size: None };
        let t = Instant::now();
        let err = download_one(&reqwest::Client::new(), &item, &[], &CancellationToken::new(), &|_, _| {}).await.unwrap_err();
        assert!(format!("{err}").contains("404"));
        assert!(t.elapsed() < Duration::from_secs(1));
    }
}
