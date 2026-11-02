//! Chat client: streaming, and the tool-calling loop used by the chat page and
//! the Agent page.
//!
//! History is kept in OpenAI's message shape. OpenAI-compatible endpoints (the
//! local llama-server, OpenAI, Google AI Studio, xAI, NVIDIA NIM, ...) get it
//! as is; Anthropic endpoints go through `crate::anthropic`.

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::future::Future;
use tokio_util::sync::CancellationToken;

/// Wire protocol of an endpoint.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ApiKind {
    /// `POST /chat/completions`, bearer key.
    #[default]
    Openai,
    /// Anthropic Messages API (`POST /messages`, `x-api-key`).
    Anthropic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Endpoint {
    /// e.g. http://127.0.0.1:8080/v1 or https://api.example.com/v1
    pub base_url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    /// Ignored by llama-server; required by most hosted APIs.
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub kind: ApiKind,
}

impl Endpoint {
    pub fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    Text { delta: String },
    Reasoning { delta: String },
    ToolCall { id: String, name: String, arguments: String },
    ToolResult { id: String, name: String, content: String, ok: bool },
    Usage { prompt_tokens: u64, completion_tokens: u64, tokens_per_sec: Option<f64> },
    /// `reason`: stop / length / max_rounds / cancelled
    Done { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// One assistant reply, assembled from the stream.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Turn {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    /// Raw content blocks of an Anthropic reply, replayed unchanged.
    pub native: Option<Value>,
}

impl Turn {
    /// The message to append to the conversation history.
    pub fn to_message(&self) -> Value {
        let mut m = json!({"role": "assistant", "content": self.content});
        if !self.tool_calls.is_empty() {
            m["tool_calls"] = self
                .tool_calls
                .iter()
                .map(|c| json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}))
                .collect();
        }
        if let Some(n) = &self.native {
            m[crate::anthropic::NATIVE_KEY] = n.clone();
        }
        m
    }
}

/// History for an OpenAI-compatible server: drop our private `_` keys.
fn openai_messages(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| match m {
            Value::Object(o) if o.keys().any(|k| k.starts_with('_')) => Value::Object(o.iter().filter(|(k, _)| !k.starts_with('_')).map(|(k, v)| (k.clone(), v.clone())).collect()),
            other => other.clone(),
        })
        .collect()
}

/// Read an SSE body, calling `f` with each `data:` payload until it returns
/// `Ok(true)`, the body ends, or `cancel` fires.
async fn for_each_sse(resp: reqwest::Response, cancel: &CancellationToken, mut f: impl FnMut(&str) -> Result<bool>) -> Result<()> {
    let mut buf: Vec<u8> = vec![];
    let mut stream = resp.bytes_stream();
    loop {
        let chunk = tokio::select! {
            c = stream.next() => c,
            _ = cancel.cancelled() => bail!(crate::download::Cancelled),
        };
        let Some(chunk) = chunk else { return Ok(()) };
        buf.extend_from_slice(&chunk?);
        while let Some(nl) = buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&line);
            if let Some(data) = sse_data(line.trim_end_matches('\n')) {
                if f(data)? {
                    return Ok(());
                }
            }
        }
    }
}

async fn send(req: reqwest::RequestBuilder, ep: &Endpoint, cancel: &CancellationToken) -> Result<reqwest::Response> {
    let resp = tokio::select! {
        r = req.send() => r.with_context(|| format!("cannot reach {}", ep.base_url))?,
        _ = cancel.cancelled() => bail!(crate::download::Cancelled),
    };
    if !resp.status().is_success() {
        let st = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("{st}: {}", crate::tools::truncate(&text, 2000));
    }
    Ok(resp)
}

/// Assembles streamed chunks. Tool call fragments arrive keyed by `index`;
/// the id and name come first and the arguments in pieces.
#[derive(Default)]
pub struct StreamAccumulator {
    turn: Turn,
    calls: BTreeMap<u64, ToolCall>,
}

/// Payload of an SSE `data:` line, if this is one.
pub fn sse_data(line: &str) -> Option<&str> {
    let l = line.trim_end_matches('\r');
    l.strip_prefix("data:").map(str::trim_start)
}

impl StreamAccumulator {
    pub fn feed(&mut self, chunk: &Value) -> Vec<ChatEvent> {
        let mut ev = vec![];
        if let Some(c) = chunk.pointer("/choices/0") {
            let d = &c["delta"];
            for (key, is_text) in [("reasoning_content", false), ("reasoning", false), ("content", true)] {
                if let Some(t) = d.get(key).and_then(Value::as_str).filter(|t| !t.is_empty()) {
                    if is_text {
                        self.turn.content.push_str(t);
                        ev.push(ChatEvent::Text { delta: t.into() });
                    } else {
                        self.turn.reasoning.push_str(t);
                        ev.push(ChatEvent::Reasoning { delta: t.into() });
                    }
                }
            }
            for tc in d.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(self.calls.len() as u64);
                let e = self.calls.entry(idx).or_default();
                if let Some(id) = tc.get("id").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                    e.id = id.into();
                }
                if let Some(n) = tc.pointer("/function/name").and_then(Value::as_str) {
                    e.name.push_str(n);
                }
                if let Some(a) = tc.pointer("/function/arguments").and_then(Value::as_str) {
                    e.arguments.push_str(a);
                }
            }
            if let Some(r) = c.get("finish_reason").and_then(Value::as_str) {
                self.turn.finish_reason = Some(r.into());
            }
        }
        let usage = chunk.get("usage").filter(|u| u.is_object());
        let tps = chunk.pointer("/timings/predicted_per_second").and_then(Value::as_f64);
        if usage.is_some() || tps.is_some() {
            let n = |k: &str| usage.and_then(|u| u.get(k)).and_then(Value::as_u64).unwrap_or(0);
            ev.push(ChatEvent::Usage { prompt_tokens: n("prompt_tokens"), completion_tokens: n("completion_tokens"), tokens_per_sec: tps });
        }
        ev
    }

    pub fn finish(mut self) -> Turn {
        self.turn.tool_calls = self
            .calls
            .into_values()
            .filter(|c| !c.name.is_empty())
            .map(|mut c| {
                if c.id.is_empty() {
                    c.id = format!("call_{}", uuid::Uuid::new_v4().simple());
                }
                c
            })
            .collect();
        self.turn
    }
}

/// Stream one completion. Events are reported as they arrive.
pub async fn stream_turn(
    http: &reqwest::Client,
    ep: &Endpoint,
    messages: &[Value],
    tools: &[Value],
    cancel: &CancellationToken,
    on_event: &(dyn Fn(ChatEvent) + Send + Sync),
) -> Result<Turn> {
    if ep.kind == ApiKind::Anthropic {
        return stream_turn_anthropic(http, ep, messages, tools, cancel, on_event).await;
    }
    let mut body = json!({
        "model": if ep.model.is_empty() { "local" } else { ep.model.as_str() },
        "messages": openai_messages(messages),
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools.to_vec());
    }
    let mut req = http.post(ep.chat_url()).json(&body);
    if let Some(k) = ep.api_key.as_deref().filter(|k| !k.is_empty()) {
        req = req.bearer_auth(k);
    }
    let resp = send(req, ep, cancel).await?;
    let mut acc = StreamAccumulator::default();
    for_each_sse(resp, cancel, |data| {
        if data == "[DONE]" {
            return Ok(true);
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else { return Ok(false) };
        if let Some(err) = v.get("error") {
            bail!("server error: {}", err.get("message").and_then(Value::as_str).map(String::from).unwrap_or_else(|| err.to_string()));
        }
        for e in acc.feed(&v) {
            on_event(e);
        }
        Ok(false)
    })
    .await?;
    Ok(acc.finish())
}

async fn stream_turn_anthropic(
    http: &reqwest::Client,
    ep: &Endpoint,
    messages: &[Value],
    tools: &[Value],
    cancel: &CancellationToken,
    on_event: &(dyn Fn(ChatEvent) + Send + Sync),
) -> Result<Turn> {
    if ep.model.is_empty() {
        bail!("choose a model for this Anthropic endpoint (Settings → AI service providers)");
    }
    let mut req = http.post(crate::anthropic::messages_url(ep)).json(&crate::anthropic::request_body(ep, messages, tools));
    for (k, v) in crate::anthropic::headers(ep) {
        req = req.header(k, v);
    }
    let resp = send(req, ep, cancel).await?;
    let mut acc = crate::anthropic::Accumulator::default();
    for_each_sse(resp, cancel, |data| {
        let Ok(v) = serde_json::from_str::<Value>(data) else { return Ok(false) };
        match acc.feed(&v) {
            Ok(evs) => evs.into_iter().for_each(on_event),
            Err(msg) => bail!("server error: {msg}"),
        }
        Ok(v.get("type").and_then(Value::as_str) == Some("message_stop"))
    })
    .await?;
    Ok(acc.finish())
}

/// Run the conversation until the model stops calling tools, the round limit
/// is reached, or `cancel` fires. Assistant and tool messages are appended to
/// `messages`. `exec(name, args)` runs one tool; a refusal from the approval
/// gate is passed back to the model as the tool result.
#[allow(clippy::too_many_arguments)]
pub async fn run_with_tools<F, Fut>(
    http: &reqwest::Client,
    ep: &Endpoint,
    messages: &mut Vec<Value>,
    tools: &[Value],
    max_rounds: usize,
    cancel: &CancellationToken,
    on_event: &(dyn Fn(ChatEvent) + Send + Sync),
    exec: F,
) -> Result<()>
where
    F: Fn(String, Value) -> Fut,
    Fut: Future<Output = Result<String>>,
{
    for _ in 0..max_rounds.max(1) {
        let turn = match stream_turn(http, ep, messages, tools, cancel, on_event).await {
            Ok(t) => t,
            Err(e) if e.is::<crate::download::Cancelled>() => {
                on_event(ChatEvent::Done { reason: "cancelled".into() });
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        messages.push(turn.to_message());
        if turn.tool_calls.is_empty() {
            on_event(ChatEvent::Done { reason: turn.finish_reason.unwrap_or_else(|| "stop".into()) });
            return Ok(());
        }
        for c in &turn.tool_calls {
            on_event(ChatEvent::ToolCall { id: c.id.clone(), name: c.name.clone(), arguments: c.arguments.clone() });
            let args: Result<Value> = if c.arguments.trim().is_empty() { Ok(json!({})) } else { serde_json::from_str(&c.arguments).context("arguments are not valid JSON") };
            let (content, ok) = match args {
                Err(e) => (format!("error: {e:#}"), false),
                Ok(a) => {
                    let r = tokio::select! {
                        r = exec(c.name.clone(), a) => r,
                        _ = cancel.cancelled() => {
                            on_event(ChatEvent::Done { reason: "cancelled".into() });
                            return Ok(());
                        }
                    };
                    match r {
                        Ok(s) => (s, true),
                        Err(e) if e.is::<crate::tools::Denied>() => ("The user declined this action. Do not retry it; ask the user how to proceed instead.".into(), false),
                        Err(e) => (format!("error: {e:#}"), false),
                    }
                }
            };
            let content = crate::tools::truncate(&content, 60_000);
            on_event(ChatEvent::ToolResult { id: c.id.clone(), name: c.name.clone(), content: content.clone(), ok });
            messages.push(json!({"role": "tool", "tool_call_id": c.id, "content": content}));
        }
    }
    on_event(ChatEvent::Done { reason: "max_rounds".into() });
    Ok(())
}

/// Model names offered by an endpoint (`GET /models`).
pub async fn list_models(http: &reqwest::Client, ep: &Endpoint) -> Result<Vec<String>> {
    let url = format!("{}/models", ep.base_url.trim_end_matches('/'));
    if ep.kind == ApiKind::Anthropic {
        let mut req = http.get(url).query(&[("limit", "1000")]);
        for (k, v) in crate::anthropic::headers(ep) {
            req = req.header(k, v);
        }
        let v: Value = req.send().await?.error_for_status()?.json().await?;
        return Ok(crate::anthropic::model_ids(&v));
    }
    let mut req = http.get(url);
    if let Some(k) = ep.api_key.as_deref().filter(|k| !k.is_empty()) {
        req = req.bearer_auth(k);
    }
    let v: Value = req.send().await?.error_for_status()?.json().await?;
    Ok(v.get("data").and_then(Value::as_array).into_iter().flatten().filter_map(|m| m.get("id").and_then(Value::as_str).map(String::from)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn sse(chunks: &[Value]) -> String {
        let mut s: String = chunks.iter().map(|c| format!("data: {c}\n\n")).collect();
        s.push_str("data: [DONE]\n\n");
        s
    }

    #[test]
    fn accumulates_text_reasoning_and_tool_calls() {
        let mut a = StreamAccumulator::default();
        let mut ev = vec![];
        for c in [
            json!({"choices": [{"delta": {"reasoning_content": "think"}}]}),
            json!({"choices": [{"delta": {"content": "Hi"}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "calc", "arguments": "{\"ex"}}]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "pression\":\"1+1\"}"}}]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"name": "now", "arguments": ""}}]}, "finish_reason": "tool_calls"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7}, "timings": {"predicted_per_second": 42.5}}),
        ] {
            ev.extend(a.feed(&c));
        }
        let t = a.finish();
        assert_eq!(t.content, "Hi");
        assert_eq!(t.reasoning, "think");
        assert_eq!(t.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(t.tool_calls.len(), 2);
        assert_eq!(t.tool_calls[0], ToolCall { id: "c1".into(), name: "calc".into(), arguments: "{\"expression\":\"1+1\"}".into() });
        assert!(t.tool_calls[1].id.starts_with("call_"));
        assert!(ev.contains(&ChatEvent::Usage { prompt_tokens: 5, completion_tokens: 7, tokens_per_sec: Some(42.5) }));
        assert_eq!(sse_data("data: {}\r"), Some("{}"));
        assert_eq!(sse_data(": keep-alive"), None);
    }

    /// Minimal HTTP server answering each POST with the next canned SSE body.
    async fn mock(bodies: Vec<String>, seen: Arc<Mutex<Vec<Value>>>) -> String {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move {
            for body in bodies {
                let (mut s, _) = l.accept().await.unwrap();
                let mut req = vec![];
                let mut tmp = [0u8; 8192];
                loop {
                    let n = s.read(&mut tmp).await.unwrap();
                    req.extend_from_slice(&tmp[..n]);
                    let text = String::from_utf8_lossy(&req).to_string();
                    if let Some(h) = text.find("\r\n\r\n") {
                        let len: usize = text[..h].lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap())).unwrap_or(0);
                        if req.len() >= h + 4 + len {
                            seen.lock().unwrap().push(serde_json::from_slice(&req[h + 4..h + 4 + len]).unwrap());
                            break;
                        }
                    }
                }
                let head = format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", body.len());
                s.write_all(head.as_bytes()).await.unwrap();
                s.write_all(body.as_bytes()).await.unwrap();
            }
        });
        format!("http://{addr}/v1")
    }

    #[tokio::test]
    async fn tool_loop_runs_tools_and_feeds_results_back() {
        let seen = Arc::new(Mutex::new(vec![]));
        let base = mock(
            vec![
                sse(&[json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "calculator", "arguments": "{\"expression\":\"6*7\"}"}}]}, "finish_reason": "tool_calls"}]})]),
                sse(&[json!({"choices": [{"delta": {"content": "It is 42."}, "finish_reason": "stop"}]})]),
            ],
            seen.clone(),
        )
        .await;
        let ep = Endpoint { base_url: base, ..Default::default() };
        let mut msgs = vec![json!({"role": "user", "content": "6*7?"})];
        let events = Arc::new(Mutex::new(vec![]));
        let ev2 = events.clone();
        run_with_tools(&crate::http_client(), &ep, &mut msgs, &crate::tools::ChatToolbox::schemas(), 5, &CancellationToken::new(), &move |e| ev2.lock().unwrap().push(e), |name, args| async move {
            assert_eq!(name, "calculator");
            crate::tools::calculator(args["expression"].as_str().unwrap())
        })
        .await
        .unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[2], json!({"role": "tool", "tool_call_id": "a", "content": "42"}));
        assert_eq!(msgs[3]["content"], "It is 42.");
        let ev = events.lock().unwrap();
        assert!(ev.contains(&ChatEvent::ToolResult { id: "a".into(), name: "calculator".into(), content: "42".into(), ok: true }));
        assert_eq!(ev.last(), Some(&ChatEvent::Done { reason: "stop".into() }));
        // The second request carried the tool result back to the model.
        let s = seen.lock().unwrap();
        assert_eq!(s[1]["messages"][2]["role"], "tool");
        assert!(s[0]["tools"].as_array().unwrap().len() >= 4);
    }

    #[tokio::test]
    async fn anthropic_tool_loop_round_trip() {
        fn ev(v: Value) -> String {
            format!("event: {}\ndata: {v}\n\n", v["type"].as_str().unwrap())
        }
        let first = [
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 10}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "calculator", "input": {}}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"expression\":\"6*7\"}"}}),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
            json!({"type": "message_stop"}),
        ];
        let second = [
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "42."}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}),
            json!({"type": "message_stop"}),
        ];
        let seen = Arc::new(Mutex::new(vec![]));
        let base = mock(vec![first.into_iter().map(ev).collect(), second.into_iter().map(ev).collect()], seen.clone()).await;
        let ep = Endpoint { base_url: base, api_key: Some("k".into()), model: "claude-opus-5-5".into(), kind: ApiKind::Anthropic };
        let mut msgs = vec![json!({"role": "system", "content": "sys"}), json!({"role": "user", "content": "6*7?"})];
        run_with_tools(&crate::http_client(), &ep, &mut msgs, &crate::tools::ChatToolbox::schemas(), 5, &CancellationToken::new(), &|_| {}, |_, args| async move {
            crate::tools::calculator(args["expression"].as_str().unwrap())
        })
        .await
        .unwrap();
        assert_eq!(msgs.last().unwrap()["content"], "42.");
        let s = seen.lock().unwrap();
        assert_eq!(s[0]["system"], "sys");
        // Second request: thinking block (with signature) + tool_use replayed, then the tool result.
        assert_eq!(s[1]["messages"][1]["content"][0]["signature"], "sig");
        assert_eq!(s[1]["messages"][1]["content"][1]["id"], "toolu_9");
        assert_eq!(s[1]["messages"][2]["content"][0], json!({"type": "tool_result", "tool_use_id": "toolu_9", "content": "42"}));
    }

    #[test]
    fn private_keys_never_reach_openai_servers() {
        let m = openai_messages(&[json!({"role": "assistant", "content": "x", "_anthropic": []})]);
        assert_eq!(m[0], json!({"role": "assistant", "content": "x"}));
    }

    #[tokio::test]
    async fn denied_tool_is_reported_to_the_model() {
        let seen = Arc::new(Mutex::new(vec![]));
        let base = mock(
            vec![
                sse(&[json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "x", "function": {"name": "run_command", "arguments": "{\"command\":\"rm -rf /\"}"}}]}}]})]),
                sse(&[json!({"choices": [{"delta": {"content": "OK, I won't."}}]})]),
            ],
            seen,
        )
        .await;
        let ep = Endpoint { base_url: base, ..Default::default() };
        let mut msgs = vec![json!({"role": "user", "content": "clean up"})];
        run_with_tools(&crate::http_client(), &ep, &mut msgs, &[], 5, &CancellationToken::new(), &|_| {}, |_, _| async { Err(crate::tools::Denied("Run: rm -rf /".into()).into()) }).await.unwrap();
        assert!(msgs[2]["content"].as_str().unwrap().contains("declined"));
    }
}
