//! Anthropic Messages API transport (raw HTTP; Rust has no official SDK).
//!
//! The chat loop keeps its history in OpenAI's message shape. This module
//! converts that history to Messages API requests and turns the Messages SSE
//! stream back into the same `ChatEvent`s and `Turn`. Each assistant turn's raw
//! content blocks (thinking signatures, tool_use ids, fallback markers) are kept
//! on the message under `_anthropic` and sent back unchanged on the next
//! request, as the API expects.

use crate::llm::{ChatEvent, Endpoint, ToolCall, Turn};
use anyhow::Result;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

pub const API_VERSION: &str = "2026-10-21";
/// Key on an assistant message that carries its raw Messages API content.
pub const NATIVE_KEY: &str = "_anthropic";
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

pub fn messages_url(ep: &Endpoint) -> String {
    format!("{}/messages", ep.base_url.trim_end_matches('/'))
}

/// Models with adaptive thinking and 128K output (Claude 4.6 and later).
fn is_current(model: &str) -> bool {
    regex::Regex::new(r"^claude-(fable|mythos|opus|sonnet|haiku)-(5|4-[6-9])").unwrap().is_match(model)
}

/// Models that take the server-side refusal fallback (`fallbacks: "default"`).
fn takes_fallback(model: &str) -> bool {
    ["claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"].contains(&model)
}

pub fn headers(ep: &Endpoint) -> Vec<(&'static str, String)> {
    let mut h = vec![("anthropic-version", API_VERSION.to_string())];
    if let Some(k) = ep.api_key.as_deref().filter(|k| !k.is_empty()) {
        h.push(("x-api-key", k.to_string()));
    }
    if takes_fallback(&ep.model) {
        h.push(("anthropic-beta", FALLBACK_BETA.to_string()));
    }
    h
}

/// An OpenAI `image_url` part as a Messages image block.
fn image_block(url: &str) -> Value {
    if let Some(rest) = url.strip_prefix("data:") {
        if let Some((meta, data)) = rest.split_once(',') {
            let media = meta.trim_end_matches(";base64");
            return json!({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}});
        }
    }
    json!({"type": "image", "source": {"type": "url", "url": url}})
}

fn user_blocks(content: &Value) -> Vec<Value> {
    match content {
        Value::String(s) if !s.is_empty() => vec![json!({"type": "text", "text": s})],
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                Some("text") => p.get("text").and_then(Value::as_str).filter(|t| !t.is_empty()).map(|t| json!({"type": "text", "text": t})),
                Some("image_url") => p.pointer("/image_url/url").and_then(Value::as_str).map(image_block),
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

fn assistant_blocks(m: &Value) -> Vec<Value> {
    if let Some(Value::Array(native)) = m.get(NATIVE_KEY) {
        return native.clone();
    }
    let mut b = vec![];
    if let Some(t) = m.get("content").and_then(Value::as_str).filter(|t| !t.is_empty()) {
        b.push(json!({"type": "text", "text": t}));
    }
    for c in m.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
        let args = c.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("{}");
        let input: Value = serde_json::from_str(args).ok().filter(Value::is_object).unwrap_or_else(|| json!({}));
        b.push(json!({"type": "tool_use", "id": c["id"], "name": c.pointer("/function/name").cloned().unwrap_or(Value::Null), "input": input}));
    }
    b
}

/// Split OpenAI-shaped history into the Messages `system` text and turns.
/// Consecutive turns of the same role are merged, so the tool results of one
/// assistant turn land in a single user message.
pub fn convert_messages(msgs: &[Value]) -> (String, Vec<Value>) {
    let mut system = String::new();
    let mut out: Vec<Value> = vec![];
    for m in msgs {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let (role, blocks) = match role {
            "system" | "developer" => {
                if let Some(t) = m.get("content").and_then(Value::as_str) {
                    if !system.is_empty() {
                        system.push_str("\n\n");
                    }
                    system.push_str(t);
                }
                continue;
            }
            "assistant" => ("assistant", assistant_blocks(m)),
            "tool" => {
                let content = m.get("content").and_then(Value::as_str).unwrap_or_default();
                ("user", vec![json!({"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": content})])
            }
            _ => ("user", user_blocks(m.get("content").unwrap_or(&Value::Null))),
        };
        if blocks.is_empty() {
            continue;
        }
        match out.last_mut() {
            Some(last) if last["role"] == role => last["content"].as_array_mut().unwrap().extend(blocks),
            _ => out.push(json!({"role": role, "content": blocks})),
        }
    }
    (system, out)
}

/// OpenAI function tools as Messages tools. Inputs stream as they are
/// generated; the loop validates them (invalid JSON goes back as an error).
pub fn convert_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .filter_map(|t| {
            let f = t.get("function")?;
            Some(json!({
                "name": f.get("name")?,
                "description": f.get("description").cloned().unwrap_or(Value::String(String::new())),
                "input_schema": f.get("parameters").cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                "eager_input_streaming": true,
            }))
        })
        .collect()
}

pub fn request_body(ep: &Endpoint, messages: &[Value], tools: &[Value]) -> Value {
    let (system, msgs) = convert_messages(messages);
    let current = is_current(&ep.model);
    let mut body = json!({
        "model": ep.model,
        "max_tokens": if current { 64000 } else { 8192 },
        "stream": true,
        "messages": msgs,
    });
    if !system.is_empty() {
        body["system"] = Value::String(system);
    }
    if !tools.is_empty() {
        body["tools"] = Value::Array(convert_tools(tools));
    }
    if current {
        // Show a readable summary of the reasoning instead of a silent pause.
        body["thinking"] = json!({"type": "adaptive", "display": "summarized"});
    }
    if takes_fallback(&ep.model) {
        body["fallbacks"] = Value::String("default".into());
    }
    body
}

/// Assembles one assistant turn from Messages stream events.
#[derive(Default)]
pub struct Accumulator {
    blocks: Vec<Value>,
    tool_json: BTreeMap<usize, String>,
    calls: Vec<ToolCall>,
    turn: Turn,
    input_tokens: u64,
}

impl Accumulator {
    fn block(&mut self, i: usize) -> &mut Value {
        if self.blocks.len() <= i {
            self.blocks.resize(i + 1, Value::Null);
        }
        &mut self.blocks[i]
    }

    fn append(&mut self, i: usize, key: &str, s: &str) {
        let b = self.block(i);
        if !b.is_object() {
            *b = Value::Object(Map::new());
        }
        let cur = b.get(key).and_then(Value::as_str).unwrap_or_default().to_string();
        b[key] = Value::String(cur + s);
    }

    /// Feed one SSE `data:` payload. Returns events and, for an `error`
    /// event, the error message.
    pub fn feed(&mut self, v: &Value) -> Result<Vec<ChatEvent>, String> {
        let mut ev = vec![];
        let idx = v.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        match v.get("type").and_then(Value::as_str).unwrap_or_default() {
            "message_start" => self.input_tokens = v.pointer("/message/usage/input_tokens").and_then(Value::as_u64).unwrap_or(0),
            "content_block_start" => {
                let cb = v.get("content_block").cloned().unwrap_or(Value::Null);
                if cb.get("type").and_then(Value::as_str) == Some("tool_use") {
                    self.tool_json.insert(idx, String::new());
                }
                *self.block(idx) = cb;
            }
            "content_block_delta" => {
                let d = &v["delta"];
                let s = |k: &str| d.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
                match d.get("type").and_then(Value::as_str).unwrap_or_default() {
                    "text_delta" => {
                        let t = s("text");
                        self.append(idx, "text", &t);
                        self.turn.content.push_str(&t);
                        ev.push(ChatEvent::Text { delta: t });
                    }
                    "thinking_delta" => {
                        let t = s("thinking");
                        self.append(idx, "thinking", &t);
                        self.turn.reasoning.push_str(&t);
                        ev.push(ChatEvent::Reasoning { delta: t });
                    }
                    "signature_delta" => self.append(idx, "signature", &s("signature")),
                    "input_json_delta" => self.tool_json.entry(idx).or_default().push_str(&s("partial_json")),
                    _ => {}
                }
            }
            "content_block_stop" => {
                if let Some(raw) = self.tool_json.remove(&idx) {
                    let input: Option<Value> = if raw.trim().is_empty() { Some(json!({})) } else { serde_json::from_str(&raw).ok().filter(Value::is_object) };
                    let b = self.block(idx);
                    // History must hold a valid object; the loop reports the bad JSON to the model.
                    b["input"] = input.clone().unwrap_or_else(|| json!({}));
                    let call = ToolCall {
                        id: b.get("id").and_then(Value::as_str).unwrap_or_default().into(),
                        name: b.get("name").and_then(Value::as_str).unwrap_or_default().into(),
                        arguments: if input.is_some() { if raw.trim().is_empty() { "{}".into() } else { raw } } else { raw },
                    };
                    self.calls.push(call);
                }
            }
            "message_delta" => {
                if let Some(r) = v.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    self.turn.finish_reason = Some(
                        match r {
                            "end_turn" | "stop_sequence" => "stop",
                            "max_tokens" => "length",
                            "tool_use" => "tool_calls",
                            other => other,
                        }
                        .into(),
                    );
                    if r == "refusal" {
                        let note = "\n\n[The model declined this request.]";
                        self.turn.content.push_str(note);
                        ev.push(ChatEvent::Text { delta: note.into() });
                    }
                }
                if let Some(out) = v.pointer("/usage/output_tokens").and_then(Value::as_u64) {
                    ev.push(ChatEvent::Usage { prompt_tokens: self.input_tokens, completion_tokens: out, tokens_per_sec: None });
                }
            }
            "error" => return Err(v.pointer("/error/message").and_then(Value::as_str).map(String::from).unwrap_or_else(|| v.to_string())),
            _ => {}
        }
        Ok(ev)
    }

    pub fn finish(mut self) -> Turn {
        self.turn.tool_calls = self.calls;
        self.turn.native = Some(Value::Array(self.blocks.into_iter().filter(|b| !b.is_null()).collect()));
        self.turn
    }
}

/// Model ids from `GET /v1/models`.
pub fn model_ids(v: &Value) -> Vec<String> {
    v.get("data").and_then(Value::as_array).into_iter().flatten().filter_map(|m| m.get("id").and_then(Value::as_str).map(String::from)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(model: &str) -> Endpoint {
        Endpoint { base_url: "https://api.anthropic.com/v1".into(), api_key: Some("k".into()), model: model.into(), kind: crate::llm::ApiKind::Anthropic }
    }

    #[test]
    fn converts_history() {
        let msgs = vec![
            json!({"role": "system", "content": "Be brief."}),
            json!({"role": "user", "content": [{"type": "text", "text": "What is this?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}),
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "calculator", "arguments": "{\"expression\":\"1+1\"}"}},
                {"id": "t2", "type": "function", "function": {"name": "now", "arguments": ""}}]}),
            json!({"role": "tool", "tool_call_id": "t1", "content": "2"}),
            json!({"role": "tool", "tool_call_id": "t2", "content": "noon"}),
            json!({"role": "assistant", "content": "2 at noon", NATIVE_KEY: [{"type": "thinking", "thinking": "x", "signature": "s"}, {"type": "text", "text": "2 at noon"}]}),
        ];
        let (system, out) = convert_messages(&msgs);
        assert_eq!(system, "Be brief.");
        assert_eq!(out.len(), 4);
        assert_eq!(out[0]["content"][1], json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}));
        assert_eq!(out[1]["content"][0]["input"], json!({"expression": "1+1"}));
        assert_eq!(out[1]["content"][1]["input"], json!({}));
        // Both tool results in one user message.
        assert_eq!(out[2]["content"].as_array().unwrap().len(), 2);
        assert_eq!(out[2]["content"][1], json!({"type": "tool_result", "tool_use_id": "t2", "content": "noon"}));
        // Raw blocks, signature included, go back unchanged.
        assert_eq!(out[3]["content"][0]["signature"], "s");
    }

    #[test]
    fn request_options_follow_the_model() {
        let tools = vec![json!({"type": "function", "function": {"name": "calc", "description": "d", "parameters": {"type": "object"}}})];
        let b = request_body(&ep("claude-opus-5-5"), &[json!({"role": "user", "content": "hi"})], &tools);
        assert_eq!(b["thinking"]["type"], "adaptive");
        assert_eq!(b["fallbacks"], "default");
        assert_eq!(b["max_tokens"], 64000);
        assert_eq!(b["tools"][0]["input_schema"], json!({"type": "object"}));
        assert_eq!(b["tools"][0]["eager_input_streaming"], true);
        assert!(b.get("temperature").is_none());
        assert!(headers(&ep("claude-opus-5-5")).iter().any(|(k, v)| *k == "anthropic-beta" && v == FALLBACK_BETA));
        let old = request_body(&ep("claude-haiku-4-5"), &[json!({"role": "user", "content": "hi"})], &[]);
        assert!(old.get("thinking").is_none() && old.get("fallbacks").is_none());
        assert!(!headers(&ep("claude-haiku-4-5")).iter().any(|(k, _)| *k == "anthropic-beta"));
    }

    #[test]
    fn accumulates_stream() {
        let mut a = Accumulator::default();
        let mut ev = vec![];
        for v in [
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 12}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Need calc."}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Let me check."}}),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "calculator", "input": {}}}),
            json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"expression\":"}}),
            json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "\"6*7\"}"}}),
            json!({"type": "content_block_stop", "index": 2}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 30}}),
            json!({"type": "message_stop"}),
        ] {
            ev.extend(a.feed(&v).unwrap());
        }
        let t = a.finish();
        assert_eq!(t.content, "Let me check.");
        assert_eq!(t.reasoning, "Need calc.");
        assert_eq!(t.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(t.tool_calls, vec![ToolCall { id: "toolu_1".into(), name: "calculator".into(), arguments: "{\"expression\":\"6*7\"}".into() }]);
        let native = t.native.clone().unwrap();
        assert_eq!(native[0]["signature"], "sig");
        assert_eq!(native[2]["input"], json!({"expression": "6*7"}));
        assert!(ev.contains(&ChatEvent::Usage { prompt_tokens: 12, completion_tokens: 30, tokens_per_sec: None }));
        // The assistant message carries the raw blocks for the next request.
        assert_eq!(t.to_message()[NATIVE_KEY], native);
        assert_eq!(Accumulator::default().feed(&json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})).unwrap_err(), "Overloaded");
    }
}
