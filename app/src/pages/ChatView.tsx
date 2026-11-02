import { useEffect, useRef, useState } from "react";
import { api, ApprovalRequest, ChatEvent, Endpoint, errText, Message, on } from "../api";
import { useApp } from "../App";
import { useI18n } from "../i18n";

type Item =
  | { kind: "user"; text: string; images: string[] }
  | { kind: "assistant"; text: string; reasoning: string }
  | { kind: "tool"; id: string; name: string; args: string; result?: string; ok?: boolean }
  | { kind: "notice"; text: string; bad?: boolean };

export type PlanStep = { step: string; status: string };

interface Props {
  /** "chat" uses the chat toolbox; "agent" the workspace toolbox. */
  mode: "chat" | "agent";
  workspace?: string;
  onPlan?: (steps: PlanStep[]) => void;
  toolbar?: React.ReactNode;
}

const newId = () => crypto.randomUUID();

export default function ChatView({ mode, workspace, onPlan, toolbar }: Props) {
  const { t } = useI18n();
  const { settings, llm } = useApp();
  const [session, setSession] = useState(newId);
  const [items, setItems] = useState<Item[]>([]);
  const [history, setHistory] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [tools, setTools] = useState(true);
  const [source, setSource] = useState<string>("local");
  const [approvals, setApprovals] = useState<ApprovalRequest[]>([]);
  const [stats, setStats] = useState<string>("");
  const scroller = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const subs = [
      on<{ session: string; event: ChatEvent }>("chat", ({ session: s, event }) => {
        if (s !== session) return;
        if (event.type === "plan") return onPlan?.(event.steps);
        if (event.type === "usage") {
          if (event.tokens_per_sec) setStats(`${event.completion_tokens} tok · ${event.tokens_per_sec.toFixed(1)} tok/s`);
          return;
        }
        setItems((xs) => apply(xs, event, t));
      }),
      on<{ session: string; request: ApprovalRequest }>("approval", ({ session: s, request }) => {
        if (s === session) setApprovals((a) => [...a, request]);
      }),
    ];
    return () => subs.forEach((p) => p.then((u) => u()));
  }, [session]);

  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [items, approvals]);

  const endpoint = (): Endpoint | null => {
    if (source === "local") return null;
    const e = settings.endpoints.find((x) => x.name === source);
    return e ? { base_url: e.base_url, api_key: e.api_key, model: e.model, kind: e.kind ?? "openai" } : null;
  };

  const send = async () => {
    const text = input.trim();
    if ((!text && images.length === 0) || busy) return;
    const content: unknown = images.length
      ? [{ type: "text", text }, ...images.map((url) => ({ type: "image_url", image_url: { url } }))]
      : text;
    const msgs: Message[] = [...history, { role: "user", content }];
    setItems((xs) => [...xs, { kind: "user", text, images }]);
    setInput("");
    setImages([]);
    setBusy(true);
    setStats("");
    try {
      const out = mode === "agent"
        ? await api.agentSend(session, endpoint(), msgs, workspace ?? "")
        : await api.chatSend(session, endpoint(), msgs, tools);
      setHistory(out);
    } catch (e) {
      setItems((xs) => [...xs, { kind: "notice", text: errText(e), bad: true }]);
      setHistory(msgs);
    } finally {
      setBusy(false);
      setApprovals([]);
    }
  };

  const decide = (r: ApprovalRequest, d: "once" | "always" | "deny") => {
    api.approve(session, r.id, d);
    setApprovals((a) => a.filter((x) => x.id !== r.id));
  };

  const reset = () => {
    if (busy) api.chatStop(session);
    setSession(newId());
    setItems([]);
    setHistory([]);
    setApprovals([]);
    setStats("");
    onPlan?.([]);
  };

  const attach = (files: FileList | null) => {
    for (const f of Array.from(files ?? [])) {
      const r = new FileReader();
      r.onload = () => setImages((xs) => [...xs, String(r.result)]);
      r.readAsDataURL(f);
    }
  };

  const localReady = llm?.state === "ready";
  const blocked = source === "local" && !localReady;

  return (
    <div className="chat">
      <div className="row">
        <label className="row small">
          {t("chat.source")}
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="local">{t("chat.local")}</option>
            {settings.endpoints.map((e) => <option key={e.name} value={e.name}>{e.name} · {e.model}</option>)}
          </select>
        </label>
        {mode === "chat" && (
          <label className="check small"><input type="checkbox" checked={tools} onChange={(e) => setTools(e.target.checked)} />{t("chat.tools")}</label>
        )}
        {toolbar}
        <span className="muted small right">{stats}</span>
        <button className="btn sm" onClick={reset}>{t("chat.new")}</button>
      </div>
      {blocked && <div className="note warn">{t("chat.no_local")}</div>}

      <div className="msgs" ref={scroller}>
        {items.map((it, i) => <ItemView key={i} it={it} />)}
        {approvals.map((r) => (
          <div key={r.id} className="approval">
            <b>{t("approval.title")} · {r.tool}</b>
            <div className="mono">{r.summary}</div>
            <div className="row">
              <button className="btn primary" onClick={() => decide(r, "once")}>{t("approval.once")}</button>
              <button className="btn" onClick={() => decide(r, "always")}>{t("approval.always")}</button>
              <button className="btn danger" onClick={() => decide(r, "deny")}>{t("approval.deny")}</button>
            </div>
          </div>
        ))}
      </div>

      {images.length > 0 && (
        <div className="row">
          {images.map((src, i) => (
            <img key={i} src={src} style={{ height: 48, borderRadius: 4, cursor: "pointer" }} title="×" onClick={() => setImages((xs) => xs.filter((_, j) => j !== i))} />
          ))}
        </div>
      )}
      <div className="composer">
        <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={(e) => { attach(e.target.files); e.target.value = ""; }} />
        <button className="btn" title={t("chat.attach")} onClick={() => fileRef.current?.click()}>＋</button>
        <textarea
          rows={2}
          placeholder={t("chat.placeholder")}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send();
            }
          }}
        />
        {busy ? (
          <button className="btn" onClick={() => api.chatStop(session)}>{t("common.stop")}</button>
        ) : (
          <button className="btn primary" disabled={blocked || (mode === "agent" && !workspace)} onClick={send}>{t("common.send")}</button>
        )}
      </div>
    </div>
  );
}

function apply(xs: Item[], e: ChatEvent, t: (k: "chat.max_rounds" | "chat.cancelled") => string): Item[] {
  const last = xs[xs.length - 1];
  switch (e.type) {
    case "text":
    case "reasoning": {
      const cur = last?.kind === "assistant" ? last : { kind: "assistant" as const, text: "", reasoning: "" };
      const next = e.type === "text" ? { ...cur, text: cur.text + e.delta } : { ...cur, reasoning: cur.reasoning + e.delta };
      return last?.kind === "assistant" ? [...xs.slice(0, -1), next] : [...xs, next];
    }
    case "tool_call":
      return [...xs, { kind: "tool", id: e.id, name: e.name, args: e.arguments }];
    case "tool_result":
      return xs.map((x) => (x.kind === "tool" && x.id === e.id ? { ...x, result: e.content, ok: e.ok } : x));
    case "done":
      if (e.reason === "max_rounds") return [...xs, { kind: "notice", text: t("chat.max_rounds") }];
      if (e.reason === "cancelled") return [...xs, { kind: "notice", text: t("chat.cancelled") }];
      return xs;
    default:
      return xs;
  }
}

function ItemView({ it }: { it: Item }) {
  const { t } = useI18n();
  switch (it.kind) {
    case "user":
      return (
        <div className="msg user">
          {it.text}
          {it.images.map((s, i) => <img key={i} className="att" src={s} />)}
        </div>
      );
    case "assistant":
      return (
        <div className="msg assistant">
          {it.reasoning && (
            <details className="think">
              <summary>{t("chat.thinking")}</summary>
              <div style={{ whiteSpace: "pre-wrap" }}>{it.reasoning}</div>
            </details>
          )}
          {it.text}
        </div>
      );
    case "tool":
      return (
        <details className="tool msg">
          <summary>
            {t("chat.tool")}: <b>{it.name}</b> {it.result === undefined ? "…" : it.ok ? "✓" : "✗"}
          </summary>
          <pre>{prettyArgs(it.args)}</pre>
          {it.result !== undefined && <pre>{it.result}</pre>}
        </details>
      );
    case "notice":
      return <div className={"note" + (it.bad ? " bad" : "")}>{it.text}</div>;
  }
}

function prettyArgs(a: string): string {
  try {
    return JSON.stringify(JSON.parse(a), null, 2);
  } catch {
    return a;
  }
}
