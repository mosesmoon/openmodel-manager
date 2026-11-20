import { useEffect, useRef, useState } from "react";
import { api, ApprovalRequest, ChatEvent, ChatMeta, Endpoint, errText, Message, on } from "../api";
import { useApp } from "../App";
import { useI18n } from "../i18n";
import { Markdown } from "../markdown";

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
  const { t, lang } = useI18n();
  const { settings, llm, fail } = useApp();
  // `session` keys streaming events and tool approvals; `convId` names the saved record.
  // Loading a record keeps neither, so approvals never carry over between conversations.
  const [session, setSession] = useState(newId);
  const [convId, setConvId] = useState<string>(newId);
  const [list, setList] = useState<ChatMeta[]>([]);
  const [listOpen, setListOpen] = useState(false);
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
  const convRef = useRef(convId);
  convRef.current = convId;
  const saveDue = useRef(false);
  const saving = settings.save_chats;

  const refresh = () => api.chatList(mode).then(setList, fail);
  useEffect(() => { if (saving) refresh(); }, [saving]);

  // Save only once the turn is over, from the committed state, never mid-stream.
  useEffect(() => {
    if (busy || !saveDue.current) return;
    saveDue.current = false;
    if (!saving || items.length === 0) return;
    api.chatSave({
      v: 1, id: convId, mode, title: "", created: 0, updated: 0, source,
      workspace: mode === "agent" ? workspace : undefined, history, items,
    }).then(refresh, fail);
  }, [busy]);

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
    const cid = convId;
    setItems((xs) => [...xs, { kind: "user", text, images }]);
    setInput("");
    setImages([]);
    setBusy(true);
    setStats("");
    try {
      const out = mode === "agent"
        ? await api.agentSend(session, endpoint(), msgs, workspace ?? "")
        : await api.chatSend(session, endpoint(), msgs, tools);
      if (convRef.current === cid) setHistory(out);
    } catch (e) {
      if (convRef.current === cid) {
        setItems((xs) => [...xs, { kind: "notice", text: errText(e), bad: true }]);
        setHistory(msgs);
      }
    } finally {
      // A conversation reset or deleted mid-reply must not be saved back.
      saveDue.current = convRef.current === cid;
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
    setConvId(newId());
    setItems([]);
    setHistory([]);
    setApprovals([]);
    setStats("");
    onPlan?.([]);
  };

  const open = async (id: string) => {
    if (busy || id === convId) return;
    let rec;
    try {
      rec = await api.chatLoad(id);
    } catch (e) {
      return fail(e);
    }
    const known = rec.source === "local" || settings.endpoints.some((x) => x.name === rec.source);
    const notes: Item[] = [];
    if (!known) notes.push({ kind: "notice", text: `${t("chat.source_missing")} ${rec.source}` });
    if (mode === "agent" && rec.workspace && rec.workspace !== workspace) notes.push({ kind: "notice", text: `${t("chat.workspace_other")} ${rec.workspace}` });
    setSession(newId());
    setConvId(rec.id);
    setItems([...(rec.items as Item[]), ...notes]);
    setHistory(rec.history as Message[]);
    setSource(known ? rec.source : "local");
    setApprovals([]);
    setStats("");
    onPlan?.([]);
  };

  const remove = async (m: ChatMeta) => {
    if (!confirm(t("chat.delete_confirm"))) return;
    try {
      await api.chatDelete(m.id);
    } catch (e) {
      return fail(e);
    }
    if (m.id === convId) reset();
    refresh();
  };

  const when = (secs: number) => {
    const d = new Date(secs * 1000);
    return d.toDateString() === new Date().toDateString()
      ? d.toLocaleTimeString(lang, { hour: "2-digit", minute: "2-digit" })
      : d.toLocaleDateString(lang, { month: "numeric", day: "numeric" });
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
    <div className="chatwrap">
      {saving && (
        <aside className={"chatlist" + (listOpen ? " open" : "")}>
          <div className="chatlist-head">
            <button className="chatlist-toggle" aria-expanded={listOpen} onClick={() => setListOpen(!listOpen)}>
              <span className="caret">{listOpen ? "▾" : "▸"}</span> {t("chat.history")} ({list.length})
            </button>
            <b className="chatlist-title">{t("chat.history")}</b>
            <button className="btn sm" onClick={reset}>＋ {t("chat.new")}</button>
          </div>
          <div className="chatlist-body">
            {list.length === 0 && <div className="muted small" style={{ padding: "4px 6px" }}>{t("chat.history_empty")}</div>}
            {list.map((m) => (
              <div key={m.id} className={"chatitem" + (m.id === convId ? " on" : "")}>
                <button className="chatitem-main" disabled={busy} title={m.title} onClick={() => { open(m.id); setListOpen(false); }}>
                  <span className="ellipsis">{m.title || "—"}</span>
                  <span className="muted small">{when(m.updated)}</span>
                </button>
                <button className="chatitem-del" title={t("common.remove")} aria-label={t("common.remove")} onClick={() => remove(m)}>×</button>
              </div>
            ))}
          </div>
        </aside>
      )}
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
          {it.text && <Markdown text={it.text} />}
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
