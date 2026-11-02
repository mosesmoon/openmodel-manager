// The UI runs in two places: the desktop window (Tauri) and the web
// management page in a browser. This file hides the difference: commands go
// through Tauri's invoke or `POST /api/invoke/<command>`, events through
// Tauri's listen or one `/api/events` stream.
import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { open as tauriOpen, OpenDialogOptions } from "@tauri-apps/plugin-dialog";
import { openUrl as tauriOpenUrl, revealItemInDir } from "@tauri-apps/plugin-opener";

export const inBrowser = !("__TAURI_INTERNALS__" in window);

/** Thrown when the web page's session has ended; the app shows the sign-in page. */
export class SignedOut extends Error {}

export async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (!inBrowser) return tauriInvoke<T>(cmd, args);
  const r = await fetch(`/api/invoke/${cmd}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(args ?? {}) });
  if (r.status === 401) {
    window.dispatchEvent(new Event("openmodel-signed-out"));
    throw new SignedOut("signed out");
  }
  const body = await r.json().catch(() => null);
  if (!r.ok) throw body?.error ?? `HTTP ${r.status}`;
  return body as T;
}

let stream: EventSource | null = null;

export function on<T>(event: string, cb: (payload: T) => void): Promise<UnlistenFn> {
  if (!inBrowser) return listen<T>(event, (e) => cb(e.payload));
  stream ??= new EventSource("/api/events");
  const handler = (e: MessageEvent) => cb(JSON.parse(e.data) as T);
  stream.addEventListener(event, handler);
  return Promise.resolve(() => stream?.removeEventListener(event, handler));
}

export async function session(): Promise<boolean> {
  return !inBrowser || (await fetch("/api/session")).ok;
}

export async function signIn(password: string): Promise<void> {
  const r = await fetch("/api/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ password }) });
  if (!r.ok) throw (await r.json().catch(() => null))?.error ?? `HTTP ${r.status}`;
  stream?.close();
  stream = null;
}

export async function signOut(): Promise<void> {
  await fetch("/api/logout", { method: "POST" });
  location.reload();
}

/**
 * A file or folder on the computer running OpenModel. The window shows the
 * system picker; a browser can't browse that computer, so it asks for a path.
 */
export async function pickPath(opts: OpenDialogOptions, ask: string): Promise<string | null> {
  if (!inBrowser) {
    const p = await tauriOpen({ multiple: false, ...opts });
    return typeof p === "string" ? p : null;
  }
  const p = window.prompt(ask, typeof opts.defaultPath === "string" ? opts.defaultPath : "");
  return p?.trim() || null;
}

/** Show a file in the file manager; only possible in the window. */
export function reveal(path: string): Promise<void> {
  return inBrowser ? Promise.resolve() : revealItemInDir(path);
}

export function openExternal(url: string): Promise<void> {
  if (inBrowser) {
    window.open(url, "_blank", "noopener");
    return Promise.resolve();
  }
  return tauriOpenUrl(url);
}
