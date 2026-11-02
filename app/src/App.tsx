import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { api, errText, JobDone, on, Progress, ServerEvent, ServerStatus, Settings, speed } from "./api";
import { inBrowser, session, signIn, signOut } from "./bridge";
import { I18nProvider, Key, Lang, useI18n } from "./i18n";
import Overview from "./pages/Overview";
import Models from "./pages/Models";
import Run from "./pages/Run";
import Chat from "./pages/Chat";
import ImagePage from "./pages/Image";
import VideoPage from "./pages/Video";
import Agent from "./pages/Agent";
import SettingsPage from "./pages/Settings";

export type Page = "overview" | "models" | "run" | "chat" | "image" | "video" | "agent" | "settings";
const PAGES: Page[] = ["overview", "models", "run", "chat", "image", "video", "agent", "settings"];

export type NavPos = "left" | "right" | "top" | "bottom";
export interface NavPref { pos: NavPos; collapsed: boolean }
const NAV_KEY = "openmodel.nav";

/** Where the menu sits is each viewer's own choice, kept in this browser. */
function loadNav(): NavPref {
  try {
    const v = JSON.parse(localStorage.getItem(NAV_KEY) ?? "null");
    if (v && ["left", "right", "top", "bottom"].includes(v.pos)) return { pos: v.pos, collapsed: !!v.collapsed };
  } catch { /* storage blocked or a bad value */ }
  return { pos: "left", collapsed: false };
}

interface Job { id: string; label: string; progress?: Progress; onDone?: (d: JobDone) => void }

interface AppCtx {
  settings: Settings;
  saveSettings: (s: Settings) => Promise<void>;
  llm: ServerStatus | null;
  sd: ServerStatus | null;
  llmEvents: ServerEvent[];
  sdEvents: ServerEvent[];
  go: (p: Page, arg?: string) => void;
  pageArg: string | undefined;
  track: (id: string, label: string, onDone?: (d: JobDone) => void) => void;
  jobs: Job[];
  fail: (e: unknown) => void;
  nav: NavPref;
  setNav: (n: NavPref) => void;
}

const Ctx = createContext<AppCtx>(null as unknown as AppCtx);
export const useApp = () => useContext(Ctx);

const MAX_EVENTS = 600;

export default function App() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  // In a browser: null while checking, false shows the sign-in page.
  const [signedIn, setSignedIn] = useState<boolean | null>(inBrowser ? null : true);
  useEffect(() => {
    if (!inBrowser) return;
    session().then(setSignedIn, () => setSignedIn(false));
    const out = () => { setSignedIn(false); setSettings(null); };
    window.addEventListener("openmodel-signed-out", out);
    return () => window.removeEventListener("openmodel-signed-out", out);
  }, []);
  useEffect(() => {
    if (signedIn) api.settings().then(setSettings, (e) => setBootError(errText(e)));
  }, [signedIn]);
  if (signedIn === false) return <SignIn onDone={() => setSignedIn(true)} />;
  if (bootError) return <div className="main"><div className="note bad">{bootError}</div></div>;
  if (!settings) return null;
  return (
    <I18nProvider lang={(settings.language as Lang) || "zh-TW"}>
      <Shell settings={settings} setSettings={setSettings} />
    </I18nProvider>
  );
}

function Shell({ settings, setSettings }: { settings: Settings; setSettings: (s: Settings) => void }) {
  const { t } = useI18n();
  const [page, setPage] = useState<Page>("overview");
  const [pageArg, setPageArg] = useState<string>();
  const [llm, setLlm] = useState<ServerStatus | null>(null);
  const [sd, setSd] = useState<ServerStatus | null>(null);
  const [llmEvents, setLlmEvents] = useState<ServerEvent[]>([]);
  const [sdEvents, setSdEvents] = useState<ServerEvent[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [errors, setErrors] = useState<{ id: number; text: string }[]>([]);
  const [nav, setNavState] = useState<NavPref>(loadNav);
  const setNav = (n: NavPref) => {
    setNavState(n);
    try { localStorage.setItem(NAV_KEY, JSON.stringify(n)); } catch { /* not kept, still applied */ }
  };
  const jobsRef = useRef<Job[]>([]);
  jobsRef.current = jobs;

  const fail = useCallback((e: unknown) => {
    const id = Date.now() + Math.random();
    setErrors((x) => [...x, { id, text: errText(e) }]);
  }, []);

  useEffect(() => {
    const refresh = () => {
      api.serverStatus("llm").then(setLlm).catch(() => {});
      api.serverStatus("sd").then(setSd).catch(() => {});
    };
    refresh();
    const subs = [
      on<{ which: "llm" | "sd"; event: ServerEvent }>("server", ({ which, event }) => {
        const add = (xs: ServerEvent[]) => {
          // Keep only the latest progress event to bound memory.
          const base = event.type === "progress" ? xs.filter((x) => x.type !== "progress") : xs;
          return [...base.slice(-MAX_EVENTS), event];
        };
        if (which === "llm") setLlmEvents(add); else setSdEvents(add);
        if (event.type === "status" || event.type === "crashed") refresh();
      }),
      on<Progress>("progress", (p) => setJobs((js) => js.map((j) => (j.id === p.job_id ? { ...j, progress: p } : j)))),
      on<JobDone>("job-done", (d) => {
        const j = jobsRef.current.find((x) => x.id === d.job_id);
        setJobs((js) => js.filter((x) => x.id !== d.job_id));
        if (!d.ok && d.error && !d.error.includes("cancelled")) fail(d.error);
        j?.onDone?.(d);
      }),
    ];
    const timer = setInterval(refresh, 4000);
    return () => {
      clearInterval(timer);
      subs.forEach((s) => s.then((u) => u()));
    };
  }, [fail]);

  const ctx: AppCtx = {
    settings,
    saveSettings: async (s) => {
      await api.saveSettings(s);
      setSettings(s);
    },
    llm,
    sd,
    llmEvents,
    sdEvents,
    go: (p, arg) => {
      setPage(p);
      setPageArg(arg);
    },
    pageArg,
    track: (id, label, onDone) => setJobs((js) => [...js, { id, label, onDone }]),
    jobs,
    fail,
    nav,
    setNav,
  };

  const fill = page === "chat" || page === "agent";
  return (
    <Ctx.Provider value={ctx}>
      <div className={`app pos-${nav.pos}` + (nav.collapsed ? " collapsed" : "")}>
        <div className="navwrap">
          {!nav.collapsed && (
            <nav className="side">
              <select
                className="navlang"
                aria-label="語言 / Language"
                value={settings.language}
                onChange={(e) => ctx.saveSettings({ ...settings, language: e.target.value }).catch(fail)}
              >
                <option value="zh-TW">繁體中文</option>
                <option value="en">English</option>
              </select>
              <div className="brand"><img src="/icon.svg" alt="" />OpenModel</div>
              {PAGES.map((p) => (
                <button key={p} className={"nav" + (p === page ? " on" : "")} onClick={() => ctx.go(p)}>
                  {t(("nav." + p) as Key)}
                </button>
              ))}
              <div className="side-foot">
                <ServerLine label={t("nav.chat")} st={llm} />
                <ServerLine label={t("nav.image")} st={sd} />
                {inBrowser && <button className="btn sm" onClick={() => signOut()}>{t("web.sign_out")}</button>}
              </div>
            </nav>
          )}
          <button
            className="navfold"
            title={t(nav.collapsed ? "nav.expand" : "nav.collapse")}
            aria-label={t(nav.collapsed ? "nav.expand" : "nav.collapse")}
            onClick={() => setNav({ ...nav, collapsed: !nav.collapsed })}
          >
            {FOLD_ARROW[nav.pos][nav.collapsed ? 1 : 0]}
          </button>
        </div>
        <main className={"main" + (fill ? " fill" : "")}>
          {page === "overview" && <Overview />}
          {page === "models" && <Models />}
          {page === "run" && <Run />}
          {page === "chat" && <Chat />}
          {page === "image" && <ImagePage />}
          {page === "video" && <VideoPage />}
          {page === "agent" && <Agent />}
          {page === "settings" && <SettingsPage />}
        </main>
      </div>
      <div className="jobs">
        {jobs.map((j) => <JobCard key={j.id} job={j} />)}
        {errors.map((e) => (
          <div className="toast" key={e.id}>
            <span className="grow">{e.text}</span>
            <button className="btn sm" onClick={() => setErrors((x) => x.filter((y) => y.id !== e.id))}>×</button>
          </div>
        ))}
      </div>
    </Ctx.Provider>
  );
}

// [arrow while open (toward the edge the menu folds into), arrow while folded]
const FOLD_ARROW: Record<NavPos, [string, string]> = { left: ["‹", "›"], right: ["›", "‹"], top: ["˄", "˅"], bottom: ["˅", "˄"] };

/** The web page's sign-in. Shown in both languages: settings need a session. */
function SignIn({ onDone }: { onDone: () => void }) {
  const zh = navigator.language.toLowerCase().startsWith("zh");
  const [pw, setPw] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    signIn(pw).then(onDone, (x) => { setError(zh ? "密碼錯誤" : errText(x)); setBusy(false); });
  };
  return (
    <div className="signin">
      <form className="card stack" onSubmit={submit}>
        <div className="brand" style={{ padding: 0 }}><img src="/icon.svg" alt="" />OpenModel</div>
        <h2 style={{ margin: 0 }}>{zh ? "登入 OpenModel" : "Sign in to OpenModel"}</h2>
        <label className="field"><span>{zh ? "密碼" : "Password"}</span><input type="password" autoFocus autoComplete="current-password" value={pw} onChange={(e) => setPw(e.target.value)} /></label>
        {error && <div className="note bad small">{error}</div>}
        <button className="btn primary" disabled={busy || !pw}>{zh ? "登入" : "Sign in"}</button>
      </form>
    </div>
  );
}

function ServerLine({ label, st }: { label: string; st: ServerStatus | null }) {
  const { t } = useI18n();
  const state = st?.state ?? "stopped";
  return (
    <div className="ellipsis">
      <span className={"dot " + state} />
      {label}: {t(("server." + state) as Key)}
    </div>
  );
}

function JobCard({ job }: { job: Job }) {
  const { t } = useI18n();
  const p = job.progress;
  const frac = p?.total ? Math.min(1, p.done / p.total) : null;
  return (
    <div className="job">
      <div className="row">
        <b className="grow ellipsis">{job.label}</b>
        <button className="btn sm" onClick={() => api.cancelJob(job.id)}>{t("common.cancel")}</button>
      </div>
      <div className="bar"><div style={{ width: `${(frac ?? 0) * 100}%` }} /></div>
      <div className="muted small">
        {p ? `${p.stage}${p.file_count > 1 ? ` · ${p.file_index + 1}/${p.file_count}` : ""}${frac != null ? ` · ${(frac * 100).toFixed(1)}%` : ""} ${speed(p.speed_bps)}` : t("common.loading")}
      </div>
    </div>
  );
}
