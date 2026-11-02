import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";
import { api, Backend, EngineKind, Engines, NamedEndpoint, Settings } from "../api";
import { useApp } from "../App";
import { Key, useI18n } from "../i18n";

const BACKENDS: Backend[] = ["cuda", "rocm", "vulkan", "metal", "sycl", "cpu"];
const GIB = 1024 ** 3;

export default function SettingsPage() {
  const { t } = useI18n();
  const { settings, saveSettings, fail } = useApp();
  const [s, setS] = useState<Settings>(settings);
  const [saved, setSaved] = useState(false);
  const [dataDir, setDataDir] = useState("");
  useEffect(() => { api.paths().then((p) => setDataDir(p.data_dir)).catch(() => {}); }, []);

  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => {
    setS({ ...s, [k]: v });
    setSaved(false);
  };
  const save = () => saveSettings(s).then(() => setSaved(true), fail);
  const pickDir = async (k: "models_dir" | "engines_dir" | "outputs_dir") => {
    const p = await open({ directory: true, multiple: false, defaultPath: s[k] });
    if (typeof p === "string") set(k, p);
  };

  return (
    <>
      <div className="row spread">
        <h1>{t("nav.settings")}</h1>
        <div className="row">
          {saved && <span className="muted small">{t("common.saved")}</span>}
          <button className="btn primary" onClick={save}>{t("common.save")}</button>
        </div>
      </div>

      <div className="card stack">
        <h2>{t("settings.general")}</h2>
        <label className="field" style={{ maxWidth: 240 }}>
          <span>{t("settings.language")}</span>
          <select value={s.language} onChange={(e) => set("language", e.target.value)}>
            <option value="zh-TW">繁體中文</option>
            <option value="en">English</option>
          </select>
        </label>
      </div>

      <div className="card stack">
        <h2>{t("settings.dirs")}</h2>
        {(["models_dir", "engines_dir", "outputs_dir"] as const).map((k) => (
          <div key={k} className="row">
            <span style={{ width: 90 }} className="muted">{t(("settings." + k) as Key)}</span>
            <input type="text" className="grow mono" value={s[k]} onChange={(e) => set(k, e.target.value)} />
            <button className="btn sm" onClick={() => pickDir(k)}>{t("common.browse")}</button>
            <button className="btn sm" onClick={() => openPath(s[k]).catch(fail)}>{t("common.open_folder")}</button>
          </div>
        ))}
        <div className="row">
          <span style={{ width: 90 }} className="muted">{t("settings.data_dir")}</span>
          <span className="grow mono small">{dataDir}</span>
          <button className="btn sm" onClick={() => openPath(dataDir).catch(fail)}>{t("common.open_folder")}</button>
        </div>
      </div>

      <EnginesCard backend={s.backend} setBackend={(b) => set("backend", b)} />

      <div className="card stack">
        <h2>{t("settings.network")}</h2>
        <label className="field"><span>{t("settings.hf_endpoint")}</span><input type="text" value={s.hf_endpoint} onChange={(e) => set("hf_endpoint", e.target.value)} /></label>
        <label className="field"><span>{t("settings.hf_token")}</span><input type="password" value={s.hf_token ?? ""} onChange={(e) => set("hf_token", e.target.value || null)} /></label>
        <label className="field"><span>{t("settings.github_prefix")}</span><input type="text" value={s.github_mirror_prefix} onChange={(e) => set("github_mirror_prefix", e.target.value)} /></label>
        <label className="check"><input type="checkbox" checked={s.lan_access} onChange={(e) => set("lan_access", e.target.checked)} />{t("settings.lan")}</label>
        <div className="row">
          <span className="muted">{t("settings.ports")}</span>
          <label className="row small">{t("settings.llm_port")}<input type="number" style={{ width: 90 }} value={s.server_port} onChange={(e) => set("server_port", Number(e.target.value))} /></label>
          <label className="row small">{t("settings.image_port")}<input type="number" style={{ width: 90 }} value={s.image_port} onChange={(e) => set("image_port", Number(e.target.value))} /></label>
        </div>
        <div className="muted small">{t("settings.privacy")}</div>
      </div>

      <div className="card stack">
        <h2>{t("settings.memory")}</h2>
        <div className="row">
          <label className="row small">{t("settings.ram_reserve")}<input type="number" step={0.5} min={0} style={{ width: 90 }} value={s.ram_reserve_bytes / GIB} onChange={(e) => set("ram_reserve_bytes", Math.round(Number(e.target.value) * GIB))} /></label>
          <label className="row small">{t("settings.vram_reserve")}<input type="number" step={0.25} min={0} style={{ width: 90 }} value={s.vram_reserve_bytes / GIB} onChange={(e) => set("vram_reserve_bytes", Math.round(Number(e.target.value) * GIB))} /></label>
        </div>
        <label className="check">
          <input type="checkbox" checked={!!s.gpu_override} onChange={(e) => set("gpu_override", e.target.checked ? { vendor: "nvidia", name: "GPU", vram_gb: 8 } : null)} />
          {t("settings.gpu_override")}
        </label>
        {s.gpu_override && (
          <div className="row">
            <label className="row small">{t("settings.vendor")}
              <select value={s.gpu_override.vendor} onChange={(e) => set("gpu_override", { ...s.gpu_override!, vendor: e.target.value })}>
                {["nvidia", "amd", "intel", "apple", "other"].map((v) => <option key={v}>{v}</option>)}
              </select>
            </label>
            <label className="row small">{t("settings.name")}<input type="text" value={s.gpu_override.name} onChange={(e) => set("gpu_override", { ...s.gpu_override!, name: e.target.value })} /></label>
            <label className="row small">{t("settings.vram_gb")}<input type="number" style={{ width: 80 }} value={s.gpu_override.vram_gb} onChange={(e) => set("gpu_override", { ...s.gpu_override!, vram_gb: Number(e.target.value) })} /></label>
          </div>
        )}
      </div>

      <EndpointsCard list={s.endpoints} setList={(l) => set("endpoints", l)} />
    </>
  );
}

function EnginesCard({ backend, setBackend }: { backend: Backend | null; setBackend: (b: Backend | null) => void }) {
  const { t } = useI18n();
  const { track, fail } = useApp();
  const [eng, setEng] = useState<Engines | null>(null);
  const load = () => api.engines().then(setEng, fail);
  useEffect(() => { load(); }, []);

  const install = (kind: EngineKind) => api.engineInstall(kind, backend).then((id) => track(id, `${kind === "llama" ? "llama.cpp" : "stable-diffusion.cpp"} (${backend ?? eng?.recommended[0] ?? "auto"})`, () => load()), fail);
  const importIt = async (kind: EngineKind, dir: boolean) => {
    const p = await open(dir ? { directory: true } : { filters: [{ name: "Archive", extensions: ["zip", "gz", "tgz"] }] });
    if (typeof p === "string") api.engineImport(kind, p).then(load, fail);
  };

  return (
    <div className="card stack">
      <h2>{t("settings.engines")}</h2>
      <label className="row small">
        {t("settings.backend")}
        <select value={backend ?? ""} onChange={(e) => setBackend((e.target.value || null) as Backend | null)}>
          <option value="">{t("common.auto")}{eng ? ` (${eng.recommended[0] ?? "cpu"})` : ""}</option>
          {BACKENDS.map((b) => <option key={b} value={b}>{b}</option>)}
        </select>
      </label>
      {(["llama", "sd"] as EngineKind[]).map((kind) => (
        <div key={kind} className="stack" style={{ gap: 6 }}>
          <div className="row">
            <b className="grow">{kind === "llama" ? "llama.cpp" : "stable-diffusion.cpp"}</b>
            <button className="btn sm primary" onClick={() => install(kind)}>{t("settings.install")}</button>
            <button className="btn sm" onClick={() => importIt(kind, false)}>{t("settings.import")}</button>
            <button className="btn sm" onClick={() => importIt(kind, true)}>📁</button>
          </div>
          {eng && eng[kind].length === 0 && <div className="muted small">{t("common.none")}</div>}
          {eng?.[kind].map((e) => (
            <div key={e.binary} className="row small">
              <span className="pill">{e.backend}</span>
              <span>{e.version}</span>
              <span className="muted mono ellipsis grow" title={e.binary}>{e.binary}</span>
              <button className="btn sm danger" onClick={() => api.engineRemove(kind, e.binary).then(load, fail)}>{t("common.remove")}</button>
            </div>
          ))}
        </div>
      ))}
      <div className="stack" style={{ gap: 6 }}>
        <div className="row">
          <b className="grow">Strata</b>
          {eng?.strata_available && (
            <button className="btn sm primary" onClick={() => api.strataInstall().then((id) => track(id, "Strata", () => load()), fail)}>{t("settings.install")}</button>
          )}
        </div>
        <div className="muted small">{t("settings.strata_hint")}</div>
        {eng && !eng.strata_available && <div className="muted small">{t("settings.strata_unavailable")}</div>}
        {eng && eng.strata_available && !eng.strata && <div className="muted small">{t("common.none")}</div>}
        {eng?.strata && (
          <div className="row small">
            <span>{eng.strata.version}</span>
            <span className="muted mono ellipsis grow" title={eng.strata.launcher}>{eng.strata.launcher}</span>
            <button className="btn sm" onClick={() => openPath(eng.strata_data_dir).catch(fail)} title={t("settings.strata_data")}>{t("common.open_folder")}</button>
            <button className="btn sm danger" onClick={() => api.strataRemove().then(load, fail)}>{t("common.remove")}</button>
          </div>
        )}
      </div>
    </div>
  );
}

function EndpointsCard({ list, setList }: { list: NamedEndpoint[]; setList: (l: NamedEndpoint[]) => void }) {
  const { t } = useI18n();
  const { fail } = useApp();
  const [models, setModels] = useState<Record<number, string[]>>({});
  const upd = (i: number, patch: Partial<NamedEndpoint>) => setList(list.map((e, j) => (j === i ? { ...e, ...patch } : e)));
  const fetchModels = (i: number) => api.endpointModels(list[i]).then((m) => setModels({ ...models, [i]: m }), fail);
  return (
    <div className="card stack">
      <h2>{t("settings.endpoints")}</h2>
      <div className="muted small">{t("settings.endpoint_hint")}</div>
      {list.map((e, i) => (
        <div key={i} className="row">
          <input type="text" style={{ width: 120 }} placeholder={t("settings.name")} value={e.name} onChange={(x) => upd(i, { name: x.target.value })} />
          <input type="text" className="grow" placeholder="https://api.example.com/v1" value={e.base_url} onChange={(x) => upd(i, { base_url: x.target.value })} />
          <input type="password" style={{ width: 160 }} placeholder="API key" value={e.api_key ?? ""} onChange={(x) => upd(i, { api_key: x.target.value || null })} />
          <input type="text" list={`models-${i}`} style={{ width: 180 }} placeholder="model" value={e.model} onChange={(x) => upd(i, { model: x.target.value })} />
          <datalist id={`models-${i}`}>{(models[i] ?? []).map((m) => <option key={m} value={m} />)}</datalist>
          <button className="btn sm" onClick={() => fetchModels(i)}>↻</button>
          <button className="btn sm danger" onClick={() => setList(list.filter((_, j) => j !== i))}>×</button>
        </div>
      ))}
      <div>
        <button className="btn sm" onClick={() => setList([...list, { name: `API ${list.length + 1}`, base_url: "", api_key: null, model: "" }])}>{t("settings.add")}</button>
      </div>
    </div>
  );
}
