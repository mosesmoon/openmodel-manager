import { useEffect, useState } from "react";
import { api, Engines, gb, Hardware, NamedEndpoint, VramUse } from "../api";
import { useApp } from "../App";
import { Key, useI18n } from "../i18n";
import { findProvider } from "../providers";

type ProvState = { state: "checking" } | { state: "nokey" } | { state: "ok"; models: string[]; ms: number } | { state: "error"; message: string; ms: number };

export default function Overview() {
  const { t } = useI18n();
  const { llm, sd, settings, go, fail } = useApp();
  const [hw, setHw] = useState<Hardware | null>(null);
  const [vram, setVram] = useState<VramUse[]>([]);
  const [eng, setEng] = useState<Engines | null>(null);

  const load = (refresh: boolean) => {
    api.hardware(refresh).then(setHw, fail);
    api.engines().then(setEng, fail);
  };
  useEffect(() => {
    load(false);
    const tick = () => api.liveVram().then(setVram).catch(() => {});
    tick();
    const id = setInterval(tick, 3000);
    return () => clearInterval(id);
  }, []);

  return (
    <>
      <div className="row spread">
        <h1>{t("overview.title")}</h1>
        <button className="btn" onClick={() => load(true)}>{t("common.refresh")}</button>
      </div>
      {eng && eng.llama.length === 0 && (
        <div className="note warn row" style={{ marginBottom: 14 }}>
          <span className="grow">{t("overview.no_engine")}</span>
          <button className="btn sm" onClick={() => go("settings")}>{t("nav.settings")}</button>
        </div>
      )}
      {!hw ? (
        <p className="muted">{t("common.loading")}</p>
      ) : (
        <div className="grid">
          <div className="card">
            <h3 className="muted">{t("overview.cpu")}</h3>
            <div>{hw.cpu_name}</div>
            <div className="muted small">{hw.cpu_cores}C / {hw.cpu_threads}T · {hw.os} {hw.arch}</div>
          </div>
          <div className="card">
            <h3 className="muted">{t("overview.ram")}</h3>
            <div className="big">{gb(hw.ram_total, 0)}</div>
            <div className="muted small">{hw.ram_type ?? ""} · {gb(hw.ram_available)} free · ~{hw.ram_bandwidth_gbs.toFixed(0)} GB/s</div>
          </div>
          <div className="card">
            <h3 className="muted">{t("overview.gpu")}</h3>
            {hw.gpus.length === 0 && <div className="muted">{t("overview.no_gpu")}</div>}
            {hw.gpus.map((g, i) => (
              <div key={i} style={{ marginBottom: 6 }}>
                <div>{g.name}</div>
                <div className="muted small">
                  {gb(g.vram_total)}{g.unified_memory ? " (shared)" : ""} · ~{g.bandwidth_gbs.toFixed(0)} GB/s{g.driver ? ` · ${g.driver}` : ""}
                </div>
              </div>
            ))}
          </div>
          <div className="card">
            <h3 className="muted">{t("overview.disk")}</h3>
            {hw.model_disk ? (
              <>
                <div>{hw.model_disk.mount} {hw.model_disk.is_ssd === true ? "SSD" : hw.model_disk.is_ssd === false ? "HDD" : ""}</div>
                <div className="muted small">{gb(hw.model_disk.available)} / {gb(hw.model_disk.total)}</div>
              </>
            ) : <div className="muted">—</div>}
            <div className="muted small ellipsis" title={settings.models_dir}>{settings.models_dir}</div>
          </div>
          <div className="card">
            <h3 className="muted">{t("overview.backends")}</h3>
            <div className="row">{hw.recommended_backends.map((b) => <span key={b} className="pill">{b}</span>)}</div>
          </div>
        </div>
      )}

      {vram.length > 0 && (
        <div className="card">
          <h2>{t("overview.vram_live")}</h2>
          {vram.map((v, i) => (
            <div key={i} className="stack" style={{ marginBottom: 8 }}>
              <div className="row spread small"><span>{v.name}</span><span>{gb(v.used)} / {gb(v.total)}</span></div>
              <div className="bar"><div style={{ width: `${(v.used / Math.max(1, v.total)) * 100}%` }} /></div>
              {v.spill_suspected && <div className="note warn">{t("overview.spill")}</div>}
            </div>
          ))}
        </div>
      )}

      <div className="grid">
        <div className="card">
          <h2>{t("overview.llm")}</h2>
          <div><span className={"dot " + (llm?.state ?? "stopped")} />{t(("server." + (llm?.state ?? "stopped")) as Key)}</div>
          {llm?.model && <div className="muted small ellipsis" title={llm.model}>{llm.model}</div>}
          {llm?.state === "ready" && (
            <div className="small" style={{ marginTop: 6 }}>
              {t("overview.api")}:
              {["127.0.0.1", ...llm.exposed].map((ip) => <div key={ip} className="mono">http://{ip}:{llm.port}/v1</div>)}
              {llm.api_key && (
                <div className="row" style={{ marginTop: 4 }}>
                  {t("overview.api_key")}: <span className="mono">{llm.api_key}</span>
                  <button className="btn" onClick={() => navigator.clipboard.writeText(llm.api_key ?? "")}>{t("overview.copy")}</button>
                </div>
              )}
            </div>
          )}
          <div className="row" style={{ marginTop: 10 }}>
            <button className="btn" onClick={() => go("run")}>{t("nav.run")}</button>
          </div>
        </div>
        <div className="card">
          <h2>{t("overview.sd")}</h2>
          <div><span className={"dot " + (sd?.state ?? "stopped")} />{t(("server." + (sd?.state ?? "stopped")) as Key)}</div>
          <div className="row" style={{ marginTop: 10 }}>
            <button className="btn" onClick={() => go("image")}>{t("nav.image")}</button>
          </div>
        </div>
      </div>

      <ProvidersCard list={settings.endpoints} />
    </>
  );
}

function ProvidersCard({ list }: { list: NamedEndpoint[] }) {
  const { t } = useI18n();
  const { go } = useApp();
  const [st, setSt] = useState<Record<number, ProvState>>({});
  const check = () => {
    list.forEach((e, i) => {
      const preset = findProvider(e.provider);
      if (preset.id !== "custom" && !e.api_key) {
        setSt((s) => ({ ...s, [i]: { state: "nokey" } }));
        return;
      }
      setSt((s) => ({ ...s, [i]: { state: "checking" } }));
      const t0 = performance.now();
      api.endpointModels(e).then(
        (models) => setSt((s) => ({ ...s, [i]: { state: "ok", models, ms: performance.now() - t0 } })),
        (err) => setSt((s) => ({ ...s, [i]: { state: "error", message: String(err), ms: performance.now() - t0 } })),
      );
    });
  };
  // Re-check when the saved list changes (keys, addresses).
  useEffect(check, [JSON.stringify(list)]);
  const dot = (s?: ProvState) => (s?.state === "ok" ? "ready" : s?.state === "error" ? "crashed" : s?.state === "checking" ? "starting" : "stopped");
  return (
    <div className="card stack">
      <div className="row spread">
        <h2 style={{ margin: 0 }}>{t("overview.providers")}</h2>
        <div className="row">
          {list.length > 0 && <button className="btn sm" onClick={check}>{t("overview.prov_recheck")}</button>}
          <button className="btn sm" onClick={() => go("settings")}>{t("nav.settings")}</button>
        </div>
      </div>
      {list.length === 0 && <div className="muted">{t("overview.prov_none")}</div>}
      {list.map((e, i) => {
        const s = st[i];
        const listed = s?.state === "ok" && (!e.model || s.models.length === 0 || s.models.some((m) => m === e.model || m.endsWith("/" + e.model)));
        return (
          <div key={i} className="row" style={{ alignItems: "flex-start" }}>
            <span className={"dot " + dot(s)} style={{ marginTop: 7 }} />
            <div className="grow stack" style={{ gap: 2, minWidth: 0 }}>
              <div className="row" style={{ gap: 8 }}>
                <b>{e.name}</b>
                <span className="muted small">{findProvider(e.provider).name}{e.model ? ` · ${e.model}` : ""}</span>
              </div>
              <div className="small">
                {!s || s.state === "checking" ? <span className="muted">{t("overview.prov_checking")}</span>
                  : s.state === "nokey" ? <span className="muted">{t("overview.prov_nokey")}</span>
                  : s.state === "ok" ? <span>{t("overview.prov_ok")} · {t("overview.prov_models").replace("{n}", String(s.models.length))} · {Math.round(s.ms)} ms</span>
                  : <span style={{ color: "var(--bad)" }}>{t("overview.prov_err")} · {Math.round(s.ms)} ms</span>}
              </div>
              {s?.state === "error" && <div className="muted small ellipsis" title={s.message}>{s.message}</div>}
              {s?.state === "ok" && !listed && <div className="note warn small">{t("overview.prov_model_missing").replace("{m}", e.model)}</div>}
            </div>
          </div>
        );
      })}
      {list.length > 0 && <div className="muted small">{t("overview.prov_hint")}</div>}
    </div>
  );
}
