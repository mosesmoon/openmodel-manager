import { useEffect, useMemo, useRef, useState } from "react";
import { api, gb, JitStatus, LlmEngine, LocalModel, PlanPreview, PyEngineInfo, PyKind } from "../api";
import { useApp } from "../App";
import { Key, useI18n } from "../i18n";
import { SplitBar } from "./Models";
import { Switch } from "./Settings";

const CTX_CHOICES = [4096, 8192, 16384, 32768, 65536, 131072, 262144];

export default function Run() {
  const { t } = useI18n();
  const { llm, llmEvents, pageArg, fail, go, settings, saveSettings } = useApp();
  const [jit, setJit] = useState<JitStatus | null>(null);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [path, setPath] = useState<string>(pageArg ?? llm?.model ?? "");
  const [ctx, setCtx] = useState<number | null>(null);
  const [vision, setVision] = useState(true);
  const [custom, setCustom] = useState<string | null>(null);
  const [preview, setPreview] = useState<PlanPreview | null>(null);
  const [planErr, setPlanErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [engine, setEngine] = useState<LlmEngine>("llama");
  const [hasStrata, setHasStrata] = useState(false);
  const [py, setPy] = useState<PyEngineInfo[]>([]);
  const [hfModel, setHfModel] = useState("");
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    api.engines().then((e) => setHasStrata(!!e.strata), () => {});
    api.pyEngines().then(setPy, () => {});
    api.library().then((l) => {
      setModels(l);
      // A Strata-managed model is running under a label, not a file path.
      if (l.length && !l.some((x) => x.path === path)) setPath(l[0].path);
    }, fail);
  }, []);

  useEffect(() => {
    const poll = () => api.jitStatus().then(setJit, () => {});
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, [settings.jit_api]);

  // New model selected: forget the custom args typed for the previous one.
  useEffect(() => setCustom(null), [path]);

  useEffect(() => {
    if (!path || !models.some((x) => x.path === path)) return;
    const id = setTimeout(() => {
      api.planPreview({ path, ctx, custom_args: custom, vision }).then((p) => {
        setPreview(p);
        setPlanErr(null);
        if (custom === null) setCustom(p.saved_custom_args);
        if (custom === null) setEngine((cur) => (cur === "vllm" || cur === "mlx" ? cur : p.strata_supported && hasStrata ? "strata" : "llama"));
      }, (e) => { setPreview(null); setPlanErr(String(e)); });
    }, 250);
    return () => clearTimeout(id);
  }, [path, ctx, vision, custom, hasStrata, models]);

  const logs = useMemo(() => llmEvents.flatMap((e) => (e.type === "log" ? [e.line] : e.type === "note" ? [`* ${e.message}`] : [])), [llmEvents]);
  const progress = [...llmEvents].reverse().find((e) => e.type === "progress");
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs.length]);

  const pyMode = engine === "vllm" || engine === "mlx";
  const start = () => {
    setStarting(true);
    if (pyMode) {
      api.pyEngineStart(engine as PyKind, hfModel, engine === "vllm" ? ctx : null).then(() => {}, fail).finally(() => setStarting(false));
      return;
    }
    api.llmStart({ path, ctx, custom_args: custom ?? "", vision, engine }).then(() => {}, fail).finally(() => setStarting(false));
  };
  const running = llm?.state === "ready" || llm?.state === "starting" || starting;
  const strata = engine === "strata";
  const plan = strata || pyMode ? undefined : preview?.plan;
  const mine = llm?.model && llm.model === (pyMode ? hfModel.trim() : path);
  const pyInfo = (k: PyKind) => py.find((x) => x.kind === k);
  const pyLabel = (k: PyKind) => {
    const i = pyInfo(k);
    if (!i) return "";
    if (i.unsupported) return ` — ${t(i.unsupported === "linux" ? "engine.unsupported_linux" : "engine.unsupported_mac")}`;
    return i.installed ? "" : ` — ${t("run.py_missing")}`;
  };

  return (
    <>
      <h1>{t("nav.run")}</h1>
      <div className="two">
        <div className="card stack">
          <label className="field">
            <span>{t("run.engine")}</span>
            <select value={engine} onChange={(e) => setEngine(e.target.value as LlmEngine)}>
              <option value="llama">llama.cpp</option>
              <option value="strata" disabled={!hasStrata}>Strata{hasStrata ? "" : ` — ${t("run.strata_missing")}`}</option>
              {(["vllm", "mlx"] as PyKind[]).map((k) => (
                <option key={k} value={k} disabled={!pyInfo(k)?.installed || !!pyInfo(k)?.unsupported}>{pyInfo(k)?.name ?? k}{pyLabel(k)}</option>
              ))}
            </select>
          </label>
          {pyMode ? (
            <label className="field">
              <span>{t("run.hf_model")}</span>
              <input type="text" className="mono" value={hfModel} placeholder={engine === "vllm" ? "Qwen/Qwen3-8B" : "mlx-community/Qwen3-8B-4bit"} onChange={(e) => setHfModel(e.target.value)} />
              <span className="small muted">{t(engine === "vllm" ? "run.hf_hint_vllm" : "run.hf_hint_mlx")}</span>
            </label>
          ) : (
            <label className="field">
              <span>{t("run.pick")}</span>
              <select value={path} onChange={(e) => setPath(e.target.value)}>
                {models.length === 0 && <option value="">{t("models.empty")}</option>}
                {models.map((m) => <option key={m.path} value={m.path}>{m.name} {m.quant ? `(${m.quant})` : ""}</option>)}
              </select>
            </label>
          )}
          {strata && <div className="note">{t("run.strata_note")}</div>}
          {strata && preview && !preview.strata_supported && <div className="note bad">{t("run.strata_unsupported")}</div>}
          {engine !== "mlx" && <label className="field">
            <span>{t("run.ctx")}</span>
            <select value={ctx ?? ""} onChange={(e) => setCtx(e.target.value ? Number(e.target.value) : null)}>
              <option value="">{t("common.auto")}{plan && ctx === null ? ` (${plan.ctx / 1024}K)` : ""}</option>
              {CTX_CHOICES.filter((c) => pyMode || !preview || c <= Math.max(preview.summary.context_length, 4096)).map((c) => <option key={c} value={c}>{c / 1024}K</option>)}
            </select>
          </label>}
          {!pyMode && preview?.mmproj && (
            <label className="check"><input type="checkbox" checked={vision} onChange={(e) => setVision(e.target.checked)} />{t("run.vision")}</label>
          )}
          {!strata && !pyMode && (
            <label className="field">
              <span>{t("run.custom")}</span>
              <input type="text" className="mono" value={custom ?? ""} placeholder="-ngl 99 --temp 0.7" onChange={(e) => setCustom(e.target.value)} />
              <span className="small muted">{t("run.custom_hint")}</span>
            </label>
          )}
          <div className="row">
            {!running || !mine ? (
              <button className="btn primary" disabled={(pyMode ? !hfModel.trim() : !path || (!!planErr && !strata)) || starting} onClick={start}>{t("common.start")}</button>
            ) : null}
            {(llm?.state === "ready" || llm?.state === "starting") && (
              <button className="btn" onClick={() => api.serverStop("llm").catch(fail)}>{t("common.stop")}</button>
            )}
            {llm?.state === "ready" && <button className="btn" onClick={() => go("chat")}>{t("nav.chat")}</button>}
          </div>
          <div className="jit stack" style={{ gap: 4 }}>
            <div className="row">
              <Switch on={settings.jit_api} label={t("run.jit")} onChange={(v) => saveSettings({ ...settings, jit_api: v }).catch(fail)} />
              <b className="small">{t("run.jit")}</b>
            </div>
            <div className="small muted">{t("run.jit_hint")}</div>
            {jit && (
              <div className="small">
                <span className="muted">{t("overview.api")}:</span>
                {jit.urls.map((u) => <div key={u} className="mono">{u}</div>)}
                {jit.error && <div className="note warn small">{t("run.jit_waiting").replace("{port}", String(jit.port))}</div>}
                {jit.loaded.length > 0 && <div className="muted">{t("run.jit_also")}: {jit.loaded.map((x) => x.name).join("、")}</div>}
              </div>
            )}
          </div>
          {llm && (
            <div className="small">
              <span className={"dot " + llm.state} />{t(("server." + llm.state) as Key)}
              {llm.engine === "strata" && <> · Strata{llm.model && !models.some((x) => x.path === llm.model) ? ` · ${llm.model}` : ""}</>}
              {(llm.engine === "vllm" || llm.engine === "mlx") && <> · {llm.engine === "vllm" ? "vLLM" : "MLX"} · {llm.model}</>}
                            {llm.safe_level > 0 && <> · {t("run.safe")} {llm.safe_level}</>}
              {llm.gpu_layers && <> · {t("run.offloaded")} {llm.gpu_layers[0]}/{llm.gpu_layers[1]}</>}
            </div>
          )}
          {llm?.state === "starting" && llm.engine === "llama" && progress?.type === "progress" && (
            <div className="stack" style={{ gap: 4 }}>
              <div className="bar"><div style={{ width: `${progress.fraction * 100}%` }} /></div>
              <div className="muted small">{progress.stage} · {progress.elapsed_secs.toFixed(0)}s</div>
            </div>
          )}
          {llm?.state === "crashed" && llm.last_crash && (
            <div className="note bad">
              {t("run.crash")}{llm.last_crash.exit_code != null ? ` (exit ${llm.last_crash.exit_code})` : ""}
              {llm.last_crash.problems.length > 0 && <>: {llm.last_crash.problems.join(", ")}</>}
            </div>
          )}
        </div>

        <div className="stack">
          {planErr && <div className="note bad">{planErr}</div>}
          {plan && preview && (
            <div className="card stack">
              <div className="row spread">
                <h2 style={{ margin: 0 }}>{t("run.plan")}</h2>
                <span className={"pill " + plan.tier}>{plan.tier_label}</span>
              </div>
              <div className="stats">
                <div><span className="muted small">{t("run.first_load")}</span><b>≈ {plan.est_load_secs.toFixed(0)} s</b></div>
                <div><span className="muted small">{t("run.tps")}</span><b>{plan.est_tps.toFixed(1)} tok/s</b></div>
              </div>
              <div>{plan.summary}</div>
              <SplitBar gpu={plan.gpu_bytes} ram={plan.ram_bytes} disk={plan.disk_bytes} />
              <div className="row small">
                <span>{t("run.kv")}: {gb(plan.kv_bytes, 2)}</span>
                <span>{t("models.ctx")}: {plan.ctx / 1024}K</span>
                <span className="muted">{preview.summary.arch} · {preview.summary.block_count} layers{preview.summary.expert_count > 1 ? ` · MoE ${preview.summary.expert_used_count}/${preview.summary.expert_count}` : ""}</span>
              </div>
              {[plan.gpu_note, plan.ram_note, plan.disk_note].filter(Boolean).map((n, i) => <div key={i} className="small muted">· {n}</div>)}
              <div>
                <div className="small muted" style={{ marginBottom: 4 }}>{t("run.command")}</div>
                <code className="cmd">llama-server {plan.command}</code>
              </div>
            </div>
          )}
          <div className="card">
            <h2>{t("run.logs")}</h2>
            <pre className="log" ref={logRef}>{logs.join("\n") || "—"}</pre>
          </div>
        </div>
      </div>
    </>
  );
}
