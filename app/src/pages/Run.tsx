import { useEffect, useMemo, useRef, useState } from "react";
import { api, gb, LlmEngine, LocalModel, PlanPreview } from "../api";
import { useApp } from "../App";
import { Key, useI18n } from "../i18n";
import { SplitBar } from "./Models";

const CTX_CHOICES = [4096, 8192, 16384, 32768, 65536, 131072, 262144];

export default function Run() {
  const { t } = useI18n();
  const { llm, llmEvents, pageArg, fail, go } = useApp();
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
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    api.engines().then((e) => setHasStrata(!!e.strata), () => {});
    api.library().then((l) => {
      setModels(l);
      if (!path && l.length) setPath(l[0].path);
    }, fail);
  }, []);

  // New model selected: forget the custom args typed for the previous one.
  useEffect(() => setCustom(null), [path]);

  useEffect(() => {
    if (!path) return;
    const id = setTimeout(() => {
      api.planPreview({ path, ctx, custom_args: custom, vision }).then((p) => {
        setPreview(p);
        setPlanErr(null);
        if (custom === null) setCustom(p.saved_custom_args);
        if (custom === null) setEngine(p.strata_supported && hasStrata ? "strata" : "llama");
      }, (e) => { setPreview(null); setPlanErr(String(e)); });
    }, 250);
    return () => clearTimeout(id);
  }, [path, ctx, vision, custom, hasStrata]);

  const logs = useMemo(() => llmEvents.flatMap((e) => (e.type === "log" ? [e.line] : e.type === "note" ? [`* ${e.message}`] : [])), [llmEvents]);
  const progress = [...llmEvents].reverse().find((e) => e.type === "progress");
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs.length]);

  const start = () => {
    setStarting(true);
    api.llmStart({ path, ctx, custom_args: custom ?? "", vision, engine }).then(() => {}, fail).finally(() => setStarting(false));
  };
  const running = llm?.state === "ready" || llm?.state === "starting" || starting;
  const strata = engine === "strata";
  const plan = strata ? undefined : preview?.plan;
  const mine = llm?.model && llm.model === path;

  return (
    <>
      <h1>{t("nav.run")}</h1>
      <div className="two">
        <div className="card stack">
          <label className="field">
            <span>{t("run.pick")}</span>
            <select value={path} onChange={(e) => setPath(e.target.value)}>
              {models.length === 0 && <option value="">{t("models.empty")}</option>}
              {models.map((m) => <option key={m.path} value={m.path}>{m.name} {m.quant ? `(${m.quant})` : ""}</option>)}
            </select>
          </label>
          {(hasStrata || preview?.strata_supported) && (
            <label className="field">
              <span>{t("run.engine")}</span>
              <select value={engine} onChange={(e) => setEngine(e.target.value as LlmEngine)}>
                <option value="llama">llama.cpp</option>
                <option value="strata" disabled={!hasStrata}>Strata{hasStrata ? "" : ` — ${t("run.strata_missing")}`}</option>
              </select>
            </label>
          )}
          {strata && <div className="note">{t("run.strata_note")}</div>}
          {strata && preview && !preview.strata_supported && <div className="note bad">{t("run.strata_unsupported")}</div>}
          <label className="field">
            <span>{t("run.ctx")}</span>
            <select value={ctx ?? ""} onChange={(e) => setCtx(e.target.value ? Number(e.target.value) : null)}>
              <option value="">{t("common.auto")}{plan && ctx === null ? ` (${plan.ctx / 1024}K)` : ""}</option>
              {CTX_CHOICES.filter((c) => !preview || c <= Math.max(preview.summary.context_length, 4096)).map((c) => <option key={c} value={c}>{c / 1024}K</option>)}
            </select>
          </label>
          {preview?.mmproj && (
            <label className="check"><input type="checkbox" checked={vision} onChange={(e) => setVision(e.target.checked)} />{t("run.vision")}</label>
          )}
          {!strata && (
            <label className="field">
              <span>{t("run.custom")}</span>
              <input type="text" className="mono" value={custom ?? ""} placeholder="-ngl 99 --temp 0.7" onChange={(e) => setCustom(e.target.value)} />
              <span className="small muted">{t("run.custom_hint")}</span>
            </label>
          )}
          <div className="row">
            {!running || !mine ? (
              <button className="btn primary" disabled={!path || (!!planErr && !strata) || starting} onClick={start}>{t("common.start")}</button>
            ) : null}
            {(llm?.state === "ready" || llm?.state === "starting") && (
              <button className="btn" onClick={() => api.serverStop("llm").catch(fail)}>{t("common.stop")}</button>
            )}
            {llm?.state === "ready" && <button className="btn" onClick={() => go("chat")}>{t("nav.chat")}</button>}
          </div>
          {llm && (
            <div className="small">
              <span className={"dot " + llm.state} />{t(("server." + llm.state) as Key)}
              {llm.engine === "strata" && <> · Strata</>}
              {llm.state === "ready" && <> · <span className="mono">http://127.0.0.1:{llm.port}/v1</span></>}
              {llm.safe_level > 0 && <> · {t("run.safe")} {llm.safe_level}</>}
              {llm.gpu_layers && <> · {t("run.offloaded")} {llm.gpu_layers[0]}/{llm.gpu_layers[1]}</>}
            </div>
          )}
          {llm?.state === "starting" && llm.engine !== "strata" && progress?.type === "progress" && (
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
              <div>{plan.summary}</div>
              <SplitBar gpu={plan.gpu_bytes} ram={plan.ram_bytes} disk={plan.disk_bytes} />
              <div className="row small">
                <span>{t("run.tps")}: <b>{plan.est_tps.toFixed(1)} tok/s</b></span>
                <span>{t("run.kv")}: {gb(plan.kv_bytes, 2)}</span>
                <span>{t("models.ctx")}: {plan.ctx / 1024}K</span>
                <span>{t("run.load")} {plan.est_load_secs.toFixed(0)}s</span>
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
