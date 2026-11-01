import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { useApp } from "../App";
import { useI18n } from "../i18n";
import ChatView, { PlanStep } from "./ChatView";

export default function Agent() {
  const { t } = useI18n();
  const { settings, saveSettings, fail } = useApp();
  const [plan, setPlan] = useState<PlanStep[]>([]);
  const ws = settings.agent_workspace ?? "";

  const pick = async () => {
    const p = await open({ directory: true, multiple: false });
    if (typeof p === "string") saveSettings({ ...settings, agent_workspace: p }).catch(fail);
  };

  return (
    <>
      <h1>{t("nav.agent")}</h1>
      <div className="row" style={{ marginBottom: 6 }}>
        <span className="muted small">{t("agent.workspace")}:</span>
        <span className="mono ellipsis grow" title={ws}>{ws || "—"}</span>
        <button className="btn sm" onClick={pick}>{t("agent.pick")}</button>
      </div>
      <div className="muted small" style={{ marginBottom: 10 }}>{t("agent.hint")}</div>
      {plan.length > 0 && (
        <div className="card" style={{ padding: "10px 14px", marginBottom: 10 }}>
          <h3>{t("agent.plan")}</h3>
          {plan.map((s, i) => (
            <div key={i} className="small">
              {s.status === "done" ? "✓" : s.status === "in_progress" ? "▸" : "○"} {s.step}
            </div>
          ))}
        </div>
      )}
      {/* Remount on workspace change so a conversation never spans two folders. */}
      <ChatView key={ws} mode="agent" workspace={ws} onPlan={setPlan} />
    </>
  );
}
