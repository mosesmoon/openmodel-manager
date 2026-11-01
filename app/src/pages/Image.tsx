import { useEffect, useState } from "react";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { api, MediaModel, MediaOutput } from "../api";
import { useApp } from "../App";
import { Key, useI18n } from "../i18n";

const ASPECTS: Record<string, [number, number]> = { "1:1": [1, 1], "3:4": [3, 4], "4:3": [4, 3], "9:16": [9, 16], "16:9": [16, 9] };

/** About base×base pixels at the given aspect, multiples of 16. */
export function sizeFor(aspect: string, base: number): [number, number] {
  const [a, b] = ASPECTS[aspect] ?? [1, 1];
  const scale = Math.sqrt((base * base) / (a * b));
  const r = (x: number) => Math.max(256, Math.round((x * scale) / 16) * 16);
  return [r(a), r(b)];
}

export function readImage(f: File): Promise<string> {
  return new Promise((ok, bad) => {
    const r = new FileReader();
    r.onload = () => ok(String(r.result));
    r.onerror = () => bad(r.error);
    r.readAsDataURL(f);
  });
}

/** Model picker + load button shared by the image and video pages. */
export function MediaLoader({ kind, models, id, setId }: { kind: "image" | "video"; models: MediaModel[]; id: string; setId: (s: string) => void }) {
  const { t } = useI18n();
  const { sd, fail, go } = useApp();
  const [loading, setLoading] = useState(false);
  const m = models.find((x) => x.id === id);
  const quant = m?.installed[0];
  const load = () => {
    if (!m || !quant) return;
    setLoading(true);
    api.sdStart(m.id, quant).catch(fail).finally(() => setLoading(false));
  };
  const state = loading ? "starting" : sd?.state ?? "stopped";
  return (
    <div className="stack">
      <label className="field">
        <span>{t("image.model")}</span>
        <select value={id} onChange={(e) => setId(e.target.value)}>
          {models.filter((x) => x.kind === kind).map((x) => (
            <option key={x.id} value={x.id}>{x.name}{x.installed.length ? " ✓" : ""}</option>
          ))}
        </select>
      </label>
      {m && !quant && (
        <div className="note warn row">
          <span className="grow">{t("image.not_installed")}</span>
          <button className="btn sm" onClick={() => go("models")}>{t("nav.models")}</button>
        </div>
      )}
      <div className="row">
        <button className="btn" disabled={!quant || loading} onClick={load}>{t("image.load")}</button>
        <span className="small"><span className={"dot " + state} />{t(("server." + state) as Key)}</span>
      </div>
    </div>
  );
}

export default function ImagePage() {
  const { t } = useI18n();
  const { sd, fail, pageArg } = useApp();
  const [models, setModels] = useState<MediaModel[]>([]);
  const [id, setId] = useState(pageArg ?? "");
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [aspect, setAspect] = useState("1:1");
  const [steps, setSteps] = useState<number | "">("");
  const [seed, setSeed] = useState("");
  const [ref, setRef] = useState<string | null>(null);
  const [strength, setStrength] = useState(0.75);
  const [busy, setBusy] = useState(false);
  const [outputs, setOutputs] = useState<MediaOutput[]>([]);

  useEffect(() => {
    api.mediaModels().then((l) => {
      setModels(l);
      if (!id) setId((l.find((x) => x.kind === "image" && x.installed.length) ?? l.find((x) => x.kind === "image"))?.id ?? "");
    }, fail);
  }, []);
  const m = models.find((x) => x.id === id);

  const generate = () => {
    const [width, height] = sizeFor(aspect, 1024);
    // Edit models take the picture as a reference; others do img2img.
    const edit = !!m?.edit;
    setBusy(true);
    api
      .imageGenerate({
        prompt,
        negative_prompt: negative,
        width,
        height,
        steps: steps === "" ? null : steps,
        cfg_scale: null,
        seed: seed.trim() ? Number(seed) : null,
        init_image: ref && !edit ? ref : null,
        denoising_strength: ref && !edit ? strength : null,
        extra_images: ref && edit ? [ref] : [],
      })
      .then((o) => setOutputs((x) => [...o, ...x]), fail)
      .finally(() => setBusy(false));
  };

  return (
    <>
      <h1>{t("nav.image")}</h1>
      <div className="two">
        <div className="card stack">
          <MediaLoader kind="image" models={models} id={id} setId={setId} />
          <label className="field"><span>{t("image.prompt")}</span><textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} /></label>
          <label className="field"><span>{t("image.negative")}</span><input type="text" value={negative} onChange={(e) => setNegative(e.target.value)} /></label>
          <div className="row">
            <label className="field">
              <span>{t("image.aspect")}</span>
              <select value={aspect} onChange={(e) => setAspect(e.target.value)}>{Object.keys(ASPECTS).map((a) => <option key={a}>{a}</option>)}</select>
            </label>
            <label className="field" style={{ width: 90 }}>
              <span>{t("image.steps")}</span>
              <input type="number" min={1} max={150} placeholder={String(m?.steps ?? "")} value={steps} onChange={(e) => setSteps(e.target.value ? Number(e.target.value) : "")} />
            </label>
            <label className="field grow">
              <span>{t("image.seed")}</span>
              <input type="text" value={seed} onChange={(e) => setSeed(e.target.value.replace(/[^\d-]/g, ""))} />
            </label>
          </div>
          <label className="field">
            <span>{t("image.ref")}</span>
            <input type="file" accept="image/*" onChange={async (e) => { const f = e.target.files?.[0]; setRef(f ? await readImage(f) : null); }} />
          </label>
          {ref && (
            <div className="row">
              <img src={ref} style={{ height: 64, borderRadius: 4 }} />
              {!m?.edit && (
                <label className="field grow">
                  <span>{t("image.strength")} {strength.toFixed(2)}</span>
                  <input type="range" min={0.1} max={1} step={0.05} value={strength} onChange={(e) => setStrength(Number(e.target.value))} />
                </label>
              )}
              <button className="btn sm" onClick={() => setRef(null)}>×</button>
            </div>
          )}
          <button className="btn primary" disabled={busy || sd?.state !== "ready" || !prompt.trim()} onClick={generate}>
            {busy ? t("common.loading") : t("image.generate")}
          </button>
        </div>
        <div className="card">
          <h2>{t("image.outputs")}</h2>
          <div className="gallery">
            {outputs.map((o) => <img key={o.path} src={o.data_url} title={o.path} onClick={() => revealItemInDir(o.path)} style={{ cursor: "pointer" }} />)}
          </div>
        </div>
      </div>
    </>
  );
}
