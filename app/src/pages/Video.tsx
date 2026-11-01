import { useEffect, useState } from "react";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { api, MediaModel } from "../api";
import { useApp } from "../App";
import { useI18n } from "../i18n";
import { MediaLoader, readImage } from "./Image";

// Frame counts the video models accept (4n+1); mirrors diffusion::frames_for.
function framesFor(seconds: number, fps: number): number {
  const table = fps >= 24 ? [25, 25, 49, 73, 121] : [17, 17, 33, 49, 81];
  return table[Math.min(4, Math.max(0, seconds))];
}

export default function VideoPage() {
  const { t } = useI18n();
  const { sd, fail, track, pageArg } = useApp();
  const [models, setModels] = useState<MediaModel[]>([]);
  const [id, setId] = useState(pageArg ?? "");
  const [prompt, setPrompt] = useState("");
  const [seconds, setSeconds] = useState(2);
  const [portrait, setPortrait] = useState(false);
  const [hd, setHd] = useState(false);
  const [start, setStart] = useState<string | null>(null);
  const [videos, setVideos] = useState<{ path: string; url: string }[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.mediaModels().then((l) => {
      setModels(l);
      if (!id) setId((l.find((x) => x.kind === "video" && x.installed.length) ?? l.find((x) => x.kind === "video"))?.id ?? "");
    }, fail);
  }, []);
  const m = models.find((x) => x.id === id);
  const fps = 24;

  const generate = () => {
    const [w, h] = hd ? [1280, 704] : [832, 480];
    const [width, height] = portrait ? [h, w] : [w, h];
    setBusy(true);
    api
      .videoGenerate({ prompt, negative_prompt: "", width, height, frames: framesFor(seconds, fps), fps, steps: null, seed: null, init_image: start })
      .then((job) =>
        track(job, `${m?.name ?? "video"} · ${seconds}s`, async (d) => {
          setBusy(false);
          for (const p of d.outputs) {
            const url = await api.readMedia(p).catch(() => "");
            setVideos((v) => [{ path: p, url }, ...v]);
          }
        }),
      )
      .catch((e) => {
        fail(e);
        setBusy(false);
      });
  };

  return (
    <>
      <h1>{t("nav.video")}</h1>
      <div className="two">
        <div className="card stack">
          <MediaLoader kind="video" models={models} id={id} setId={setId} />
          <label className="field"><span>{t("image.prompt")}</span><textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} /></label>
          <div className="row">
            <label className="field">
              <span>{t("video.seconds")}</span>
              <select value={seconds} onChange={(e) => setSeconds(Number(e.target.value))}>{[1, 2, 3, 5].map((s) => <option key={s} value={s}>{s}</option>)}</select>
            </label>
            <label className="check"><input type="checkbox" checked={portrait} onChange={(e) => setPortrait(e.target.checked)} />{t("video.portrait")}</label>
            {m?.hd && <label className="check"><input type="checkbox" checked={hd} onChange={(e) => setHd(e.target.checked)} />{t("video.hd")}</label>}
          </div>
          <label className="field">
            <span>{t("video.start_image")}{m?.image_to_video ? " *" : ""}</span>
            <input type="file" accept="image/*" onChange={async (e) => { const f = e.target.files?.[0]; setStart(f ? await readImage(f) : null); }} />
          </label>
          {start && <img src={start} style={{ height: 64, borderRadius: 4, alignSelf: "start" }} />}
          <button className="btn primary" disabled={busy || sd?.state !== "ready" || !prompt.trim() || (!!m?.image_to_video && !start)} onClick={generate}>
            {t("image.generate")}
          </button>
        </div>
        <div className="card">
          <h2>{t("image.outputs")}</h2>
          <div className="gallery">
            {videos.map((v) => (
              <div key={v.path} className="stack" style={{ gap: 4 }}>
                <video src={v.url} controls loop />
                <button className="btn sm" onClick={() => revealItemInDir(v.path)}>{t("common.open_folder")}</button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
