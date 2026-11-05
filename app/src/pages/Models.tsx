import { useEffect, useRef, useState } from "react";
import { inBrowser, pickPath, reveal } from "../bridge";
import { api, CatalogEntry, DiskInfo, driveKind, gb, HfModel, LocalModel, ManagedModel, MediaModel, RepoListing } from "../api";
import { useApp } from "../App";
import { Key, useI18n } from "../i18n";

const TAGS = ["chat", "code", "vision", "reasoning"];
// Country of the developer: the badge shows the flag and the country name.
const ORIGINS: Record<string, { zh: string; en: string }> = {
  CN: { zh: "中國", en: "China" },
  US: { zh: "美國", en: "USA" },
  FR: { zh: "法國", en: "France" },
  JP: { zh: "日本", en: "Japan" },
  KR: { zh: "韓國", en: "Korea" },
  TW: { zh: "台灣", en: "Taiwan" },
  DE: { zh: "德國", en: "Germany" },
  GB: { zh: "英國", en: "UK" },
  CA: { zh: "加拿大", en: "Canada" },
  AE: { zh: "阿聯", en: "UAE" },
};
/** Flag emoji from an ISO 3166 alpha-2 code (two regional indicator symbols). */
const flag = (code: string) => (/^[A-Z]{2}$/.test(code) ? String.fromCodePoint(...[...code].map((c) => 0x1f1e6 + c.charCodeAt(0) - 65)) : "");
const countryName = (code: string, lang: string) => (lang.startsWith("zh") ? ORIGINS[code]?.zh : ORIGINS[code]?.en) ?? code;
const originBadge = (code: string, lang: string) => `${flag(code)} ${countryName(code, lang)}`.trim();

function OriginPill({ code }: { code?: string | null }) {
  const { t, lang } = useI18n();
  if (!code) return null;
  return <span className="pill origin" title={t("models.origin_tip").replace("{country}", countryName(code, lang))}>{originBadge(code, lang)}</span>;
}

/** Country filter: a drop-down list of check boxes. None ticked shows every country. */
function CountryPicker({ all, chosen, count, onChange }: { all: string[]; chosen: string[]; count: (o: string) => number; onChange: (v: string[]) => void }) {
  const { t, lang } = useI18n();
  const ref = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (ref.current?.open && !ref.current.contains(e.target as Node)) ref.current.open = false;
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);
  const label = chosen.length === 0 ? t("models.country_all") : chosen.map((o) => originBadge(o, lang)).join("、");
  return (
    <details className="dropdown" ref={ref}>
      <summary className={"chip" + (chosen.length ? " on" : "")}>{t("models.country")}：{label} ▾</summary>
      <div className="menu">
        {all.map((o) => (
          <label key={o} className="check">
            <input type="checkbox" checked={chosen.includes(o)} onChange={() => onChange(chosen.includes(o) ? chosen.filter((x) => x !== o) : [...chosen, o])} />
            {originBadge(o, lang)} <span className="muted">{count(o)}</span>
          </label>
        ))}
      </div>
    </details>
  );
}
const tagLabel = (t: (k: Key) => string, x: string) => (TAGS.includes(x) ? t(("tag." + x) as Key) : x);

type Tab = "recommended" | "reasoning" | "installed" | "media" | "search";

export default function Models() {
  const { t } = useI18n();
  const [tab, setTab] = useState<Tab>("recommended");
  return (
    <>
      <h1>{t("nav.models")}</h1>
      <div className="tabs">
        {(["recommended", "reasoning", "installed", "media", "search"] as Tab[]).map((x) => (
          <button key={x} className={"tab" + (x === tab ? " on" : "")} onClick={() => setTab(x)}>{t(`models.${x}`)}</button>
        ))}
      </div>
      {tab === "recommended" && <Recommended />}
      {tab === "reasoning" && <Recommended key="reasoning" only="reasoning" />}
      {tab === "installed" && <Installed />}
      {tab === "media" && <Media />}
      {tab === "search" && <Search />}
    </>
  );
}

/** Catalog cards; `only` limits them to models with that tag. */
function Recommended({ only }: { only?: string }) {
  const { t } = useI18n();
  const { track, fail } = useApp();
  const [list, setList] = useState<CatalogEntry[] | null>(null);
  const [choice, setChoice] = useState<Record<string, string>>({});
  const [vision, setVision] = useState(true);
  const [tags, setTags] = useState<string[]>([]);
  const [strataOnly, setStrataOnly] = useState(false);
  const [origins, setOrigins] = useState<string[]>([]);
  const [managed, setManaged] = useState<ManagedModel[]>([]);
  const [hasStrata, setHasStrata] = useState(false);
  const keep = <T extends { tags: string[] }>(xs: T[]) => (only ? xs.filter((m) => m.tags.includes(only)) : xs);
  const load = () => api.catalog().then((l) => setList(keep(l)), fail);
  useEffect(() => {
    load();
    api.strataManaged().then((m) => setManaged(keep(m)), () => {});
    api.engines().then((e) => setHasStrata(!!e.strata), () => {});
  }, []);
  if (!list) return <p className="muted">{t("common.loading")}</p>;
  const toggle = (x: string) => setTags(tags.includes(x) ? tags.filter((y) => y !== x) : [...tags, x]);
  // Countries are alternatives (USA or China); capability tags must all match.
  const fromOrigin = (o?: string | null) => origins.length === 0 || (!!o && origins.includes(o));
  const shown = list.filter((m) => tags.every((x) => m.tags.includes(x)) && (!strataOnly || !!m.strata) && fromOrigin(m.origin));
  const shownManaged = managed.filter((m) => tags.every((x) => m.tags.includes(x)) && fromOrigin(m.origin));
  const allOrigins = [...new Set([...list.map((m) => m.origin), ...managed.map((m) => m.origin)].filter((o): o is string => !!o))];
  const byOrigin = (o: string) => list.filter((m) => m.origin === o).length + managed.filter((m) => m.origin === o).length;
  allOrigins.sort((a, b) => byOrigin(b) - byOrigin(a));
  const count = (x: string) => list.filter((m) => m.tags.includes(x)).length + managed.filter((m) => m.tags.includes(x)).length;
  return (
    <>
      <div className="chips" style={{ marginBottom: 10 }}>
        <span className="muted small">{t("models.filter")}</span>
        <button className={"chip" + (tags.length === 0 && !strataOnly && origins.length === 0 ? " on" : "")} onClick={() => { setTags([]); setStrataOnly(false); setOrigins([]); }}>{t("models.filter_all")}</button>
        {TAGS.filter((x) => x !== only).map((x) => (
          <button key={x} className={"chip" + (tags.includes(x) ? " on" : "")} onClick={() => toggle(x)}>
            {tagLabel(t, x)} <span className="muted">{count(x)}</span>
          </button>
        ))}
        <button className={"chip" + (strataOnly ? " on" : "")} onClick={() => setStrataOnly(!strataOnly)} title={t("models.strata_tip")}>
          Strata <span className="muted">{list.filter((m) => m.strata).length + managed.length}</span>
        </button>
        <CountryPicker all={allOrigins} chosen={origins} count={byOrigin} onChange={setOrigins} />
      </div>
      <label className="check" style={{ marginBottom: 12 }}>
        <input type="checkbox" checked={vision} onChange={(e) => setVision(e.target.checked)} />
        {t("models.with_vision")}
      </label>
      {shown.length + shownManaged.length === 0 && <p className="muted">{t("models.filter_none")}</p>}
      <div className="grid">
        {shown.map((m) => {
          const q = choice[m.id] ?? m.recommended ?? m.quants[0];
          const fit = m.fits.find((f) => f.quant === q);
          const sp = m.strata;
          const sameRepo = sp?.repo.toLowerCase() === m.repo.toLowerCase();
          const spOk = (x: string) => !!sp && sameRepo && sp.quants.some((y) => y.toLowerCase() === x.toLowerCase());
          const spQuant = sp ? (sp.quants.find((y) => y.toLowerCase() === q.toLowerCase()) ?? sp.quants[0]) : "";
          return (
            <div key={m.id} className="card stack">
              <div>
                <div className="cardhead">
                  <h3>{m.name}</h3>
                  <span className="pillcol">
                    <OriginPill code={m.origin} />
                    {sp && <span className="pill strata" title={t("models.strata_tip")}>Strata</span>}
                    {fit && <span className={"pill " + fit.tier}>{fit.tier_label}</span>}
                  </span>
                </div>
                <div className="muted small">{m.org} · {m.release} · {m.params_b}B{m.active_b ? ` (${m.active_b}B active)` : ""}</div>
              </div>
              <div className="small">{m.description}</div>
              <div className="row">{m.tags.map((x) => <span key={x} className="pill">{tagLabel(t, x)}</span>)}</div>
              <div className="row">
                <select value={q} onChange={(e) => setChoice({ ...choice, [m.id]: e.target.value })}>
                  {m.fits.map((f) => (
                    <option key={f.quant} value={f.quant}>
                      {f.quant} · {gb(f.size_bytes)}{f.quant === m.recommended ? " ★" : ""}{f.installed ? " ✓" : ""}{spOk(f.quant) ? ` · Strata${sp?.mac_limit ? `（${t(`models.mac_short_${sp.mac_limit}` as Key)}）` : ""}` : ""}
                    </option>
                  ))}
                </select>
                {fit && <span className="muted small">{t("models.est")} {fit.est_tps.toFixed(0)} tok/s · {t("models.ctx")} {fit.ctx / 1024}K</span>}
              </div>
              {fit && <SplitBar gpu={fit.gpu_bytes} ram={fit.ram_bytes} disk={fit.disk_bytes} />}
              {sp && sameRepo && (
                <div className="small muted">
                  {spOk(q) ? t("models.strata_quant_ok") : <>{t("models.strata_quants")} {sp.quants.join(", ")}</>}
                </div>
              )}
              {sp && !sameRepo && (
                <div className="note small stack" style={{ gap: 6 }}>
                  <div>{t("models.strata_other_repo")} <span className="mono">{sp.repo}</span> ({sp.quants.join(", ")})</div>
                  {sp.mac_limit && <div className="note warn small">{t(`models.mac_warn_${sp.mac_limit}` as Key)}</div>}
                  <div>
                    <button className="btn sm" onClick={() => api.hfDownload(sp.repo, spQuant, vision && m.vision).then((id) => track(id, `${m.name} MTP ${spQuant}`, () => load()), fail)}>
                      {t("models.strata_download")} ({spQuant})
                    </button>
                  </div>
                </div>
              )}
              <div className="row">
                {fit?.installed ? (
                  <span className="pill fast">{t("common.installed")}</span>
                ) : (
                  <button
                    className="btn primary"
                    onClick={() => api.hfDownload(m.repo, q, vision && m.vision).then((id) => track(id, `${m.name} ${q}`, () => load()), fail)}
                  >
                    {t("common.download")} {fit ? gb(fit.size_bytes) : ""}
                  </button>
                )}
              </div>
            </div>
          );
        })}
        {shownManaged.map((m) => <ManagedCard key={m.id} m={m} hasStrata={hasStrata} />)}
      </div>
    </>
  );
}

function ManagedCard({ m, hasStrata }: { m: ManagedModel; hasStrata: boolean }) {
  const { t } = useI18n();
  const { fail, go } = useApp();
  const [set, setSet] = useState(m.sets[1]?.model ?? m.sets[0].model);
  const [plan, setPlan] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const cur = m.sets.find((x) => x.model === set)!;
  const preview = () => {
    setBusy(true);
    setPlan(null);
    api.strataPlan(m.id, set).then(setPlan, fail).finally(() => setBusy(false));
  };
  const start = () => {
    if (!confirm(t("models.strata_confirm").replace("{gb}", String(cur.disk_gb)))) return;
    api.strataStartManaged(m.id, set, null).catch(fail);
    go("run");
  };
  return (
    <div className="card stack">
      <div>
        <div className="cardhead">
          <h3>{m.name}</h3>
          <span className="pillcol">
            <OriginPill code={m.origin} />
            <span className="pill strata" title={t("models.strata_tip")}>{t("models.strata_only")}</span>
          </span>
        </div>
        <div className="muted small">{m.org}</div>
      </div>
      <div className="small">{m.description}</div>
      <div className="row">{m.tags.map((x) => <span key={x} className="pill">{tagLabel(t, x)}</span>)}</div>
      <div className="row">
        <select value={set} onChange={(e) => { setSet(e.target.value); setPlan(null); }}>
          {m.sets.map((x) => <option key={x.model} value={x.model}>{x.label} · {x.disk_gb} GB{m.mac_limit ? ` · ${t(`models.mac_short_${m.mac_limit}` as Key)}` : ""}</option>)}
        </select>
      </div>
      {m.mac_limit && <div className="note warn small">{t(`models.mac_warn_${m.mac_limit}` as Key)}</div>}
      <div className="muted small ellipsis" title={m.dir}>{t("models.strata_dir")}: <span className="mono">{m.dir}</span></div>
      {hasStrata ? (
        <div className="row">
          <button className="btn" disabled={busy} onClick={preview}>{busy ? t("common.loading") : t("models.strata_preview")}</button>
          <button className="btn primary" onClick={start}>{t("models.strata_start")}</button>
        </div>
      ) : (
        <div className="row">
          <span className="muted small">{t("models.strata_need")}</span>
          <button className="btn sm" onClick={() => go("settings")}>{t("nav.settings")}</button>
        </div>
      )}
      {plan && <pre className="log" style={{ maxHeight: 260 }}>{plan}</pre>}
    </div>
  );
}

// One hardware request shared by every bar, redone when the models folder moves.
let diskCache: { dir: string; disk: Promise<DiskInfo | null> } | null = null;

function useModelDisk(): DiskInfo | null {
  const { settings } = useApp();
  const [disk, setDisk] = useState<DiskInfo | null>(null);
  useEffect(() => {
    if (diskCache?.dir !== settings.models_dir) {
      diskCache = { dir: settings.models_dir, disk: api.hardware().then((h) => h.model_disk, () => null) };
    }
    let live = true;
    diskCache.disk.then((d) => live && setDisk(d));
    return () => { live = false; };
  }, [settings.models_dir]);
  return disk;
}

export function SplitBar({ gpu, ram, disk }: { gpu: number; ram: number; disk: number }) {
  const { t } = useI18n();
  const drive = driveKind(useModelDisk()) || t("run.disk");
  const total = Math.max(1, gpu + ram + disk);
  return (
    <div className="stack" style={{ gap: 4 }}>
      <div className="split">
        <div className="g" style={{ width: `${(gpu / total) * 100}%` }} />
        <div className="r" style={{ width: `${(ram / total) * 100}%` }} />
        <div className="d" style={{ width: `${(disk / total) * 100}%` }} />
      </div>
      <div className="legend small">
        <span><i className="g" />{t("run.gpu")} {gb(gpu)}</span>
        <span><i className="r" />{t("run.ram")} {gb(ram)}</span>
        <span><i className="d" />{drive} {gb(disk)}</span>
      </div>
    </div>
  );
}

function Installed() {
  const { t } = useI18n();
  const { go, fail } = useApp();
  const [list, setList] = useState<LocalModel[] | null>(null);
  const load = () => api.library().then(setList, fail);
  useEffect(() => { load(); }, []);
  const importFile = async () => {
    const p = await pickPath({ filters: [{ name: "GGUF", extensions: ["gguf"] }] }, t("common.enter_path"));
    if (p) api.libraryImport(p).then(load, fail);
  };
  const remove = (m: LocalModel) => {
    if (!m.external && !confirm(t("models.remove_confirm"))) return;
    api.libraryRemove(m.path).then(load, fail);
  };
  return (
    <>
      <div className="row" style={{ marginBottom: 12 }}>
        <button className="btn" onClick={importFile}>{t("models.import")}</button>
        <span className="muted small">{t("models.import_hint")}</span>
      </div>
      {list && list.length === 0 && <p className="muted">{t("models.empty")}</p>}
      {list && list.length > 0 && (
        <div className="card">
          <table>
            <tbody>
              {list.map((m) => (
                <tr key={m.path}>
                  <td>
                    <div>{m.name}</div>
                    <div className="muted small ellipsis" style={{ maxWidth: 520 }} title={m.path}>{m.path}</div>
                  </td>
                  <td>{m.quant ?? ""}</td>
                  <td>{gb(m.size_bytes)}</td>
                  <td>
                    {m.strata && <span className="pill strata" title={t("models.strata_tip")}>Strata</span>}{" "}
                    {m.mmproj && <span className="pill">vision</span>} {m.external && <span className="pill">{t("models.external")}</span>}
                  </td>
                  <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                    <button className="btn sm primary" onClick={() => go("run", m.path)}>{t("models.run")}</button>{" "}
                    {!inBrowser && <><button className="btn sm" onClick={() => reveal(m.path)}>{t("common.open_folder")}</button>{" "}</>}
                    <button className="btn sm danger" onClick={() => remove(m)}>{t("common.remove")}</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function Media() {
  const { t } = useI18n();
  const { track, fail, go } = useApp();
  const [list, setList] = useState<MediaModel[] | null>(null);
  const [choice, setChoice] = useState<Record<string, string>>({});
  const load = () => api.mediaModels().then(setList, fail);
  useEffect(() => { load(); }, []);
  if (!list) return <p className="muted">{t("common.loading")}</p>;
  return (
    <div className="grid">
      {list.map((m) => {
        const q = choice[m.id] ?? m.default_quant;
        const have = m.installed.includes(q);
        return (
          <div key={m.id} className="card stack">
            <div className="row spread">
              <h3 style={{ margin: 0 }}>{m.name}</h3>
              <span className="pill">{m.kind === "image" ? t("nav.image") : t("nav.video")}</span>
            </div>
            <div className="row">
              {m.edit && <span className="pill">edit</span>}
              {m.image_to_video && <span className="pill">image→video</span>}
              {m.hd && <span className="pill">720P</span>}
            </div>
            <div className="row">
              <select value={q} onChange={(e) => setChoice({ ...choice, [m.id]: e.target.value })}>
                {m.quants.map((x) => <option key={x} value={x}>{x}{m.installed.includes(x) ? " ✓" : ""}</option>)}
              </select>
              {have ? (
                <button className="btn" onClick={() => go(m.kind, m.id)}>{m.kind === "image" ? t("nav.image") : t("nav.video")}</button>
              ) : (
                <button className="btn primary" onClick={() => api.mediaDownload(m.id, q).then((id) => track(id, `${m.name} ${q}`, () => load()), fail)}>{t("common.download")}</button>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Search() {
  const { t } = useI18n();
  const { track, fail } = useApp();
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<HfModel[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const [listing, setListing] = useState<RepoListing | null>(null);
  const [vision, setVision] = useState(true);

  const search = () => {
    if (!q.trim()) return;
    setBusy(true);
    api.hfSearch(q.trim()).then(setResults, fail).finally(() => setBusy(false));
  };
  const expand = (repo: string) => {
    if (open === repo) return setOpen(null);
    setOpen(repo);
    setListing(null);
    api.hfListing(repo).then(setListing, fail);
  };
  return (
    <>
      <div className="row" style={{ marginBottom: 12 }}>
        <input type="text" className="grow" style={{ maxWidth: 420 }} placeholder={t("models.search_ph")} value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && search()} />
        <button className="btn primary" disabled={busy} onClick={search}>{t("models.search")}</button>
        <label className="check small"><input type="checkbox" checked={vision} onChange={(e) => setVision(e.target.checked)} />{t("models.with_vision")}</label>
      </div>
      {results.map((r) => (
        <div key={r.id} className="card">
          <div className="row spread" style={{ cursor: "pointer" }} onClick={() => expand(r.id)}>
            <b>{r.id}</b>
            <span className="muted small">↓ {r.downloads.toLocaleString()} · ♥ {r.likes}</span>
          </div>
          {open === r.id && (
            <div style={{ marginTop: 10 }}>
              {!listing ? <span className="muted">{t("common.loading")}</span> : (
                <table>
                  <tbody>
                    {listing.quants.map((g) => (
                      <tr key={g.quant}>
                        <td>{g.quant}</td>
                        <td>{gb(g.total_bytes)}</td>
                        <td className="muted small">{g.files.length > 1 ? `${g.files.length} files` : g.files[0]?.path}</td>
                        <td style={{ textAlign: "right" }}>
                          <button className="btn sm primary" onClick={() => api.hfDownload(r.id, g.quant, vision && listing.mmproj.length > 0).then((id) => track(id, `${r.id} ${g.quant}`), fail)}>{t("common.download")}</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </div>
      ))}
    </>
  );
}
