// Typed wrappers around the Tauri commands in src-tauri/src/lib.rs.
import { invoke } from "./bridge";
export { on } from "./bridge";

export type Backend = "cuda" | "rocm" | "vulkan" | "metal" | "sycl" | "cpu";
export type EngineKind = "llama" | "sd";
export type Tier = "fast" | "ok" | "slow" | "crawl";

export type ApiKind = "openai" | "anthropic";
export interface Endpoint { base_url: string; api_key?: string | null; model: string; kind?: ApiKind }
export interface NamedEndpoint extends Endpoint { name: string; provider?: string | null }

export interface Settings {
  models_dir: string;
  engines_dir: string;
  outputs_dir: string;
  hf_endpoint: string;
  hf_token: string | null;
  github_api: string;
  github_mirror_prefix: string;
  ram_reserve_bytes: number;
  vram_reserve_bytes: number;
  language: string;
  installer_language: string | null;
  server_port: number;
  image_port: number;
  lan_access: boolean;
  server_api_key: string | null;
  min_ctx: number;
  jit_api: boolean;
  lan_addresses: string[];
  web_ui: boolean;
  web_port: number;
  web_lan: boolean;
  web_addresses: string[];
  web_password: string | null;
  backend: Backend | null;
  gpu_override: { vendor: string; name: string; vram_gb: number } | null;
  custom_args: Record<string, string>;
  endpoints: NamedEndpoint[];
  agent_workspace: string | null;
}

export interface GpuInfo { vendor: string; name: string; vram_total: number; vram_free: number | null; driver: string | null; unified_memory: boolean; bandwidth_gbs: number }
export type DiskBus = "nvme" | "sata" | "usb" | "thunderbolt" | "apple" | "mmc" | "other";
export interface DiskInfo {
  mount: string; total: number; available: number; is_ssd: boolean | null;
  bus: DiskBus | null; pcie_gen: number | null; pcie_lanes: number | null; read_bps: number;
}
export interface Hardware {
  os: string; arch: string; cpu_name: string; cpu_cores: number; cpu_threads: number;
  ram_total: number; ram_available: number; ram_type: string | null; ram_bandwidth_gbs: number;
  ram_speed_mts: number | null; ram_bus_bits: number | null; ram_bandwidth_detected: boolean;
  gpus: GpuInfo[]; model_disk: DiskInfo | null; recommended_backends: Backend[];
}
export interface VramUse { name: string; used: number; total: number; spill_suspected: boolean }

export interface QuantFit { quant: string; size_bytes: number; tier: Tier; tier_label: string; est_tps: number; ctx: number; gpu_bytes: number; ram_bytes: number; disk_bytes: number; installed: boolean }
export interface CatalogEntry {
  id: string; name: string; org: string; repo: string; release: string; tags: string[];
  params_b: number; active_b: number | null; ctx_max: number; vision: boolean; quants: string[];
  description: string; fits: QuantFit[]; recommended: string | null;
  strata: StrataPack | null;
  origin?: string | null;
}
export type MacLimit = "unsupported" | "text_64gb";
export interface StrataPack { family: string; repo: string; quants: string[]; mac_limit: MacLimit | null }
export interface ManagedSet { model: string; label: string; disk_gb: number }
export interface ManagedModel { id: string; name: string; org: string; origin: string; family: string; tags: string[]; mac_limit: MacLimit | null; sets: ManagedSet[]; description: string; dir: string }

export interface Plan {
  tier: Tier; tier_label: string; ctx: number; gpu_layers: number; n_cpu_moe: number;
  gpu_bytes: number; ram_bytes: number; disk_bytes: number; kv_bytes: number;
  est_tps: number; est_load_secs: number; summary: string; gpu_note: string; ram_note: string; disk_note: string;
  args: string[]; env: [string, string][]; command: string; custom_args: string; manual_placement: boolean;
}
export interface ModelSummary { arch: string; name: string | null; quant: string; block_count: number; context_length: number; expert_count: number; expert_used_count: number; total_bytes: number }
export interface PlanPreview { plan: Plan; summary: ModelSummary; mmproj: string | null; saved_custom_args: string; strata_supported: boolean }
export type LlmEngine = "llama" | "strata" | "vllm" | "mlx";
export type PyKind = "vllm" | "mlx";
export type PyVariant = "default" | "rocm";
export interface PyInstall { kind: PyKind; variant: PyVariant; version: string; python: string; dir: string }
export interface PyEngineInfo { kind: PyKind; name: string; package: string; unsupported: "macos-arm64" | "linux" | null; installed: PyInstall | null; suggested: PyVariant }
export interface RunOptions { path: string; ctx: number | null; custom_args: string | null; vision: boolean; engine?: LlmEngine }

export interface LocalModel { name: string; path: string; files: string[]; size_bytes: number; quant: string | null; mmproj: string | null; external: boolean; strata?: boolean }
export interface HfModel { id: string; downloads: number; likes: number; lastModified: string | null; gated: unknown }
export interface RepoFile { path: string; size: number }
export interface RepoListing { quants: { quant: string; files: RepoFile[]; total_bytes: number }[]; mmproj: RepoFile[] }

export interface InstalledEngine { kind: EngineKind; version: string; backend: Backend; binary: string; dir: string; source: string }
export interface StrataInstall { version: string; launcher: string; dir: string }
export interface Engines { llama: InstalledEngine[]; sd: InstalledEngine[]; recommended: Backend[]; strata: StrataInstall | null; strata_available: boolean; strata_data_dir: string }

export type ServerState = "stopped" | "starting" | "ready" | "crashed";
export interface ServerStatus { state: ServerState; port: number; model: string | null; engine: string; safe_level: number; gpu_layers: [number, number] | null; api_key: string | null; exposed: string[]; last_crash: { problems: string[]; exit_code: number | null; tail: string[] } | null }
export interface WebStatus { running: boolean; urls: string[]; error: string | null }
export interface NetAddr { name: string; ip: string }
export interface JitStatus { running: boolean; port: number; urls: string[]; error: string | null; loaded: { name: string; port: number }[] }
export interface LoadProgress { stage: string; fraction: number; elapsed_secs: number }
export type ServerEvent =
  | { type: "status"; state: ServerState; port: number }
  | { type: "log"; line: string }
  | { type: "note"; message: string }
  | ({ type: "progress" } & LoadProgress)
  | { type: "crashed"; problems: string[]; exit_code: number | null; safe_level: number };

export interface MediaModel { id: string; name: string; kind: "image" | "video"; edit: boolean; image_to_video: boolean; hd: boolean; quants: string[]; default_quant: string; steps: number; cfg_scale: number; installed: string[] }
export interface MediaOutput { path: string; data_url: string }
export interface ImageRequest { prompt: string; negative_prompt: string; width: number; height: number; steps: number | null; cfg_scale: number | null; seed: number | null; init_image: string | null; denoising_strength: number | null; extra_images: string[] }
export interface VideoRequest { prompt: string; negative_prompt: string; width: number; height: number; frames: number; fps: number; steps: number | null; seed: number | null; init_image: string | null }

export interface Progress { job_id: string; stage: string; done: number; total: number | null; speed_bps: number | null; file_index: number; file_count: number; message: string | null }
export interface JobDone { job_id: string; ok: boolean; error: string | null; outputs: string[] }

export type ChatEvent =
  | { type: "text"; delta: string }
  | { type: "reasoning"; delta: string }
  | { type: "tool_call"; id: string; name: string; arguments: string }
  | { type: "tool_result"; id: string; name: string; content: string; ok: boolean }
  | { type: "usage"; prompt_tokens: number; completion_tokens: number; tokens_per_sec: number | null }
  | { type: "done"; reason: string }
  | { type: "plan"; steps: { step: string; status: string }[] };
export interface ApprovalRequest { id: string; tool: string; summary: string }
export type Decision = "once" | "always" | "deny";

// Chat messages in OpenAI format; content may be a string or parts (images).
export type Message = { role: string; content?: unknown; tool_calls?: unknown; tool_call_id?: string };

export const api = {
  settings: () => invoke<Settings>("get_settings"),
  saveSettings: (settings: Settings) => invoke<void>("save_settings", { settings }),
  paths: () => invoke<{ data_dir: string; version: string }>("app_paths"),
  hardware: (refresh = false) => invoke<Hardware>("get_hardware", { refresh }),
  liveVram: () => invoke<VramUse[]>("live_vram"),
  catalog: () => invoke<CatalogEntry[]>("catalog"),
  planPreview: (opts: RunOptions) => invoke<PlanPreview>("plan_preview", { opts }),
  library: () => invoke<LocalModel[]>("library_list"),
  libraryImport: (path: string) => invoke<LocalModel>("library_import", { path }),
  libraryRemove: (path: string) => invoke<void>("library_remove", { path }),
  hfSearch: (query: string) => invoke<HfModel[]>("hf_search", { query }),
  hfListing: (repo: string) => invoke<RepoListing>("hf_listing", { repo }),
  hfDownload: (repo: string, quant: string, mmproj: boolean) => invoke<string>("hf_download", { repo, quant, mmproj }),
  cancelJob: (id: string) => invoke<boolean>("cancel_job", { id }),
  engines: () => invoke<Engines>("engines"),
  engineInstall: (kind: EngineKind, backend: Backend | null) => invoke<string>("engine_install", { kind, backend }),
  engineImport: (kind: EngineKind, path: string) => invoke<InstalledEngine>("engine_import", { kind, path }),
  engineRemove: (kind: EngineKind, binary: string) => invoke<void>("engine_remove", { kind, binary }),
  serverStatus: (which: "llm" | "sd") => invoke<ServerStatus>("server_status", { which }),
  serverLogs: (which: "llm" | "sd", last = 400) => invoke<string[]>("server_logs", { which, last }),
  serverStop: (which: "llm" | "sd") => invoke<void>("server_stop", { which }),
  llmStart: (opts: RunOptions) => invoke<Plan | null>("llm_start", { opts }),
  pyEngines: () => invoke<PyEngineInfo[]>("py_engines"),
  pyEngineInstall: (kind: PyKind, variant: PyVariant | null = null) => invoke<string>("py_engine_install", { kind, variant }),
  pyEngineLatest: (kind: PyKind, variant: PyVariant | null = null) => invoke<string>("py_engine_latest", { kind, variant }),
  pyEngineRemove: (kind: PyKind) => invoke<void>("py_engine_remove", { kind }),
  pyEngineStart: (kind: PyKind, model: string, ctx: number | null) => invoke<number>("py_engine_start", { kind, model, ctx }),
  openFolder: (path: string) => invoke<void>("open_folder", { path }),
  lanAddresses: () => invoke<NetAddr[]>("lan_addresses"),
  strataInstall: () => invoke<string>("strata_install"),
  strataRemove: () => invoke<void>("strata_remove"),
  strataManaged: () => invoke<ManagedModel[]>("strata_managed"),
  strataPlan: (id: string, set: string) => invoke<string>("strata_plan", { id, set }),
  strataStartManaged: (id: string, set: string, ctx: number | null) => invoke<void>("strata_start_managed", { id, set, ctx }),
  mediaModels: () => invoke<MediaModel[]>("media_models"),
  mediaDownload: (id: string, quant: string) => invoke<string>("media_download", { id, quant }),
  sdStart: (id: string, quant: string) => invoke<void>("sd_start", { id, quant }),
  imageGenerate: (req: ImageRequest) => invoke<MediaOutput[]>("image_generate", { req }),
  videoGenerate: (req: VideoRequest) => invoke<string>("video_generate", { req }),
  readMedia: (path: string) => invoke<string>("read_media", { path }),
  chatSend: (session: string, endpoint: Endpoint | null, messages: Message[], tools: boolean) => invoke<Message[]>("chat_send", { session, endpoint, messages, tools }),
  agentSend: (session: string, endpoint: Endpoint | null, messages: Message[], workspace: string) => invoke<Message[]>("agent_send", { session, endpoint, messages, workspace }),
  chatStop: (session: string) => invoke<void>("chat_stop", { session }),
  approve: (session: string, id: string, decision: Decision) => invoke<boolean>("approval_resolve", { session, id, decision }),
  endpointModels: (endpoint: Endpoint) => invoke<string[]>("endpoint_models", { endpoint }),
  webStatus: () => invoke<WebStatus>("web_ui_status"),
  jitStatus: () => invoke<JitStatus>("jit_status"),
  webSetPassword: (password: string) => invoke<void>("web_ui_set_password", { password }),
};


export function gb(bytes: number | null | undefined, digits = 1): string {
  if (bytes == null) return "—";
  return `${(bytes / 1024 ** 3).toFixed(digits)} GB`;
}

export function speed(bps: number | null | undefined): string {
  if (!bps) return "";
  return bps > 1024 ** 2 ? `${(bps / 1024 ** 2).toFixed(1)} MB/s` : `${(bps / 1024).toFixed(0)} KB/s`;
}

export function errText(e: unknown): string {
  return typeof e === "string" ? e : e instanceof Error ? e.message : JSON.stringify(e);
}
