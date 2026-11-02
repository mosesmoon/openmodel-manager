import { ApiKind } from "./api";

/** Presets for Settings → AI service providers. Models come from each
 *  provider's model list (↻); only Anthropic gets a default. */
export interface ProviderPreset { id: string; name: string; kind: ApiKind; base_url: string; model: string; key_url: string }

export const PROVIDERS: ProviderPreset[] = [
  { id: "anthropic", name: "Anthropic (Claude)", kind: "anthropic", base_url: "https://api.anthropic.com/v1", model: "claude-opus-5-5", key_url: "https://console.anthropic.com/settings/keys" },
  { id: "openai", name: "OpenAI", kind: "openai", base_url: "https://api.openai.com/v1", model: "", key_url: "https://platform.openai.com/api-keys" },
  { id: "google", name: "Google AI Studio (Gemini)", kind: "openai", base_url: "https://generativelanguage.googleapis.com/v1beta/openai", model: "", key_url: "https://aistudio.google.com/apikey" },
  { id: "xai", name: "xAI (Grok)", kind: "openai", base_url: "https://api.x.ai/v1", model: "", key_url: "https://console.x.ai" },
  { id: "nvidia", name: "NVIDIA NIM", kind: "openai", base_url: "https://integrate.api.nvidia.com/v1", model: "", key_url: "https://build.nvidia.com" },
  { id: "custom", name: "OpenAI-compatible", kind: "openai", base_url: "", model: "", key_url: "" },
];

export const findProvider = (id?: string | null) => PROVIDERS.find((p) => p.id === (id ?? "custom")) ?? PROVIDERS[PROVIDERS.length - 1];
