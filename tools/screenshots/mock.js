// Stand-in for the Tauri IPC, so the built frontend runs in a plain browser
// page and shows fake data. Loaded before the app (capture.mjs injects it into
// index.html). Every command in app/src/api.ts needs an entry in `handlers`
// below; an unknown command logs `unmocked <cmd>` and fails the capture run.
//
// Data comes from fixtures/hw.json, fixtures/data.json and the real catalog
// (served as /__mock/catalog.json). Nothing here touches the machine it runs on.
(() => {
  "use strict";
  const GIB = 1024 ** 3;
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const clone = (x) => (x === undefined ? null : JSON.parse(JSON.stringify(x)));
  const getJson = (path) => fetch(path).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${path}: HTTP ${r.status}`))));

  // ---- Tauri internals the JS API reaches for ----
  const callbacks = new Map();
  const listeners = new Map(); // event -> Map(eventId -> callback id)
  let nextId = 1;

  function transformCallback(cb, once) {
    const id = nextId++;
    callbacks.set(id, (data) => {
      if (once) callbacks.delete(id);
      return cb && cb(data);
    });
    return id;
  }

  function emit(event, payload) {
    for (const [eventId, cbId] of listeners.get(event) ?? []) callbacks.get(cbId)?.({ event, id: eventId, payload });
  }

  // ---- State ----
  let hw, data, catalog;
  const state = {
    settings: null,
    llm: null,
    sd: null,
    pending: new Map(), // agent session -> resolver, until the approval is answered
  };
  const ready = Promise.all([getJson("/__mock/hw.json"), getJson("/__mock/data.json"), getJson("/__mock/catalog.json")]).then(([h, d, c]) => {
    hw = h;
    data = d;
    catalog = c.models;
    state.settings = clone(d.settings);
    const first = d.library[0];
    state.llm = status("ready", { model: first.path, gpu_layers: [49, 49] });
    state.sd = status("stopped", { port: d.settings.image_port, engine: "sd" });
  });

  function status(st, extra = {}) {
    return { state: st, port: 8080, model: null, engine: "llama", safe_level: 0, gpu_layers: null, api_key: null, exposed: [], last_crash: null, ...extra };
  }
  const lang = () => (state.settings.language === "en" ? "en" : "zh-TW");
  const tr = (zh, en) => (lang() === "en" ? en : zh);

  // ---- Fit estimates for the catalog (rough, only has to look plausible) ----
  const TIER_LABEL = { fast: ["流暢", "Fast"], ok: ["可用", "OK"], slow: ["偏慢", "Slow"], crawl: ["非常慢", "Very slow"] };
  const PREFERRED = ["UD-Q4_K_XL", "Q4_K_M", "UD-Q4_K_M", "UD-Q3_K_XL", "UD-IQ4_XS", "IQ3_S", "Q2_0", "UD-Q2_K_XL"];

  function placement(bytes) {
    const gpu = hw.gpus[0];
    const gpuCap = Math.max(0, gpu.vram_total - state.settings.vram_reserve_bytes);
    const ramCap = Math.max(0, hw.ram_available - state.settings.ram_reserve_bytes);
    const g = Math.min(bytes, gpuCap);
    const r = Math.min(bytes - g, ramCap);
    return { gpu_bytes: g, ram_bytes: r, disk_bytes: bytes - g - r };
  }

  function kvBytes(m, ctx) {
    return (m.kv_layers ?? m.layers ?? 32) * (m.kv_heads ?? 8) * (m.head_dim ?? 128) * 2 * ctx;
  }

  function fit(m, quant, ctx) {
    const size = (m.size_gb?.[quant] ?? (m.params_b * 4.8) / 8) * GIB;
    const kv = kvBytes(m, ctx);
    const p = placement(size + kv);
    const total = size + kv;
    // Per token the weights that are read: all of them, or just the active experts.
    const perTok = m.active_b ? size * (m.active_b / m.params_b) : size;
    const secs =
      (((p.gpu_bytes / total) * perTok) / (hw.gpus[0].bandwidth_gbs * 1e9)) +
      (((p.ram_bytes / total) * perTok) / (hw.ram_bandwidth_gbs * 1e9)) +
      (((p.disk_bytes / total) * perTok) / hw.model_disk.read_bps);
    const tps = Math.min(160, 0.6 / secs);
    const tier = tps >= 25 ? "fast" : tps >= 10 ? "ok" : tps >= 3 ? "slow" : "crawl";
    return { size, kv, tps, tier, ...p };
  }

  function quantFit(m, quant) {
    const f = fit(m, quant, 32768);
    return {
      quant,
      size_bytes: Math.round(f.size),
      tier: f.tier,
      tier_label: tr(...TIER_LABEL[f.tier]),
      est_tps: f.tps,
      ctx: 32768,
      gpu_bytes: Math.round(f.gpu_bytes),
      ram_bytes: Math.round(f.ram_bytes),
      disk_bytes: Math.round(f.disk_bytes),
      installed: data.library.some((l) => l.catalog === m.id && l.quant === quant),
    };
  }

  function catalogEntry(m) {
    const fits = m.quants.map((q) => quantFit(m, q));
    const pick = PREFERRED.map((q) => fits.find((f) => f.quant === q && f.disk_bytes === 0)).find(Boolean);
    const smallest = fits.reduce((a, b) => (a.size_bytes <= b.size_bytes ? a : b));
    const pack = data.strata_packs[m.id];
    return {
      id: m.id, name: m.name, org: m.org, repo: m.repo, release: m.release, tags: m.tags,
      params_b: m.params_b, active_b: m.active_b ?? null, ctx_max: m.ctx_max, vision: !!m.vision, quants: m.quants,
      description: m.desc?.[lang()] ?? m.desc?.en ?? "",
      fits, recommended: (pick ?? smallest).quant,
      strata: pack ? { repo: m.repo, ...pack } : null,
      origin: m.origin ?? null,
    };
  }

  function planPreview(opts) {
    const lib = data.library.find((l) => l.path === opts.path) ?? data.library[0];
    const m = catalog.find((x) => x.id === lib.catalog);
    const ctx = Math.min(opts.ctx ?? 65536, m.ctx_max);
    const f = fit(m, lib.quant, ctx);
    const layers = m.layers ?? 32;
    const moe = !!m.experts && f.ram_bytes > 0;
    const cpuMoe = moe ? Math.max(1, Math.round((layers * f.ram_bytes) / (f.gpu_bytes + f.ram_bytes))) : 0;
    const custom = opts.custom_args ?? "";
    const args = ["-m", lib.path, "-c", String(ctx), "-ngl", "99", ...(moe ? ["--n-cpu-moe", String(cpuMoe)] : []), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--host", "127.0.0.1", "--port", String(state.settings.server_port), "--jinja"];
    return {
      plan: {
        tier: f.tier, tier_label: tr(...TIER_LABEL[f.tier]), ctx, gpu_layers: layers, n_cpu_moe: cpuMoe,
        gpu_bytes: Math.round(f.gpu_bytes), ram_bytes: Math.round(f.ram_bytes), disk_bytes: Math.round(f.disk_bytes), kv_bytes: Math.round(f.kv),
        est_tps: f.tps, est_load_secs: f.size / hw.model_disk.read_bps + 3,
        summary: moe
          ? tr(`把 ${cpuMoe} 層的專家權重放到記憶體，其餘放在 GPU。`, `Experts of ${cpuMoe} layers sit in RAM; the rest is on the GPU.`)
          : tr("整個模型放得進 GPU。", "The whole model fits on the GPU."),
        gpu_note: tr("GPU：注意力層、共用權重與 KV 快取", "GPU: attention layers, shared weights and the KV cache"),
        ram_note: moe ? tr(`記憶體：${cpuMoe} 層的 MoE 專家`, `RAM: MoE experts of ${cpuMoe} layers`) : "",
        disk_note: "",
        args, env: [], command: args.join(" ") + (custom ? " " + custom : ""), custom_args: custom, manual_placement: false,
      },
      summary: {
        arch: m.id.split(/[-.]/)[0] + (m.experts ? "moe" : ""), name: m.name, quant: lib.quant, block_count: layers, context_length: m.ctx_max,
        expert_count: m.experts ?? 0, expert_used_count: m.experts_used ?? 0, total_bytes: lib.size_bytes,
      },
      mmproj: lib.mmproj,
      saved_custom_args: "",
      // Keeps the run page on llama.cpp; Strata would take over as the default engine.
      strata_supported: false,
    };
  }

  // ---- Jobs (downloads and installs): a short progress run, then done ----
  let jobSeq = 0;
  function startJob(outputs = []) {
    const id = `job-${++jobSeq}`;
    (async () => {
      const total = 100e6;
      for (let i = 1; i <= 4; i++) {
        await wait(150);
        emit("progress", { job_id: id, stage: "download", done: (total * i) / 4, total, speed_bps: 48e6, file_index: 0, file_count: 1, message: null });
      }
      emit("job-done", { job_id: id, ok: true, error: null, outputs });
    })();
    return id;
  }

  // ---- Fake outputs ----
  function artwork(seed) {
    const hue = (seed * 67) % 360;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="hsl(${hue},70%,62%)"/><stop offset="1" stop-color="hsl(${(hue + 70) % 360},65%,32%)"/></linearGradient></defs><rect width="512" height="512" fill="url(#g)"/><circle cx="380" cy="150" r="56" fill="hsla(${(hue + 160) % 360},90%,85%,.85)"/><path d="M0 400 L150 260 L270 360 L370 280 L512 410 V512 H0Z" fill="hsla(${hue},40%,12%,.55)"/></svg>`;
    return "data:image/svg+xml;base64," + btoa(svg);
  }

  // ---- Chat: scripted replies sent as events, like the backend streams them ----
  const say = async (session, event) => { emit("chat", { session, event }); await wait(40); };

  async function chatSend({ session, messages, tools }) {
    await say(session, { type: "reasoning", delta: tr("使用者要算 17 × 23：17 × 20 = 340，17 × 3 = 51，加起來是 391。", "17 × 23: 17 × 20 = 340 and 17 × 3 = 51, so 391.") });
    if (tools) {
      await say(session, { type: "tool_call", id: "call-1", name: "calculator", arguments: JSON.stringify({ expression: "17 * 23" }) });
      await say(session, { type: "tool_result", id: "call-1", name: "calculator", content: "391", ok: true });
    }
    const text = "17 × 23 = 391。";
    await say(session, { type: "text", delta: text });
    await say(session, { type: "usage", prompt_tokens: 42, completion_tokens: 18, tokens_per_sec: 64.3 });
    await say(session, { type: "done", reason: "stop" });
    return [...messages, { role: "assistant", content: text }];
  }

  // The agent stops at an approval request until `approval_resolve` answers it.
  async function agentSend({ session, messages, workspace }) {
    await say(session, { type: "plan", steps: [{ step: tr("讀取專案結構", "Read the project layout"), status: "done" }, { step: tr("執行測試", "Run the tests"), status: "in_progress" }] });
    await say(session, { type: "tool_call", id: "call-2", name: "run_command", arguments: JSON.stringify({ command: "npm test" }) });
    emit("approval", { session, request: { id: "approval-1", tool: "run_command", summary: `Run: npm test (in ${workspace})` } });
    const decision = await new Promise((resolve) => state.pending.set(session, resolve));
    const ok = decision !== "deny";
    await say(session, { type: "tool_result", id: "call-2", name: "run_command", content: ok ? "12 passed" : "denied", ok });
    await say(session, { type: "done", reason: "stop" });
    return messages;
  }

  // ---- Commands: one entry per invoke() name in app/src/api.ts ----
  const handlers = {
    get_settings: () => state.settings,
    save_settings: ({ settings }) => { state.settings = clone(settings); },
    app_paths: () => data.paths,
    get_hardware: () => hw,
    live_vram: () => [{ name: hw.gpus[0].name, used: state.llm.state === "ready" ? 17.4 * GIB : 1.6 * GIB, total: hw.gpus[0].vram_total, spill_suspected: false }],
    catalog: () => catalog.map(catalogEntry),
    plan_preview: ({ opts }) => planPreview(opts),
    library_list: () => data.library.map(({ catalog: _c, ...m }) => m),
    library_import: () => { const { catalog: _c, ...m } = data.library[0]; return m; },
    library_remove: () => {},
    hf_search: () => data.hf_results,
    hf_listing: () => data.hf_listing,
    hf_download: () => startJob(),
    cancel_job: () => true,
    engines: () => data.engines,
    engine_install: () => startJob(),
    engine_import: () => data.engines.llama[0],
    engine_remove: () => {},
    server_status: ({ which }) => (which === "sd" ? state.sd : state.llm),
    server_logs: () => [],
    server_stop: ({ which }) => {
      const st = which === "sd" ? state.sd : state.llm;
      st.state = "stopped";
      emit("server", { which, event: { type: "status", state: "stopped", port: st.port } });
    },
    llm_start: ({ opts }) => {
      Object.assign(state.llm, { state: "ready", model: opts.path, gpu_layers: [49, 49] });
      emit("server", { which: "llm", event: { type: "status", state: "ready", port: state.llm.port } });
      return planPreview(opts).plan;
    },
    py_engines: () => data.py_engines,
    py_engine_install: () => startJob(),
    py_engine_latest: () => "0.0.0",
    py_engine_remove: () => {},
    py_engine_start: () => state.llm.port,
    open_folder: () => {},
    lan_addresses: () => data.net_addrs,
    strata_install: () => startJob(),
    strata_remove: () => {},
    strata_managed: () => data.strata_managed.map((m) => ({ ...m, description: m.description[lang()] ?? m.description.en })),
    strata_plan: ({ id, set }) => `${id} / ${set}\nGPU: attention layers, shared weights\nRAM: routed experts\nDisk: none`,
    strata_start_managed: () => {},
    media_models: () => data.media_models,
    media_download: () => startJob(),
    sd_start: () => {
      state.sd.state = "ready";
      emit("server", { which: "sd", event: { type: "status", state: "ready", port: state.sd.port } });
    },
    image_generate: async ({ req }) => {
      await wait(300);
      return [1, 2, 3, 4].map((i) => ({ path: `F:\\OM\\outputs\\image-${i}.png`, data_url: artwork(i + (req.seed ?? 0)) }));
    },
    video_generate: () => startJob(),
    read_media: () => "",
    chat_send: chatSend,
    agent_send: agentSend,
    chat_stop: ({ session }) => { state.pending.get(session)?.("deny"); state.pending.delete(session); },
    approval_resolve: ({ session, decision }) => { state.pending.get(session)?.(decision); state.pending.delete(session); return true; },
    endpoint_models: ({ endpoint }) => (endpoint.model ? [endpoint.model] : []),
    web_ui_status: () => ({ running: false, urls: [], error: null }),
    jit_status: () => ({ running: true, port: state.settings.server_port, urls: [`http://127.0.0.1:${state.settings.server_port}/v1`], error: null, loaded: [] }),
    web_ui_set_password: () => {},
    // Plugins the app uses through @tauri-apps/api.
    "plugin:event|listen": ({ event, handler }) => {
      const eventId = nextId++;
      if (!listeners.has(event)) listeners.set(event, new Map());
      listeners.get(event).set(eventId, handler);
      return eventId;
    },
    "plugin:event|unlisten": ({ event, eventId }) => { listeners.get(event)?.delete(eventId); },
    "plugin:dialog|open": () => null,
    "plugin:opener|open_url": () => {},
    "plugin:opener|reveal_item_in_dir": () => {},
  };

  async function invoke(cmd, args = {}) {
    await ready;
    const h = handlers[cmd];
    if (!h) {
      console.warn(`unmocked ${cmd}`);
      throw `unmocked command: ${cmd}`;
    }
    return clone(await h(args));
  }

  window.__TAURI_INTERNALS__ = {
    invoke,
    transformCallback,
    unregisterCallback: (id) => callbacks.delete(id),
    convertFileSrc: (p) => p,
    metadata: { currentWindow: { label: "main" }, currentWebview: { label: "main", windowLabel: "main" } },
  };
  window.__TAURI_EVENT_PLUGIN_INTERNALS__ = { unregisterListener: () => {} };

  // Hooks for the capture script.
  window.__mock = {
    ready,
    emit,
    commands: Object.keys(handlers).filter((k) => !k.startsWith("plugin:")),
    /** Pretend the model server has been running for a while. */
    seedLogs: () => {
      const lines = [
        "llama_model_loader: loaded meta data with 41 key-value pairs and 733 tensors",
        "load_tensors: offloaded 49/49 layers to GPU",
        "llama_context: n_ctx = 65536, flash_attn = 1, type_k = q8_0, type_v = q8_0",
        "server is listening on http://127.0.0.1:8080 - starting the main loop",
        "srv  update_slots: all slots are idle",
      ];
      for (const line of lines) emit("server", { which: "llm", event: { type: "log", line } });
    },
  };
})();
