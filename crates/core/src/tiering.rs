//! Offload planning: decide how a model is split across GPU, RAM and disk,
//! estimate speed, and build the llama-server command line.
//!
//! Strategy, in order:
//! 1. Everything (weights + KV cache + compute buffers) on the GPU.
//! 2. MoE: keep attention/shared weights on the GPU and move the expert
//!    tensors of the first N layers to system RAM (`--n-cpu-moe N`).
//! 3. Dense: offload as many whole layers as fit (`-ngl N`).
//! 4. Whatever does not fit in RAM is streamed from disk through mmap.

use crate::config::{GIB, MIB};
use serde::{Deserialize, Serialize};

/// Model layout used for planning. Built from a GGUF header
/// ([`crate::gguf::ModelSummary`]) or, before download, from the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelShape {
    pub total_bytes: u64,
    pub layer_bytes: Vec<u64>,
    pub layer_expert_bytes: Vec<u64>,
    pub other_bytes: u64,
    pub kv_layers: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub ctx_max: u64,
    pub expert_count: u64,
    pub expert_used: u64,
    pub mmproj_bytes: u64,
}

impl ModelShape {
    pub fn layers(&self) -> u64 {
        self.layer_bytes.len() as u64
    }
    pub fn is_moe(&self) -> bool {
        self.expert_count > 1 && self.layer_expert_bytes.iter().any(|b| *b > 0)
    }
    fn expert_total(&self) -> u64 {
        self.layer_expert_bytes.iter().sum()
    }
    /// Fraction of expert weights touched per token.
    fn expert_active_frac(&self) -> f64 {
        if self.expert_count == 0 {
            1.0
        } else {
            (self.expert_used.max(1) as f64 / self.expert_count as f64).min(1.0)
        }
    }

    pub fn from_summary(s: &crate::gguf::ModelSummary, mmproj_bytes: u64) -> ModelShape {
        ModelShape {
            total_bytes: s.total_bytes,
            layer_bytes: s.layer_bytes.clone(),
            layer_expert_bytes: s.layer_expert_bytes.clone(),
            other_bytes: s.other_bytes,
            kv_layers: s.kv_layers,
            kv_heads: s.head_count_kv,
            head_dim: s.head_dim,
            ctx_max: if s.context_length > 0 { s.context_length } else { 32768 },
            expert_count: s.expert_count,
            expert_used: s.expert_used_count,
            mmproj_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum KvType {
    F16,
    #[default]
    Q8_0,
}

impl KvType {
    pub fn as_str(&self) -> &'static str {
        match self {
            KvType::F16 => "f16",
            KvType::Q8_0 => "q8_0",
        }
    }
    fn bytes_per_elem(&self) -> f64 {
        match self {
            KvType::F16 => 2.0,
            KvType::Q8_0 => 34.0 / 32.0,
        }
    }
}

/// Memory/bandwidth budget of the machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Machine {
    /// Usable GPU memory (0 = CPU only).
    pub vram_bytes: u64,
    pub gpu_bandwidth_gbs: f64,
    /// GPU shares system RAM (Apple Silicon, APUs).
    pub unified_memory: bool,
    pub ram_available: u64,
    pub ram_bandwidth_gbs: f64,
    pub disk_read_bps: u64,
    pub vram_reserve: u64,
    pub ram_reserve: u64,
}

impl Machine {
    pub fn from_report(hw: &crate::hardware::HardwareReport, s: &crate::config::Settings) -> Machine {
        let g = hw.primary_gpu();
        Machine {
            vram_bytes: g.map(|g| g.vram_total).unwrap_or(0),
            gpu_bandwidth_gbs: g.map(|g| g.bandwidth_gbs).unwrap_or(0.0),
            unified_memory: g.map(|g| g.unified_memory).unwrap_or(false),
            ram_available: hw.ram_available,
            ram_bandwidth_gbs: hw.ram_bandwidth_gbs,
            disk_read_bps: hw.model_disk.as_ref().map(|d| d.read_speed_bps()).unwrap_or(1500 * MIB),
            vram_reserve: s.vram_reserve_bytes,
            ram_reserve: s.ram_reserve_bytes,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanOptions {
    /// None = largest context that does not slow generation down.
    pub ctx: Option<u64>,
    /// Smallest context the automatic choice may pick (capped at the model's
    /// maximum). Agents such as Hermes refuse windows under 64K.
    pub min_ctx: u64,
    pub kv_type: KvType,
    pub flash_attn: bool,
    pub model_path: String,
    pub mmproj_path: Option<String>,
    pub host: String,
    pub port: u16,
    /// User arguments, appended last; same-name flags win.
    pub custom_args: String,
    /// Compatibility level 0-5 used after a failed start.
    pub safe_level: u8,
    pub lang: String,
}

impl Default for PlanOptions {
    fn default() -> Self {
        PlanOptions {
            ctx: None,
            min_ctx: 0,
            kv_type: KvType::Q8_0,
            flash_attn: true,
            model_path: "model.gguf".into(),
            mmproj_path: None,
            host: "127.0.0.1".into(),
            port: 8080,
            custom_args: String::new(),
            safe_level: 0,
            lang: "zh-TW".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Fast,
    Ok,
    Slow,
    Crawl,
}

impl Tier {
    pub fn label(&self, lang: &str) -> &'static str {
        let zh = lang.starts_with("zh");
        match (self, zh) {
            (Tier::Fast, true) => "流暢",
            (Tier::Ok, true) => "可用",
            (Tier::Slow, true) => "偏慢",
            (Tier::Crawl, true) => "非常慢",
            (Tier::Fast, false) => "Fast",
            (Tier::Ok, false) => "Usable",
            (Tier::Slow, false) => "Slow",
            (Tier::Crawl, false) => "Very slow",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub tier: Tier,
    pub tier_label: String,
    pub ctx: u64,
    /// -ngl value (layers on GPU); equals layer count when fully offloaded.
    pub gpu_layers: u64,
    /// --n-cpu-moe value (0 = none).
    pub n_cpu_moe: u64,
    pub gpu_bytes: u64,
    pub ram_bytes: u64,
    pub disk_bytes: u64,
    pub kv_bytes: u64,
    pub est_tps: f64,
    pub est_load_secs: f64,
    pub summary: String,
    pub gpu_note: String,
    pub ram_note: String,
    pub disk_note: String,
    /// Final argument list (without the executable).
    pub args: Vec<String>,
    /// Extra environment variables (set by compatibility levels).
    pub env: Vec<(String, String)>,
    /// `args` rendered as one shell-style line for display.
    pub command: String,
    pub custom_args: String,
    /// True when custom args took over layer placement.
    pub manual_placement: bool,
}

const COMPUTE_OVERHEAD: u64 = 600 * MIB;
const GPU_EFF: f64 = 0.65;
const CPU_EFF: f64 = 0.55;
const AUTO_CTX: &[u64] = &[8192, 16384, 32768, 65536];

pub fn kv_bytes(shape: &ModelShape, ctx: u64, kv: KvType) -> u64 {
    (2.0 * shape.kv_layers as f64 * ctx as f64 * shape.kv_heads as f64 * shape.head_dim as f64 * kv.bytes_per_elem()) as u64
}

struct Placement {
    gpu_layers: u64,
    n_cpu_moe: u64,
    gpu_weights: u64,
    gpu_kv: u64,
}

fn place(shape: &ModelShape, m: &Machine, kv: u64, mmproj: u64) -> Placement {
    let n = shape.layers();
    let budget = m.vram_bytes.saturating_sub(m.vram_reserve);
    let fixed = COMPUTE_OVERHEAD + mmproj;
    if m.vram_bytes == 0 || budget <= fixed {
        return Placement { gpu_layers: 0, n_cpu_moe: 0, gpu_weights: 0, gpu_kv: 0 };
    }
    let budget = budget - fixed;
    if shape.total_bytes + kv <= budget {
        return Placement { gpu_layers: n, n_cpu_moe: 0, gpu_weights: shape.total_bytes, gpu_kv: kv };
    }
    if shape.is_moe() {
        let mut on_gpu = shape.total_bytes + kv;
        for (i, e) in shape.layer_expert_bytes.iter().enumerate() {
            on_gpu -= e;
            if on_gpu <= budget {
                return Placement { gpu_layers: n, n_cpu_moe: i as u64 + 1, gpu_weights: on_gpu - kv, gpu_kv: kv };
            }
        }
    }
    // Dense split: llama.cpp offloads the last N layers; the output head goes
    // with them, the embeddings stay on the CPU.
    let kv_per_layer = kv.checked_div(n).unwrap_or(0);
    let mut used = shape.other_bytes / 2;
    let mut layers = 0;
    for b in shape.layer_bytes.iter().rev() {
        if used + b + kv_per_layer > budget {
            break;
        }
        used += b + kv_per_layer;
        layers += 1;
    }
    if layers == 0 {
        return Placement { gpu_layers: 0, n_cpu_moe: 0, gpu_weights: 0, gpu_kv: 0 };
    }
    Placement { gpu_layers: layers, n_cpu_moe: 0, gpu_weights: used - kv_per_layer * layers, gpu_kv: kv_per_layer * layers }
}

struct Numbers {
    p: Placement,
    kv: u64,
    ram: u64,
    disk: u64,
    tps: f64,
}

fn evaluate(shape: &ModelShape, m: &Machine, ctx: u64, kv_type: KvType, mmproj: u64) -> Numbers {
    let kv = kv_bytes(shape, ctx, kv_type);
    let p = place(shape, m, kv, mmproj);
    let cpu_weights = shape.total_bytes - p.gpu_weights;
    let cpu_kv = kv - p.gpu_kv;
    let mut ram_free = m.ram_available.saturating_sub(m.ram_reserve);
    if m.unified_memory {
        ram_free = ram_free.saturating_sub(p.gpu_weights + p.gpu_kv);
    }
    let in_ram = (cpu_weights + cpu_kv).min(ram_free);
    let disk = (cpu_weights + cpu_kv).saturating_sub(ram_free);

    // Bytes read per generated token, by location.
    let frac = shape.expert_active_frac();
    let active = |weights: u64, experts: u64| (weights - experts) as f64 + experts as f64 * frac;
    let total_exp = shape.expert_total();
    let cpu_exp: u64 = if p.n_cpu_moe > 0 {
        shape.layer_expert_bytes.iter().take(p.n_cpu_moe as usize).sum()
    } else if p.gpu_weights == 0 {
        total_exp
    } else {
        total_exp * cpu_weights / shape.total_bytes.max(1)
    };
    let gpu_exp = total_exp - cpu_exp.min(total_exp);
    let gpu_active = active(p.gpu_weights, gpu_exp.min(p.gpu_weights));
    let cpu_active = active(cpu_weights, cpu_exp.min(cpu_weights));
    let disk_share = if cpu_weights > 0 { disk as f64 / (cpu_weights + cpu_kv) as f64 } else { 0.0 };
    let gbs = 1e9;
    let mut t = 0.0;
    if m.gpu_bandwidth_gbs > 0.0 {
        t += (gpu_active + p.gpu_kv as f64 * 0.5) / (m.gpu_bandwidth_gbs * gbs * GPU_EFF);
    }
    t += (cpu_active * (1.0 - disk_share) + cpu_kv as f64 * 0.5) / (m.ram_bandwidth_gbs.max(1.0) * gbs * CPU_EFF);
    t += cpu_active * disk_share / m.disk_read_bps.max(1) as f64;
    let tps = if t > 0.0 { 1.0 / t } else { 0.0 };
    Numbers { p, kv, ram: in_ram, disk, tps }
}

fn tier_for(tps: f64, disk: u64) -> Tier {
    let t = if tps >= 25.0 {
        Tier::Fast
    } else if tps >= 10.0 {
        Tier::Ok
    } else if tps >= 4.0 {
        Tier::Slow
    } else {
        Tier::Crawl
    };
    if disk > 0 && t == Tier::Fast {
        Tier::Slow
    } else {
        t
    }
}

/// Pick the context: explicit value (capped at the model max) or the largest
/// automatic choice whose speed stays within 10 % of the smallest one.
fn choose_ctx(shape: &ModelShape, m: &Machine, o: &PlanOptions, mmproj: u64) -> u64 {
    let cap = shape.ctx_max.max(4096);
    if let Some(c) = o.ctx {
        return c.clamp(512, cap);
    }
    let floor = o.min_ctx.min(cap);
    let mut cands: Vec<u64> = AUTO_CTX.iter().copied().filter(|c| *c <= cap && *c >= floor).collect();
    if floor > 0 && !cands.contains(&floor) {
        cands.insert(0, floor);
    }
    let Some(&first) = cands.first() else { return cap };
    let base = evaluate(shape, m, first, o.kv_type, mmproj);
    let mut best = first;
    for c in cands.into_iter().skip(1) {
        let n = evaluate(shape, m, c, o.kv_type, mmproj);
        if n.tps >= base.tps * 0.9 && n.disk <= base.disk && n.p.gpu_layers >= base.p.gpu_layers {
            best = c;
        } else {
            break;
        }
    }
    best
}

pub fn plan(shape: &ModelShape, m: &Machine, o: &PlanOptions) -> Plan {
    let zh = o.lang.starts_with("zh");
    let mut kv_type = o.kv_type;
    if o.safe_level >= 1 {
        kv_type = KvType::F16;
    }
    let use_mmproj = o.mmproj_path.is_some() && o.safe_level < 4;
    let mmproj = if use_mmproj { shape.mmproj_bytes } else { 0 };
    let ctx = choose_ctx(shape, m, o, mmproj);
    let n = evaluate(shape, m, ctx, kv_type, mmproj);
    let tier = tier_for(n.tps, n.disk);
    let layers = shape.layers();

    let gb = |b: u64| b as f64 / GIB as f64;
    let gpu_note = match (m.vram_bytes, n.p.gpu_layers) {
        (0, _) => if zh { "沒有可用的 GPU，全部在 CPU 上執行".into() } else { "No usable GPU; running on the CPU".into() },
        (_, 0) => if zh { "顯示記憶體不足以放下任何一層".into() } else { "Not enough VRAM for a single layer".into() },
        _ if n.p.gpu_layers == layers && n.p.n_cpu_moe == 0 => {
            if zh { format!("全部 {layers} 層放進 GPU（{:.1} GB）", gb(n.p.gpu_weights + n.p.gpu_kv)) } else { format!("All {layers} layers on the GPU ({:.1} GB)", gb(n.p.gpu_weights + n.p.gpu_kv)) }
        }
        _ if n.p.n_cpu_moe > 0 => {
            if zh { format!("注意力與共用權重在 GPU，前 {} 層的專家權重放到記憶體", n.p.n_cpu_moe) } else { format!("Attention/shared weights on the GPU; experts of the first {} layers in RAM", n.p.n_cpu_moe) }
        }
        _ => if zh { format!("{}/{layers} 層放進 GPU", n.p.gpu_layers) } else { format!("{}/{layers} layers on the GPU", n.p.gpu_layers) },
    };
    let ram_note = if n.ram == 0 {
        String::new()
    } else if zh {
        format!("系統記憶體使用約 {:.1} GB", gb(n.ram))
    } else {
        format!("About {:.1} GB in system RAM", gb(n.ram))
    };
    let disk_note = if n.disk == 0 {
        String::new()
    } else if zh {
        format!("記憶體不夠，約 {:.1} GB 會從硬碟讀取，速度會明顯變慢", gb(n.disk))
    } else {
        format!("Not enough RAM: about {:.1} GB is read from disk, which is much slower", gb(n.disk))
    };
    let summary = if zh {
        format!("{} · 約 {:.0} tok/s · 上下文 {}K", tier.label(&o.lang), n.tps, ctx / 1024)
    } else {
        format!("{} · ~{:.0} tok/s · {}K context", tier.label(&o.lang), n.tps, ctx / 1024)
    };

    let mut auto: Vec<String> = vec!["-m".into(), o.model_path.clone(), "--ctx-size".into(), ctx.to_string()];
    let ngl = if n.p.gpu_layers == layers { 999 } else { n.p.gpu_layers };
    auto.extend(["-ngl".into(), ngl.to_string()]);
    if n.p.n_cpu_moe > 0 {
        auto.extend(["--n-cpu-moe".into(), n.p.n_cpu_moe.to_string()]);
    }
    let fa = o.flash_attn && o.safe_level < 2;
    auto.extend(["--flash-attn".into(), if fa { "on" } else { "off" }.into()]);
    if kv_type != KvType::F16 && fa {
        auto.extend(["--cache-type-k".into(), kv_type.as_str().into(), "--cache-type-v".into(), kv_type.as_str().into()]);
    }
    if use_mmproj {
        auto.extend(["--mmproj".into(), o.mmproj_path.clone().unwrap_or_default()]);
    }
    if o.safe_level >= 5 {
        auto.extend(["--batch-size".into(), "512".into(), "--ubatch-size".into(), "256".into()]);
    }
    auto.extend(["--jinja".into(), "--host".into(), o.host.clone(), "--port".into(), o.port.to_string()]);

    let custom = split_args(&o.custom_args);
    let manual = custom.iter().any(|a| PLACEMENT_FLAGS.contains(&canonical_flag(a)));
    if manual {
        auto = remove_flags(&auto, AUTO_PLACEMENT_FLAGS);
    }
    let args = merge_args(&auto, &custom);

    let mut env = Vec::new();
    if o.safe_level >= 3 {
        env.push(("GGML_CUDA_DISABLE_GRAPHS".into(), "1".into()));
    }
    if o.safe_level >= 5 {
        env.push(("GGML_CUDA_FORCE_CUBLAS".into(), "1".into()));
    }

    Plan {
        tier,
        tier_label: tier.label(&o.lang).into(),
        ctx,
        gpu_layers: n.p.gpu_layers,
        n_cpu_moe: n.p.n_cpu_moe,
        gpu_bytes: n.p.gpu_weights + n.p.gpu_kv,
        ram_bytes: n.ram,
        disk_bytes: n.disk,
        kv_bytes: n.kv,
        est_tps: (n.tps * 10.0).round() / 10.0,
        est_load_secs: shape.total_bytes as f64 / m.disk_read_bps.max(1) as f64 + 5.0,
        summary,
        gpu_note,
        ram_note,
        disk_note,
        command: render_command(&args),
        args,
        env,
        custom_args: o.custom_args.clone(),
        manual_placement: manual,
    }
}

/// Flags that, when given by the user, disable automatic layer placement.
const PLACEMENT_FLAGS: &[&str] = &["--n-gpu-layers", "--ctx-size", "--override-tensor", "--n-cpu-moe", "--cpu-moe"];
/// Automatic flags dropped in that case. The context size stays (unless the
/// user overrides it) so llama-server does not fall back to the model maximum.
const AUTO_PLACEMENT_FLAGS: &[&str] = &["--n-gpu-layers", "--override-tensor", "--n-cpu-moe", "--cpu-moe"];

/// Map short aliases to the long llama-server flag.
pub fn canonical_flag(f: &str) -> &str {
    match f {
        "-ngl" | "--gpu-layers" => "--n-gpu-layers",
        "-c" => "--ctx-size",
        "-ot" => "--override-tensor",
        "-ncmoe" => "--n-cpu-moe",
        "-cmoe" => "--cpu-moe",
        "-fa" => "--flash-attn",
        "-ctk" => "--cache-type-k",
        "-ctv" => "--cache-type-v",
        "-b" => "--batch-size",
        "-ub" => "--ubatch-size",
        "-ts" => "--tensor-split",
        "-sm" => "--split-mode",
        "-mg" => "--main-gpu",
        "-m" => "--model",
        "-t" => "--threads",
        other => other,
    }
}

/// Flags that never take a value.
const BOOL_FLAGS: &[&str] = &["--no-mmap", "--mlock", "--cpu-moe", "--jinja", "--no-warmup", "--cont-batching", "--metrics", "--verbose", "-v", "--embedding", "--no-webui", "--swa-full", "--check-tensors", "--no-kv-offload"];

fn is_flag(s: &str) -> bool {
    s.starts_with('-') && s.len() > 1 && s.parse::<f64>().is_err()
}

/// Group a flat arg list into (flag, values) pairs; positional args get an empty flag.
fn group(args: &[String]) -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if is_flag(a) {
            if let Some((f, v)) = a.split_once('=').filter(|_| a.starts_with("--")) {
                out.push((f.to_string(), vec![v.to_string()]));
            } else if !BOOL_FLAGS.contains(&canonical_flag(a)) && i + 1 < args.len() && !is_flag(&args[i + 1]) {
                out.push((a.clone(), vec![args[i + 1].clone()]));
                i += 1;
            } else {
                out.push((a.clone(), vec![]));
            }
        } else {
            out.push((String::new(), vec![a.clone()]));
        }
        i += 1;
    }
    out
}

fn flatten(g: Vec<(String, Vec<String>)>) -> Vec<String> {
    g.into_iter().flat_map(|(f, v)| if f.is_empty() { v } else { std::iter::once(f).chain(v).collect() }).collect()
}

fn remove_flags(args: &[String], flags: &[&str]) -> Vec<String> {
    flatten(group(args).into_iter().filter(|(f, _)| !flags.contains(&canonical_flag(f))).collect())
}

/// Append `custom` after `base`; a flag present in `custom` replaces the same
/// flag (by canonical name) in `base`.
pub fn merge_args(base: &[String], custom: &[String]) -> Vec<String> {
    let c = group(custom);
    let overridden: Vec<&str> = c.iter().filter(|(f, _)| !f.is_empty()).map(|(f, _)| canonical_flag(f)).collect();
    let mut out: Vec<(String, Vec<String>)> = group(base).into_iter().filter(|(f, _)| !overridden.contains(&canonical_flag(f))).collect();
    out.extend(c);
    flatten(out)
}

/// Split a user-typed argument string, honouring single and double quotes.
pub fn split_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut has = false;
    for ch in s.chars() {
        match (quote, ch) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => cur.push(c),
            (None, '"' | '\'') => {
                quote = Some(ch);
                has = true;
            }
            (None, c) if c.is_whitespace() => {
                if has || !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                    has = false;
                }
            }
            (None, c) => cur.push(c),
        }
    }
    if has || !cur.is_empty() {
        out.push(cur);
    }
    out
}

pub fn render_command(args: &[String]) -> String {
    args.iter()
        .map(|a| if a.is_empty() || a.contains(char::is_whitespace) { format!("\"{a}\"") } else { a.clone() })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(layers: usize, layer_gb: f64) -> ModelShape {
        let lb = (layer_gb * GIB as f64) as u64;
        ModelShape {
            total_bytes: lb * layers as u64 + GIB,
            layer_bytes: vec![lb; layers],
            layer_expert_bytes: vec![0; layers],
            other_bytes: GIB,
            kv_layers: layers as u64,
            kv_heads: 8,
            head_dim: 128,
            ctx_max: 131072,
            expert_count: 0,
            expert_used: 0,
            mmproj_bytes: 0,
        }
    }

    fn gpu(vram_gb: u64) -> Machine {
        Machine {
            vram_bytes: vram_gb * GIB,
            gpu_bandwidth_gbs: 912.0,
            unified_memory: false,
            ram_available: 48 * GIB,
            ram_bandwidth_gbs: 60.0,
            disk_read_bps: 1500 * MIB,
            vram_reserve: GIB / 2,
            ram_reserve: 4 * GIB,
        }
    }

    #[test]
    fn auto_context_respects_the_minimum() {
        // A model too big for the GPU: without a floor auto stays small.
        let shape = dense(60, 0.5);
        let fast = plan(&shape, &gpu(12), &PlanOptions::default());
        let agent = plan(&shape, &gpu(12), &PlanOptions { min_ctx: 65536, ..Default::default() });
        assert!(agent.ctx >= 65536, "got {}", agent.ctx);
        assert!(fast.ctx <= agent.ctx);
        // An explicit context still wins.
        assert_eq!(plan(&shape, &gpu(12), &PlanOptions { ctx: Some(8192), min_ctx: 65536, ..Default::default() }).ctx, 8192);
    }

    #[test]
    fn kv_formula() {
        let s = dense(32, 0.1);
        // 2 * 32 layers * 4096 ctx * 8 heads * 128 dim * 2 bytes = 512 MiB
        assert_eq!(kv_bytes(&s, 4096, KvType::F16), 512 * MIB);
    }

    #[test]
    fn small_model_fits_on_gpu() {
        let p = plan(&dense(32, 0.2), &gpu(12), &PlanOptions::default());
        assert_eq!(p.gpu_layers, 32);
        assert_eq!(p.n_cpu_moe, 0);
        assert_eq!(p.disk_bytes, 0);
        assert_eq!(p.tier, Tier::Fast);
        assert!(p.args.windows(2).any(|w| w == ["-ngl", "999"]));
        // 64K would no longer fit the KV cache on a 12 GB card, so auto stops at 32K.
        assert_eq!(p.ctx, 32768);
    }

    #[test]
    fn dense_partial_offload() {
        // 60 layers x 0.5 GB = 31 GB model on a 12 GB card
        let p = plan(&dense(60, 0.5), &gpu(12), &PlanOptions { ctx: Some(8192), ..Default::default() });
        assert!(p.gpu_layers > 10 && p.gpu_layers < 60, "{}", p.gpu_layers);
        assert!(p.ram_bytes > 15 * GIB);
        assert!(p.est_tps < plan(&dense(60, 0.5), &gpu(48), &PlanOptions { ctx: Some(8192), ..Default::default() }).est_tps);
    }

    #[test]
    fn moe_moves_experts_first() {
        let c = crate::catalog::Catalog::embedded();
        let shape = c.find("qwen3.6-35b-a3b").unwrap().shape("UD-Q4_K_XL");
        let p = plan(&shape, &gpu(12), &PlanOptions { ctx: Some(32768), ..Default::default() });
        assert_eq!(p.gpu_layers, shape.layers());
        assert!(p.n_cpu_moe > 0 && p.n_cpu_moe < shape.layers());
        assert!(p.args.contains(&"--n-cpu-moe".to_string()));
        assert!(p.est_tps > 10.0, "MoE with experts in RAM should still be usable: {}", p.est_tps);
    }

    #[test]
    fn cpu_only_and_disk_spill() {
        let mut m = gpu(0);
        m.ram_available = 16 * GIB;
        let p = plan(&dense(40, 0.5), &m, &PlanOptions { ctx: Some(4096), ..Default::default() });
        assert_eq!(p.gpu_layers, 0);
        assert!(p.disk_bytes > 0);
        assert!(!p.disk_note.is_empty());
        assert_ne!(p.tier, Tier::Fast);
    }

    #[test]
    fn safe_levels_degrade() {
        let o = |lvl| PlanOptions { safe_level: lvl, mmproj_path: Some("mm.gguf".into()), ..Default::default() };
        let s = dense(8, 0.1);
        let p1 = plan(&s, &gpu(12), &o(1));
        assert!(!p1.args.contains(&"--cache-type-k".to_string()));
        let p2 = plan(&s, &gpu(12), &o(2));
        assert!(p2.args.windows(2).any(|w| w == ["--flash-attn", "off"]));
        let p4 = plan(&s, &gpu(12), &o(4));
        assert!(!p4.args.contains(&"--mmproj".to_string()));
        assert!(p4.env.iter().any(|(k, _)| k == "GGML_CUDA_DISABLE_GRAPHS"));
        let p5 = plan(&s, &gpu(12), &o(5));
        assert!(p5.env.iter().any(|(k, _)| k == "GGML_CUDA_FORCE_CUBLAS"));
    }

    #[test]
    fn custom_args_override_and_disable_auto_placement() {
        let o = PlanOptions { custom_args: "-ngl 20 --flash-attn off --temp 0.6 --no-mmap".into(), ..Default::default() };
        let p = plan(&dense(32, 0.2), &gpu(12), &o);
        assert!(p.manual_placement);
        assert_eq!(p.args.iter().filter(|a| *a == "-ngl").count(), 1);
        assert!(p.args.contains(&"--ctx-size".to_string()), "context stays when only -ngl is given");
        assert_eq!(p.args.iter().filter(|a| *a == "--flash-attn").count(), 1);
        assert!(p.command.ends_with("-ngl 20 --flash-attn off --temp 0.6 --no-mmap"));
    }

    #[test]
    fn arg_helpers() {
        assert_eq!(split_args(r#"--chat-template-file "C:\My Templates\a.jinja" -t 8"#), vec!["--chat-template-file", r"C:\My Templates\a.jinja", "-t", "8"]);
        let base: Vec<String> = ["-c", "4096", "--port", "8080"].map(String::from).to_vec();
        assert_eq!(merge_args(&base, &split_args("--ctx-size=8192")), vec!["--port", "8080", "--ctx-size", "8192"]);
        assert_eq!(render_command(&["-m".into(), "a b.gguf".into()]), "-m \"a b.gguf\"");
    }
}
