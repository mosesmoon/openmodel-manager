//! `openmodel` — headless front end for testing on machines without a GUI.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use openmodel_core::config::{data_dir, Settings, GIB};
use openmodel_core::engine::{self, EngineKind};
use openmodel_core::hardware::{self, Backend};
use openmodel_core::tiering::{self, Machine, ModelShape, PlanOptions};
use openmodel_core::{catalog::Catalog, download, gguf, hf::HfClient, library, server, Progress, ProgressFn};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[derive(Parser)]
#[command(name = "openmodel", version, about = "OpenModel-Manager CLI")]
struct Cli {
    /// Print JSON instead of text where supported.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show detected hardware and recommended backends.
    Hw,
    /// Inspect a GGUF file's header.
    Gguf { path: PathBuf },
    /// List the built-in model catalog.
    Catalog,
    /// Plan GPU/RAM/disk placement for a local GGUF or a catalog model.
    Plan {
        /// Local .gguf file (any split part).
        #[arg(long, conflicts_with = "catalog")]
        gguf: Option<PathBuf>,
        /// Catalog id, e.g. qwen3.6-35b-a3b.
        #[arg(long, requires = "quant")]
        catalog: Option<String>,
        #[arg(long)]
        quant: Option<String>,
        /// Context size (default: automatic).
        #[arg(long)]
        ctx: Option<u64>,
        /// Extra llama-server arguments, e.g. "--temp 0.6 -ngl 20".
        #[arg(long, default_value = "")]
        custom: String,
    },
    /// Hugging Face search / file listing / download.
    Hf {
        #[command(subcommand)]
        cmd: HfCmd,
    },
    /// Locally available models.
    Models,
    /// Inference engines (llama.cpp, stable-diffusion.cpp).
    Engine {
        #[command(subcommand)]
        cmd: EngineCmd,
    },
    /// Start llama-server for a GGUF and keep it running until Ctrl+C.
    Serve {
        path: PathBuf,
        #[arg(long)]
        ctx: Option<u64>,
        /// Open the server on every LAN address of this computer.
        #[arg(long)]
        lan: bool,
        #[arg(long)]
        port: Option<u16>,
        #[arg(long, default_value = "")]
        custom: String,
        /// Do not attach the mmproj found next to the model.
        #[arg(long)]
        no_vision: bool,
        /// Print llama-server output.
        #[arg(long, short)]
        verbose: bool,
    },
}

#[derive(Subcommand)]
enum HfCmd {
    Search { query: String },
    Files { repo: String },
    /// Download one quant (all split parts) and the best mmproj.
    Download {
        repo: String,
        quant: String,
        #[arg(long)]
        no_mmproj: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum KindArg {
    Llama,
    Sd,
}

impl From<KindArg> for EngineKind {
    fn from(k: KindArg) -> Self {
        match k {
            KindArg::Llama => EngineKind::Llama,
            KindArg::Sd => EngineKind::Sd,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum BackendArg {
    Cuda,
    Rocm,
    Vulkan,
    Metal,
    Sycl,
    Cpu,
}

impl From<BackendArg> for Backend {
    fn from(b: BackendArg) -> Self {
        match b {
            BackendArg::Cuda => Backend::Cuda,
            BackendArg::Rocm => Backend::Rocm,
            BackendArg::Vulkan => Backend::Vulkan,
            BackendArg::Metal => Backend::Metal,
            BackendArg::Sycl => Backend::Sycl,
            BackendArg::Cpu => Backend::Cpu,
        }
    }
}

#[derive(Subcommand)]
enum EngineCmd {
    List {
        #[arg(long, value_enum, default_value = "llama")]
        kind: KindArg,
    },
    /// Download the latest official release build.
    Install {
        #[arg(long, value_enum, default_value = "llama")]
        kind: KindArg,
        /// Default: best backend for this machine.
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    /// Import a release .zip / .tar.gz, or register an existing folder.
    Import {
        path: PathBuf,
        #[arg(long, value_enum, default_value = "llama")]
        kind: KindArg,
    },
}

fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / GIB as f64)
}

fn print_json<T: serde::Serialize>(v: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}

fn progress_printer() -> ProgressFn {
    Arc::new(|p: Progress| {
        let pct = p.total.map(|t| format!("{:5.1}%", p.done as f64 * 100.0 / t.max(1) as f64)).unwrap_or_default();
        let speed = p.speed_bps.map(|s| format!(" {:.1} MB/s", s as f64 / 1e6)).unwrap_or_default();
        eprint!("\r[{}/{}] {} {}{}      ", p.file_index + 1, p.file_count, gb(p.done), pct, speed);
        if p.stage == "done" {
            eprintln!();
        }
    })
}

fn hardware(s: &Settings) -> hardware::HardwareReport {
    let mut hw = hardware::detect(&s.models_dir);
    hw.apply_override(s.gpu_override.as_ref());
    hw
}

fn ctrl_c_token() -> CancellationToken {
    let t = CancellationToken::new();
    let t2 = t.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        t2.cancel();
    });
    t
}

fn shape_for_file(path: &Path, with_vision: bool) -> Result<(ModelShape, gguf::ModelSummary, Option<PathBuf>)> {
    let g = gguf::open(path)?;
    let sum = g.summary();
    if sum.is_mmproj {
        bail!("{} is a vision projector, not a model", path.display());
    }
    if sum.is_diffusion {
        bail!("{} is a diffusion model; use the image page / sd-server", path.display());
    }
    let mmproj = if with_vision { path.parent().and_then(library::find_mmproj) } else { None };
    let mm_bytes = mmproj.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0);
    Ok((ModelShape::from_summary(&sum, mm_bytes), sum, mmproj))
}

fn print_plan(p: &tiering::Plan) {
    println!("{}", p.summary);
    for n in [&p.gpu_note, &p.ram_note, &p.disk_note] {
        if !n.is_empty() {
            println!("  - {n}");
        }
    }
    println!("  GPU {} · RAM {} · disk {} · KV {}", gb(p.gpu_bytes), gb(p.ram_bytes), gb(p.disk_bytes), gb(p.kv_bytes));
    println!("  first load ≈ {:.0}s", p.est_load_secs);
    println!("\nllama-server {}", p.command);
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let s = Settings::load();
    match cli.cmd {
        Cmd::Hw => {
            let hw = hardware(&s);
            if cli.json {
                return print_json(&hw);
            }
            println!("{} {} · {} ({} cores / {} threads)", hw.os, hw.arch, hw.cpu_name, hw.cpu_cores, hw.cpu_threads);
            let ram_kind = match (&hw.ram_type, hw.ram_speed_mts) {
                (Some(t), Some(s)) => format!(" ({t}-{s})"),
                (Some(t), None) => format!(" ({t})"),
                (None, Some(s)) => format!(" ({s} MT/s)"),
                (None, None) => String::new(),
            };
            let bus = hw.ram_bus_bits.map(|b| format!(" · {b}-bit bus")).unwrap_or_default();
            let how = if hw.ram_bandwidth_detected { "" } else { " (typical, not detected)" };
            println!("RAM {} total, {} free{ram_kind}{bus} · ~{:.0} GB/s{how}", gb(hw.ram_total), gb(hw.ram_available), hw.ram_bandwidth_gbs);
            for g in &hw.gpus {
                println!("GPU {:?} {} · {}{} · ~{:.0} GB/s", g.vendor, g.name, gb(g.vram_total), if g.unified_memory { " (shared)" } else { "" }, g.bandwidth_gbs);
            }
            if let Some(d) = &hw.model_disk {
                let media = match d.is_ssd {
                    Some(true) => "SSD",
                    Some(false) => "HDD",
                    None => "unknown type",
                };
                let bus = d.bus.map(|b| format!(" · {b:?}")).unwrap_or_default();
                let link = match (d.pcie_gen, d.pcie_lanes) {
                    (Some(g), Some(n)) => format!(" PCIe {g}.0 x{n}"),
                    _ => String::new(),
                };
                println!("Model disk {} · {} free · {media}{bus}{link} · ~{:.1} GB/s read", d.mount, gb(d.available), d.read_speed_bps() as f64 / 1e9);
            }
            println!("Backends: {}", hw.recommended_backends.iter().map(|b| b.as_str()).collect::<Vec<_>>().join(" > "));
            println!("Data dir: {}", data_dir().display());
        }
        Cmd::Gguf { path } => {
            let g = gguf::open(&path)?;
            let sum = g.summary();
            if cli.json {
                return print_json(&sum);
            }
            println!("{} ({}) · {} · {} layers ({} with KV) · ctx {}", sum.name.as_deref().unwrap_or("?"), sum.arch, sum.quant, sum.block_count, sum.kv_layers, sum.context_length);
            println!("heads {} / kv {} · head_dim {} · experts {}/{}", sum.head_count, sum.head_count_kv, sum.head_dim, sum.expert_used_count, sum.expert_count);
            println!("weights {} in {} file(s){}{}", gb(sum.total_bytes), g.files.len(), if sum.is_diffusion { " · diffusion" } else { "" }, if sum.is_mmproj { " · mmproj" } else { "" });
        }
        Cmd::Catalog => {
            let c = Catalog::load(&data_dir());
            if cli.json {
                return print_json(&c);
            }
            for m in &c.models {
                let moe = m.active_b.map(|a| format!(" (A{a}B)")).unwrap_or_default();
                println!("{:<24} {:>5}B{:<8} {:<8} {}", m.id, m.params_b, moe, m.release, m.description(&s.language));
            }
        }
        Cmd::Plan { gguf: file, catalog, quant, ctx, custom } => {
            let (shape, path) = match (file, catalog) {
                (Some(f), _) => (shape_for_file(&f, true)?.0, f.display().to_string()),
                (None, Some(id)) => {
                    let c = Catalog::load(&data_dir());
                    let m = c.find(&id).with_context(|| format!("no catalog model '{id}'"))?;
                    let q = quant.unwrap();
                    (m.shape(&q), format!("{}-{q}.gguf", m.name.replace(' ', "-")))
                }
                _ => bail!("give --gguf <file> or --catalog <id> --quant <quant>"),
            };
            let hw = hardware(&s);
            let o = PlanOptions { ctx, model_path: path, custom_args: custom, lang: s.language.clone(), port: s.server_port, ..Default::default() };
            let p = tiering::plan(&shape, &Machine::from_report(&hw, &s), &o);
            if cli.json {
                return print_json(&p);
            }
            print_plan(&p);
        }
        Cmd::Hf { cmd } => {
            let hf = HfClient::from_settings(&s);
            match cmd {
                HfCmd::Search { query } => {
                    let r = hf.search(&query).await?;
                    if cli.json {
                        return print_json(&r);
                    }
                    for m in r {
                        println!("{:<60} ↓{:<10} ♥{}", m.id, m.downloads, m.likes);
                    }
                }
                HfCmd::Files { repo } => {
                    let l = hf.repo_listing(&repo).await?;
                    if cli.json {
                        return print_json(&l);
                    }
                    for q in &l.quants {
                        println!("{:<14} {:>9}  {} file(s)", q.quant, gb(q.total_bytes), q.files.len());
                    }
                    for m in &l.mmproj {
                        println!("mmproj         {:>9}  {}", gb(m.size), m.path);
                    }
                }
                HfCmd::Download { repo, quant, no_mmproj } => {
                    let l = hf.repo_listing(&repo).await?;
                    let g = l.quants.iter().find(|g| g.quant.eq_ignore_ascii_case(&quant)).with_context(|| {
                        format!("{repo} has no {quant}; available: {}", l.quants.iter().map(|g| g.quant.as_str()).collect::<Vec<_>>().join(", "))
                    })?;
                    let dir = library::repo_dir(&s.models_dir, &repo);
                    let mut files: Vec<_> = g.files.clone();
                    if !no_mmproj {
                        files.extend(openmodel_core::hf::best_mmproj(&l.mmproj).cloned());
                    }
                    let items: Vec<download::DownloadItem> = files
                        .iter()
                        .map(|f| download::DownloadItem { url: hf.resolve_url(&repo, &f.path), dest: dir.join(f.path.rsplit('/').next().unwrap()), size: Some(f.size) })
                        .collect();
                    let headers: Vec<(String, String)> = hf.auth_header().into_iter().collect();
                    download::download_all(&openmodel_core::http_client(), &items, &headers, &ctrl_c_token(), "hf", &progress_printer()).await?;
                    println!("saved to {}", dir.display());
                }
            }
        }
        Cmd::Models => {
            let l = library::list(&s.models_dir, &data_dir());
            if cli.json {
                return print_json(&l);
            }
            if l.is_empty() {
                println!("no models in {}", s.models_dir.display());
            }
            for m in l {
                println!("{:<50} {:>9} {}{}", m.name, gb(m.size_bytes), m.quant.unwrap_or_default(), if m.mmproj.is_some() { " +vision" } else { "" });
            }
        }
        Cmd::Engine { cmd } => match cmd {
            EngineCmd::List { kind } => {
                let l = engine::list_installed(&s, kind.into());
                if cli.json {
                    return print_json(&l);
                }
                for e in l {
                    println!("{:<12} {:<8} {}", e.version, e.backend.as_str(), e.binary.display());
                }
            }
            EngineCmd::Install { kind, backend } => {
                let backend: Backend = match backend {
                    Some(b) => b.into(),
                    None => s.backend.unwrap_or_else(|| hardware(&s).recommended_backends[0]),
                };
                eprintln!("installing {} ({})", EngineKind::from(kind).repo(), backend.as_str());
                let e = engine::install(&s, kind.into(), backend, &ctrl_c_token(), "engine", &progress_printer()).await?;
                println!("installed {} → {}", e.version, e.binary.display());
            }
            EngineCmd::Import { path, kind } => {
                let e = if path.is_dir() { engine::import_dir(&s, kind.into(), &path, None)? } else { engine::import_archive(&s, kind.into(), &path)? };
                println!("imported {} ({}) → {}", e.version, e.backend.as_str(), e.binary.display());
            }
        },
        Cmd::Serve { path, ctx, lan, port, custom, no_vision, verbose } => {
            let eng = engine::list_installed(&s, EngineKind::Llama).into_iter().next().context("no llama.cpp engine installed; run `openmodel engine install` first")?;
            let (shape, _sum, mmproj) = shape_for_file(&path, !no_vision)?;
            let hw = hardware(&s);
            let machine = Machine::from_report(&hw, &s);
            let custom = if custom.is_empty() { s.custom_args.get(&path.file_name().unwrap_or_default().to_string_lossy().to_string()).cloned().unwrap_or_default() } else { custom };
            let base = PlanOptions {
                ctx,
                model_path: path.display().to_string(),
                mmproj_path: mmproj.map(|p| p.display().to_string()),
                host: "127.0.0.1".into(),
                port: port.unwrap_or(s.server_port),
                custom_args: custom,
                lang: s.language.clone(),
                ..Default::default()
            };
            let srv = server::ManagedServer::new();
            let mut rx = srv.subscribe();
            tokio::spawn(async move {
                while let Ok(e) = rx.recv().await {
                    match e {
                        server::ServerEvent::Log { line } if verbose => eprintln!("{line}"),
                        server::ServerEvent::Note { message } => eprintln!("\n* {message}"),
                        server::ServerEvent::Progress(p) if !verbose => eprint!("\r{:>5.1}% {:?}      ", p.fraction * 100.0, p.stage),
                        _ => {}
                    }
                }
            });
            // --lan opens every LAN address; otherwise the ones chosen in Settings.
            srv.set_exposure(if lan { openmodel_core::net::lan_addresses().iter().filter_map(|a| a.ip.parse().ok()).collect() } else { s.exposed_ips() });
            let plan = srv.start_llama(&eng.binary, shape.total_bytes, |lvl| tiering::plan(&shape, &machine, &PlanOptions { safe_level: lvl, ..base.clone() }), &s.language, s.server_key()).await;
            let plan = match plan {
                Ok(p) => p,
                Err(e) => {
                    if let Some(c) = srv.status().last_crash {
                        eprintln!("\n--- last output ---\n{}", c.tail.join("\n"));
                    }
                    return Err(e);
                }
            };
            let st = srv.status();
            eprintln!();
            print_plan(&plan);
            println!("\nready: http://127.0.0.1:{}/v1  (OpenAI-compatible) · web UI http://127.0.0.1:{}/", st.port, st.port);
            for ip in &st.exposed {
                println!("       http://{ip}:{}/v1", st.port);
            }
            if st.api_key.is_some() {
                println!("API key required (Authorization: Bearer <key>)");
            }
            if let Some((a, b)) = st.gpu_layers {
                println!("llama.cpp offloaded {a}/{b} layers to the GPU");
            }
            println!("Ctrl+C to stop");
            let _ = tokio::signal::ctrl_c().await;
            srv.stop().await;
        }
    }
    Ok(())
}
