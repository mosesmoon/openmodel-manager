//! Hardware detection: CPU, RAM, disk, GPU (NVIDIA / AMD / Intel / Apple).
//!
//! Only standard OS tools are used (nvidia-smi, sysfs, PowerShell CIM/registry,
//! sysctl). Nothing is installed and no driver is touched.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other,
}

/// Inference backend a llama.cpp / sd.cpp build is compiled for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Cuda,
    Rocm,
    Vulkan,
    Metal,
    Sycl,
    Cpu,
}

impl Backend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Rocm => "rocm",
            Backend::Vulkan => "vulkan",
            Backend::Metal => "metal",
            Backend::Sycl => "sycl",
            Backend::Cpu => "cpu",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub name: String,
    /// Dedicated VRAM in bytes (for Apple / APUs: memory usable by the GPU).
    pub vram_total: u64,
    pub vram_free: Option<u64>,
    pub driver: Option<String>,
    /// True when the GPU shares system RAM (Apple Silicon, AMD/Intel iGPU).
    pub unified_memory: bool,
    /// Estimated memory bandwidth in GB/s (used for tok/s estimates).
    pub bandwidth_gbs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskInfo {
    pub mount: String,
    pub total: u64,
    pub available: u64,
    /// None when the OS does not report the media type.
    pub is_ssd: Option<bool>,
}

impl DiskInfo {
    /// Sequential read speed assumed for load-time estimates (bytes/s).
    pub fn read_speed_bps(&self) -> u64 {
        match self.is_ssd {
            Some(false) => 150 * crate::config::MIB,
            _ => 1500 * crate::config::MIB,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareReport {
    pub os: String,
    pub arch: String,
    pub cpu_name: String,
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub ram_total: u64,
    pub ram_available: u64,
    /// "DDR4", "DDR5", "LPDDR5"... when the OS exposes it.
    pub ram_type: Option<String>,
    /// Rough system RAM bandwidth estimate (GB/s).
    pub ram_bandwidth_gbs: f64,
    pub gpus: Vec<GpuInfo>,
    pub model_disk: Option<DiskInfo>,
    /// Backends that make sense on this machine, best first.
    pub recommended_backends: Vec<Backend>,
}

impl HardwareReport {
    /// The GPU used for planning (largest VRAM).
    pub fn primary_gpu(&self) -> Option<&GpuInfo> {
        self.gpus.iter().max_by_key(|g| g.vram_total)
    }

    /// Replace detected GPUs with the user's manual override, if any.
    pub fn apply_override(&mut self, o: Option<&crate::config::GpuOverride>) {
        let Some(o) = o else { return };
        let vram = (o.vram_gb * crate::config::GIB as f64) as u64;
        self.gpus = vec![GpuInfo {
            vendor: o.vendor,
            bandwidth_gbs: estimate_bandwidth(o.vendor, &o.name),
            name: o.name.clone(),
            vram_total: vram,
            vram_free: None,
            driver: None,
            unified_memory: matches!(o.vendor, GpuVendor::Apple),
        }];
        self.recommended_backends = recommend_backends(&self.gpus);
    }
}

/// Live VRAM usage per GPU (bytes used, bytes total), for the stats panel.
pub fn live_vram() -> Vec<(String, u64, u64)> {
    #[allow(unused_mut)]
    let mut out: Vec<(String, u64, u64)> = run(
        "nvidia-smi",
        &["--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
    )
    .map(|s| {
        s.lines()
            .filter_map(|l| {
                let f: Vec<&str> = l.split(',').map(str::trim).collect();
                Some((f.first()?.to_string(), f.get(1)?.parse::<u64>().ok()? * crate::config::MIB, f.get(2)?.parse::<u64>().ok()? * crate::config::MIB))
            })
            .collect()
    })
    .unwrap_or_default();
    #[cfg(target_os = "linux")]
    if let Ok(rd) = std::fs::read_dir("/sys/class/drm") {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if !n.starts_with("card") || n.contains('-') {
                continue;
            }
            let dev = e.path().join("device");
            let num = |f: &str| std::fs::read_to_string(dev.join(f)).ok().and_then(|s| s.trim().parse::<u64>().ok());
            if let (Some(used), Some(total)) = (num("mem_info_vram_used"), num("mem_info_vram_total")) {
                out.push((n, used, total));
            }
        }
    }
    out
}

fn detect_ram_type() -> Option<String> {
    #[cfg(windows)]
    {
        let out = run(
            "powershell",
            &["-NoProfile", "-NonInteractive", "-Command", "(Get-CimInstance Win32_PhysicalMemory | Select-Object -First 1).SMBIOSMemoryType"],
        )?;
        return smbios_memory_type(out.trim().parse().ok()?).map(String::from);
    }
    #[cfg(target_os = "macos")]
    {
        let out = run("system_profiler", &["SPMemoryDataType"])?;
        return out.lines().find_map(|l| l.trim().strip_prefix("Type:").map(|t| t.trim().to_string()));
    }
    #[allow(unreachable_code)]
    None
}

/// SMBIOS type 17 "Memory Type" codes.
pub fn smbios_memory_type(code: u32) -> Option<&'static str> {
    Some(match code {
        24 => "DDR3",
        26 => "DDR4",
        27 => "LPDDR",
        28 => "LPDDR2",
        29 => "LPDDR3",
        30 => "LPDDR4",
        34 => "DDR5",
        35 => "LPDDR5",
        _ => return None,
    })
}

pub fn detect(models_dir: &Path) -> HardwareReport {
    use sysinfo::{Disks, System};
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();

    let cpu_name = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Unknown CPU".into());
    let cpu_threads = sys.cpus().len();
    let cpu_cores = sys.physical_core_count().unwrap_or(cpu_threads);

    let disks = Disks::new_with_refreshed_list();
    let model_disk = best_disk_for(models_dir, &disks);

    let mut gpus = detect_gpus(sys.total_memory());
    gpus.sort_by_key(|g| std::cmp::Reverse(g.vram_total));

    let recommended_backends = recommend_backends(&gpus);
    let ram_bandwidth_gbs = if cfg!(target_os = "macos") {
        gpus.first().map(|g| g.bandwidth_gbs).unwrap_or(100.0)
    } else {
        60.0
    };

    HardwareReport {
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        cpu_name,
        cpu_cores,
        cpu_threads,
        ram_total: sys.total_memory(),
        ram_available: sys.available_memory(),
        ram_type: detect_ram_type(),
        ram_bandwidth_gbs,
        gpus,
        model_disk,
        recommended_backends,
    }
}

fn best_disk_for(path: &Path, disks: &sysinfo::Disks) -> Option<DiskInfo> {
    let mut p = path.to_path_buf();
    while !p.exists() {
        if !p.pop() {
            break;
        }
    }
    let canon = std::fs::canonicalize(&p).unwrap_or(p);
    // Windows canonical paths are verbatim (\\?\C:\...) but mount points are not.
    let canon = match canon.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(rest) if !rest.starts_with("UNC") => std::path::PathBuf::from(rest),
        _ => canon,
    };
    disks
        .list()
        .iter()
        .filter(|d| canon.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| DiskInfo {
            mount: d.mount_point().display().to_string(),
            total: d.total_space(),
            available: d.available_space(),
            is_ssd: match d.kind() {
                sysinfo::DiskKind::SSD => Some(true),
                sysinfo::DiskKind::HDD => Some(false),
                _ => None,
            },
        })
}

pub fn recommend_backends(gpus: &[GpuInfo]) -> Vec<Backend> {
    let mut out = Vec::new();
    for g in gpus {
        let list: &[Backend] = match g.vendor {
            GpuVendor::Nvidia => &[Backend::Cuda, Backend::Vulkan],
            GpuVendor::Amd => {
                if cfg!(target_os = "macos") {
                    &[Backend::Metal]
                } else {
                    &[Backend::Vulkan, Backend::Rocm]
                }
            }
            GpuVendor::Intel => &[Backend::Vulkan, Backend::Sycl],
            GpuVendor::Apple => &[Backend::Metal],
            GpuVendor::Other => &[Backend::Vulkan],
        };
        for b in list {
            if !out.contains(b) {
                out.push(*b);
            }
        }
    }
    out.push(Backend::Cpu);
    out
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let mut c = Command::new(cmd);
    c.args(args);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let out = c.output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn detect_gpus(ram_total: u64) -> Vec<GpuInfo> {
    let mut gpus = detect_nvidia();
    #[cfg(target_os = "linux")]
    gpus.extend(linux_sysfs_gpus(ram_total));
    #[cfg(windows)]
    gpus.extend(windows_gpus());
    #[cfg(target_os = "macos")]
    gpus.extend(macos_gpu(ram_total));
    let _ = ram_total;
    gpus
}

fn detect_nvidia() -> Vec<GpuInfo> {
    let Some(out) = run(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
    ) else {
        return vec![];
    };
    parse_nvidia_smi(&out)
}

pub fn parse_nvidia_smi(out: &str) -> Vec<GpuInfo> {
    out.lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').map(|s| s.trim()).collect();
            if f.len() < 4 {
                return None;
            }
            let total_mib: u64 = f[1].parse().ok()?;
            let free_mib: Option<u64> = f[2].parse().ok();
            Some(GpuInfo {
                vendor: GpuVendor::Nvidia,
                name: f[0].to_string(),
                vram_total: total_mib * 1024 * 1024,
                vram_free: free_mib.map(|m| m * 1024 * 1024),
                driver: Some(f[3].to_string()),
                unified_memory: false,
                bandwidth_gbs: estimate_bandwidth(GpuVendor::Nvidia, f[0]),
            })
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn linux_sysfs_gpus(ram_total: u64) -> Vec<GpuInfo> {
    let mut out = vec![];
    let Ok(rd) = std::fs::read_dir("/sys/class/drm") else {
        return out;
    };
    let read = |p: std::path::PathBuf| std::fs::read_to_string(p).ok().map(|s| s.trim().to_string());
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let dev = e.path().join("device");
        let vendor = match read(dev.join("vendor")).as_deref() {
            Some("0x1002") => GpuVendor::Amd,
            Some("0x8086") => GpuVendor::Intel,
            _ => continue, // NVIDIA handled by nvidia-smi
        };
        let vram: u64 = read(dev.join("mem_info_vram_total")).and_then(|s| s.parse().ok()).unwrap_or(0);
        let vram_used: Option<u64> = read(dev.join("mem_info_vram_used")).and_then(|s| s.parse().ok());
        let gtt: u64 = read(dev.join("mem_info_gtt_total")).and_then(|s| s.parse().ok()).unwrap_or(0);
        let pretty = read(dev.join("product_name"))
            .filter(|s| !s.is_empty())
            .or_else(|| lspci_name(&dev))
            .unwrap_or_else(|| format!("{:?} GPU", vendor));
        // APU heuristic: small carve-out VRAM with a large GTT window.
        let unified = vram < 4 * crate::config::GIB && gtt > vram;
        let usable = if unified { (vram + gtt).min(ram_total * 3 / 4) } else { vram };
        if usable == 0 {
            continue;
        }
        out.push(GpuInfo {
            vendor,
            bandwidth_gbs: if unified { 90.0 } else { estimate_bandwidth(vendor, &pretty) },
            name: pretty,
            vram_total: usable,
            vram_free: if unified { None } else { vram_used.map(|u| vram.saturating_sub(u)) },
            driver: None,
            unified_memory: unified,
        });
    }
    out
}

#[cfg(target_os = "linux")]
fn lspci_name(dev: &Path) -> Option<String> {
    let link = std::fs::read_link(dev).ok()?;
    let slot = link.file_name()?.to_string_lossy().to_string();
    let out = run("lspci", &["-s", &slot])?;
    out.split(": ").nth(1).map(|s| s.trim().to_string())
}

#[cfg(windows)]
fn windows_gpus() -> Vec<GpuInfo> {
    // Win32_VideoController.AdapterRAM is a uint32 (caps at 4 GB), so read the
    // 64-bit qwMemorySize from the display class registry keys instead.
    let script = r#"
$ErrorActionPreference='SilentlyContinue'
Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' |
 ForEach-Object { $p = Get-ItemProperty $_.PSPath
  if ($p.DriverDesc) { "{0}|{1}|{2}" -f $p.DriverDesc, $p.'HardwareInformation.qwMemorySize', $p.DriverVersion } }
"#;
    let Some(out) = run("powershell", &["-NoProfile", "-NonInteractive", "-Command", script]) else {
        return vec![];
    };
    let mut seen = std::collections::HashSet::new();
    out.lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.trim().split('|').collect();
            if f.len() < 3 || !seen.insert(f[0].to_string()) {
                return None;
            }
            let name = f[0].to_string();
            let lower = name.to_lowercase();
            let vendor = if lower.contains("nvidia") {
                return None; // nvidia-smi already covered it
            } else if lower.contains("amd") || lower.contains("radeon") {
                GpuVendor::Amd
            } else if lower.contains("intel") || lower.contains("arc") {
                GpuVendor::Intel
            } else {
                return None; // virtual / remote display adapters
            };
            let mem: u64 = f[1].parse().unwrap_or(0);
            let unified = mem < 4 * crate::config::GIB;
            Some(GpuInfo {
                vendor,
                bandwidth_gbs: if unified { 90.0 } else { estimate_bandwidth(vendor, &name) },
                name,
                vram_total: mem,
                vram_free: None,
                driver: Some(f[2].to_string()),
                unified_memory: unified,
            })
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn macos_gpu(ram_total: u64) -> Vec<GpuInfo> {
    let brand = run("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default();
    let brand = brand.trim().to_string();
    if !brand.starts_with("Apple") {
        return vec![];
    }
    // Metal's recommendedMaxWorkingSetSize is ~2/3 of RAM below 36 GB and ~3/4 above.
    let frac = if ram_total > 36 * crate::config::GIB { 0.75 } else { 0.66 };
    vec![GpuInfo {
        vendor: GpuVendor::Apple,
        bandwidth_gbs: estimate_bandwidth(GpuVendor::Apple, &brand),
        name: brand,
        vram_total: (ram_total as f64 * frac) as u64,
        vram_free: None,
        driver: None,
        unified_memory: true,
    }]
}

/// Memory bandwidth lookup (GB/s). Approximate public specs; used only for estimates.
pub fn estimate_bandwidth(vendor: GpuVendor, name: &str) -> f64 {
    let n = name.to_lowercase();
    const TABLE: &[(&str, f64)] = &[
        ("5090", 1792.0), ("5080", 960.0), ("5070 ti", 896.0), ("5070", 672.0), ("5060 ti", 448.0), ("5060", 448.0),
        ("4090", 1008.0), ("4080", 717.0), ("4070 ti super", 672.0), ("4070 ti", 504.0), ("4070 super", 504.0),
        ("4070", 504.0), ("4060 ti", 288.0), ("4060", 272.0),
        ("3090 ti", 1008.0), ("3090", 936.0), ("3080 ti", 912.0), ("3080", 760.0), ("3070", 448.0), ("3060", 360.0),
        ("2080 ti", 616.0), ("2080", 448.0), ("2070", 448.0), ("2060", 336.0),
        ("h100", 3350.0), ("a100", 1935.0), ("l40", 864.0), ("a6000", 768.0), ("l4", 300.0), ("t4", 320.0),
        ("7900 xtx", 960.0), ("7900 xt", 800.0), ("7800 xt", 624.0), ("7700 xt", 432.0), ("7600", 288.0),
        ("9070 xt", 640.0), ("9070", 640.0), ("9060 xt", 320.0), ("8060s", 256.0), ("890m", 120.0), ("780m", 90.0),
        ("arc b580", 456.0), ("arc a770", 560.0), ("arc a750", 512.0),
        ("m4 max", 546.0), ("m4 pro", 273.0), ("m4", 120.0), ("m3 ultra", 819.0), ("m3 max", 400.0),
        ("m3 pro", 150.0), ("m3", 100.0), ("m2 ultra", 800.0), ("m2 max", 400.0), ("m2 pro", 200.0),
        ("m2", 100.0), ("m1 ultra", 800.0), ("m1 max", 400.0), ("m1 pro", 200.0), ("m1", 68.0),
    ];
    for (k, v) in TABLE {
        if n.contains(k) {
            return *v;
        }
    }
    match vendor {
        GpuVendor::Nvidia => 400.0,
        GpuVendor::Amd => 400.0,
        GpuVendor::Intel => 250.0,
        GpuVendor::Apple => 100.0,
        GpuVendor::Other => 100.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvidia_smi() {
        let g = parse_nvidia_smi("NVIDIA GeForce RTX 3080 Ti, 12288, 11000, 576.02\n");
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].vram_total, 12288 * 1024 * 1024);
        assert_eq!(g[0].bandwidth_gbs, 912.0);
    }

    #[test]
    fn smbios_codes() {
        assert_eq!(smbios_memory_type(34), Some("DDR5"));
        assert_eq!(smbios_memory_type(0), None);
    }

    #[test]
    fn amd_prefers_vulkan_then_rocm() {
        let g = vec![GpuInfo {
            vendor: GpuVendor::Amd,
            name: "AMD Radeon 8060S".into(),
            vram_total: 64 << 30,
            vram_free: None,
            driver: None,
            unified_memory: true,
            bandwidth_gbs: estimate_bandwidth(GpuVendor::Amd, "AMD Radeon 8060S"),
        }];
        assert_eq!(g[0].bandwidth_gbs, 256.0);
        if !cfg!(target_os = "macos") {
            assert_eq!(recommend_backends(&g), vec![Backend::Vulkan, Backend::Rocm, Backend::Cpu]);
        }
    }

    #[test]
    fn backend_order() {
        let g = parse_nvidia_smi("NVIDIA GeForce RTX 4090, 24564, 23000, 580.1");
        assert_eq!(recommend_backends(&g), vec![Backend::Cuda, Backend::Vulkan, Backend::Cpu]);
    }
}
