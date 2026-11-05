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

/// How the model drive is attached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiskBus {
    Nvme,
    Sata,
    Usb,
    Thunderbolt,
    /// Apple Silicon internal storage ("Apple Fabric").
    Apple,
    /// eMMC / SD card.
    Mmc,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskInfo {
    pub mount: String,
    pub total: u64,
    pub available: u64,
    /// None when the OS does not report the media type.
    pub is_ssd: Option<bool>,
    #[serde(default)]
    pub bus: Option<DiskBus>,
    /// PCIe generation (3 = PCIe 3.0) and lane count of an NVMe drive's link.
    #[serde(default)]
    pub pcie_gen: Option<u8>,
    #[serde(default)]
    pub pcie_lanes: Option<u8>,
    /// Estimated sequential read speed (bytes/s), from the fields above.
    #[serde(default)]
    pub read_bps: u64,
}

impl DiskInfo {
    /// Sequential read speed assumed for load-time and disk-spill estimates (bytes/s).
    pub fn read_speed_bps(&self) -> u64 {
        if self.read_bps > 0 {
            self.read_bps
        } else {
            estimate_disk_read_bps(self.is_ssd, self.bus, self.pcie_gen, self.pcie_lanes)
        }
    }
}

const MB: u64 = 1_000_000;

/// Usable bandwidth of one PCIe lane, GB/s (after line encoding).
fn pcie_lane_gbs(gen: u8) -> Option<f64> {
    Some(match gen {
        1 => 0.25,
        2 => 0.5,
        3 => 0.985,
        4 => 1.969,
        5 => 3.938,
        6 => 7.563,
        _ => return None,
    })
}

/// Typical sequential read speed for a drive. NVMe drives reach about 85 % of
/// their PCIe link; the other figures are common real-world values.
pub fn estimate_disk_read_bps(is_ssd: Option<bool>, bus: Option<DiskBus>, pcie_gen: Option<u8>, pcie_lanes: Option<u8>) -> u64 {
    match (is_ssd, bus) {
        (Some(false), Some(DiskBus::Usb)) => 120 * MB,
        (Some(false), _) => 150 * crate::config::MIB,
        (_, Some(DiskBus::Nvme)) => match (pcie_gen.and_then(pcie_lane_gbs), pcie_lanes) {
            (Some(lane), Some(n)) if n > 0 => (lane * n as f64 * 0.85 * 1e9) as u64,
            // Unknown link: a mid-range PCIe 3.0 x4 drive.
            _ => 3000 * MB,
        },
        (_, Some(DiskBus::Sata)) => 530 * MB,
        (_, Some(DiskBus::Usb)) => 400 * MB,
        (_, Some(DiskBus::Thunderbolt)) => 2500 * MB,
        (_, Some(DiskBus::Apple)) => 3000 * MB,
        (_, Some(DiskBus::Mmc)) => 250 * MB,
        _ => 1500 * crate::config::MIB,
    }
}

/// One installed memory module, as reported by SMBIOS (type 17).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MemModule {
    /// "DDR4", "DDR5", "LPDDR5"...
    pub kind: Option<String>,
    /// Running (configured) speed in MT/s, or the rated speed when unknown.
    pub speed_mts: Option<u32>,
    /// Data bus width in bits (64 for a DIMM; 16/32 for LPDDR packages).
    pub data_width: Option<u32>,
}

/// System RAM bandwidth worked out from the installed modules.
#[derive(Debug, Clone, PartialEq)]
pub struct RamBandwidth {
    pub gbs: f64,
    pub speed_mts: u32,
    pub bus_bits: u32,
}

/// Memory channels the CPU's platform has; extra DIMMs share a channel.
fn platform_channels(cpu_name: &str) -> u32 {
    let n = cpu_name.to_lowercase();
    if n.contains("epyc") {
        12
    } else if n.contains("threadripper") {
        if n.contains("pro") { 8 } else { 4 }
    } else if n.contains("xeon") {
        if n.contains("xeon(r) e-") || n.contains("xeon e-") {
            2
        } else if n.contains("w-2") {
            4
        } else {
            8
        }
    } else {
        2
    }
}

/// Peak bandwidth = transfer rate x bus width. DIMMs fill one 64-bit channel
/// each (up to the platform's channel count); soldered LPDDR reports its
/// packages, whose widths add up to the bus.
pub fn ram_bandwidth(modules: &[MemModule], cpu_name: &str) -> Option<RamBandwidth> {
    let mods: Vec<&MemModule> = modules.iter().filter(|m| m.speed_mts.is_some_and(|s| s > 0)).collect();
    let speed = mods.iter().filter_map(|m| m.speed_mts).min()?;
    let lpddr = mods.iter().any(|m| m.kind.as_deref().is_some_and(|k| k.to_uppercase().starts_with("LPDDR")));
    let bus_bits = if lpddr {
        let widths: Option<u32> = mods.iter().map(|m| m.data_width.filter(|w| *w > 0)).sum();
        widths.unwrap_or(128).clamp(32, 1024)
    } else {
        (mods.len() as u32).min(platform_channels(cpu_name)) * 64
    };
    let gbs = speed as f64 * bus_bits as f64 / 8.0 / 1000.0;
    (5.0..3000.0).contains(&gbs).then_some(RamBandwidth { gbs, speed_mts: speed, bus_bits })
}

/// Bandwidth guess from the memory type alone (typical dual-channel desktop
/// or 128-bit laptop configuration).
pub fn typical_ram_bandwidth(kind: Option<&str>) -> f64 {
    let k = kind.unwrap_or("").to_uppercase();
    match k.as_str() {
        "DDR3" => 25.6,
        "DDR4" => 51.2,
        "DDR5" => 89.6,
        k if k.starts_with("LPDDR4") => 68.3,
        k if k.starts_with("LPDDR5") => 102.4,
        _ => 60.0,
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
    /// System RAM bandwidth (GB/s): from the modules' speed and channels when
    /// they could be read, otherwise a typical value for `ram_type`.
    pub ram_bandwidth_gbs: f64,
    /// Running speed of the memory in MT/s, when detected.
    #[serde(default)]
    pub ram_speed_mts: Option<u32>,
    /// Total memory bus width in bits (64 per channel), when detected.
    #[serde(default)]
    pub ram_bus_bits: Option<u32>,
    /// True when `ram_bandwidth_gbs` comes from the detected modules.
    #[serde(default)]
    pub ram_bandwidth_detected: bool,
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

/// Installed memory modules, plus the memory type when only that is known
/// (Apple Silicon reports a type but no modules).
fn detect_memory() -> (Vec<MemModule>, Option<String>) {
    #[cfg(windows)]
    {
        let out = run(
            "powershell",
            &[
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_PhysicalMemory | ForEach-Object { '{0}|{1}|{2}|{3}' -f $_.SMBIOSMemoryType, $_.ConfiguredClockSpeed, $_.Speed, $_.DataWidth }",
            ],
        );
        return (out.map(|o| parse_windows_memory(&o)).unwrap_or_default(), None);
    }
    #[cfg(target_os = "linux")]
    {
        // systemd's udev rules copy the SMBIOS memory table into a file anyone
        // can read; dmidecode needs root and is only the last resort.
        let found = std::fs::read_to_string("/run/udev/data/+dmi:id")
            .ok()
            .map(|s| parse_udev_memory(&s))
            .filter(|m| !m.is_empty())
            .or_else(|| {
                run("udevadm", &["info", "--query=property", "--path=/sys/devices/virtual/dmi/id"])
                    .map(|s| parse_udev_memory(&s))
                    .filter(|m| !m.is_empty())
            })
            .or_else(|| run("dmidecode", &["--type", "17"]).map(|s| parse_dmidecode_memory(&s)).filter(|m| !m.is_empty()));
        return (found.unwrap_or_default(), None);
    }
    #[cfg(target_os = "macos")]
    {
        let out = run("system_profiler", &["SPMemoryDataType"]).unwrap_or_default();
        let kind = out.lines().find_map(|l| l.trim().strip_prefix("Type:").map(|t| t.trim().to_string()));
        return (parse_system_profiler_memory(&out), kind);
    }
    #[allow(unreachable_code)]
    (vec![], None)
}

fn first_number(s: &str) -> Option<u32> {
    s.split_whitespace().next()?.parse::<u32>().ok().filter(|n| *n > 0)
}

fn known_type(s: &str) -> Option<String> {
    let s = s.trim();
    (!s.is_empty() && !matches!(s, "Unknown" | "Other" | "<OUT OF SPEC>")).then(|| s.to_string())
}

/// `MEMORY_DEVICE_<n>_<FIELD>=value` lines from udev (`E:` prefix optional).
pub fn parse_udev_memory(text: &str) -> Vec<MemModule> {
    use std::collections::BTreeMap;
    let mut by: BTreeMap<u32, BTreeMap<String, String>> = BTreeMap::new();
    for l in text.lines() {
        let l = l.trim();
        let l = l.strip_prefix("E:").unwrap_or(l);
        let Some((k, v)) = l.split_once('=') else { continue };
        let Some((idx, field)) = k.strip_prefix("MEMORY_DEVICE_").and_then(|r| r.split_once('_')) else { continue };
        let Ok(i) = idx.parse::<u32>() else { continue };
        by.entry(i).or_default().insert(field.to_string(), v.trim().to_string());
    }
    by.into_values()
        .filter(|f| f.get("PRESENT").map(String::as_str) != Some("0") && f.get("SIZE").map(String::as_str) != Some("0"))
        .map(|f| {
            let num = |k: &str| f.get(k).and_then(|v| first_number(v));
            MemModule {
                kind: f.get("TYPE").and_then(|t| known_type(t)),
                speed_mts: num("CONFIGURED_SPEED_MTS").or_else(|| num("SPEED_MTS")),
                data_width: num("DATA_WIDTH"),
            }
        })
        .collect()
}

/// `dmidecode --type 17` output.
pub fn parse_dmidecode_memory(text: &str) -> Vec<MemModule> {
    let mut out = Vec::new();
    let mut cur: Option<(MemModule, bool, Option<u32>)> = None;
    let flush = |c: Option<(MemModule, bool, Option<u32>)>, out: &mut Vec<MemModule>| {
        if let Some((mut m, empty, rated)) = c {
            if !empty {
                m.speed_mts = m.speed_mts.or(rated);
                out.push(m);
            }
        }
    };
    for l in text.lines() {
        let t = l.trim();
        if t == "Memory Device" {
            flush(cur.take(), &mut out);
            cur = Some((MemModule::default(), false, None));
            continue;
        }
        let Some((m, empty, rated)) = cur.as_mut() else { continue };
        let Some((k, v)) = t.split_once(':') else { continue };
        let v = v.trim();
        match k.trim() {
            "Size" => *empty = v.contains("No Module") || v.starts_with('0'),
            "Type" => m.kind = known_type(v),
            "Configured Memory Speed" | "Configured Clock Speed" => m.speed_mts = first_number(v),
            "Speed" => *rated = first_number(v),
            "Data Width" => m.data_width = first_number(v),
            _ => {}
        }
    }
    flush(cur, &mut out);
    out
}

/// `SMBIOSMemoryType|ConfiguredClockSpeed|Speed|DataWidth` lines from Win32_PhysicalMemory.
pub fn parse_windows_memory(text: &str) -> Vec<MemModule> {
    text.lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.trim().split('|').collect();
            if f.len() < 4 {
                return None;
            }
            Some(MemModule {
                kind: f[0].trim().parse().ok().and_then(smbios_memory_type).map(String::from),
                speed_mts: first_number(f[1]).or_else(|| first_number(f[2])),
                data_width: first_number(f[3]),
            })
        })
        .collect()
}

/// `system_profiler SPMemoryDataType` on Intel Macs (one block per DIMM slot).
pub fn parse_system_profiler_memory(text: &str) -> Vec<MemModule> {
    let mut out: Vec<MemModule> = Vec::new();
    let mut in_module = false;
    for l in text.lines() {
        let Some((k, v)) = l.trim().split_once(':') else { continue };
        let v = v.trim();
        match k {
            "Size" => {
                in_module = !v.is_empty() && v != "Empty";
                if in_module {
                    out.push(MemModule { data_width: Some(64), ..Default::default() });
                }
            }
            "Type" if in_module => out.last_mut().unwrap().kind = known_type(v),
            "Speed" if in_module => out.last_mut().unwrap().speed_mts = first_number(v),
            _ => {}
        }
    }
    out
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
        36 => "HBM3",
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

    let (modules, type_hint) = detect_memory();
    let ram_type = modules.iter().find_map(|m| m.kind.clone()).or(type_hint);
    let measured = ram_bandwidth(&modules, &cpu_name);
    let apple = gpus.iter().find(|g| g.vendor == GpuVendor::Apple).map(|g| g.bandwidth_gbs);
    let (ram_bandwidth_gbs, ram_bandwidth_detected) = match (apple, &measured) {
        // Apple Silicon: CPU and GPU share the SoC's memory bandwidth.
        (Some(bw), _) => (bw, true),
        (None, Some(m)) => (m.gbs, true),
        (None, None) => (typical_ram_bandwidth(ram_type.as_deref()), false),
    };
    // Integrated AMD / Intel GPUs read from system RAM, so they get its bandwidth.
    if ram_bandwidth_detected || ram_type.is_some() {
        for g in gpus.iter_mut().filter(|g| g.unified_memory && g.vendor != GpuVendor::Apple) {
            g.bandwidth_gbs = ram_bandwidth_gbs;
        }
    }

    HardwareReport {
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        cpu_name,
        cpu_cores,
        cpu_threads,
        ram_total: sys.total_memory(),
        ram_available: sys.available_memory(),
        ram_type,
        ram_bandwidth_gbs,
        ram_speed_mts: measured.as_ref().map(|m| m.speed_mts),
        ram_bus_bits: measured.as_ref().map(|m| m.bus_bits),
        ram_bandwidth_detected,
        gpus,
        model_disk,
        recommended_backends,
    }
}

/// The drive holding `models_dir`, read on its own (when the folder moves).
pub fn model_disk(models_dir: &Path) -> Option<DiskInfo> {
    best_disk_for(models_dir, &sysinfo::Disks::new_with_refreshed_list())
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
        .map(|d| {
            let link = disk_link(d);
            let is_ssd = link.ssd.or(match d.kind() {
                sysinfo::DiskKind::SSD => Some(true),
                sysinfo::DiskKind::HDD => Some(false),
                _ => None,
            });
            DiskInfo {
                mount: d.mount_point().display().to_string(),
                total: d.total_space(),
                available: d.available_space(),
                is_ssd,
                bus: link.bus,
                pcie_gen: link.pcie_gen,
                pcie_lanes: link.pcie_lanes,
                read_bps: estimate_disk_read_bps(is_ssd, link.bus, link.pcie_gen, link.pcie_lanes),
            }
        })
}

/// How a drive is attached, as far as the OS tells.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DiskLink {
    pub bus: Option<DiskBus>,
    pub ssd: Option<bool>,
    pub pcie_gen: Option<u8>,
    pub pcie_lanes: Option<u8>,
}

fn disk_link(d: &sysinfo::Disk) -> DiskLink {
    #[cfg(target_os = "linux")]
    {
        // sysinfo reports the device node (/dev/nvme0n1p2, /dev/mapper/...).
        let dev = std::path::PathBuf::from(d.name());
        let real = std::fs::canonicalize(&dev).unwrap_or(dev);
        return real
            .file_name()
            .and_then(|n| linux_block_link(Path::new("/sys"), &n.to_string_lossy()))
            .unwrap_or_default();
    }
    #[cfg(windows)]
    {
        let Some(letter) = d.mount_point().to_str().and_then(|m| m.chars().next()).filter(|c| c.is_ascii_alphabetic()) else {
            return DiskLink::default();
        };
        let script = format!(
            r#"$ErrorActionPreference='SilentlyContinue'
$d = Get-Partition -DriveLetter {letter} | Get-Disk
$p = Get-PhysicalDisk | Where-Object {{ $_.DeviceId -eq [string]$d.Number }}
'disk|{{0}}|{{1}}' -f $d.BusType, $p.MediaType
if ("$($d.BusType)" -in 'NVMe','17') {{
 Get-PnpDevice -PresentOnly | Where-Object {{ $_.InstanceId -like 'PCI\*' -and $_.FriendlyName -match 'NVM ?Express|NVMe' }} | ForEach-Object {{
  $s = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_PciDevice_CurrentLinkSpeed).Data
  $w = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_PciDevice_CurrentLinkWidth).Data
  'link|{{0}}|{{1}}' -f $s, $w }} }}"#
        );
        return run("powershell", &["-NoProfile", "-NonInteractive", "-Command", &script])
            .map(|o| parse_windows_disk(&o))
            .unwrap_or_default();
    }
    #[cfg(target_os = "macos")]
    {
        let mount = d.mount_point().to_string_lossy().to_string();
        return run("diskutil", &["info", &mount]).map(|o| parse_diskutil(&o)).unwrap_or_default();
    }
    #[allow(unreachable_code)]
    {
        let _ = d;
        DiskLink::default()
    }
}

/// "16.0 GT/s PCIe" -> PCIe generation 4.
pub fn pcie_gen_from_speed(s: &str) -> Option<u8> {
    let gts: f64 = s.split_whitespace().next()?.parse().ok()?;
    Some(match gts {
        x if x < 3.0 => 1,
        x if x < 6.0 => 2,
        x if x < 10.0 => 3,
        x if x < 20.0 => 4,
        x if x < 40.0 => 5,
        _ => 6,
    })
}

/// Bus, media type and PCIe link of the block device `name` (a partition,
/// whole disk, or dm/md device) under a sysfs tree rooted at `sys`.
pub fn linux_block_link(sys: &Path, name: &str) -> Option<DiskLink> {
    let sys = std::fs::canonicalize(sys).unwrap_or_else(|_| sys.to_path_buf());
    linux_block_link_at(&sys, name, 0)
}

fn linux_block_link_at(sys: &Path, name: &str, depth: u8) -> Option<DiskLink> {
    if depth > 4 {
        return None;
    }
    let dir = std::fs::canonicalize(sys.join("class/block").join(name)).ok()?;
    // dm-crypt, LVM and md RAID: follow the device underneath.
    if let Ok(rd) = std::fs::read_dir(dir.join("slaves")) {
        let mut under: Vec<String> = rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        under.sort();
        if let Some(first) = under.first() {
            return linux_block_link_at(sys, first, depth + 1);
        }
    }
    let disk = if dir.join("partition").exists() { dir.parent()?.to_path_buf() } else { dir };
    let disk_name = disk.file_name()?.to_string_lossy().into_owned();
    let read = |p: &Path| std::fs::read_to_string(p).ok().map(|s| s.trim().to_string());
    let ssd = read(&disk.join("queue/rotational")).map(|r| r == "0");
    let path = disk.to_string_lossy();
    let bus = if path.contains("/usb") {
        DiskBus::Usb
    } else if disk_name.starts_with("nvme") {
        DiskBus::Nvme
    } else if disk_name.starts_with("mmcblk") {
        DiskBus::Mmc
    } else if path.contains("/ata") {
        DiskBus::Sata
    } else {
        DiskBus::Other
    };
    let mut link = DiskLink { bus: Some(bus), ssd, ..Default::default() };
    if bus == DiskBus::Nvme {
        // The PCIe function is an ancestor of the namespace, except with native
        // NVMe multipath (namespace under nvme-subsystem); then go through the
        // controller: nvme0n1 / multipath path nvme0c0n1 -> nvme0.
        let mut pci = disk.ancestors().find(|a| a.join("current_link_speed").exists() && a.starts_with(sys)).map(Path::to_path_buf);
        if pci.is_none() {
            let path_name = std::fs::read_dir(disk.join("multipath"))
                .ok()
                .and_then(|rd| rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).min())
                .unwrap_or_else(|| disk_name.clone());
            let digits: String = path_name.trim_start_matches("nvme").chars().take_while(char::is_ascii_digit).collect();
            pci = std::fs::canonicalize(sys.join("class/nvme").join(format!("nvme{digits}")).join("device")).ok();
        }
        if let Some(pci) = pci {
            link.pcie_gen = read(&pci.join("current_link_speed")).and_then(|s| pcie_gen_from_speed(&s));
            link.pcie_lanes = read(&pci.join("current_link_width")).and_then(|s| s.parse().ok()).filter(|n| *n > 0);
        }
    }
    Some(link)
}

/// Output of the Windows disk script: `disk|<BusType>|<MediaType>` then one
/// `link|<gen>|<width>` line per NVMe controller. Names or numeric codes.
pub fn parse_windows_disk(text: &str) -> DiskLink {
    let mut link = DiskLink::default();
    let mut links: Vec<(Option<u8>, Option<u8>)> = Vec::new();
    for l in text.lines() {
        let f: Vec<&str> = l.trim().split('|').map(str::trim).collect();
        match f.as_slice() {
            ["disk", bus, media] => {
                link.bus = match bus.to_ascii_lowercase().as_str() {
                    "nvme" | "17" => Some(DiskBus::Nvme),
                    "sata" | "11" | "ata" | "3" => Some(DiskBus::Sata),
                    "usb" | "7" => Some(DiskBus::Usb),
                    "sd" | "12" | "mmc" | "13" => Some(DiskBus::Mmc),
                    "" => None,
                    _ => Some(DiskBus::Other),
                };
                link.ssd = match media.to_ascii_lowercase().as_str() {
                    "ssd" | "4" | "scm" | "5" => Some(true),
                    "hdd" | "3" => Some(false),
                    _ => None,
                };
            }
            ["link", gen, width] => links.push((gen.parse().ok().filter(|g| (1..=6).contains(g)), width.parse().ok().filter(|w| *w > 0))),
            _ => {}
        }
    }
    if link.bus == Some(DiskBus::Nvme) {
        if link.ssd.is_none() {
            link.ssd = Some(true);
        }
        // Several NVMe controllers can't be told apart from the drive letter;
        // only trust the link when they all agree.
        if let Some(first) = links.first().copied() {
            if links.iter().all(|l| *l == first) {
                (link.pcie_gen, link.pcie_lanes) = first;
            }
        }
    }
    link
}

/// `diskutil info <mount>` on macOS.
pub fn parse_diskutil(text: &str) -> DiskLink {
    let mut link = DiskLink::default();
    for l in text.lines() {
        let Some((k, v)) = l.trim().split_once(':') else { continue };
        let v = v.trim();
        match k.trim() {
            "Protocol" => {
                link.bus = Some(match v {
                    "Apple Fabric" => DiskBus::Apple,
                    "PCI-Express" | "PCI" | "NVMExpress" => DiskBus::Nvme,
                    "SATA" => DiskBus::Sata,
                    "USB" => DiskBus::Usb,
                    "Thunderbolt" => DiskBus::Thunderbolt,
                    "Secure Digital" => DiskBus::Mmc,
                    _ => DiskBus::Other,
                })
            }
            "Solid State" => link.ssd = Some(v.eq_ignore_ascii_case("yes")),
            _ => {}
        }
    }
    link
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

    fn dimm(kind: &str, mts: u32, width: u32) -> MemModule {
        MemModule { kind: Some(kind.into()), speed_mts: Some(mts), data_width: Some(width) }
    }

    #[test]
    fn ram_bandwidth_from_modules() {
        // Dual-channel DDR5-5600: 5600 MT/s x 128 bit = 89.6 GB/s.
        let two = vec![dimm("DDR5", 5600, 64); 2];
        let bw = ram_bandwidth(&two, "AMD Ryzen 7 9700X").unwrap();
        assert_eq!((bw.speed_mts, bw.bus_bits), (5600, 128));
        assert!((bw.gbs - 89.6).abs() < 0.01);
        // A single DIMM is single-channel: half the bandwidth.
        assert!((ram_bandwidth(&two[..1], "AMD Ryzen 7 9700X").unwrap().gbs - 44.8).abs() < 0.01);
        // Four DIMMs on a desktop still share two channels; the slowest sets the speed.
        let mut four = vec![dimm("DDR4", 3200, 64); 4];
        four[3].speed_mts = Some(2933);
        let bw = ram_bandwidth(&four, "Intel(R) Core(TM) i7-12700K").unwrap();
        assert_eq!((bw.speed_mts, bw.bus_bits), (2933, 128));
        // ...but fill four channels on a Threadripper.
        assert_eq!(ram_bandwidth(&four, "AMD Ryzen Threadripper 7960X").unwrap().bus_bits, 256);
        // Soldered LPDDR5X: the package widths add up to the bus (8 x 32 = 256 bit).
        let strix = vec![dimm("LPDDR5", 8000, 32); 8];
        assert!((ram_bandwidth(&strix, "AMD RYZEN AI MAX+ 395").unwrap().gbs - 256.0).abs() < 0.01);
        // No speed: nothing to work from.
        assert!(ram_bandwidth(&[MemModule { kind: Some("DDR4".into()), ..Default::default() }], "x").is_none());
        assert_eq!(typical_ram_bandwidth(Some("DDR5")), 89.6);
        assert_eq!(typical_ram_bandwidth(None), 60.0);
    }

    #[test]
    fn parses_udev_memory() {
        let text = "E:MEMORY_ARRAY_NUM_DEVICES=4\nE:MEMORY_DEVICE_0_PRESENT=1\nE:MEMORY_DEVICE_0_SIZE=17179869184\n\
E:MEMORY_DEVICE_0_TYPE=DDR5\nE:MEMORY_DEVICE_0_DATA_WIDTH=64\nE:MEMORY_DEVICE_0_SPEED_MTS=6000\n\
E:MEMORY_DEVICE_0_CONFIGURED_SPEED_MTS=4800\nE:MEMORY_DEVICE_1_PRESENT=0\nE:MEMORY_DEVICE_2_PRESENT=1\n\
E:MEMORY_DEVICE_2_TYPE=DDR5\nE:MEMORY_DEVICE_2_DATA_WIDTH=64\nE:MEMORY_DEVICE_2_SPEED_MTS=6000\n";
        let m = parse_udev_memory(text);
        assert_eq!(m, vec![dimm("DDR5", 4800, 64), dimm("DDR5", 6000, 64)]);
        // udevadm prints the same without the E: prefix.
        assert_eq!(parse_udev_memory(&text.replace("E:", "")).len(), 2);
    }

    #[test]
    fn parses_dmidecode_memory() {
        let text = "Handle 0x0040, DMI type 17, 92 bytes\nMemory Device\n\tData Width: 64 bits\n\tSize: 16 GB\n\
\tType: DDR4\n\tSpeed: 3200 MT/s\n\tConfigured Memory Speed: 2666 MT/s\n\n\
Handle 0x0041, DMI type 17, 92 bytes\nMemory Device\n\tData Width: Unknown\n\tSize: No Module Installed\n\tType: Unknown\n\tSpeed: Unknown\n\n\
Handle 0x0042, DMI type 17, 92 bytes\nMemory Device\n\tData Width: 64 bits\n\tSize: 16 GB\n\tType: DDR4\n\tSpeed: 3200 MT/s\n\tConfigured Memory Speed: Unknown\n";
        assert_eq!(parse_dmidecode_memory(text), vec![dimm("DDR4", 2666, 64), dimm("DDR4", 3200, 64)]);
    }

    #[test]
    fn parses_windows_memory() {
        let m = parse_windows_memory("34|5600|6000|64\r\n34|5600|6000|64\r\n0|0|2400|64\r\n");
        assert_eq!(m[0], dimm("DDR5", 5600, 64));
        assert_eq!(m[2], MemModule { kind: None, speed_mts: Some(2400), data_width: Some(64) });
    }

    #[test]
    fn parses_system_profiler_memory() {
        let text = "Memory:\n\n    Memory Slots:\n\n      ECC: Disabled\n\n        BANK 0/ChannelA-DIMM0:\n\n\
          Size: 8 GB\n          Type: DDR4\n          Speed: 2667 MHz\n\n        BANK 2/ChannelB-DIMM0:\n\n\
          Size: Empty\n          Type: Empty\n          Speed: Empty\n";
        assert_eq!(parse_system_profiler_memory(text), vec![dimm("DDR4", 2667, 64)]);
        // Apple Silicon has no slots.
        assert!(parse_system_profiler_memory("Memory:\n\n      Memory: 32 GB\n      Type: LPDDR5\n").is_empty());
    }

    #[test]
    fn disk_read_speed() {
        let gb = |b: u64| b as f64 / 1e9;
        // PCIe 4.0 x4 NVMe: 4 x 1.969 GB/s x 0.85 = 6.7 GB/s.
        assert!((gb(estimate_disk_read_bps(Some(true), Some(DiskBus::Nvme), Some(4), Some(4))) - 6.69).abs() < 0.01);
        assert!((gb(estimate_disk_read_bps(Some(true), Some(DiskBus::Nvme), Some(3), Some(4))) - 3.35).abs() < 0.01);
        assert_eq!(estimate_disk_read_bps(Some(true), Some(DiskBus::Nvme), None, None), 3000 * MB);
        assert_eq!(estimate_disk_read_bps(Some(true), Some(DiskBus::Sata), None, None), 530 * MB);
        assert_eq!(estimate_disk_read_bps(Some(false), Some(DiskBus::Sata), None, None), 150 * crate::config::MIB);
        assert_eq!(estimate_disk_read_bps(None, None, None, None), 1500 * crate::config::MIB);
        assert_eq!(pcie_gen_from_speed("16.0 GT/s PCIe"), Some(4));
        assert_eq!(pcie_gen_from_speed("8.0 GT/s PCIe"), Some(3));
        assert_eq!(pcie_gen_from_speed("2.5 GT/s"), Some(1));
        assert_eq!(pcie_gen_from_speed("Unknown"), None);
    }

    #[test]
    fn parses_windows_disk() {
        let l = parse_windows_disk("disk|NVMe|SSD\r\nlink|4|4\r\nlink|4|4\r\n");
        assert_eq!(l, DiskLink { bus: Some(DiskBus::Nvme), ssd: Some(true), pcie_gen: Some(4), pcie_lanes: Some(4) });
        // Controllers disagree: the drive's link is unknown.
        let l = parse_windows_disk("disk|17|4\nlink|4|4\nlink|3|2\n");
        assert_eq!((l.bus, l.ssd, l.pcie_gen), (Some(DiskBus::Nvme), Some(true), None));
        let l = parse_windows_disk("disk|SATA|HDD\n");
        assert_eq!((l.bus, l.ssd), (Some(DiskBus::Sata), Some(false)));
        assert_eq!(parse_windows_disk("disk|RAID|Unspecified\n").bus, Some(DiskBus::Other));
    }

    #[test]
    fn parses_diskutil() {
        let l = parse_diskutil("   Device Node:              /dev/disk3s5\n   Protocol:                 Apple Fabric\n   Solid State:              Yes\n");
        assert_eq!((l.bus, l.ssd), (Some(DiskBus::Apple), Some(true)));
        assert_eq!(parse_diskutil("   Protocol:                 USB\n   Solid State:              No\n").ssd, Some(false));
    }

    #[cfg(unix)]
    #[test]
    fn linux_sysfs_links() {
        use std::os::unix::fs::symlink;
        let t = tempfile::tempdir().unwrap();
        let sys = t.path();
        let mk = |p: &str| std::fs::create_dir_all(sys.join(p)).unwrap();
        let put = |p: &str, v: &str| std::fs::write(sys.join(p), v).unwrap();
        // NVMe on PCIe 4.0 x4, partition nvme0n1p2 inside dm-0 (LUKS).
        let pci = "devices/pci0000:00/0000:00:01.1/0000:01:00.0";
        mk(&format!("{pci}/nvme/nvme0/nvme0n1/nvme0n1p2"));
        mk(&format!("{pci}/nvme/nvme0/nvme0n1/queue"));
        put(&format!("{pci}/current_link_speed"), "16.0 GT/s PCIe\n");
        put(&format!("{pci}/current_link_width"), "4\n");
        put(&format!("{pci}/nvme/nvme0/nvme0n1/queue/rotational"), "0\n");
        put(&format!("{pci}/nvme/nvme0/nvme0n1/nvme0n1p2/partition"), "2\n");
        mk("devices/virtual/block/dm-0/slaves");
        mk("class/block");
        symlink(sys.join(format!("{pci}/nvme/nvme0/nvme0n1/nvme0n1p2")), sys.join("class/block/nvme0n1p2")).unwrap();
        symlink(sys.join("devices/virtual/block/dm-0"), sys.join("class/block/dm-0")).unwrap();
        symlink(sys.join("class/block/nvme0n1p2"), sys.join("devices/virtual/block/dm-0/slaves/nvme0n1p2")).unwrap();
        let want = DiskLink { bus: Some(DiskBus::Nvme), ssd: Some(true), pcie_gen: Some(4), pcie_lanes: Some(4) };
        assert_eq!(linux_block_link(sys, "nvme0n1p2"), Some(want.clone()));
        assert_eq!(linux_block_link(sys, "dm-0"), Some(want));

        // Native NVMe multipath: namespace under nvme-subsystem, link via the controller.
        mk("devices/virtual/nvme-subsystem/nvme-subsys1/nvme1n1/multipath");
        mk("devices/virtual/nvme-subsystem/nvme-subsys1/nvme1n1/queue");
        put("devices/virtual/nvme-subsystem/nvme-subsys1/nvme1n1/queue/rotational", "0");
        std::fs::write(sys.join("devices/virtual/nvme-subsystem/nvme-subsys1/nvme1n1/multipath/nvme1c1n1"), "").unwrap();
        let pci1 = "devices/pci0000:00/0000:00:1d.0/0000:3d:00.0";
        mk(pci1);
        put(&format!("{pci1}/current_link_speed"), "8.0 GT/s PCIe");
        put(&format!("{pci1}/current_link_width"), "2");
        mk("class/nvme/nvme1");
        symlink(sys.join(pci1), sys.join("class/nvme/nvme1/device")).unwrap();
        symlink(sys.join("devices/virtual/nvme-subsystem/nvme-subsys1/nvme1n1"), sys.join("class/block/nvme1n1")).unwrap();
        let l = linux_block_link(sys, "nvme1n1").unwrap();
        assert_eq!((l.pcie_gen, l.pcie_lanes), (Some(3), Some(2)));

        // SATA hard disk and a USB stick.
        let ata = "devices/pci0000:00/0000:00:17.0/ata1/host0/target0:0:0/0:0:0:0/block/sda";
        mk(&format!("{ata}/queue"));
        put(&format!("{ata}/queue/rotational"), "1");
        symlink(sys.join(ata), sys.join("class/block/sda")).unwrap();
        let l = linux_block_link(sys, "sda").unwrap();
        assert_eq!((l.bus, l.ssd), (Some(DiskBus::Sata), Some(false)));
        let usb = "devices/pci0000:00/0000:00:14.0/usb2/2-1/2-1:1.0/host1/target1:0:0/1:0:0:0/block/sdb";
        mk(&format!("{usb}/queue"));
        put(&format!("{usb}/queue/rotational"), "0");
        symlink(sys.join(usb), sys.join("class/block/sdb")).unwrap();
        assert_eq!(linux_block_link(sys, "sdb").unwrap().bus, Some(DiskBus::Usb));
        assert_eq!(linux_block_link(sys, "missing"), None);
    }
}
