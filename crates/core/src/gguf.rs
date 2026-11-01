//! GGUF header / metadata / tensor-info reader.
//!
//! Only the header is read (never the tensor data), so this is fast even for
//! 100 GB files. Split models (`-00001-of-00003.gguf`) are aggregated.
//! Format reference: ggml/docs/gguf.md.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"GGUF";
/// Arrays longer than this (tokenizer vocab, merges) are skipped, only their length is kept.
const MAX_STORED_ARRAY: u64 = 4096;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    /// Array that was too long to keep; holds its length.
    SkippedArray(u64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::UInt(v) => Some(*v),
            Value::Int(v) if *v >= 0 => Some(*v as u64),
            Value::Float(f) if *f >= 0.0 => Some(*f as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
    pub bytes: u64,
}

impl TensorInfo {
    /// Layer index for `blk.N.*` tensors.
    pub fn layer(&self) -> Option<usize> {
        let rest = self.name.strip_prefix("blk.")?;
        rest.split('.').next()?.parse().ok()
    }
    /// MoE expert weights (`ffn_up_exps`, `ffn_gate_up_exps`, ...).
    pub fn is_expert(&self) -> bool {
        self.name.contains("_exps")
    }
}

/// (elements per block, bytes per block) for a ggml tensor type.
pub fn ggml_block(t: u32) -> Option<(u64, u64)> {
    Some(match t {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M
        30 => (1, 2),     // BF16
        34 => (256, 54),  // TQ1_0
        35 => (256, 66),  // TQ2_0
        39 => (32, 17),   // MXFP4
        _ => return None,
    })
}

pub fn tensor_bytes(ggml_type: u32, dims: &[u64]) -> u64 {
    let n: u64 = dims.iter().product();
    match ggml_block(ggml_type) {
        Some((blk, sz)) => n.div_ceil(blk) * sz,
        None => n * 2, // unknown type: assume 16-bit
    }
}

/// `general.file_type` (llama_ftype) to a quant label.
pub fn file_type_name(ft: u64) -> &'static str {
    match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        _ => "unknown",
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// All files that make up this model (one unless split).
    pub files: Vec<PathBuf>,
    /// Sum of file sizes on disk.
    pub file_bytes: u64,
}

struct Rd<R: Read> {
    r: R,
    version: u32,
}

impl<R: Read> Rd<R> {
    fn u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.r.read_exact(&mut b)?;
        Ok(b[0])
    }
    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.r.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    /// Counts and string lengths are u32 in GGUF v1, u64 afterwards.
    fn len(&mut self) -> Result<u64> {
        if self.version == 1 {
            Ok(self.u32()? as u64)
        } else {
            self.u64()
        }
    }
    fn string(&mut self) -> Result<String> {
        let n = self.len()?;
        if n > 64 * 1024 * 1024 {
            bail!("string too long ({n} bytes); file is corrupt");
        }
        let mut b = vec![0u8; n as usize];
        self.r.read_exact(&mut b)?;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }
    fn skip(&mut self, n: u64) -> Result<()> {
        let copied = std::io::copy(&mut (&mut self.r).take(n), &mut std::io::sink())?;
        if copied != n {
            bail!("unexpected end of file");
        }
        Ok(())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::UInt(self.u8()? as u64),
            1 => Value::Int(self.u8()? as i8 as i64),
            2 => {
                let mut b = [0u8; 2];
                self.r.read_exact(&mut b)?;
                Value::UInt(u16::from_le_bytes(b) as u64)
            }
            3 => {
                let mut b = [0u8; 2];
                self.r.read_exact(&mut b)?;
                Value::Int(i16::from_le_bytes(b) as i64)
            }
            4 => Value::UInt(self.u32()? as u64),
            5 => Value::Int(self.u32()? as i32 as i64),
            6 => Value::Float(f32::from_bits(self.u32()?) as f64),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let inner = self.u32()?;
                let n = self.len()?;
                if n > MAX_STORED_ARRAY {
                    self.skip_array(inner, n)?;
                    Value::SkippedArray(n)
                } else {
                    let mut v = Vec::with_capacity(n as usize);
                    for _ in 0..n {
                        v.push(self.value(inner)?);
                    }
                    Value::Array(v)
                }
            }
            10 => Value::UInt(self.u64()?),
            11 => Value::Int(self.u64()? as i64),
            12 => Value::Float(f64::from_bits(self.u64()?)),
            _ => bail!("unknown GGUF value type {ty}"),
        })
    }
    fn skip_array(&mut self, inner: u32, n: u64) -> Result<()> {
        let fixed = match inner {
            0 | 1 | 7 => Some(1),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        };
        if let Some(sz) = fixed {
            return self.skip(sz * n);
        }
        for _ in 0..n {
            match inner {
                8 => {
                    let l = self.len()?;
                    self.skip(l)?;
                }
                _ => {
                    self.value(inner)?;
                }
            }
        }
        Ok(())
    }
}

/// Parse one GGUF file's header.
pub fn read_header(path: &Path) -> Result<GgufFile> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let size = f.metadata()?.len();
    let mut r = Rd { r: BufReader::with_capacity(1 << 20, f), version: 3 };
    let mut magic = [0u8; 4];
    r.r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("{} is not a GGUF file", path.display());
    }
    r.version = r.u32()?;
    if !(1..=3).contains(&r.version) {
        bail!("unsupported GGUF version {}", r.version);
    }
    let n_tensors = r.len()?;
    let n_kv = r.len()?;
    let mut metadata = BTreeMap::new();
    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.u32()?;
        let v = r.value(ty).with_context(|| format!("metadata key {key}"))?;
        metadata.insert(key, v);
    }
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = r.string()?;
        let nd = r.u32()?;
        if nd > 8 {
            bail!("tensor {name} has {nd} dims; file is corrupt");
        }
        let dims = (0..nd).map(|_| r.len()).collect::<Result<Vec<_>>>()?;
        let ggml_type = r.u32()?;
        let offset = r.u64()?;
        let bytes = tensor_bytes(ggml_type, &dims);
        tensors.push(TensorInfo { name, dims, ggml_type, offset, bytes });
    }
    Ok(GgufFile { version: r.version, metadata, tensors, files: vec![path.to_path_buf()], file_bytes: size })
}

/// `(stem, index, count)` for `name-00001-of-00003.gguf`.
pub fn split_parts(file_name: &str) -> Option<(String, u32, u32)> {
    let re = regex::Regex::new(r"(?i)^(.*)-(\d{5})-of-(\d{5})\.gguf$").unwrap();
    let c = re.captures(file_name)?;
    Some((c[1].to_string(), c[2].parse().ok()?, c[3].parse().ok()?))
}

/// Open a model, aggregating all parts if `path` is any part of a split model.
pub fn open(path: &Path) -> Result<GgufFile> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let Some((stem, _, count)) = split_parts(name) else {
        return read_header(path);
    };
    let dir = path.parent().unwrap_or(Path::new("."));
    let part = |i: u32| dir.join(format!("{stem}-{i:05}-of-{count:05}.gguf"));
    let mut first = read_header(&part(1))?;
    for i in 2..=count {
        let p = part(i);
        let g = read_header(&p).with_context(|| format!("split part {i}/{count} missing or unreadable"))?;
        first.tensors.extend(g.tensors);
        first.files.push(p);
        first.file_bytes += g.file_bytes;
    }
    Ok(first)
}

/// Facts needed for planning, distilled from metadata + tensor table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelSummary {
    pub arch: String,
    pub name: Option<String>,
    pub quant: String,
    pub block_count: u64,
    pub context_length: u64,
    pub embedding_length: u64,
    pub head_count: u64,
    pub head_count_kv: u64,
    pub head_dim: u64,
    /// Layers that actually hold a KV cache (hybrid / SWA-free layers).
    pub kv_layers: u64,
    pub expert_count: u64,
    pub expert_used_count: u64,
    pub total_bytes: u64,
    /// Bytes per `blk.N` layer, index = layer.
    pub layer_bytes: Vec<u64>,
    /// Expert-tensor bytes per layer.
    pub layer_expert_bytes: Vec<u64>,
    /// Bytes outside the repeating blocks (embeddings, output head).
    pub other_bytes: u64,
    pub is_diffusion: bool,
    pub is_mmproj: bool,
}

impl ModelSummary {
    pub fn is_moe(&self) -> bool {
        self.expert_count > 1 && self.layer_expert_bytes.iter().any(|b| *b > 0)
    }
}

const DIFFUSION_ARCHES: &[&str] = &["flux", "sd1", "sd2", "sdxl", "sd3", "wan", "qwen_image", "lumina2", "ltxv", "hidream", "chroma", "z_image"];

impl GgufFile {
    pub fn get(&self, k: &str) -> Option<&Value> {
        self.metadata.get(k)
    }
    pub fn arch(&self) -> String {
        self.get("general.architecture").and_then(Value::as_str).unwrap_or("unknown").to_string()
    }
    fn arch_u64(&self, suffix: &str) -> Option<u64> {
        self.get(&format!("{}.{suffix}", self.arch())).and_then(Value::as_u64)
    }

    pub fn summary(&self) -> ModelSummary {
        let arch = self.arch();
        let block_count = self.arch_u64("block_count").unwrap_or_else(|| {
            self.tensors.iter().filter_map(TensorInfo::layer).max().map(|m| m as u64 + 1).unwrap_or(0)
        });
        let head_count = self.arch_u64("attention.head_count").unwrap_or(0);
        let embedding_length = self.arch_u64("embedding_length").unwrap_or(0);

        // head_count_kv may be a per-layer array; 0 entries mark layers without attention.
        let kv_key = format!("{arch}.attention.head_count_kv");
        let (head_count_kv, kv_layers_from_array) = match self.get(&kv_key) {
            Some(Value::Array(a)) => {
                let per: Vec<u64> = a.iter().filter_map(Value::as_u64).collect();
                (per.iter().copied().max().unwrap_or(0), Some(per.iter().filter(|v| **v > 0).count() as u64))
            }
            Some(v) => (v.as_u64().unwrap_or(head_count), None),
            None => (head_count, None),
        };
        let head_dim = self
            .arch_u64("attention.key_length")
            .unwrap_or(embedding_length.checked_div(head_count).unwrap_or(0));
        let kv_layers = kv_layers_from_array
            .or_else(|| self.arch_u64("full_attention_interval").filter(|n| *n > 0).map(|n| block_count.div_ceil(n)))
            .unwrap_or(block_count);

        let mut layer_bytes = vec![0u64; block_count as usize];
        let mut layer_expert_bytes = vec![0u64; block_count as usize];
        let mut other = 0u64;
        for t in &self.tensors {
            match t.layer() {
                Some(l) if l < layer_bytes.len() => {
                    layer_bytes[l] += t.bytes;
                    if t.is_expert() {
                        layer_expert_bytes[l] += t.bytes;
                    }
                }
                _ => other += t.bytes,
            }
        }
        let total_bytes = self.tensors.iter().map(|t| t.bytes).sum();
        let quant = self
            .get("general.file_type")
            .and_then(Value::as_u64)
            .map(|f| file_type_name(f).to_string())
            .unwrap_or_else(|| "unknown".into());
        let is_mmproj = arch == "clip" || self.metadata.keys().any(|k| k.starts_with("clip.vision"));
        let has_tokenizer = self.metadata.contains_key("tokenizer.ggml.model");
        let is_diffusion = !is_mmproj
            && (DIFFUSION_ARCHES.contains(&arch.as_str())
                || (!has_tokenizer
                    && self.tensors.iter().any(|t| {
                        t.name.contains("double_blocks") || t.name.contains("diffusion_model") || t.name.contains("joint_blocks")
                    })));

        ModelSummary {
            name: self.get("general.name").and_then(Value::as_str).map(String::from),
            quant,
            block_count,
            context_length: self.arch_u64("context_length").unwrap_or(0),
            embedding_length,
            head_count,
            head_count_kv,
            head_dim,
            kv_layers,
            expert_count: self.arch_u64("expert_count").unwrap_or(0),
            expert_used_count: self.arch_u64("expert_used_count").unwrap_or(0),
            total_bytes,
            layer_bytes,
            layer_expert_bytes,
            other_bytes: other,
            is_diffusion,
            is_mmproj,
            arch,
        }
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    //! Minimal GGUF writer for tests.
    pub enum V<'a> {
        U32(u32),
        Str(&'a str),
        ArrU32(Vec<u32>),
        ArrStr(Vec<&'a str>),
    }

    fn s(out: &mut Vec<u8>, x: &str) {
        out.extend((x.len() as u64).to_le_bytes());
        out.extend(x.as_bytes());
    }

    pub fn build(kv: &[(&str, V)], tensors: &[(&str, &[u64], u32)]) -> Vec<u8> {
        let mut o = b"GGUF".to_vec();
        o.extend(3u32.to_le_bytes());
        o.extend((tensors.len() as u64).to_le_bytes());
        o.extend((kv.len() as u64).to_le_bytes());
        for (k, v) in kv {
            s(&mut o, k);
            match v {
                V::U32(x) => {
                    o.extend(4u32.to_le_bytes());
                    o.extend(x.to_le_bytes());
                }
                V::Str(x) => {
                    o.extend(8u32.to_le_bytes());
                    s(&mut o, x);
                }
                V::ArrU32(a) => {
                    o.extend(9u32.to_le_bytes());
                    o.extend(4u32.to_le_bytes());
                    o.extend((a.len() as u64).to_le_bytes());
                    for x in a {
                        o.extend(x.to_le_bytes());
                    }
                }
                V::ArrStr(a) => {
                    o.extend(9u32.to_le_bytes());
                    o.extend(8u32.to_le_bytes());
                    o.extend((a.len() as u64).to_le_bytes());
                    for x in a {
                        s(&mut o, x);
                    }
                }
            }
        }
        let mut off = 0u64;
        for (name, dims, ty) in tensors {
            s(&mut o, name);
            o.extend((dims.len() as u32).to_le_bytes());
            for d in *dims {
                o.extend(d.to_le_bytes());
            }
            o.extend(ty.to_le_bytes());
            o.extend(off.to_le_bytes());
            off += super::tensor_bytes(*ty, dims);
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::{build, V};
    use super::*;

    fn moe_file() -> Vec<u8> {
        let vocab: Vec<&str> = (0..5000).map(|_| "tok").collect();
        build(
            &[
                ("general.architecture", V::Str("qwen3moe")),
                ("general.name", V::Str("Test MoE")),
                ("general.file_type", V::U32(15)),
                ("qwen3moe.block_count", V::U32(2)),
                ("qwen3moe.context_length", V::U32(32768)),
                ("qwen3moe.embedding_length", V::U32(256)),
                ("qwen3moe.attention.head_count", V::U32(8)),
                ("qwen3moe.attention.head_count_kv", V::ArrU32(vec![2, 0])),
                ("qwen3moe.attention.key_length", V::U32(64)),
                ("qwen3moe.expert_count", V::U32(4)),
                ("qwen3moe.expert_used_count", V::U32(2)),
                ("tokenizer.ggml.model", V::Str("gpt2")),
                ("tokenizer.ggml.tokens", V::ArrStr(vocab)),
            ],
            &[
                ("token_embd.weight", &[256, 1000], 12),
                ("blk.0.attn_q.weight", &[256, 256], 8),
                ("blk.0.ffn_up_exps.weight", &[256, 512, 4], 12),
                ("blk.1.attn_q.weight", &[256, 256], 8),
                ("blk.1.ffn_up_exps.weight", &[256, 512, 4], 12),
                ("output.weight", &[256, 1000], 14),
            ],
        )
    }

    #[test]
    fn block_sizes() {
        assert_eq!(tensor_bytes(12, &[256, 2]), 2 * 144);
        assert_eq!(tensor_bytes(8, &[32]), 34);
        assert_eq!(tensor_bytes(1, &[10, 10]), 200);
    }

    #[test]
    fn parses_moe_header() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.gguf");
        std::fs::write(&p, moe_file()).unwrap();
        let g = open(&p).unwrap();
        assert_eq!(g.version, 3);
        assert!(matches!(g.get("tokenizer.ggml.tokens"), Some(Value::SkippedArray(5000))));
        let s = g.summary();
        assert_eq!(s.arch, "qwen3moe");
        assert_eq!(s.quant, "Q4_K_M");
        assert_eq!(s.block_count, 2);
        assert_eq!(s.kv_layers, 1, "layer 1 has no KV heads");
        assert_eq!(s.head_count_kv, 2);
        assert_eq!(s.head_dim, 64);
        assert!(s.is_moe());
        assert_eq!(s.layer_expert_bytes[0], tensor_bytes(12, &[256, 512, 4]));
        assert_eq!(s.other_bytes, tensor_bytes(12, &[256, 1000]) + tensor_bytes(14, &[256, 1000]));
        assert!(!s.is_diffusion);
    }

    #[test]
    fn aggregates_split_files() {
        let dir = tempfile::tempdir().unwrap();
        let a = build(&[("general.architecture", V::Str("llama")), ("llama.block_count", V::U32(2))], &[("blk.0.attn_q.weight", &[64, 64], 1)]);
        let b = build(&[], &[("blk.1.attn_q.weight", &[64, 64], 1)]);
        std::fs::write(dir.path().join("x-00001-of-00002.gguf"), a).unwrap();
        std::fs::write(dir.path().join("x-00002-of-00002.gguf"), b).unwrap();
        let g = open(&dir.path().join("x-00002-of-00002.gguf")).unwrap();
        assert_eq!(g.files.len(), 2);
        let s = g.summary();
        assert_eq!(s.layer_bytes, vec![8192, 8192]);
        assert_eq!(s.kv_layers, 2);
    }

    #[test]
    fn full_attention_interval_sets_kv_layers() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("h.gguf");
        std::fs::write(
            &p,
            build(&[("general.architecture", V::Str("qwen3next")), ("qwen3next.block_count", V::U32(48)), ("qwen3next.full_attention_interval", V::U32(4))], &[]),
        )
        .unwrap();
        assert_eq!(open(&p).unwrap().summary().kv_layers, 12);
    }

    #[test]
    fn detects_diffusion_and_rejects_garbage() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.gguf");
        std::fs::write(&p, build(&[], &[("model.diffusion_model.double_blocks.0.img_attn.qkv.weight", &[64, 64], 1)])).unwrap();
        assert!(open(&p).unwrap().summary().is_diffusion);
        let bad = dir.path().join("bad.gguf");
        std::fs::write(&bad, b"NOPE1234").unwrap();
        assert!(open(&bad).is_err());
    }

    #[test]
    fn split_name() {
        assert_eq!(split_parts("Qwen-UD-Q4_K_XL-00002-of-00003.gguf"), Some(("Qwen-UD-Q4_K_XL".into(), 2, 3)));
        assert_eq!(split_parts("model.gguf"), None);
    }
}
