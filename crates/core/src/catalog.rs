//! Curated model catalog (recommended LLMs).
//!
//! A copy ships inside the binary. A newer `catalog.json` in the data dir (for
//! example one the user downloaded from the project repo) takes precedence when
//! its `version` is higher. Nothing is fetched automatically.

use crate::tiering::ModelShape;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

const EMBEDDED: &str = include_str!("../data/catalog.json");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    pub version: u64,
    pub models: Vec<CatalogModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub name: String,
    pub org: String,
    pub repo: String,
    pub release: String,
    /// Language code -> description.
    pub desc: BTreeMap<String, String>,
    /// chat / code / vision / reasoning
    pub tags: Vec<String>,
    pub params_b: f64,
    #[serde(default)]
    pub active_b: Option<f64>,
    pub layers: u64,
    /// Layers with a KV cache; defaults to `layers`.
    #[serde(default)]
    pub kv_layers: Option<u64>,
    pub kv_heads: u64,
    pub head_dim: u64,
    #[serde(default)]
    pub experts: Option<u64>,
    #[serde(default)]
    pub experts_used: Option<u64>,
    pub ctx_max: u64,
    #[serde(default)]
    pub vision: bool,
    pub quants: Vec<String>,
    /// Known download sizes in GB (overrides the bits-per-weight estimate).
    #[serde(default)]
    pub size_gb: BTreeMap<String, f64>,
}

/// Average bits per weight for common quants (used when the size is unknown).
pub fn bits_per_weight(quant: &str) -> f64 {
    let q = quant.to_uppercase();
    let q = q.trim_start_matches("UD-");
    let base = q.trim_end_matches("_XL");
    match base {
        "Q2_K" | "Q2_K_S" | "IQ2_M" => 3.0,
        "IQ2_XXS" | "IQ2_XS" | "IQ2_S" => 2.5,
        "IQ1_S" | "IQ1_M" => 1.9,
        "Q3_K_S" | "Q3_K_M" | "Q3_K_L" | "Q3_K" | "IQ3_XXS" | "IQ3_XS" | "IQ3_S" | "IQ3_M" => 3.9,
        "Q4_0" | "Q4_1" | "IQ4_NL" | "IQ4_XS" | "MXFP4" | "MXFP4_MOE" => 4.5,
        "Q4_K_S" | "Q4_K_M" | "Q4_K" => 4.85,
        "Q5_0" | "Q5_1" | "Q5_K_S" | "Q5_K_M" | "Q5_K" => 5.7,
        "Q6_K" => 6.6,
        "Q8_0" | "Q8_K" => 8.5,
        "F16" | "BF16" => 16.0,
        "F32" => 32.0,
        _ => 4.85,
    }
}

impl Catalog {
    pub fn embedded() -> Catalog {
        serde_json::from_str(EMBEDDED).expect("embedded catalog.json is valid")
    }

    /// Embedded catalog, or a newer one from `data_dir/catalog.json`.
    pub fn load(data_dir: &Path) -> Catalog {
        let base = Catalog::embedded();
        match std::fs::read(data_dir.join("catalog.json")).ok().and_then(|b| serde_json::from_slice::<Catalog>(&b).ok()) {
            Some(c) if c.version > base.version => c,
            _ => base,
        }
    }

    pub fn find(&self, id: &str) -> Option<&CatalogModel> {
        self.models.iter().find(|m| m.id == id)
    }
}

impl CatalogModel {
    pub fn is_moe(&self) -> bool {
        self.experts.unwrap_or(0) > 1
    }

    pub fn size_bytes(&self, quant: &str) -> u64 {
        let gb = self.size_gb.get(quant).copied().unwrap_or_else(|| self.params_b * bits_per_weight(quant) / 8.0);
        (gb * 1e9) as u64
    }

    pub fn description(&self, lang: &str) -> &str {
        self.desc.get(lang).or_else(|| self.desc.get("en")).map(String::as_str).unwrap_or("")
    }

    /// Approximate layer layout for planning before the file is downloaded.
    pub fn shape(&self, quant: &str) -> ModelShape {
        let total = self.size_bytes(quant);
        let layers = self.layers.max(1);
        // Embeddings + output head: roughly 4 % of a model this size.
        let other = total / 25;
        let per_layer = (total - other) / layers;
        let expert_frac = match (self.is_moe(), self.active_b) {
            (true, Some(a)) if a < self.params_b => ((self.params_b - a) / self.params_b).clamp(0.0, 0.97),
            (true, _) => 0.9,
            _ => 0.0,
        };
        let per_expert = (per_layer as f64 * expert_frac) as u64;
        ModelShape {
            total_bytes: total,
            layer_bytes: vec![per_layer; layers as usize],
            layer_expert_bytes: vec![per_expert; layers as usize],
            other_bytes: other,
            kv_layers: self.kv_layers.unwrap_or(self.layers),
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            ctx_max: self.ctx_max,
            expert_count: self.experts.unwrap_or(0),
            expert_used: self.experts_used.unwrap_or(0),
            mmproj_bytes: if self.vision { 900 * crate::config::MIB } else { 0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_catalog_is_valid() {
        let c = Catalog::embedded();
        assert!(c.models.len() >= 10);
        for m in &c.models {
            assert!(m.repo.contains('/'), "{}", m.id);
            assert!(!m.description("zh-TW").is_empty(), "{}", m.id);
            assert!(!m.description("en").is_empty(), "{}", m.id);
            assert!(!m.quants.is_empty());
        }
    }

    #[test]
    fn sizes() {
        let c = Catalog::embedded();
        let g = c.find("gpt-oss-20b").unwrap();
        assert_eq!(g.size_bytes("Q4_K_M"), 11_600_000_000);
        let q = c.find("qwen3.6-27b").unwrap();
        assert_eq!(q.size_bytes("UD-Q4_K_XL"), 17_600_000_000);
        // No listed size: 27B x 4.85 bpw / 8 ~ 16.4 GB
        let b = q.size_bytes("Q4_K_M") as f64 / 1e9;
        assert!((16.0..17.0).contains(&b), "{b}");
    }

    #[test]
    fn moe_shape_has_experts() {
        let c = Catalog::embedded();
        let s = c.find("qwen3.6-35b-a3b").unwrap().shape("Q4_K_M");
        assert!(s.is_moe());
        assert_eq!(s.kv_layers, 10);
        let dense = c.find("qwen3.6-27b").unwrap().shape("Q4_K_M");
        assert!(!dense.is_moe());
    }
}
