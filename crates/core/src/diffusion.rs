//! Image and video generation through stable-diffusion.cpp's `sd-server`.
//!
//! Each model is a manifest of components (diffusion weights, text encoder,
//! VAE...) that live in Hugging Face repos. Files are resolved per quant,
//! downloaded into `models_dir/diffusion/<model id>/`, and passed to
//! sd-server as flags.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaKind {
    Image,
    Video,
}

/// A file inside a HF repo; `by_quant` lists alternatives keyed by quant.
#[derive(Debug, Clone, Serialize)]
pub struct Component {
    /// sd-server flag that receives the local path.
    pub flag: &'static str,
    pub repo: &'static str,
    /// Fixed file, used when `by_quant` is empty.
    pub file: &'static str,
    pub by_quant: &'static [(&'static str, &'static str)],
}

impl Component {
    pub fn file_for(&self, quant: &str) -> &'static str {
        self.by_quant.iter().find(|(q, _)| q.eq_ignore_ascii_case(quant)).map(|(_, f)| *f).unwrap_or(self.file)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MediaModel {
    pub id: &'static str,
    pub name: &'static str,
    pub kind: MediaKind,
    /// Can edit / use reference images.
    pub edit: bool,
    /// Video model that needs a start image.
    pub image_to_video: bool,
    /// Supports 720P video.
    pub hd: bool,
    pub quants: &'static [&'static str],
    pub default_quant: &'static str,
    pub components: &'static [Component],
    pub steps: u32,
    pub cfg_scale: f32,
    pub sampler: &'static str,
    pub flow_shift: Option<f32>,
    /// Extra sd-server flags this model needs.
    pub extra_args: &'static [&'static str],
}

const fn c(flag: &'static str, repo: &'static str, file: &'static str) -> Component {
    Component { flag, repo, file, by_quant: &[] }
}

/// Built-in models. Repos and file names are public Hugging Face facts.
pub const MODELS: &[MediaModel] = &[
    MediaModel {
        id: "z-image-turbo",
        name: "Z-Image Turbo",
        kind: MediaKind::Image,
        edit: false,
        image_to_video: false,
        hd: false,
        quants: &["Q3_K", "Q4_K", "Q6_K", "Q8_0"],
        default_quant: "Q4_K",
        components: &[
            Component {
                flag: "--diffusion-model",
                repo: "leejet/Z-Image-Turbo-GGUF",
                file: "z_image_turbo-Q4_K.gguf",
                by_quant: &[("Q3_K", "z_image_turbo-Q3_K.gguf"), ("Q4_K", "z_image_turbo-Q4_K.gguf"), ("Q6_K", "z_image_turbo-Q6_K.gguf"), ("Q8_0", "z_image_turbo-Q8_0.gguf")],
            },
            c("--llm", "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
            c("--vae", "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
        ],
        steps: 8,
        cfg_scale: 1.0,
        sampler: "euler",
        flow_shift: None,
        extra_args: &[],
    },
    MediaModel {
        id: "flux2-klein-4b",
        name: "FLUX.2 klein 4B",
        kind: MediaKind::Image,
        edit: true,
        image_to_video: false,
        hd: false,
        quants: &["Q4_0", "Q8_0"],
        default_quant: "Q4_0",
        components: &[
            Component {
                flag: "--diffusion-model",
                repo: "leejet/FLUX.2-klein-4B-GGUF",
                file: "flux-2-klein-4b-Q4_0.gguf",
                by_quant: &[("Q4_0", "flux-2-klein-4b-Q4_0.gguf"), ("Q8_0", "flux-2-klein-4b-Q8_0.gguf")],
            },
            c("--llm", "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
            c("--vae", "Comfy-Org/flux2-klein-4B", "split_files/vae/flux2-vae.safetensors"),
        ],
        steps: 4,
        cfg_scale: 1.0,
        sampler: "euler",
        flow_shift: None,
        extra_args: &[],
    },
    MediaModel {
        id: "flux1-schnell",
        name: "FLUX.1 schnell",
        kind: MediaKind::Image,
        edit: false,
        image_to_video: false,
        hd: false,
        quants: &["Q3_K", "Q4_K", "Q8_0"],
        default_quant: "Q4_K",
        components: &[
            Component {
                flag: "--diffusion-model",
                repo: "leejet/FLUX.1-schnell-gguf",
                file: "flux1-schnell-q4_k.gguf",
                by_quant: &[("Q3_K", "flux1-schnell-q3_k.gguf"), ("Q4_K", "flux1-schnell-q4_k.gguf"), ("Q8_0", "flux1-schnell-q8_0.gguf")],
            },
            c("--clip_l", "comfyanonymous/flux_text_encoders", "clip_l.safetensors"),
            c("--t5xxl", "comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors"),
            c("--vae", "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
        ],
        steps: 4,
        cfg_scale: 1.0,
        sampler: "euler",
        flow_shift: None,
        extra_args: &[],
    },
    MediaModel {
        id: "qwen-image",
        name: "Qwen-Image",
        kind: MediaKind::Image,
        edit: false,
        image_to_video: false,
        hd: false,
        quants: &["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
        default_quant: "Q4_K_M",
        components: &[
            Component {
                flag: "--diffusion-model",
                repo: "QuantStack/Qwen-Image-GGUF",
                file: "Qwen_Image-Q4_K_M.gguf",
                by_quant: &[
                    ("Q2_K", "Qwen_Image-Q2_K.gguf"),
                    ("Q3_K_M", "Qwen_Image-Q3_K_M.gguf"),
                    ("Q4_K_M", "Qwen_Image-Q4_K_M.gguf"),
                    ("Q5_K_M", "Qwen_Image-Q5_K_M.gguf"),
                    ("Q6_K", "Qwen_Image-Q6_K.gguf"),
                    ("Q8_0", "Qwen_Image-Q8_0.gguf"),
                ],
            },
            c("--llm", "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf"),
            c("--vae", "QuantStack/Qwen-Image-GGUF", "VAE/Qwen_Image-VAE.safetensors"),
        ],
        steps: 20,
        cfg_scale: 2.5,
        sampler: "euler",
        flow_shift: Some(3.0),
        extra_args: &[],
    },
    MediaModel {
        id: "qwen-image-edit",
        name: "Qwen-Image-Edit 2511",
        kind: MediaKind::Image,
        edit: true,
        image_to_video: false,
        hd: false,
        quants: &["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
        default_quant: "Q4_K_M",
        components: &[
            Component {
                flag: "--diffusion-model",
                repo: "unsloth/Qwen-Image-Edit-2511-GGUF",
                file: "qwen-image-edit-2511-Q4_K_M.gguf",
                by_quant: &[
                    ("Q2_K", "qwen-image-edit-2511-Q2_K.gguf"),
                    ("Q3_K_M", "qwen-image-edit-2511-Q3_K_M.gguf"),
                    ("Q4_K_M", "qwen-image-edit-2511-Q4_K_M.gguf"),
                    ("Q5_K_M", "qwen-image-edit-2511-Q5_K_M.gguf"),
                    ("Q6_K", "qwen-image-edit-2511-Q6_K.gguf"),
                    ("Q8_0", "qwen-image-edit-2511-Q8_0.gguf"),
                ],
            },
            c("--llm", "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf"),
            c("--llm_vision", "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf"),
            c("--vae", "QuantStack/Qwen-Image-GGUF", "VAE/Qwen_Image-VAE.safetensors"),
        ],
        steps: 20,
        cfg_scale: 2.5,
        sampler: "euler",
        flow_shift: Some(3.0),
        extra_args: &[],
    },
    MediaModel {
        id: "wan2.2-ti2v-5b",
        name: "Wan2.2 TI2V 5B",
        kind: MediaKind::Video,
        edit: false,
        image_to_video: false,
        hd: true,
        quants: &["Q4_K_M", "Q6_K", "Q8_0"],
        default_quant: "Q4_K_M",
        components: &[
            Component {
                flag: "--diffusion-model",
                repo: "QuantStack/Wan2.2-TI2V-5B-GGUF",
                file: "Wan2.2-TI2V-5B-Q4_K_M.gguf",
                by_quant: &[("Q4_K_M", "Wan2.2-TI2V-5B-Q4_K_M.gguf"), ("Q6_K", "Wan2.2-TI2V-5B-Q6_K.gguf"), ("Q8_0", "Wan2.2-TI2V-5B-Q8_0.gguf")],
            },
            c("--t5xxl", "city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q8_0.gguf"),
            c("--vae", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan2.2_vae.safetensors"),
        ],
        steps: 20,
        cfg_scale: 5.0,
        sampler: "euler",
        flow_shift: Some(5.0),
        extra_args: &[],
    },
    MediaModel {
        id: "wan2.2-i2v-a14b",
        name: "Wan2.2 I2V A14B",
        kind: MediaKind::Video,
        edit: false,
        image_to_video: true,
        hd: false,
        quants: &["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"],
        default_quant: "Q4_K_M",
        components: &[
            Component {
                flag: "--diffusion-model",
                repo: "QuantStack/Wan2.2-I2V-A14B-GGUF",
                file: "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf",
                by_quant: &[
                    ("Q3_K_M", "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_M.gguf"),
                    ("Q4_K_M", "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"),
                    ("Q5_K_M", "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf"),
                    ("Q6_K", "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q6_K.gguf"),
                ],
            },
            Component {
                flag: "--high-noise-diffusion-model",
                repo: "QuantStack/Wan2.2-I2V-A14B-GGUF",
                file: "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
                by_quant: &[
                    ("Q3_K_M", "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_M.gguf"),
                    ("Q4_K_M", "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"),
                    ("Q5_K_M", "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf"),
                    ("Q6_K", "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q6_K.gguf"),
                ],
            },
            c("--t5xxl", "city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q8_0.gguf"),
            c("--vae", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors"),
        ],
        steps: 10,
        cfg_scale: 3.5,
        sampler: "euler",
        flow_shift: Some(3.0),
        extra_args: &["--high-noise-cfg-scale", "3.5", "--high-noise-sampling-method", "euler", "--high-noise-steps", "10"],
    },
];

pub fn find_model(id: &str) -> Option<&'static MediaModel> {
    MODELS.iter().find(|m| m.id == id)
}

/// Where a model's files are stored.
pub fn model_dir(models_dir: &Path, m: &MediaModel) -> PathBuf {
    models_dir.join("diffusion").join(m.id)
}

/// Local path of a component (keeps the repo sub-folder to avoid name clashes).
pub fn component_path(models_dir: &Path, m: &MediaModel, comp: &Component, quant: &str) -> PathBuf {
    let mut p = model_dir(models_dir, m).join(comp.repo.replace('/', "__"));
    for seg in comp.file_for(quant).split('/') {
        p.push(seg);
    }
    p
}

/// (url, local path) for every component.
pub fn download_plan(hf: &crate::hf::HfClient, models_dir: &Path, m: &MediaModel, quant: &str) -> Vec<(String, PathBuf)> {
    m.components.iter().map(|c| (hf.resolve_url(c.repo, c.file_for(quant)), component_path(models_dir, m, c, quant))).collect()
}

pub fn is_installed(models_dir: &Path, m: &MediaModel, quant: &str) -> bool {
    m.components.iter().all(|c| component_path(models_dir, m, c, quant).exists())
}

/// sd-server command line for a model.
pub fn server_args(models_dir: &Path, m: &MediaModel, quant: &str, host: &str, port: u16) -> Vec<String> {
    let mut a: Vec<String> = vec!["--listen-ip".into(), host.into(), "--listen-port".into(), port.to_string(), "--model-dir".into(), model_dir(models_dir, m).display().to_string()];
    for comp in m.components {
        a.push(comp.flag.into());
        a.push(component_path(models_dir, m, comp, quant).display().to_string());
    }
    a.extend(["--steps".into(), m.steps.to_string(), "--cfg-scale".into(), m.cfg_scale.to_string(), "--sampling-method".into(), m.sampler.into()]);
    if let Some(fs) = m.flow_shift {
        a.extend(["--flow-shift".into(), fs.to_string()]);
    }
    a.extend(m.extra_args.iter().map(|s| s.to_string()));
    a
}

pub const ASPECTS: &[&str] = &["1:1", "3:4", "4:3", "9:16", "16:9"];

/// Image size for an aspect ratio with about `base`x`base` pixels, multiples of 16.
pub fn image_size(aspect: &str, base: u32) -> (u32, u32) {
    let (w, h) = aspect.split_once(':').and_then(|(a, b)| Some((a.parse::<f64>().ok()?, b.parse::<f64>().ok()?))).unwrap_or((1.0, 1.0));
    let area = (base * base) as f64;
    let width = (area * w / h).sqrt();
    let height = area / width;
    let r16 = |x: f64| ((x / 16.0).round() as u32).max(1) * 16;
    (r16(width), r16(height))
}

/// Video frame size: 480P = 832x480, 720P = 1280x704 (Wan2.2 5B only).
pub fn video_size(portrait: bool, hd: bool) -> (u32, u32) {
    let (w, h) = if hd { (1280, 704) } else { (832, 480) };
    if portrait {
        (h, w)
    } else {
        (w, h)
    }
}

/// Frame count for a clip: 4n+1 frames, as the Wan/LTX VAEs require.
pub fn frames_for(seconds: u32, fps: u32) -> u32 {
    if fps >= 24 {
        match seconds {
            0 | 1 => 25,
            2 => 49,
            3 => 73,
            _ => 121,
        }
    } else {
        match seconds {
            0 | 1 => 17,
            2 => 33,
            3 => 49,
            _ => 81,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImageRequest {
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: Option<u32>,
    pub cfg_scale: Option<f32>,
    pub seed: Option<i64>,
    /// Base64 PNG/JPEG for img2img.
    pub init_image: Option<String>,
    pub denoising_strength: Option<f32>,
    /// Reference images for edit models.
    #[serde(default)]
    pub extra_images: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VideoRequest {
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub steps: Option<u32>,
    pub seed: Option<i64>,
    pub init_image: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum JobState {
    Running(Option<f64>),
    Completed(Vec<String>),
    Failed(String),
}

/// Pull base64 images out of either `{"images": [...]}` or `{"data": [{"b64_json": ...}]}`.
pub fn extract_images(v: &Value) -> Vec<String> {
    if let Some(a) = v.get("images").and_then(Value::as_array) {
        return a.iter().filter_map(|x| x.as_str().map(String::from)).collect();
    }
    if let Some(a) = v.get("data").and_then(Value::as_array) {
        return a.iter().filter_map(|x| x.get("b64_json").and_then(Value::as_str).map(String::from)).collect();
    }
    if let Some(s) = v.get("b64_json").and_then(Value::as_str) {
        return vec![s.to_string()];
    }
    vec![]
}

pub fn parse_job(v: &Value) -> JobState {
    let status = v.get("status").and_then(Value::as_str).unwrap_or("running");
    match status {
        "completed" | "succeeded" | "done" => {
            let r = v.get("result").unwrap_or(v);
            JobState::Completed(extract_images(r))
        }
        "failed" | "error" | "cancelled" => JobState::Failed(v.get("error").map(|e| e.as_str().map(String::from).unwrap_or_else(|| e.to_string())).unwrap_or_else(|| status.into())),
        _ => JobState::Running(v.get("progress").and_then(Value::as_f64)),
    }
}

/// HTTP client for a running sd-server.
pub struct SdClient {
    http: reqwest::Client,
    base: String,
}

impl SdClient {
    pub fn new(base: &str) -> SdClient {
        SdClient { http: crate::http_client(), base: base.trim_end_matches('/').to_string() }
    }

    async fn post(&self, path: &str, body: Value) -> Result<Value> {
        let r = self.http.post(format!("{}{}", self.base, path)).json(&body).send().await?;
        let status = r.status();
        let v: Value = r.json().await.unwrap_or(Value::Null);
        if !status.is_success() {
            bail!("sd-server {path}: HTTP {status}: {v}");
        }
        Ok(v)
    }

    pub async fn capabilities(&self) -> Result<Value> {
        Ok(self.http.get(format!("{}/sdcpp/v1/capabilities", self.base)).send().await?.json().await?)
    }

    /// Returns base64-encoded PNGs.
    pub async fn generate_image(&self, m: &MediaModel, req: &ImageRequest) -> Result<Vec<String>> {
        let mut body = json!({
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps.unwrap_or(m.steps),
            "cfg_scale": req.cfg_scale.unwrap_or(m.cfg_scale),
            "sampler_name": m.sampler,
            "seed": req.seed.unwrap_or(-1),
            "batch_size": 1,
        });
        let path = if let Some(init) = &req.init_image {
            body["init_images"] = json!([init]);
            body["denoising_strength"] = json!(req.denoising_strength.unwrap_or(0.75));
            "/sdapi/v1/img2img"
        } else {
            "/sdapi/v1/txt2img"
        };
        if !req.extra_images.is_empty() {
            body["extra_images"] = json!(req.extra_images);
        }
        let v = self.post(path, body).await?;
        let imgs = extract_images(&v);
        if imgs.is_empty() {
            bail!("sd-server returned no image");
        }
        Ok(imgs)
    }

    pub async fn start_video(&self, m: &MediaModel, req: &VideoRequest) -> Result<String> {
        let mut body = json!({
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "video_frames": req.frames,
            "fps": req.fps,
            "steps": req.steps.unwrap_or(m.steps),
            "cfg_scale": m.cfg_scale,
            "seed": req.seed.unwrap_or(-1),
            "output_format": "webm",
        });
        if let Some(i) = &req.init_image {
            body["init_image"] = json!(i);
        }
        let v = self.post("/sdcpp/v1/vid_gen", body).await?;
        v.get("id").or_else(|| v.get("job_id")).and_then(|x| x.as_str().map(String::from).or_else(|| Some(x.to_string()))).context("sd-server did not return a job id")
    }

    pub async fn job(&self, id: &str) -> Result<JobState> {
        let v: Value = self.http.get(format!("{}/sdcpp/v1/jobs/{id}", self.base)).send().await?.json().await?;
        Ok(parse_job(&v))
    }

    pub async fn cancel(&self, id: &str) -> Result<()> {
        self.post(&format!("/sdcpp/v1/jobs/{id}/cancel"), json!({})).await.map(|_| ())
    }

    /// Poll a video job until it finishes; `on_progress` gets 0..1 when known.
    pub async fn wait_video(&self, id: &str, cancel: &tokio_util::sync::CancellationToken, on_progress: impl Fn(Option<f64>)) -> Result<Vec<String>> {
        loop {
            if cancel.is_cancelled() {
                let _ = self.cancel(id).await;
                bail!("cancelled");
            }
            match self.job(id).await? {
                JobState::Completed(v) => return Ok(v),
                JobState::Failed(e) => bail!("video generation failed: {e}"),
                JobState::Running(p) => on_progress(p),
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }
}

/// Decode base64 output and write it to `dir` with a timestamped name.
pub fn save_output(dir: &Path, b64: &str, ext: &str) -> Result<PathBuf> {
    use base64::Engine;
    std::fs::create_dir_all(dir)?;
    let s = b64.trim();
    // Accept data URLs ("data:image/png;base64,....") as well as bare base64.
    let s = if s.starts_with("data:") { s.split_once(',').map(|x| x.1).unwrap_or_default() } else { s };
    let data = base64::engine::general_purpose::STANDARD.decode(s)?;
    let name = format!("{}-{}.{ext}", chrono::Local::now().format("%Y%m%d-%H%M%S"), &uuid::Uuid::new_v4().simple().to_string()[..6]);
    let p = dir.join(name);
    std::fs::write(&p, data)?;
    Ok(p)
}

/// Rough "draw me a picture" intent check for the chat page.
pub fn wants_image(prompt: &str) -> bool {
    let p = prompt.trim().to_lowercase();
    const ZH: &[&str] = &["畫一張", "画一张", "畫一個", "画一个", "生成一張圖", "生成一张图", "幫我畫", "帮我画", "畫出", "画出"];
    const EN: &[&str] = &["draw ", "generate an image", "generate a picture", "make an image", "create an image", "paint "];
    ZH.iter().any(|k| p.contains(k)) || EN.iter().any(|k| p.starts_with(k) || p.contains(&format!(" {k}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifests_are_consistent() {
        for m in MODELS {
            assert!(m.quants.contains(&m.default_quant), "{}", m.id);
            assert!(m.components.iter().any(|c| c.flag == "--diffusion-model"), "{}", m.id);
            for c in m.components {
                assert!(c.repo.contains('/'));
                for q in m.quants {
                    if !c.by_quant.is_empty() {
                        assert!(c.by_quant.iter().any(|(x, _)| x == q), "{} missing {q}", m.id);
                    }
                }
            }
        }
    }

    #[test]
    fn args_use_quant_files() {
        let m = find_model("wan2.2-i2v-a14b").unwrap();
        let a = server_args(Path::new("/models"), m, "Q3_K_M", "127.0.0.1", 7860);
        let s = a.join(" ");
        assert!(s.contains("LowNoise-Q3_K_M.gguf"));
        assert!(s.contains("--high-noise-diffusion-model"));
        assert!(s.contains("--listen-port 7860"));
        assert!(s.contains("--flow-shift 3"));
    }

    #[test]
    fn sizes_and_frames() {
        assert_eq!(image_size("1:1", 1024), (1024, 1024));
        let (w, h) = image_size("16:9", 1024);
        assert!(w > h && w % 16 == 0 && h % 16 == 0);
        assert_eq!(video_size(true, false), (480, 832));
        assert_eq!(frames_for(5, 24), 121);
        assert_eq!(frames_for(2, 16), 33);
    }

    #[test]
    fn job_parsing() {
        assert_eq!(parse_job(&json!({"status":"running","progress":0.4})), JobState::Running(Some(0.4)));
        assert_eq!(parse_job(&json!({"status":"completed","result":{"data":[{"b64_json":"AAA"}]}})), JobState::Completed(vec!["AAA".into()]));
        assert_eq!(parse_job(&json!({"status":"failed","error":"oom"})), JobState::Failed("oom".into()));
        assert_eq!(extract_images(&json!({"images":["x","y"]})).len(), 2);
    }

    #[test]
    fn saves_and_detects_intent() {
        let d = tempfile::tempdir().unwrap();
        let p = save_output(d.path(), "aGVsbG8=", "png").unwrap();
        assert_eq!(std::fs::read(p).unwrap(), b"hello");
        let p2 = save_output(d.path(), "data:image/png;base64,aGk=", "png").unwrap();
        assert_eq!(std::fs::read(p2).unwrap(), b"hi");
        assert!(wants_image("幫我畫一隻貓"));
        assert!(wants_image("Draw a cat in space"));
        assert!(!wants_image("寫一首詩"));
    }
}
