# 進度與下一步

最後更新：2026-11-02

## 已完成（`crates/core`）

| 模組 | 內容 | 測試 |
|---|---|---|
| `config` | 設定檔、資料夾位置（`OPENMODEL_HOME`、可攜模式 `portable.txt`）、LAN、GPU 手動覆寫、每個模型的自訂參數、GitHub 鏡像前綴（預設空白） | 2 |
| `hardware` | NVIDIA（nvidia-smi）、AMD/Intel（Linux sysfs，APU 用 GTT 判斷共用記憶體；Windows 讀顯示卡登錄檔 64-bit VRAM）、Apple Silicon；SSD/HDD、RAM 類型（DDR4/5）、即時 VRAM | 4 |
| `gguf` | GGUF v1–v3 header、metadata、tensor 表；分卷合併；每層 / 專家權重大小；`kv_layers`（逐層 head_count_kv 或 `full_attention_interval`）；擴散 / mmproj 偵測 | 6 |
| `catalog` | 內建 16 個 LLM（`data/catalog.json`，描述重新撰寫為 zh-TW / en）；量化大小估算；下載前的層配置估計 | 3 |
| `tiering` | 全放 GPU → MoE `--n-cpu-moe` → 稠密 `-ngl` → mmap 溢出到硬碟；KV 大小；tok/s 估算；自動 context（不降速的最大值）；相容模式 1–5；自訂參數合併（同名覆寫、指定 `-ngl/-ot/--n-cpu-moe` 時停用自動配置） | 8 |
| `hf` | 搜尋、檔案列表（分頁）、量化分組（含分卷）、mmproj 偵測、鏡像與 token | 3 |
| `download` | `.part` 續傳（Range）、取消、重試（404/403 不重試）、多檔進度與速度 | 3 |
| `engine` | GitHub release 資產挑選（llama.cpp 與 sd.cpp 命名、CUDA 版本依驅動、Windows cudart）、下載解壓（防 zip-slip）、匯入 zip / 既有資料夾 | 5 |
| `server` | 啟動 / 停止、log 擷取與載入階段解析、`/health`、port 自動往後找、崩潰原因分類、相容模式自動重試 | 5 |
| `diffusion` | 7 個圖片 / 影片模型清單（Z-Image、FLUX.2 klein、FLUX.1 schnell、Qwen-Image、Qwen-Image-Edit、Wan2.2 TI2V 5B、Wan2.2 I2V A14B）、sd-server 參數、txt2img / img2img / 影片 job API、尺寸與幀數 | 5 |
| `tools` | 審批閘門（後端強制、本次 / 本對話總是允許 / 拒絕）、計算機（evalexpr，無程式執行）、Bing RSS 搜尋、網頁擷取（拒絕本機與私有網段，逐跳檢查轉址）、檔案與指令工具 | 7 |
| `agent` | 工作資料夾沙箱（解析 `..` 與 symlink 後判斷）、11 個工具的 schema、資料夾外讀寫與 `run_command` / `delete_path` 一律詢問 | 5 |
| `library` | 掃描本機模型（分卷只列一次、自動配對 mmproj）、登記外部模型（不複製、移除時不刪檔） | 1 |

`crates/cli`：`openmodel hw | gguf | catalog | plan | hf search/files/download | models | engine list/install/import | serve`。

## 尚未驗證（需要使用者在測試機上跑）

- `engine install` 與 `serve` 尚未實際下載引擎與載入模型；只在單元測試中用假程序驗證崩潰流程。
- 相容模式的環境變數（`GGML_CUDA_DISABLE_GRAPHS`、`GGML_CUDA_FORCE_CUBLAS`）和 llama.cpp 目前版本的對應。
- sd-server 的元件參數名稱（`--llm`、`--llm_vision`、`--clip_l`、`--t5xxl`、`--vae`）以 stable-diffusion.cpp 現行版本為準，需實測。
- tok/s 估算的效率係數（GPU 0.65、CPU 0.55）需用三台測試機的實測值校正。

建議的測試指令（每台機器）：

```bash
openmodel hw
openmodel engine install
openmodel hf download unsloth/gpt-oss-20b-GGUF Q4_K_M
openmodel serve <models_dir>/unsloth__gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf -v
```

## 下一步

1. **Tauri 2 桌面程式**（`app/src-tauri` + React/Vite/TS）：概覽、模型（推薦 / 已安裝 / 畫圖影片 / HF 搜尋）、執行、對話、畫圖、影片、Agent、設定；i18n zh-TW / en；全新 UI 設計。Tauri command 直接包 core 的函式，事件轉發 `ServerEvent` 與下載進度；審批請求以事件送前端、`resolve` 由前端呼叫。
2. **對話 / Agent 的 LLM 迴圈**：OpenAI 相容串流客戶端（本機 llama-server 與雲端供應商共用），工具呼叫最多 12 輪 / Agent 150 輪。
3. **CI**：`.github/workflows/ci.yml` 已有 `cargo test`（Windows / Ubuntu / macOS）；加上 `tauri build`。
4. Phase 2：computer use（`xcap` + `enigo`）、附件解析（PDF / DOCX / PPTX / XLSX）、LTX-2.3 與 MiniMax-H3 影片模型（元件參數待確認）。
