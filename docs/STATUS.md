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
| `engine` | GitHub release 資產挑選（llama.cpp 與 sd.cpp 命名、CUDA 版本依驅動（含新版 `CUDA UMD Version` 標頭）、Windows cudart）、掃描最近 15 個 release 取第一個有對應檔案的版本（llama.cpp 的 latest 不含 build）、下載解壓（防 zip-slip）、匯入 zip / 既有資料夾 | 7 |
| `pyengine` | vLLM（Linux，NVIDIA CUDA 用 PyPI、AMD ROCm 用 wheels.vllm.ai/rocm 套件庫，依偏好後端自動選擇，可手動切換）與 MLX／mlx-lm（macOS Apple Silicon）：用 uv（從 GitHub release 下載並核對 SHA-256）建 Python 3.12 venv、從 PyPI 安裝與 `--upgrade` 更新；PyPI 查最新版；`python -m vllm.entrypoints.openai.api_server` / `python -m mlx_lm.server` 啟動，`/v1/models` 判斷就緒，模型下載到 `<models>/hf`；不支援的系統在介面上反灰。已實測 uv 下載與建 venv（Windows），vLLM／MLX 本身未在其平台實測 | 3（+1 個 `--ignored`） |
| `strata` | 第三方引擎 [Strata](https://github.com/Niko1221/Strata)（MIT）：安裝其 Terminal 套件（核對 install-manifest 的 SHA-256、跑 `strata update --check` 驗簽章），資料放在 `<engines>/strata/data`（`STRATA_DATA_DIR`）；用 `strata launch --model-path … --no-tui --restart` 啟動、`/health` 判斷就緒、`strata launch --stop` 停止；LAN 模式自動產生 API key。只呼叫它的程式，不含它的程式碼 | 4（+1 個 `--ignored` 實機安裝測試） |
| `server` | 啟動 / 停止、log 擷取與載入階段解析、`/health`、port 自動往後找、崩潰原因分類、相容模式自動重試 | 5 |
| `diffusion` | 7 個圖片 / 影片模型清單（Z-Image、FLUX.2 klein、FLUX.1 schnell、Qwen-Image、Qwen-Image-Edit、Wan2.2 TI2V 5B、Wan2.2 I2V A14B）、sd-server 參數、txt2img / img2img / 影片 job API、尺寸與幀數 | 5 |
| `tools` | 審批閘門（後端強制、本次 / 本對話總是允許 / 拒絕）、計算機（evalexpr，無程式執行）、Bing RSS 搜尋、網頁擷取（拒絕本機與私有網段，逐跳檢查轉址）、檔案與指令工具 | 7 |
| `agent` | 工作資料夾沙箱（解析 `..` 與 symlink 後判斷）、11 個工具的 schema、資料夾外讀寫與 `run_command` / `delete_path` 一律詢問 | 5 |
| `library` | 掃描本機模型（分卷只列一次、自動配對 mmproj）、登記外部模型（不複製、移除時不刪檔） | 1 |
| `net` | 偵測有連線的 IPv4 網卡（排除 loopback 與 169.254）；伺服器一律只聽 127.0.0.1，勾選的網卡由 TCP 轉接到同一個 port。設定的 API key 經環境變數交給 llama.cpp（`LLAMA_API_KEY`）、vLLM（`VLLM_API_KEY`）、Strata（`--api-key-env`）；MLX 無驗證。已實測 llama-server 的 key（無 key／錯 key 401、正確 200）與經 LAN IP 轉接連線 | 2 |
| `jit`（app） | OpenAI API 閘道：一律佔用設定的連接埠（127.0.0.1 與勾選的網卡），模型伺服器改用下一個空的連接埠且只聽 127.0.0.1。回應 `GET /v1`、`/v1/models`、`/v1/models/{id}`、`/props`，其他 `/v1/*` 轉給模型（串流照轉）；檢查 API key。JIT 開啟時列出全部已下載模型並在呼叫時掛載，GPU 剩餘 VRAM 放得下（`nvidia-smi`）就並存，否則先停掉已載入的模型。已實測：JIT 關／開、未知模型、按需載入 Gemma 4 26B（約 12 秒）、串流、`/props`、API key 401／200 | 2 |
| `llm` | OpenAI 相容串流（SSE，含 `reasoning_content`、分段 tool call、usage / tok/s）；工具迴圈（拒絕的工具以結果回報給模型、可取消）；`/models` 列表；本機與遠端 endpoint 共用 | 3 |
| `anthropic` | Anthropic Messages API（raw HTTP，`anthropic-version: 2026-10-21`）：把 OpenAI 形狀的對話轉成 Messages 格式（system、圖片、`tool_use`／`tool_result` 合併）、解析 SSE（text／thinking／signature／`input_json_delta`／usage／錯誤）；每輪的原始內容區塊存在訊息的 `_anthropic` 欄位、下一輪原樣送回；新模型開 adaptive thinking（summarized）與 `fallbacks: "default"`。設定頁「AI 服務供應商」：Anthropic、OpenAI、Google AI Studio、xAI、NVIDIA NIM、自訂 OpenAI 相容。訂閱登入（ChatGPT／Codex、Claude OAuth）不做：目前做法都要冒用官方 CLI 身分 | 4 |

`crates/cli`：`openmodel hw | gguf | catalog | plan | hf search/files/download | models | engine list/install/import | serve`。

## 桌面程式（`app/`）

- `app/src-tauri/src/lib.rs`：33 個 Tauri command，直接包 core。下載 / 安裝 / 影片是背景 job，透過 `progress`、`job-done` 事件回報、可取消；伺服器輸出走 `server` 事件；對話串流走 `chat` 事件；工具審批走 `approval` 事件，前端只能回答正在等待的請求（`approval_resolve`），無法直接執行受保護的工具。關閉視窗時會停掉 llama-server / sd-server / Strata。執行頁可選 llama.cpp 或 Strata（模型屬於 Strata 支援的系列且已安裝時預設 Strata）。
- `app/src`（React + Vite + TS）：概覽（硬體、即時 VRAM、VRAM 溢出警告、伺服器狀態）、模型（推薦含每個量化的預估速度與 GPU/RAM/硬碟分配、已安裝、畫圖 / 影片模型、HF 搜尋）、執行（配置計畫、實際指令、自訂參數每模型記住、載入進度、log、崩潰原因）、對話（本機或遠端 API、工具開關、附加圖片、思考過程、工具呼叫與結果、審批卡片）、畫圖、影片、Agent（工作資料夾、步驟清單、審批）、設定（資料夾、引擎安裝 / 匯入 / 移除、偏好後端、HF 鏡像與 token、GitHub 前綴、LAN、連接埠、保留記憶體、手動 GPU、遠端 API）。
- i18n：`app/src/i18n.tsx`，繁中為預設，英文完整對應（缺字會在型別檢查時報錯）。
- 已驗證：`tauri build --debug --no-bundle` 在 Windows 建置成功並能啟動；用模擬後端在瀏覽器跑過全部 8 頁、一輪含工具呼叫的對話、一次 Agent 審批。**尚未**用真的引擎與模型在 app 裡完整跑過。

## 網頁管理介面（`app/src-tauri/src/web.rs`、`app/src/bridge.ts`）

- 桌面程式內建 HTTP 伺服器（axum），提供程式內打包的同一份前端；`POST /api/invoke/<command>` 執行與視窗相同的 command，`GET /api/events` 以 SSE 轉送 `server`／`progress`／`job-done`／`chat`／`approval` 事件。
- 登入：密碼以加鹽 SHA-256（10 萬次）存在設定檔；登入後發 HttpOnly、SameSite=Strict 的 session cookie，錯誤密碼延遲 1 秒；改密碼會登出所有瀏覽器。`save_settings` 不接受前端送來的密碼雜湊。
- 預設只聽 127.0.0.1:8765；可另外勾選網卡。開關與網卡變更後重新啟動，先等舊的 socket 關閉再綁定。
- 瀏覽器裡沒有檔案選擇器：改為輸入路徑；「開啟資料夾」按鈕隱藏。
- 已實測（Windows debug 版）：登入／錯誤密碼／未登入擋下、指令與 SSE 事件、儲存後重新啟動（本機與 LAN IP）、換連接埠、換密碼後舊 session 失效；瀏覽器內登入、選單四個位置與收合（偏好存在 localStorage）。

## 供應鏈安全（`.github/workflows/security.yml`、`ci.yml`）

- `cargo audit`（RustSec）與 `npm audit`：推送、PR、每週一；有漏洞即失敗，JSON 報告存為 artifact。
- SBOM：syft 產生 SPDX 與 CycloneDX JSON，涵蓋 Cargo.lock 與 app/package-lock.json。
- 版本標籤的打包：`actions/attest-build-provenance` 與 `actions/attest-sbom` 為每個安裝檔簽署來源證明。
- 2026-11-02 本機掃描：`zip` 2.6.1 有 RUSTSEC-2026-0367／0368／0369，已升級到 9.2.2；解壓縮時另外遮掉 setuid／setgid／sticky 位元。之後 Rust 與 npm 均為 0 個漏洞；剩 5 個 Tauri Linux 相依的 unmaintained／unsound 警告。
- 雲端尚未實際跑過這兩個新流程。

## 尚未驗證（需要使用者在測試機上跑）

- `engine install` 與 `serve` 尚未實際下載引擎與載入模型；只在單元測試中用假程序驗證崩潰流程。
- 相容模式的環境變數（`GGML_CUDA_DISABLE_GRAPHS`、`GGML_CUDA_FORCE_CUBLAS`）和 llama.cpp 目前版本的對應。
- sd-server 的元件參數名稱（`--llm`、`--llm_vision`、`--clip_l`、`--t5xxl`、`--vae`）以 stable-diffusion.cpp 現行版本為準，需實測。
- Strata：安裝流程已實測（下載、SHA-256、簽章、`launch --help` 參數）；尚未用真的模型啟動過（需要 Qwen3.8-Flash-Next / Qwen3.6-35B-A3B / DeepSeekV4.1-Flash 的檔案，第一次啟動會再下載 GPU 引擎）。
- tok/s 估算的效率係數（GPU 0.65、CPU 0.55）需用三台測試機的實測值校正。

建議的測試指令（每台機器）：

```bash
openmodel hw
openmodel engine install
openmodel hf download unsloth/gpt-oss-20b-GGUF Q4_K_M
openmodel serve <models_dir>/unsloth__gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf -v
```

## 封裝（`packaging/`）

| 平台 | 產物 | 建置方式 |
|---|---|---|
| Windows | MSI 安裝檔（zh-TW、en-US 各一個） | `cd app && npx tauri build --bundles msi`。`packaging/windows/language.wxs` 把安裝語言寫進程式旁的 `install.ini`，程式第一次開啟（或換語言重裝後）用同一種語言 |
| Windows | 免安裝 zip | 先建 MSI，再 `powershell -File packaging/windows-portable.ps1`。zip 內含 `portable.txt` 與 `OpenModelData/{models,engines,outputs}`；設定中資料夾內的路徑存成相對路徑，WebView2 資料也放在 `OpenModelData/webview` |
| Linux | AppImage、deb | `packaging/linux/build.sh appimage`：在 Ubuntu 22.04 容器內建置（glibc 2.35），Node 24.22.1 以 SHA-256 核對 |
| Linux | Flatpak | `packaging/linux/build.sh flatpak`：GNOME 51 runtime，用 deb 重新包裝；使用者層級的 `org.flatpak.Builder`，不需 sudo |
| macOS | DMG（拖到「應用程式」） | 在 Mac 上 `packaging/macos/build.sh`，universal（Apple Silicon + Intel），未簽章 |

2026-11-02 已建置：Windows（本機）、Linux（moses-server，Docker + Flatpak）。macOS 尚未建置（需要 Mac）。CI 的 `bundle` 工作會在三平台產生上述全部格式。

## 下一步

1. **實機試用桌面程式**：安裝引擎 → 下載模型 → 執行 → 對話 / Agent；畫圖與影片需要 sd.cpp 引擎與模型。
2. 對話訊息目前以純文字顯示；加 Markdown 與程式碼區塊渲染。對話紀錄尚未存檔。
3. **CI**：`ci.yml` 已加上前端建置、Linux WebKit 套件與三平台 `tauri build`（上傳安裝檔為 artifact），尚未在 GitHub 上實際跑過。
4. Phase 2：computer use（`xcap` + `enigo`）、附件解析（PDF / DOCX / PPTX / XLSX）、LTX-2.3 與 MiniMax-H3 影片模型（元件參數待確認）。
