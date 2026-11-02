# OpenModel-Manager

**繁體中文** | [English](README.en.md)

[![CI](https://github.com/mosesmoon/openmodel-manager/actions/workflows/ci.yml/badge.svg)](https://github.com/mosesmoon/openmodel-manager/actions/workflows/ci.yml) [![Security](https://github.com/mosesmoon/openmodel-manager/actions/workflows/security.yml/badge.svg)](https://github.com/mosesmoon/openmodel-manager/actions/workflows/security.yml)

開源、跨平台（Windows / Linux / macOS）的本機 AI 模型管理工具。它會偵測你的硬體，規劃模型要怎麼分配到 GPU、記憶體和硬碟，接著幫你安裝推論引擎、從 Hugging Face 下載模型，並啟動 OpenAI 相容的本機伺服器。程式內建對話、Agent、AI 生成圖片與 AI 生成影片。

![概覽](screenshots/01-概覽.png)

## 功能

- **硬體偵測**：支援 NVIDIA、AMD、Intel 和 Apple Silicon，會讀取 VRAM（含 APU 共用記憶體）、RAM 類型、SSD / HDD。
- **配置規劃**：在下載前就估算每個量化版本的 GPU / RAM / 硬碟分配和預估 tok/s。放不進 GPU 時，依序改用 MoE 專家卸載、層卸載、mmap 溢出到硬碟。也會自動選最大且不降速的 context。
- **推論引擎**，都能從程式內下載與更新：
  - [llama.cpp](https://github.com/ggml-org/llama.cpp)：CUDA / Vulkan / ROCm / Metal / CPU，自動選擇。
  - [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)：生成圖片與影片。
  - [Strata](https://github.com/Niko1221/Strata)：超大 MoE 模型（例如 DeepSeek），支援 Mac 與 AMD GPU。
  - [vLLM](https://github.com/vllm-project/vllm)：Linux，NVIDIA CUDA 或 AMD ROCm。
  - [MLX](https://github.com/ml-explore/mlx-lm)：macOS Apple Silicon。
- **模型目錄**：推薦清單可依能力（chat / code / vision / reasoning）、來源國家（🇺🇸 🇨🇳 🇫🇷 …）和 Strata 支援篩選，也能直接搜尋 Hugging Face。下載可續傳，鏡像與 token 可自訂。
- **崩潰自動處理**：載入失敗時會判斷原因，並自動用相容模式重試。
- **對話**：可接本機模型或 AI 服務供應商，包括 Anthropic、OpenAI、Google AI Studio、xAI Grok、NVIDIA NIM 及任何 OpenAI 相容 API。支援工具呼叫、圖片輸入、顯示思考過程。
- **Agent**：在指定的工作資料夾內讀寫檔案、執行指令，每一步都有清單可看。
- **AI 生成圖片 / 影片**：Z-Image、FLUX.2 klein、FLUX.1 schnell、Qwen-Image、Wan2.2 等。
- **OpenAI 相容 API**：啟動的模型可以給其他程式和 Agent（例如 Hermes Agent）使用。可設定 API key（也可留空），並勾選要開放的網路卡。開啟 JIT 後，沒有執行模型時 API 仍會列出已下載的模型，呼叫時自動掛載。
- **網頁管理介面**：其他電腦或手機可用瀏覽器開啟和桌面版相同的介面，有密碼登入和綠／紅啟用開關，也能選要開放的網路卡。
- **自訂版面**：導覽選單可以放在左、右、上、下，邊緣的半圓按鈕可以收合選單。
- **供應鏈安全**：CI 掃描相依套件漏洞，產生 SPDX 與 CycloneDX 格式的 SBOM，並為安裝檔簽署建置來源證明。
- **介面語言**：繁體中文與英文，從導覽選單最上方切換。中文版 MSI 第一次開啟是中文，英文版 MSI 和免安裝版是英文。

更新內容見 [CHANGELOG.md](CHANGELOG.md)。

## 設計原則

- **沒有遙測**：不上報任何安裝、裝置或使用資料。
- **沒有中轉代理或付費通道**：只連你設定的主機，例如 Hugging Face 或你選的鏡像、GitHub 或你選的加速前綴，以及你自己加入的 AI 服務供應商。
- **工具審批在後端強制**：執行指令、刪除檔案、存取工作資料夾以外的路徑，一律由 Rust 後端等你同意，前端無法繞過。
- **AMD / Intel 是一等公民**：Vulkan 與 ROCm 和 CUDA 一樣優先支援。

## 下載與安裝

請到 [Releases](../../releases) 下載。

| 平台 | 檔案 | 說明 |
|---|---|---|
| Windows | `*_zh-TW.msi` | 安裝版（繁體中文：安裝畫面與程式都是中文） |
| Windows | `*_en-US.msi` | 安裝版（英文：安裝畫面與程式都是英文） |
| Windows | `*_portable.zip` | 免安裝版（預設英文）：解壓即可用，設定、模型、引擎、輸出都放在旁邊的 `OpenModelData/` |
| Linux | `.AppImage` | 需要 glibc 2.35 以上（Ubuntu 22.04+） |
| Linux | `.flatpak` | `flatpak install --user OpenModel-Manager.flatpak` |
| Linux | `.deb` | Debian / Ubuntu |
| macOS | `.dmg` | 打開後把程式拖到 Applications。未簽章，第一次開啟要按右鍵 → 打開 |

## 讓其他程式透過 OpenAI API 連進來

OpenModel-Manager 在 `http://127.0.0.1:8080/v1`（連接埠可在設定修改，概覽頁會顯示網址）提供 OpenAI 相容的 API。任何支援 OpenAI API 的工具都能直接連，例如 Hermes Agent、Open WebUI、Continue、Cline、或 `openai` SDK。在這些工具填「自訂端點」時，網址填到 `/v1` 為止：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "你好"}],
)
print(r.choices[0].message.content)
```

要讓其他電腦連進來：

1. 到「設定」勾選「允許本機以外的電腦連線」。
2. 勾選要開放的網路卡。清單只列出有連上網路的 IPv4 位址，都沒勾時會開放全部網卡。
3. 概覽頁會列出每個可連線的網址。

模型伺服器本身只聽 `127.0.0.1`，OpenModel-Manager 的 API 閘道在你勾選的位址上接收連線再轉過去，所以沒勾的網卡連不進來。沒勾「允許本機以外的電腦連線」時，用區網 IP（例如 `http://192.168.0.101:8080/v1`）連線會被拒絕。

**保持 API 開啟（JIT）**：在「執行」頁打開這個開關後：

- 沒有執行模型時，`GET /v1/models` 仍會列出所有已下載的模型。
- 其他程式呼叫某個模型時（請求中的 `model` 填清單裡的名稱），會自動掛載該模型再回應。
- 已經有模型在執行時，GPU 放得下就讓兩個模型並存；放不下就先卸載既有的模型。

沒開 JIT 時，所有呼叫都交給目前執行中的模型；沒有模型在執行時會回傳 503。

**API key**：可在「設定」填入或按「產生」，留空就是不需要 key。設定後，用戶端要帶 `Authorization: Bearer <key>`（上面範例的 `api_key=`）。llama.cpp、vLLM、Strata 都會檢查 key，MLX 沒有驗證功能。開放給其他電腦時建議設定 key。

## 網頁管理介面

可以從其他電腦或手機用瀏覽器開啟和桌面程式一樣的管理介面：

1. 在桌面程式的「設定 → 網頁管理介面」設定登入密碼（至少 6 個字元）。
2. 打開右上角的開關（綠色是啟用，紅色是停用）。
3. 預設只開放本機 `http://127.0.0.1:8765/`。要讓其他裝置連線，勾選「允許本機以外的電腦開啟網頁介面」，再勾選要開放的網路卡。

網頁版沒辦法瀏覽那台電腦的檔案，需要選資料夾或檔案時會改成輸入路徑。網頁介面使用一般 HTTP，請只在信任的網路開放。

導覽選單可以在「設定 → 一般」改放在左、右、上、下，選單邊緣的半圓按鈕可以收合選單。這個偏好只存在目前的裝置與瀏覽器。

## 截圖

| | |
|---|---|
| ![模型](screenshots/02-模型-推薦.png) | ![執行](screenshots/06-執行.png) |
| ![對話](screenshots/07-對話.png) | ![Agent](screenshots/10-Agent.png) |
| ![AI 生成圖片](screenshots/08-AI生成圖片.png) | ![設定](screenshots/11-設定.png) |

## 供應鏈安全

上方的 **Security** 徽章顯示最新一次掃描結果（綠色＝通過）。

- **套件掃描**：每次推送、每個 Pull Request、以及每週一自動掃描所有相依套件，有任何已知漏洞就會失敗。
  - Rust：`cargo audit`，比對 [RustSec](https://rustsec.org/) 資料庫。
  - 前端：`npm audit`，比對 npm 漏洞資料庫。
  - 完整報告（JSON）在每次執行的 `security-report` artifact。
- **SBOM（軟體物料清單）**：每次掃描通過後產生，同時涵蓋 Rust 與前端套件，提供兩種通用格式：
  - SPDX：`openmodel-manager.spdx.json`
  - CycloneDX：`openmodel-manager.cdx.json`
  - 在 [Security 執行紀錄](https://github.com/mosesmoon/openmodel-manager/actions/workflows/security.yml) 的 `sbom` artifact 下載；正式發行版也會附在 Release 檔案中。
- **建置來源證明**：推送版本標籤時，CI 會替每個安裝檔產生 GitHub 簽署的證明，載明它是由這個儲存庫、哪個提交、哪次 CI 執行建出來的，並把 SBOM 綁到安裝檔上。下載後可以驗證：

  ```bash
  gh attestation verify OpenModel-Manager_0.1.0_x64_en-US.msi -R mosesmoon/openmodel-manager
  ```

已知警告：Tauri 在 Linux 使用的幾個元件（`glib`、`zbus` 系列、`proc-macro-error`）被標為「不再維護」或「有不安全用法」。這些不是漏洞，掃描會列出但不判定失敗，會隨 Tauri 更新處理。

## 從原始碼建置

需要 Rust stable（1.80+）與 Node.js 24.22.1 以上（較舊版本受 CVE-2026-111497 影響）。Linux 另需 `libwebkit2gtk-4.1-dev` 等套件，詳見 [Tauri 前置需求](https://v2.tauri.app/start/prerequisites/)。

```bash
cd app && npm ci && npm run build && cd ..   # 先建前端（編譯桌面程式需要 app/dist）
cargo test
cargo build --release -p openmodel-cli       # 只建 CLI

cd app
npx tauri dev                                # 開發模式
npx tauri build                              # 產生安裝檔（target/release/bundle/）
```

打包腳本在 [`packaging/`](packaging/)：

| 腳本 | 產出 |
|---|---|
| `packaging/windows-portable.ps1` | Windows 免安裝 zip |
| `packaging/linux/build.sh` | 在 Ubuntu 22.04 容器內建 AppImage、deb、Flatpak |
| `packaging/macos/build.sh` | macOS 通用版 DMG（要在 Mac 上執行） |

## CLI

```bash
openmodel hw                                             # 硬體與建議後端
openmodel plan --catalog qwen3.6-35b-a3b --quant UD-Q4_K_XL
openmodel hf files unsloth/gpt-oss-20b-GGUF
openmodel hf download unsloth/gpt-oss-20b-GGUF Q4_K_M
openmodel engine install                                 # 自動選 CUDA / Vulkan / ROCm / Metal / CPU
openmodel engine install --backend vulkan
openmodel serve <path/to/model.gguf> --lan               # 失敗時自動以相容模式重試
```

資料預設放在系統的應用程式資料夾。設定環境變數 `OPENMODEL_HOME` 可以改位置；在執行檔旁放一個 `portable.txt` 則會啟用可攜模式，資料放在 `OpenModelData/`。

## 專案結構

| 路徑 | 內容 |
|---|---|
| `crates/core` | 核心函式庫：硬體、GGUF、配置規劃、下載、引擎、伺服器、LLM 串流、工具、Agent |
| `crates/cli` | `openmodel` 命令列工具 |
| `app/` | Tauri 2 桌面程式（React + Vite + TypeScript） |
| `packaging/` | 各平台打包腳本 |
| `docs/STATUS.md` | 進度、已驗證與未驗證的項目 |

## 目前狀態

這是 0.1.0 版。核心功能都有單元測試，桌面程式能在 Windows 建置和執行。各引擎與模型還沒有在所有平台實測，詳見 [docs/STATUS.md](docs/STATUS.md)。歡迎回報問題。

## 授權

Apache-2.0 OR MIT，任選其一。見 [LICENSE-APACHE](LICENSE-APACHE)、[LICENSE-MIT](LICENSE-MIT)。

第三方素材：國旗字型 `app/public/fonts/TwemojiCountryFlags.woff2` 的圖案來自 [Twemoji](https://github.com/twitter/twemoji)（CC BY 4.0），字型子集由 [country-flag-emoji-polyfill](https://github.com/talkjs/country-flag-emoji-polyfill)（MIT）製作，詳見同目錄的 `TwemojiCountryFlags.LICENSE.txt`。推論引擎是在執行時另外下載的獨立程式，各自有自己的授權。
