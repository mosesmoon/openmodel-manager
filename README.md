# OpenModel-Manager

開源、跨平台（Windows / Linux / macOS）的本機 AI 模型管理工具：偵測硬體、規劃模型怎麼分配到 GPU / 記憶體 / 硬碟、安裝 llama.cpp 與 stable-diffusion.cpp、從 Hugging Face 下載模型、啟動 OpenAI 相容的本機伺服器，並提供對話工具與 Agent。

An open-source, cross-platform local AI model manager: hardware detection, GPU/RAM/disk offload planning, llama.cpp and stable-diffusion.cpp installs, Hugging Face downloads, an OpenAI-compatible local server, chat tools and a coding agent.

## 設計原則 / Principles

- **沒有遙測**：不上報任何安裝、裝置或使用資料。No telemetry of any kind.
- **沒有代理 / 付費通道**：只連你設定的主機（Hugging Face 或你選的鏡像、GitHub 或你選的加速前綴）。Only the hosts you configure are contacted.
- **工具審批在後端強制**：執行指令、刪除檔案、存取工作資料夾以外的路徑，一律在 Rust 後端等使用者同意；前端無法繞過。Tool approval is enforced in the backend, not the UI.
- **AMD / Intel 是一等公民**：Vulkan 與 ROCm 和 CUDA 一樣優先支援。

## 目前狀態 / Status

| 模組 | 狀態 |
|---|---|
| `openmodel-core`：硬體偵測、GGUF 解析、分層規劃、HF、續傳下載、引擎安裝、llama-server / sd-server 管理、對話工具、Agent 沙箱 | 完成，55 個單元測試 |
| `openmodel` CLI | 完成（`hw` `gguf` `catalog` `plan` `hf` `models` `engine` `serve`） |
| Tauri 2 桌面程式 | 尚未開始 |

詳見 [docs/STATUS.md](docs/STATUS.md)。

## 建置 / Build

需要 Rust stable（1.80+）。

```bash
cargo test
cargo build --release
```

## CLI 範例

```bash
openmodel hw                                             # 硬體與建議後端
openmodel plan --catalog qwen3.6-35b-a3b --quant UD-Q4_K_XL
openmodel hf files unsloth/gpt-oss-20b-GGUF
openmodel hf download unsloth/gpt-oss-20b-GGUF Q4_K_M
openmodel engine install                                 # 自動選 CUDA / Vulkan / ROCm / Metal / CPU
openmodel engine install --backend vulkan
openmodel serve <path/to/model.gguf> --lan               # 失敗時自動以相容模式重試
```

資料預設放在系統的應用程式資料夾；設定環境變數 `OPENMODEL_HOME` 可改位置，或在執行檔旁放一個 `portable.txt` 啟用可攜模式（資料放在 `OpenModelData/`）。

## 授權 / License

Apache-2.0 OR MIT，任選其一。見 [LICENSE-APACHE](LICENSE-APACHE)、[LICENSE-MIT](LICENSE-MIT)。

本專案是獨立的 clean-room 實作，沒有使用任何閉源產品的程式碼。This is an independent clean-room implementation.
