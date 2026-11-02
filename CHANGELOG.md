# Changelog

本檔記錄每個版本的變更，格式參考 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)。英文版在每個版本的中文段落之後。

This file records the changes in each version, loosely following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The English text follows the Chinese in each version.

> Beta 版的程式內版本號仍是 0.1.0，安裝檔檔名也是 `0.1.0`，正式發行 0.1.1 時才會更新。
> Beta builds still carry version 0.1.0 inside the app and in the installer file names; it changes at the 0.1.1 release.

## [Unreleased]

### 新增
- Windows MSI 分成繁體中文（`zh-TW`）和英文（`en-US`）兩個，安裝畫面和程式第一次開啟的語言都跟著安裝檔。用另一種語言的 MSI 重裝，介面也會跟著換。
- 語言選單移到導覽選單最上方，選了立即套用。

### 變更
- 免安裝版第一次開啟改為英文。

### Added
- The Windows MSI comes in Traditional Chinese (`zh-TW`) and English (`en-US`); both the installer and the app's first launch use that language, and reinstalling with the other MSI switches the UI too.
- The language menu is now at the top of the navigation menu and applies at once.

### Changed
- The portable build now opens in English the first time.

## [0.1.1-beta.2] - 2026-11-02

### 新增
- **供應鏈安全**：
  - CI 用 `cargo audit`（RustSec）和 `npm audit` 掃描相依套件，推送、PR 和每週一都會跑，有漏洞就判定失敗。
  - 掃描通過後產生 SPDX 與 CycloneDX 格式的 SBOM。
  - 版本標籤的安裝檔附 GitHub 簽署的建置來源證明與 SBOM 證明，可用 `gh attestation verify` 驗證。
- README 加入 CI／Security 徽章、「供應鏈安全」說明和本 Changelog。

### 安全性
- `zip` 從 2.6.1 升級到 9.2.2，修正 RUSTSEC-2026-0367、0368、0369。
- 解壓縮引擎時一律去掉 setuid、setgid、sticky 權限位元。

### 修正
- Linux 版 CI：Flatpak 改在 D-Bus 工作階段中打包。Beta 1 在這一步失敗，所以沒有產出 Linux 安裝檔。

### Added
- **Supply-chain security**:
  - CI scans dependencies with `cargo audit` (RustSec) and `npm audit` on every push, every pull request and every Monday, and fails on any vulnerability.
  - A passing scan produces SBOMs in SPDX and CycloneDX format.
  - Installers from version tags carry GitHub-signed build provenance and SBOM attestations, which you can check with `gh attestation verify`.
- README: CI and Security badges, a supply-chain section, and this changelog.

### Security
- `zip` upgraded from 2.6.1 to 9.2.2, fixing RUSTSEC-2026-0367, 0368 and 0369.
- Engine archives are now always unpacked without setuid, setgid or sticky bits.

### Fixed
- Linux CI: the Flatpak is now built inside a D-Bus session. Beta 1 failed at this step, so it produced no Linux installers.

## [0.1.1-beta.1] - 2026-11-02

### 新增
- **網頁管理介面**：
  - 其他電腦或手機可以用瀏覽器開啟和桌面版相同的介面。
  - 用密碼登入（加鹽雜湊儲存、HttpOnly session），可選擇開放在哪些網路卡。
  - 設定頁有綠／紅滑動開關可以啟用或停用。
- **導覽選單位置**：可放在左、右、上、下，邊緣的半圓按鈕可以收合。每台裝置和瀏覽器各自記住。
- **本機伺服器 API key**：
  - 可自訂或自動產生，留空就是不需要。
  - llama.cpp、vLLM、Strata 會檢查 key；MLX 本身沒有驗證功能。
  - 概覽頁會顯示 key，可以複製。
- **選擇開放的網路卡**：
  - 只列出有連上網路的 IPv4 位址，可逐一勾選。
  - 模型伺服器只聽 127.0.0.1，勾選的位址由程式轉接過去。

### 變更
- 「允許區網連線（0.0.0.0）」改為「允許本機以外的電腦連線」，並可選擇網路卡。

### 修正
- 測試在 Linux／macOS 誤把 `D:/…` 當成相對路徑。

### Added
- **Web management page**:
  - Other computers and phones can open the same UI as the desktop app in a browser.
  - Password sign-in (salted hash, HttpOnly session) and a choice of network cards to open it on.
  - A green/red switch in Settings turns it on or off.
- **Menu position**: left, right, top or bottom, with a half-circle button to fold the menu. Each device and browser remembers its own choice.
- **API key for the local server**:
  - Enter your own or generate one; leave it blank for no key.
  - llama.cpp, vLLM and Strata check the key; MLX has no authentication of its own.
  - The Overview page shows the key with a Copy button.
- **Choice of network cards**:
  - Only connected IPv4 addresses are listed, each with its own checkbox.
  - Model servers listen only on 127.0.0.1, and the app forwards the ticked addresses to them.

### Changed
- "Allow LAN connections (0.0.0.0)" is now "Allow connections from other computers", with a choice of network cards.

### Fixed
- A test treated `D:/…` as a relative path on Linux and macOS.

## [0.1.0] - 2026-11-02

第一版（未正式發行）。First version (not released).

- **硬體與規劃**：偵測硬體；規劃 GPU／RAM／硬碟配置並估算速度；崩潰時自動用相容模式重試。
- **引擎**：llama.cpp、stable-diffusion.cpp、Strata、vLLM（CUDA／ROCm）、MLX，都能從程式內下載與更新。
- **模型目錄**：可依能力、來源國家（附國旗）和 Strata 支援篩選；可搜尋 Hugging Face，下載能續傳。
- **使用功能**：對話（本機模型或 Anthropic、OpenAI、Google AI Studio、xAI、NVIDIA NIM）、Agent（後端強制工具審批）、AI 生成圖片與影片。
- **打包**：Windows MSI 與免安裝版、Linux AppImage／deb／Flatpak、macOS DMG。
- **文件**：README 提供繁體中文與英文兩版，附各頁面截圖。

Summary in English:

- **Hardware and planning**: hardware detection; GPU/RAM/disk placement planning with speed estimates; automatic retry in compatibility mode after a crash.
- **Engines**: llama.cpp, stable-diffusion.cpp, Strata, vLLM (CUDA/ROCm) and MLX, each installed and updated from inside the app.
- **Model catalog**: filters by capability, country of origin (with flags) and Strata support; Hugging Face search; resumable downloads.
- **Features**: chat with local models or with Anthropic, OpenAI, Google AI Studio, xAI or NVIDIA NIM; an agent whose tool approval is enforced in the backend; AI image and video generation.
- **Packaging**: Windows MSI and portable zip, Linux AppImage/deb/Flatpak, macOS DMG.
- **Docs**: README in Traditional Chinese and English, with screenshots of every page.

[Unreleased]: https://github.com/mosesmoon/openmodel-manager/compare/v0.1.1-beta.2...main
[0.1.1-beta.2]: https://github.com/mosesmoon/openmodel-manager/compare/v0.1.1-beta.1...v0.1.1-beta.2
[0.1.1-beta.1]: https://github.com/mosesmoon/openmodel-manager/releases/tag/v0.1.1-beta.1
[0.1.0]: https://github.com/mosesmoon/openmodel-manager/tree/d71c8d1
