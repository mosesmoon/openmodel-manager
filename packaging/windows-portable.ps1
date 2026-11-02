# Build the portable (no-install) Windows package from the release build.
#
#   cd app; npx tauri build --bundles msi     # builds target\release\openmodel-app.exe
#   powershell -File packaging\windows-portable.ps1
#
# The zip holds OpenModel-Manager.exe and portable.txt; with portable.txt next
# to the executable all data goes to OpenModelData\ beside it.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$conf = Get-Content -Raw "$root\app\src-tauri\tauri.conf.json" | ConvertFrom-Json
$version = $conf.version
$exe = "$root\target\release\openmodel-app.exe"
if (-not (Test-Path $exe)) { throw "Build first: cd app; npx tauri build --bundles msi" }

$name = "OpenModel-Manager_${version}_x64_portable"
$outDir = "$root\target\release\bundle\portable"
$stage = "$outDir\$name"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null

Copy-Item $exe "$stage\OpenModel-Manager.exe"
# Default folders, already in place after unpacking (Settings points here).
foreach ($d in 'models', 'engines', 'outputs') {
  New-Item -ItemType Directory -Force "$stage\OpenModelData\$d" | Out-Null
  Set-Content -Encoding UTF8 "$stage\OpenModelData\$d\README.txt" "Default $d folder of the portable OpenModel-Manager. 免安裝版預設的 $d 資料夾。"
}
Set-Content -Encoding Ascii "$stage\portable.txt" "This file turns on portable mode: settings, models and engines are kept in OpenModelData\ next to the program."
$readme = @"
OpenModel-Manager $version - portable

Run OpenModel-Manager.exe. Nothing is installed. Settings, downloaded models,
engines and outputs live in OpenModelData next to the program:
  OpenModelData\models    models
  OpenModelData\engines   llama.cpp / stable-diffusion.cpp / Strata / ...
  OpenModelData\outputs   generated images and videos
Paths inside the folder are stored relative to it, so the whole folder can be
moved or copied to another drive, PC or USB stick.

Delete portable.txt to keep data in the normal per-user folders instead.
Needs Windows 10/11 x64 with the Microsoft Edge WebView2 runtime (included in
Windows 11 and current Windows 10).

可攜版：直接執行 OpenModel-Manager.exe，不需安裝。設定、模型、引擎與輸出都放在
程式旁邊的 OpenModelData（models、engines、outputs）。資料夾內的路徑以相對路徑
保存，整個資料夾可以搬到別的磁碟、別台電腦或隨身碟。
刪除 portable.txt 則改用一般的使用者資料夾。
"@
Set-Content -Encoding UTF8 "$stage\README.txt" $readme

$zip = "$outDir\$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path "$stage\*" -DestinationPath $zip
Remove-Item -Recurse -Force $stage
Get-Item $zip | Select-Object FullName, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 2) } }
