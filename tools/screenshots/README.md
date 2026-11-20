# README screenshots

Regenerates the 11 README screenshots (1920x1080 PNG) without running the
desktop app. The built frontend is loaded in a headless browser where `mock.js`
stands in for the Tauri IPC and returns fake data, so the pictures never show
real paths, keys or hardware.

## Run

```sh
# once
npm --prefix tools/screenshots install

# build the frontend (any output dir; app/dist works too)
npm --prefix app run build -- --outDir ../tools/screenshots/.build --emptyOutDir

# capture: node capture.mjs [frontend-build-dir] [output-dir]
node tools/screenshots/capture.mjs tools/screenshots/.build
```

Without arguments it uses `app/dist` and writes to `tools/screenshots/out`
(git-ignored, as is `.build`). It starts its own local HTTP server on a free
port, so nothing else needs to be running. The script exits non-zero if a page
logs a console error, if anything logs `unmocked <command>`, if a request fails,
if `app/src/api.ts` calls a command `mock.js` lacks, or if a PNG is not
1920x1080.

The browser is the first Edge or Chrome found in the usual install paths. Set
`OPENMODEL_BROWSER` to an executable to override. Edge needs
`--edge-skip-compat-layer-relaunch`, which the script passes: without it
`msedge.exe` relaunches itself and exits at once, and puppeteer reports
"Failed to launch the browser process".

## The mock

- `mock.js` defines `window.__TAURI_INTERNALS__` (so the app takes its desktop
  code path) with one handler per Tauri command, plus the event plumbing
  (`plugin:event|listen`) used for server logs, chat streams and progress.
- `fixtures/hw.json` is the fake machine (including the RAM speed and bus width
  and the model disk's bus, PCIe link and read speed); `fixtures/data.json`
  holds settings, installed models, engines, image models and search results.
- The catalog is read from `crates/core/data/catalog.json` at run time; the mock
  computes the per-quant fit and speed estimates for the fake machine.

When you add a Tauri command to `app/src/api.ts`, add an entry to `handlers` in
`mock.js` (and fixture data if it returns any). Otherwise the capture fails with
a message naming the command.
