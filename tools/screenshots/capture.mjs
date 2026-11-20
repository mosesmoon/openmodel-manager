#!/usr/bin/env node
// Captures the README screenshots: serves a built frontend plus mock.js over a
// local HTTP server, drives it with puppeteer-core and saves 1920x1080 PNGs.
//
//   node capture.mjs [frontend-build-dir] [output-dir]
//
// Defaults: ../../app/dist and ./out. OPENMODEL_BROWSER overrides the browser.
// Exits non-zero on any console error, "unmocked" warning, failed request,
// command missing from the mock, or PNG that is not 1920x1080.
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "../..");
const dist = path.resolve(process.argv[2] ?? path.join(repo, "app/dist"));
const out = path.resolve(process.argv[3] ?? path.join(here, "out"));
const W = 1920;
const H = 1080;

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8", ".svg": "image/svg+xml", ".woff2": "font/woff2", ".woff": "font/woff",
  ".ttf": "font/ttf", ".png": "image/png", ".ico": "image/x-icon",
};

// ---- Local server: the build, plus the mock under /__mock/ ----
const MOCK_FILES = {
  "/__mock/mock.js": path.join(here, "mock.js"),
  "/__mock/hw.json": path.join(here, "fixtures/hw.json"),
  "/__mock/data.json": path.join(here, "fixtures/data.json"),
  // The real catalog, read from the repo so it never goes stale.
  "/__mock/catalog.json": path.join(repo, "crates/core/data/catalog.json"),
};

function serve() {
  const indexPath = path.join(dist, "index.html");
  if (!fs.existsSync(indexPath)) throw new Error(`no index.html in ${dist}; build the frontend first (see README.md)`);
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
    let file = MOCK_FILES[url];
    if (!file) {
      file = path.join(dist, url === "/" ? "index.html" : url);
      if (!file.startsWith(dist + path.sep)) return void res.writeHead(403).end();
    }
    fs.readFile(file, (err, body) => {
      if (err) return void res.writeHead(404).end("not found");
      if (file === indexPath) {
        // The mock must be defined before the app's own script runs.
        body = Buffer.from(body.toString("utf8").replace("<head>", '<head><link rel="icon" href="data:,"><script src="/__mock/mock.js"></script>'));
      }
      res.writeHead(200, { "Content-Type": MIME[path.extname(file)] ?? "application/octet-stream", "Cache-Control": "no-store" });
      res.end(body);
    });
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

// ---- Browser: OPENMODEL_BROWSER, else the first Edge or Chrome that starts ----
function candidates() {
  const env = process.env.OPENMODEL_BROWSER;
  if (env) return [env];
  const e = process.env;
  if (process.platform === "win32") {
    const roots = [e["ProgramFiles(x86)"], e.ProgramFiles, e.LOCALAPPDATA].filter(Boolean);
    return roots.flatMap((r) => [`${r}\\Microsoft\\Edge\\Application\\msedge.exe`, `${r}\\Google\\Chrome\\Application\\chrome.exe`]);
  }
  if (process.platform === "darwin") {
    return ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"];
  }
  return ["/usr/bin/microsoft-edge", "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser"];
}

// The browser can still hold the profile for a moment after it closes.
function cleanup(dir) {
  try {
    fs.rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 300 });
  } catch {
    console.warn(`could not remove ${dir}`);
  }
}

async function launch() {
  const errors = [];
  for (const exe of candidates()) {
    if (!fs.existsSync(exe)) continue;
    const profile = fs.mkdtempSync(path.join(os.tmpdir(), "openmodel-shot-"));
    const args = ["--lang=zh-TW", "--hide-scrollbars", "--force-device-scale-factor=1", "--no-first-run", "--no-default-browser-check", `--window-size=${W},${H}`];
    // Edge's launcher relaunches itself under a compatibility layer and exits
    // with code 0, which puppeteer reports as "Failed to launch the browser
    // process". This switch skips the relaunch (harmless elsewhere).
    if (/msedge|edge/i.test(path.basename(exe))) args.push("--edge-skip-compat-layer-relaunch");
    try {
      const browser = await puppeteer.launch({ executablePath: exe, headless: true, userDataDir: profile, args, timeout: 30000, defaultViewport: { width: W, height: H, deviceScaleFactor: 1 } });
      console.log(`browser: ${exe} (${await browser.version()})`);
      return { browser, profile };
    } catch (err) {
      errors.push(`${exe}: ${String(err.message).split("\n")[0]}`);
      cleanup(profile);
    }
  }
  throw new Error("no usable browser; set OPENMODEL_BROWSER to an Edge or Chrome executable.\n" + (errors.join("\n") || "none of the usual install paths exist"));
}

// ---- Every command the frontend calls must have a mock entry ----
function apiCommands() {
  const src = fs.readFileSync(path.join(repo, "app/src/api.ts"), "utf8");
  return [...new Set([...src.matchAll(/invoke<[^>]*(?:<[^>]*>[^>]*)*>\(\s*"([a-z_]+)"/g)].map((m) => m[1]))];
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}

// ---- The pages, in README order ----
const NAV = { overview: 0, models: 1, run: 2, chat: 3, image: 4, video: 5, agent: 6, settings: 7 };
const PAGE_WAIT = 400;

async function main() {
  fs.mkdirSync(out, { recursive: true });
  const server = await serve();
  const base = `http://127.0.0.1:${server.address().port}/`;
  const { browser, profile } = await launch();
  const problems = [];
  let page;
  let step = "start";
  try {
    page = await browser.newPage();
    await page.setExtraHTTPHeaders({ "Accept-Language": "zh-TW,zh;q=0.9" });
    const flag = (msg) => problems.push(`[${step}] ${msg}`);
    page.on("console", (m) => {
      const text = m.text();
      if (m.type() === "error") flag(`console error: ${text}`);
      else if (m.type() === "warning" && text.startsWith("unmocked")) flag(text);
    });
    page.on("pageerror", (e) => flag(`page error: ${e.message}`));
    page.on("requestfailed", (r) => flag(`request failed: ${r.url()} (${r.failure()?.errorText})`));

    const click = (sel, nth = 0) => page.evaluate((s, n) => document.querySelectorAll(s)[n]?.click(), sel, nth);
    const nav = async (name) => {
      await click("nav.side button.nav", NAV[name]);
      await wait(PAGE_WAIT);
    };
    const tab = async (i) => {
      await click(".tabs .tab", i);
      await wait(PAGE_WAIT);
    };
    const type = async (sel, text) => {
      await page.click(sel);
      await page.type(sel, text);
    };
    const shot = async (name) => {
      step = name;
      await page.mouse.move(W - 4, H - 4); // keep hover effects out of the picture
      await wait(PAGE_WAIT);
      const file = path.join(out, `${name}.png`);
      await page.screenshot({ path: file, type: "png" });
      const [w, h] = pngSize(file);
      console.log(`${name}.png ${w}x${h}`);
      if (w !== W || h !== H) flag(`${name}.png is ${w}x${h}, expected ${W}x${H}`);
    };

    await page.goto(base, { waitUntil: "load" });
    await page.waitForFunction("window.__mock !== undefined");
    const known = await page.evaluate(() => window.__mock.commands);
    for (const c of apiCommands()) if (!known.includes(c)) flag(`api.ts calls "${c}" but mock.js has no entry for it`);

    // 01 overview: wait until the provider checks have answered too.
    step = "01-概覽";
    await page.waitForSelector(".grid .card");
    await page.waitForFunction("document.body.innerText.includes('ms')");
    await page.evaluate(() => window.__mock.seedLogs());
    await shot("01-概覽");

    // 02-05 models
    step = "02-模型-推薦";
    await nav("models");
    await page.waitForSelector(".hwcard");
    await page.waitForSelector(".grid .card .pill");
    await shot("02-模型-推薦");

    step = "03-模型-已安裝";
    await tab(2);
    await page.waitForSelector("table tbody tr");
    await shot("03-模型-已安裝");

    step = "04-模型-AI生圖影片";
    await tab(3);
    await page.waitForSelector(".grid .card select");
    await shot("04-模型-AI生圖影片");

    step = "05-模型-搜尋";
    await tab(4);
    await type('input[type="text"]', "qwen3.6 27b gguf");
    await page.keyboard.press("Enter");
    await page.waitForSelector(".card .row.spread b");
    await click(".card .row.spread");
    await page.waitForSelector(".card table tbody tr");
    await shot("05-模型-搜尋");

    // 06 run
    step = "06-執行";
    await nav("run");
    await page.waitForSelector(".stats");
    await page.waitForSelector("code.cmd");
    await shot("06-執行");

    // 07 chat
    step = "07-對話";
    await nav("chat");
    await type("textarea", "17 × 23 等於多少？");
    await page.keyboard.press("Enter");
    await page.waitForFunction("[...document.querySelectorAll('.msg.assistant')].some((m) => m.innerText.includes('391')) && document.querySelector('.chat .right')?.innerText.includes('tok/s')");
    await shot("07-對話");

    // 08 image: load the model, generate, then stop it again for the next page.
    step = "08-AI生成圖片";
    await nav("image");
    await page.waitForSelector(".two .card .row button.btn:not([disabled])");
    await click(".two .card .row button.btn");
    await page.waitForSelector(".two .card .small .dot.ready");
    await type(".two .card textarea", "一隻坐在窗邊的橘貓，水彩風格");
    await page.waitForFunction("!document.querySelector('.two .card button.btn.primary').disabled");
    await click(".two .card button.btn.primary");
    await page.waitForSelector(".gallery img");
    await shot("08-AI生成圖片");
    await page.evaluate(() => window.__TAURI_INTERNALS__.invoke("server_stop", { which: "sd" }));

    // 09 video
    step = "09-AI生成影片";
    await nav("video");
    await page.waitForSelector(".two .card select");
    await shot("09-AI生成影片");

    // 10 agent: stops at the approval request.
    step = "10-Agent";
    await nav("agent");
    await type("textarea", "幫我跑測試");
    await page.keyboard.press("Enter");
    await page.waitForSelector(".approval");
    await shot("10-Agent");

    // 11 settings
    step = "11-設定";
    await nav("settings");
    await page.waitForSelector(".card h2");
    await page.waitForFunction("document.body.innerText.includes('0.1.55')");
    await shot("11-設定");
  } catch (err) {
    problems.push(`[${step}] ${err.message}`);
    // Leave a picture of where it stopped.
    await page?.screenshot({ path: path.join(out, "_failed.png") }).catch(() => {});
  } finally {
    await browser.close().catch(() => {});
    server.close();
    cleanup(profile);
  }

  if (problems.length) {
    console.error(`\n${problems.length} problem(s):\n` + problems.map((p) => "  " + p).join("\n"));
    process.exit(1);
  }
  console.log(`\nok: 11 screenshots in ${out}`);
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
