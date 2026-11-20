import { Fragment, memo, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { openExternal } from "./bridge";
import { useI18n } from "./i18n";

/*
 * Small Markdown renderer for model output (untrusted). It builds React
 * elements only, so raw HTML is always escaped text; links are limited to
 * http/https and images are never loaded.
 */

type Align = "left" | "center" | "right" | undefined;
type Body =
  | { t: "p"; text: string }
  | { t: "h"; n: number; text: string }
  | { t: "hr" }
  | { t: "code"; lang: string; text: string }
  | { t: "quote"; kids: Block[] }
  | { t: "list"; ordered: boolean; start: number; items: Block[][] }
  | { t: "table"; head: string[]; align: Align[]; rows: string[][] };
type Block = Body & { src: string };

const MAX_DEPTH = 8;
const FENCE = /^ {0,3}(`{3,}|~{3,})\s*([^`\s]*)[^`]*$/;
const HEADING = /^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$/;
const HR = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
const MARKER = /^( *)([-*+]|\d{1,9}[.)])( +|$)/;
const SEPARATOR = /^ {0,3}\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$/;

const blank = (l: string) => l.trim() === "";
const indentOf = (l: string) => l.length - l.trimStart().length;

function startsBlock(l: string): boolean {
  if (FENCE.test(l) || HEADING.test(l) || HR.test(l) || /^ {0,3}>/.test(l)) return true;
  const m = MARKER.exec(l);
  return !!m && l.trim().length > m[2].length && (/^[-*+]/.test(m[2]) || /^1[.)]$/.test(m[2]));
}

function splitRow(line: string): string[] {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  const cells: string[] = [];
  let cur = "";
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === "\\" && s[i + 1] === "|") { cur += "|"; i++; }
    else if (c === "|") { cells.push(cur.trim()); cur = ""; }
    else cur += c;
  }
  if (cur.trim() !== "" || cells.length === 0) cells.push(cur.trim());
  return cells;
}

function parseBlocks(lines: string[], depth: number): Block[] {
  if (depth > MAX_DEPTH) return [{ t: "p", text: lines.join("\n"), src: lines.join("\n") }];
  const out: Block[] = [];
  let i = 0;
  const push = (b: Body, from: number) => out.push({ ...b, src: lines.slice(from, i).join("\n") });
  while (i < lines.length) {
    const line = lines[i];
    if (blank(line)) { i++; continue; }
    const from = i;

    const f = FENCE.exec(line);
    if (f) {
      const ch = f[1][0], len = f[1].length;
      const body: string[] = [];
      i++;
      // An unclosed fence (still streaming) runs to the end and renders the same.
      while (i < lines.length) {
        const c = lines[i].trim();
        if (c.length >= len && [...c].every((x) => x === ch)) break;
        body.push(lines[i]);
        i++;
      }
      if (i < lines.length) i++;
      push({ t: "code", lang: f[2].slice(0, 30), text: body.join("\n") }, from);
      continue;
    }

    const h = HEADING.exec(line);
    if (h) { i++; push({ t: "h", n: h[1].length, text: h[2] }, from); continue; }
    if (HR.test(line)) { i++; push({ t: "hr" }, from); continue; }

    if (/^ {0,3}>/.test(line)) {
      const inner: string[] = [];
      while (i < lines.length && !blank(lines[i]) && (/^ {0,3}>/.test(lines[i]) || inner.length > 0 && !startsBlock(lines[i]))) {
        inner.push(lines[i].replace(/^ {0,3}> ?/, ""));
        i++;
      }
      push({ t: "quote", kids: parseBlocks(inner, depth + 1) }, from);
      continue;
    }

    if (line.includes("|") && i + 1 < lines.length && lines[i + 1].includes("-") && SEPARATOR.test(lines[i + 1])) {
      const head = splitRow(line);
      const seps = splitRow(lines[i + 1]);
      if (head.length === seps.length) {
        const align = seps.map((s): Align => (s.startsWith(":") ? (s.endsWith(":") ? "center" : "left") : s.endsWith(":") ? "right" : undefined));
        i += 2;
        const rows: string[][] = [];
        while (i < lines.length && !blank(lines[i]) && lines[i].includes("|")) rows.push(splitRow(lines[i++]));
        push({ t: "table", head, align, rows }, from);
        continue;
      }
    }

    const m = MARKER.exec(line);
    if (m && line.trim().length > m[2].length) {
      const base = m[1].length;
      const ordered = /\d/.test(m[2]);
      const start = ordered ? parseInt(m[2], 10) : 1;
      const items: Block[][] = [];
      while (i < lines.length) {
        const im = MARKER.exec(lines[i]);
        if (!im || im[1].length !== base || /\d/.test(im[2]) !== ordered) break;
        const offset = im[0].length - (im[3].length > 4 ? im[3].length - 1 : 0);
        const body: string[] = [lines[i].slice(offset)];
        i++;
        while (i < lines.length) {
          const l = lines[i];
          if (blank(l)) {
            let j = i + 1;
            while (j < lines.length && blank(lines[j])) j++;
            if (j >= lines.length) { i = j; break; }
            const nm = MARKER.exec(lines[j]);
            const sibling = !!nm && nm[1].length === base && /\d/.test(nm[2]) === ordered;
            if (indentOf(lines[j]) >= offset || sibling) {
              for (; i < j; i++) body.push("");
              if (sibling) break;
              continue;
            }
            break;
          }
          const ind = indentOf(l);
          if (ind >= offset) { body.push(l.slice(offset)); i++; continue; }
          const sm = MARKER.exec(l);
          if (sm && sm[1].length <= base + 1) break;
          if (ind <= base && startsBlock(l)) break;
          if (body[body.length - 1] === "") break;
          body.push(l.trim()); // lazy continuation of the item's paragraph
          i++;
        }
        while (body.length && body[body.length - 1] === "") body.pop();
        items.push(parseBlocks(body, depth + 1));
      }
      push({ t: "list", ordered, start, items }, from);
      continue;
    }

    const para: string[] = [line];
    i++;
    while (i < lines.length && !blank(lines[i]) && !startsBlock(lines[i])) para.push(lines[i++]);
    push({ t: "p", text: para.map((l) => l.trim()).join("\n") }, from);
  }
  return out;
}

// Inline

const PUNCT = /[!-/:-@[-`{-~]/;
const isAlnum = (c: string | undefined) => !!c && /[\p{L}\p{N}]/u.test(c);

function safeUrl(raw: string): string | null {
  const u = raw.trim();
  if (!u || /[\s\u0000-\u001f]/.test(u)) return null;
  try {
    const p = new URL(u);
    return p.protocol === "http:" || p.protocol === "https:" ? p.href : null;
  } catch {
    return null;
  }
}

function Ext({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a
      href={href}
      title={href}
      rel="noopener noreferrer"
      onClick={(e) => { e.preventDefault(); openExternal(href).catch(() => {}); }}
      onAuxClick={(e) => e.preventDefault()}
    >
      {children}
    </a>
  );
}

/** Index of the `]` closing the `[` at `open`, or -1. */
function closeBracket(s: string, open: number): number {
  let d = 0;
  for (let i = open; i < s.length; i++) {
    const c = s[i];
    if (c === "\\") i++;
    else if (c === "[") d++;
    else if (c === "]" && --d === 0) return i;
  }
  return -1;
}

/** Parse `(url "title")` starting at `s[k] === "("`. */
function linkTail(s: string, k: number): { url: string; end: number } | null {
  let i = k + 1;
  while (s[i] === " ") i++;
  let url = "";
  if (s[i] === "<") {
    const e = s.indexOf(">", i);
    if (e < 0) return null;
    url = s.slice(i + 1, e);
    i = e + 1;
  } else {
    let d = 0;
    const st = i;
    for (; i < s.length; i++) {
      const c = s[i];
      if (c === "\\") i++;
      else if (c === "(") d++;
      else if (c === ")") { if (d === 0) break; d--; }
      else if (c === " " || c === "\n") break;
    }
    url = s.slice(st, i);
  }
  while (s[i] === " ") i++;
  if (s[i] === '"' || s[i] === "'") {
    const e = s.indexOf(s[i], i + 1);
    if (e < 0) return null;
    i = e + 1;
    while (s[i] === " ") i++;
  }
  return s[i] === ")" ? { url, end: i + 1 } : null;
}

/** End index (exclusive) of the closing run for an emphasis opener, or -1. */
function closer(s: string, from: number, d: string): number {
  const c = d[0];
  for (let j = s.indexOf(d, from); j >= 0; j = s.indexOf(d, j + 1)) {
    if (j === from || /\s/.test(s[j - 1])) continue;
    if (d.length === 1 && (s[j + 1] === c || s[j - 1] === c)) continue;
    if (c === "_" && isAlnum(s[j + d.length])) continue;
    return j;
  }
  return -1;
}

function inline(s: string, noLink = false, depth = 0): ReactNode[] {
  const out: ReactNode[] = [];
  let buf = "";
  let key = 0;
  const flush = () => { if (buf) { out.push(buf); buf = ""; } };
  const add = (n: ReactNode) => { flush(); out.push(<Fragment key={key++}>{n}</Fragment>); };
  const rec = (x: string, nl = noLink) => (depth >= 6 ? [x] : inline(x, nl, depth + 1));

  let i = 0;
  while (i < s.length) {
    const c = s[i];

    if (c === "\\" && i + 1 < s.length && PUNCT.test(s[i + 1])) { buf += s[i + 1]; i += 2; continue; }

    if (c === "`") {
      let n = 1;
      while (s[i + n] === "`") n++;
      const tick = "`".repeat(n);
      let e = s.indexOf(tick, i + n);
      while (e >= 0 && s[e + n] === "`") e = s.indexOf(tick, e + n + 1);
      if (e > 0) {
        let code = s.slice(i + n, e).replace(/\n/g, " ");
        if (code.length > 2 && code.startsWith(" ") && code.endsWith(" ") && code.trim()) code = code.slice(1, -1);
        add(<code>{code}</code>);
        i = e + n;
        continue;
      }
      buf += tick;
      i += n;
      continue;
    }

    const img = c === "!" && s[i + 1] === "[";
    if (c === "[" || img) {
      const open = img ? i + 1 : i;
      const close = closeBracket(s, open);
      if (close > 0 && s[close + 1] === "(") {
        const tail = linkTail(s, close + 1);
        const label = s.slice(open + 1, close);
        if (tail) {
          const href = safeUrl(tail.url);
          // Images are never fetched: shown as a link (or just the alt text).
          if (img) add(href ? <Ext href={href}>{label || href}</Ext> : label);
          else if (href && !noLink) add(<Ext href={href}>{rec(label, true)}</Ext>);
          else add(rec(label, noLink));
          i = tail.end;
          continue;
        }
      }
    }

    if (c === "<") {
      const m = /^<(https?:\/\/[^\s<>]+)>/.exec(s.slice(i, i + 2100));
      const href = m && !noLink ? safeUrl(m[1]) : null;
      if (m && href) { add(<Ext href={href}>{m[1]}</Ext>); i += m[0].length; continue; }
    }

    if (c === "h" && !noLink && !isAlnum(s[i - 1]) && (s.startsWith("http://", i) || s.startsWith("https://", i))) {
      const m = /^https?:\/\/[^\s<>]+/.exec(s.slice(i, i + 2100))!;
      let url = m[0];
      for (;;) {
        const last = url[url.length - 1];
        const unbalanced = last === ")" && url.split(")").length > url.split("(").length;
        if (/[.,;:!?'"*_\]]/.test(last) || unbalanced) url = url.slice(0, -1);
        else break;
      }
      const href = safeUrl(url);
      if (href) { add(<Ext href={href}>{url}</Ext>); i += url.length; continue; }
    }

    if (c === "*" || c === "_" || c === "~") {
      const tryDelim = (d: string, wrap: (n: ReactNode[]) => ReactNode): boolean => {
        if (!s.startsWith(d, i)) return false;
        const nx = s[i + d.length];
        if (!nx || /\s/.test(nx)) return false;
        if (c === "_" && isAlnum(s[i - 1])) return false;
        const e = closer(s, i + d.length, d);
        if (e < 0) return false;
        add(wrap(rec(s.slice(i + d.length, e))));
        i = e + d.length;
        return true;
      };
      if (c === "~") {
        if (tryDelim("~~", (n) => <del>{n}</del>)) continue;
      } else if (
        tryDelim(c.repeat(3), (n) => <strong><em>{n}</em></strong>) ||
        tryDelim(c.repeat(2), (n) => <strong>{n}</strong>) ||
        tryDelim(c, (n) => <em>{n}</em>)
      ) continue;
    }

    buf += c;
    i++;
  }
  flush();
  return out;
}

// Rendering

function CodeBlock({ lang, text }: { lang: string; text: string }) {
  const { t } = useI18n();
  const [done, setDone] = useState(false);
  const timer = useRef<number>();
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const copy = () => {
    // navigator.clipboard is missing on plain-http web pages; ignore then.
    navigator.clipboard?.writeText(text).then(() => {
      setDone(true);
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setDone(false), 1500);
    }).catch(() => {});
  };
  return (
    <div className="md-code">
      <div className="md-code-head">
        <span>{lang}</span>
        <button className="btn sm" onClick={copy}>{done ? "✓" : t("overview.copy")}</button>
      </div>
      <pre><code>{text}</code></pre>
    </div>
  );
}

function render(b: Block): ReactNode {
  switch (b.t) {
    case "p":
      return <p>{inline(b.text)}</p>;
    case "h": {
      const H = `h${Math.min(b.n + 1, 6)}` as "h2";
      return <H>{inline(b.text)}</H>;
    }
    case "hr":
      return <hr />;
    case "code":
      return <CodeBlock lang={b.lang} text={b.text} />;
    case "quote":
      return <blockquote><Blocks blocks={b.kids} /></blockquote>;
    case "list": {
      const L = b.ordered ? "ol" : "ul";
      return (
        <L start={b.ordered && b.start !== 1 ? b.start : undefined}>
          {b.items.map((it, i) => <li key={i}><Blocks blocks={it} /></li>)}
        </L>
      );
    }
    case "table": {
      const n = b.head.length;
      return (
        <div className="md-table">
          <table>
            <thead>
              <tr>{b.head.map((c, i) => <th key={i} style={{ textAlign: b.align[i] }}>{inline(c)}</th>)}</tr>
            </thead>
            <tbody>
              {b.rows.map((r, ri) => (
                <tr key={ri}>
                  {Array.from({ length: n }, (_, i) => <td key={i} style={{ textAlign: b.align[i] }}>{inline(r[i] ?? "")}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
  }
}

// Finished blocks keep the same source text, so only the block being streamed re-renders.
const BlockView = memo(({ b }: { b: Block }) => <>{render(b)}</>, (a, c) => a.b.src === c.b.src);

function Blocks({ blocks }: { blocks: Block[] }) {
  return <>{blocks.map((b, i) => <BlockView key={i} b={b} />)}</>;
}

export const Markdown = memo(function Markdown({ text }: { text: string }) {
  const blocks = useMemo(() => parseBlocks(text.split(/\r?\n/), 0), [text]);
  return <div className="md"><Blocks blocks={blocks} /></div>;
});
