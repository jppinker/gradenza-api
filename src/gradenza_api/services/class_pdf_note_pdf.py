"""
Class PDF Notes → HTML renderer for backend PDF export.

The frontend renders notes with ReactMarkdown + KaTeX + a canvas graph component.
For server-side PDF export, we:
- render Markdown to HTML on the backend
- use KaTeX auto-render in the browser to typeset LaTeX
- re-implement the small, safe canvas graph renderer in vanilla JS
"""

from __future__ import annotations

import asyncio
import html
import json
import re
from typing import Any
from urllib.parse import unquote, urlparse

from gradenza_api.settings import settings


_KATEX_VERSION = "0.16.45"
_MD = None


def _slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def build_export_filename(title: str) -> str:
    slug = _slugify(title) or "class-pdf-note"
    return f"{slug}.pdf"


def _normalize_math_delimiters(text: str) -> str:
    # Convert \( ... \) -> $...$ and \[ ... \] -> $$...$$ (same as frontend).
    if not text:
        return ""
    text = re.sub(r"\\\(([\s\S]*?)\\\)", lambda m: f"${m.group(1).strip()}$", text)
    text = re.sub(r"\\\[([\s\S]*?)\\\]", lambda m: f"$${m.group(1).strip()}$$", text)
    return text


def _normalize_inline_html(text: str) -> str:
    """
    Best-effort normalization for a small subset of inline HTML that may appear in
    saved markdown content.

    We intentionally do not allow arbitrary raw HTML in the renderer (escape=True),
    but we also don't want common tags like <br> to show up as literal text.
    """
    if not text:
        return ""

    # Line breaks
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)

    # Bold / italic (strip attributes if present)
    text = re.sub(
        r"(?is)<\s*(strong|b)(?:\s+[^>]*)?>(.*?)<\s*/\s*\1\s*>",
        r"**\2**",
        text,
    )
    text = re.sub(
        r"(?is)<\s*(em|i)(?:\s+[^>]*)?>(.*?)<\s*/\s*\1\s*>",
        r"*\2*",
        text,
    )

    # Sup/subscript: keep content, avoid rendering raw tags.
    text = re.sub(r"(?is)<\s*sup(?:\s+[^>]*)?>(.*?)<\s*/\s*sup\s*>", r"^\1", text)
    text = re.sub(r"(?is)</?\s*sub(?:\s+[^>]*)?>", "", text)

    return text


def _markdown_renderer():
    global _MD
    if _MD is not None:
        return _MD
    try:
        import mistune  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "mistune is required to render markdown for PDF export. "
            "Add 'mistune' to requirements.txt and reinstall."
        ) from exc

    # escape=True ensures any raw HTML in markdown is escaped (avoid XSS in headless browser).
    _MD = mistune.create_markdown(escape=True)
    return _MD


def _render_markdown(md: str) -> str:
    md = _normalize_inline_html(md or "")
    md = _normalize_math_delimiters(md)
    render = _markdown_renderer()
    return render(md)


def _render_markdown_inline(md: str) -> str:
    rendered = _render_markdown(md)
    # Strip the common "<p>...</p>" wrapper for inline contexts (objectives/checklist items).
    rendered = rendered.strip()
    if rendered.startswith("<p>") and rendered.endswith("</p>"):
        rendered = rendered[3:-4].strip()
    return rendered


def _try_parse_supabase_storage_object_url(url: str) -> tuple[str, str] | None:
    """
    Parse a Supabase Storage object URL (public/authenticated/signed) and return (bucket, object_path).

    Supported URL formats:
      - {SUPABASE_URL}/storage/v1/object/public/{bucket}/{path...}
      - {SUPABASE_URL}/storage/v1/object/authenticated/{bucket}/{path...}
      - {SUPABASE_URL}/storage/v1/object/sign/{bucket}/{path...}?token=...
    """
    url = (url or "").strip()
    if not url or url.startswith("data:"):
        return None
    try:
        base = urlparse(settings.supabase_url)
        parsed = urlparse(url)
        if parsed.scheme != base.scheme or parsed.netloc != base.netloc:
            return None
        parts = [p for p in (parsed.path or "").split("/") if p]
        if len(parts) < 6:
            return None
        if parts[0] != "storage" or parts[1] != "v1" or parts[2] != "object":
            return None
        if parts[3] not in ("public", "authenticated", "sign"):
            return None
        bucket = parts[4]
        object_path = "/".join(parts[5:])
        if not bucket or not object_path:
            return None
        return bucket, unquote(object_path)
    except Exception:
        return None


async def ensure_note_images_accessible(*, svc: Any, note_content: dict[str, Any]) -> dict[str, Any]:
    """
    Best-effort: if note images point at this project's Supabase Storage, replace their URLs with
    fresh signed URLs so the Playwright renderer can always fetch them (even if the bucket is private).
    """
    if not isinstance(note_content, dict):
        return note_content

    sections = note_content.get("sections")
    if not isinstance(sections, list) or not sections:
        return note_content

    cache: dict[str, str] = {}

    async def _maybe_sign(url: str) -> str:
        url = (url or "").strip()
        if not url:
            return url
        cached = cache.get(url)
        if cached is not None:
            return cached
        parsed = _try_parse_supabase_storage_object_url(url)
        if not parsed:
            cache[url] = url
            return url
        bucket, object_path = parsed

        def _sign() -> str:
            signed = svc.storage.from_(bucket).create_signed_url(path=object_path, expires_in=3600)
            return signed.get("signedURL") or signed.get("signed_url") or ""

        try:
            signed_url = await asyncio.to_thread(_sign)
        except Exception:
            signed_url = ""
        cache[url] = signed_url or url
        return cache[url]

    updated_sections: list[Any] = []
    changed = False
    for section in sections:
        if not isinstance(section, dict):
            updated_sections.append(section)
            continue
        images = section.get("sectionImages")
        if not isinstance(images, list) or not images:
            updated_sections.append(section)
            continue

        new_images: list[Any] = []
        images_changed = False
        for img in images:
            if not isinstance(img, dict):
                new_images.append(img)
                continue
            url = img.get("url")
            if isinstance(url, str) and url.strip():
                new_url = await _maybe_sign(url)
                if new_url != url:
                    img = dict(img)
                    img["url"] = new_url
                    images_changed = True
            new_images.append(img)

        if images_changed:
            new_section = dict(section)
            new_section["sectionImages"] = new_images
            updated_sections.append(new_section)
            changed = True
        else:
            updated_sections.append(section)

    if not changed:
        return note_content

    updated = dict(note_content)
    updated["sections"] = updated_sections
    return updated


def _section_images_html(section: dict[str, Any]) -> str:
    images = section.get("sectionImages") or []
    if not isinstance(images, list) or not images:
        return ""
    figures: list[str] = []
    for img in images:
        if not isinstance(img, dict):
            continue
        url = (img.get("url") or "").strip()
        if not url:
            continue
        caption = (img.get("caption") or "").strip()
        alt = (img.get("alt") or "").strip() or caption
        figures.append(
            "\n".join(
                [
                    '<figure class="sectionImageFigure">',
                    f'  <img class="sectionImageImg" src="{html.escape(url)}" alt="{html.escape(alt)}" />',
                    f'  <figcaption class="sectionImageCaption">{html.escape(caption)}</figcaption>' if caption else "",
                    "</figure>",
                ]
            )
        )
    if not figures:
        return ""
    return f'<div class="sectionImageGrid">{"".join(figures)}</div>'


def _graph_html(section: dict[str, Any]) -> str:
    exprs = section.get("graphExpressions") or []
    if not isinstance(exprs, list):
        return ""
    exprs = [e for e in exprs if isinstance(e, str) and e.strip()]
    if not exprs:
        return ""
    exprs_json = html.escape(json.dumps(exprs[:3], ensure_ascii=False))
    caption = section.get("graphCaption")
    caption_html = ""
    if isinstance(caption, str) and caption.strip():
        caption_html = f'<figcaption class="graphCaption">{html.escape(caption.strip())}</figcaption>'
    return (
        "\n".join(
            [
                '<figure class="graphFigure">',
                f'  <canvas class="graphCanvas" width="320" height="180" data-expressions="{exprs_json}"></canvas>',
                f"  {caption_html}" if caption_html else "",
                "</figure>",
            ]
        )
    )


def build_class_pdf_note_html(note_content: dict[str, Any]) -> str:
    """
    Build a standalone HTML document for a Class PDF note.

    The output includes:
    - Minimal document styling
    - KaTeX auto-render for math
    - JS renderer for graphs (canvas)
    - A readiness flag (`window.__PDF_READY`) for Playwright
    """
    title = (note_content.get("title") or "").strip() if isinstance(note_content, dict) else ""
    subtitle = (note_content.get("subtitle") or "").strip() if isinstance(note_content, dict) else ""

    overview_md = note_content.get("overviewMarkdown") if isinstance(note_content, dict) else ""
    objectives = note_content.get("learningObjectives") if isinstance(note_content, dict) else []
    sections = note_content.get("sections") if isinstance(note_content, dict) else []
    checklist = note_content.get("practiceChecklist") if isinstance(note_content, dict) else []
    summary_md = note_content.get("studentSummaryMarkdown") if isinstance(note_content, dict) else ""

    overview_html = _render_markdown(str(overview_md)) if isinstance(overview_md, str) and overview_md.strip() else ""

    objectives_html = ""
    if isinstance(objectives, list) and objectives:
        items = [
            f'<li class="objectiveItem">{_render_markdown_inline(obj)}</li>'
            for obj in objectives
            if isinstance(obj, str) and obj.strip()
        ]
        if items:
            objectives_html = f'<ul class="objectiveList">{"".join(items)}</ul>'

    sections_html_parts: list[str] = []
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            sec_title = (section.get("title") or "").strip()
            content_md = section.get("contentMarkdown") or ""
            worked_md = section.get("workedExampleMarkdown")

            content_html = _render_markdown(str(content_md)) if isinstance(content_md, str) and content_md.strip() else ""
            worked_html = _render_markdown(str(worked_md)) if isinstance(worked_md, str) and worked_md.strip() else ""

            images_html = _section_images_html(section)
            graph_html = _graph_html(section)

            worked_block = ""
            if worked_html:
                worked_block = (
                    "\n".join(
                        [
                            '<div class="workedExample">',
                            '<div class="workedExampleLabel">Worked example</div>',
                            f'<div class="prose">{worked_html}</div>',
                            "</div>",
                        ]
                    )
                )

            sections_html_parts.append(
                "\n".join(
                    [
                        '<section class="docSection">',
                        f'  <h2 class="docSectionTitle">{html.escape(sec_title)}</h2>' if sec_title else "",
                        f'  <div class="prose">{content_html}</div>' if content_html else "",
                        f"  {graph_html}" if graph_html else "",
                        f"  {images_html}" if images_html else "",
                        f"  {worked_block}" if worked_block else "",
                        "</section>",
                    ]
                )
            )

    checklist_html = ""
    if isinstance(checklist, list) and checklist:
        items = [
            "\n".join(
                [
                    '<li class="checklistItem">',
                    '  <span class="checklistBox" aria-hidden="true">☐</span>',
                    f'  <span class="checklistText">{_render_markdown_inline(item)}</span>',
                    "</li>",
                ]
            )
            for item in checklist
            if isinstance(item, str) and item.strip()
        ]
        if items:
            checklist_html = f'<ul class="checklist">{"".join(items)}</ul>'

    summary_html = _render_markdown(str(summary_md)) if isinstance(summary_md, str) and summary_md.strip() else ""

    def section_block(title_text: str, inner_html: str) -> str:
        if not inner_html:
            return ""
        return (
            "\n".join(
                [
                    '<section class="docSection">',
                    f'  <h2 class="docSectionTitle">{html.escape(title_text)}</h2>',
                    f'  <div class="prose">{inner_html}</div>',
                    "</section>",
                ]
            )
        )

    # KaTeX auto-render + graph renderer + readiness flag.
    js = r"""
(function () {
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  var warnings = [];
  window.__PDF_WARNINGS = warnings;

  function waitForPredicate(pred, timeoutMs) {
    var start = Date.now();
    return new Promise(function (resolve) {
      (function tick() {
        try {
          if (pred()) return resolve(true);
        } catch (_) {
          // ignore predicate errors
        }
        if (Date.now() - start >= timeoutMs) return resolve(false);
        setTimeout(tick, 50);
      })();
    });
  }
  function waitForFonts(timeoutMs) {
    try {
      if (!document.fonts || !document.fonts.ready) return Promise.resolve();
      return Promise.race([document.fonts.ready, sleep(timeoutMs)]);
    } catch (_) {
      return Promise.resolve();
    }
  }

  function waitForImages(timeoutMs) {
    var imgs = Array.prototype.slice.call(document.images || []);
    if (imgs.length === 0) return Promise.resolve();
    var done = false;
    var timer = setTimeout(function () { done = true; }, timeoutMs);

    return Promise.all(imgs.map(function (img) {
      if (img.complete) return Promise.resolve();
      return new Promise(function (resolve) {
        var finish = function () { resolve(); };
        img.addEventListener('load', finish, { once: true });
        img.addEventListener('error', finish, { once: true });
      });
    })).finally(function () {
      clearTimeout(timer);
      done = true;
    });
  }

  function replaceBrokenSectionImages() {
    try {
      var imgs = Array.prototype.slice.call(document.querySelectorAll('img.sectionImageImg'));
      if (imgs.length === 0) return;

      var broken = imgs.filter(function (img) { return img.complete && img.naturalWidth === 0; });
      if (broken.length === 0) return;

      var urls = broken
        .map(function (img) { return img.currentSrc || img.src; })
        .filter(Boolean);
      if (urls.length) window.__PDF_BROKEN_IMAGES = urls;

      warnings.push('Some images failed to load and may be missing in the PDF');

      broken.forEach(function (img) {
        try {
          var alt = (img.getAttribute('alt') || '').trim();
          var ph = document.createElement('div');
          ph.className = 'sectionImageBroken';
          ph.textContent = alt ? ('Image failed to load: ' + alt) : 'Image failed to load';
          img.replaceWith(ph);
        } catch (_) {
          // ignore
        }
      });
    } catch (_) {
      // ignore
    }
  }

  // ── Graph renderer (ported from frontend FunctionGraph.tsx) ────────────────

  var ALLOWED_FUNCS = {
    sin: Math.sin,
    cos: Math.cos,
    tan: Math.tan,
    sqrt: Math.sqrt,
    abs: Math.abs,
    exp: Math.exp,
    ln: Math.log,
    log: Math.log10
  };

  function ExprParser(expr) {
    this.pos = 0;
    this.src = String(expr || '').replace(/\s+/g, '').replace(/\*\*/g, '^');
  }
  ExprParser.prototype.peek = function () { return this.src[this.pos] || ''; };
  ExprParser.prototype.consume = function () { return this.src[this.pos++] || ''; };
  ExprParser.prototype.evaluate = function (x) { this.pos = 0; return this.parseExpr(x); };
  ExprParser.prototype.parseExpr = function (x) {
    var left = this.parseTerm(x);
    while (this.pos < this.src.length) {
      var ch = this.peek();
      if (ch === '+') { this.consume(); left += this.parseTerm(x); }
      else if (ch === '-') { this.consume(); left -= this.parseTerm(x); }
      else break;
    }
    return left;
  };
  ExprParser.prototype.parseTerm = function (x) {
    var left = this.parsePower(x);
    while (this.pos < this.src.length) {
      var ch = this.peek();
      if (ch === '*') { this.consume(); left *= this.parsePower(x); }
      else if (ch === '/') { this.consume(); var d = this.parsePower(x); left = d === 0 ? NaN : left / d; }
      else break;
    }
    return left;
  };
  ExprParser.prototype.parsePower = function (x) {
    var base = this.parseUnary(x);
    if (this.peek() === '^') {
      this.consume();
      var exp = this.parseUnary(x);
      return Math.pow(base, exp);
    }
    return base;
  };
  ExprParser.prototype.parseUnary = function (x) {
    if (this.peek() === '-') { this.consume(); return -this.parseAtom(x); }
    if (this.peek() === '+') { this.consume(); return this.parseAtom(x); }
    return this.parseAtom(x);
  };
  ExprParser.prototype.parseAtom = function (x) {
    if (/[0-9.]/.test(this.peek())) {
      var num = '';
      while (/[0-9.]/.test(this.peek())) num += this.consume();
      return parseFloat(num);
    }
    if (this.peek() === '(') {
      this.consume();
      var val = this.parseExpr(x);
      if (this.peek() === ')') this.consume();
      return val;
    }
    if (/[a-zA-Z]/.test(this.peek())) {
      var name = '';
      while (/[a-zA-Z]/.test(this.peek())) name += this.consume();
      if (name === 'x') return x;
      if (name === 'pi') return Math.PI;
      if (name === 'e') return Math.E;
      var fn = ALLOWED_FUNCS[name];
      if (fn && this.peek() === '(') {
        this.consume();
        var arg = this.parseExpr(x);
        if (this.peek() === ')') this.consume();
        return fn(arg);
      }
      throw new Error('Unknown identifier: ' + name);
    }
    throw new Error('Unexpected char: ' + this.peek());
  };

  function safeEvaluate(expr, x) {
    try {
      var parser = new ExprParser(expr);
      var result = parser.evaluate(x);
      return isFinite(result) ? result : null;
    } catch (_) {
      return null;
    }
  }

  function compileExpression(expr) {
    try {
      var parser = new ExprParser(expr);
      parser.evaluate(1);
      return function (x) { return safeEvaluate(expr, x); };
    } catch (_) {
      return null;
    }
  }

  var CURVE_COLORS = ['#4B36CC', '#E05B23', '#159947'];
  var W = 320, H = 180, PAD = 28;

  function renderGraphCanvas(canvas) {
    try {
      var raw = canvas.getAttribute('data-expressions') || '[]';
      var expressions = JSON.parse(raw);
      if (!Array.isArray(expressions)) return;
      var validFns = expressions.slice(0, 3)
        .map(function (expr) { return { expr: expr, fn: compileExpression(expr) }; })
        .filter(function (item) { return item.fn !== null; });
      if (validFns.length === 0) return;

      var ctx = canvas.getContext('2d');
      if (!ctx) return;

      var plotW = W - 2 * PAD;
      var plotH = H - 2 * PAD;

      var STEPS = 200;
      var xMin = -6, xMax = 6;
      var allY = [];
      for (var i = 0; i <= STEPS; i++) {
        var x = xMin + (i / STEPS) * (xMax - xMin);
        for (var j = 0; j < validFns.length; j++) {
          var y = validFns[j].fn(x);
          if (y !== null) allY.push(y);
        }
      }
      var yMin = Math.min.apply(Math, allY);
      var yMax = Math.max.apply(Math, allY);
      if (!isFinite(yMin) || !isFinite(yMax) || yMin === yMax) { yMin = -5; yMax = 5; }
      var yPad = (yMax - yMin) * 0.1;
      yMin -= yPad; yMax += yPad;

      function toCanvasX(x) { return PAD + ((x - xMin) / (xMax - xMin)) * plotW; }
      function toCanvasY(y) { return PAD + plotH - ((y - yMin) / (yMax - yMin)) * plotH; }

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#FAFAF8';
      ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = '#E8E4DE';
      ctx.lineWidth = 0.5;
      for (var gx = Math.ceil(xMin); gx <= Math.floor(xMax); gx++) {
        var cx = toCanvasX(gx);
        ctx.beginPath(); ctx.moveTo(cx, PAD); ctx.lineTo(cx, H - PAD); ctx.stroke();
      }
      var yStep = (yMax - yMin) / 5;
      for (var k = 0; k <= 5; k++) {
        var gy = yMin + k * yStep;
        var cy = toCanvasY(gy);
        ctx.beginPath(); ctx.moveTo(PAD, cy); ctx.lineTo(W - PAD, cy); ctx.stroke();
      }

      ctx.strokeStyle = '#1A1A1A';
      ctx.lineWidth = 1;
      if (yMin <= 0 && yMax >= 0) {
        var axY = toCanvasY(0);
        ctx.beginPath(); ctx.moveTo(PAD, axY); ctx.lineTo(W - PAD, axY); ctx.stroke();
      }
      if (xMin <= 0 && xMax >= 0) {
        var axX = toCanvasX(0);
        ctx.beginPath(); ctx.moveTo(axX, PAD); ctx.lineTo(axX, H - PAD); ctx.stroke();
      }

      for (var fi = 0; fi < validFns.length; fi++) {
        ctx.strokeStyle = CURVE_COLORS[fi % CURVE_COLORS.length];
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        var penDown = false;
        for (var si = 0; si <= STEPS; si++) {
          var sx = xMin + (si / STEPS) * (xMax - xMin);
          var sy = validFns[fi].fn(sx);
          if (sy === null) { penDown = false; continue; }
          var px = toCanvasX(sx);
          var py = toCanvasY(sy);
          if (!penDown) { ctx.moveTo(px, py); penDown = true; }
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
    } catch (_) {
      // ignore graph errors
    }
  }

	  function renderAllGraphs() {
	    var canvases = Array.prototype.slice.call(document.querySelectorAll('canvas.graphCanvas'));
	    canvases.forEach(renderGraphCanvas);
	  }

	  function decodeMathEntities(math) {
	    if (!math || typeof math !== 'string') return math;
	    return math
	      .replace(/&lt;/g, '<')
	      .replace(/&gt;/g, '>')
	      .replace(/&le;/g, '\\\\le')
	      .replace(/&ge;/g, '\\\\ge')
	      .replace(/&amp;/g, '&')
	      .replace(/&nbsp;/g, ' ');
	  }

	  function renderMath() {
	    return waitForPredicate(function () { return typeof window.renderMathInElement === 'function'; }, 8000)
	      .then(function (ok) {
	        if (!ok) {
	          warnings.push('KaTeX did not load; math may be unrendered');
          return;
        }
        try {
	          window.renderMathInElement(document.body, {
	            delimiters: [
	              { left: '$$', right: '$$', display: true },
	              { left: '$', right: '$', display: false },
	              { left: '\\\\(', right: '\\\\)', display: false },
	              { left: '\\\\[', right: '\\\\]', display: true }
	            ],
	            preProcess: decodeMathEntities,
	            throwOnError: false,
	            strict: false
	          });
	        } catch (_) {
	          // ignore KaTeX render errors
	        }
      });
  }

  function domReady() {
    if (document.readyState === 'loading') {
      return new Promise(function (resolve) {
        document.addEventListener('DOMContentLoaded', function () { resolve(); }, { once: true });
      });
    }
    return Promise.resolve();
  }

  domReady()
    .then(function () { return renderMath(); })
    .then(function () {
      try { renderAllGraphs(); } catch (_) { /* ignore */ }
      return Promise.all([
        waitForFonts(2500),
        waitForImages(10_000)
      ]).then(function () {
        try { replaceBrokenSectionImages(); } catch (_) { /* ignore */ }
      });
    })
    .catch(function () {
      // ignore
    })
    .finally(function () {
      window.__PDF_READY = true;
    });
})();
"""

    css = f"""
html, body {{
  margin: 0;
  padding: 0;
  background: #ffffff;
  color: #1A1A1A;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.7;
}}
.document {{
  max-width: 720px;
  margin: 0 auto;
  padding: 32px 36px;
}}
.docHeader {{
  margin-bottom: 28px;
  padding-bottom: 20px;
  border-bottom: 2px solid #E8E0D6;
}}
.docTitle {{
  font-size: 22px;
  font-weight: 700;
  margin: 0 0 6px;
  line-height: 1.3;
}}
.docSubtitle {{
  font-size: 14px;
  color: #767676;
  margin: 0;
  font-style: italic;
}}
.docSection {{
  margin-bottom: 28px;
}}
.docSectionTitle {{
  font-size: 15px;
  font-weight: 700;
  color: #4B36CC;
  margin: 0 0 10px;
  padding-bottom: 4px;
  border-bottom: 1px solid #EDE9FF;
}}
.prose p {{ margin: 0 0 10px; }}
.prose ul, .prose ol {{ padding-left: 1.4em; margin: 0 0 10px; }}
.prose li {{ margin-bottom: 4px; }}
.objectiveList {{ padding-left: 1.4em; margin: 0; }}
.objectiveItem {{ margin-bottom: 6px; }}
.checklist {{ list-style: none; padding: 0; margin: 0; }}
.checklistItem {{ display: flex; gap: 10px; margin-bottom: 8px; align-items: flex-start; }}
.checklistBox {{ display: inline-block; width: 18px; height: 18px; border: 1px solid #CFC7BD; border-radius: 4px; text-align: center; line-height: 18px; font-size: 12px; color: #6B6B6B; margin-top: 2px; }}
.workedExample {{
  margin-top: 14px;
  padding: 12px 14px;
  border-left: 4px solid #E8E0D6;
  background: #FAFAF8;
  border-radius: 8px;
}}
.workedExampleLabel {{
  font-weight: 700;
  margin-bottom: 6px;
}}
.sectionImageGrid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
  margin-top: 12px;
}}
.sectionImageFigure {{
  margin: 0;
}}
.sectionImageImg {{
  width: 100%;
  height: auto;
  display: block;
  border: 1px solid #E8E0D6;
  border-radius: 10px;
  background: #fff;
}}
.sectionImageBroken {{
  width: 100%;
  min-height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 14px;
  text-align: center;
  border: 1px dashed #E8E0D6;
  border-radius: 10px;
  background: #FAFAF8;
  color: #767676;
  font-size: 12px;
}}
.sectionImageCaption {{
  font-size: 12px;
  color: #767676;
  margin-top: 6px;
}}
.graphFigure {{
  margin: 12px 0 0;
}}
.graphCanvas {{
  border: 1px solid #E8E0D6;
  border-radius: 10px;
  background: #FAFAF8;
}}
.graphCaption {{
  font-size: 12px;
  color: #767676;
  margin-top: 6px;
}}
@page {{
  size: A4;
}}
"""

    title_esc = html.escape(title or "Class PDF Note")
    subtitle_esc = html.escape(subtitle)

    html_parts: list[str] = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8" />',
        '<meta name="viewport" content="width=device-width, initial-scale=1" />',
        f"<title>{title_esc}</title>",
        f'<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@{_KATEX_VERSION}/dist/katex.min.css" />',
        f'<script defer src="https://cdn.jsdelivr.net/npm/katex@{_KATEX_VERSION}/dist/katex.min.js"></script>',
        f'<script defer src="https://cdn.jsdelivr.net/npm/katex@{_KATEX_VERSION}/dist/contrib/auto-render.min.js"></script>',
        "<style>",
        css,
        "</style>",
        "</head>",
        "<body>",
        '<article class="document">',
        '<header class="docHeader">',
        f'  <h1 class="docTitle">{title_esc}</h1>',
        f'  <p class="docSubtitle">{subtitle_esc}</p>' if subtitle_esc else "",
        "</header>",
        section_block("Overview", overview_html),
        (
            "\n".join(
                [
                    '<section class="docSection">',
                    '  <h2 class="docSectionTitle">Learning objectives</h2>',
                    f"  {objectives_html}",
                    "</section>",
                ]
            )
            if objectives_html
            else ""
        ),
        "".join(sections_html_parts),
        (
            "\n".join(
                [
                    '<section class="docSection">',
                    '  <h2 class="docSectionTitle">Practice checklist</h2>',
                    f"  {checklist_html}",
                    "</section>",
                ]
            )
            if checklist_html
            else ""
        ),
        section_block("Summary", summary_html),
        "</article>",
        "<script>",
        js,
        "</script>",
        "</body>",
        "</html>",
    ]

    return "\n".join(p for p in html_parts if p)
