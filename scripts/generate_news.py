#!/usr/bin/env python3
"""Agente de noticias de IA.

Junta noticias de feeds RSS (sitios + canales de YouTube), le pide a un LLM
(Gemini primero, Groq de respaldo) que elija las 3-5 mas importantes (con al
menos un video/episodio si hay), escribe docs/news.json (contrato news_format.md)
y manda el digest completo (con detalle y links) por Telegram.
"""
import calendar
import html
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import feedparser
import requests

# ---------------------------------------------------------------------------
# Fuentes
# kind: "article" | "video"
# ai_filter: el feed es generalista, solo se queda con entradas que hablen de IA
# skip_categories: categorias de WordPress a descartar
# desc_only: usar solo la descripcion (el contenido completo es de paywall/largo)
# ---------------------------------------------------------------------------
FEEDS = [
    {"name": "arXiv cs.AI", "url": "http://export.arxiv.org/rss/cs.AI", "kind": "article", "max": 6},
    {"name": "Hacker News", "url": "https://hnrss.org/best", "kind": "article", "ai_filter": True},
    {"name": "OpenAI", "url": "https://openai.com/news/rss.xml", "kind": "article"},
    {"name": "TechCrunch IA", "url": "https://techcrunch.com/category/artificial-intelligence/feed/", "kind": "article"},
    {"name": "The Verge IA", "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "kind": "article"},
    {"name": "Google News (Anthropic/DeepMind)",
     "url": "https://news.google.com/rss/search?q=Anthropic+OR+%22Google+DeepMind%22+when:2d&hl=en-US&gl=US&ceid=US:en",
     "kind": "article"},
    {"name": "Academia Artificial", "url": "https://academiartificial.com/feed/", "kind": "article",
     "skip_categories": ["actualidad ia"]},
    {"name": "La Prensa IA", "url": "https://laprensaia.com/feed/", "kind": "article", "desc_only": True},
    {"name": "Xataka", "url": "https://www.xataka.com/index.xml", "kind": "article", "ai_filter": True},
    {"name": "Infobae Tecno", "url": "https://www.infobae.com/arc/outboundfeeds/rss/category/tecno/",
     "kind": "article", "ai_filter": True},
    {"name": "Wired ES Negocios", "url": "https://es.wired.com/feed/negocios/rss", "kind": "article", "ai_filter": True},
    {"name": "Wired ES Ciencia", "url": "https://es.wired.com/feed/ciencia/rss", "kind": "article", "ai_filter": True},
    # Canales de YouTube (video / episodio)
    {"name": "YouTube: canal 1", "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCxcDzs-4quJV4QsairlFYNg", "kind": "video"},
    {"name": "YouTube: canal 2", "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCl5-lvQyfILb-l2abPk4Ntg", "kind": "video"},
    {"name": "YouTube: canal 3", "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCnjgxChqYYnyoqO4k_Q1d6Q", "kind": "video"},
]

MAX_ITEMS_PER_FEED = 10
ARTICLE_MAX_AGE = timedelta(hours=48)
VIDEO_MAX_AGE = timedelta(days=7)  # los canales no publican todos los dias
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "news.json")
ARG_TZ = timezone(timedelta(hours=-3))
GEMINI_MODEL = "gemini-3.8-flash"
GROQ_MODEL = "openai/gpt-oss-120b"
TELEGRAM_LIMIT = 4000  # el maximo real es 4096

AI_RE = re.compile(
    r"\b(ia|inteligencia artificial|ai|a\.i\.|llm|gpt|chatgpt|openai|anthropic|claude|gemini|"
    r"deepmind|copilot|machine learning|aprendizaje autom|neural|modelo de lenguaje|"
    r"generativ|agente|nvidia|mistral|llama|grok|deepseek|robot)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """Sos un editor de noticias de inteligencia artificial para un programador argentino que quiere informarse sin sesgo politico.
Recibis una lista numerada de entradas candidatas. Cada una tiene un numero [n], un tipo (ARTICULO o VIDEO), la fuente, el titulo y un resumen.
Elegi las 3 a 5 mas importantes e interesantes del dia (lanzamientos, decisiones de directivos, declaraciones, impacto en la sociedad, avances tecnicos relevantes). Evita duplicados del mismo tema y notas de poca relevancia.
Si hay entradas de tipo VIDEO, incluí al menos una de ellas entre las elegidas (la mas interesante); en ese caso mencioná el nombre del canal en el brief, por ejemplo "En su ultimo video, el canal X cuenta que...".

Respondé SOLO con JSON valido, sin texto extra, con esta forma exacta:
{"items":[{"ref": 3, "title": "...", "brief": "...", "detail": "..."}]}

Reglas estrictas:
- ref: el numero [n] de la entrada elegida, tal cual.
- Entre 3 y 5 items.
- title: corto, sin caracteres raros.
- brief: UNA sola oracion autocontenida de 15 a 25 palabras.
- detail: entre 2 y 4 oraciones que amplian el brief.
- Todo en espanol rioplatense, tono conversacional, nada de lenguaje de agencia de noticias.
- Texto plano: sin markdown, sin asteriscos, sin guiones de lista, sin numerales, sin emojis, sin saltos de linea dentro de los textos. Todo se lee en voz alta.
- Explica las siglas o evitalas (no se pueden deletrear bien al hablar).
- No inventes datos: usa solo lo que dicen las entradas.
- Tono neutral, sin sesgo politico."""

FORBIDDEN_CHARS_RE = re.compile(r"[*_#`•\U0001F300-\U0001FAFF]")
TAG_RE = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------------------
# Recoleccion
# ---------------------------------------------------------------------------
def _clean(text):
    text = html.unescape(TAG_RE.sub(" ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _entry_datetime(entry):
    st = entry.get("published_parsed") or entry.get("updated_parsed")
    if not st:
        return None
    return datetime.fromtimestamp(calendar.timegm(st), tz=timezone.utc)


def _entry_summary(entry, feed_cfg):
    if feed_cfg.get("desc_only"):
        raw = entry.get("summary", "")
    else:
        raw = entry.get("summary") or ""
        if not raw and entry.get("content"):
            raw = entry["content"][0].get("value", "")
    text = _clean(raw)
    # YouTube: la descripcion viene en media_description
    if not text:
        text = _clean(entry.get("media_description", ""))
    return text[:600]


def collect_candidates():
    """Devuelve lista de dicts {kind, source, title, summary, url}."""
    now = datetime.now(timezone.utc)
    candidates = []
    for cfg in FEEDS:
        try:
            resp = requests.get(cfg["url"], timeout=20, headers={"User-Agent": "agente-noticias-ia/1.0"})
            resp.raise_for_status()
            parsed = feedparser.parse(resp.content)
        except Exception as exc:  # una fuente caida no debe frenar al resto
            print(f"[feed] {cfg['name']}: ERROR {exc}")
            continue

        max_age = VIDEO_MAX_AGE if cfg["kind"] == "video" else ARTICLE_MAX_AGE
        kept = 0
        for entry in parsed.entries:
            if kept >= cfg.get("max", MAX_ITEMS_PER_FEED):
                break
            published = _entry_datetime(entry)
            if published and now - published > max_age:
                continue
            title = _clean(entry.get("title", ""))
            if not title:
                continue
            cats = [t.get("term", "").lower() for t in entry.get("tags", [])]
            if any(skip in cats for skip in cfg.get("skip_categories", [])):
                continue
            summary = _entry_summary(entry, cfg)
            if cfg.get("ai_filter") and not AI_RE.search(f"{title} {summary}"):
                continue
            source = cfg["name"]
            if cfg["kind"] == "video":
                author = _clean(entry.get("author", ""))
                if author:
                    source = f"YouTube: {author}"
            candidates.append({
                "kind": cfg["kind"],
                "source": source,
                "title": title,
                "summary": summary,
                "url": entry.get("link", ""),
            })
            kept += 1
        print(f"[feed] {cfg['name']}: {kept} entradas")
    return candidates


def build_prompt_text(candidates):
    blocks = []
    for i, c in enumerate(candidates):
        tipo = "VIDEO" if c["kind"] == "video" else "ARTICULO"
        blocks.append(f"[{i}] ({tipo}) Fuente: {c['source']} - {c['title']} :: {c['summary']}")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# LLMs
# ---------------------------------------------------------------------------
def _raise_with_body(resp):
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")


def call_gemini(prompt_text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"}
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    _raise_with_body(resp)
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def call_groq(prompt_text):
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
        },
        timeout=90,
    )
    _raise_with_body(resp)
    return json.loads(resp.json()["choices"][0]["message"]["content"])


# ---------------------------------------------------------------------------
# Validacion y salida
# ---------------------------------------------------------------------------
def validate_items(payload, candidates):
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not 3 <= len(items) <= 5:
        raise ValueError("Se esperaban entre 3 y 5 items")
    seen = set()
    clean = []
    for it in items:
        ref = it.get("ref")
        if not isinstance(ref, int) or not 0 <= ref < len(candidates) or ref in seen:
            raise ValueError(f"ref invalido o repetido: {ref!r}")
        seen.add(ref)
        out = {}
        for key in ("title", "brief", "detail"):
            val = it.get(key)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"Falta {key}")
            val = re.sub(r"\s+", " ", val).strip()
            if FORBIDDEN_CHARS_RE.search(val):
                raise ValueError(f"Caracteres prohibidos en {key}")
            out[key] = val
        words = len(out["brief"].split())
        if not 10 <= words <= 35:
            raise ValueError(f"brief con {words} palabras")
        out["ref"] = ref
        clean.append(out)
    has_video_candidate = any(c["kind"] == "video" for c in candidates)
    if has_video_candidate and not any(candidates[i["ref"]]["kind"] == "video" for i in clean):
        print("[aviso] habia videos candidatos pero el LLM no eligio ninguno")
    return clean


def build_output(items):
    now = datetime.now(ARG_TZ)
    return {
        "date": now.strftime("%Y-%m-%d"),
        "updated_at": now.isoformat(timespec="seconds"),
        # news.json respeta el contrato: solo title/brief/detail (se lee por TTS)
        "items": [{k: i[k] for k in ("title", "brief", "detail")} for i in items],
    }


def write_news_json(payload):
    path = os.path.abspath(OUTPUT_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[ok] escrito {path}")


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def format_telegram_messages(payload, items, candidates):
    header = f"Noticias de IA - {payload['date']}"
    blocks = []
    for n, (pub, it) in enumerate(zip(payload["items"], items), start=1):
        cand = candidates[it["ref"]]
        tag = "[VIDEO] " if cand["kind"] == "video" else ""
        lines = [f"{n}. {tag}{pub['title']}", pub["brief"], "", pub["detail"]]
        if cand["url"]:
            lines.append(f"{cand['source']}: {cand['url']}")
        blocks.append("\n".join(lines))

    messages, current = [], header
    for block in blocks:
        if len(current) + 2 + len(block) > TELEGRAM_LIMIT:
            messages.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}"
    messages.append(current)
    return messages


def notify_telegram(payload, items, candidates):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[telegram] sin credenciales, se omite")
        return
    try:
        for text in format_telegram_messages(payload, items, candidates):
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
                timeout=20,
            )
            if resp.ok:
                print(f"[telegram] enviado ({len(text)} caracteres)")
            else:
                print(f"[telegram] ERROR HTTP {resp.status_code}: {resp.text[:300]}")
            time.sleep(1)
    except Exception as exc:
        print(f"[telegram] aviso: {exc}")


# ---------------------------------------------------------------------------
def main():
    candidates = collect_candidates()
    print(f"[info] {len(candidates)} candidatas en total")
    if len(candidates) < 3:
        print("[error] muy pocas candidatas; se conserva news.json anterior")
        return 1
    prompt_text = build_prompt_text(candidates)

    items = None
    for name, fn in (("Gemini", call_gemini), ("Groq", call_groq)):
        try:
            items = validate_items(fn(prompt_text), candidates)
            print(f"[ok] {name} genero {len(items)} items")
            break
        except Exception as exc:
            print(f"[error] {name}: {exc}")
    if items is None:
        print("[error] fallaron ambos LLMs; se conserva news.json anterior")
        return 1

    payload = build_output(items)
    write_news_json(payload)
    notify_telegram(payload, items, candidates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
