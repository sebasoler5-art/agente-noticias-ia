"""
Agente de noticias de IA para Jarvis (+ notificación opcional por Telegram).

Flujo:
1. Junta ítems recientes de una lista fija de fuentes RSS de IA.
2. Le pide a Gemini (y si falla, a Groq) que elija las 3-5 noticias más
   importantes del día y las devuelva en el formato que necesita Jarvis
   (texto plano, apto para leerse en voz alta por TTS).
3. Valida la respuesta. Si algo falla en cualquier paso, NO TOCA
   docs/news.json — se queda el del día anterior, tal como pide el
   contrato de news_format.md.
4. Si hay credenciales de Telegram configuradas, manda el mismo digest.

Variables de entorno esperadas (se configuran como Secrets en GitHub Actions):
  GEMINI_API_KEY        (obligatoria)
  GROQ_API_KEY          (opcional, fallback)
  TELEGRAM_BOT_TOKEN    (opcional)
  TELEGRAM_CHAT_ID      (opcional)
"""

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import feedparser
import requests

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

FEEDS = [
    "http://export.arxiv.org/rss/cs.AI",
    "https://hnrss.org/best",
    "https://openai.com/news/rss.xml",
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://news.google.com/rss/search?q=Anthropic+OR+%22Google+DeepMind%22+when:2d&hl=en-US&gl=US&ceid=US:en",
]

MAX_ITEMS_PER_FEED = 10
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "news.json")
ARG_TZ = timezone(timedelta(hours=-3))  # Argentina no tiene horario de verano

GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """Sos el editor que arma el resumen diario de inteligencia artificial \
para Jarvis, un asistente de voz. Vas a recibir una lista de titulares y resúmenes \
recientes de distintas fuentes. Tu trabajo es elegir SOLO las noticias de IA más \
importantes del día (lanzamientos, investigación relevante, modelos nuevos, movimientos \
de las empresas grandes del sector, política/regulación de IA) y devolverlas en un \
formato estricto, porque el resultado se lee en voz alta con un sintetizador de voz.

Reglas OBLIGATORIAS, sin excepción:
- Devolvé SOLO un JSON válido con esta forma exacta, nada de texto antes o después:
  {"items": [{"title": "...", "brief": "...", "detail": "..."}]}
- Entre 3 y 5 ítems. Priorizá impacto real sobre ruido o notas menores.
- "title": corto (para logs, no se lee en voz alta), sin caracteres raros.
- "brief": UNA sola oración autocontenida, entre 15 y 25 palabras, que alguien pueda \
entender sin contexto previo.
- "detail": entre 2 y 4 oraciones ampliando el brief.
- Español rioplatense, tono conversacional (como le explicarías esto a un amigo), \
nunca tono de cable de agencia ni acartonado.
- Nada de markdown: prohibido usar *, #, guiones de lista, numeración, emojis o \
cualquier caracter que no sea texto plano dentro de title/brief/detail.
- No inventes ni completes con datos que no estén en el material que te paso. Si una \
noticia es ambigua o no tenés detalle suficiente, elegí otra.
- No uses siglas sin explicarlas la primera vez (Jarvis no puede "deletrear" bien).
"""

FORBIDDEN_CHARS_RE = re.compile(r"[*_#`\u2022\U0001F300-\U0001FAFF]")


# ---------------------------------------------------------------------------
# 1. Recolección de fuentes
# ---------------------------------------------------------------------------

def collect_raw_items() -> str:
    blocks = []
    for url in FEEDS:
        try:
            parsed = feedparser.parse(url)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] no pude leer {url}: {exc}", file=sys.stderr)
            continue

        source_name = parsed.feed.get("title", url)
        entries = parsed.entries[:MAX_ITEMS_PER_FEED]
        if not entries:
            continue

        lines = [f"### Fuente: {source_name}"]
        for entry in entries:
            title = entry.get("title", "").strip()
            summary = entry.get("summary", entry.get("description", "")).strip()
            summary = re.sub(r"<[^>]+>", " ", summary)  # saca HTML crudo
            summary = re.sub(r"\s+", " ", summary)[:600]
            lines.append(f"- {title} :: {summary}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 2. Llamadas a los LLM
# ---------------------------------------------------------------------------

def _raise_with_body(resp: requests.Response) -> None:
    """Como raise_for_status() pero mostrando el cuerpo de la respuesta,
    que es donde el proveedor explica la causa real del error."""
    if not resp.ok:
        raise RuntimeError(
            f"{resp.status_code} {resp.reason} — respuesta del servidor: {resp.text[:500]}"
        )


def call_gemini(raw_items: str) -> dict:
    api_key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {
        "x-goog-api-key": api_key,  # la key va en el header, no en ?key=
        "Content-Type": "application/json",
    }
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": raw_items}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.3,
        },
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    _raise_with_body(resp)
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def call_groq(raw_items: str) -> dict:
    api_key = os.environ["GROQ_API_KEY"]
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_items},
        ],
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    _raise_with_body(resp)
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return json.loads(text)


# ---------------------------------------------------------------------------
# 3. Validación (las reglas que le importan al firmware)
# ---------------------------------------------------------------------------

def validate_items(payload: dict) -> list:
    items = payload.get("items")
    if not isinstance(items, list) or not (3 <= len(items) <= 5):
        raise ValueError(f"'items' inválido o fuera de rango (3-5): {items!r}")

    for item in items:
        for field in ("title", "brief", "detail"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"item sin '{field}' válido: {item!r}")
            if FORBIDDEN_CHARS_RE.search(value):
                raise ValueError(f"'{field}' tiene markdown/emoji prohibido: {value!r}")

        brief_words = len(item["brief"].split())
        if not (10 <= brief_words <= 35):  # margen razonable sobre el 15-25 pedido
            raise ValueError(f"'brief' con longitud rara ({brief_words} palabras): {item['brief']!r}")

    return items


# ---------------------------------------------------------------------------
# 4. Salida
# ---------------------------------------------------------------------------

def build_output(items: list) -> dict:
    now = datetime.now(ARG_TZ)
    return {
        "date": now.strftime("%Y-%m-%d"),
        "updated_at": now.isoformat(timespec="seconds"),
        "items": items,
    }


def write_news_json(payload: dict) -> None:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def notify_telegram(payload: dict) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    lines = [f"Resumen de IA — {payload['date']}", ""]
    for item in payload["items"]:
        lines.append(f"• {item['title']}")
        lines.append(item["brief"])
        lines.append("")
    text = "\n".join(lines).strip()

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=30)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] no pude mandar Telegram: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    raw_items = collect_raw_items()
    if not raw_items.strip():
        print("[error] no se pudo leer ninguna fuente RSS, no toco news.json", file=sys.stderr)
        return 1

    payload = None
    for engine_name, call in (("Gemini", call_gemini), ("Groq", call_groq)):
        try:
            raw = call(raw_items)
            items = validate_items(raw)
            payload = build_output(items)
            print(f"[ok] generado con {engine_name}: {len(items)} ítems")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] falló {engine_name}: {exc}", file=sys.stderr)

    if payload is None:
        print("[error] Gemini y Groq fallaron, no toco news.json", file=sys.stderr)
        return 1

    write_news_json(payload)
    notify_telegram(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
