# agente-noticias-ia

Junta noticias de IA (sitios + canales de YouTube), un LLM elige las 3-5 mas importantes (con al menos un video si hay), y publica:
- `docs/news.json` (GitHub Pages, contrato `news_format.md`) para Jarvis.
- Un mensaje de Telegram con titulo, resumen, detalle y link de cada noticia.

Secrets necesarios: `GEMINI_API_KEY`, `GROQ_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

Ejecucion manual: Actions > daily-news > **Run workflow** (no "Re-run": ese reusa el commit viejo).
Automatica: cron `0 11 * * *` (8:00 Argentina). Para desactivarla, borrar el bloque `schedule` del workflow.

Si fallan los LLMs o hay muy pocas noticias, se conserva el `news.json` anterior.
