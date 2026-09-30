# agente-noticias-ia

Junta noticias de IA de varias fuentes RSS, les pide a Gemini (con Groq como
fallback) que elija las 3-5 más importantes del día, y publica el resultado
como `docs/news.json` — el archivo que Jarvis consume por HTTPS/GitHub Pages
(ver `news_format.md`). Opcionalmente también manda el mismo resumen por
Telegram.

## Puesta en marcha

1. **Crear el repo en GitHub** (público, para que Pages y Actions salgan
   gratis sin límite de minutos) y subir este contenido:

   ```bash
   cd agente-noticias-ia
   git init
   git add .
   git commit -m "init"
   git branch -M main
   git remote add origin https://github.com/<tu-usuario>/agente-noticias-ia.git
   git push -u origin main
   ```

2. **Cargar los Secrets** en Settings → Secrets and variables → Actions:
   - `GEMINI_API_KEY` (obligatoria)
   - `GROQ_API_KEY` (opcional, fallback si Gemini falla)
   - `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID` (opcionales, si querés el aviso
     por Telegram además del `news.json` para Jarvis)

3. **Activar GitHub Pages**: Settings → Pages → Source: "Deploy from a
   branch" → Branch: `main`, carpeta `/docs`. La URL que te da Pages (algo
   como `https://<tu-usuario>.github.io/agente-noticias-ia/news.json`) es la
   que va en `config::kNewsUrl` del firmware de Jarvis.

4. **Probar manualmente** antes de esperar al cron: pestaña Actions →
   `daily-news` → "Run workflow". Revisá que `docs/news.json` se haya
   actualizado con la fecha de hoy.

5. El cron corre solo todos los días a las 08:00 (hora Argentina). Se puede
   ajustar editando el `cron` en `.github/workflows/daily-news.yml`.

## Cómo conseguir el `TELEGRAM_CHAT_ID`

1. Creá el bot con [@BotFather](https://t.me/BotFather) → `/newbot` → te da
   el `TELEGRAM_BOT_TOKEN`.
2. Mandale cualquier mensaje al bot desde tu cuenta de Telegram.
3. Abrí `https://api.telegram.org/bot<TOKEN>/getUpdates` en el navegador y
   buscá `"chat":{"id": ...}` en la respuesta — ese número es el
   `TELEGRAM_CHAT_ID`.

## Si un día falla la generación

El script nunca deja `docs/news.json` vacío o corrupto: si Gemini y Groq
fallan, o las fuentes RSS no responden, el workflow no commitea nada y se
queda el archivo del día anterior (con su `date` vieja) — Jarvis, al ver que
`date` no es hoy, avisa que no hay noticias nuevas en vez de leer algo roto.
