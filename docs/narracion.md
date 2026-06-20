# Narración automática en vivo

Este módulo convierte acciones detectadas durante el partido en narración de
comentarista y audio reproducible en una web app local.

La cadena completa es:

```text
detector de acciones -> POST /api/actions -> GPT -> ElevenLabs -> MP3 -> navegador
```

Los equipos esperados para el torneo son:

- `blanco`
- `negro`

Usa esos valores en el campo `team` y en las llaves de `score`.

## Archivos principales

```text
futbot_narration/
  narrator.py                      # GPT + ElevenLabs + contexto del partido

scripts/
  run_narration_stream.py          # procesa un JSONL por terminal
  live_narration_server.py         # servidor web local en vivo
  stop_live_narration_server.py    # detiene servidores locales de narración
  test_elevenlabs_stream.py        # prueba directa de TTS streaming
  benchmark_elevenlabs_latency.py  # compara TTS normal vs streaming

web/narration_live/
  index.html                       # UI
  app.css                          # estilos
  app.js                           # SSE, cola de audio, actualización de acciones

examples/
  actions_stream.jsonl             # acciones de ejemplo blanco/negro
```

## Variables de entorno

Crea `.env` en la raíz de `futbotmx26_JARDI`. El archivo está ignorado por git.
Nunca subas llaves reales al repositorio.

```bash
OPENAI_API_KEY=tu_api_key_de_openai
ELEVENLABS_API_KEY=tu_api_key_de_elevenlabs
OPENAI_MODEL=gpt-5.5
OPENAI_REASONING_EFFORT=low
NARRATION_REQUEST_TIMEOUT_SECONDS=12
NARRATION_OPENING_PHRASE=Amigos aficionados que viven la intensidad del futbol
ELEVENLABS_VOICE_ID=QpDQJR3frbDwOhTIo8nW
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
ELEVENLABS_OUTPUT_FORMAT=mp3_44100_128
ELEVENLABS_VOICE_SPEED=1.18
ELEVENLABS_STREAMING_LATENCY=3
```

Notas:

- `OPENAI_REASONING_EFFORT=low` prioriza baja latencia para narración en vivo.
- `NARRATION_REQUEST_TIMEOUT_SECONDS=12` evita que una llamada lenta deje la UI atorada en `NARRANDO`.
- `NARRATION_OPENING_PHRASE` se agrega al audio de la primera acción de cada corrida/reset.
- `ELEVENLABS_VOICE_ID` ya está configurado con la voz solicitada.
- `ELEVENLABS_VOICE_SPEED=1.18` acelera la voz sin sonar demasiado atropellada.
- `ELEVENLABS_STREAMING_LATENCY=3` pide optimización de latencia al endpoint streaming.
- Puedes copiar `.env.example` como punto de partida.

## Levantar el servidor en vivo

Desde la raíz del repo:

```bash
cd /Users/juanterven/dev/Copa_FutBolMX/futbotmx26_JARDI
python scripts/live_narration_server.py
```

Modo streaming de audio:

```bash
python scripts/live_narration_server.py --stream-audio
```

En modo normal, el servidor espera a que ElevenLabs termine el MP3 y luego lo
sirve desde `outputs/live_narration_audio/`. En modo `--stream-audio`, la UI
recibe una URL `/audio-stream/<id>.mp3` y el servidor proxyea los chunks del
endpoint streaming de ElevenLabs al navegador.

Abre:

```text
http://127.0.0.1:8060
```

En la web app:

1. Presiona `Audio`.
2. Manda acciones al endpoint `POST /api/actions`.
3. La acción aparece inmediatamente.
4. Cuando terminan GPT y ElevenLabs, se muestra el texto narrado y se reproduce el MP3.

Si el navegador bloquea autoplay, el botón cambia a `Reintentar audio`. Da click
otra vez y se reproduce lo que haya quedado en cola.

## Cerrar el servidor

Si el servidor está corriendo en la terminal actual:

```text
Ctrl+C
```

Si se quedó corriendo en otra terminal y ocupa el puerto `8060`:

```bash
python scripts/stop_live_narration_server.py
```

Para ver qué mataría sin detener nada:

```bash
python scripts/stop_live_narration_server.py --dry-run
```

Si no se detiene con `SIGTERM`:

```bash
python scripts/stop_live_narration_server.py --force
```

También puedes cambiar el puerto:

```bash
python scripts/live_narration_server.py --port 8061
```

## Prueba rápida sin gastar APIs

Modo mock:

```bash
python scripts/live_narration_server.py --mock
```

En otra terminal:

```bash
curl -X POST http://127.0.0.1:8060/api/actions \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"00:42","type":"pase","team":"blanco","robot_id":"B2","target_robot_id":"B4","confidence":0.91}'
```

El modo mock no llama a OpenAI ni a ElevenLabs; genera texto y archivos de audio
simulados para validar el flujo.

## Prueba con APIs reales

Con `.env` configurado:

```bash
python scripts/live_narration_server.py
```

En otra terminal:

```bash
curl -X POST http://127.0.0.1:8060/api/actions \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"00:51","type":"gol","team":"blanco","robot_id":"B4","score":{"blanco":2,"negro":0},"confidence":0.97}'
```

Los MP3 quedan en:

```text
outputs/live_narration_audio/
```

El manifiesto de acciones/narraciones queda en:

```text
outputs/live_narration_manifest.jsonl
```

`outputs/` está ignorado por git.

## Probar solo la voz en streaming

Este comando no usa GPT. Solo manda una frase a ElevenLabs usando el endpoint
streaming y guarda el MP3 resultante:

```bash
python scripts/test_elevenlabs_stream.py \
  --text "Gol del equipo blanco, B4 la manda a guardar y se prende la cancha." \
  --output outputs/streaming_test.mp3
```

El script imprime:

- `time_to_first_chunk_ms`: tiempo hasta recibir el primer chunk de audio.
- `total_ms`: tiempo total hasta guardar el MP3 completo.
- `chunks` y `bytes`: datos recibidos del stream.

La documentación de ElevenLabs indica que el endpoint streaming de TTS es:

```text
POST /v1/text-to-speech/:voice_id/stream
```

También documenta `voice_settings.speed`: `1.0` es la velocidad normal, valores
mayores que `1.0` aceleran la voz.

## Comparar tiempos con y sin streaming

```bash
python scripts/benchmark_elevenlabs_latency.py --trials 3
```

Genera audios y un reporte JSON en:

```text
outputs/latency_benchmark/
```

Métricas:

- `normal.summary.mean_ms`: tiempo promedio esperando el MP3 completo.
- `stream.time_to_first_chunk_summary.mean_ms`: tiempo promedio hasta el primer chunk.
- `stream.total_summary.mean_ms`: tiempo promedio hasta terminar de recibir todo el MP3.

## Formato del evento

Cada evento es un objeto JSON. Campos recomendados:

```json
{
  "timestamp": "00:15",
  "type": "gol",
  "team": "blanco",
  "robot_id": "B4",
  "target_robot_id": "B2",
  "outcome": "a_porteria",
  "confidence": 0.96,
  "score": {
    "blanco": 1,
    "negro": 0
  }
}
```

Campos:

- `timestamp`: tiempo de partido, frame o reloj del sistema.
- `type`: acción detectada, por ejemplo `pase`, `gol`, `tiro`, `fuera_de_lugar`.
- `team`: `blanco` o `negro`.
- `robot_id`: robot principal de la acción.
- `target_robot_id`: robot destino en pases.
- `outcome`: resultado detectado, por ejemplo `completo`, `a_porteria`, `fallado`.
- `confidence`: confianza del detector entre `0` y `1`.
- `score`: marcador actual cuando esté disponible.

El narrador conserva un contexto corto de acciones recientes y marcador para que
GPT no narre cada evento como si estuviera aislado.

## Endpoints del servidor

### `GET /`

Sirve la interfaz web.

### `GET /events`

Canal SSE usado por la web app. No lo tienen que llamar los detectores.

### `POST /api/actions`

Recibe una acción:

```bash
curl -X POST http://127.0.0.1:8060/api/actions \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"00:10","type":"tiro","team":"negro","robot_id":"N7","confidence":0.88}'
```

También recibe una lista de acciones:

```bash
curl -X POST http://127.0.0.1:8060/api/actions \
  -H "Content-Type: application/json" \
  -d '[
    {"timestamp":"00:07","type":"pase","team":"blanco","robot_id":"B2","target_robot_id":"B4","confidence":0.91},
    {"timestamp":"00:13","type":"tiro","team":"blanco","robot_id":"B4","confidence":0.84}
  ]'
```

Respuesta:

```json
{
  "queued": {
    "id": 1,
    "received_at": 1781827070.95152,
    "action": {
      "timestamp": "00:10",
      "type": "tiro",
      "team": "negro",
      "robot_id": "N7",
      "confidence": 0.88
    }
  }
}
```

### `POST /api/demo`

Manda las acciones de `examples/actions_stream.jsonl` con pausa entre eventos.
Sirve para demostración desde la UI.

### `POST /api/reset`

Invalida la corrida actual, limpia historial, vacía la cola pendiente del
servidor y hace que cualquier respuesta vieja de GPT/ElevenLabs que llegue tarde
sea ignorada. En el navegador también detiene el audio actual, cancela la voz
fallback local y vacía la cola de reproducción.

Después de un reset, la siguiente acción vuelve a incluir la frase de apertura
configurada en `NARRATION_OPENING_PHRASE`.

Respuesta:

```json
{
  "ok": true,
  "generation": 3
}
```

### `GET /api/status`

Devuelve estado básico:

```json
{
  "queued": 0,
  "demo_running": false,
  "generation": 3,
  "output_dir": "outputs/live_narration_audio",
  "stream_audio": false
}
```

### `GET /audio/<archivo.mp3>`

Sirve los MP3 generados para que el navegador los reproduzca.

## Integración desde el detector de acciones

Ejemplo mínimo desde Python:

```python
import json
import urllib.request


def send_action(action):
    body = json.dumps(action).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:8060/api/actions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


send_action({
    "timestamp": "00:42",
    "type": "pase",
    "team": "blanco",
    "robot_id": "B2",
    "target_robot_id": "B4",
    "confidence": 0.91,
})
```

Ejemplo dentro de un loop:

```python
for detected_action in action_stream:
    send_action({
        "timestamp": detected_action["match_time"],
        "type": detected_action["label"],
        "team": detected_action["team"],       # blanco o negro
        "robot_id": detected_action["robot_id"],
        "target_robot_id": detected_action.get("target_robot_id"),
        "confidence": detected_action["confidence"],
        "score": detected_action.get("score"),
    })
```

## Procesar un archivo JSONL por terminal

Si no necesitas web app:

```bash
python scripts/run_narration_stream.py \
  --actions examples/actions_stream.jsonl \
  --output-dir outputs/narration_audio \
  --manifest outputs/narration_manifest.jsonl
```

Solo texto, sin ElevenLabs:

```bash
python scripts/run_narration_stream.py \
  --actions examples/actions_stream.jsonl \
  --no-audio
```

Leer desde stdin:

```bash
tail -f acciones.jsonl | python scripts/run_narration_stream.py
```

## Troubleshooting

### El puerto ya está ocupado

```bash
python scripts/stop_live_narration_server.py --dry-run
python scripts/stop_live_narration_server.py
```

O usa otro puerto:

```bash
python scripts/live_narration_server.py --port 8061
```

### No suena el audio

- Presiona `Audio` antes de mandar acciones.
- Si aparece `Reintentar audio`, vuelve a presionar el botón.
- Verifica que existan MP3 en `outputs/live_narration_audio/`.
- Si ElevenLabs se tarda o falla, el servidor publica texto narrado con
  `browser_tts: true`; la web app lo reproduce con la voz local del navegador.

### Se queda en `NARRANDO`

- El servidor tiene `NARRATION_REQUEST_TIMEOUT_SECONDS=12`; si GPT o ElevenLabs
  exceden ese tiempo, publica una narración fallback en lugar de dejar la fila
  atorada.
- Si presionaste `Limpiar`, el servidor invalida la generación anterior. Las
  respuestas viejas que lleguen después del reset no deben reaparecer en la UI.
- Revisa estado con:

```bash
curl http://127.0.0.1:8060/api/status
curl http://127.0.0.1:8060/api/history
```

### Falla por API key

Revisa `.env`:

```bash
OPENAI_API_KEY=...
ELEVENLABS_API_KEY=...
```

### La narración tarda mucho

- `OPENAI_REASONING_EFFORT=low` ya está configurado para bajar latencia.
- Usa `python scripts/live_narration_server.py --stream-audio` para empezar a
  reproducir conforme ElevenLabs devuelve chunks.
- ElevenLabs puede dominar la latencia total; para demos rápidas usa `--mock`.
- Evita mandar demasiadas acciones redundantes; el servidor procesa en cola.
