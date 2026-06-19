# futbotmx26_JARDI

Copa FutBotMX 2026 - Vision por Computadora.

Esta rama agrega una primera version del modulo de **homografia movil** para convertir frames de futbol de robots a una vista cenital basada en `cancha.png`.

## Contenido

```text
futbot_homography/
  tracking.py                      # libreria reusable de homografia movil

scripts/
  calibrate_homography.py          # calibracion manual frame -> cancha
  estimate_video_homographies.py   # estima H_t para todos los frames

examples/
  homography_tracking_example.py   # ejemplo de uso programatico

configs/
  homography_points.example.json   # plantilla de puntos de calibracion
  calibrations/                    # calibraciones listas por video

docs/
  homografia_video.md              # explicacion teorica y tecnica
  narracion.md                     # narracion GPT + audio ElevenLabs

futbot_narration/
  narrator.py                      # stream de acciones -> texto + MP3
```

## Instalacion

```bash
pip install -r requirements.txt
```

El codigo fue probado con Python 3.11, NumPy y OpenCV.

Para la narracion se usan APIs HTTP de OpenAI y ElevenLabs sin dependencias
extra. Configura tus llaves en `.env` a partir de `.env.example`.

## Flujo de uso

### 1. Calibrar el frame de referencia

Prepara un JSON de puntos a partir de la plantilla:

```bash
cp configs/homography_points.example.json homography_points.json
```

Edita `homography_points.json` para apuntar a tus imagenes:

```json
{
  "frame_image": "data/frame_0001.jpg",
  "field_image": "data/cancha.png",
  "points": []
}
```

Abre la herramienta de calibracion:

```bash
python scripts/calibrate_homography.py --points homography_points.json
```

En la ventana:

- click en el frame,
- click en el punto correspondiente de la cancha,
- repetir al menos 4 veces,
- presionar `Enter` para guardar.

Se recomienda usar 6 a 12 puntos bien distribuidos sobre las lineas de la cancha.

### Calibraciones incluidas

Esta rama incluye calibraciones ya capturadas para tres videos:

```text
configs/calibrations/homography_points_9913.json
configs/calibrations/homography_points_9933.json
configs/calibrations/homography_points_9938.json
```

En el layout local usado durante el desarrollo, estas calibraciones apuntan a `../videos/...` y `../cancha.png` desde el repo clonado dentro del proyecto padre.

### 2. Estimar homografias por frame

```bash
python scripts/estimate_video_homographies.py \
  --frames-dir data/frames \
  --points homography_points.json \
  --output homographies.json \
  --csv-output homographies.csv \
  --write-preview
```

Esto calcula una matriz por frame:

```text
H_t: frame_t -> cancha.png
```

Las vistas cenitales de depuracion quedan en:

```text
homography_previews/
```

### 3. Usar desde codigo

```python
from pathlib import Path
import numpy as np

from futbot_homography import (
    HomographyTrackingConfig,
    compute_reference_homography,
    estimate_homographies_for_frame_paths,
    sorted_frame_paths,
    transform_points_to_field,
)

frames = sorted_frame_paths("data/frames")
reference = compute_reference_homography("homography_points.json")

results = estimate_homographies_for_frame_paths(
    frames,
    Path("data/frames/frame_0001.jpg"),
    reference.H_ref,
    config=HomographyTrackingConfig(tracking_mode="sequential"),
)

robot_points_px = np.array([[960.0, 540.0]], dtype=np.float32)
robot_points_field = transform_points_to_field(
    robot_points_px,
    results[0].H_frame_to_field,
)
```

## Documentacion tecnica

La explicacion completa del metodo esta en:

[docs/homografia_video.md](docs/homografia_video.md)

La integracion de narracion automatica esta en:

[docs/narracion.md](docs/narracion.md)

Incluye:

- teoria de la homografia base `H_ref`,
- actualizacion frame a frame,
- tracking secuencial,
- uso de ECC,
- procesamiento de lineas blancas,
- proyeccion de puntos a vista cenital.

## Narracion automatica

Esta rama agrega una tuberia que recibe acciones del partido en JSONL, genera
narracion estilo comentarista de futbol mexicano con OpenAI y convierte el texto
a audio con ElevenLabs usando la voz `QpDQJR3frbDwOhTIo8nW`.

Configura `.env` con tus llaves:

```bash
OPENAI_API_KEY=
ELEVENLABS_API_KEY=
OPENAI_MODEL=gpt-5.5
OPENAI_REASONING_EFFORT=low
NARRATION_OPENING_PHRASE=Amigos aficionados que viven la intensidad del futbol
ELEVENLABS_VOICE_ID=QpDQJR3frbDwOhTIo8nW
ELEVENLABS_VOICE_SPEED=1.18
```

La frase de `NARRATION_OPENING_PHRASE` se escucha antes de la primera accion de
cada corrida. Si presionas `Limpiar`, se vuelve a activar para la siguiente
accion.

Prueba sin consumir APIs:

```bash
python scripts/run_narration_stream.py \
  --actions examples/actions_stream.jsonl \
  --mock
```

Uso real:

```bash
python scripts/run_narration_stream.py \
  --actions examples/actions_stream.jsonl
```

Web app en vivo:

```bash
python scripts/live_narration_server.py
```

Web app usando streaming de ElevenLabs:

```bash
python scripts/live_narration_server.py --stream-audio
```

Abre `http://127.0.0.1:8060`, activa `Audio` y envia acciones a
`POST /api/actions`.

Ejemplo:

```bash
curl -X POST http://127.0.0.1:8060/api/actions \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"00:42","type":"pase","team":"blanco","robot_id":"B2","target_robot_id":"B4","confidence":0.91}'
```

Para cerrar el servidor usa `Ctrl+C` en la terminal donde esta corriendo. Si el
puerto quedo ocupado, usa:

```bash
python scripts/stop_live_narration_server.py
```

Para revisar antes de detener:

```bash
python scripts/stop_live_narration_server.py --dry-run
```

La documentacion completa para integrarlo con los detectores esta en
[docs/narracion.md](docs/narracion.md).

Para revisar el estado de la web app:

```bash
curl http://127.0.0.1:8060/api/status
curl http://127.0.0.1:8060/api/history
```

El boton `Limpiar` llama a `POST /api/reset`: vacia la cola, detiene audio local
e invalida respuestas viejas de GPT/ElevenLabs para que no reaparezcan acciones
anteriores ni se queden filas en `NARRANDO`.

Prueba directa de la voz en streaming:

```bash
python scripts/test_elevenlabs_stream.py \
  --text "Gol del equipo blanco, B4 la manda a guardar y se prende la cancha."
```

Benchmark de latencia con y sin streaming:

```bash
python scripts/benchmark_elevenlabs_latency.py --trials 3
```
