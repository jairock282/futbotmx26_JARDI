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
```

## Instalacion

```bash
pip install -r requirements.txt
```

El codigo fue probado con Python 3.11, NumPy y OpenCV.

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

Incluye:

- teoria de la homografia base `H_ref`,
- actualizacion frame a frame,
- tracking secuencial,
- uso de ECC,
- procesamiento de lineas blancas,
- proyeccion de puntos a vista cenital.
