# futbot_sam

Módulo de **tracking multi-objeto** basado en SAM3 (Segment Anything Model 3) para videos de fútbol de robots.

## Contenido

```text
futbot_sam/
  __init__.py          # API publica del paquete
  tracker.py           # SAMTracker, dataclasses y helpers de asignacion
  visualization.py     # renderizado de mascaras y video overlay

configs/tracking/
  tracking_classes_9913.json   # configuracion de clases para IMG_9913
  tracking_classes_9933.json   # configuracion de clases para IMG_9933
  tracking_classes_9938.json   # configuracion de clases para IMG_9938

examples/
  sam_tracking_example.py      # ejemplo de uso independiente
```

## Instalación

```bash
conda env create -f futbotmx26.yml
```

Requiere además el paquete `sam3` (SAM3 video predictor) instalado y accesible.

## Arquitectura del módulo

### Clases principales

- **`TrackingClass`** - define un objeto a rastrear: `prompt` (texto para SAM), `label` (etiqueta de clase) y `bbox` opcional (bounding box normalizado `[x, y, w, h]`).
- **`TrackingConfig`** - parámetros del tracker: `offload_video_to_cpu`, `mask_alpha`, `font_scale`, `font_thickness`.
- **`SAMTracker`** - orquesta el tracking completo: agrupa clases por prompt, ejecuta sesiones SAM3 y asigna objetos detectados a sub-clases.
- **`TrackingResult`** - contiene los resultados: diccionario de `frame_idx` -> `FrameResult`.
- **`FrameResult`** - máscaras y sus índices de clase para un frame.

### Flujo interno del tracker

1. Las clases se agrupan por `prompt` (ej: todas las entradas con prompt `"robot"` se ejecutan en una sola sesión SAM).
2. Para cada grupo se inicia una sesión SAM3 con `start_session`, se agrega el prompt con `add_prompt` (texto + bbox opcional) y se propaga con `propagate_in_video`.
3. Si un grupo tiene múltiples sub-clases con bbox (ej: `robot_a` y `robot_b`), los objetos detectados se asignan a la sub-clase más cercana comparando el centroide de la máscara en el frame 0 con el centro del bbox de referencia.
4. Las sesiones se cierran después de cada grupo para liberar memoria GPU.

### Helpers

- **`load_tracking_classes(config_path)`** - carga clases desde un JSON de configuración, normalizando bboxes de píxeles a `[0, 1]`.
- **`bbox_center(bbox_norm, h, w)`** - centro en píxeles de un bbox normalizado.
- **`mask_centroid(mask)`** - centroide `(cy, cx)` de una máscara binaria.

### Visualización

- **`save_mask_frames(result, output_dir, labels, colors)`** - guarda PNGs con máscaras coloreadas por clase.
- **`render_overlay_video(result, frames_dir, output_path, labels, fps, colors)`** - genera un video MP4 con máscaras y etiquetas superpuestas sobre los frames originales.

## Configuración por muestra

Cada video tiene su archivo JSON en `configs/tracking/`:

```json
{
  "image_width": 1920,
  "image_height": 1080,
  "tracking_classes": [
    {
      "prompt": "orange ball",
      "label": "ball",
      "bbox": [753, 393, 27, 26]
    },
    {
      "prompt": "robot",
      "label": "robot_a",
      "bbox": [277, 379, 185, 145]
    },
    {
      "prompt": "robot",
      "label": "robot_b",
      "bbox": [787, 275, 119, 154]
    }
  ]
}
```

- **`image_width`, `image_height`**: dimensiones del frame original (para normalizar bboxes).
- **`bbox`**: `[x, y, w, h]` en píxeles del frame 0. Se normaliza automáticamente al cargar.
- **`prompt`**: texto enviado a SAM3. Clases con el mismo prompt comparten sesión.
- **`label`**: etiqueta de la clase. Múltiples entradas pueden tener el mismo label (ej: dos robots del mismo equipo).

### Colores

Los colores se definen como un diccionario `label -> (R, G, B)`:

```python
COLORS = {
    "ball": (255, 0, 0),
    "robot_a": (255, 255, 0),
    "robot_b": (0, 100, 255),
}
```

## Uso desde código

```python
from futbot_sam import SAMTracker, TrackingConfig, load_tracking_classes
from futbot_sam import save_mask_frames, render_overlay_video

# 1. Cargar configuración
tracking_classes = load_tracking_classes("configs/tracking/tracking_classes_9913.json")
labels = [c.label for c in tracking_classes]

# 2. Ejecutar tracking
config = TrackingConfig(offload_video_to_cpu=True)
tracker = SAMTracker(tracking_classes, config)
result = tracker.track("data/frames/IMG_9913")

# 3. Guardar resultados
COLORS = {"ball": (255, 0, 0), "robot_a": (255, 255, 0), "robot_b": (0, 100, 255)}

save_mask_frames(result, "output/mask_frames", labels=labels, colors=COLORS)

render_overlay_video(
    result,
    "data/frames/IMG_9913",
    "output/tracking_overlay.mp4",
    labels=labels,
    fps=20.0,
    colors=COLORS,
)
```

## Integración con el pipeline principal

El módulo se integra con `futbot_homography` en `main.py`:

1. **SAM tracking** produce máscaras por frame con etiquetas de clase.
2. Se extraen centroides de las máscaras (base del robot).
3. **Homografía** transforma los centroides de píxeles a coordenadas de cancha.
4. Se genera un video de 2 paneles: overlay SAM | mapa de cancha.

## API pública

```python
# Clases
TrackingClass, TrackingConfig, TrackingResult, FrameResult, SAMTracker

# Funciones
load_tracking_classes, bbox_center, mask_centroid
save_mask_frames, render_overlay_video
```
