# futbot_activity_recognition

Modulo de **reconocimiento de actividad basado en reglas** usando coordenadas de cancha por frame.

## Contenido

```text
futbot_activity_recognition/
  __init__.py     # API publica
  detector.py     # GoalDetector, dataclasses y loader de zonas

configs/roi/
  goal_zones_{video_id}.json   # zonas de gol por video
```

## Eventos detectados

### Goal

Se detecta un gol cuando:

1. En el frame `t`, la pelota esta cerca de un robot de clase X (dentro de `proximity_threshold`).
2. En el frame `t+1`, la pelota entra en la zona de gol de clase X.

Esto significa que el equipo que tenia posesion recibio un gol (el equipo contrario anoto).

## Configuracion

Archivo `configs/roi/goal_zones_{video_id}.json`:

```json
{
  "robot_a": {
    "bbox": [871, 232, 45, 226]
  },
  "robot_b": {
    "bbox": [3, 231, 45, 226]
  }
}
```

- **bbox**: `[x, y, w, h]` en pixeles de la imagen de cancha (homografia).
- Las zonas se definen por clase de robot — la zona de `robot_a` es donde `robot_a` defiende.

## Uso desde codigo

```python
from futbot_activity_recognition import GoalDetector, load_goal_zones

# Cargar zonas
zones = load_goal_zones("configs/roi/goal_zones_9913.json")

# Inicializar detector
detector = GoalDetector(goal_zones=zones, proximity_threshold=100.0)

# Por cada frame, pasar las posiciones en coordenadas de cancha
for frame_idx, field_positions in frame_data:
    events = detector.update(frame_idx, field_positions)
    for event in events:
        print(f"Frame {event.frame_idx}: {event.event_type} — {event.details}")
```

## API publica

```python
# Clases
GoalZone, ActivityEvent, GoalDetector

# Funciones
load_goal_zones
```
