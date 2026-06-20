# futbot_activity_recognition

Módulo de **reconocimiento de actividad basado en reglas** usando coordenadas de cancha por frame.

## Contenido

```text
futbot_activity_recognition/
  __init__.py     # API pública
  detector.py     # GoalDetector, PassingDetector, ControlDetector, dataclasses y loader de zonas

configs/roi/
  goal_zones_{video_id}.json   # zonas de gol por video
```

## Eventos detectados

### Goal

Se detecta un gol cuando:

1. En el frame `t`, la pelota está cerca de un robot de clase X (dentro de `proximity_threshold`).
2. En el frame `t+1`, la pelota entra en la zona de gol de clase X.

Esto significa que el equipo que tenía posesión recibió un gol (el equipo contrario anotó).

### Pase (Pass)

Se detecta un pase cuando:

1. En el frame `t`, la pelota está cerca de un robot `X` de clase `C` (dentro de `proximity_threshold`).
2. En el frame `t+1`, la pelota pasa a estar cerca de otro robot `Y` de la misma clase `C`, pero de identidad diferente.

Es decir, el balón se transfiere entre dos robots del mismo equipo. Los robots deben tener etiquetas distintas para distinguir identidades, por ejemplo `robot_a_0`, `robot_a_1`, `robot_b_0`, etc.

El evento incluye:

- `team`: clase del equipo que realiza el pase (`robot_a`, `robot_b`, etc.).
- `from_robot`: etiqueta del robot que tenía la pelota antes.
- `to_robot`: etiqueta del robot que recibe la pelota.
- `ball_position`: posición de la pelota en coordenadas de cancha cuando se detecta el pase.

### Posesión del balón (Control)

Se detecta que un robot controla la pelota cuando el balón permanece dentro de `proximity_threshold` del mismo robot durante al menos `hold_frames` consecutivos.

Esto modela la posesión del balón: el robot mantiene el balón cerca sin que otro robot lo intercepte. Incluye un `cooldown` para evitar eventos repetidos de control.

El evento incluye:

- `team`: clase del equipo que posee el balón.
- `robot`: etiqueta del robot que controla la pelota.
- `ball_position`: posición de la pelota en coordenadas de cancha.
- `hold_frames`: número de frames consecutivos que el robot mantuvo la pelota.

## Configuración

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

- **bbox**: `[x, y, w, h]` en píxeles de la imagen de cancha (homografía).
- Las zonas se definen por clase de robot - la zona de `robot_a` es donde `robot_a` defiende.

## Uso desde código

```python
from futbot_activity_recognition import (
    GoalDetector,
    PassingDetector,
    ControlDetector,
    load_goal_zones,
)

# Cargar zonas de gol
zones = load_goal_zones("configs/roi/goal_zones_9913.json")

# Inicializar detectores
goal_detector = GoalDetector(
    goal_zones=zones,
    proximity_threshold=100.0,
    cooldown_frames=30,
)
pass_detector = PassingDetector(
    proximity_threshold=100.0,
    cooldown_frames=30,
)
control_detector = ControlDetector(
    proximity_threshold=50.0,
    hold_frames=20,
    cooldown_frames=60,
)

# Por cada frame, pasar las posiciones en coordenadas de cancha
for frame_idx, field_positions in frame_data:
    goal_event = goal_detector.update(frame_idx, field_positions)
    if goal_event:
        print(f"Frame {goal_event.frame_idx}: {goal_event.event_type} — {goal_event.details}")

    pass_event = pass_detector.update(frame_idx, field_positions)
    if pass_event:
        print(f"Frame {pass_event.frame_idx}: {pass_event.event_type} — {pass_event.details}")

    control_event = control_detector.update(frame_idx, field_positions)
    if control_event:
        print(f"Frame {control_event.frame_idx}: {control_event.event_type} — {control_event.details}")
```

## API pública

```python
# Clases
GoalZone, ActivityEvent, GoalDetector, PassingDetector, ControlDetector

# Funciones
load_goal_zones
```
