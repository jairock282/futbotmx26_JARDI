# futbot_heatmap

Módulo de **acumulación y visualización de posiciones** de robots sobre el mapa de cancha, generando un mapa de calor superpuesto en tiempo real.

## Contenido

```text
futbot_heatmap/
  __init__.py   # API pública del paquete
  tracker.py    # HeatmapTracker: acumulación, suavizado y renderizado
```

## API pública

```python
from futbot_heatmap import HeatmapTracker
```

## Clase principal: `HeatmapTracker`

Acumula posiciones de robots frame a frame y genera un mapa de calor superpuesto sobre la imagen de la cancha.

### Constructor

```python
HeatmapTracker(field_h: int, field_w: int, sigma: float = 30.0)
```

| Parámetro | Tipo    | Descripción |
|-----------|---------|-------------|
| `field_h` | `int`   | Alto del mapa de cancha en píxeles. |
| `field_w` | `int`   | Ancho del mapa de cancha en píxeles. |
| `sigma`   | `float` | Desviación estándar del suavizado gaussiano. Por defecto `30.0`. |

Internamente inicializa `_counts`, una matriz `float32` de ceros con forma `(field_h, field_w)` donde se acumulan las visitas por celda.

### Métodos

#### `update(field_positions)`

Acumula las posiciones de un frame. Ignora el objeto con etiqueta `"ball"`.

```python
tracker.update(field_positions: list[tuple[str, float, float]])
```

- Recibe una lista de tuplas `(label, fx, fy)` en coordenadas de cancha.
- Para cada robot, convierte `fx, fy` a entero y suma `1` en la celda correspondiente de `_counts`.
- Coordenadas fuera de los límites del campo se descartan silenciosamente.

#### `get_normalized()`

Devuelve el mapa de calor suavizado y normalizado como imagen `uint8` (valores 0–255).

```python
heatmap: np.ndarray = tracker.get_normalized()
```

Proceso interno:
1. **Suavizado**: `cv2.GaussianBlur` aplicado sobre `_counts` con la `sigma` configurada.
2. **Raíz cuadrada**: reduce el contraste extremo entre zonas de alta y baja actividad.
3. **Normalización**: escala el resultado al rango 0–255 con `cv2.NORM_MINMAX`.

#### `render(field_base, alpha=0.5)`

Superpone el mapa de calor sobre una imagen base de la cancha y devuelve la imagen combinada.

```python
output: np.ndarray = tracker.render(field_base: np.ndarray, alpha: float = 0.5)
```

| Parámetro    | Descripción |
|--------------|-------------|
| `field_base` | Imagen BGR de la cancha con las mismas dimensiones `field_h × field_w`. |
| `alpha`      | Peso del mapa de calor en la mezcla. `0.0` = solo imagen base, `1.0` = solo mapa. |

Proceso interno:
1. Obtiene el mapa normalizado con `get_normalized()`.
2. Aplica la paleta de colores **JET** con `cv2.applyColorMap`.
3. Combina ambas imágenes con `cv2.addWeighted(field_base, 1-alpha, heatmap_bgr, alpha, 0)`.

#### `reset()`

Reinicia el acumulador de posiciones a cero.

```python
tracker.reset()
```

## Flujo de uso

```python
from futbot_heatmap import HeatmapTracker

tracker = HeatmapTracker(field_h=400, field_w=600, sigma=30.0)

for frame_idx, frame_path in enumerate(frame_paths):
    # field_positions: lista de (label, fx, fy) en coordenadas de cancha
    tracker.update(field_positions)

    # Renderizar sobre la imagen base de la cancha
    panel_field = tracker.render(field_image, alpha=0.35)
```

## Flujo del programa

1. Inicializar `HeatmapTracker` con las dimensiones del mapa de cancha.
2. Por cada frame, llamar a `update()` con las posiciones actuales de los robots.
3. Llamar a `render()` para obtener la imagen de cancha con el heatmap superpuesto.
4. Incorporar el panel resultante al video de salida.
5. Opcionalmente llamar a `reset()` para reiniciar el acumulador entre partidos.

## Integración con el pipeline principal

En `main.py`, `HeatmapTracker` se usa dentro del bucle de procesamiento de frames:

1. **SAM tracking** detecta robots y pelota; se extraen sus centroides en píxeles.
2. **Homografía** transforma los centroides a coordenadas de cancha (`field_positions`).
3. `tracker.update(field_positions)` acumula las posiciones del frame actual.
4. `tracker.render(field_image, alpha=0.35)` genera el panel del mapa de cancha con el heatmap superpuesto.
5. El panel se combina con el panel SAM para formar el video de salida de dos paneles.

## Dependencias

| Librería | Uso |
|----------|-----|
| `numpy`  | Matriz de acumulación `_counts` y operaciones numéricas. |
| `opencv` | Suavizado gaussiano, normalización, paleta de color JET y mezcla de imágenes. |