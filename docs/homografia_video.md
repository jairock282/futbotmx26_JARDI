# Homografía móvil para vista cenital

Este documento explica el método usado para convertir frames de un video de fútbol de robots a una vista cenital de la cancha, incluso cuando la cámara tiene movimientos suaves.

El objetivo es obtener, para cada frame del video, una matriz:

```text
H_t: frame_t -> cancha.png
```

Con esa homografía se pueden proyectar puntos detectados en el frame original, por ejemplo centros de robots o de la pelota, hacia coordenadas de la imagen cenital `cancha.png`.

## Idea general

Primero se calibra manualmente un frame de referencia, normalmente el primer frame:

```text
H_ref: frame_0001 -> cancha.png
```

Esa homografía se obtiene con puntos correspondientes entre `frame_0001.jpg` y `cancha.png`. En este proyecto esos puntos viven en:

```text
homography_points.json
```

El problema es que la cámara no está completamente fija. Si la cámara se mueve, aunque sea poco, `H_ref` ya no sirve perfectamente para todos los frames.

Para corregirlo, estimamos el movimiento de cámara entre frames consecutivos:

```text
A_2: frame_0002 -> frame_0001
A_3: frame_0003 -> frame_0002
A_4: frame_0004 -> frame_0003
...
```

Luego acumulamos esas transformaciones:

```text
T_t: frame_t -> frame_0001
```

Y componemos:

```text
H_t = H_ref @ T_t
```

Así obtenemos una homografía distinta para cada frame:

```text
frame_t -> frame_0001 -> cancha.png
```

## Por qué usar tracking secuencial

Una opción sería alinear cada frame directamente contra `frame_0001`:

```text
frame_t -> frame_0001
```

Pero esto falla más fácilmente cuando la cámara se mueve y partes de la cancha salen de la imagen. Por ejemplo, si una esquina visible en el frame de referencia ya no aparece en un frame posterior, el algoritmo intentaría alinear información que ya no existe.

Por eso se usa modo secuencial:

```text
frame_0005 -> frame_0004 -> frame_0003 -> frame_0002 -> frame_0001
```

Entre frames consecutivos el movimiento suele ser pequeño y hay mucho más traslape visual, así que la alineación es más estable.

## Qué transformación se estima entre frames

Para el movimiento entre frames se usa normalmente un modelo afín:

```text
A_t =
[ a b tx
  c d ty
  0 0 1  ]
```

Este modelo permite:

- traslación,
- rotación,
- escala leve,
- shear leve.

Aunque el movimiento entre frames se estime con una transformación afín, el resultado final sigue siendo una homografía porque se compone con `H_ref`:

```text
H_t = H_ref @ T_t
```

También se puede usar una homografía completa entre frames, pero para movimientos suaves de cámara el modelo afín suele ser más estable.

## Cómo se alinean los frames

La alineación se hace con ECC, disponible en OpenCV como:

```python
cv2.findTransformECC(...)
```

ECC busca la transformación geométrica que maximiza la similitud entre dos imágenes.

No se usa el frame RGB completo, porque hay objetos que se mueven independientemente de la cámara:

- robots,
- manos,
- sombras,
- personas,
- celulares,
- pelota.

En vez de eso se construye una imagen de alineación donde se intenta conservar principalmente la estructura fija de la cancha:

- líneas blancas,
- bordes del campo,
- marcas estables sobre el piso.

## Procesamiento para extraer líneas de cancha

La función principal está en:

```text
homography_tracking.py
```

Función:

```python
make_line_alignment(frame_bgr)
```

El flujo es:

```text
frame BGR
 -> HSV
 -> máscara de blanco
 -> máscara de verde/cancha
 -> blanco AND zona de cancha
 -> limpieza morfológica
 -> engrosar líneas
 -> blur suave
 -> imagen float para ECC
```

Código base:

```python
def make_line_alignment(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(
        hsv,
        np.array([0, 0, 130], dtype=np.uint8),
        np.array([180, 95, 255], dtype=np.uint8),
    )

    green = cv2.inRange(
        hsv,
        np.array([35, 30, 35], dtype=np.uint8),
        np.array([105, 255, 255], dtype=np.uint8),
    )

    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, odd_kernel(3))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, odd_kernel(17))
    green = cv2.dilate(green, odd_kernel(23))

    lines = cv2.bitwise_and(white, green)
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, odd_kernel(5))
    lines = cv2.dilate(lines, odd_kernel(3))
    lines = cv2.GaussianBlur(lines, (5, 5), 0)

    return lines.astype(np.float32) / 255.0, lines
```

### Máscara de blanco

```python
white = cv2.inRange(
    hsv,
    np.array([0, 0, 130], dtype=np.uint8),
    np.array([180, 95, 255], dtype=np.uint8),
)
```

Busca píxeles claros y con poca saturación.

Esto captura líneas blancas, aunque también puede capturar partes blancas de robots, ropa u otros objetos.

### Máscara de cancha

```python
green = cv2.inRange(
    hsv,
    np.array([35, 30, 35], dtype=np.uint8),
    np.array([105, 255, 255], dtype=np.uint8),
)
```

Busca la zona verde/turquesa de la cancha.

Esta máscara sirve para saber qué regiones pertenecen al campo.

### Expansión de la cancha

```python
green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, odd_kernel(17))
green = cv2.dilate(green, odd_kernel(23))
```

Se limpia y expande la zona de cancha para incluir líneas blancas que están encima o justo junto al campo verde.

### Intersección blanco-cancha

```python
lines = cv2.bitwise_and(white, green)
```

Esto conserva píxeles blancos que están dentro o cerca de la cancha.

Ayuda a descartar blancos externos, como paredes, playeras, celulares o reflejos fuera del campo.

### Limpieza morfológica

```python
white = cv2.morphologyEx(white, cv2.MORPH_OPEN, odd_kernel(3))
lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, odd_kernel(5))
```

`MORPH_OPEN` elimina ruido pequeño.

`MORPH_CLOSE` conecta segmentos de líneas que quedaron cortados.

### Engrosar y suavizar líneas

```python
lines = cv2.dilate(lines, odd_kernel(3))
lines = cv2.GaussianBlur(lines, (5, 5), 0)
```

Se engrosan las líneas para darle más señal a ECC.

Luego se suaviza la imagen para que la optimización sea menos frágil.

## Máscara válida para ECC

Además de la imagen de alineación, se construye una máscara válida:

```python
valid_mask = cv2.dilate(line_mask, odd_kernel(120))
```

Esto le dice a ECC que se concentre alrededor de las zonas donde hay líneas de cancha.

La máscara no tiene que ser perfecta. Solo necesita contener suficiente estructura fija para que ECC pueda estimar el movimiento de cámara.

## Composición de homografías

Si:

```text
H_ref: frame_0001 -> cancha.png
A_t: frame_t -> frame_{t-1}
T_{t-1}: frame_{t-1} -> frame_0001
```

Entonces:

```text
T_t = T_{t-1} @ A_t
H_t = H_ref @ T_t
```

En código, el estimador mantiene el estado acumulado:

```python
result = estimator.process(frame, frame_name="frame_0005.jpg")
H_t = result.H_frame_to_field
```

## Proyectar robots a cancha

Una vez que existe `H_t`, cualquier punto detectado en el frame original se puede proyectar a la vista cenital:

```python
from futbot_homography.tracking import transform_points_to_field

robot_centers_px = np.array([
    [x_robot, y_robot],
], dtype=np.float32)

robot_centers_field = transform_points_to_field(
    robot_centers_px,
    H_t,
)
```

Si el tracking de robots devuelve el centro de la base del robot en píxeles del frame, esta función devuelve su posición en píxeles de `cancha.png`.

## Archivos principales

### Calibración manual

```text
homography_viewer.py
```

Permite seleccionar puntos correspondientes entre el frame de referencia y `cancha.png`.

Genera/lee:

```text
homography_points.json
```

### Librería reusable

```text
homography_tracking.py
```

Contiene la lógica reusable:

- cargar puntos,
- calcular `H_ref`,
- procesar líneas,
- estimar movimiento frame a frame,
- componer homografías,
- proyectar puntos.

### Script de estimación por carpeta

```text
estimate_video_homographies.py
```

Procesa todos los frames de una carpeta y guarda una homografía por frame.

Ejemplo:

```bash
python scripts/estimate_video_homographies.py \
  --frames-dir data/frames \
  --points homography_points.json \
  --output homographies_IMG_9913.json \
  --csv-output homographies_IMG_9913.csv \
  --write-preview
```

### Ejemplo de uso

```text
example_homography_tracking.py
```

Muestra cómo usar la librería para:

- estimar `H_t`,
- proyectar puntos,
- generar una vista cenital de prueba.

## Salidas

El script puede generar:

```text
homographies_IMG_9913.json
homographies_IMG_9913.csv
homography_previews/
```

Cada registro contiene:

```text
frame
status
ecc_score
ref_to_current
current_to_ref
H_frame_to_field
```

El campo más importante para el gemelo digital es:

```text
H_frame_to_field
```

Esa matriz convierte puntos del frame original hacia la vista cenital.

## Consideraciones prácticas

- La calidad de `H_ref` es crítica. Si los puntos manuales iniciales están mal, todas las homografías posteriores heredan ese error.
- Conviene usar más de 4 puntos para la calibración inicial. Idealmente 6-12 puntos distribuidos por la cancha.
- Los puntos deben estar sobre el plano del piso, no sobre robots, postes altos o paredes.
- ECC puede fallar si hay oclusiones fuertes, blur, cambios grandes de cámara o pocas líneas visibles.
- El modo secuencial reduce fallos por partes de cancha que salen de la imagen, pero puede acumular drift en videos largos.
- Para videos largos puede convenir reenganchar contra un frame clave o recalibrar cada cierto tiempo.

## Resumen

El método separa el problema en dos partes:

```text
1. Perspectiva de la cancha:
   H_ref = frame_0001 -> cancha.png

2. Movimiento suave de cámara:
   T_t = frame_t -> frame_0001
```

La homografía final por frame es:

```text
H_t = H_ref @ T_t
```

Con esto se puede construir una visualización cenital dinámica del partido y proyectar detecciones de robots/pelota sobre `cancha.png`.
