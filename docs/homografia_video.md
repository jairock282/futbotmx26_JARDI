# Homografia movil para vista cenital

Este documento explica el metodo usado para convertir frames de un video de futbol de robots a una vista cenital de la cancha, incluso cuando la camara tiene movimientos suaves.

El objetivo es obtener, para cada frame del video, una matriz:

```text
H_t: frame_t -> cancha.png
```

Con esa homografia se pueden proyectar puntos detectados en el frame original, por ejemplo centros de robots o de la pelota, hacia coordenadas de la imagen cenital `cancha.png`.

## Idea general

Primero se calibra manualmente un frame de referencia, normalmente el primer frame:

```text
H_ref: frame_0001 -> cancha.png
```

Esa homografia se obtiene con puntos correspondientes entre `frame_0001.jpg` y `cancha.png`. En este proyecto esos puntos viven en:

```text
homography_points.json
```

El problema es que la camara no esta completamente fija. Si la camara se mueve, aunque sea poco, `H_ref` ya no sirve perfectamente para todos los frames.

Para corregirlo, estimamos el movimiento de camara entre frames consecutivos:

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

Asi obtenemos una homografia distinta para cada frame:

```text
frame_t -> frame_0001 -> cancha.png
```

## Por que usar tracking secuencial

Una opcion seria alinear cada frame directamente contra `frame_0001`:

```text
frame_t -> frame_0001
```

Pero esto falla mas facilmente cuando la camara se mueve y partes de la cancha salen de la imagen. Por ejemplo, si una esquina visible en el frame de referencia ya no aparece en un frame posterior, el algoritmo intentaria alinear informacion que ya no existe.

Por eso se usa modo secuencial:

```text
frame_0005 -> frame_0004 -> frame_0003 -> frame_0002 -> frame_0001
```

Entre frames consecutivos el movimiento suele ser pequeno y hay mucho mas traslape visual, asi que la alineacion es mas estable.

## Que transformacion se estima entre frames

Para el movimiento entre frames se usa normalmente un modelo afin:

```text
A_t =
[ a b tx
  c d ty
  0 0 1  ]
```

Este modelo permite:

- traslacion,
- rotacion,
- escala leve,
- shear leve.

Aunque el movimiento entre frames se estime con una transformacion afin, el resultado final sigue siendo una homografia porque se compone con `H_ref`:

```text
H_t = H_ref @ T_t
```

Tambien se puede usar una homografia completa entre frames, pero para movimientos suaves de camara el modelo afin suele ser mas estable.

## Como se alinean los frames

La alineacion se hace con ECC, disponible en OpenCV como:

```python
cv2.findTransformECC(...)
```

ECC busca la transformacion geometrica que maximiza la similitud entre dos imagenes.

No se usa el frame RGB completo, porque hay objetos que se mueven independientemente de la camara:

- robots,
- manos,
- sombras,
- personas,
- celulares,
- pelota.

En vez de eso se construye una imagen de alineacion donde se intenta conservar principalmente la estructura fija de la cancha:

- lineas blancas,
- bordes del campo,
- marcas estables sobre el piso.

## Procesamiento para extraer lineas de cancha

La funcion principal esta en:

```text
homography_tracking.py
```

Funcion:

```python
make_line_alignment(frame_bgr)
```

El flujo es:

```text
frame BGR
 -> HSV
 -> mascara de blanco
 -> mascara de verde/cancha
 -> blanco AND zona de cancha
 -> limpieza morfologica
 -> engrosar lineas
 -> blur suave
 -> imagen float para ECC
```

Codigo base:

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

### Mascara de blanco

```python
white = cv2.inRange(
    hsv,
    np.array([0, 0, 130], dtype=np.uint8),
    np.array([180, 95, 255], dtype=np.uint8),
)
```

Busca pixeles claros y con poca saturacion.

Esto captura lineas blancas, aunque tambien puede capturar partes blancas de robots, ropa u otros objetos.

### Mascara de cancha

```python
green = cv2.inRange(
    hsv,
    np.array([35, 30, 35], dtype=np.uint8),
    np.array([105, 255, 255], dtype=np.uint8),
)
```

Busca la zona verde/turquesa de la cancha.

Esta mascara sirve para saber que regiones pertenecen al campo.

### Expansion de la cancha

```python
green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, odd_kernel(17))
green = cv2.dilate(green, odd_kernel(23))
```

Se limpia y expande la zona de cancha para incluir lineas blancas que estan encima o justo junto al campo verde.

### Interseccion blanco-cancha

```python
lines = cv2.bitwise_and(white, green)
```

Esto conserva pixeles blancos que estan dentro o cerca de la cancha.

Ayuda a descartar blancos externos, como paredes, playeras, celulares o reflejos fuera del campo.

### Limpieza morfologica

```python
white = cv2.morphologyEx(white, cv2.MORPH_OPEN, odd_kernel(3))
lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, odd_kernel(5))
```

`MORPH_OPEN` elimina ruido pequeno.

`MORPH_CLOSE` conecta segmentos de lineas que quedaron cortados.

### Engrosar y suavizar lineas

```python
lines = cv2.dilate(lines, odd_kernel(3))
lines = cv2.GaussianBlur(lines, (5, 5), 0)
```

Se engrosan las lineas para darle mas senal a ECC.

Luego se suaviza la imagen para que la optimizacion sea menos fragil.

## Mascara valida para ECC

Ademas de la imagen de alineacion, se construye una mascara valida:

```python
valid_mask = cv2.dilate(line_mask, odd_kernel(120))
```

Esto le dice a ECC que se concentre alrededor de las zonas donde hay lineas de cancha.

La mascara no tiene que ser perfecta. Solo necesita contener suficiente estructura fija para que ECC pueda estimar el movimiento de camara.

## Composicion de homografias

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

En codigo, el estimador mantiene el estado acumulado:

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

Si el tracking de robots devuelve el centro de la base del robot en pixeles del frame, esta funcion devuelve su posicion en pixeles de `cancha.png`.

## Archivos principales

### Calibracion manual

```text
homography_viewer.py
```

Permite seleccionar puntos correspondientes entre el frame de referencia y `cancha.png`.

Genera/lee:

```text
homography_points.json
```

### Libreria reusable

```text
homography_tracking.py
```

Contiene la logica reusable:

- cargar puntos,
- calcular `H_ref`,
- procesar lineas,
- estimar movimiento frame a frame,
- componer homografias,
- proyectar puntos.

### Script de estimacion por carpeta

```text
estimate_video_homographies.py
```

Procesa todos los frames de una carpeta y guarda una homografia por frame.

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

Muestra como usar la libreria para:

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

El campo mas importante para el gemelo digital es:

```text
H_frame_to_field
```

Esa matriz convierte puntos del frame original hacia la vista cenital.

## Consideraciones practicas

- La calidad de `H_ref` es critica. Si los puntos manuales iniciales estan mal, todas las homografias posteriores heredan ese error.
- Conviene usar mas de 4 puntos para la calibracion inicial. Idealmente 6-12 puntos distribuidos por la cancha.
- Los puntos deben estar sobre el plano del piso, no sobre robots, postes altos o paredes.
- ECC puede fallar si hay oclusiones fuertes, blur, cambios grandes de camara o pocas lineas visibles.
- El modo secuencial reduce fallos por partes de cancha que salen de la imagen, pero puede acumular drift en videos largos.
- Para videos largos puede convenir reenganchar contra un frame clave o recalibrar cada cierto tiempo.

## Resumen

El metodo separa el problema en dos partes:

```text
1. Perspectiva de la cancha:
   H_ref = frame_0001 -> cancha.png

2. Movimiento suave de camara:
   T_t = frame_t -> frame_0001
```

La homografia final por frame es:

```text
H_t = H_ref @ T_t
```

Con esto se puede construir una visualizacion cenital dinamica del partido y proyectar detecciones de robots/pelota sobre `cancha.png`.
