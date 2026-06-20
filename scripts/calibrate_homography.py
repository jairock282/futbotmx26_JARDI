#!/usr/bin/env python3
"""
Interactive homography viewer for mapping points from frame_01.jpg to cancha.png.

Edit homography_points.json with corresponding pixel coordinates in both images,
then run:

    conda run -n futbol python homography_viewer.py

Controls:
    move mouse over the left image  -> show transformed point on the right image
    left click                       -> print image coordinates in the terminal
    c                                -> start manual calibration
    r                               -> reload homography_points.json
    q or Esc                         -> quit

Calibration controls:
    click left image, then right image -> add one correspondence pair
    u                                  -> undo last complete pair
    s                                  -> save points
    Enter                              -> save and return to viewer
    q or Esc                            -> return to viewer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit(
        "No pude importar OpenCV (cv2). Instala OpenCV en el entorno futbol, por ejemplo:\n"
        "  conda run -n futbol python -m pip install opencv-python\n"
    ) from exc


WINDOW_NAME = "Homografia: frame -> cancha"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualiza una homografia interactiva entre frame_01.jpg y cancha.png."
    )
    parser.add_argument(
        "--points",
        default="homography_points.json",
        help="Archivo JSON con pares de puntos correspondientes.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1700,
        help="Ancho maximo de la ventana combinada.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=900,
        help="Alto maximo de la ventana combinada.",
    )
    parser.add_argument(
        "--field-display-scale",
        type=float,
        default=3.0,
        help="Multiplicador visual para la imagen cancha.png respecto al frame.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Valida imagenes/puntos y termina sin abrir la ventana.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Abre primero la etapa interactiva para capturar puntos.",
    )
    return parser.parse_args()


def load_config(points_path: Path) -> dict[str, Any]:
    with points_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(points_path: Path, config: dict[str, Any]) -> None:
    with points_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=True)
        f.write("\n")


def resolve_image_path(points_path: Path, image_name: str) -> Path:
    image_path = Path(image_name)
    if image_path.is_absolute():
        return image_path
    return points_path.parent / image_path


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No pude leer la imagen: {path}")
    return image


def as_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def load_correspondences(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame_points: list[tuple[float, float]] = []
    field_points: list[tuple[float, float]] = []
    names: list[str] = []

    for index, item in enumerate(config.get("points", []), start=1):
        if not isinstance(item, dict):
            continue
        frame_point = as_point(item.get("frame"))
        field_point = as_point(item.get("field"))
        if frame_point is None or field_point is None:
            continue
        frame_points.append(frame_point)
        field_points.append(field_point)
        names.append(str(item.get("name", f"punto_{index}")))

    return (
        np.asarray(frame_points, dtype=np.float32),
        np.asarray(field_points, dtype=np.float32),
        names,
    )


def compute_homography(
    frame_points: np.ndarray, field_points: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(frame_points) < 4:
        return None, None

    method = 0 if len(frame_points) == 4 else cv2.RANSAC
    homography, mask = cv2.findHomography(frame_points, field_points, method, 3.0)
    return homography, mask


def fit_side_by_side(
    frame: np.ndarray,
    field: np.ndarray,
    max_width: int,
    max_height: int,
    field_display_scale: float,
) -> tuple[float, float]:
    frame_h, frame_w = frame.shape[:2]
    field_h, field_w = field.shape[:2]
    field_display_scale = max(field_display_scale, 0.1)

    scale = min(
        max_width / (frame_w + field_w * field_display_scale),
        max_height / max(frame_h, field_h * field_display_scale),
        1.0,
    )
    return scale, scale * field_display_scale


def resize_for_display(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image.copy()
    height, width = image.shape[:2]
    return cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def transform_point(homography: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    src = np.asarray([[[point[0], point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


def append_correspondence(
    config: dict[str, Any],
    frame_point: tuple[float, float],
    field_point: tuple[float, float],
) -> dict[str, Any]:
    points = config.setdefault("points", [])
    complete_count = sum(
        1
        for item in points
        if isinstance(item, dict)
        and as_point(item.get("frame")) is not None
        and as_point(item.get("field")) is not None
    )
    new_point = {
        "name": f"punto_{complete_count + 1}",
        "frame": [round(frame_point[0], 1), round(frame_point[1], 1)],
        "field": [round(field_point[0], 1), round(field_point[1], 1)],
    }

    for index, item in enumerate(points):
        if not isinstance(item, dict):
            points[index] = new_point
            return config
        if as_point(item.get("frame")) is None or as_point(item.get("field")) is None:
            points[index] = {**item, **new_point}
            return config

    points.append(new_point)
    return config


def remove_last_complete_correspondence(config: dict[str, Any]) -> bool:
    points = config.get("points", [])
    for index in range(len(points) - 1, -1, -1):
        item = points[index]
        if (
            isinstance(item, dict)
            and as_point(item.get("frame")) is not None
            and as_point(item.get("field")) is not None
        ):
            del points[index]
            return True
    return False


class HomographyViewer:
    def __init__(
        self,
        points_path: Path,
        max_width: int,
        max_height: int,
        field_display_scale: float,
    ) -> None:
        self.points_path = points_path
        self.max_width = max_width
        self.max_height = max_height
        self.field_display_scale_arg = field_display_scale
        self.mouse_xy: tuple[int, int] | None = None
        self.last_click_text = ""

        self.config: dict[str, Any] = {}
        self.frame: np.ndarray
        self.field: np.ndarray
        self.frame_points = np.empty((0, 2), dtype=np.float32)
        self.field_points = np.empty((0, 2), dtype=np.float32)
        self.point_names: list[str] = []
        self.homography: np.ndarray | None = None
        self.inlier_mask: np.ndarray | None = None
        self.start_calibration = False

        self.frame_scale = 1.0
        self.field_scale = 1.0
        self.frame_display: np.ndarray
        self.field_display: np.ndarray
        self.frame_display_width = 0
        self.canvas: np.ndarray

        self.reload()

    def reload(self) -> None:
        self.config = load_config(self.points_path)

        frame_path = resolve_image_path(self.points_path, self.config["frame_image"])
        field_path = resolve_image_path(self.points_path, self.config["field_image"])
        self.frame = read_image(frame_path)
        self.field = read_image(field_path)

        self.frame_points, self.field_points, self.point_names = load_correspondences(self.config)
        self.homography, self.inlier_mask = compute_homography(
            self.frame_points, self.field_points
        )

        self.frame_scale, self.field_scale = fit_side_by_side(
            self.frame,
            self.field,
            self.max_width,
            self.max_height,
            self.field_display_scale_arg,
        )
        self.frame_display = resize_for_display(self.frame, self.frame_scale)
        self.field_display = resize_for_display(self.field, self.field_scale)
        self.frame_display_width = self.frame_display.shape[1]

        print("")
        print(f"Coordenadas completas: {len(self.frame_points)}")
        if self.homography is None:
            print("Homografia: no calculada. Necesitas al menos 4 pares de puntos.")
        else:
            print("Homografia calculada:")
            print(self.homography)
            if self.inlier_mask is not None:
                inliers = int(self.inlier_mask.ravel().sum())
                print(f"Inliers: {inliers}/{len(self.frame_points)}")

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.mouse_xy = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < self.frame_display_width:
                frame_x = x / self.frame_scale
                frame_y = y / self.frame_scale
                self.last_click_text = f'"frame": [{frame_x:.1f}, {frame_y:.1f}]'
            else:
                field_x = (x - self.frame_display_width) / self.field_scale
                field_y = y / self.field_scale
                self.last_click_text = f'"field": [{field_x:.1f}, {field_y:.1f}]'
            print(self.last_click_text)

    def draw_reference_points(self, frame_disp: np.ndarray, field_disp: np.ndarray) -> None:
        mask = None
        if self.inlier_mask is not None:
            mask = self.inlier_mask.ravel().astype(bool)

        for index, (frame_point, field_point) in enumerate(
            zip(self.frame_points, self.field_points, strict=False)
        ):
            is_inlier = True if mask is None else bool(mask[index])
            color = (0, 220, 255) if is_inlier else (0, 0, 255)

            fx = int(round(frame_point[0] * self.frame_scale))
            fy = int(round(frame_point[1] * self.frame_scale))
            cx = int(round(field_point[0] * self.field_scale))
            cy = int(round(field_point[1] * self.field_scale))

            cv2.circle(frame_disp, (fx, fy), 5, color, -1)
            cv2.circle(field_disp, (cx, cy), 5, color, -1)
            cv2.putText(
                frame_disp,
                str(index + 1),
                (fx + 8, fy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                field_disp,
                str(index + 1),
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    def build_canvas(self) -> np.ndarray:
        frame_disp = self.frame_display.copy()
        field_disp = self.field_display.copy()
        self.draw_reference_points(frame_disp, field_disp)

        status_lines = [
            "q/Esc: salir",
            "r: recargar puntos",
            "c: calibrar",
            "click: imprimir coordenada",
        ]
        if self.last_click_text:
            status_lines.append(f"ultimo click: {self.last_click_text}")
        if self.homography is None:
            status_lines.append("Faltan >= 4 pares en homography_points.json")

        if self.mouse_xy is not None and self.mouse_xy[0] < self.frame_display_width:
            mx, my = self.mouse_xy
            frame_x = mx / self.frame_scale
            frame_y = my / self.frame_scale
            cv2.circle(frame_disp, (mx, my), 7, (0, 0, 255), 2)
            cv2.drawMarker(
                frame_disp,
                (mx, my),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                22,
                2,
                cv2.LINE_AA,
            )

            if self.homography is not None:
                field_x, field_y = transform_point(self.homography, (frame_x, frame_y))
                display_x = int(round(field_x * self.field_scale))
                display_y = int(round(field_y * self.field_scale))
                cv2.circle(field_disp, (display_x, display_y), 8, (0, 0, 255), 2)
                cv2.drawMarker(
                    field_disp,
                    (display_x, display_y),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    26,
                    2,
                    cv2.LINE_AA,
                )
                status_lines.append(
                    f"frame: ({frame_x:.1f}, {frame_y:.1f}) -> cancha: ({field_x:.1f}, {field_y:.1f})"
                )
            else:
                status_lines.append(f"frame: ({frame_x:.1f}, {frame_y:.1f})")
        elif self.mouse_xy is not None:
            mx, my = self.mouse_xy
            field_x = (mx - self.frame_display_width) / self.field_scale
            field_y = my / self.field_scale
            cv2.circle(field_disp, (mx - self.frame_display_width, my), 7, (255, 0, 0), 2)
            cv2.drawMarker(
                field_disp,
                (mx - self.frame_display_width, my),
                (255, 0, 0),
                cv2.MARKER_CROSS,
                22,
                2,
                cv2.LINE_AA,
            )
            status_lines.append(f"cancha: ({field_x:.1f}, {field_y:.1f})")

        height = max(frame_disp.shape[0], field_disp.shape[0])
        width = frame_disp.shape[1] + field_disp.shape[1]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[: frame_disp.shape[0], : frame_disp.shape[1]] = frame_disp
        canvas[: field_disp.shape[0], frame_disp.shape[1] :] = field_disp

        cv2.line(
            canvas,
            (self.frame_display_width, 0),
            (self.frame_display_width, height),
            (255, 255, 255),
            1,
        )
        self.draw_status(canvas, status_lines)
        return canvas

    @staticmethod
    def draw_status(canvas: np.ndarray, lines: list[str]) -> None:
        line_height = 24
        box_height = 12 + line_height * len(lines)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, 8), (760, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        y = 32
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += line_height

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        while True:
            canvas = self.build_canvas()
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                self.start_calibration = True
                break
            if key == ord("r"):
                try:
                    self.reload()
                except Exception as exc:  # noqa: BLE001
                    print(f"No pude recargar los puntos: {exc}", file=sys.stderr)

        cv2.destroyAllWindows()


class CalibrationTool:
    def __init__(
        self,
        points_path: Path,
        max_width: int,
        max_height: int,
        field_display_scale: float,
    ) -> None:
        self.points_path = points_path
        self.max_width = max_width
        self.max_height = max_height
        self.field_display_scale_arg = field_display_scale
        self.pending_frame_point: tuple[float, float] | None = None
        self.mouse_xy: tuple[int, int] | None = None
        self.status = "Click en frame_01.jpg y luego en cancha.png para crear un par."
        self.config: dict[str, Any] = {}
        self.frame: np.ndarray
        self.field: np.ndarray
        self.frame_points = np.empty((0, 2), dtype=np.float32)
        self.field_points = np.empty((0, 2), dtype=np.float32)
        self.point_names: list[str] = []
        self.frame_scale = 1.0
        self.field_scale = 1.0
        self.frame_display: np.ndarray
        self.field_display: np.ndarray
        self.frame_display_width = 0

        self.reload()

    def reload(self) -> None:
        self.config = load_config(self.points_path)
        frame_path = resolve_image_path(self.points_path, self.config["frame_image"])
        field_path = resolve_image_path(self.points_path, self.config["field_image"])
        self.frame = read_image(frame_path)
        self.field = read_image(field_path)
        self.frame_points, self.field_points, self.point_names = load_correspondences(self.config)
        self.frame_scale, self.field_scale = fit_side_by_side(
            self.frame,
            self.field,
            self.max_width,
            self.max_height,
            self.field_display_scale_arg,
        )
        self.frame_display = resize_for_display(self.frame, self.frame_scale)
        self.field_display = resize_for_display(self.field, self.field_scale)
        self.frame_display_width = self.frame_display.shape[1]

    def save(self) -> None:
        save_config(self.points_path, self.config)
        self.reload()
        self.status = f"Guardado: {len(self.frame_points)} pares completos."
        print(self.status)

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.mouse_xy = (x, y)
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if x < self.frame_display_width:
            frame_x = x / self.frame_scale
            frame_y = y / self.frame_scale
            self.pending_frame_point = (frame_x, frame_y)
            self.status = f"Frame seleccionado: ({frame_x:.1f}, {frame_y:.1f}). Ahora click en cancha."
            print(f'"frame": [{frame_x:.1f}, {frame_y:.1f}]')
            return

        field_x = (x - self.frame_display_width) / self.field_scale
        field_y = y / self.field_scale
        print(f'"field": [{field_x:.1f}, {field_y:.1f}]')

        if self.pending_frame_point is None:
            self.status = "Primero selecciona un punto en el frame izquierdo."
            return

        append_correspondence(self.config, self.pending_frame_point, (field_x, field_y))
        self.pending_frame_point = None
        self.save()

    def draw_points(self, frame_disp: np.ndarray, field_disp: np.ndarray) -> None:
        for index, (frame_point, field_point) in enumerate(
            zip(self.frame_points, self.field_points, strict=False)
        ):
            color = (0, 220, 255)
            fx = int(round(frame_point[0] * self.frame_scale))
            fy = int(round(frame_point[1] * self.frame_scale))
            cx = int(round(field_point[0] * self.field_scale))
            cy = int(round(field_point[1] * self.field_scale))
            cv2.circle(frame_disp, (fx, fy), 5, color, -1)
            cv2.circle(field_disp, (cx, cy), 5, color, -1)
            cv2.putText(
                frame_disp,
                str(index + 1),
                (fx + 8, fy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                field_disp,
                str(index + 1),
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )

        if self.pending_frame_point is not None:
            fx = int(round(self.pending_frame_point[0] * self.frame_scale))
            fy = int(round(self.pending_frame_point[1] * self.frame_scale))
            cv2.drawMarker(
                frame_disp,
                (fx, fy),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                28,
                2,
                cv2.LINE_AA,
            )

    def build_canvas(self) -> np.ndarray:
        frame_disp = self.frame_display.copy()
        field_disp = self.field_display.copy()
        self.draw_points(frame_disp, field_disp)

        if self.mouse_xy is not None:
            mx, my = self.mouse_xy
            if mx < self.frame_display_width:
                cv2.drawMarker(
                    frame_disp,
                    (mx, my),
                    (255, 0, 0),
                    cv2.MARKER_CROSS,
                    20,
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.drawMarker(
                    field_disp,
                    (mx - self.frame_display_width, my),
                    (255, 0, 0),
                    cv2.MARKER_CROSS,
                    20,
                    1,
                    cv2.LINE_AA,
                )

        height = max(frame_disp.shape[0], field_disp.shape[0])
        width = frame_disp.shape[1] + field_disp.shape[1]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[: frame_disp.shape[0], : frame_disp.shape[1]] = frame_disp
        canvas[: field_disp.shape[0], frame_disp.shape[1] :] = field_disp
        cv2.line(
            canvas,
            (self.frame_display_width, 0),
            (self.frame_display_width, height),
            (255, 255, 255),
            1,
        )

        lines = [
            "CALIBRACION",
            "click izquierda -> click derecha: agregar par",
            "u: deshacer ultimo par | s: guardar | Enter/q/Esc: volver",
            f"pares completos: {len(self.frame_points)}",
            self.status,
        ]
        HomographyViewer.draw_status(canvas, lines)
        return canvas

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        while True:
            cv2.imshow(WINDOW_NAME, self.build_canvas())
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27, 13):
                if key == 13:
                    self.save()
                break
            if key == ord("s"):
                self.save()
            if key == ord("u"):
                if remove_last_complete_correspondence(self.config):
                    self.save()
                    self.status = "Se elimino el ultimo par completo."
                else:
                    self.status = "No hay pares completos para deshacer."

        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    points_path = Path(args.points).expanduser().resolve()
    if args.check:
        viewer = HomographyViewer(
            points_path,
            args.max_width,
            args.max_height,
            args.field_display_scale,
        )
        return

    if args.calibrate:
        CalibrationTool(
            points_path,
            args.max_width,
            args.max_height,
            args.field_display_scale,
        ).run()

    viewer = HomographyViewer(
        points_path,
        args.max_width,
        args.max_height,
        args.field_display_scale,
    )
    viewer.run()
    while viewer.start_calibration:
        CalibrationTool(
            points_path,
            args.max_width,
            args.max_height,
            args.field_display_scale,
        ).run()
        viewer = HomographyViewer(
            points_path,
            args.max_width,
            args.max_height,
            args.field_display_scale,
        )
        viewer.run()


if __name__ == "__main__":
    main()
