from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


MOTION_TYPES = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,
}


@dataclass(frozen=True)
class HomographyTrackingConfig:
    motion: str = "affine"
    tracking_mode: str = "sequential"
    ecc_scale: float = 0.5
    iterations: int = 150
    eps: float = 1e-6
    line_mask_dilate: int = 120
    accept_score: float = 0.30


@dataclass(frozen=True)
class ReferenceHomography:
    H_ref: np.ndarray
    inliers: int | None
    total_points: int


@dataclass(frozen=True)
class HomographyFrameResult:
    frame_name: str
    index: int
    status: str
    ecc_score: float | None
    ref_to_current: np.ndarray
    current_to_ref: np.ndarray
    H_frame_to_field: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame_name,
            "index": self.index,
            "status": self.status,
            "ecc_score": self.ecc_score,
            "ref_to_current": self.ref_to_current.tolist(),
            "current_to_ref": self.current_to_ref.tolist(),
            "H_frame_to_field": self.H_frame_to_field.tolist(),
        }


def load_points_config(points_path: str | Path) -> dict[str, Any]:
    with Path(points_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def as_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def load_correspondences(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    frame_points: list[tuple[float, float]] = []
    field_points: list[tuple[float, float]] = []

    for item in config.get("points", []):
        if not isinstance(item, dict):
            continue
        frame_point = as_point(item.get("frame"))
        field_point = as_point(item.get("field"))
        if frame_point is None or field_point is None:
            continue
        frame_points.append(frame_point)
        field_points.append(field_point)

    if len(frame_points) < 4:
        raise ValueError("Necesitas al menos 4 pares completos en el JSON de puntos.")

    return (
        np.asarray(frame_points, dtype=np.float32),
        np.asarray(field_points, dtype=np.float32),
    )


def compute_reference_homography(points_path: str | Path) -> ReferenceHomography:
    config = load_points_config(points_path)
    frame_points, field_points = load_correspondences(config)
    method = 0 if len(frame_points) == 4 else cv2.RANSAC
    homography, mask = cv2.findHomography(frame_points, field_points, method, 3.0)
    if homography is None:
        raise RuntimeError("No se pudo calcular H_ref con los puntos manuales.")

    inliers = int(mask.ravel().sum()) if mask is not None else None
    return ReferenceHomography(
        H_ref=homography.astype(np.float64),
        inliers=inliers,
        total_points=len(frame_points),
    )


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No pude leer la imagen: {path}")
    return image


def odd_kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return np.ones((size, size), np.uint8)


def make_line_alignment(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def resize_gray_float(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image.astype(np.float32)
    height, width = image.shape[:2]
    return cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)


def resize_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return mask
    height, width = mask.shape[:2]
    return cv2.resize(
        mask,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_NEAREST,
    )


def as_homogeneous(warp: np.ndarray) -> np.ndarray:
    if warp.shape == (3, 3):
        return warp.astype(np.float64)
    homogeneous = np.eye(3, 3, dtype=np.float64)
    homogeneous[:2, :] = warp.astype(np.float64)
    return homogeneous


def to_motion_shape(warp_h: np.ndarray, motion: str) -> np.ndarray:
    if motion == "homography":
        return warp_h.astype(np.float32)
    return warp_h[:2, :].astype(np.float32)


def scale_warp_to_full_resolution(warp_scaled: np.ndarray, scale: float) -> np.ndarray:
    warp_h = as_homogeneous(warp_scaled)
    if scale == 1.0:
        return warp_h

    scale_matrix = np.array(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return np.linalg.inv(scale_matrix) @ warp_h @ scale_matrix


def scale_warp_to_ecc_resolution(warp_full: np.ndarray, scale: float, motion: str) -> np.ndarray:
    if scale == 1.0:
        return to_motion_shape(warp_full, motion)

    scale_matrix = np.array(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    warp_scaled = scale_matrix @ warp_full @ np.linalg.inv(scale_matrix)
    return to_motion_shape(warp_scaled, motion)


def estimate_template_to_current_warp(
    template_align: np.ndarray,
    current_align: np.ndarray,
    valid_mask: np.ndarray,
    initial_warp_full: np.ndarray,
    config: HomographyTrackingConfig,
) -> tuple[np.ndarray, float]:
    ecc_scale = max(0.05, min(float(config.ecc_scale), 1.0))
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.iterations,
        config.eps,
    )
    template_small = resize_gray_float(template_align, ecc_scale)
    current_small = resize_gray_float(current_align, ecc_scale)
    mask_small = resize_mask(valid_mask, ecc_scale)
    initial_warp_small = scale_warp_to_ecc_resolution(
        initial_warp_full,
        ecc_scale,
        config.motion,
    )

    score, warp_small = cv2.findTransformECC(
        template_small,
        current_small,
        initial_warp_small,
        MOTION_TYPES[config.motion],
        criteria,
        inputMask=mask_small,
        gaussFiltSize=5,
    )
    warp_full = scale_warp_to_full_resolution(warp_small, ecc_scale)
    return warp_full, float(score)


def candidate_initial_warps(previous_warp: np.ndarray) -> list[tuple[str, np.ndarray]]:
    candidates = [
        ("previous", previous_warp),
        ("identity", np.eye(3, 3, dtype=np.float64)),
    ]

    for dx in (20.0, 40.0, -20.0, -40.0):
        guess = np.eye(3, 3, dtype=np.float64)
        guess[0, 2] = dx
        candidates.append((f"translate_x_{dx:g}", guess))

    for dy in (-30.0, 30.0):
        guess = np.eye(3, 3, dtype=np.float64)
        guess[1, 2] = dy
        candidates.append((f"translate_y_{dy:g}", guess))

    return candidates


def estimate_with_retries(
    template_align: np.ndarray,
    current_align: np.ndarray,
    template_lines: np.ndarray,
    base_mask: np.ndarray,
    initial_warp: np.ndarray,
    config: HomographyTrackingConfig,
) -> tuple[np.ndarray, float, str]:
    wide_size = int(round(config.line_mask_dilate * 2.0))
    wide_mask = cv2.dilate(template_lines, odd_kernel(wide_size))

    initial_warps = candidate_initial_warps(initial_warp)
    attempts: list[tuple[str, np.ndarray, str, np.ndarray]] = [
        ("base", base_mask, initial_warps[0][0], initial_warps[0][1]),
        (f"dilate_{wide_size}", wide_mask, initial_warps[0][0], initial_warps[0][1]),
        ("base", base_mask, initial_warps[1][0], initial_warps[1][1]),
        (f"dilate_{wide_size}", wide_mask, initial_warps[1][0], initial_warps[1][1]),
    ]
    for init_name, initial in initial_warps[2:]:
        attempts.append(("base", base_mask, init_name, initial))
        attempts.append((f"dilate_{wide_size}", wide_mask, init_name, initial))

    best: tuple[np.ndarray, float, str] | None = None
    errors: list[str] = []

    for mask_name, mask, init_name, initial in attempts:
        try:
            warp, score = estimate_template_to_current_warp(
                template_align,
                current_align,
                mask,
                initial,
                config,
            )
        except cv2.error as exc:
            errors.append(f"{mask_name}/{init_name}: {exc.code}")
            continue

        attempt_name = f"{mask_name}/{init_name}"
        if best is None or score > best[1]:
            best = (warp, score, attempt_name)
        if score >= config.accept_score:
            return warp, score, attempt_name

    if best is None:
        raise cv2.error(f"ECC no convergio en ningun reintento: {', '.join(errors)}")
    return best


class ConsecutiveHomographyEstimator:
    def __init__(
        self,
        reference_frame_bgr: np.ndarray,
        H_ref: np.ndarray,
        config: HomographyTrackingConfig | None = None,
        reference_name: str = "reference",
    ) -> None:
        self.config = config or HomographyTrackingConfig()
        if self.config.motion not in MOTION_TYPES:
            raise ValueError(f"motion invalido: {self.config.motion}")
        if self.config.tracking_mode not in {"sequential", "reference"}:
            raise ValueError(f"tracking_mode invalido: {self.config.tracking_mode}")

        self.reference_name = reference_name
        self.reference_shape = reference_frame_bgr.shape
        self.H_ref = H_ref.astype(np.float64)
        self.ref_align, self.ref_lines = make_line_alignment(reference_frame_bgr)
        self.ref_mask = cv2.dilate(self.ref_lines, odd_kernel(self.config.line_mask_dilate))

        if int((self.ref_mask > 0).sum()) < 500:
            raise RuntimeError("La mascara de lineas del frame de referencia quedo demasiado vacia.")

        self.last_ref_to_current = np.eye(3, 3, dtype=np.float64)
        self.last_step_warp = np.eye(3, 3, dtype=np.float64)
        self.previous_align = self.ref_align
        self.previous_lines = self.ref_lines
        self.previous_mask = self.ref_mask
        self.processed = 0

    def process(
        self,
        frame_bgr: np.ndarray,
        frame_name: str | None = None,
        is_reference: bool = False,
    ) -> HomographyFrameResult:
        if frame_bgr.shape != self.reference_shape:
            raise ValueError(
                f"El frame tiene shape {frame_bgr.shape}, pero la referencia tiene {self.reference_shape}."
            )

        self.processed += 1
        name = frame_name or f"frame_{self.processed:04d}"

        if is_reference:
            self.last_ref_to_current = np.eye(3, 3, dtype=np.float64)
            self.last_step_warp = np.eye(3, 3, dtype=np.float64)
            self.previous_align = self.ref_align
            self.previous_lines = self.ref_lines
            self.previous_mask = self.ref_mask
            return self._make_result(name, "reference", 1.0, self.last_ref_to_current)

        current_align, current_lines = make_line_alignment(frame_bgr)
        try:
            if self.config.tracking_mode == "sequential":
                template_align = self.previous_align
                template_lines = self.previous_lines
                template_mask = self.previous_mask
                initial_warp = self.last_step_warp
            else:
                template_align = self.ref_align
                template_lines = self.ref_lines
                template_mask = self.ref_mask
                initial_warp = self.last_ref_to_current

            template_to_current, ecc_score, attempt_name = estimate_with_retries(
                template_align,
                current_align,
                template_lines,
                template_mask,
                initial_warp,
                self.config,
            )

            if self.config.tracking_mode == "sequential":
                ref_to_current = template_to_current @ self.last_ref_to_current
                self.last_step_warp = template_to_current
            else:
                ref_to_current = template_to_current

            self.last_ref_to_current = ref_to_current
            self.previous_align = current_align
            self.previous_lines = current_lines
            self.previous_mask = cv2.dilate(
                current_lines,
                odd_kernel(self.config.line_mask_dilate),
            )
            status = f"ok:{self.config.tracking_mode}:{attempt_name}"
            return self._make_result(name, status, ecc_score, ref_to_current)
        except cv2.error:
            return self._make_result(name, "fallback_previous", None, self.last_ref_to_current)

    def _make_result(
        self,
        frame_name: str,
        status: str,
        ecc_score: float | None,
        ref_to_current: np.ndarray,
    ) -> HomographyFrameResult:
        current_to_ref = np.linalg.inv(ref_to_current)
        H_frame_to_field = self.H_ref @ current_to_ref
        H_frame_to_field /= H_frame_to_field[2, 2]
        return HomographyFrameResult(
            frame_name=frame_name,
            index=self.processed,
            status=status,
            ecc_score=ecc_score,
            ref_to_current=ref_to_current.copy(),
            current_to_ref=current_to_ref,
            H_frame_to_field=H_frame_to_field,
        )


def sorted_frame_paths(frames_dir: str | Path, pattern: str = "*.jpg") -> list[Path]:
    paths = sorted(Path(frames_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No encontre frames {pattern} en: {frames_dir}")
    return paths


def estimate_homographies_for_frame_paths(
    frame_paths: Iterable[str | Path],
    reference_frame_path: str | Path,
    H_ref: np.ndarray,
    config: HomographyTrackingConfig | None = None,
) -> list[HomographyFrameResult]:
    paths = [Path(path) for path in frame_paths]
    reference_path = Path(reference_frame_path)
    reference_frame = read_image(reference_path)
    estimator = ConsecutiveHomographyEstimator(
        reference_frame,
        H_ref,
        config=config,
        reference_name=reference_path.name,
    )

    results: list[HomographyFrameResult] = []
    for path in paths:
        frame = read_image(path)
        result = estimator.process(
            frame,
            frame_name=path.name,
            is_reference=path.resolve() == reference_path.resolve(),
        )
        results.append(result)
    return results


def warp_frame_to_field(
    frame_bgr: np.ndarray,
    H_frame_to_field: np.ndarray,
    field_size: tuple[int, int],
) -> np.ndarray:
    width, height = field_size
    return cv2.warpPerspective(frame_bgr, H_frame_to_field, (width, height))


def transform_points_to_field(
    points_xy: np.ndarray,
    H_frame_to_field: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, H_frame_to_field)
    return transformed.reshape(-1, 2)
