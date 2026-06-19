"""Heatmap tracker for accumulating and visualizing field positions."""
from __future__ import annotations

import cv2
import numpy as np


class HeatmapTracker:
    """Accumulates robot positions and renders a heatmap overlay on the field."""

    def __init__(self, field_h: int, field_w: int, sigma: float = 30.0) -> None:
        self.field_h = field_h
        self.field_w = field_w
        self.sigma = sigma
        self._counts = np.zeros((field_h, field_w), dtype=np.float32)

    def reset(self) -> None:
        self._counts[:] = 0

    def update(self, field_positions: list[tuple[str, float, float]]) -> None:
        """Accumulate one frame of robot positions (skip the ball)."""
        for label, fx, fy in field_positions:
            if label == "ball":
                continue
            ix, iy = int(round(fx)), int(round(fy))
            if 0 <= ix < self.field_w and 0 <= iy < self.field_h:
                self._counts[iy, ix] += 1

    def get_normalized(self) -> np.ndarray:
        """Return smoothed and normalized heatmap (0-255 uint8)."""
        blurred = cv2.GaussianBlur(self._counts, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        scaled = np.sqrt(blurred)
        norm = cv2.normalize(scaled, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)

    def render(self, field_base: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Render the heatmap blended on top of the field image."""
        heatmap_gray = self.get_normalized()
        heatmap_bgr = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        return cv2.addWeighted(field_base, 1.0 - alpha, heatmap_bgr, alpha, 0)
