from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .tracker import TrackingConfig, TrackingResult


def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a boolean mask to (h, w) if dimensions don't match."""
    if mask.shape == (h, w):
        return mask
    return cv2.resize(
        mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def save_mask_frames(
    result: TrackingResult,
    output_dir: str | Path,
    labels: list[str],
    colors: dict[str, tuple[int, int, int]],
) -> None:
    """Save color-coded mask PNGs for each frame.

    Args:
        result: TrackingResult from SAMTracker.track().
        output_dir: Directory to save mask frame images.
        labels: Label per class index.
        colors: Dict mapping label -> RGB color tuple.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in result.sorted_frame_indices():
        frame_result = result.frames[frame_idx]
        if not frame_result.entries:
            continue
        H, W = frame_result.entries[0][1].shape
        overlay = np.zeros((H, W, 3), dtype=np.uint8)
        for class_idx, mask in frame_result.entries:
            color = colors[labels[class_idx]]
            overlay[mask] = color
        Image.fromarray(overlay).save(str(output_dir / f"frame_{frame_idx:04d}.png"))

    print(f"Saved {len(result.frames)} mask frames to {output_dir}")


def render_overlay_video(
    result: TrackingResult,
    frames_dir: str | Path,
    output_path: str | Path,
    labels: list[str],
    fps: float,
    colors: dict[str, tuple[int, int, int]],
    config: TrackingConfig | None = None,
) -> None:
    """Render an overlay video with masks and labels on original frames.

    Args:
        result: TrackingResult from SAMTracker.track().
        frames_dir: Directory containing the extracted frame images.
        output_path: Path for the output MP4 video.
        labels: Display label per class index.
        fps: Output video frame rate.
        colors: Dict mapping label -> RGB color tuple.
        config: TrackingConfig for rendering parameters.
    """
    config = config or TrackingConfig()
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(os.listdir(frames_dir))
    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return

    # Get frame dimensions from the first extracted frame
    sample_frame = cv2.imread(str(frames_dir / frame_files[0]))
    h, w = sample_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame_idx in range(len(frame_files)):
        frame_bgr = cv2.imread(str(frames_dir / frame_files[frame_idx]))

        if frame_idx in result.frames:
            frame_result = result.frames[frame_idx]
            overlay = frame_bgr.copy()
            for class_idx, mask in frame_result.entries:
                color_bgr = colors[labels[class_idx]][::-1]
                mask_resized = _resize_mask(mask, h, w)
                overlay[mask_resized] = color_bgr
            frame_bgr = cv2.addWeighted(
                overlay, config.mask_alpha, frame_bgr, 1 - config.mask_alpha, 0
            )

            # Draw labels at mask centroids
            for class_idx, mask in frame_result.entries:
                mask_resized = _resize_mask(mask, h, w)
                ys, xs = np.where(mask_resized)
                if len(ys) == 0:
                    continue
                cx, cy = int(xs.mean()), int(ys.mean())
                label = labels[class_idx]
                color_bgr = colors[label][::-1]
                cv2.putText(
                    frame_bgr, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, config.font_scale,
                    color_bgr, config.font_thickness, cv2.LINE_AA,
                )

        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"Overlay video saved to {output_path}")
