from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sam3.model_builder import build_sam3_video_predictor


@dataclass(frozen=True)
class TrackingClass:
    prompt: str
    label: str
    bbox: list[float] | None = None


def load_tracking_classes(config_path: str | Path) -> list[TrackingClass]:
    """Load tracking classes from a JSON config file.

    The JSON should contain:
        - image_width, image_height: original frame dimensions (for bbox normalization)
        - tracking_classes: list of {prompt, label, bbox?} where bbox is [x, y, w, h] in pixels

    Args:
        config_path: Path to the JSON config file.

    Returns:
        List of TrackingClass instances with normalized bboxes.
    """
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = json.load(f)

    img_w = config["image_width"]
    img_h = config["image_height"]

    classes = []
    for entry in config["tracking_classes"]:
        bbox = entry.get("bbox")
        if bbox is not None:
            x, y, w, h = bbox
            bbox = [x / img_w, y / img_h, w / img_w, h / img_h]
        classes.append(
            TrackingClass(
                prompt=entry["prompt"],
                label=entry["label"],
                bbox=bbox,
            )
        )

    return classes


@dataclass(frozen=True)
class TrackingConfig:
    # target_fps: float = 20.0 #TODO: remove
    offload_video_to_cpu: bool = True
    mask_alpha: float = 0.5
    font_scale: float = 0.7
    font_thickness: int = 2


@dataclass
class FrameResult:
    frame_idx: int
    entries: list[tuple[int, np.ndarray]] = field(default_factory=list)

    def class_indices(self) -> list[int]:
        return [ci for ci, _ in self.entries]

    def masks(self) -> list[np.ndarray]:
        return [m for _, m in self.entries]


@dataclass
class TrackingResult:
    frames: dict[int, FrameResult] = field(default_factory=dict)
    num_frames_processed: int = 0

    def sorted_frame_indices(self) -> list[int]:
        return sorted(self.frames.keys())


def bbox_center(bbox_norm: list[float], h: int, w: int) -> tuple[float, float]:
    """Return (cy, cx) pixel coords of a normalized [x, y, w, h] bbox."""
    x, y, bw, bh = bbox_norm
    cx = (x + bw / 2) * w
    cy = (y + bh / 2) * h
    return cy, cx


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Return (cy, cx) of the mask centroid, or None if mask is empty."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def _assign_objects_to_classes(
    per_obj_results: dict[Any, list[tuple[int, np.ndarray]]],
    group: list[tuple[int, TrackingClass]],
    tracking_classes: list[TrackingClass],
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Assign detected objects to sub-classes by bbox proximity on frame 0.

    Returns a dict of frame_idx -> list of (class_idx, mask).
    """
    assigned: dict[int, list[tuple[int, np.ndarray]]] = {}
    sub_classes_with_bbox = [(ci, c) for ci, c in group if c.bbox is not None]

    if len(sub_classes_with_bbox) > 1:
        # Multiple sub-classes: assign each object to nearest bbox
        sample_mask = next(iter(per_obj_results.values()))[0][1]
        mh, mw = sample_mask.shape

        # Get bbox centers for each sub-class
        bbox_centers = {}
        for ci, c in sub_classes_with_bbox:
            bbox_centers[ci] = bbox_center(c.bbox, mh, mw)

        # For each object, find its frame-0 centroid and assign to nearest bbox
        for obj_id, frames_data in per_obj_results.items():
            frame0_mask = None
            for fidx, mask in frames_data:
                if fidx == 0:
                    frame0_mask = mask
                    break
            if frame0_mask is None:
                continue

            centroid = mask_centroid(frame0_mask)
            if centroid is None:
                continue

            # Find nearest sub-class bbox center
            best_ci = None
            best_dist = float("inf")
            for ci, (bcy, bcx) in bbox_centers.items():
                dist = np.sqrt((centroid[0] - bcy) ** 2 + (centroid[1] - bcx) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_ci = ci

            label = tracking_classes[best_ci].label
            print(f"  obj_id={obj_id} -> class '{label}' (dist={best_dist:.1f}px)")
            for fidx, mask in frames_data:
                if fidx not in assigned:
                    assigned[fidx] = []
                assigned[fidx].append((best_ci, mask))
    else:
        # Single sub-class or no bbox: assign all objects to this class
        class_idx = group[0][0]
        for obj_id, frames_data in per_obj_results.items():
            for frame_idx, mask in frames_data:
                if frame_idx not in assigned:
                    assigned[frame_idx] = []
                assigned[frame_idx].append((class_idx, mask))

    return assigned


class SAMTracker:
    """Multi-object video tracker using SAM3."""

    def __init__(
        self,
        tracking_classes: list[TrackingClass],
        config: TrackingConfig | None = None,
    ) -> None:
        self.tracking_classes = tracking_classes
        self.config = config or TrackingConfig()
        self.labels = [c.label for c in tracking_classes]
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = build_sam3_video_predictor()
        return self._predictor

    def track(self, frames_dir: str) -> TrackingResult:
        """Run tracking on pre-extracted frames and return results.

        Args:
            frames_dir: Path to directory containing extracted frame images.

        Returns:
            TrackingResult with per-frame mask assignments.
        """
        all_results: dict[int, list[tuple[int, np.ndarray]]] = {}

        # Group tracking classes by prompt so shared prompts run only once
        prompt_groups: dict[str, list[tuple[int, TrackingClass]]] = defaultdict(list)
        for class_idx, cls in enumerate(self.tracking_classes):
            prompt_groups[cls.prompt].append((class_idx, cls))

        for prompt, group in prompt_groups.items():
            print(f"Running prompt: '{prompt}' ({len(group)} sub-classes)")

            # Start session
            response = self.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=frames_dir,
                    offload_video_to_cpu=self.config.offload_video_to_cpu,
                )
            )
            session_id = response["session_id"]

            # Add prompt (use bbox from first sub-class that has one, if any)
            add_prompt_request = dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=prompt,
            )
            first_bbox = next((c.bbox for _, c in group if c.bbox is not None), None)
            if first_bbox is not None:
                add_prompt_request["bounding_boxes"] = [first_bbox]
                add_prompt_request["bounding_box_labels"] = [1]
            self.predictor.handle_request(request=add_prompt_request)

            # Propagate and collect per-object results
            per_obj_results: dict[Any, list[tuple[int, np.ndarray]]] = {}
            for result in self.predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                frame_idx = result["frame_index"]
                outputs = result["outputs"]
                obj_ids = outputs["out_obj_ids"]
                masks = outputs["out_binary_masks"]

                for obj_id, mask in zip(obj_ids, masks):
                    if obj_id not in per_obj_results:
                        per_obj_results[obj_id] = []
                    per_obj_results[obj_id].append((frame_idx, mask))

            # Assign objects to sub-classes
            assigned = _assign_objects_to_classes(
                per_obj_results, group, self.tracking_classes
            )
            for fidx, entries in assigned.items():
                if fidx not in all_results:
                    all_results[fidx] = []
                all_results[fidx].extend(entries)

            # Close session
            self.predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
            print(f"Tracking session closed.")

        # Build TrackingResult
        tracking_result = TrackingResult(num_frames_processed=len(all_results))
        for fidx in sorted(all_results.keys()):
            tracking_result.frames[fidx] = FrameResult(
                frame_idx=fidx,
                entries=all_results[fidx],
            )

        return tracking_result
