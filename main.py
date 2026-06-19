"""Main pipeline: SAM tracking + homography → field positions + 3-panel video."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from futbot_homography import (
    ConsecutiveHomographyEstimator,
    HomographyTrackingConfig,
    compute_reference_homography,
    read_image,
    sorted_frame_paths,
    transform_points_to_field,
)
from futbot_activity_recognition import GoalDetector, load_goal_zones
from futbot_sam import (
    SAMTracker,
    TrackingConfig,
    load_tracking_classes,
)

# ── Configuration ──────────────────────────────────────────────────────────────
SAMPLE_ID = "IMG_9913"
OUTPUT_FPS = 20

COLORS = {
    "ball": (255, 0, 0),
    "robot_a": (255, 255, 0),
    "robot_b": (0, 100, 255),
}
ICON_PATHS = {
    "ball": Path("assets/ball.png"),
    "robot_a": Path("assets/robot_a.png"),
    "robot_b": Path("assets/robot_b.png"),
}
ICON_SCALE_MAP = {
    "ball": 0.02,
    "robot_a": 0.05,
    "robot_b": 0.05,
}

# Paths
frames_dir = Path(f"data/frames/{SAMPLE_ID}")
sid = SAMPLE_ID.split("_")[-1]
tracking_config_path = Path(f"configs/tracking/tracking_classes_{sid}.json")
calibration_path = Path(f"configs/calibrations/homography_points_{sid}.json")
goal_zones_path = Path(f"configs/roi/goal_zones_{sid}.json")
field_image_path = Path("assets/cancha_1_10.png")
output_dir = Path(f"/mnt/HDD/model_outputs/futbot/pipeline_output/{SAMPLE_ID}")
output_dir.mkdir(parents=True, exist_ok=True)


def extract_mask_centroids(
    frame_result,
    labels: list[str],
    h: int,
    w: int,
) -> list[tuple[str, float, float]]:
    """Extract (label, cx, cy) for each mask in a frame result.

    Uses the bottom-center of each mask as the object position (base of robot).
    Resizes masks to (h, w) if needed.
    """
    centroids = []
    for class_idx, mask in frame_result.entries:
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.max())  # bottom of mask = robot base
        centroids.append((labels[class_idx], cx, cy))
    return centroids


def draw_sam_overlay(
    frame_bgr: np.ndarray,
    frame_result,
    labels: list[str],
    colors: dict[str, tuple[int, int, int]],
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw SAM masks + labels on a frame."""
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()
    for class_idx, mask in frame_result.entries:
        label = labels[class_idx]
        color_bgr = colors[label][::-1]
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        overlay[mask] = color_bgr
    blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

    for class_idx, mask_raw in frame_result.entries:
        label = labels[class_idx]
        if mask_raw.shape != (h, w):
            mask_raw = cv2.resize(
                mask_raw.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        ys, xs = np.where(mask_raw)
        if len(ys) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        color_bgr = colors[label][::-1]
        cv2.putText(blended, label, (cx, cy),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)
    return blended


def load_icons(
    icon_paths: dict[str, Path],
    scale_map: dict[str, float],
) -> dict[str, np.ndarray]:
    """Load RGBA icon images and resize by scale factor."""
    icons = {}
    for label, path in icon_paths.items():
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        new_size = (
            int(round(w * scale_map[label])),
            int(round(h * scale_map[label]))
        )
        icons[label] = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return icons


def _overlay_icon(canvas: np.ndarray, icon_bgra: np.ndarray, cx: int, cy: int) -> None:
    """Composite an RGBA icon onto a BGR canvas, centered at (cx, cy)."""
    ih, iw = icon_bgra.shape[:2]
    x1 = cx - iw // 2
    y1 = cy - ih // 2
    x2, y2 = x1 + iw, y1 + ih

    ch, cw = canvas.shape[:2]
    # Clamp to canvas bounds
    sx1, sy1 = max(0, -x1), max(0, -y1)
    dx1, dy1 = max(0, x1), max(0, y1)
    dx2, dy2 = min(cw, x2), min(ch, y2)
    sx2 = sx1 + (dx2 - dx1)
    sy2 = sy1 + (dy2 - dy1)

    if dx2 <= dx1 or dy2 <= dy1:
        return

    patch = icon_bgra[sy1:sy2, sx1:sx2]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    bgr = patch[:, :, :3].astype(np.float32)
    roi = canvas[dy1:dy2, dx1:dx2].astype(np.float32)
    canvas[dy1:dy2, dx1:dx2] = (bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)


def draw_field_positions(
    field_base: np.ndarray,
    positions: list[tuple[str, float, float]],
    icons: dict[str, np.ndarray],
) -> np.ndarray:
    """Draw object icons on a copy of the field image."""
    field = field_base.copy()
    for label, fx, fy in positions:
        icon = icons.get(label)
        if icon is not None:
            _overlay_icon(field, icon, int(round(fx)), int(round(fy)))
    return field


def main() -> None:
    # ── 1. SAM tracking ────────────────────────────────────────────────────────
    print("=== Step 1: SAM Tracking ===")
    tracking_classes = load_tracking_classes(tracking_config_path)
    labels = [c.label for c in tracking_classes]

    sam_config = TrackingConfig(offload_video_to_cpu=True)
    tracker = SAMTracker(tracking_classes, sam_config)
    sam_result = tracker.track(str(frames_dir))

    # ── 2. Homography setup ────────────────────────────────────────────────────
    print("=== Step 2: Homography ===")
    reference = compute_reference_homography(calibration_path)
    reference_frame_path = sorted_frame_paths(frames_dir)[0]
    reference_frame = read_image(reference_frame_path)

    homography_config = HomographyTrackingConfig(
        tracking_mode="sequential",
        motion="affine",
        ecc_scale=0.5,
    )
    estimator = ConsecutiveHomographyEstimator(
        reference_frame, reference.H_ref,
        config=homography_config,
        reference_name=reference_frame_path.name,
    )

    field_image = read_image(field_image_path)
    field_h, field_w = field_image.shape[:2]
    icons = load_icons(icon_paths=ICON_PATHS, scale_map=ICON_SCALE_MAP)

    goal_zones = load_goal_zones(goal_zones_path)
    goal_detector = GoalDetector(
        goal_zones=goal_zones,
        proximity_threshold=200,
        cooldown_frames=OUTPUT_FPS*2
    )

    # ── 3. Build 3-panel video ─────────────────────────────────────────────────
    print("=== Step 3: Rendering 3-panel video ===")
    frame_paths = sorted_frame_paths(frames_dir)
    sample_frame = read_image(frame_paths[0])
    fh, fw = sample_frame.shape[:2]

    # Scale frames to match field image height (field is the reference)
    frame_scale = field_h / fh
    fw_scaled = int(round(fw * frame_scale))

    panel_w = fw_scaled + field_w
    video_path = str(output_dir / "pipeline_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, OUTPUT_FPS, (panel_w, field_h))

    match_score = {
        "robot_a": 0,
        "robot_b": 0
    }
    for frame_idx, frame_path in enumerate(frame_paths):
        frame_bgr = read_image(frame_path)

        # Homography for this frame
        h_result = estimator.process(
            frame_bgr,
            frame_name=frame_path.name,
            is_reference=(frame_path.name == reference_frame_path.name),
        )

        # Panel 1: SAM overlay (scaled to field height)
        if frame_idx in sam_result.frames:
            fr = sam_result.frames[frame_idx]
            panel_sam = draw_sam_overlay(frame_bgr, fr, labels, COLORS)
            panel_sam = cv2.resize(panel_sam, (fw_scaled, field_h))

            # Extract centroids and transform to field
            centroids = extract_mask_centroids(fr, labels, fh, fw)
            if centroids:
                pixel_pts = np.array(
                    [[cx, cy] for _, cx, cy in centroids], dtype=np.float32
                )
                field_pts = transform_points_to_field(
                    pixel_pts, h_result.H_frame_to_field
                )
                field_positions = [
                    (centroids[i][0], float(field_pts[i][0]), float(field_pts[i][1]))
                    for i in range(len(centroids))
                ]
            else:
                field_positions = []
        else:
            panel_sam = cv2.resize(frame_bgr, (fw_scaled, field_h))
            field_positions = []

        # Activity recognition
        event = goal_detector.update(frame_idx, field_positions)
        if event:
            # print(f"  Frame {event.frame_idx}: {event.event_type} | {event.details}")
            if event.event_type == "goal":
                match_score[event.details["scoring_class"]] += 1


        # Panel 2: field map with positions (original scale)
        panel_field = draw_field_positions(field_image, field_positions, icons)

        # Combine panels
        combined = np.hstack([panel_sam, panel_field])
        writer.write(combined)

        print(f"match_score: {match_score}")

    writer.release()
    print(f"Pipeline video saved to {video_path}")


if __name__ == "__main__":
    main()
