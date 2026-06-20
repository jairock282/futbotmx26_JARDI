"""Usage example for the futbot_sam package."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from futbot_sam import (
    SAMTracker,
    TrackingConfig,
    load_tracking_classes,
    render_overlay_video,
    save_mask_frames,
)

# --- Configuration ---
COLORS = {
    "ball": (255, 0, 0),
    "robot_a": (255, 255, 0),
    "robot_b": (0, 100, 255)
}
OUTPUT_FPS = 20.0

sample_id = "IMG_9938"
tracking_config_path = f"configs/tracking/tracking_classes_{sample_id.split('_')[-1]}.json"
tracking_classes = load_tracking_classes(tracking_config_path)

config = TrackingConfig(offload_video_to_cpu=True)

frames_dir = f"data/frames/{sample_id}"
output_dir = f"/mnt/HDD/model_outputs/futbot/{sample_id}"
os.makedirs(output_dir, exist_ok=True)

# --- Step 1: Track ---
tracker = SAMTracker(tracking_classes, config)
result = tracker.track(frames_dir)

# --- Step 2: Save outputs ---
labels = [c.label for c in tracking_classes]

save_mask_frames(result, f"{output_dir}/mask_frames", labels=labels, colors=COLORS)

render_overlay_video(
    result,
    frames_dir,
    f"{output_dir}/tracking_overlay.mp4",
    labels=labels,
    fps=OUTPUT_FPS,
    colors=COLORS,
    config=config,
)
