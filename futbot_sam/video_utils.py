# from __future__ import annotations
#
# from pathlib import Path
#
# import cv2
# from PIL import Image
#
#
# def extract_frames(
#     video_path: str | Path,
#     output_dir: str | Path,
#     target_fps: float = 20.0,
# ) -> tuple[int, float]:
#     """Extract frames from a video at a target FPS.
#
#     Args:
#         video_path: Path to the source video file.
#         output_dir: Directory to save extracted frame images.
#         target_fps: Desired frames per second for extraction.
#
#     Returns:
#         Tuple of (number of saved frames, original video FPS).
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise ValueError(f"Could not open video: {video_path}")
#
#     original_fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_interval = max(1, round(original_fps / target_fps))
#     frame_count = 0
#     saved_count = 0
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_count % frame_interval == 0:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             Image.fromarray(frame_rgb).save(str(output_dir / f"{saved_count:05d}.jpg"))
#             saved_count += 1
#         frame_count += 1
#
#     cap.release()
#     print(
#         f"Extracted {saved_count} frames at ~{target_fps} FPS "
#         f"(original: {original_fps:.1f} FPS, every {frame_interval}th frame)"
#     )
#     return saved_count, original_fps
#TODO: remove