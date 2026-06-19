"""Main pipeline: SAM tracking + homography + narration web app."""
from __future__ import annotations

import json
import mimetypes
import queue
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
import concurrent.futures
import sys

import cv2
import numpy as np

from futbot_activity_recognition import ControlDetector, GoalDetector, PassingDetector, load_goal_zones
from futbot_homography import (
    ConsecutiveHomographyEstimator,
    HomographyTrackingConfig,
    compute_reference_homography,
    read_image,
    sorted_frame_paths,
    transform_points_to_field,
)
from futbot_narration.narrator import (
    FutbotNarrationPipeline,
    MatchAction,
    MockCommentaryGenerator,
    iter_jsonl_actions,
)
from futbot_sam import (
    SAMTracker,
    TrackingConfig,
    load_tracking_classes,
)

# ── Configuration ──────────────────────────────────────────────────────────────
SAMPLE_ID = "IMG_9913"
OUTPUT_FPS = 20
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8060

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

WEB_ROOT = Path("web/narration_live")
ASSETS_ROOT = Path("assets")


# ── Visualization helpers ─────────────────────────────────────────────────────

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


# ── App State ─────────────────────────────────────────────────────────────────
class AppState:
    def __init__(
        self,
        pipeline: FutbotNarrationPipeline,
        output_dir: Path,
        manifest_path: Path,
        demo_actions_path: Path,
        stream_audio: bool,
        action_timeout_seconds: float,
    ) -> None:
        self.pipeline = pipeline
        self.output_dir = output_dir
        self.manifest_path = manifest_path
        self.demo_actions_path = demo_actions_path
        self.stream_audio = stream_audio
        self.action_timeout_seconds = action_timeout_seconds
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self.actions: queue.Queue[dict[str, Any]] = queue.Queue()
        self.records: dict[int, dict[str, Any]] = {}
        self.records_lock = threading.Lock()
        self.stream_texts: dict[int, str] = {}
        self.stream_texts_lock = threading.Lock()
        self.clients: list[queue.Queue[dict[str, Any]]] = []
        self.clients_lock = threading.Lock()
        self.counter = 0
        self.counter_lock = threading.Lock()
        self.generation = 0
        self.generation_lock = threading.Lock()
        self.demo_lock = threading.Lock()
        self.demo_running = False
        # Pipeline (vision)
        self.pipeline_running = False
        self.pipeline_lock = threading.Lock()
        self.match_score: dict[str, int] = {"robot_a": 0, "robot_b": 0}
        # MJPEG frame
        self._current_frame: bytes | None = None
        self._frame_lock = threading.Lock()

    def next_id(self) -> int:
        with self.counter_lock:
            self.counter += 1
            return self.counter

    def set_frame(self, jpeg_bytes: bytes) -> None:
        with self._frame_lock:
            self._current_frame = jpeg_bytes

    def get_frame(self) -> bytes | None:
        with self._frame_lock:
            return self._current_frame

    def current_generation(self) -> int:
        with self.generation_lock:
            return self.generation

    def is_current_generation(self, generation: int) -> bool:
        return generation == self.current_generation()

    def add_client(self) -> queue.Queue[dict[str, Any]]:
        client: queue.Queue[dict[str, Any]] = queue.Queue()
        with self.clients_lock:
            self.clients.append(client)
        return client

    def remove_client(self, client: queue.Queue[dict[str, Any]]) -> None:
        with self.clients_lock:
            if client in self.clients:
                self.clients.remove(client)

    def broadcast(self, event: str, payload: dict[str, Any]) -> None:
        message = {"event": event, "payload": payload}
        with self.clients_lock:
            clients = list(self.clients)
        for client in clients:
            client.put(message)

    def enqueue_action(self, action: dict[str, Any]) -> dict[str, Any]:
        action_id = self.next_id()
        generation = self.current_generation()
        queued = {
            "id": action_id,
            "generation": generation,
            "received_at": time.time(),
            "action": action,
            "status": "queued",
        }
        self._save_record(queued)
        self.actions.put(queued)
        self.broadcast("action_received", queued)
        return queued

    def _save_record(self, record: dict[str, Any]) -> None:
        with self.records_lock:
            current = self.records.get(record["id"], {})
            current.update(record)
            self.records[record["id"]] = current

    def history(self) -> list[dict[str, Any]]:
        with self.records_lock:
            return [self.records[key] for key in sorted(self.records)]

    def reset(self) -> int:
        with self.generation_lock:
            self.generation += 1
            generation = self.generation
        with self.demo_lock:
            self.demo_running = False
        with self.records_lock:
            self.records.clear()
        with self.stream_texts_lock:
            self.stream_texts.clear()
        self.pipeline.reset_opening_phrase()
        with self.counter_lock:
            self.counter = 0
        while True:
            try:
                self.actions.get_nowait()
            except queue.Empty:
                break
            else:
                self.actions.task_done()
        self.broadcast("reset", {"generation": generation})
        return generation

    def run_worker(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            queued = self.actions.get()
            action_id = queued["id"]
            generation = queued.get("generation", 0)
            action = queued["action"]
            try:
                if not self.is_current_generation(generation):
                    continue
                self._save_record({"id": action_id, "generation": generation, "status": "narrating"})
                self.broadcast("narration_started", {"id": action_id, "generation": generation})
                future = self.executor.submit(
                    self.pipeline.process_action,
                    action,
                    not self.stream_audio,
                )
                result = future.result(timeout=self.action_timeout_seconds)
                if not self.is_current_generation(generation):
                    continue
                audio_path = result.audio_path
                record = result.to_manifest_record()
                record["id"] = action_id
                record["generation"] = generation
                record["status"] = "ready"
                if self.stream_audio:
                    with self.stream_texts_lock:
                        self.stream_texts[action_id] = result.text
                    record["audio_url"] = f"/audio-stream/{action_id}.mp3"
                    record["audio_streaming"] = True
                elif audio_path:
                    record["audio_url"] = f"/audio/{audio_path.name}"
                self._publish_record(record)
            except concurrent.futures.TimeoutError:
                record = self._fallback_record(
                    action_id,
                    action,
                    f"timeout despues de {self.action_timeout_seconds:.0f}s",
                    generation,
                )
                if self.is_current_generation(generation):
                    self._publish_record(record)
            except Exception as exc:
                record = self._fallback_record(action_id, action, str(exc), generation)
                if self.is_current_generation(generation):
                    self._publish_record(record)
            finally:
                self.actions.task_done()

    def _publish_record(self, record: dict[str, Any]) -> None:
        if record.get("status") == "ready":
            with self.manifest_path.open("a", encoding="utf-8") as manifest:
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._save_record(record)
        event = "narration_ready" if record.get("status") == "ready" else "narration_error"
        self.broadcast(event, record)

    def _fallback_record(
        self,
        action_id: int,
        action: dict[str, Any],
        error: str,
        generation: int,
    ) -> dict[str, Any]:
        text = MockCommentaryGenerator().generate(MatchAction.from_mapping(action), {})
        text, _ = self.pipeline.prepend_opening_if_needed(text)
        record: dict[str, Any] = {
            "id": action_id,
            "generation": generation,
            "action": MatchAction.from_mapping(action).compact(),
            "status": "ready",
            "text": text,
            "fallback": True,
            "fallback_reason": error,
            "audio_path": None,
            "audio_url": None,
            "browser_tts": True,
        }
        return record

    def start_demo(self, delay_seconds: float) -> bool:
        with self.demo_lock:
            if self.demo_running:
                return False
            self.demo_running = True
        thread = threading.Thread(
            target=self._run_demo,
            args=(delay_seconds,),
            daemon=True,
        )
        thread.start()
        return True

    def _run_demo(self, delay_seconds: float) -> None:
        generation = self.current_generation()
        self.broadcast("demo_started", {"path": str(self.demo_actions_path), "generation": generation})
        try:
            for action in iter_jsonl_actions(self.demo_actions_path):
                if not self.is_current_generation(generation):
                    break
                self.enqueue_action(action)
                time.sleep(delay_seconds)
        finally:
            with self.demo_lock:
                if self.is_current_generation(generation):
                    self.demo_running = False
            if self.is_current_generation(generation):
                self.broadcast("demo_finished", {"generation": generation})
                threading.Thread(target=_wait_and_broadcast_complete, args=(self,), daemon=True).start()


# ── Pipeline runner ───────────────────────────────────────────────────────────
def run_pipeline(state: AppState) -> None:
    """Run the full SAM + homography + activity recognition pipeline."""
    state.broadcast("pipeline_status", {"status": "running", "step": "generating_sam_tracking"})

    # ── 1. SAM tracking ────────────────────────────────────────────────────────
    print("=== Step 1: SAM Tracking ===")
    tracking_classes = load_tracking_classes(tracking_config_path)
    labels = [c.label for c in tracking_classes]

    sam_config = TrackingConfig(offload_video_to_cpu=True)
    tracker = SAMTracker(tracking_classes, sam_config)
    sam_result = tracker.track(str(frames_dir))

    state.broadcast("pipeline_status", {"status": "running", "step": "homography_setup"})

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
    heatmap_tracker = HeatmapTracker(field_h, field_w, sigma=8.0)

    goal_zones = load_goal_zones(goal_zones_path)
    goal_detector = GoalDetector(
        goal_zones=goal_zones,
        proximity_threshold=200,
        cooldown_frames=OUTPUT_FPS * 2,
    )
    pass_detector = PassingDetector(
        proximity_threshold=200,
        cooldown_frames=OUTPUT_FPS * 2,
    )
    control_detector = ControlDetector(
        proximity_threshold=50,
        hold_frames=5,
        cooldown_frames=OUTPUT_FPS * 5,
    )

    state.broadcast("pipeline_status", {"status": "running", "step": "rendering"})

    # ── 3. Frame-by-frame processing ──────────────────────────────────────────
    print("=== Step 3: Rendering video + streaming ===")
    frame_paths = sorted_frame_paths(frames_dir)
    sample_frame = read_image(frame_paths[0])
    fh, fw = sample_frame.shape[:2]

    frame_scale = field_h / fh
    fw_scaled = int(round(fw * frame_scale))

    panel_w = fw_scaled + field_w
    video_path = str(output_dir / "pipeline_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, OUTPUT_FPS, (panel_w, field_h))

    state.match_score = {"robot_a": 0, "robot_b": 0}
    frame_delay = 1.0 / OUTPUT_FPS

    for frame_idx, frame_path in enumerate(frame_paths):
        t0 = time.time()
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
                # Build indexed labels for PassingDetector (e.g. robot_a_0, robot_a_1)
                label_counts: dict[str, int] = {}
                field_positions_indexed = []
                for label, fx, fy in field_positions:
                    idx = label_counts.get(label, 0)
                    label_counts[label] = idx + 1
                    field_positions_indexed.append((f"{label}_{idx}", fx, fy))
            else:
                field_positions = []
                field_positions_indexed = []
        else:
            panel_sam = cv2.resize(frame_bgr, (fw_scaled, field_h))
            field_positions = []
            field_positions_indexed = []

        # Activity recognition
        event = goal_detector.update(frame_idx, field_positions)
        if event and event.event_type == "goal":
            scoring_class = event.details["scoring_class"]
            state.match_score[scoring_class] += 1
            # Enqueue narration action
            action_data = {
                "type": "gol",
                "team": scoring_class,
                "timestamp": f"frame_{frame_idx}",
                "score": dict(state.match_score),
                "confidence": 0.9,
            }
            state.enqueue_action(action_data)
            print(f"  GOAL! {scoring_class} scores | {state.match_score}")

        pass_event = pass_detector.update(frame_idx, field_positions_indexed)
        if pass_event:
            action_data = {
                "type": "pase",
                "team": pass_event.details["team"],
                "robot_id": pass_event.details["from_robot"],
                "target_robot_id": pass_event.details["to_robot"],
                "timestamp": f"frame_{frame_idx}",
                "confidence": 0.7,
            }
            state.enqueue_action(action_data)
            print(f"  PASS: {pass_event.details['from_robot']} -> {pass_event.details['to_robot']}")

        control_event = control_detector.update(frame_idx, field_positions_indexed)
        if control_event:
            action_data = {
                "type": "controla",
                "team": control_event.details["team"],
                "robot_id": control_event.details["robot"],
                "timestamp": f"frame_{frame_idx}",
                "confidence": 0.6,
            }
            state.enqueue_action(action_data)
            print(f"  CONTROL: {control_event.details['robot']} holds ball ({control_event.details['hold_frames']} frames)")

        # Panel 2: field map with heatmap overlay
        heatmap_tracker.update(field_positions)
        panel_field = heatmap_tracker.render(field_image, alpha=0.35)
        panel_field = draw_field_positions(panel_field, field_positions, icons)

        # Combine panels: sam | field
        combined = np.hstack([panel_sam, panel_field])
        writer.write(combined)

        # Encode and publish frame for MJPEG stream
        _, jpeg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 70])
        state.set_frame(jpeg.tobytes())

        # Broadcast frame progress
        if frame_idx % 10 == 0:
            state.broadcast("pipeline_progress", {
                "frame": frame_idx,
                "total": len(frame_paths),
                "score": state.match_score,
            })

        # Throttle to ~real-time playback
        elapsed = time.time() - t0
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    writer.release()
    print(f"Pipeline video saved to {video_path}")
    state.broadcast("pipeline_status", {"status": "finished", "score": state.match_score})
    with state.pipeline_lock:
        state.pipeline_running = False
    threading.Thread(target=_wait_and_broadcast_complete, args=(state,), daemon=True).start()


def _wait_and_broadcast_complete(state: AppState) -> None:
    """Wait until all queued narrations finish, then notify the frontend."""
    state.actions.join()
    time.sleep(5)
    state.broadcast("narration_complete", {"score": state.match_score})


# ── HTTP Server ───────────────────────────────────────────────────────────────
def _ensure_action_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Cada accion debe ser un objeto JSON.")
    return value

def _parse_stream_id(path: str) -> int:
    name = Path(unquote(path.removeprefix("/audio-stream/"))).stem
    return int(name)

def make_handler(state: AppState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "FutBot/1.0"

        def handle(self) -> None:
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._serve_file(WEB_ROOT / "index.html")
            elif parsed.path == "/events":
                self._serve_events()
            elif parsed.path == "/video-feed":
                self._serve_mjpeg()
            elif parsed.path.startswith("/static/"):
                relative = parsed.path.removeprefix("/static/")
                self._serve_file(WEB_ROOT / relative)
            elif parsed.path.startswith("/assets/"):
                relative = parsed.path.removeprefix("/assets/")
                self._serve_file(ASSETS_ROOT / relative)
            elif parsed.path.startswith("/audio/"):
                name = Path(unquote(parsed.path.removeprefix("/audio/"))).name
                self._serve_file(state.output_dir / name)
            elif parsed.path.startswith("/audio-stream/"):
                try:
                    stream_id = _parse_stream_id(parsed.path)
                except ValueError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._serve_elevenlabs_stream(stream_id)
            elif parsed.path == "/api/status":
                self._send_json(
                    {
                        "queued": state.actions.qsize(),
                        "demo_running": state.demo_running,
                        "generation": state.current_generation(),
                        "output_dir": str(state.output_dir),
                        "stream_audio": state.stream_audio,
                    }
                )
            elif parsed.path == "/api/history":
                self._send_json({"records": state.history()})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/audio/"):
                name = Path(unquote(parsed.path.removeprefix("/audio/"))).name
                self._serve_file(state.output_dir / name, head_only=True)
            elif parsed.path.startswith("/audio-stream/"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "audio/mpeg")
                self.end_headers()
            elif parsed.path.startswith("/static/"):
                relative = parsed.path.removeprefix("/static/")
                self._serve_file(WEB_ROOT / relative, head_only=True)
            elif parsed.path == "/":
                self._serve_file(WEB_ROOT / "index.html", head_only=True)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/actions":
                payload = self._read_json()
                if isinstance(payload, list):
                    queued = [state.enqueue_action(_ensure_action_dict(item)) for item in payload]
                else:
                    queued = state.enqueue_action(_ensure_action_dict(payload))
                self._send_json({"queued": queued}, status=HTTPStatus.ACCEPTED)
            elif parsed.path == "/api/pipeline":
                with state.pipeline_lock:
                    if state.pipeline_running:
                        self._send_json({"started": False, "reason": "already running"}, HTTPStatus.CONFLICT)
                        return
                    state.pipeline_running = True
                thread = threading.Thread(target=run_pipeline, args=(state,), daemon=True)
                thread.start()
                self._send_json({"started": True}, status=HTTPStatus.ACCEPTED)
            elif parsed.path == "/api/demo":
                started = state.start_demo(delay_seconds=2.5)
                status = HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT
                self._send_json({"started": started}, status=status)
            elif parsed.path == "/api/reset":
                generation = state.reset()
                self._send_json({"ok": True, "generation": generation})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), format % args))

        def _serve_events(self) -> None:
            client = state.add_client()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.flush()
            try:
                self._write_sse("connected", {"ok": True})
                while True:
                    try:
                        message = client.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        continue
                    self._write_sse(message["event"], message["payload"])
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                state.remove_client(client)

        def _write_sse(self, event: str, payload: dict[str, Any]) -> None:
            body = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            self.wfile.write(body.encode("utf-8"))
            self.wfile.flush()

        def _serve_mjpeg(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    frame = state.get_frame()
                    if frame is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    time.sleep(1.0 / OUTPUT_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _serve_file(self, path: Path, head_only: bool = False) -> None:
            resolved = path.resolve()
            allowed_roots = [WEB_ROOT.resolve(), state.output_dir.resolve(), ASSETS_ROOT.resolve()]
            if not any(resolved == root or root in resolved.parents for root in allowed_roots):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not resolved.exists() or not resolved.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
            data = resolved.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if not head_only:
                self.wfile.write(data)

        def _serve_elevenlabs_stream(self, stream_id: int) -> None:
            with state.stream_texts_lock:
                text = state.stream_texts.get(stream_id)
            if text is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                for chunk in state.pipeline.speech_generator.synthesize_stream(text):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            try:
                return json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalido: {exc}") from exc

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    pipeline = FutbotNarrationPipeline.from_env(".env", mock=False)
    pipeline.config.output_dir = output_dir

    manifest_path = output_dir / "narration_manifest.jsonl"
    demo_actions_path = Path("examples/actions_stream.jsonl")

    state = AppState(
        pipeline=pipeline,
        output_dir=output_dir,
        manifest_path=manifest_path,
        demo_actions_path=demo_actions_path,
        stream_audio=False,
        action_timeout_seconds=60.0,
    )

    # Start narration worker
    worker = threading.Thread(target=state.run_worker, daemon=True)
    worker.start()

    # Start HTTP server
    server = ThreadingHTTPServer((SERVER_HOST, SERVER_PORT), make_handler(state))
    print(f"FutBotMX app: http://{SERVER_HOST}:{SERVER_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
