"""Example: rule-based goal detection using field coordinates."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from futbot_activity_recognition import GoalDetector, load_goal_zones

# ── Configuration ──────────────────────────────────────────────────────────────
goal_zones_path = "configs/roi/goal_zones_9913.json"
proximity_threshold = 100.0  # max distance (field px) to consider ball "with" a robot

# ── Setup ──────────────────────────────────────────────────────────────────────
zones = load_goal_zones(goal_zones_path)
detector = GoalDetector(goal_zones=zones, proximity_threshold=proximity_threshold)

print("Goal zones loaded:")
for cls, zone in zones.items():
    print(f"  {cls}: bbox=({zone.x}, {zone.y}, {zone.w}, {zone.h})")

# ── Simulated frame data (field coordinates) ──────────────────────────────────
# Each entry: list of (label, field_x, field_y)
# Simulates ball moving from near robot_a into robot_a's goal zone
simulated_frames = [
    # Frame 0: ball near robot_a, far from goal
    [("ball", 500.0, 345.0), ("robot_a", 480.0, 340.0), ("robot_b", 200.0, 345.0)],
    # Frame 1: ball still near robot_a, approaching goal
    [("ball", 700.0, 345.0), ("robot_a", 680.0, 340.0), ("robot_b", 200.0, 345.0)],
    # Frame 2: ball still near robot_a, very close to goal
    [("ball", 850.0, 345.0), ("robot_a", 830.0, 340.0), ("robot_b", 200.0, 345.0)],
    # Frame 3: ball enters robot_a's goal zone → GOAL!
    [("ball", 880.0, 345.0), ("robot_a", 830.0, 340.0), ("robot_b", 200.0, 345.0)],
    # Frame 4: after goal, ball reset to center
    [("ball", 450.0, 345.0), ("robot_a", 400.0, 340.0), ("robot_b", 500.0, 345.0)],
]

# ── Run detection ─────────────────────────────────────────────────────────────
print("\nProcessing frames...")
for frame_idx, positions in enumerate(simulated_frames):
    events = detector.update(frame_idx, positions)

    ball = next((p for p in positions if p[0] == "ball"), None)
    print(f"  Frame {frame_idx}: ball=({ball[1]:.0f}, {ball[2]:.0f})")

    for event in events:
        print(f"EVENT: {event.event_type} | {event.details}")

print("\nDone.")
