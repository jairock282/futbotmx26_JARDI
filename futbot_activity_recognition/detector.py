"""Rule-based activity detection using field coordinates per frame."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoalZone:
    """A goal zone defined as a bounding box in field image coordinates."""
    robot_class: str
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        """Check if point (px, py) is inside this zone."""
        return (self.x <= px <= self.x + self.w) and (self.y <= py <= self.y + self.h)


@dataclass(frozen=True)
class ActivityEvent:
    """A detected activity event."""
    event_type: str
    frame_idx: int
    details: dict


def load_goal_zones(config_path: str | Path) -> dict[str, GoalZone]:
    """Load goal zones from a JSON config file.

    Expected format:
        {
            "robot_a": {"bbox": [x, y, w, h]},
            "robot_b": {"bbox": [x, y, w, h]}
        }

    Args:
        config_path: Path to the goal zones JSON.

    Returns:
        Dict mapping robot class label -> GoalZone.
    """
    with Path(config_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    zones = {}
    for robot_class, entry in data.items():
        x, y, w, h = entry["bbox"]
        zones[robot_class] = GoalZone(
            robot_class=robot_class, x=x, y=y, w=w, h=h
        )
    return zones


class GoalDetector:
    """Detect goal events based on ball position transitions.

    A goal is detected when:
        1. At frame t, the ball is closest to a robot of class X.
        2. At frame t+1, the ball enters the goal zone of class Y.
    """

    def __init__(
        self,
        goal_zones: dict[str, GoalZone],
        proximity_threshold: float = 100.0,
        cooldown_frames: int = 30,
    ) -> None:
        """
        Args:
            goal_zones: Dict mapping robot class -> GoalZone (defensive zone).
            proximity_threshold: Max distance (field px) to consider ball "with" a robot.
            cooldown_frames: Frames to ignore after a goal is detected.
        """
        self.goal_zones = goal_zones
        self.proximity_threshold = proximity_threshold
        self.cooldown_frames = cooldown_frames
        self._prev_ball_pos: tuple[float, float] | None = None
        self._prev_nearest_class: str | None = None
        self._cooldown_remaining: int = 0

    def reset(self) -> None:
        """Reset internal state."""
        self._prev_ball_pos = None
        self._prev_nearest_class = None
        self._cooldown_remaining = 0

    def update(
        self,
        frame_idx: int,
        field_positions: list[tuple[str, float, float]],
    ) -> ActivityEvent | None:
        """Process one frame and return a goal event if detected.

        Args:
            frame_idx: Current frame index.
            field_positions: List of (label, field_x, field_y) for all objects.

        Returns:
            ActivityEvent if a goal is detected, None otherwise.
        """
        # Cooldown active — skip detection
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return None

        # Find ball position
        ball_pos = None
        for label, fx, fy in field_positions:
            if label == "ball":
                ball_pos = (fx, fy)
                break

        if ball_pos is None:
            self._prev_ball_pos = None
            self._prev_nearest_class = None
            return None

        # Find nearest robot class to the ball
        nearest_class = None
        nearest_dist = float("inf")
        for label, fx, fy in field_positions:
            if label == "ball":
                continue
            dist = ((ball_pos[0] - fx) ** 2 + (ball_pos[1] - fy) ** 2) ** 0.5
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_class = label

        # Check goal condition: ball was with class X and now enters the opponent's zone
        event = None
        if self._prev_nearest_class is not None and self._prev_ball_pos is not None:
            for zone_class, zone in self.goal_zones.items():
                if zone_class == self._prev_nearest_class:
                    continue  # skip own zone — a team scores at the opponent's goal
                if zone.contains(ball_pos[0], ball_pos[1]):
                    event = ActivityEvent(
                        event_type="goal",
                        frame_idx=frame_idx,
                        details={
                            "scoring_class": self._prev_nearest_class,
                            "goal_zone": zone_class,
                            "ball_position": ball_pos,
                            "prev_ball_position": self._prev_ball_pos,
                        },
                    )
                    self._cooldown_remaining = self.cooldown_frames
                    break

        # Update state for next frame
        if nearest_class is not None and nearest_dist <= self.proximity_threshold:
            self._prev_nearest_class = nearest_class
        else:
            self._prev_nearest_class = None
        self._prev_ball_pos = ball_pos

        return event
