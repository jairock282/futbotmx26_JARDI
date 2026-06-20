"""Rule-based activity recognition from field coordinates."""

from .detector import (
    ActivityEvent,
    ControlDetector,
    GoalDetector,
    GoalZone,
    PassingDetector,
    load_goal_zones,
)

__all__ = [
    "ActivityEvent",
    "ControlDetector",
    "GoalDetector",
    "GoalZone",
    "PassingDetector",
    "load_goal_zones",
]
