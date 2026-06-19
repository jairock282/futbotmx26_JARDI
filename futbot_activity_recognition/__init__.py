"""Rule-based activity recognition from field coordinates."""

from .detector import (
    ActivityEvent,
    GoalDetector,
    GoalZone,
    load_goal_zones,
)

__all__ = [
    "ActivityEvent",
    "GoalDetector",
    "GoalZone",
    "load_goal_zones",
]
