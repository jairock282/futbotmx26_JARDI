#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from futbot_homography.tracking import (
    ConsecutiveHomographyEstimator,
    HomographyTrackingConfig,
    compute_reference_homography,
    read_image,
    sorted_frame_paths,
    transform_points_to_field,
    warp_frame_to_field,
)


def main() -> None:
    frames_dir = Path("data/frames")
    points_path = Path("homography_points.json")
    field_image_path = Path("cancha.png")
    reference_frame_path = frames_dir / "frame_0001.jpg"

    reference = compute_reference_homography(points_path)
    reference_frame = read_image(reference_frame_path)

    config = HomographyTrackingConfig(
        tracking_mode="sequential",
        motion="affine",
        ecc_scale=0.5,
    )
    estimator = ConsecutiveHomographyEstimator(
        reference_frame,
        reference.H_ref,
        config=config,
        reference_name=reference_frame_path.name,
    )

    field_image = read_image(field_image_path)
    field_h, field_w = field_image.shape[:2]
    output_dir = Path("example_homography_tracking_output")
    output_dir.mkdir(exist_ok=True)

    for frame_path in sorted_frame_paths(frames_dir):
        frame = read_image(frame_path)
        result = estimator.process(
            frame,
            frame_name=frame_path.name,
            is_reference=frame_path.name == reference_frame_path.name,
        )

        # Despues, cuando tengas detecciones de robots, sustituye estos puntos
        # por los centros de la base de cada robot en pixeles del frame original.
        robot_centers_px = np.array(
            [
                [960.0, 540.0],
            ],
            dtype=np.float32,
        )
        robot_centers_field = transform_points_to_field(
            robot_centers_px,
            result.H_frame_to_field,
        )

        top_view = warp_frame_to_field(
            frame,
            result.H_frame_to_field,
            field_size=(field_w, field_h),
        )

        for x, y in robot_centers_field:
            cv2.circle(top_view, (int(round(x)), int(round(y))), 8, (0, 0, 255), 2)

        cv2.imwrite(str(output_dir / frame_path.name), top_view)
        print(
            frame_path.name,
            result.status,
            "ecc=",
            result.ecc_score,
            "robot_field_px=",
            robot_centers_field.tolist(),
        )


if __name__ == "__main__":
    main()
