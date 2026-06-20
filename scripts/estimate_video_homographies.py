#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from futbot_homography.tracking import (
    HomographyTrackingConfig,
    compute_reference_homography,
    estimate_homographies_for_frame_paths,
    load_points_config,
    read_image,
    sorted_frame_paths,
    warp_frame_to_field,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Actualiza la homografia por frame usando alineacion ECC."
    )
    parser.add_argument(
        "--frames-dir",
        default="data/frames",
        help="Directorio con frames frame_0001.jpg, frame_0002.jpg, ...",
    )
    parser.add_argument(
        "--points",
        default="homography_points.json",
        help="JSON con puntos manuales del frame de referencia a cancha.png.",
    )
    parser.add_argument(
        "--output",
        default="homographies_IMG_9913.json",
        help="Archivo JSON de salida con H por frame.",
    )
    parser.add_argument(
        "--csv-output",
        default="homographies_IMG_9913.csv",
        help="CSV de salida con H aplanada por frame.",
    )
    parser.add_argument(
        "--reference-frame",
        default="frame_0001.jpg",
        help="Frame de referencia dentro de --frames-dir.",
    )
    parser.add_argument(
        "--motion",
        choices=["affine", "euclidean", "homography", "translation"],
        default="affine",
        help="Modelo ECC entre frames.",
    )
    parser.add_argument(
        "--tracking-mode",
        choices=["sequential", "reference"],
        default="sequential",
        help=(
            "sequential alinea cada frame contra el anterior y acumula; "
            "reference alinea cada frame directamente contra el frame de referencia."
        ),
    )
    parser.add_argument(
        "--ecc-scale",
        type=float,
        default=0.5,
        help="Escala usada para ECC. Menor es mas rapido; 1.0 usa resolucion completa.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=150,
        help="Iteraciones maximas de ECC.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Criterio de convergencia de ECC.",
    )
    parser.add_argument(
        "--line-mask-dilate",
        type=int,
        default=120,
        help="Dilatacion en pixeles para la mascara de lineas.",
    )
    parser.add_argument(
        "--accept-score",
        type=float,
        default=0.30,
        help="Score ECC a partir del cual se acepta un intento sin probar mas alternativas.",
    )
    parser.add_argument(
        "--write-preview",
        action="store_true",
        help="Guarda una vista cenital de cada frame usando la homografia estimada.",
    )
    parser.add_argument(
        "--preview-dir",
        default="homography_previews",
        help="Directorio para previews si --write-preview esta activo.",
    )
    return parser.parse_args()


def save_outputs(
    output_path: Path,
    csv_path: Path,
    records: list[dict[str, Any]],
    H_ref: np.ndarray,
    args: argparse.Namespace,
) -> None:
    payload = {
        "frames_dir": str(Path(args.frames_dir)),
        "reference_frame": args.reference_frame,
        "points_file": str(Path(args.points)),
        "motion_model": args.motion,
        "tracking_mode": args.tracking_mode,
        "ecc_scale": args.ecc_scale,
        "H_ref": H_ref.tolist(),
        "frames": records,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "status",
                "ecc_score",
                "h00",
                "h01",
                "h02",
                "h10",
                "h11",
                "h12",
                "h20",
                "h21",
                "h22",
            ]
        )
        for record in records:
            h = np.asarray(record["H_frame_to_field"], dtype=np.float64).reshape(-1)
            writer.writerow(
                [
                    record["frame"],
                    record["status"],
                    record["ecc_score"],
                    *[f"{value:.12g}" for value in h],
                ]
            )


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir).expanduser().resolve()
    points_path = Path(args.points).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    csv_path = Path(args.csv_output).expanduser().resolve()

    reference = compute_reference_homography(points_path)
    if reference.inliers is not None:
        print(f"H_ref calculada con {reference.inliers}/{reference.total_points} inliers.")

    config = HomographyTrackingConfig(
        motion=args.motion,
        tracking_mode=args.tracking_mode,
        ecc_scale=args.ecc_scale,
        iterations=args.iterations,
        eps=args.eps,
        line_mask_dilate=args.line_mask_dilate,
        accept_score=args.accept_score,
    )

    frame_paths = sorted_frame_paths(frames_dir)
    reference_path = frames_dir / args.reference_frame
    results = estimate_homographies_for_frame_paths(
        frame_paths,
        reference_path,
        reference.H_ref,
        config=config,
    )

    preview_dir = Path(args.preview_dir).expanduser().resolve()
    if args.write_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)
        field_config = load_points_config(points_path)
        field_path = Path(field_config.get("field_image", "cancha.png"))
        if not field_path.is_absolute():
            field_path = points_path.parent / field_path
        field_image = read_image(field_path)
        field_h, field_w = field_image.shape[:2]

    records: list[dict[str, Any]] = []
    for frame_path, result in zip(frame_paths, results, strict=True):
        records.append(result.to_dict())
        print(f"{result.frame_name}: {result.status}, ecc={result.ecc_score}", flush=True)

        if args.write_preview:
            frame = read_image(frame_path)
            warped = warp_frame_to_field(
                frame,
                result.H_frame_to_field,
                field_size=(field_w, field_h),
            )
            cv2.imwrite(str(preview_dir / frame_path.name), warped)

    save_outputs(output_path, csv_path, records, reference.H_ref, args)
    print(f"\nGuardado JSON: {output_path}")
    print(f"Guardado CSV:  {csv_path}")
    if args.write_preview:
        print(f"Previews:      {preview_dir}")


if __name__ == "__main__":
    main()
