#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from futbot_narration.narrator import FutbotNarrationPipeline, iter_jsonl_actions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera narracion y audio desde un stream JSONL de acciones FutBotMX.",
    )
    parser.add_argument(
        "--actions",
        default="-",
        help="Archivo JSONL de acciones. Usa '-' para leer desde stdin.",
    )
    parser.add_argument("--env", default=".env", help="Ruta al archivo .env.")
    parser.add_argument(
        "--output-dir",
        default="outputs/narration_audio",
        help="Carpeta para MP3 generados.",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/narration_manifest.jsonl",
        help="JSONL con accion, texto y ruta de audio.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Solo genera texto; no llama a ElevenLabs.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Prueba local sin llamar a OpenAI ni ElevenLabs.",
    )
    args = parser.parse_args()

    pipeline = FutbotNarrationPipeline.from_env(
        args.env,
        mock=args.mock,
        require_elevenlabs=not args.no_audio,
    )
    pipeline.config.output_dir = Path(args.output_dir)

    actions = _read_stdin_actions() if args.actions == "-" else iter_jsonl_actions(args.actions)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for result in pipeline.process_stream(actions, write_audio=not args.no_audio):
            record = result.to_manifest_record()
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            print(json.dumps(record, ensure_ascii=False), flush=True)


def _read_stdin_actions() -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(sys.stdin, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON invalido en stdin linea {line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"La linea {line_number} debe ser un objeto JSON.")
        yield payload


if __name__ == "__main__":
    main()
