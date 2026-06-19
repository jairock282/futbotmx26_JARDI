#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from futbot_narration.narrator import ElevenLabsSpeechGenerator, NarrationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba streaming de ElevenLabs para FutBotMX.")
    parser.add_argument(
        "--text",
        default="Gol del equipo blanco, el robot B4 la manda a guardar y se prende la cancha.",
    )
    parser.add_argument("--env", default=".env")
    parser.add_argument("--output", default="outputs/streaming_test.mp3")
    args = parser.parse_args()

    config = NarrationConfig.from_env(args.env)
    speech = ElevenLabsSpeechGenerator(
        config.elevenlabs_api_key,
        config.elevenlabs_voice_id,
        config.elevenlabs_model_id,
        config.elevenlabs_output_format,
        config.elevenlabs_voice_speed,
        config.elevenlabs_streaming_latency,
        config.language_code,
        config.request_timeout_seconds,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    first_chunk_at: float | None = None
    chunk_count = 0
    byte_count = 0

    with output_path.open("wb") as output:
        for chunk in speech.synthesize_stream(args.text):
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
            chunk_count += 1
            byte_count += len(chunk)
            output.write(chunk)

    finished = time.perf_counter()
    first_ms = (first_chunk_at - start) * 1000 if first_chunk_at else 0.0
    total_ms = (finished - start) * 1000
    print(f"output={output_path}")
    print(f"chunks={chunk_count}")
    print(f"bytes={byte_count}")
    print(f"time_to_first_chunk_ms={first_ms:.0f}")
    print(f"total_ms={total_ms:.0f}")
    print(f"voice_speed={config.elevenlabs_voice_speed}")
    print(f"streaming_latency={config.elevenlabs_streaming_latency}")


if __name__ == "__main__":
    main()
