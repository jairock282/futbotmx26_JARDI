#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from futbot_narration.narrator import ElevenLabsSpeechGenerator, NarrationConfig


DEFAULT_TEXT = "Gol del equipo blanco, B4 la manda a guardar y se prende la cancha."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara latencia de ElevenLabs con y sin streaming.",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--env", default=".env")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/latency_benchmark")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normal_results = []
    stream_results = []

    for index in range(1, args.trials + 1):
        normal_results.append(run_normal_trial(speech, args.text, output_dir, index))
        stream_results.append(run_stream_trial(speech, args.text, output_dir, index))

    report = {
        "text": args.text,
        "trials": args.trials,
        "voice_speed": config.elevenlabs_voice_speed,
        "streaming_latency": config.elevenlabs_streaming_latency,
        "normal": {
            "trials": normal_results,
            "summary": summarize(normal_results, "total_ms"),
        },
        "stream": {
            "trials": stream_results,
            "time_to_first_chunk_summary": summarize(stream_results, "time_to_first_chunk_ms"),
            "total_summary": summarize(stream_results, "total_ms"),
        },
    }

    report_path = output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report={report_path}")


def run_normal_trial(
    speech: ElevenLabsSpeechGenerator,
    text: str,
    output_dir: Path,
    index: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    audio = speech.synthesize(text)
    finished = time.perf_counter()
    output_path = output_dir / f"normal_{index:02d}.mp3"
    output_path.write_bytes(audio)
    return {
        "trial": index,
        "total_ms": round((finished - start) * 1000),
        "bytes": len(audio),
        "output": str(output_path),
    }


def run_stream_trial(
    speech: ElevenLabsSpeechGenerator,
    text: str,
    output_dir: Path,
    index: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    first_chunk_at: float | None = None
    chunk_count = 0
    byte_count = 0
    output_path = output_dir / f"stream_{index:02d}.mp3"

    with output_path.open("wb") as output:
        for chunk in speech.synthesize_stream(text):
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
            chunk_count += 1
            byte_count += len(chunk)
            output.write(chunk)

    finished = time.perf_counter()
    return {
        "trial": index,
        "time_to_first_chunk_ms": round((first_chunk_at - start) * 1000) if first_chunk_at else 0,
        "total_ms": round((finished - start) * 1000),
        "chunks": chunk_count,
        "bytes": byte_count,
        "output": str(output_path),
    }


def summarize(results: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(result[key]) for result in results]
    return {
        "min_ms": min(values),
        "mean_ms": round(statistics.fmean(values), 1),
        "max_ms": max(values),
    }


if __name__ == "__main__":
    main()
