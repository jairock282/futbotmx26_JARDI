#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import mimetypes
import queue
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from futbot_narration.narrator import (
    FutbotNarrationPipeline,
    MatchAction,
    MockCommentaryGenerator,
    iter_jsonl_actions,
)


WEB_ROOT = REPO_ROOT / "web" / "narration_live"


class LiveNarrationState:
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

    def next_id(self) -> int:
        with self.counter_lock:
            self.counter += 1
            return self.counter

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


def make_handler(state: LiveNarrationState) -> type[BaseHTTPRequestHandler]:
    class LiveNarrationHandler(BaseHTTPRequestHandler):
        server_version = "FutBotLiveNarration/1.0"

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
            elif parsed.path.startswith("/static/"):
                relative = parsed.path.removeprefix("/static/")
                self._serve_file(WEB_ROOT / relative)
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

        def _serve_file(self, path: Path, head_only: bool = False) -> None:
            resolved = path.resolve()
            allowed_roots = [WEB_ROOT.resolve(), state.output_dir.resolve()]
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

    return LiveNarrationHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Web app local para narracion en vivo de FutBotMX.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--env", default=".env")
    parser.add_argument("--mock", action="store_true", help="No llama a OpenAI ni ElevenLabs.")
    parser.add_argument("--output-dir", default="outputs/live_narration_audio")
    parser.add_argument("--manifest", default="outputs/live_narration_manifest.jsonl")
    parser.add_argument("--demo-actions", default="examples/actions_stream.jsonl")
    parser.add_argument(
        "--action-timeout",
        type=float,
        default=12.0,
        help="Segundos maximos para GPT+TTS antes de usar narracion fallback.",
    )
    parser.add_argument(
        "--stream-audio",
        action="store_true",
        help="Usa el endpoint streaming de ElevenLabs y reproduce desde /audio-stream/<id>.mp3.",
    )
    args = parser.parse_args()

    pipeline = FutbotNarrationPipeline.from_env(args.env, mock=args.mock)
    pipeline.config.output_dir = Path(args.output_dir)
    state = LiveNarrationState(
        pipeline=pipeline,
        output_dir=Path(args.output_dir),
        manifest_path=Path(args.manifest),
        demo_actions_path=Path(args.demo_actions),
        stream_audio=args.stream_audio,
        action_timeout_seconds=args.action_timeout,
    )
    worker = threading.Thread(target=state.run_worker, daemon=True)
    worker.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Live narration app: http://{args.host}:{args.port}", flush=True)
    print("POST actions to: /api/actions", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping live narration app.", flush=True)
    finally:
        server.server_close()


def _ensure_action_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Cada accion debe ser un objeto JSON.")
    return value


def _parse_stream_id(path: str) -> int:
    name = Path(unquote(path.removeprefix("/audio-stream/"))).stem
    return int(name)


if __name__ == "__main__":
    main()
