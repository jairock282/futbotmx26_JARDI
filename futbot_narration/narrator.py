from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


ELEVENLABS_DEFAULT_VOICE_ID = "QpDQJR3frbDwOhTIo8nW"
DEFAULT_OPENING_PHRASE = "Amigos aficionados que viven la intensidad del futbol"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_TTS_STREAM_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"


COMMENTARY_INSTRUCTIONS = """Eres un comentarista de futbol mexicano narrando un partido de robots.
Convierte cada accion detectada en una narracion breve, natural y emocionante.
Reglas:
- Habla en espanol mexicano, con energia de transmision deportiva.
- Los equipos del torneo son blanco y negro; usalos exactamente cuando vengan en el evento.
- No inventes jugadores, marcador, faltas o datos que no vengan en el evento/contexto.
- Si la confianza del detector es baja, usa lenguaje prudente.
- Para goles puedes usar mas emocion, pero no alargues exageradamente la palabra gol.
- Para acciones normales usa una sola frase de 8 a 22 palabras.
- Devuelve solo el texto que se debe leer en voz alta."""


@dataclass(frozen=True)
class MatchAction:
    """Evento normalizado que llega desde el detector de acciones."""

    action_type: str
    timestamp: str | None = None
    team: str | None = None
    robot_id: str | None = None
    target_robot_id: str | None = None
    outcome: str | None = None
    confidence: float | None = None
    score: Mapping[str, int] | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MatchAction":
        action_type = str(
            data.get("type")
            or data.get("action")
            or data.get("accion")
            or data.get("event")
            or "accion"
        )
        confidence = data.get("confidence", data.get("confianza"))
        return cls(
            action_type=action_type,
            timestamp=_optional_str(data.get("timestamp") or data.get("time") or data.get("t")),
            team=_optional_str(data.get("team") or data.get("equipo")),
            robot_id=_optional_str(data.get("robot_id") or data.get("robot") or data.get("player")),
            target_robot_id=_optional_str(
                data.get("target_robot_id") or data.get("target") or data.get("receiver")
            ),
            outcome=_optional_str(data.get("outcome") or data.get("resultado")),
            confidence=float(confidence) if confidence is not None else None,
            score=_parse_score(data.get("score") or data.get("marcador")),
            raw=dict(data),
        )

    def compact(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.action_type,
            "timestamp": self.timestamp,
            "team": self.team,
            "robot_id": self.robot_id,
            "target_robot_id": self.target_robot_id,
            "outcome": self.outcome,
            "confidence": self.confidence,
            "score": dict(self.score) if self.score else None,
        }
        extras = {
            key: value
            for key, value in self.raw.items()
            if key not in payload and key not in {"action", "accion", "event"}
        }
        if extras:
            payload["extra"] = extras
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class NarrationConfig:
    openai_api_key: str
    elevenlabs_api_key: str
    openai_model: str = "gpt-5.5"
    openai_reasoning_effort: str = "low"
    elevenlabs_voice_id: str = ELEVENLABS_DEFAULT_VOICE_ID
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    elevenlabs_output_format: str = "mp3_44100_128"
    elevenlabs_voice_speed: float = 1.18
    elevenlabs_streaming_latency: int | None = 3
    language_code: str = "es"
    output_dir: Path = Path("outputs/narration_audio")
    history_size: int = 6
    request_timeout_seconds: int = 12
    opening_phrase: str = DEFAULT_OPENING_PHRASE

    @classmethod
    def from_env(
        cls,
        env_path: str | Path = ".env",
        require_elevenlabs: bool = True,
    ) -> "NarrationConfig":
        load_env_file(env_path)
        return cls(
            openai_api_key=_required_env("OPENAI_API_KEY"),
            elevenlabs_api_key=(
                _required_env("ELEVENLABS_API_KEY")
                if require_elevenlabs
                else os.getenv("ELEVENLABS_API_KEY", "")
            ),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
            openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "low"),
            elevenlabs_voice_id=os.getenv(
                "ELEVENLABS_VOICE_ID",
                ELEVENLABS_DEFAULT_VOICE_ID,
            ),
            elevenlabs_model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
            elevenlabs_output_format=os.getenv(
                "ELEVENLABS_OUTPUT_FORMAT",
                "mp3_44100_128",
            ),
            elevenlabs_voice_speed=float(os.getenv("ELEVENLABS_VOICE_SPEED", "1.18")),
            elevenlabs_streaming_latency=_optional_int(
                os.getenv("ELEVENLABS_STREAMING_LATENCY", "3")
            ),
            request_timeout_seconds=int(os.getenv("NARRATION_REQUEST_TIMEOUT_SECONDS", "12")),
            opening_phrase=os.getenv("NARRATION_OPENING_PHRASE", DEFAULT_OPENING_PHRASE),
        )


@dataclass(frozen=True)
class NarrationResult:
    action: MatchAction
    text: str
    audio_path: Path | None = None

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "action": self.action.compact(),
            "text": self.text,
            "audio_path": str(self.audio_path) if self.audio_path else None,
        }


class OpenAICommentaryGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
        reasoning_effort: str,
        timeout_seconds: int,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds

    def generate(self, action: MatchAction, context: Mapping[str, Any]) -> str:
        payload = {
            "model": self.model,
            "instructions": COMMENTARY_INSTRUCTIONS,
            "input": json.dumps(
                {
                    "accion_actual": action.compact(),
                    "contexto_partido": context,
                },
                ensure_ascii=False,
            ),
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": 90,
        }
        response = _post_json(
            OPENAI_RESPONSES_URL,
            payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout_seconds=self.timeout_seconds,
        )
        text = _extract_openai_text(response).strip()
        return _clean_commentary_text(text)


class ElevenLabsSpeechGenerator:
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str,
        output_format: str,
        voice_speed: float,
        streaming_latency: int | None,
        language_code: str,
        timeout_seconds: int,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.voice_speed = voice_speed
        self.streaming_latency = streaming_latency
        self.language_code = language_code
        self.timeout_seconds = timeout_seconds

    def synthesize(self, text: str) -> bytes:
        query = urllib.parse.urlencode({"output_format": self.output_format})
        url = f"{ELEVENLABS_TTS_URL.format(voice_id=self.voice_id)}?{query}"
        return _post_bytes(
            url,
            self._payload(text),
            headers=self._headers(),
            timeout_seconds=self.timeout_seconds,
        )

    def synthesize_stream(self, text: str) -> Iterable[bytes]:
        query_params: dict[str, str | int] = {"output_format": self.output_format}
        if self.streaming_latency is not None:
            query_params["optimize_streaming_latency"] = self.streaming_latency
        query = urllib.parse.urlencode(query_params)
        url = f"{ELEVENLABS_TTS_STREAM_URL.format(voice_id=self.voice_id)}?{query}"
        yield from _post_stream(
            url,
            self._payload(text),
            headers=self._headers(),
            timeout_seconds=self.timeout_seconds,
        )

    def _payload(self, text: str) -> dict[str, Any]:
        return {
            "text": text,
            "model_id": self.model_id,
            "language_code": self.language_code,
            "voice_settings": {
                "stability": 0.42,
                "similarity_boost": 0.8,
                "style": 0.55,
                "speed": self.voice_speed,
                "use_speaker_boost": True,
            },
        }

    def _headers(self) -> dict[str, str]:
        return {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }


class MockCommentaryGenerator:
    def generate(self, action: MatchAction, context: Mapping[str, Any]) -> str:
        team = f" del {action.team}" if action.team else ""
        robot = f" el robot {action.robot_id}" if action.robot_id else ""
        target = f" para {action.target_robot_id}" if action.target_robot_id else ""
        low_confidence = action.confidence is not None and action.confidence < 0.65
        maybe = "Parece que " if low_confidence else ""
        kind = _normalize_action_type(action.action_type)

        if kind in {"gol", "goal"}:
            return f"{maybe}Gol{team}! La mando a guardar{robot} y se prende la cancha."
        if kind in {"pase", "pass"}:
            return f"{maybe}Buen pase{team}{robot}{target}, moviendo la pelota con mucha calma."
        if kind in {"fuera_de_lugar", "offside"}:
            return f"{maybe}Se levanta el fuera de lugar{team}, la jugada queda invalidada."
        if kind in {"tiro", "shot"}:
            return f"{maybe}Disparo{team}{robot}, buscando sorprender desde esa zona."
        if kind in {"controla", "control"}:
            return f"{maybe}La controla{team}{robot}, es suya la pelota."
        return f"{maybe}Accion detectada{team}{robot}, el partido sigue tomando ritmo."


class MockSpeechGenerator:
    def synthesize(self, text: str) -> bytes:
        return f"MOCK AUDIO: {text}\n".encode("utf-8")

    def synthesize_stream(self, text: str) -> Iterable[bytes]:
        yield self.synthesize(text)


class FutbotNarrationPipeline:
    def __init__(
        self,
        config: NarrationConfig,
        commentary_generator: OpenAICommentaryGenerator | MockCommentaryGenerator | None = None,
        speech_generator: ElevenLabsSpeechGenerator | MockSpeechGenerator | None = None,
    ) -> None:
        self.config = config
        self.history: deque[dict[str, Any]] = deque(maxlen=config.history_size)
        self.scoreboard: dict[str, int] = {}
        self._opening_spoken = False
        self.commentary_generator = commentary_generator or OpenAICommentaryGenerator(
            config.openai_api_key,
            config.openai_model,
            config.openai_reasoning_effort,
            config.request_timeout_seconds,
        )
        self.speech_generator = speech_generator or ElevenLabsSpeechGenerator(
            config.elevenlabs_api_key,
            config.elevenlabs_voice_id,
            config.elevenlabs_model_id,
            config.elevenlabs_output_format,
            config.elevenlabs_voice_speed,
            config.elevenlabs_streaming_latency,
            config.language_code,
            config.request_timeout_seconds,
        )

    @classmethod
    def from_env(
        cls,
        env_path: str | Path = ".env",
        mock: bool = False,
        require_elevenlabs: bool = True,
    ) -> "FutbotNarrationPipeline":
        if mock:
            load_env_file(env_path)
            config = NarrationConfig(
                openai_api_key=os.getenv("OPENAI_API_KEY", "mock-openai-key"),
                elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", "mock-elevenlabs-key"),
                openai_model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
                openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "low"),
                elevenlabs_voice_speed=float(os.getenv("ELEVENLABS_VOICE_SPEED", "1.18")),
                elevenlabs_streaming_latency=_optional_int(
                    os.getenv("ELEVENLABS_STREAMING_LATENCY", "3")
                ),
                request_timeout_seconds=int(os.getenv("NARRATION_REQUEST_TIMEOUT_SECONDS", "12")),
                elevenlabs_voice_id=os.getenv(
                    "ELEVENLABS_VOICE_ID",
                    ELEVENLABS_DEFAULT_VOICE_ID,
                ),
                opening_phrase=os.getenv("NARRATION_OPENING_PHRASE", DEFAULT_OPENING_PHRASE),
            )
            return cls(config, MockCommentaryGenerator(), MockSpeechGenerator())
        return cls(NarrationConfig.from_env(env_path, require_elevenlabs=require_elevenlabs))

    def process_action(self, action_data: Mapping[str, Any], write_audio: bool = True) -> NarrationResult:
        action = MatchAction.from_mapping(action_data)
        self._update_scoreboard(action)
        context = {
            "marcador_actual": self.scoreboard or None,
            "acciones_recientes": list(self.history),
        }
        text = self.commentary_generator.generate(action, context)
        opening_applied = False
        audio_path: Path | None = None
        try:
            text, opening_applied = self.prepend_opening_if_needed(text)
            if write_audio:
                self.config.output_dir.mkdir(parents=True, exist_ok=True)
                audio_bytes = self.speech_generator.synthesize(text)
                audio_path = self._audio_path_for(action)
                audio_path.write_bytes(audio_bytes)
        except Exception:
            if opening_applied:
                self.restore_opening_phrase()
            raise

        record = action.compact()
        record["narration"] = text
        self.history.append(record)
        return NarrationResult(action=action, text=text, audio_path=audio_path)

    def process_stream(
        self,
        actions: Iterable[Mapping[str, Any]],
        write_audio: bool = True,
    ) -> Iterable[NarrationResult]:
        for action in actions:
            yield self.process_action(action, write_audio=write_audio)

    def _update_scoreboard(self, action: MatchAction) -> None:
        if action.score:
            self.scoreboard = {str(team): int(goals) for team, goals in action.score.items()}
            return
        if _normalize_action_type(action.action_type) in {"gol", "goal"} and action.team:
            self.scoreboard[action.team] = self.scoreboard.get(action.team, 0) + 1

    def _audio_path_for(self, action: MatchAction) -> Path:
        timestamp = action.timestamp or str(int(time.time() * 1000))
        stem = "_".join(
            value
            for value in [
                _slugify(timestamp),
                _slugify(action.action_type),
                _slugify(action.team or ""),
                _slugify(action.robot_id or ""),
            ]
            if value
        )
        return self.config.output_dir / f"{stem}.mp3"

    def prepend_opening_if_needed(self, text: str) -> tuple[str, bool]:
        phrase = self.config.opening_phrase.strip()
        if not phrase or self._opening_spoken:
            return text, False
        self._opening_spoken = True
        return f"{phrase}. {text}", True

    def restore_opening_phrase(self) -> None:
        self._opening_spoken = False

    def reset_opening_phrase(self) -> None:
        self._opening_spoken = False


def load_env_file(env_path: str | Path = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def iter_jsonl_actions(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalido en linea {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"La linea {line_number} debe ser un objeto JSON.")
            yield payload


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Error HTTP {exc.code} llamando {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo conectar con {url}: {exc.reason}") from exc


def _post_bytes(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_seconds: int,
) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Error HTTP {exc.code} llamando {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo conectar con {url}: {exc.reason}") from exc


def _post_stream(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_seconds: int,
    chunk_size: int = 8192,
) -> Iterable[bytes]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Error HTTP {exc.code} llamando {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo conectar con {url}: {exc.reason}") from exc


def _extract_openai_text(response: Mapping[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, Mapping):
            continue
        for content in item.get("content", []):
            if isinstance(content, Mapping):
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    if chunks:
        return " ".join(chunks)
    raise RuntimeError(f"La respuesta de OpenAI no contiene texto: {response}")


def _clean_commentary_text(text: str) -> str:
    cleaned = " ".join(text.replace("\n", " ").split())
    return cleaned.strip('"')


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta configurar {name} en .env o en variables de entorno.")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_score(value: Any) -> Mapping[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    score: dict[str, int] = {}
    for team, goals in value.items():
        try:
            score[str(team)] = int(goals)
        except (TypeError, ValueError):
            continue
    return score or None


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in {"none", "null"}:
        return None
    return int(stripped)


def _normalize_action_type(value: str) -> str:
    return _slugify(value).replace("-", "_")


def _slugify(value: str) -> str:
    text = value.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")
