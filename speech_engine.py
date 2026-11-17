"""Local Kokoro neural TTS backend for NinuGaming Live Assistant.

The app installer places Kokoro model files in ProgramData. On Windows,
ONNX Runtime DirectML is preferred so supported GPUs can accelerate inference
without requiring a separate CUDA/cuDNN installation. CPU uses the int8 model.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Callable

APP_DATA_NAME = "NinuGamingLiveAssistant"
DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0
DEFAULT_VOLUME = 100

# The English voice set shipped in voices-v1.0.bin. Voice IDs are upstream Kokoro IDs.
VOICE_CATALOG = [
    {"id": "af_heart", "label": "Heart · American · Female", "language": "en-us"},
    {"id": "af_bella", "label": "Bella · American · Female", "language": "en-us"},
    {"id": "af_nicole", "label": "Nicole · American · Female", "language": "en-us"},
    {"id": "af_sarah", "label": "Sarah · American · Female", "language": "en-us"},
    {"id": "af_sky", "label": "Sky · American · Female", "language": "en-us"},
    {"id": "af_alloy", "label": "Alloy · American · Female", "language": "en-us"},
    {"id": "af_aoede", "label": "Aoede · American · Female", "language": "en-us"},
    {"id": "af_jessica", "label": "Jessica · American · Female", "language": "en-us"},
    {"id": "af_kore", "label": "Kore · American · Female", "language": "en-us"},
    {"id": "af_nova", "label": "Nova · American · Female", "language": "en-us"},
    {"id": "af_river", "label": "River · American · Female", "language": "en-us"},
    {"id": "am_michael", "label": "Michael · American · Male", "language": "en-us"},
    {"id": "am_fenrir", "label": "Fenrir · American · Male", "language": "en-us"},
    {"id": "am_puck", "label": "Puck · American · Male", "language": "en-us"},
    {"id": "am_adam", "label": "Adam · American · Male", "language": "en-us"},
    {"id": "am_echo", "label": "Echo · American · Male", "language": "en-us"},
    {"id": "am_eric", "label": "Eric · American · Male", "language": "en-us"},
    {"id": "am_liam", "label": "Liam · American · Male", "language": "en-us"},
    {"id": "am_onyx", "label": "Onyx · American · Male", "language": "en-us"},
    {"id": "bf_emma", "label": "Emma · British · Female", "language": "en-gb"},
    {"id": "bf_isabella", "label": "Isabella · British · Female", "language": "en-gb"},
    {"id": "bf_alice", "label": "Alice · British · Female", "language": "en-gb"},
    {"id": "bf_lily", "label": "Lily · British · Female", "language": "en-gb"},
    {"id": "bm_george", "label": "George · British · Male", "language": "en-gb"},
    {"id": "bm_lewis", "label": "Lewis · British · Male", "language": "en-gb"},
    {"id": "bm_fable", "label": "Fable · British · Male", "language": "en-gb"},
    {"id": "bm_daniel", "label": "Daniel · British · Male", "language": "en-gb"},
]
VOICE_BY_ID = {item["id"]: item for item in VOICE_CATALOG}


def find_model_directory(app_root: Path) -> Path:
    candidates: list[Path] = []
    program_data = Path(__import__("os").environ.get("PROGRAMDATA", r"C:\ProgramData"))
    candidates.append(program_data / APP_DATA_NAME / "models")
    candidates.append(Path.home() / "AppData" / "Roaming" / APP_DATA_NAME / "models")
    candidates.append(app_root / "models")
    for candidate in candidates:
        if (candidate / "voices-v1.0.bin").is_file():
            return candidate
    return candidates[0]


class NeuralSpeechEngine:
    """Persistent Kokoro runtime, warm-started once and reused for every chat line."""

    def __init__(
        self,
        app_root: Path,
        on_status: Callable[[str, str], None] | None = None,
        voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_SPEED,
        volume: int = DEFAULT_VOLUME,
    ) -> None:
        self.model_dir = find_model_directory(app_root)
        self.voice = voice if voice in VOICE_BY_ID else DEFAULT_VOICE
        self.speed = self._clamp_speed(speed)
        self.volume = self._clamp_volume(volume)
        self.on_status = on_status
        self.lock = threading.RLock()
        self.model = None
        self.session = None
        self.ready = False
        self.status = "Loading Kokoro neural voice model…"
        self.provider = "Initializing"
        self.model_precision = "Not loaded"
        self.phonemizer = "Kokoro eSpeak NG"
        self.error = ""
        self.cancel_event = threading.Event()
        self._g2p_us = None
        self._g2p_gb = None
        self.loader = threading.Thread(target=self._load, name="kokoro-model-loader", daemon=True)
        self.loader.start()

    @staticmethod
    def _clamp_speed(speed: float) -> float:
        return max(0.70, min(1.40, float(speed)))

    @staticmethod
    def _clamp_volume(volume: int) -> int:
        return max(0, min(100, int(volume)))

    def _notify(self, status: str, level: str = "info") -> None:
        self.status = status
        if self.on_status:
            try:
                self.on_status(status, level)
            except Exception:
                pass

    def _load_g2p(self) -> None:
        # kokoro-onnx bundles a tokenizer backed by espeakng-loader. Avoid the
        # optional Misaki English extras here: they pull in heavyweight NLP
        # stacks that are unnecessary for chat-message speech.
        self._g2p_us = self._g2p_gb = None
        self.phonemizer = "Kokoro eSpeak NG tokenizer"

    @staticmethod
    def _make_session(ort, model_path: Path, provider: str):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if provider == "DmlExecutionProvider":
            # DirectML requires sequential graph execution and memory-pattern
            # allocation to be disabled by ONNX Runtime.
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.enable_mem_pattern = False
            return ort.InferenceSession(
                str(model_path), sess_options=options,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            )
        return ort.InferenceSession(
            str(model_path), sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def _load(self) -> None:
        try:
            from kokoro_onnx import Kokoro
            import onnxruntime as ort
            import sounddevice  # noqa: F401 - fail during setup, not mid-stream
        except Exception as exc:
            self.error = f"Neural TTS dependencies are missing: {exc}"
            self._notify(self.error, "error")
            return

        voice_file = self.model_dir / "voices-v1.0.bin"
        fp16_file = self.model_dir / "kokoro-v1.0.fp16.onnx"
        int8_file = self.model_dir / "kokoro-v1.0.int8.onnx"
        if not voice_file.is_file():
            self.error = f"Kokoro voice pack is missing: {voice_file}"
            self._notify(self.error + " Re-run the NinuGaming Live Assistant installer.", "error")
            return

        available = set(ort.get_available_providers())
        attempts: list[tuple[str, Path, str]] = []
        if "DmlExecutionProvider" in available and fp16_file.is_file():
            attempts.append(("DmlExecutionProvider", fp16_file, "FP16 high-quality GPU model"))
        if int8_file.is_file():
            attempts.append(("CPUExecutionProvider", int8_file, "INT8 CPU fallback model"))
        if fp16_file.is_file():
            attempts.append(("CPUExecutionProvider", fp16_file, "FP16 CPU fallback model"))

        if not attempts:
            self.error = f"Kokoro model files are missing from {self.model_dir}"
            self._notify(self.error + " Re-run the NinuGaming Live Assistant installer.", "error")
            return

        last_error = None
        for provider, model_file, description in attempts:
            try:
                self._notify(f"Loading {description}…")
                session = self._make_session(ort, model_file, provider)
                model = Kokoro.from_session(session, str(voice_file))
                self.session = session
                self.model = model
                selected_providers = session.get_providers()
                self.provider = selected_providers[0] if selected_providers else provider
                self.model_precision = "FP16" if "fp16" in model_file.name else "INT8"
                self._load_g2p()
                # Warm the graph once so the first viewer message does not pay
                # the model's one-time initialization cost.
                self._notify("Warming up Kokoro voice model…")
                model.create("NinuGaming Live Assistant is ready.", voice=DEFAULT_VOICE, speed=1.0, lang="en-us")
                self.ready = True
                self.error = ""
                self._notify(f"Kokoro ready · {self.provider} · {self.model_precision} · {self.phonemizer}")
                return
            except Exception as exc:
                last_error = exc
                self.model = None
                self.session = None
                self._notify(f"{provider} could not load the model; trying the next safe backend: {exc}", "warning")

        self.error = f"Kokoro could not start on GPU or CPU: {last_error}"
        self._notify(self.error, "error")

    def configure(self, voice: str | None = None, speed: float | None = None, volume: int | None = None) -> dict:
        with self.lock:
            if voice is not None:
                self.voice = voice if voice in VOICE_BY_ID else DEFAULT_VOICE
            if speed is not None:
                self.speed = self._clamp_speed(speed)
            if volume is not None:
                self.volume = self._clamp_volume(volume)
            return self.settings()

    def settings(self) -> dict:
        with self.lock:
            return {
                "voice": self.voice,
                "voice_label": VOICE_BY_ID[self.voice]["label"],
                "speed": self.speed,
                "volume": self.volume,
                "ready": self.ready,
                "status": self.status,
                "provider": self.provider,
                "model_precision": self.model_precision,
                "phonemizer": self.phonemizer,
                "model_directory": str(self.model_dir),
                "error": self.error,
                "voices": VOICE_CATALOG,
            }

    def speak(self, text: str) -> bool:
        """Generate audio ahead while a single output stream plays it continuously.

        A producer queue prevents the previous implementation's per-chunk
        play/wait cycle from inserting avoidable gaps between Kokoro chunks.
        """
        if not self.ready or self.model is None:
            raise RuntimeError(self.error or self.status or "Kokoro voice model is not ready yet.")
        import queue
        import numpy as np
        import sounddevice as sd

        with self.lock:
            voice = self.voice
            speed = self.speed
            volume = self.volume
        self.cancel_event.clear()
        voice_info = VOICE_BY_ID[voice]
        language = voice_info["language"]
        g2p = self._g2p_gb if language == "en-gb" else self._g2p_us
        is_phonemes = False
        if g2p is not None:
            phonemes, _tokens = g2p(text)
            speech_input = phonemes
            is_phonemes = True
        else:
            speech_input = text

        audio_queue: queue.Queue[tuple[object, int]] = queue.Queue(maxsize=5)
        producer_done = threading.Event()
        producer_errors: list[BaseException] = []

        async def produce() -> None:
            try:
                stream = self.model.create_stream(
                    speech_input, voice=voice, speed=speed,
                    lang=language, is_phonemes=is_phonemes,
                )
                async for samples, sample_rate in stream:
                    if self.cancel_event.is_set():
                        break
                    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
                    if volume <= 0:
                        audio = np.zeros_like(audio)
                    elif volume < 100:
                        audio = np.clip(audio * (volume / 100.0), -1.0, 1.0)
                    # A small bounded queue balances low latency and generation
                    # headroom without allowing chat messages to consume RAM.
                    while not self.cancel_event.is_set():
                        try:
                            await asyncio.to_thread(audio_queue.put, (audio, int(sample_rate)), True, 0.2)
                            break
                        except queue.Full:
                            continue
                    if self.cancel_event.is_set():
                        break
            except BaseException as exc:
                producer_errors.append(exc)
            finally:
                producer_done.set()

        async def play() -> bool:
            producer_task = asyncio.create_task(produce())
            first_chunk = None
            while not self.cancel_event.is_set() and first_chunk is None:
                try:
                    first_chunk = await asyncio.to_thread(audio_queue.get, True, 0.2)
                except queue.Empty:
                    if producer_done.is_set():
                        break
            if first_chunk is None:
                await producer_task
                if producer_errors:
                    raise producer_errors[0]
                return False

            sample_rate = first_chunk[1]
            playback = {"chunk": first_chunk[0], "offset": 0}

            def audio_callback(outdata, frames, _time_info, _status):
                outdata.fill(0)
                written = 0
                while written < frames:
                    chunk = playback["chunk"]
                    if chunk is None or playback["offset"] >= len(chunk):
                        try:
                            next_chunk, next_rate = audio_queue.get_nowait()
                            if int(next_rate) != sample_rate:
                                raise RuntimeError("Kokoro changed the sample rate during one utterance.")
                            playback["chunk"] = next_chunk
                            playback["offset"] = 0
                            chunk = next_chunk
                        except queue.Empty:
                            if producer_done.is_set():
                                raise sd.CallbackStop()
                            break
                    available = len(playback["chunk"]) - playback["offset"]
                    count = min(frames - written, available)
                    begin = playback["offset"]
                    outdata[written:written + count, 0] = playback["chunk"][begin:begin + count]
                    playback["offset"] += count
                    written += count
                if producer_done.is_set() and audio_queue.empty() and (
                    playback["chunk"] is None or playback["offset"] >= len(playback["chunk"])
                ):
                    raise sd.CallbackStop()

            try:
                with sd.OutputStream(
                    samplerate=sample_rate, channels=1, dtype="float32",
                    callback=audio_callback, latency="low",
                ) as output_stream:
                    while output_stream.active and not self.cancel_event.is_set():
                        await asyncio.sleep(0.02)
                    if self.cancel_event.is_set():
                        output_stream.abort()
            finally:
                if not producer_task.done():
                    producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
            if producer_errors:
                raise producer_errors[0]
            return not self.cancel_event.is_set()

        return asyncio.run(play())

    def stop_current(self) -> None:
        self.cancel_event.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    def close(self) -> None:
        self.stop_current()