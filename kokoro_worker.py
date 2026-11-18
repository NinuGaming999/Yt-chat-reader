"""Isolated Kokoro-v1.0 streaming worker using NVIDIA ONNX Runtime CUDA only.

stdout is a JSON-line control channel; model and audio logs go to stderr.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr

MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.1/kokoro-v1.0.fp16.onnx"
)
VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.1/voices-v1.0.bin"
)
MODEL_SHA256 = "f3a290d384fbb27966d462905c71a46cef9e5fd00516b40df32a0b4afe77ac96"
VOICES_SHA256 = "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"
SAMPLE_RATE = 24_000
MODEL_DIR = Path(os.environ.get("KOKORO_MODEL_DIR", "models/kokoro-v1.0"))
MODEL_FILE = MODEL_DIR / "kokoro-v1.0.fp16.onnx"
VOICES_FILE = MODEL_DIR / "voices-v1.0.bin"
VOICE_LANGUAGES = {
    "af_heart": "en-us", "af_bella": "en-us", "af_nicole": "en-us",
    "af_sarah": "en-us", "af_sky": "en-us", "af_alloy": "en-us",
    "af_aoede": "en-us", "af_jessica": "en-us", "af_kore": "en-us",
    "af_nova": "en-us", "af_river": "en-us", "am_michael": "en-us",
    "am_fenrir": "en-us", "am_puck": "en-us", "am_adam": "en-us",
    "am_echo": "en-us", "am_eric": "en-us", "am_liam": "en-us",
    "am_onyx": "en-us", "bf_emma": "en-gb", "bf_isabella": "en-gb",
    "bf_alice": "en-gb", "bf_lily": "en-gb", "bm_george": "en-gb",
    "bm_lewis": "en-gb", "bm_fable": "en-gb", "bm_daniel": "en-gb",
}
output_lock = threading.Lock()


def emit(event: str, **data) -> None:
    with output_lock:
        PROTOCOL_OUT.write(json.dumps({"event": event, **data}, ensure_ascii=False) + "\n")
        PROTOCOL_OUT.flush()


def status(message: str) -> None:
    emit("status", status=message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "NinuGamingLiveAssistant/2.3"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, part.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if not part.is_file() or part.stat().st_size < 1_000_000:
            raise RuntimeError(f"Download was incomplete: {destination.name}")
        actual = sha256_file(part)
        if actual != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {destination.name}: expected {expected_sha256}, got {actual}"
            )
        part.replace(destination)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def ensure_model_files() -> None:
    status("Checking Kokoro-v1.0 model files…")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_FILE.is_file() or sha256_file(MODEL_FILE) != MODEL_SHA256:
        status("Downloading Kokoro-v1.0 FP16 model (about 156 MB)…")
        download_verified(MODEL_URL, MODEL_FILE, MODEL_SHA256)
    if not VOICES_FILE.is_file() or sha256_file(VOICES_FILE) != VOICES_SHA256:
        status("Downloading Kokoro v1.0 voice pack (about 27 MB)…")
        download_verified(VOICES_URL, VOICES_FILE, VOICES_SHA256)


def gpu_name() -> str:
    smi = os.environ.get("WINDIR", r"C:\Windows")
    candidates = [
        "nvidia-smi",
        str(Path(smi) / "System32" / "nvidia-smi.exe"),
        str(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"),
    ]
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().splitlines()[0]
        except Exception:
            continue
    return "NVIDIA GPU"


class Runtime:
    def __init__(self):
        import numpy as np
        import onnxruntime as ort
        import sounddevice as sd
        from kokoro_onnx import Kokoro

        self.np = np
        self.ort = ort
        self.sd = sd
        self.device = "CUDAExecutionProvider"
        self.voice = "af_heart"
        self.speed = 1.0
        self.volume = 100
        self.lock = threading.RLock()
        self.active_stop: threading.Event | None = None
        self.model = None
        self.session = None
        self.backend = "NVIDIA ONNX Runtime CUDA"
        self.precision = "FP16 ONNX"

        if os.environ.get("ONNX_PROVIDER", self.device) != self.device:
            raise RuntimeError("Kokoro is configured for CUDAExecutionProvider only.")
        available = ort.get_available_providers()
        if self.device not in available:
            raise RuntimeError(
                "ONNX Runtime cannot find CUDAExecutionProvider. "
                f"Available providers: {available}. CPU fallback is disabled."
            )
        try:
            ort.preload_dlls(directory="")
        except Exception as exc:
            raise RuntimeError(f"Could not load NVIDIA CUDA/cuDNN runtime libraries: {exc}") from exc

        ensure_model_files()
        status("Loading Kokoro-v1.0 FP16 on the NVIDIA GPU…")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 4))
        # Never silently execute unsupported graph nodes on the CPU.
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        try:
            self.session = ort.InferenceSession(
                str(MODEL_FILE),
                sess_options=options,
                providers=[self.device],
            )
        except Exception as exc:
            raise RuntimeError(
                "Kokoro could not initialize its CUDA-only ONNX session. "
                "Check the NVIDIA driver and CUDA/cuDNN runtime. CPU fallback is disabled. "
                f"Details: {exc}"
            ) from exc
        active_providers = self.session.get_providers()
        if not active_providers or active_providers[0] != self.device:
            raise RuntimeError(
                f"Kokoro did not start on NVIDIA CUDA. Active providers: {active_providers}"
            )
        self.model = Kokoro.from_session(self.session, str(VOICES_FILE))
        available_voices = set(self.model.get_voices())
        missing = sorted(set(VOICE_LANGUAGES) - available_voices)
        if missing:
            raise RuntimeError(f"Kokoro voice pack is missing presets: {', '.join(missing)}")

        status("Warming up Kokoro CUDA inference…")
        self.model.create(
            "NinuGaming Live Assistant is ready.",
            voice=self.voice, speed=1.0, lang="en-us",
        )
        status("Kokoro CUDA model ready.")

    def configure(self, message: dict) -> None:
        with self.lock:
            voice = str(message.get("voice") or self.voice)
            if voice in VOICE_LANGUAGES:
                self.voice = voice
            try:
                self.speed = max(0.70, min(1.40, float(message.get("speed", self.speed))))
                self.volume = max(0, min(100, int(message.get("volume", self.volume))))
            except (TypeError, ValueError):
                pass

    def speak(self, text: str) -> int:
        with self.lock:
            voice = self.voice
            speed = self.speed
            volume = self.volume
        stop_event = threading.Event()
        with self.lock:
            self.active_stop = stop_event
        try:
            return self._speak_stream(text, voice, speed, volume, stop_event)
        finally:
            with self.lock:
                if self.active_stop is stop_event:
                    self.active_stop = None

    def _speak_stream(
        self,
        text: str,
        voice: str,
        speed: float,
        volume: int,
        stop_event: threading.Event,
    ) -> int:
        audio_queue: queue.Queue[tuple[object, int]] = queue.Queue(maxsize=3)
        producer_done = threading.Event()
        producer_errors: list[BaseException] = []
        sample_count = [0]
        language = VOICE_LANGUAGES[voice]

        def producer() -> None:
            async def generate() -> None:
                try:
                    stream = self.model.create_stream(
                        text,
                        voice=voice,
                        speed=speed,
                        lang=language,
                        sentence_pause=0.22,
                        clause_pause=0.08,
                    )
                    async for samples, sample_rate in stream:
                        if stop_event.is_set():
                            break
                        audio = self.np.asarray(samples, dtype=self.np.float32).reshape(-1)
                        if volume <= 0:
                            audio = self.np.zeros_like(audio)
                        elif volume < 100:
                            audio = self.np.clip(audio * (volume / 100.0), -1.0, 1.0)
                        while not stop_event.is_set():
                            try:
                                await asyncio.to_thread(
                                    audio_queue.put, (audio, int(sample_rate)), True, 0.25
                                )
                                sample_count[0] += int(audio.size)
                                break
                            except queue.Full:
                                continue
                except BaseException as exc:
                    producer_errors.append(exc)
                finally:
                    producer_done.set()

            asyncio.run(generate())

        thread = threading.Thread(target=producer, name="kokoro-audio-generator", daemon=True)
        thread.start()
        first = None
        while first is None and not producer_done.is_set() and not stop_event.is_set():
            try:
                first = audio_queue.get(timeout=0.20)
            except queue.Empty:
                continue

        if first is None:
            thread.join(timeout=10)
            if producer_errors:
                raise RuntimeError(f"Kokoro audio generation failed: {producer_errors[0]}") from producer_errors[0]
            if stop_event.is_set():
                return 0
            raise RuntimeError("Kokoro did not produce an audio chunk.")

        first_audio, sample_rate = first
        playback = {"chunk": first_audio, "offset": 0}

        def audio_callback(outdata, frames, _time_info, _status) -> None:
            outdata.fill(0)
            written = 0
            while written < frames:
                chunk = playback["chunk"]
                if chunk is None or playback["offset"] >= len(chunk):
                    try:
                        next_audio, next_rate = audio_queue.get_nowait()
                    except queue.Empty:
                        if producer_done.is_set():
                            raise self.sd.CallbackStop()
                        break
                    if int(next_rate) != int(sample_rate):
                        raise self.sd.CallbackAbort
                    playback["chunk"] = next_audio
                    playback["offset"] = 0
                    chunk = next_audio

                available = len(chunk) - playback["offset"]
                count = min(frames - written, available)
                start = playback["offset"]
                outdata[written:written + count, 0] = chunk[start:start + count]
                playback["offset"] += count
                written += count

            current = playback["chunk"]
            if producer_done.is_set() and audio_queue.empty() and (
                current is None or playback["offset"] >= len(current)
            ):
                raise self.sd.CallbackStop()

        try:
            with self.sd.OutputStream(
                samplerate=int(sample_rate),
                channels=1,
                dtype="float32",
                callback=audio_callback,
                latency="low",
            ) as output:
                while output.active and not stop_event.is_set():
                    time.sleep(0.02)
                if stop_event.is_set():
                    output.abort()
        finally:
            if stop_event.is_set() and thread.is_alive():
                thread.join(timeout=10)
            elif thread.is_alive():
                thread.join(timeout=30)
            if thread.is_alive():
                raise TimeoutError("Kokoro audio generation did not stop cleanly.")

        if producer_errors:
            raise RuntimeError(f"Kokoro audio generation failed: {producer_errors[0]}") from producer_errors[0]
        return sample_count[0]

    def stop(self) -> None:
        with self.lock:
            event = self.active_stop
        if event is not None:
            event.set()


def main():
    try:
        runtime = Runtime()
    except Exception as exc:
        emit("error", message=f"{type(exc).__name__}: {exc}")
        return

    emit(
        "ready",
        provider="NVIDIA CUDAExecutionProvider",
        precision=runtime.precision,
        gpu_name=gpu_name(),
        backend="Kokoro-v1.0 FP16 · CUDA-only · eSpeak NG streaming",
    )
    active = None
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        command = message.get("cmd")
        if command == "configure":
            try:
                runtime.configure(message)
            except Exception as exc:
                emit("error", message=f"Voice configuration failed: {exc}")
        elif command == "stop":
            runtime.stop()
        elif command == "speak":
            if active is not None and active.is_alive():
                emit("speech_done", success=False, error="Kokoro is already speaking.")
                continue
            text = str(message.get("text") or "").strip()

            def read_message() -> None:
                emit("status", status="Speaking with Kokoro NVIDIA CUDA…")
                try:
                    sample_count = runtime.speak(text)
                    if sample_count <= 0:
                        emit("speech_done", success=False, error="Kokoro produced no audio.")
                    else:
                        emit("speech_done", success=True, error="")
                except Exception as exc:
                    emit("speech_done", success=False, error=f"{type(exc).__name__}: {exc}")

            active = threading.Thread(target=read_message, name="kokoro-playback", daemon=True)
            active.start()
        elif command == "shutdown":
            runtime.stop()
            break

    if active is not None and active.is_alive():
        active.join(timeout=5)


if __name__ == "__main__":
    main()
