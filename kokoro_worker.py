"""Kokoro-v1.0 speech worker using native PyTorch NVIDIA CUDA inference only.

Text normalization/phonemization and audio playback use the CPU as normal desktop
I/O tasks. The neural Kokoro model and every model parameter remain on CUDA.
stdout is a JSON-line control channel; diagnostics go to stderr.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr

MODEL_REPO = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24_000
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


def gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "NVIDIA GPU"


class Runtime:
    def __init__(self):
        import numpy as np
        import sounddevice as sd
        import torch
        from kokoro import KModel, KPipeline

        self.np = np
        self.sd = sd
        self.torch = torch
        self.voice = "af_heart"
        self.speed = 1.0
        self.volume = 100
        self.lock = threading.RLock()
        self.active_stop: threading.Event | None = None
        self.device = torch.device("cuda:0")
        self.model = None
        self.pipelines = {}
        self.backend = "PyTorch NVIDIA CUDA"
        self.precision = "FP32 CUDA"

        if torch.version.cuda is None:
            raise RuntimeError(
                "The installed PyTorch build has no CUDA support. "
                "Install the NVIDIA CUDA build; CPU model inference is disabled."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch cannot initialize NVIDIA CUDA. Check the NVIDIA driver and "
                "the installed CUDA-enabled PyTorch runtime. CPU model inference is disabled."
            )

        torch.cuda.init()
        torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        status("Loading Kokoro-v1.0 weights for NVIDIA CUDA…")
        self.model = KModel(repo_id=MODEL_REPO).to(self.device).eval()
        parameter_devices = {str(parameter.device) for parameter in self.model.parameters()}
        buffer_devices = {str(buffer.device) for buffer in self.model.buffers()}
        if not parameter_devices or parameter_devices != {str(self.device)}:
            raise RuntimeError(
                "Kokoro model parameters are not all on CUDA: " + repr(sorted(parameter_devices))
            )
        if buffer_devices and buffer_devices != {str(self.device)}:
            raise RuntimeError(
                "Kokoro model buffers are not all on CUDA: " + repr(sorted(buffer_devices))
            )

        status("Preparing American and British English Kokoro pipelines…")
        self.pipelines = {
            "en-us": KPipeline(lang_code="a", repo_id=MODEL_REPO, model=self.model),
            "en-gb": KPipeline(lang_code="b", repo_id=MODEL_REPO, model=self.model),
        }

        status("Loading the default voice and warming up Kokoro on the GPU…")
        self.pipelines["en-us"].load_voice(self.voice)
        warmup = list(self.pipelines["en-us"](
            "NinuGaming Live Assistant is ready.",
            voice=self.voice,
            speed=1.0,
        ))
        if not warmup or warmup[0].audio is None or warmup[0].audio.numel() == 0:
            raise RuntimeError("Kokoro CUDA warm-up did not produce audio.")
        torch.cuda.synchronize()
        if str(self.model.device) != str(self.device):
            raise RuntimeError(f"Kokoro moved away from CUDA: {self.model.device}")

        status("Kokoro model ready on NVIDIA CUDA.")

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

    def speak(self, text: str) -> tuple[int, bool]:
        with self.lock:
            voice = self.voice
            speed = self.speed
            volume = self.volume
        stop_event = threading.Event()
        with self.lock:
            self.active_stop = stop_event
        try:
            sample_count = self._speak_stream(text, voice, speed, volume, stop_event)
            return sample_count, stop_event.is_set()
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
        audio_queue: queue.Queue[object] = queue.Queue(maxsize=4)
        producer_done = threading.Event()
        producer_errors: list[BaseException] = []
        sample_count = [0]
        language = VOICE_LANGUAGES[voice]
        pipeline = self.pipelines[language]

        def enqueue(audio) -> None:
            while not stop_event.is_set():
                try:
                    audio_queue.put(audio, timeout=0.20)
                    return
                except queue.Full:
                    continue

        def producer() -> None:
            try:
                with self.torch.inference_mode():
                    for result in pipeline(text, voice=voice, speed=speed):
                        if stop_event.is_set():
                            break
                        if result.audio is None:
                            continue
                        audio = self.np.asarray(
                            result.audio.detach().cpu().numpy(), dtype=self.np.float32
                        ).reshape(-1)
                        if audio.size == 0:
                            continue
                        if not self.np.isfinite(audio).all():
                            raise RuntimeError("Kokoro generated non-finite audio samples.")
                        if volume <= 0:
                            audio = self.np.zeros_like(audio)
                        elif volume < 100:
                            audio = self.np.clip(audio * (volume / 100.0), -1.0, 1.0)
                        enqueue(audio)
                        sample_count[0] += int(audio.size)
                        if not stop_event.is_set():
                            enqueue(self.np.zeros(int(SAMPLE_RATE * 0.10), dtype=self.np.float32))
            except BaseException as exc:
                producer_errors.append(exc)
            finally:
                producer_done.set()

        thread = threading.Thread(target=producer, name="kokoro-cuda-generator", daemon=True)
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
                raise RuntimeError(f"Kokoro CUDA generation failed: {producer_errors[0]}") from producer_errors[0]
            if stop_event.is_set():
                return 0
            raise RuntimeError("Kokoro did not produce an audio chunk.")

        playback = {"chunk": first, "offset": 0}

        def audio_callback(outdata, frames, _time_info, _status) -> None:
            outdata.fill(0)
            if stop_event.is_set():
                raise self.sd.CallbackAbort
            written = 0
            while written < frames:
                chunk = playback["chunk"]
                if chunk is None or playback["offset"] >= len(chunk):
                    try:
                        chunk = audio_queue.get_nowait()
                    except queue.Empty:
                        if producer_done.is_set():
                            raise self.sd.CallbackStop()
                        break
                    playback["chunk"] = chunk
                    playback["offset"] = 0

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
                samplerate=SAMPLE_RATE,
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
            if thread.is_alive():
                thread.join(timeout=45)
            if thread.is_alive():
                raise TimeoutError("Kokoro CUDA generation did not stop cleanly.")

        if producer_errors:
            raise RuntimeError(f"Kokoro CUDA generation failed: {producer_errors[0]}") from producer_errors[0]
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
        provider="PyTorch CUDA",
        precision=runtime.precision,
        gpu_name=gpu_name(),
        backend="Kokoro-v1.0 · native PyTorch · CUDA-only model inference",
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
                emit("status", status="Speaking with Kokoro on NVIDIA CUDA…")
                try:
                    sample_count, cancelled = runtime.speak(text)
                    if cancelled:
                        emit("speech_done", success=False, error="")
                    elif sample_count <= 0:
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
