"""Isolated CUDA-only VibeVoice Realtime worker.

stdout is a JSON-line control channel; model logs are redirected to stderr.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import threading
import traceback
from pathlib import Path

PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr

MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
UPSTREAM_SHA = "16fb2cb1217c9934a886e1948ffb06120caa2df5"
SAMPLE_RATE = 24_000
VOICE_FILES = {
    "en-Carter_man": ("1d795ef667e6641eecb8b22452bb853b089bfdbe", 4256002),
    "en-Davis_man": ("a259516c4d777016dbdc6ed925389cb35f2a6e4b", 2471258),
    "en-Emma_woman": ("9876ca31f3fad23cea06ea4b73bfea799c3c747b", 3343090),
    "en-Frank_man": ("2c37e82fc01855700bcf1f9389cc6ac11f54d789", 3359578),
    "en-Grace_woman": ("366bda87ee88da801999dd649d95f1f685cd63b5", 2772466),
    "en-Mike_man": ("f02ef03fb2c325c2be20c4c3e83b41e0a423b5a4", 2016234),
}
MODEL_DIR = Path(os.environ.get("VIBEVOICE_MODEL_DIR", "models/vibevoice"))
DATA_DIR = Path(os.environ.get("VIBEVOICE_DATA_DIR", "data"))
VOICE_DIR = DATA_DIR / "voices"
output_lock = threading.Lock()


def emit(event: str, **data) -> None:
    with output_lock:
        PROTOCOL_OUT.write(json.dumps({"event":event, **data}, ensure_ascii=False) + "\n")
        PROTOCOL_OUT.flush()


def status(message: str) -> None:
    emit("status", status=message)


def git_blob_sha1(content: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()


def ensure_files() -> None:
    import requests
    from huggingface_hub import snapshot_download

    status("Downloading VibeVoice model (about 2 GB, first launch only)…")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        allow_patterns=["*.json", "*.safetensors"],
        max_workers=2,
    )
    if not list(MODEL_DIR.glob("*.safetensors")):
        raise RuntimeError("VibeVoice download ended without model.safetensors.")

    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    for voice_id, (expected_sha, expected_size) in VOICE_FILES.items():
        target = VOICE_DIR / f"{voice_id}.pt"
        if target.is_file() and target.stat().st_size == expected_size:
            if git_blob_sha1(target.read_bytes()) == expected_sha:
                continue
        status(f"Downloading voice preset {voice_id}…")
        url = (
            f"https://raw.githubusercontent.com/microsoft/VibeVoice/{UPSTREAM_SHA}/"
            f"demo/voices/streaming_model/{voice_id}.pt"
        )
        response = requests.get(url, timeout=(20, 120))
        response.raise_for_status()
        payload = response.content
        if len(payload) != expected_size or git_blob_sha1(payload) != expected_sha:
            raise RuntimeError(f"Integrity check failed for VibeVoice voice {voice_id}.")
        temp = target.with_suffix(".pt.download")
        temp.write_bytes(payload)
        temp.replace(target)


class Runtime:
    def __init__(self):
        import torch
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from vibevoice.modular.streamer import AudioStreamer
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

        self.torch = torch
        self.DynamicCache = DynamicCache
        self.BaseModelOutputWithPast = BaseModelOutputWithPast
        self.AudioStreamer = AudioStreamer
        self.device = torch.device("cuda:0")
        self.voice = "en-Carter_man"
        self.speed = 1.0
        self.volume = 100
        self.voice_cache = {}
        self.active_stop = None
        self.model = None
        self.processor = None
        self.precision = "BF16"

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access an NVIDIA CUDA GPU. CPU fallback is disabled.")
        if not torch.cuda.is_bf16_supported():
            self.precision = "FP16"
        self.gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        if capability[0] < 7:
            raise RuntimeError(f"{self.gpu_name} has unsupported CUDA compute capability {capability}.")
        free, total = torch.cuda.mem_get_info(0)
        if free < 3 * 1024**3:
            status(f"Only {free / 1024**3:.1f} GB GPU memory is free; close GPU-heavy applications if loading fails.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        ensure_files()
        status(f"Loading VibeVoice on {self.gpu_name} using {self.precision}…")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(str(MODEL_DIR))
        dtype = torch.bfloat16 if self.precision == "BF16" else torch.float16
        self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            str(MODEL_DIR), torch_dtype=dtype, device_map="cuda",
            attn_implementation="sdpa", low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=5)
        self.get_voice(self.voice)
        status("Warming up the VibeVoice CUDA stream…")
        self.generate("NinuGaming voice is ready.", self.voice, threading.Event(), discard=True, warmup=True)
        torch.cuda.synchronize()
        status("VibeVoice CUDA model ready.")

    def configure(self, message: dict) -> None:
        if message.get("voice") in VOICE_FILES:
            self.voice = message["voice"]
        try:
            self.speed = max(0.70, min(1.40, float(message.get("speed", self.speed))))
            self.volume = max(0, min(100, int(message.get("volume", self.volume))))
        except (TypeError, ValueError):
            pass
        self.get_voice(self.voice)

    def get_voice(self, voice_id: str):
        if voice_id not in self.voice_cache:
            path = VOICE_DIR / f"{voice_id}.pt"
            with self.torch.serialization.safe_globals([self.BaseModelOutputWithPast, self.DynamicCache]):
                self.voice_cache[voice_id] = self.torch.load(
                    path, map_location=self.device, weights_only=True
                )
        return self.voice_cache[voice_id]

    def inference_steps(self) -> int:
        # Low setting = fewer diffusion steps and less GPU work.
        return max(3, min(8, int(round(3 + (self.speed - 0.70) * (5 / 0.70)))))

    def generate(self, text: str, voice_id: str, stop_event: threading.Event,
                 discard: bool = False, warmup: bool = False) -> int:
        import numpy as np
        import sounddevice as sd

        prompt = self.get_voice(voice_id)
        inputs = self.processor.process_input_with_cached_prompt(
            text=text.strip(), cached_prompt=prompt, padding=True,
            return_tensors="pt", return_attention_mask=True,
        )
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(self.device)
        self.model.set_ddpm_inference_steps(num_steps=3 if warmup else self.inference_steps())
        streamer = self.AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors = []

        def produce():
            try:
                with self.torch.inference_mode():
                    self.model.generate(
                        **inputs, max_new_tokens=None, cfg_scale=1.5,
                        tokenizer=self.processor.tokenizer,
                        generation_config={"do_sample": False},
                        audio_streamer=streamer,
                        stop_check_fn=stop_event.is_set,
                        verbose=False, refresh_negative=True,
                        all_prefilled_outputs=copy.deepcopy(prompt),
                    )
            except BaseException as exc:
                errors.append(exc)
                traceback.print_exc(file=sys.stderr)
            finally:
                streamer.end()

        thread = threading.Thread(target=produce, name="vibevoice-generation", daemon=True)
        thread.start()
        count = 0
        output = None
        try:
            for chunk in streamer.get_stream(0):
                if stop_event.is_set():
                    break
                if chunk is None:
                    continue
                if hasattr(chunk, "detach"):
                    audio = chunk.detach().to(device="cpu", dtype=self.torch.float32).numpy()
                else:
                    audio = np.asarray(chunk, dtype=np.float32)
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                if discard:
                    count += int(audio.size)
                    continue
                if self.volume <= 0:
                    audio.fill(0)
                elif self.volume < 100:
                    audio *= self.volume / 100.0
                audio = np.clip(audio, -1.0, 1.0)
                if not audio.size:
                    continue
                count += int(audio.size)
                if output is None:
                    output = sd.OutputStream(
                        samplerate=SAMPLE_RATE, channels=1, dtype="float32", latency="low"
                    )
                    output.start()
                output.write(audio.reshape(-1, 1))
        finally:
            if stop_event.is_set():
                try:
                    sd.stop()
                except Exception:
                    pass
            if output is not None:
                try:
                    output.stop()
                    output.close()
                except Exception:
                    pass
            thread.join(timeout=30)
            if thread.is_alive():
                raise TimeoutError("VibeVoice generation did not stop cleanly.")
        if errors:
            if "out of memory" in str(errors[0]).lower():
                self.torch.cuda.empty_cache()
                raise RuntimeError(
                    "GPU memory exhausted. Close GPU-heavy apps or reduce GPU compute/quality."
                ) from errors[0]
            raise RuntimeError(f"VibeVoice generation failed: {errors[0]}") from errors[0]
        return count

    def speak(self, text: str) -> None:
        stop_event = threading.Event()
        self.active_stop = stop_event
        try:
            samples = self.generate(text, self.voice, stop_event)
            if samples <= 0 and not stop_event.is_set():
                raise RuntimeError("VibeVoice did not produce any audio.")
        finally:
            self.active_stop = None

    def stop(self) -> None:
        if self.active_stop:
            self.active_stop.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass


def main():
    try:
        runtime = Runtime()
    except Exception as exc:
        emit("error", message=f"{type(exc).__name__}: {exc}")
        return
    emit("ready", provider="NVIDIA CUDA", precision=runtime.precision,
         gpu_name=runtime.gpu_name, backend="VibeVoice Realtime · SDPA")
    active = None
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        command = msg.get("cmd")
        if command == "configure":
            try:
                runtime.configure(msg)
            except Exception as exc:
                emit("error", message=f"Voice configuration failed: {exc}")
        elif command == "stop":
            runtime.stop()
        elif command == "speak":
            if active is not None and active.is_alive():
                emit("speech_done", success=False, error="VibeVoice is already speaking.")
                continue
            text = str(msg.get("text") or "").strip()
            def read_message():
                emit("status", status="Speaking with VibeVoice CUDA…")
                try:
                    runtime.speak(text)
                    emit("speech_done", success=True, error="")
                except Exception as exc:
                    emit("speech_done", success=False, error=f"{type(exc).__name__}: {exc}")
            active = threading.Thread(target=read_message, name="vibevoice-playback", daemon=True)
            active.start()
        elif command == "shutdown":
            runtime.stop()
            break
    if active is not None and active.is_alive():
        active.join(timeout=5)


if __name__ == "__main__":
    main()
