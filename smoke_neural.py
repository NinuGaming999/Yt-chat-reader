"""Headless Kokoro runtime smoke test used by the Windows CI build."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import onnxruntime as ort
from kokoro_onnx import Kokoro

MODEL_DIR = Path(__file__).resolve().parent / "models"
VOICE_FILE = MODEL_DIR / "voices-v1.0.bin"
MODEL_FILE = MODEL_DIR / "kokoro-v1.0.int8.onnx"

missing = [str(path) for path in (VOICE_FILE, MODEL_FILE) if not path.is_file()]
if missing:
    raise SystemExit("Missing smoke-test model files: " + ", ".join(missing))

session = ort.InferenceSession(str(MODEL_FILE), providers=["CPUExecutionProvider"])
model = Kokoro.from_session(session, str(VOICE_FILE))

async def check_stream() -> tuple[int, int]:
    chunk_count = 0
    sample_count = 0
    async for samples, sample_rate in model.create_stream(
        "Hello from NinuGaming Live Assistant. This is a natural neural voice test.",
        voice="af_heart", speed=1.0, lang="en-us",
    ):
        chunk_count += 1
        sample_count += len(samples)
        if sample_rate != 24000:
            raise AssertionError(f"Unexpected sample rate: {sample_rate}")
    return chunk_count, sample_count

chunks, samples = asyncio.run(check_stream())
if chunks < 1 or samples < 1000:
    raise AssertionError(f"Kokoro produced too little audio: {chunks=} {samples=}")
print(f"Kokoro stream smoke test passed: {chunks} chunks, {samples} samples, CPU provider")
try:
    model.voices.close()
except Exception:
    pass