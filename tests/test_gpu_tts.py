from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GpuOnlySpeechTests(unittest.TestCase):
    def test_kokoro_worker_compiles(self):
        source = (ROOT / "kokoro_worker.py").read_text(encoding="utf-8")
        compile(source, "kokoro_worker.py", "exec")

    def test_kokoro_uses_cuda_and_disables_cpu_execution_fallback(self):
        source = (ROOT / "kokoro_worker.py").read_text(encoding="utf-8")
        self.assertIn("CUDAExecutionProvider", source)
        self.assertIn("session.disable_cpu_ep_fallback", source)
        self.assertNotIn("CPUExecutionProvider", source)
        self.assertIn("Kokoro.from_session", source)
        self.assertIn("create_stream(", source)

    def test_kokoro_model_and_voice_integrity_are_pinned(self):
        source = (ROOT / "kokoro_worker.py").read_text(encoding="utf-8")
        self.assertIn("kokoro-v1.0.fp16.onnx", source)
        self.assertIn("voices-v1.0.bin", source)
        self.assertIn("MODEL_SHA256", source)
        self.assertIn("VOICES_SHA256", source)

    def test_engine_bootstraps_kokoro_and_nvidia_onnx_runtime(self):
        source = (ROOT / "speech_engine.py").read_text(encoding="utf-8")
        compile(source, "speech_engine.py", "exec")
        self.assertIn("onnxruntime-gpu", source)
        self.assertIn("CUDAExecutionProvider", source)
        self.assertIn("af_heart", source)
        self.assertIn("bf_emma", source)
        self.assertNotIn("torch==", source)


if __name__ == "__main__":
    unittest.main()
