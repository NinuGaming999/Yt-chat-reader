from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GpuOnlySpeechTests(unittest.TestCase):
    def test_worker_compiles(self):
        source = (ROOT / "vibevoice_worker.py").read_text(encoding="utf-8")
        compile(source, "vibevoice_worker.py", "exec")

    def test_worker_requires_cuda_without_cpu_fallback(self):
        source = (ROOT / "vibevoice_worker.py").read_text(encoding="utf-8")
        self.assertIn("torch.cuda.is_available()", source)
        self.assertIn('device_map="cuda"', source)
        self.assertNotIn("CPUExecutionProvider", source)
        self.assertNotIn('device_map="cpu"', source)

    def test_engine_uses_nvidia_cuda_and_vibevoice_presets(self):
        source = (ROOT / "speech_engine.py").read_text(encoding="utf-8")
        compile(source, "speech_engine.py", "exec")
        self.assertIn("NVIDIA CUDA", source)
        self.assertIn("en-Carter_man", source)
        self.assertIn("en-Emma_woman", source)


if __name__ == "__main__":
    unittest.main()
