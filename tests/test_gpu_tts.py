from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GpuOnlySpeechTests(unittest.TestCase):
    def test_kokoro_worker_compiles(self):
        source = (ROOT / "kokoro_worker.py").read_text(encoding="utf-8")
        compile(source, "kokoro_worker.py", "exec")

    def test_kokoro_requires_native_torch_cuda(self):
        source = (ROOT / "kokoro_worker.py").read_text(encoding="utf-8")
        self.assertIn("torch.version.cuda", source)
        self.assertIn("torch.cuda.is_available()", source)
        self.assertIn('torch.device("cuda:0")', source)
        self.assertIn(".to(self.device).eval()", source)
        self.assertIn("parameter.device", source)
        self.assertIn("KModel(repo_id=MODEL_REPO)", source)
        self.assertIn("KPipeline(lang_code=\"a\"", source)
        self.assertIn("KPipeline(lang_code=\"b\"", source)
        self.assertIn("CPU model inference is disabled", source)
        self.assertNotIn("onnxruntime", source.lower())
        self.assertNotIn("CPUExecutionProvider", source)

    def test_kokoro_uses_all_declared_voice_presets(self):
        source = (ROOT / "kokoro_worker.py").read_text(encoding="utf-8")
        engine = (ROOT / "speech_engine.py").read_text(encoding="utf-8")
        self.assertIn('"af_heart": "en-us"', source)
        self.assertIn('"bf_emma": "en-gb"', source)
        self.assertIn('"bm_daniel": "en-gb"', source)
        self.assertIn('"af_heart"', engine)
        self.assertIn('"bm_daniel"', engine)

    def test_engine_bootstraps_cuda_pytorch_and_native_kokoro(self):
        source = (ROOT / "speech_engine.py").read_text(encoding="utf-8")
        compile(source, "speech_engine.py", "exec")
        self.assertIn("torch==", source)
        self.assertIn("download.pytorch.org/whl/cu128", source)
        self.assertIn("kokoro==", source)
        self.assertIn("HF_HOME", source)
        self.assertIn("torch.cuda.is_available()", source)
        self.assertNotIn("import onnxruntime", source)
        self.assertNotIn("kokoro-onnx==", source)


if __name__ == "__main__":
    unittest.main()
