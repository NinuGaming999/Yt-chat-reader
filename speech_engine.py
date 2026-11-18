"""CUDA-only Kokoro-v1.0 process manager.

Native PyTorch CUDA and Kokoro are installed into a separate Python runtime
on first launch. CPU model inference is never permitted.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

APP_NAME = "NinuGamingLiveAssistant"
DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0
DEFAULT_VOLUME = 100
PYTHON_VERSION = "3.12.10"
PYTHON_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
TORCH_VERSION = "2.11.0"
KOKORO_VERSION = "0.9.4"
MODEL_DIR_NAME = "kokoro-v1.0-pytorch"
RUNTIME_SIGNATURE = (
    f"python-{PYTHON_VERSION};torch-{TORCH_VERSION}-cu128;"
    f"kokoro-{KOKORO_VERSION};PyTorch-CUDA;strict"
)
VOICE_CATALOG = [
    {"id":"af_heart","label":"Heart · American · Female","language":"en-us"},
    {"id":"af_bella","label":"Bella · American · Female","language":"en-us"},
    {"id":"af_nicole","label":"Nicole · American · Female","language":"en-us"},
    {"id":"af_sarah","label":"Sarah · American · Female","language":"en-us"},
    {"id":"af_sky","label":"Sky · American · Female","language":"en-us"},
    {"id":"af_alloy","label":"Alloy · American · Female","language":"en-us"},
    {"id":"af_aoede","label":"Aoede · American · Female","language":"en-us"},
    {"id":"af_jessica","label":"Jessica · American · Female","language":"en-us"},
    {"id":"af_kore","label":"Kore · American · Female","language":"en-us"},
    {"id":"af_nova","label":"Nova · American · Female","language":"en-us"},
    {"id":"af_river","label":"River · American · Female","language":"en-us"},
    {"id":"am_michael","label":"Michael · American · Male","language":"en-us"},
    {"id":"am_fenrir","label":"Fenrir · American · Male","language":"en-us"},
    {"id":"am_puck","label":"Puck · American · Male","language":"en-us"},
    {"id":"am_adam","label":"Adam · American · Male","language":"en-us"},
    {"id":"am_echo","label":"Echo · American · Male","language":"en-us"},
    {"id":"am_eric","label":"Eric · American · Male","language":"en-us"},
    {"id":"am_liam","label":"Liam · American · Male","language":"en-us"},
    {"id":"am_onyx","label":"Onyx · American · Male","language":"en-us"},
    {"id":"bf_emma","label":"Emma · British · Female","language":"en-gb"},
    {"id":"bf_isabella","label":"Isabella · British · Female","language":"en-gb"},
    {"id":"bf_alice","label":"Alice · British · Female","language":"en-gb"},
    {"id":"bf_lily","label":"Lily · British · Female","language":"en-gb"},
    {"id":"bm_george","label":"George · British · Male","language":"en-gb"},
    {"id":"bm_lewis","label":"Lewis · British · Male","language":"en-gb"},
    {"id":"bm_fable","label":"Fable · British · Male","language":"en-gb"},
    {"id":"bm_daniel","label":"Daniel · British · Male","language":"en-gb"},
]
VOICE_BY_ID = {voice["id"]: voice for voice in VOICE_CATALOG}


def _data_root() -> Path:
    return Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / APP_NAME


def _nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    paths = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
    ]
    return next((str(path) for path in paths if path.is_file()), None)


def _gpu_name(smi: str) -> str:
    try:
        result = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return "NVIDIA GPU"


class NeuralSpeechEngine:
    """Persistent manager for the external Kokoro ONNX CUDA worker."""

    def __init__(
        self,
        app_root: Path,
        on_status: Callable[[str, str], None] | None = None,
        voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_SPEED,
        volume: int = DEFAULT_VOLUME,
    ) -> None:
        self.app_root = Path(app_root)
        self.data_root = _data_root()
        self.runtime_root = self.data_root / "kokoro-runtime"
        self.python_root = self.runtime_root / f"python-{PYTHON_VERSION}"
        self.python_exe = self.python_root / "python.exe"
        self.model_dir = self.data_root / "models" / MODEL_DIR_NAME
        self.hf_home = self.model_dir / "huggingface"
        self.worker_file = self.app_root / "kokoro_worker.py"
        if not self.worker_file.is_file():
            import sys
            self.worker_file = Path(getattr(sys, "_MEIPASS", self.app_root)) / "kokoro_worker.py"
        self.voice = voice if voice in VOICE_BY_ID else DEFAULT_VOICE
        self.speed = self._clamp_speed(speed)
        self.volume = self._clamp_volume(volume)
        self.on_status = on_status
        self.lock = threading.RLock()
        self.speak_lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.reader: threading.Thread | None = None
        self.ready = False
        self.status = "Checking NVIDIA GPU…"
        self.provider = "PyTorch CUDA (required)"
        self.model_precision = "FP32 CUDA"
        self.phonemizer = "Kokoro native PyTorch · CUDA inference"
        self.error = ""
        self.closed = threading.Event()
        self.speech_done = threading.Event()
        self.speech_success = False
        self.speech_error = ""
        threading.Thread(target=self._bootstrap, name="kokoro-bootstrap", daemon=True).start()

    @staticmethod
    def _clamp_speed(value: float) -> float:
        return max(0.70, min(1.40, float(value)))

    @staticmethod
    def _clamp_volume(value: int) -> int:
        return max(0, min(100, int(value)))

    def _notify(self, message: str, level: str = "info") -> None:
        with self.lock:
            self.status = message
        if self.on_status:
            try:
                self.on_status(message, level)
            except Exception:
                pass

    def _run(self, args: list[str], label: str, log_path: Path) -> None:
        self._notify(label)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write("\n>>> " + " ".join(args) + "\n")
            log.flush()
            result = subprocess.run(
                args, stdout=log, stderr=subprocess.STDOUT, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        if result.returncode:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            raise RuntimeError(
                f"{label} failed ({result.returncode}).\n" + "\n".join(lines[-18:])[-3000:]
            )

    def _download(self, url: str, destination: Path) -> None:
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
        except Exception:
            part.unlink(missing_ok=True)
            raise
        if not part.is_file() or part.stat().st_size < 1_000_000:
            part.unlink(missing_ok=True)
            raise RuntimeError("A required first-run download was incomplete.")
        part.replace(destination)

    def _install_python(self, log_path: Path) -> None:
        self.python_root.mkdir(parents=True, exist_ok=True)
        if not self.python_exe.is_file():
            self._notify("Downloading the lightweight Python runtime…")
            archive_path = self.runtime_root / "python-embed.zip"
            self._download(PYTHON_URL, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                bad = archive.testzip()
                if bad:
                    raise RuntimeError(f"Python runtime archive is corrupt at {bad}.")
                archive.extractall(self.python_root)
            archive_path.unlink(missing_ok=True)

        pth_files = list(self.python_root.glob("*._pth"))
        if not pth_files:
            raise RuntimeError("Python runtime configuration file was not found.")
        pth = pth_files[0]
        lines = [line.strip() for line in pth.read_text(encoding="utf-8").splitlines() if line.strip()]
        lines = [line for line in lines if line.lower() not in {"# import site", "import site"}]
        if "Lib\\site-packages" not in lines:
            lines.append("Lib\\site-packages")
        lines.append("import site")
        pth.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (self.python_root / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)

        pip_ok = subprocess.run(
            [str(self.python_exe), "-m", "pip", "--version"],
            capture_output=True, text=True, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).returncode == 0
        if pip_ok:
            return
        self._notify("Preparing pip in the isolated Kokoro runtime…")
        get_pip = self.runtime_root / "get-pip.py"
        request = urllib.request.Request(GET_PIP_URL, headers={"User-Agent": "NinuGamingLiveAssistant/2.3"})
        with urllib.request.urlopen(request, timeout=120) as response:
            get_pip.write_bytes(response.read())
        self._run(
            [str(self.python_exe), str(get_pip), "--disable-pip-version-check"],
            "Installing the isolated Python package manager", log_path,
        )
        get_pip.unlink(missing_ok=True)

    def _install_kokoro_runtime(self, log_path: Path) -> None:
        marker = self.runtime_root / "kokoro-runtime.version"
        try:
            if marker.read_text(encoding="utf-8").strip() == RUNTIME_SIGNATURE:
                self._run(
                    [str(self.python_exe), "-c",
                     "import torch; assert torch.version.cuda is not None; assert torch.cuda.is_available(); "
                     "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"],
                    "Checking PyTorch NVIDIA CUDA runtime", log_path,
                )
                return
        except (OSError, RuntimeError):
            pass

        self._install_python(log_path)
        # Remove the failed ONNX backend so its packages cannot shadow the CUDA PyTorch stack.
        self._run(
            [str(self.python_exe), "-m", "pip", "uninstall", "-y",
             "kokoro-onnx", "onnxruntime-gpu", "onnxruntime"],
            "Removing the incompatible ONNX speech backend", log_path,
        )
        self._run(
            [str(self.python_exe), "-m", "pip", "install", "--disable-pip-version-check",
             "--no-warn-script-location", "--index-url", "https://download.pytorch.org/whl/cu128",
             "--extra-index-url", "https://pypi.org/simple", f"torch=={TORCH_VERSION}"],
            "Installing PyTorch with NVIDIA CUDA 12.8", log_path,
        )
        self._run(
            [str(self.python_exe), "-m", "pip", "install", "--disable-pip-version-check",
             "--no-warn-script-location", f"kokoro=={KOKORO_VERSION}", "sounddevice>=0.5",
             "espeakng-loader>=0.2.4"],
            "Installing native Kokoro and audio dependencies", log_path,
        )
        self._run(
            [str(self.python_exe), "-c",
             "import torch; assert torch.version.cuda is not None; assert torch.cuda.is_available(), "
             "'PyTorch CUDA is unavailable'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); "
             "from kokoro import KModel, KPipeline"],
            "Verifying native Kokoro CUDA runtime", log_path,
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(RUNTIME_SIGNATURE, encoding="utf-8")

    def _bootstrap(self) -> None:
        log_path = self.runtime_root / "kokoro-install.log"
        try:
            smi = _nvidia_smi()
            if not smi:
                raise RuntimeError("NVIDIA driver not detected. Install/update the NVIDIA driver; CPU TTS is disabled.")
            check = subprocess.run(
                [smi, "-L"], capture_output=True, text=True, timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if check.returncode or "GPU" not in (check.stdout or ""):
                raise RuntimeError("nvidia-smi did not find an NVIDIA GPU. Kokoro CUDA was not started.")
            self.data_root.mkdir(parents=True, exist_ok=True)
            self._install_kokoro_runtime(log_path)
            if self.closed.is_set():
                return
            if not self.worker_file.is_file():
                raise RuntimeError(f"Kokoro worker file is missing: {self.worker_file}")
            self._notify("Starting the Kokoro NVIDIA CUDA worker…")
            env = os.environ.copy()
            env.update({
                "HF_HOME": str(self.hf_home),
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
            })
            worker_log = (self.runtime_root / "kokoro-worker.log").open(
                "a", encoding="utf-8", errors="replace"
            )
            self.process = subprocess.Popen(
                [str(self.python_exe), "-u", str(self.worker_file)],
                cwd=str(self.worker_file.parent), env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=worker_log,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.reader = threading.Thread(target=self._read_worker, name="kokoro-events", daemon=True)
            self.reader.start()
            self._send({"cmd":"configure","voice":self.voice,"speed":self.speed,"volume":self.volume})
        except Exception as exc:
            self.error = str(exc)
            self.ready = False
            self._notify(f"Kokoro CUDA startup failed: {exc}", "error")

    def _send(self, payload: dict[str, Any]) -> bool:
        proc = self.process
        if proc is None or proc.poll() is not None or proc.stdin is None:
            return False
        try:
            with self.lock:
                proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                proc.stdin.flush()
            return True
        except (OSError, BrokenPipeError):
            return False

    def _read_worker(self) -> None:
        proc = self.process
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                if self.closed.is_set():
                    break
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = message.get("event")
                if kind == "status":
                    self._notify(str(message.get("status") or "Loading Kokoro…"), message.get("level", "info"))
                elif kind == "ready":
                    self.ready = True
                    self.error = ""
                    self.provider = str(message.get("provider") or "NVIDIA CUDA")
                    self.model_precision = str(message.get("precision") or "FP32 CUDA")
                    self.phonemizer = str(message.get("backend") or "Kokoro streaming")
                    self._notify(
                        f"Kokoro ready · {self.provider} · {self.model_precision} · "
                        f"{message.get('gpu_name', 'NVIDIA GPU')}"
                    )
                elif kind == "error":
                    self.ready = False
                    self.error = str(message.get("message") or "Unknown Kokoro error")
                    self._notify(f"Kokoro CUDA error: {self.error}", "error")
                elif kind == "speech_done":
                    self.speech_success = bool(message.get("success"))
                    self.speech_error = str(message.get("error") or "")
                    self.speech_done.set()
        finally:
            if not self.closed.is_set() and proc.poll() is not None:
                self.ready = False
                if not self.error:
                    self.error = f"Kokoro worker exited with code {proc.returncode}."
                self._notify(self.error, "error")

    def configure(
        self,
        voice: str | None = None,
        speed: float | None = None,
        volume: int | None = None,
    ) -> dict:
        with self.lock:
            if voice is not None:
                self.voice = voice if voice in VOICE_BY_ID else DEFAULT_VOICE
            if speed is not None:
                self.speed = self._clamp_speed(speed)
            if volume is not None:
                self.volume = self._clamp_volume(volume)
            result = self.settings()
            command = {"cmd":"configure","voice":self.voice,"speed":self.speed,"volume":self.volume}
        self._send(command)
        return result

    def settings(self) -> dict:
        with self.lock:
            return {
                "voice":self.voice, "voice_label":VOICE_BY_ID[self.voice]["label"],
                "speed":self.speed, "volume":self.volume, "ready":self.ready,
                "status":self.status, "provider":self.provider,
                "model_precision":self.model_precision, "phonemizer":self.phonemizer,
                "model_directory":str(self.model_dir), "error":self.error,
                "voices":VOICE_CATALOG,
            }

    def speak(self, text: str) -> bool:
        if not self.ready:
            raise RuntimeError(self.error or self.status or "Kokoro CUDA is not ready.")
        with self.speak_lock:
            self.speech_done.clear()
            self.speech_success = False
            self.speech_error = ""
            if not self._send({"cmd":"speak","text":str(text)[:1000]}):
                raise RuntimeError("Could not send speech to the Kokoro worker.")
            deadline = time.monotonic() + 180
            while not self.speech_done.wait(0.25):
                if self.closed.is_set():
                    return False
                if self.process is None or self.process.poll() is not None:
                    raise RuntimeError("Kokoro worker stopped during speech.")
                if time.monotonic() >= deadline:
                    self.stop_current()
                    raise TimeoutError("Kokoro did not finish within three minutes.")
            if self.speech_error:
                raise RuntimeError(self.speech_error)
            return self.speech_success

    def stop_current(self) -> None:
        self._send({"cmd":"stop"})

    def close(self) -> None:
        self.closed.set()
        self._send({"cmd":"shutdown"})
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.process.terminate()
