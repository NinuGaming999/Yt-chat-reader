"""CUDA-only VibeVoice process manager.

The desktop installer stays small. A separate embedded Python runtime, the
NVIDIA CUDA packages and the model are downloaded on first launch. There is
intentionally no CPU TTS fallback.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import requests

APP_NAME = "NinuGamingLiveAssistant"
DEFAULT_VOICE = "en-Carter_man"
DEFAULT_SPEED = 1.0
DEFAULT_VOLUME = 100
PYTHON_VERSION = "3.12.10"
PYTHON_URL = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
TORCH_VERSION = "2.7.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
VIBEVOICE_ZIP = "https://github.com/microsoft/VibeVoice/archive/16fb2cb1217c9934a886e1948ffb06120caa2df5.zip"
PACKAGES = [
    "transformers==4.51.3", "accelerate>=1.0,<2", "diffusers>=0.30,<1",
    "numpy>=1.26,<2", "scipy>=1.11", "safetensors>=0.4",
    "huggingface_hub>=0.27,<1", "sentencepiece>=0.2", "sounddevice>=0.5",
    "tqdm>=4.66", "ml-collections>=0.1", "packaging>=24",
]
VOICE_CATALOG = [
    {"id":"en-Carter_man","label":"Carter · English · Male","language":"en"},
    {"id":"en-Davis_man","label":"Davis · English · Male","language":"en"},
    {"id":"en-Emma_woman","label":"Emma · English · Female","language":"en"},
    {"id":"en-Frank_man","label":"Frank · English · Male","language":"en"},
    {"id":"en-Grace_woman","label":"Grace · English · Female","language":"en"},
    {"id":"en-Mike_man","label":"Mike · English · Male","language":"en"},
]
VOICE_BY_ID = {v["id"]: v for v in VOICE_CATALOG}


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
    return next((str(p) for p in paths if p.is_file()), None)


class NeuralSpeechEngine:
    """Persistent manager for the external VibeVoice CUDA worker."""

    def __init__(self, app_root: Path, on_status: Callable[[str,str],None] | None = None,
                 voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED,
                 volume: int = DEFAULT_VOLUME) -> None:
        self.app_root = Path(app_root)
        self.data_root = _data_root()
        self.runtime_root = self.data_root / "runtime"
        self.python_root = self.runtime_root / f"python-{PYTHON_VERSION}"
        self.python_exe = self.python_root / "python.exe"
        self.model_dir = self.data_root / "models" / "vibevoice-realtime-0.5b"
        self.worker_file = self.app_root / "vibevoice_worker.py"
        if not self.worker_file.is_file():
            import sys
            self.worker_file = Path(getattr(sys, "_MEIPASS", self.app_root)) / "vibevoice_worker.py"
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
        self.provider = "NVIDIA CUDA (required)"
        self.model_precision = "BF16 / FP16"
        self.phonemizer = "VibeVoice realtime streaming"
        self.error = ""
        self.closed = threading.Event()
        self.speech_done = threading.Event()
        self.speech_success = False
        self.speech_error = ""
        threading.Thread(target=self._bootstrap, name="vibevoice-bootstrap", daemon=True).start()

    @staticmethod
    def _clamp_speed(value: float) -> float:
        return max(1.0, min(1.40, float(value)))

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
            result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=False,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            raise RuntimeError(f"{label} failed ({result.returncode}).\n" + "\n".join(lines[-18:])[-3000:])

    def _download(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_suffix(destination.suffix + ".download")
        with requests.get(url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            with part.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        output.write(chunk)
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
        lines = [x.strip() for x in pth.read_text(encoding="utf-8").splitlines() if x.strip()]
        lines = [x for x in lines if x.lower() not in {"# import site", "import site"}]
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
        self._notify("Preparing pip in the isolated CUDA runtime…")
        get_pip = self.runtime_root / "get-pip.py"
        response = requests.get(GET_PIP_URL, timeout=(20, 120))
        response.raise_for_status()
        get_pip.write_bytes(response.content)
        self._run([str(self.python_exe), str(get_pip), "--disable-pip-version-check"],
                  "Installing the isolated Python package manager", log_path)
        get_pip.unlink(missing_ok=True)

    def _install_cuda_runtime(self, log_path: Path) -> None:
        marker = self.runtime_root / "vibevoice-runtime.version"
        signature = f"python-{PYTHON_VERSION};torch-{TORCH_VERSION};cu126;vibevoice-16fb2cb"
        try:
            if marker.read_text(encoding="utf-8").strip() == signature:
                self._run([str(self.python_exe), "-c",
                           "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"],
                          "Checking NVIDIA CUDA runtime", log_path)
                return
        except (OSError, RuntimeError):
            pass
        self._install_python(log_path)
        self._run([str(self.python_exe), "-m", "pip", "install", "--disable-pip-version-check",
                   "--no-warn-script-location", f"torch=={TORCH_VERSION}", "--index-url", TORCH_INDEX],
                  "Installing PyTorch with NVIDIA CUDA 12.6", log_path)
        self._run([str(self.python_exe), "-m", "pip", "install", "--disable-pip-version-check",
                   "--no-warn-script-location", *PACKAGES],
                  "Installing VibeVoice dependencies", log_path)
        self._run([str(self.python_exe), "-m", "pip", "install", "--disable-pip-version-check",
                   "--no-warn-script-location", "--no-deps", VIBEVOICE_ZIP],
                  "Installing pinned Microsoft VibeVoice source", log_path)
        self._run([str(self.python_exe), "-c",
                   "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
                   "print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())"],
                  "Verifying GPU access before model download", log_path)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(signature, encoding="utf-8")

    def _bootstrap(self) -> None:
        log_path = self.runtime_root / "vibevoice-install.log"
        try:
            smi = _nvidia_smi()
            if not smi:
                raise RuntimeError("NVIDIA driver not detected. Install/update the NVIDIA driver; CPU TTS is disabled.")
            check = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=15,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if check.returncode or "GPU" not in (check.stdout or ""):
                raise RuntimeError("nvidia-smi did not find an NVIDIA GPU. VibeVoice CUDA was not started.")
            self.data_root.mkdir(parents=True, exist_ok=True)
            self._install_cuda_runtime(log_path)
            if self.closed.is_set():
                return
            if not self.worker_file.is_file():
                raise RuntimeError(f"Speech worker file is missing: {self.worker_file}")
            self._notify("Starting the NVIDIA CUDA speech worker…")
            env = os.environ.copy()
            env.update({
                "VIBEVOICE_MODEL_DIR": str(self.model_dir),
                "VIBEVOICE_DATA_DIR": str(self.data_root),
                "HF_HOME": str(self.data_root / "huggingface"),
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            })
            worker_log = (self.runtime_root / "vibevoice-worker.log").open("a", encoding="utf-8", errors="replace")
            self.process = subprocess.Popen(
                [str(self.python_exe), "-u", str(self.worker_file)],
                cwd=str(self.worker_file.parent), env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=worker_log,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.reader = threading.Thread(target=self._read_worker, name="vibevoice-events", daemon=True)
            self.reader.start()
            self._send({"cmd":"configure","voice":self.voice,"speed":self.speed,"volume":self.volume})
        except Exception as exc:
            self.error = str(exc)
            self.ready = False
            self._notify(f"VibeVoice CUDA startup failed: {exc}", "error")

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
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = msg.get("event")
                if kind == "status":
                    self._notify(str(msg.get("status") or "Loading VibeVoice…"), msg.get("level", "info"))
                elif kind == "ready":
                    self.ready = True
                    self.error = ""
                    self.provider = str(msg.get("provider") or "NVIDIA CUDA")
                    self.model_precision = str(msg.get("precision") or "BF16")
                    self.phonemizer = str(msg.get("backend") or "VibeVoice realtime")
                    self._notify(f"VibeVoice ready · {self.provider} · {self.model_precision} · {msg.get('gpu_name','NVIDIA GPU')}")
                elif kind == "error":
                    self.ready = False
                    self.error = str(msg.get("message") or "Unknown VibeVoice error")
                    self._notify(f"VibeVoice CUDA error: {self.error}", "error")
                elif kind == "speech_done":
                    self.speech_success = bool(msg.get("success"))
                    self.speech_error = str(msg.get("error") or "")
                    self.speech_done.set()
        finally:
            if not self.closed.is_set() and proc.poll() is not None:
                self.ready = False
                if not self.error:
                    self.error = f"VibeVoice worker exited with code {proc.returncode}."
                self._notify(self.error, "error")

    def configure(self, voice: str | None = None, speed: float | None = None,
                  volume: int | None = None) -> dict:
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
            raise RuntimeError(self.error or self.status or "VibeVoice CUDA is not ready.")
        with self.speak_lock:
            self.speech_done.clear()
            self.speech_success = False
            self.speech_error = ""
            if not self._send({"cmd":"speak","text":str(text)[:1000]}):
                raise RuntimeError("Could not send speech to the VibeVoice worker.")
            deadline = time.monotonic() + 300
            while not self.speech_done.wait(0.25):
                if self.closed.is_set():
                    return False
                if self.process is None or self.process.poll() is not None:
                    raise RuntimeError("VibeVoice worker stopped during speech.")
                if time.monotonic() >= deadline:
                    self.stop_current()
                    raise TimeoutError("VibeVoice did not finish within five minutes.")
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
