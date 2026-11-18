# NinuGaming Live Assistant 2.3.0 RC2

A Windows desktop app combining YouTube livestream chat reading, a visible speech pipeline, and NinuGaming's Genshin account-review queue.

## Kokoro-v1.0 native NVIDIA CUDA speech

- Uses **Kokoro-82M v1.0** through the original **Kokoro PyTorch model** (kokoro==0.9.4).
- Runs the neural model directly through **PyTorch CUDA 12.8** (torch==2.11.0); it no longer uses the ONNX export that failed strict CUDA graph placement.
- CUDA is mandatory. Startup fails if PyTorch CUDA is unavailable, and the app verifies all Kokoro parameters and buffers are on cuda:0 before warming up the model.
- There is **no CPU model-inference fallback**. Text normalization/phonemization and final audio playback still use normal CPU-side desktop I/O.
- Streams generated sentence chunks to the selected Windows audio output.
- Offers 27 selected American and British English voices.
- Keeps short-chat batching, emoji/symbol cleanup for TTS input only, and the original unmodified chat text in the dashboard.

### Runtime, model storage, and compatibility

- Requires 64-bit Windows 10/11, an NVIDIA CUDA-capable GPU, a compatible current NVIDIA driver, internet access for first-time setup, and several GB of free disk space.
- The isolated Python 3.12.10 speech runtime installs the CUDA-enabled PyTorch 2.11.0 Windows wheel from the official PyTorch CUDA 12.8 index and Kokoro 0.9.4.
- Model weights and selected voice files are downloaded from Hugging Face and cached under:
  %PROGRAMDATA%\NinuGamingLiveAssistant\models\kokoro-v1.0-pytorch\huggingface\
- Speech runtime files are under %PROGRAMDATA%\NinuGamingLiveAssistant\kokoro-runtime\.
- PyTorch's CUDA wheel carries the CUDA libraries it needs. A separate full CUDA Toolkit installation is not required, but a compatible NVIDIA display driver is required.
- Kokoro is a small model, but the PyTorch CUDA runtime is substantially larger than the previous ONNX runtime. Allow several GB for first-run installation and temporary package extraction.
- Speech generation uses GPU resources and can compete with a running game. The app does not move neural inference to CPU to avoid that load; lower the number of queued messages or pause voice while gaming if needed.

**Model licensing:** Kokoro-82M is Apache-2.0 licensed. See the model card and the upstream Kokoro project.

## Install

1. Download NinuGamingLiveAssistant-Setup.exe from Releases.
2. Run the installer.
3. Start the app. First launch checks the NVIDIA driver, installs CUDA-enabled PyTorch and Kokoro in an isolated runtime, then downloads the model and selected voice from Hugging Face.
4. Wait until Engine Status says Kokoro ready · PyTorch CUDA · FP32 CUDA and shows your NVIDIA GPU.
5. Preview the selected voice before connecting the stream.

The desktop installer is intentionally lightweight; the GPU runtime and model weights are downloaded on first launch. Keep the app open and internet connected during setup.

## Speech controls

- **Voice:** 27 selected American and British English presets.
- **Preview selected voice:** generates a sample through the same CUDA model and audio playback path used by live chat.
- **Test neural voice (3 lines):** queues three consecutive test utterances.
- **Voice speed:** adjusts Kokoro's speaking rate.
- **Volume:** playback volume after generation.
- **Engine status:** displays PyTorch CUDA, FP32 model execution, the selected NVIDIA GPU, and any startup/runtime errors.
- **Diagnostics:** reports the native Kokoro PyTorch CUDA backend. Speech is not ready unless CUDA initialization and GPU warm-up succeed.

## YouTube and chat features

- Read public YouTube live-chat messages aloud.
- View received messages, routing/filter reasons, upcoming speech, current speech, and recently spoken lines.
- Combine adjacent very short chat messages into a contextual spoken update.
- Detect Genshin account-review requests and extract possible 6–12 digit UIDs.
- Keep a persistent review queue with viewer, UID, original request, confidence, and detection reason.
- Read supported Super Chats, Super Stickers, new membership events, membership milestones, and gifting events.
- Retry recoverable network/rate-limit failures and preserve YouTube continuation tokens.
- Use the local OBS overlay at /obs; OBS does not make its own YouTube API requests.

## YouTube quota and four-hour cap

The listener uses videos.list once (1 unit) and liveChatMessages.list (5 units per request). It respects YouTube's returned polling interval and enforces a 10-second minimum delay. At the 10-second floor, the four-hour estimate is approximately 7,201 units. The app stops at a local 9,000-unit safety ceiling or four cumulative hours per Pacific quota day.

The local estimate cannot measure quota used by other apps or people sharing the same Google Cloud project. A real quotaExceeded response stops the listener with a visible error.

## Run from source

Windows 10/11 64-bit and Python 3.12 are recommended.

1. Clone/download this repository.
2. Run setup.bat to install desktop dependencies in a local virtual environment.
3. Run run.bat.
4. On first launch the app bootstraps the separate CUDA PyTorch/Kokoro runtime and downloads model assets into the ProgramData cache. This does not install the speech stack into the source virtual environment.

The speech runtime pins Python 3.12.10, PyTorch 2.11.0 from the CUDA 12.8 wheel index, and Kokoro 0.9.4.

## Build and validation

- GitHub Actions checks Python syntax, unit tests, and Windows installer packaging.
- Standard GitHub-hosted CI does not have an RTX 3060 attached, so it cannot prove CUDA inference or local audio playback. The final hardware test must be performed on an NVIDIA PC.
- The worker itself performs hard runtime checks for CUDA availability, device placement of all model parameters/buffers, and successful warm-up inference before declaring the engine ready.

## Troubleshooting

- **NVIDIA GPU not detected:** install/update the NVIDIA driver and confirm nvidia-smi -L lists the GPU.
- **PyTorch CUDA unavailable:** check %PROGRAMDATA%\NinuGamingLiveAssistant\kokoro-runtime\kokoro-install.log; verify the installed runtime reports a CUDA build and a visible GPU.
- **Model downloading:** keep an internet connection. The model cache is under %PROGRAMDATA%\NinuGamingLiveAssistant\models\kokoro-v1.0-pytorch\huggingface\.
- **CUDA initialization failed:** update the NVIDIA driver and restart Windows before retrying.
- **Voice playback error:** inspect Engine Status and kokoro-worker.log in the Kokoro runtime folder.
- **No audio:** check the Windows output device and Volume Mixer.

## Official references

- [Kokoro model and inference library](https://github.com/hexgrad/kokoro)
- [Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M)
- [Kokoro PyPI package](https://pypi.org/project/kokoro/0.9.4/)
- [PyTorch CUDA installation instructions](https://pytorch.org/get-started/locally/)
