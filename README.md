# NinuGaming Live Assistant 2.3.0 RC1

A Windows desktop app combining YouTube livestream chat reading, a visible speech pipeline, and NinuGaming's Genshin account-review queue.

## New in 2.3.0: Kokoro-v1.0 NVIDIA CUDA speech

- Uses **Kokoro-82M v1.0** through **kokoro-onnx 0.6.1**.
- Uses the official **FP16 ONNX model** with NVIDIA **ONNX Runtime CUDAExecutionProvider**.
- CPU execution-provider fallback is explicitly disabled. If CUDA cannot run the model, the dashboard shows an error rather than silently switching to CPU.
- Streams sentence/clause audio chunks while the next chunk is generated.
- Offers 26 selected American and British English voices from the Kokoro v1.0 voice pack.
- Uses the compact FP16 model (about 156 MB) and voice pack (about 27 MB), downloaded and SHA-256 verified on first launch.
- Retains short-chat batching and filters emoji/symbol noise and unsupported scripts only in TTS input; the original chat remains visible in the inbox.

### Runtime and compatibility

- Requires 64-bit Windows 10/11, an NVIDIA CUDA-capable GPU, a compatible NVIDIA driver, an internet connection for first-time setup, and several hundred MB of free disk space.
- The separate Python 3.12.10 runtime installs ONNX Runtime GPU 1.26.0 with its CUDA/cuDNN dependencies and kokoro-onnx 0.6.1 under `%PROGRAMDATA%\NinuGamingLiveAssistant\kokoro-runtime`.
- The model and voice pack are stored under `%PROGRAMDATA%\NinuGamingLiveAssistant\models\kokoro-v1.0`.
- ONNX Runtime 1.26.0 uses CUDA 12.8 and cuDNN 9.x. Keep the NVIDIA driver up to date. See the official [ONNX Runtime CUDA compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
- Kokoro is much smaller than the previous VibeVoice 0.5B/PyTorch runtime, but speech synthesis still uses GPU resources and can compete with a running game.
- The selected voice and speed apply to the next utterance. The speed control changes speaking rate, not model quality.

**Model licensing:** Kokoro-82M is Apache-2.0 licensed. The kokoro-onnx runtime is MIT licensed. See the upstream project and model card for the exact terms.

## Install

1. Download `NinuGamingLiveAssistant-Setup.exe` from Releases.
2. Run the installer.
3. Start the app. On first launch it checks for the NVIDIA driver, installs the isolated ONNX Runtime CUDA speech environment, and downloads the FP16 model and voice pack.
4. Wait until Engine Status says `Kokoro ready · NVIDIA CUDA`.
5. Choose a voice, preview it, and adjust the speaking speed before connecting the stream.

The lightweight installer does not bundle CUDA libraries or model weights. The separate runtime and the verified model files are downloaded on first launch.

## Speech controls

- **Voice:** 26 selected American and British English presets.
- **Preview selected voice:** queues a sample through the same streaming playback path used by live chat.
- **Test neural voice (3 lines):** queues three consecutive test utterances.
- **Voice speed:** adjusts Kokoro's speaking rate.
- **Volume:** post-generation playback level.
- **Engine status:** displays the NVIDIA CUDA provider, FP16 ONNX model, voice, and any startup/runtime errors.
- **Diagnostics:** shows the selected speech backend. Speech is not ready unless CUDA initialization succeeds.

## YouTube and chat features

- Read public YouTube live-chat messages aloud.
- View received messages, routing/filter reasons, upcoming speech, current speech, and recently spoken lines.
- Combine adjacent very short chat messages into a contextual spoken update.
- Detect Genshin account-review requests and extract possible 6–12 digit UIDs.
- Keep a persistent review queue with viewer, UID, original request, confidence, and detection reason.
- Read supported Super Chats, Super Stickers, new membership events, membership milestones, and gifting events.
- Retry recoverable network/rate-limit failures and preserve YouTube continuation tokens.
- Use the local OBS overlay at `/obs`; OBS does not make its own YouTube API requests.

## YouTube quota and four-hour cap

The listener uses `videos.list` once (1 unit) and `liveChatMessages.list` (5 units per request). It respects YouTube's returned polling interval and enforces a 10-second minimum delay. At the 10-second floor, the four-hour estimate is approximately 7,201 units. The app stops at a local 9,000-unit safety ceiling or four cumulative hours per Pacific quota day.

The local estimate cannot measure quota used by other apps or people sharing the same Google Cloud project. A real `quotaExceeded` response stops the listener with a visible error.

## Run from source

Windows 10/11 64-bit and Python 3.12 are recommended.

1. Clone/download this repository.
2. Run `setup.bat` to install desktop dependencies in a local virtual environment.
3. Run `run.bat`.
4. On first launch, the app bootstraps the isolated Kokoro ONNX CUDA runtime and downloads the verified model files. This does not modify the source virtual environment.

The speech runtime pins ONNX Runtime GPU 1.26.0, kokoro-onnx 0.6.1, and the Kokoro-v1.0 FP16 ONNX model/voice pack hashes embedded in `kokoro_worker.py`.

## Build and test

- `.github/workflows/tests.yml` runs unit tests and syntax checks without requiring a GPU.
- `.github/workflows/windows-build.yml` builds the lightweight Windows desktop installer. The CUDA runtime and model are downloaded only after installation.
- GitHub-hosted CI has no RTX 3060 attached, so it cannot verify actual CUDA inference or local audio playback. That final hardware test remains necessary.

## Troubleshooting

- **NVIDIA GPU not detected:** install/update the NVIDIA driver and confirm `nvidia-smi -L` lists the GPU.
- **CUDA provider unavailable:** inspect `%PROGRAMDATA%\NinuGamingLiveAssistant\kokoro-runtime\kokoro-install.log`.
- **Model downloading:** first-run downloads are about 190 MB total. Keep the app open and internet connected.
- **CUDA/cuDNN initialization failed:** update the NVIDIA driver; check that the installed driver supports the CUDA runtime used by ONNX Runtime 1.26.0.
- **Voice playback error:** inspect Engine Status and `kokoro-worker.log` in the Kokoro runtime folder.
- **No audio:** check the Windows output device and Volume Mixer.

## Official references

- [Kokoro-ONNX](https://github.com/thewh1teagle/kokoro-onnx)
- [Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M)
- [Kokoro model and voice releases](https://github.com/thewh1teagle/kokoro-onnx/releases)
- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [YouTube LiveChatMessages.list](https://developers.google.com/youtube/v3/live/docs/liveChatMessages/list)
- [YouTube Data API quota costs](https://developers.google.com/youtube/v3/determine_quota_cost)
