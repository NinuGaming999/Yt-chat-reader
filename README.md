# NinuGaming Live Assistant 2.2.0 RC3

A Windows desktop app combining YouTube livestream chat reading, a visible speech pipeline, and NinuGaming's Genshin account-review queue.

## New in 2.2.0 RC3: VibeVoice CUDA streaming speech

- Replaces Kokoro/DirectML with **Microsoft VibeVoice Realtime 0.5B**.
- Uses **NVIDIA CUDA through PyTorch**. There is deliberately **no CPU TTS fallback**.
- Streams generated audio chunks while speech generation continues.
- Offers six English presets: Carter, Davis, Emma, Frank, Grace, and Mike.
- Keeps PyTorch, CUDA packages, and the model outside the installer. The first launch downloads the separate runtime and approximately 2 GB model; later launches reuse the local files.
- Restores the original VibeVoice noise scheduler and uses Microsoft's five-step baseline, with the quality control scaling up to eight steps.
- Selects FlashAttention 2 when available and working; otherwise PyTorch SDPA uses the fastest fused CUDA kernels supported by the installed GPU and PyTorch build.
- Combines adjacent very short chat messages into a contextual spoken update, and removes emoji/symbol noise and unsupported scripts from TTS input without altering the visible chat.
- Uses a solid red app icon consistently in the Windows executable, installer, and app UI.

### Compatibility and limitations

- Requires 64-bit Windows 10/11, a supported NVIDIA GPU, a current NVIDIA driver, an internet connection for first-time setup, and several GB of free disk space.
- The engine uses BF16 when supported and FP16 otherwise, always on CUDA.
- If CUDA is unavailable or model loading fails, the dashboard shows the error. It never silently falls back to CPU.
- The TTS model occupies GPU memory while the app is open and can compete with a game for GPU resources. Close other GPU-heavy apps if loading fails or audio falls behind.
- This realtime model supports one speaker at a time and is primarily intended for English.
- Microsoft describes first-audio latency around 200–300 ms as hardware-dependent. This is not a guarantee for every RTX 3060 configuration.

**Research-use warning:** the model card says VibeVoice is intended for research and development and is not recommended for untested real-world/commercial use. Generated audio includes an audible AI-generated-speech disclosure and an imperceptible provenance watermark. Review the current model card before using it on a public stream.

## Install

1. Download `NinuGamingLiveAssistant-Setup.exe` from Releases or the Windows workflow artifact.
2. Run the installer. The desktop app is installed without bundling PyTorch, CUDA libraries, or model weights.
3. Start the app. On first launch it checks for the NVIDIA driver, installs an isolated Python/PyTorch CUDA runtime under `%PROGRAMDATA%\NinuGamingLiveAssistant\runtime`, and downloads the model under `%PROGRAMDATA%\NinuGamingLiveAssistant\models`.
4. Wait until Engine Status says `VibeVoice ready · NVIDIA CUDA`.
5. Choose a voice, preview it, and adjust GPU compute/quality before connecting the stream.

The separated first-run downloads keep the setup file much smaller than bundling both the CUDA runtime and multi-gigabyte speech model.

## Speech controls

- **Voice:** six English presets.
- **Preview selected voice:** queues a short sample through the same live speech path.
- **Test neural voice (3 lines):** queues three consecutive test utterances.
- **GPU compute / voice quality:** the baseline is five diffusion steps; higher values use up to eight steps and more GPU compute.
- **Volume:** post-generation playback level.
- **Engine status:** displays the CUDA device, precision, model directory, loading stage, and failure details.
- **Diagnostics:** shows the actual provider. Speech is not ready unless it reports NVIDIA CUDA.

## YouTube and chat features

- Read public YouTube live-chat messages aloud.
- View received messages, routing/filter reasons, upcoming speech, current speech, and recently spoken lines.
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
4. On first launch, the app bootstraps the isolated CUDA speech runtime and downloads VibeVoice. This does not modify the source virtual environment.

The speech runtime is pinned to PyTorch 2.7.1 CUDA 12.6, Transformers 4.51.3, and Microsoft VibeVoice source commit `16fb2cb1217c9934a886e1948ffb06120caa2df5`.

## Build and test

- `.github/workflows/tests.yml` runs unit tests and syntax checks without requiring a GPU.
- `.github/workflows/windows-build.yml` builds the lightweight Windows desktop installer. The CUDA runtime and model are downloaded only after installation.
- GitHub-hosted CI has no RTX 3060 attached, so it cannot verify CUDA inference or local playback on your PC. That final hardware test remains necessary; the app will show explicit CUDA errors rather than silently use CPU.

## Troubleshooting

- **NVIDIA GPU not detected:** install/update the NVIDIA driver and confirm `nvidia-smi -L` lists the GPU.
- **CUDA not available:** inspect `%PROGRAMDATA%\NinuGamingLiveAssistant\runtime\vibevoice-install.log`.
- **Model downloading:** first-run download is approximately 2 GB. Keep the app open and internet connected.
- **Out of GPU memory:** close GPU-heavy applications or lower the GPU compute/quality setting. The game and TTS share the RTX 3060.
- **Voice playback error:** inspect Engine Status and `vibevoice-worker.log` in the runtime folder.
- **No audio:** check the Windows output device and Volume Mixer.

## Official references

- [Microsoft VibeVoice repository](https://github.com/microsoft/VibeVoice)
- [VibeVoice Realtime 0.5B model card](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
- [PyTorch CUDA installation guide](https://pytorch.org/get-started/locally/)
- [YouTube LiveChatMessages.list](https://developers.google.com/youtube/v3/live/docs/liveChatMessages/list)
- [YouTube Data API quota costs](https://developers.google.com/youtube/v3/determine_quota_cost)
