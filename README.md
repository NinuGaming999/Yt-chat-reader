# NinuGaming Live Assistant 2.1.0

A Windows desktop assistant combining YouTube livestream chat reading, a visible speech pipeline, and NinuGaming's Genshin account-review queue.

## New in 2.1.0: local neural speech

- Replaces Windows SAPI voices with **Kokoro-82M** natural neural text-to-speech.
- Offers 27 selectable English voices across American/British accents and female/male voices.
- Provides a voice preview, three-line test, speed and volume controls.
- Uses ONNX Runtime DirectML GPU acceleration where available; it automatically falls back to the INT8 CPU model if DirectML or the GPU model cannot be used.
- Warms up the model in the background at launch and keeps one persistent model session for all messages.
- Generates audio in chunks and buffers playback ahead to reduce gaps between generated chunks.
- Runs locally after installation; chat messages and speech are not sent to a cloud TTS service.

The installer contains both the FP16 and INT8 Kokoro models plus the shared voice pack. Expect a larger download and an initial first-run model warm-up. The FP16 model is used on supported DirectML GPUs; the smaller INT8 model is the CPU fallback.

## Download and install (recommended)

1. Open the repository's **Releases** page or the latest successful **Build Windows desktop app** workflow artifact.
2. Download `NinuGamingLiveAssistant-Setup.exe`.
3. Run the installer. It installs the app, the Kokoro model files, Microsoft's Visual C++ runtime, and WebView2 runtime.
4. Launch **NinuGaming Live Assistant** from the Start menu or desktop shortcut.
5. On the first launch, wait for the TTS status to say `Kokoro ready`. The first load initializes the neural model; later messages reuse the loaded session.
6. Choose a voice, press **Preview selected voice**, and adjust speed/volume before connecting your livestream.

The installer writes model files under `%PROGRAMDATA%\\NinuGamingLiveAssistant\\models`. User settings, API key, local usage ledger, and review queue are kept under `%APPDATA%\\NinuGamingLiveAssistant`.

## Speech controls

- **Choose a natural voice:** 27 Kokoro English voices. The dropdown identifies accent and voice style.
- **Preview selected voice:** speaks a fixed sample so you can compare voices.
- **Test neural voice (3 lines):** queues three separate lines to verify consecutive speech.
- **Speed:** 0.70× to 1.40×. Start near 1.00× for a natural pace.
- **Volume:** adjusts generated audio level.
- **Engine status:** shows model loading, selected ONNX provider, precision, phonemizer and model path. If it reports an error, open Diagnostics and copy the exact message.

The app prefers `DmlExecutionProvider` with the FP16 model. If that cannot be created, it tries the INT8 CPU model. The dashboard displays the backend actually selected; GPU acceleration is never silently assumed.

## YouTube and chat features

- Read public YouTube live-chat messages aloud through the local Kokoro voice.
- View received messages, why each message was routed/filtered, exact upcoming speech, current speech and recently spoken lines.
- Detect Genshin account-review/queue requests and extract possible 6–12 digit UIDs.
- Keep a persistent review queue with viewer, UID, original request, confidence and detection reason.
- Read supported Super Chats, Super Stickers, new membership events, membership milestones and gifting events.
- Retry recoverable network/rate-limit failures and preserve YouTube continuation tokens.
- Use the local OBS overlay at `/obs`; OBS does not make its own YouTube API requests.

## YouTube quota and four-hour cap

The listener uses `videos.list` once (1 unit) and `liveChatMessages.list` (5 units per request). It respects YouTube's returned polling interval and enforces a 10-second minimum delay. At the 10-second floor, the four-hour estimate is approximately 7,201 units. The app stops at a local 9,000-unit safety ceiling or four cumulative hours per Pacific quota day.

This local estimate cannot measure quota used by other apps or people sharing your Google Cloud project. A real `quotaExceeded` API response therefore stops the listener with a visible error.

Official references:
- [YouTube LiveChatMessages.list](https://developers.google.com/youtube/v3/live/docs/liveChatMessages/list)
- [YouTube Data API quota costs](https://developers.google.com/youtube/v3/determine_quota_cost)

## Run from source (developer mode)

Windows 10 1903+ or Windows 11, 64-bit Python 3.12, and an internet connection for downloading dependencies/models are recommended.

1. Clone/download this repository.
2. Run `setup.bat`. It installs Python packages, ONNX Runtime DirectML, Kokoro, and downloads/checksums all three model files.
3. Run `run.bat`.
4. Use `build.bat` only if Inno Setup 7 is installed; it builds the onedir app and the final Windows installer.

The normal user should use the installer, not developer mode.

## Build pipeline

GitHub Actions runs unit tests, installs the Windows speech runtime, downloads and verifies official model hashes, runs a headless neural streaming smoke test, packages the desktop app with PyInstaller, and creates `installer/NinuGamingLiveAssistant-Setup.exe` using Inno Setup.

## Troubleshooting

- **Model is still loading:** wait for `Kokoro ready`. Model initialization is heavier than normal speech and happens once per app start.
- **DirectML fails:** the engine attempts the bundled INT8 CPU fallback. Check the provider shown in Diagnostics.
- **No audio:** press Preview selected voice; check Windows Volume Mixer and the selected output device. A working model does not guarantee that the Windows audio device is available.
- **Voice test returns an error:** copy the full TTS error shown in Engine status/Diagnostics.
- **GPU not selected:** update the NVIDIA driver and verify Windows supports DirectX 12. The app reports CPU fallback rather than claiming GPU use.
- **API stops:** inspect the YouTube API error, daily quota, live status and public live-chat setting.

## Licensing

The Kokoro-82M model and voice files are Apache 2.0 licensed; `kokoro-onnx` is MIT licensed. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for component notices.