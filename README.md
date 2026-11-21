# NinuGaming Live Assistant 2.2.0

A Windows desktop assistant combining YouTube livestream chat reading, a visible speech pipeline, and NinuGaming's Genshin account-review queue.

## New in 2.2.0: smarter chat, priority queue, better queue tools

The Kokoro neural voice engine is unchanged. Everything below is around it.

**Chat reading**
- Cleans text before speaking: links become "link", emoji and `:emoji:` codes are removed, `looooool` and ALL-CAPS are tamed, long numbers (UIDs) are read digit by digit, usernames are tidied (`@Ninu_Gaming12345` → "Ninu Gaming"). Add your own **pronunciations** (`Ninu=Nee noo`).
- **Priority queue:** Super Chats / memberships → review requests → mods / members / host → normal chat. If chat floods in, the oldest normal lines are dropped so the voice never falls minutes behind.
- **Filters** (Tools tab): per-viewer cooldown, duplicate and copy-paste spam detection, blocked words, ignored viewers, `!command` ignore, optional skipping of non-English scripts (Bangla, Hindi, Arabic, CJK…) that an English voice cannot pronounce, and no repeated name for back-to-back messages. Review requests and Super Chats always get through.
- New read mode: **only Super Chats, members, mods and review requests**.

**Review queue**
- UID validation (Genshin UIDs are normally 9 digits) with region hint, "Check UID" and **duplicate UID** warnings.
- Move to top / up / down, edit a UID, mark **Done**, remove; finished reviews are kept in a history list and can be **exported to CSV**.

**OBS overlays** (all options go in the URL)
- `/obs?view=queue`, `view=chat`, `view=speaking`, `view=all`
- `&theme=glass|dark|light|neon`, `&limit=8`, `&scale=1.3`, `&accent=ff4fa3`, `&header=0`, `&bg=solid`, `&order=chat,speaking,queue`
- Uses a lightweight `/api/overlay` endpoint instead of the full dashboard state.

**Reliability and control**
- Quota and runtime warnings at 70% / 90% and 30 / 10 minutes left.
- Optional **backup API key** (from a second Google project) used automatically if the main key hits its quota.
- **Auto-reconnect:** if the live chat ends or the stream drops, the app re-checks once a minute for 10 minutes.
- **Update check** against GitHub Releases.
- **Global hotkeys** (Windows): `Ctrl+Alt+N` next review viewer, `Ctrl+Alt+S` stop current speech, `Ctrl+Alt+P` pause/resume voice, `Ctrl+Alt+K` skip next queued line. Can be turned off.
- **Stream profiles:** save and re-apply voice, speed, read mode and filters.
- **Session insights** and exports (review CSV, activity log, session summary) are saved to `%APPDATA%\NinuGamingLiveAssistant\exports`.
- API keys are now **encrypted with Windows DPAPI** (current user only). Old plain-text keys are migrated automatically.
- Tabs: Live / Review queue / Insights / Tools. Fixed a malformed `<html>` tag in the OBS overlay.

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
- Detect Genshin account-review/queue requests and extract possible UIDs (9-digit numbers preferred).
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