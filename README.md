# NinuGaming Live Assistant

A Windows desktop app that merges NinuGaming's YouTube live-chat voice reader with the Genshin account-review queue. It includes a live inbox, transparent speech queue, diagnostics and an OBS browser overlay.

## Features

- Read new public YouTube live-chat messages aloud using a persistent Windows System.Speech voice process.
- See every message received, its reading decision, the exact next line, the current line and recently completed lines.
- Detect Genshin account-review / queue requests and extract a possible 6–12 digit UID.
- Keep a persistent local review queue with viewer name, UID, original comment and detection reason.
- Handle supported Super Chats, Super Stickers, memberships, milestones and gift events.
- Retry recoverable network and rate-limit errors, preserve YouTube page tokens and explain terminal errors.
- Enforce a cumulative four-hour daily reading limit and a local 9,000-unit quota safety ceiling.
- Provide a local OBS Browser Source overlay. OBS uses the app's local state and does not make YouTube API calls.

The Genshin intent detector is rule-based and local, not a paid AI service. Its confidence and matching reason are displayed so the streamer can correct false positives.

## Windows installation

1. Install 64-bit Python 3.10+ and enable **Add Python to PATH**.
2. Download this repository (Code → Download ZIP) and extract it to a writable folder.
3. Double-click **setup.bat** once.
4. Double-click **run.bat**.
5. Enter a YouTube Data API v3 key and a public livestream URL. Press **Test connection**.
6. Press **Test voice (3 messages)**. This checks local voice output without consuming YouTube quota.
7. Press **Start listener**, then send a fresh message in the livestream chat.

The application stores its API key and local usage/review queue under `%APPDATA%\\NinuGamingLiveAssistant`. These files are not committed to GitHub. Do not share your config file or API key.

## Dashboard guide

- **Live chat received:** newest incoming messages and their read/skip/filter reasons. Use **Read this message** to enqueue an individual message manually.
- **Currently speaking:** the line sent to Windows speech.
- **Up next:** FIFO queue of lines waiting to be spoken. The first item is next. **Skip next** removes only the waiting item; it does not interrupt the current line.
- **Recently read:** completed or failed speech items.
- **Genshin review queue:** detected viewers, UID, confidence, raw message and matching reason. **Next viewer** advances the queue.
- **Diagnostics:** poll count, retry count, estimated quota, remaining runtime and recent errors.

### Reading modes

- **All readable chat messages** — speak every new text-bearing message/event.
- **Only Genshin review requests** — show all messages but speak detected review requests.
- **Off** — continue monitoring and classifying without adding speech.

By default, existing chat backlog is skipped on connect. Enable **Read recent chat already on screen at connection** to include it.

## Quota and four-hour limit

The app calls `videos.list` once to find the live chat ID (1 quota unit), then `liveChatMessages.list` (5 units per call). It observes YouTube's `pollingIntervalMillis` and enforces a 10-second minimum delay. At a 10-second delay, four hours is estimated at 7,201 units; a 15-second delay is estimated at 4,801 units.

The app tracks usage and runtime locally and stops at 9,000 estimated units or four cumulative hours in the Pacific quota day. The local ledger cannot see other applications or users consuming the same Google Cloud project's quota. The four-hour limit is therefore a safety cap, not a guarantee of availability. A YouTube `quotaExceeded` response stops the listener with a visible error. Recoverable `rateLimitExceeded` and network failures retry with backoff.

YouTube's API requires the client to follow the returned page token and polling interval. See the official [LiveChatMessages.list documentation](https://developers.google.com/youtube/data/live/reference/rest/v3/liveChatMessages/list) and [quota costs](https://developers.google.com/youtube/v3/determine_quota_cost).

## OBS overlay

Add an OBS **Browser Source** with the local URL printed in the app console, normally `http://127.0.0.1:8765/obs`. The overlay is read-only and reflects the app state. If port 8765 is busy, the app chooses another port and prints the actual URL.

## Build a standalone EXE

On Windows, double-click **build.bat**. The resulting file is `dist\\NinuGamingLiveAssistant.exe`. The GitHub Actions workflow also builds a Windows EXE and uploads it as a downloadable workflow artifact after pushes to `main`.

## Troubleshooting

- **No chat appears:** confirm that the stream is live and public live chat is enabled; send a new chat message after connection. Existing backlog is skipped by default.
- **Messages appear but are not spoken:** inspect each row's read status/reason; choose the correct reading mode, unpause voice and check **Up next**.
- **Voice test is silent:** check Windows Settings → System → Sound → Volume mixer and available Windows speech voices. The test does not use YouTube quota.
- **Listener reconnects repeatedly:** open Diagnostics and inspect the latest error. Invalid API keys, disabled API access, ended streams and exhausted quota are not transient network failures.
- **Quota exhausted:** review the shared project quota in Google Cloud Console; this app cannot measure quota spent by other clients.

## Tests

```bat
.venv\\Scripts\\python.exe -m unittest discover -s tests -v
```

## License

MIT. See [LICENSE](LICENSE).
