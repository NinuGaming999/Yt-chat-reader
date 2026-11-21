from __future__ import annotations

import atexit
import base64
import csv
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections import Counter, OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from speech_engine import DEFAULT_SPEED, DEFAULT_VOICE, DEFAULT_VOLUME, NeuralSpeechEngine, VOICE_CATALOG
from chat_tools import (DEFAULT_FILTERS, PAID_KINDS, PRIORITY_LABELS, SpamFilter, clean_for_speech, clean_name_for_speech,
                        enqueue_with_priority, mark_duplicate_uids, non_latin_ratio, pick_uid, sanitize_filters, speech_priority)
from hotkeys import HotkeyManager
from secure_store import protect_secret, unprotect_secret

import requests
from flask import Flask, jsonify, request, send_from_directory

try:
    import webview
except ImportError:
    webview = None

APP_NAME = "NinuGaming Live Assistant"
APP_VERSION = "2.2.0"
GITHUB_REPO = "NinuGaming999/Yt-chat-reader"
YOUTUBE_API = "https://www.googleapis.com/youtube/v3"
PACIFIC = ZoneInfo("America/Los_Angeles")
DAILY_RUNTIME_LIMIT_SECONDS = 4 * 60 * 60
DAILY_QUOTA_LIMIT_UNITS = 10_000
SAFE_QUOTA_BUDGET_UNITS = 9_000
DEFAULT_MIN_DELAY = 10
MAX_CHAT_HISTORY = 300
MAX_SPEECH_HISTORY = 100
MAX_PENDING_SPEECH = 500
MAX_SEEN_IDS = 20_000
MAX_LOGS = 250
SOFT_LOW_PRIORITY_LIMIT = 40
READ_MODES = {"all", "review_only", "priority", "off"}
SECRET_KEYS = ("api_key", "backup_api_key")
MAX_CHATTERS = 20_000

if os.name == "nt":
    DATA_ROOT = Path(os.environ.get("APPDATA", str(Path.home() / "AppData/Roaming"))) / "NinuGamingLiveAssistant"
else:
    DATA_ROOT = Path.home() / ".ninugaming_live_assistant"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
CONFIG_FILE = DATA_ROOT / "config.json"
QUEUE_FILE = DATA_ROOT / "review_queue.json"
EXPORT_DIR = DATA_ROOT / "exports"

INTENT_PHRASES: dict[str, int] = {
    "review my account": 4, "review my acc": 4, "account review": 4,
    "please review": 3, "pls review": 3, "plz review": 3,
    "review me": 3, "check my account": 3, "check my acc": 3,
    "rate my account": 3, "rate my acc": 3, "can i get a review": 3,
    "i want a review": 3, "need a review": 3, "join the queue": 3,
    "join queue": 3, "join que": 3, "put me in queue": 3,
    "put me in the queue": 3, "add me": 3, "addme": 3,
    "sign me up": 3, "my turn": 2, "waitlist": 2, "wait list": 2,
    "waiting list": 2, "join me": 2, "i want to join": 2,
    "wanna join": 2, "want to join": 2, "in queue": 2, "in que": 2,
    "queue": 1, "que": 1, "join": 1, "spot": 1, "slot": 1,
    "my uid": 1, "uid:": 1,
}
UID_PATTERN = re.compile(r"(?<!\d)(\d{6,12})(?!\d)")
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def extract_video_id(value: str) -> str:
    text = (value or "").strip()
    if VIDEO_ID_PATTERN.fullmatch(text):
        return text
    parsed = urlparse(text if "://" in text else f"https://{text}")
    host = parsed.netloc.lower().removeprefix("www.")
    candidate = ""
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com", "gaming.youtube.com"}:
        query = parse_qs(parsed.query)
        if query.get("v"):
            candidate = query["v"][0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0] in {"live", "shorts", "embed", "v"}:
                candidate = parts[1]
    if not VIDEO_ID_PATTERN.fullmatch(candidate):
        raise ValueError("Paste a valid YouTube live URL or its 11-character video ID.")
    return candidate


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def extract_uid(text: str) -> str | None:
    return pick_uid(text or "")


def classify_review_request(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    uid = extract_uid(text)
    matched: list[str] = []
    score = 3 if uid else 0
    for phrase, weight in INTENT_PHRASES.items():
        p = normalize_text(phrase)
        if not p:
            continue
        # Word boundaries prevent "que" matching arbitrary parts of words.
        if re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", normalized):
            matched.append(phrase)
            score += weight
    if any(word in normalized.split() for word in ("genshin", "account", "acc", "uid")):
        score += 1
    is_candidate = score >= 3 and bool(matched or uid)
    confidence = "High" if score >= 6 else ("Likely" if is_candidate else "Low")
    if is_candidate:
        reason = "Matched: " + ", ".join(matched[:4])
        if uid:
            reason += ("; " if matched else "") + f"UID detected: {uid}"
    elif not (text or "").strip():
        reason = "No readable text in this chat event"
    else:
        reason = "No clear review/queue intent phrase or UID was detected"
    return {"candidate": is_candidate, "score": score, "confidence": confidence, "reason": reason, "uid": uid}


def clean_request(text: str) -> str:
    cleaned = UID_PATTERN.sub("", text or "").strip()
    for phrase in sorted(INTENT_PHRASES, key=len, reverse=True):
        cleaned = re.sub(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(please|pls|plz)\b\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:,.!?|[]()")
    return cleaned or "Account review"


def message_text(item: dict[str, Any]) -> tuple[str, str]:
    snippet = item.get("snippet") or {}
    kind = str(snippet.get("type") or "textMessageEvent")
    text = snippet.get("displayMessage") or snippet.get("textMessageDetails", {}).get("messageText") or ""
    if text:
        return " ".join(str(text).split()), kind
    details = snippet.get("superChatDetails") or {}
    if details:
        amount = details.get("amountDisplayString", "a Super Chat")
        comment = details.get("userComment", "")
        return f"sent a Super Chat of {amount}" + (f" and said: {comment}" if comment else ""), kind
    details = snippet.get("superStickerDetails") or {}
    if details:
        amount = details.get("amountDisplayString", "a Super Sticker")
        sticker = details.get("superStickerMetadata", {}).get("altText", "")
        return f"sent a Super Sticker worth {amount}" + (f" ({sticker})" if sticker else ""), kind
    details = snippet.get("newSponsorDetails") or {}
    if details:
        level = details.get("memberLevelName") or "channel membership"
        return f"became a member ({level})", kind
    details = snippet.get("memberMilestoneChatDetails") or {}
    if details:
        months = details.get("memberMonth", "")
        comment = details.get("userComment", "")
        msg = f"celebrated {months} months as a member" if months else "sent a membership milestone"
        return msg + (f": {comment}" if comment else ""), kind
    details = snippet.get("membershipGiftingDetails") or snippet.get("giftMembershipsDetails") or {}
    if details:
        count = details.get("giftMembershipsCount", 1)
        return f"gifted {count} channel membership(s)", kind
    details = snippet.get("giftMembershipReceivedDetails") or {}
    if details:
        gifter = details.get("gifterChannelName") or "a viewer"
        return f"received a gifted membership from {gifter}", kind
    if kind == "messageDeletedEvent":
        return "a message was deleted", kind
    if kind == "userBannedEvent":
        return "a user was timed out or banned", kind
    return "", kind


def make_chat_record(item: dict[str, Any]) -> dict[str, Any]:
    snippet = item.get("snippet") or {}
    author = item.get("authorDetails") or {}
    text, kind = message_text(item)
    classification = classify_review_request(text)
    record = {
        "id": str(item.get("id") or f"no-id-{time.time_ns()}"),
        "name": str(author.get("displayName") or "Viewer"),
        "channel_id": str(author.get("channelId") or ""),
        "text": text or "[Non-text chat event]",
        "full_comment": text,
        "published_at": snippet.get("publishedAt") or now_iso(),
        "kind": kind,
        "uid": classification["uid"],
        "review_candidate": classification["candidate"],
        "intent_score": classification["score"],
        "confidence": classification["confidence"],
        "reason": classification["reason"],
        "read_state": "pending",
        "read_reason": "Waiting for routing decision",
        "is_mod": bool(author.get("isChatModerator")),
        "is_owner": bool(author.get("isChatOwner")),
        "is_member": bool(author.get("isChatSponsor")),
    }
    record["priority"] = speech_priority(record)
    return record


def speech_text_for(record: dict[str, Any], settings: dict[str, Any] | None = None, include_name: bool = True) -> str | None:
    raw = " ".join(str(record.get("text") or "").split())
    if not raw or raw == "[Non-text chat event]":
        return None
    text = clean_for_speech(raw, settings)
    if not text:
        return None
    if not include_name:
        return text
    return f"{clean_name_for_speech(record.get('name') or 'Viewer', settings)}: {text}"


def estimate_four_hour_quota(min_delay_seconds: int) -> int:
    seconds = max(10, int(min_delay_seconds))
    polls = (DAILY_RUNTIME_LIMIT_SECONDS + seconds - 1) // seconds
    return 1 + polls * 5  # videos.list=1, liveChatMessages.list=5


class UsageLedger:
    """Local conservative usage estimate. YouTube quota day rolls over at midnight Pacific."""
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (DATA_ROOT / "usage.json")
        self.lock = threading.RLock()
        self.data: dict[str, Any] = {}
        self.slot_key = "quota_units"
        self._load()

    @property
    def using_backup(self) -> bool:
        return self.slot_key == "backup_quota_units"

    def use_backup(self, flag: bool) -> None:
        with self.lock:
            self.slot_key = "backup_quota_units" if flag else "quota_units"

    @staticmethod
    def today() -> str:
        return datetime.now(PACIFIC).date().isoformat()

    def _load(self) -> None:
        data = safe_read_json(self.path, {})
        if not isinstance(data, dict) or data.get("date") != self.today():
            data = {"date": self.today(), "quota_units": 0, "runtime_seconds": 0.0, "api_calls": 0}
        data.setdefault("quota_units", 0)
        data.setdefault("runtime_seconds", 0.0)
        data.setdefault("api_calls", 0)
        data.setdefault("backup_quota_units", 0)
        self.data = data
        self._save()

    def _ensure_day(self) -> None:
        if self.data.get("date") != self.today():
            self.data = {"date": self.today(), "quota_units": 0, "backup_quota_units": 0, "runtime_seconds": 0.0, "api_calls": 0}

    def _save(self) -> None:
        atomic_write_json(self.path, self.data)

    def reserve_quota(self, units: int) -> bool:
        with self.lock:
            self._ensure_day()
            used = int(self.data.get(self.slot_key, 0))
            if used + units > SAFE_QUOTA_BUDGET_UNITS:
                self._save()
                return False
            self.data[self.slot_key] = used + units
            self.data["api_calls"] = int(self.data["api_calls"]) + 1
            self._save()
            return True

    def add_runtime(self, seconds: float) -> None:
        if seconds <= 0:
            return
        with self.lock:
            self._ensure_day()
            self.data["runtime_seconds"] = min(DAILY_RUNTIME_LIMIT_SECONDS, float(self.data["runtime_seconds"]) + seconds)
            self._save()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            self._ensure_day()
            return dict(self.data)

    def remaining_runtime(self) -> float:
        return max(0.0, DAILY_RUNTIME_LIMIT_SECONDS - float(self.snapshot()["runtime_seconds"]))

    def remaining_quota(self) -> int:
        return max(0, SAFE_QUOTA_BUDGET_UNITS - int(self.snapshot().get(self.slot_key, 0)))


class YouTubeAPIError(RuntimeError):
    def __init__(self, status_code: int, reason: str, detail: str) -> None:
        self.status_code = status_code
        self.reason = reason
        self.detail = detail
        super().__init__(f"YouTube API error {status_code}" + (f" ({reason})" if reason else "") + f": {detail}")

    @property
    def fatal(self) -> bool:
        if self.reason == "rateLimitExceeded":
            return False
        return self.reason in {"quotaExceeded", "dailyLimitExceeded", "liveChatEnded", "liveChatDisabled", "liveChatNotFound", "forbidden"} or self.status_code in {400, 401} or self.status_code == 403


class YouTubeClient:
    def __init__(self, api_key: str, ledger: UsageLedger) -> None:
        self.api_key = api_key.strip()
        self.ledger = ledger
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"NinuGamingLiveAssistant/{APP_VERSION}"})

    def get_json(self, endpoint: str, params: dict[str, Any], units: int) -> dict[str, Any]:
        if not self.ledger.reserve_quota(units):
            raise RuntimeError("Daily local safety budget reached (9,000 estimated quota units).")
        response = self.session.get(f"{YOUTUBE_API}/{endpoint}", params={**params, "key": self.api_key}, timeout=(10, 35))
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not response.ok:
            err = payload.get("error") or {}
            reasons = [e.get("reason", "") for e in err.get("errors", []) if isinstance(e, dict)]
            reason = next((r for r in reasons if r), "")
            detail = str(err.get("message") or response.text[:400] or response.reason)
            raise YouTubeAPIError(response.status_code, reason, detail)
        if not isinstance(payload, dict):
            raise RuntimeError("YouTube returned an unexpected response format.")
        return payload

    def get_live_chat_id(self, video_id: str) -> str:
        data = self.get_json("videos", {"part": "liveStreamingDetails", "id": video_id}, 1)
        items = data.get("items") or []
        if not items:
            raise RuntimeError("YouTube could not find this video. Check the URL and API key.")
        chat_id = (items[0].get("liveStreamingDetails") or {}).get("activeLiveChatId")
        if not chat_id:
            raise RuntimeError("No active live chat. Make sure the stream is live and live chat is enabled.")
        return str(chat_id)

    def list_messages(self, chat_id: str, page_token: str | None) -> dict[str, Any]:
        params: dict[str, Any] = {"part": "snippet,authorDetails", "liveChatId": chat_id, "maxResults": 2000}
        if page_token:
            params["pageToken"] = page_token
        return self.get_json("liveChat/messages", params, 5)


class ReviewQueue:
    def __init__(self, path: Path = QUEUE_FILE, history_path: Path | None = None) -> None:
        self.path = path
        self.history_path = history_path or path.with_name(path.stem + "_history.json")
        self.lock = threading.RLock()
        loaded = safe_read_json(path, [])
        self.items: list[dict[str, Any]] = loaded if isinstance(loaded, list) else []
        history = safe_read_json(self.history_path, [])
        self.history: list[dict[str, Any]] = history if isinstance(history, list) else []
        self._save()

    def _save(self) -> None:
        atomic_write_json(self.path, self.items)

    def _log_history(self, item: dict[str, Any], status: str) -> None:
        self.history.insert(0, {**item, "status": status, "finished_at": now_iso()})
        del self.history[500:]
        atomic_write_json(self.history_path, self.history)

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return mark_duplicate_uids(self.items)

    def history_snapshot(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(x) for x in self.history[:limit]]

    def upsert(self, record: dict[str, Any]) -> str:
        channel_id = record.get("channel_id") or record.get("id")
        with self.lock:
            for i, old in enumerate(self.items):
                if old.get("channel_id") == channel_id:
                    updated = dict(record)
                    updated["joined_at"] = old.get("joined_at", updated.get("joined_at", now_iso()))
                    updated["updated_at"] = now_iso()
                    self.items[i] = updated
                    self._save()
                    return "updated"
            new = dict(record)
            new["channel_id"] = channel_id
            new.setdefault("joined_at", now_iso())
            new["updated_at"] = now_iso()
            self.items.append(new)
            self._save()
            return "added"

    def _index(self, channel_id: str) -> int:
        return next((i for i, x in enumerate(self.items) if x.get("channel_id") == channel_id), -1)

    def remove(self, channel_id: str) -> bool:
        with self.lock:
            i = self._index(channel_id)
            if i < 0:
                return False
            self._log_history(self.items.pop(i), "removed")
            self._save()
            return True

    def pop_next(self) -> dict[str, Any] | None:
        return self.complete(None)

    def complete(self, channel_id: str | None) -> dict[str, Any] | None:
        """Mark a viewer as reviewed (default: the first in line) and move them to history."""
        with self.lock:
            i = 0 if channel_id is None and self.items else (self._index(channel_id) if channel_id else -1)
            if i < 0 or i >= len(self.items):
                return None
            result = self.items.pop(i)
            self._log_history(result, "done")
            self._save()
            return result

    def move(self, channel_id: str, where: str) -> bool:
        with self.lock:
            i = self._index(channel_id)
            if i < 0:
                return False
            target = {"up": i - 1, "down": i + 1, "top": 0, "bottom": len(self.items) - 1}.get(where, i)
            target = max(0, min(len(self.items) - 1, target))
            if target == i:
                return True
            self.items.insert(target, self.items.pop(i))
            self._save()
            return True

    def edit(self, channel_id: str, uid: str | None = None, request: str | None = None) -> bool:
        with self.lock:
            i = self._index(channel_id)
            if i < 0:
                return False
            if uid is not None:
                clean = re.sub(r"\D", "", uid)
                self.items[i]["uid"] = clean or "not given"
            if request is not None:
                self.items[i]["request"] = " ".join(request.split())[:200] or "Account review"
            self.items[i]["updated_at"] = now_iso()
            self._save()
            return True

    def clear(self) -> None:
        with self.lock:
            self.items.clear()
            self._save()


class LiveAssistant:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.speech_condition = threading.Condition(threading.RLock())
        self.ledger = UsageLedger()
        self.review_queue = ReviewQueue()
        self.running = False
        self.connected = False
        self.paused = False
        self.speech_paused = False
        self.status = "Ready — add your YouTube API key and live stream URL"
        self.last_error = ""
        self.video_id = ""
        self.chat_id = ""
        self.min_delay = DEFAULT_MIN_DELAY
        self.read_mode = "all"
        self.read_existing = False
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.messages: deque[dict[str, Any]] = deque(maxlen=MAX_CHAT_HISTORY)
        self.pending_speech: deque[dict[str, Any]] = deque()
        self.speech_history: deque[dict[str, Any]] = deque(maxlen=MAX_SPEECH_HISTORY)
        self.current_speech: dict[str, Any] | None = None
        self.logs: deque[dict[str, str]] = deque(maxlen=MAX_LOGS)
        self.seen_ids: OrderedDict[str, None] = OrderedDict()
        self.message_count = 0
        self.text_count = 0
        self.review_added_count = 0
        self.poll_count = 0
        self.retry_count = 0
        self.last_poll_at = ""
        self.last_message_at = ""
        self.next_poll_in = 0
        self.last_runtime_tick = 0.0
        self.session_started = ""
        stored = safe_read_json(CONFIG_FILE, {})
        if not isinstance(stored, dict):
            stored = {}
        self.filter = SpamFilter(sanitize_filters(stored.get("filters") or {}))
        self.auto_reconnect = bool(stored.get("auto_reconnect", True))
        self.read_counts: Counter = Counter()
        self.kind_counts: Counter = Counter()
        self.chatters: dict[str, list[Any]] = {}
        self.msg_times: deque[float] = deque(maxlen=900)
        self.spoken_count = 0
        self.dropped_count = 0
        self._alerts_logged: set[str] = set()
        self._last_name = ""
        self._last_name_at = 0.0
        self.shutdown_event = threading.Event()
        self.tts = NeuralSpeechEngine(
            Path(__file__).resolve().parent,
            on_status=self._on_tts_status,
            voice=stored.get("tts_voice", DEFAULT_VOICE),
            speed=stored.get("tts_speed", DEFAULT_SPEED),
            volume=stored.get("tts_volume", DEFAULT_VOLUME),
        )
        self.speech_thread = threading.Thread(target=self._speech_loop, name="speech-worker", daemon=True)
        self.speech_thread.start()
        self.hotkeys = HotkeyManager({
            "next_review": self._hotkey_next_review,
            "stop_current": lambda: self.tts.stop_current(),
            "pause_voice": lambda: self.speech_pause(not self.speech_paused),
            "skip_next": lambda: self.speech_skip_next(),
        })
        if stored.get("hotkeys_enabled", True):
            self.hotkeys.start()
        self.log("Desktop assistant initialized. Kokoro neural voice model is loading in the background.")

    def _hotkey_next_review(self) -> None:
        item = self.review_queue.pop_next()
        self.log(f"Hotkey: next review viewer is {item.get('name', 'viewer')}." if item else "Hotkey: review queue is empty.")

    def set_hotkeys(self, enabled: bool) -> dict[str, Any]:
        if enabled:
            self.hotkeys.start()
        else:
            self.hotkeys.stop()
        return self.hotkeys.status()

    def set_filters(self, payload: dict[str, Any]) -> dict[str, Any]:
        cfg = sanitize_filters(payload, self.filter.raw)
        self.filter.configure(cfg)
        return cfg

    def alerts(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        used = SAFE_QUOTA_BUDGET_UNITS - self.ledger.remaining_quota()
        pct = used * 100 // SAFE_QUOTA_BUDGET_UNITS
        if pct >= 90:
            out.append({"key": "quota90", "level": "critical", "text": f"Quota almost used up: {pct}% of today's safety budget. The listener stops at 100%."})
        elif pct >= 70:
            out.append({"key": "quota70", "level": "warn", "text": f"Quota warning: {pct}% of today's safety budget is used."})
        left = self.ledger.remaining_runtime()
        if self.running and left <= 600:
            out.append({"key": "run10", "level": "critical", "text": f"Only {int(left // 60)} min of today's 4-hour runtime are left."})
        elif self.running and left <= 1800:
            out.append({"key": "run30", "level": "warn", "text": f"{int(left // 60)} min of today's 4-hour runtime are left."})
        if self.ledger.using_backup:
            out.append({"key": "backup", "level": "info", "text": "Using the backup API key (the primary key reached its quota)."})
        return out

    def _check_alerts(self) -> None:
        for a in self.alerts():
            if a["key"] not in self._alerts_logged:
                self._alerts_logged.add(a["key"])
                self.log(a["text"], "warning" if a["level"] != "info" else "info")

    def _rotate_key(self, backup_key: str) -> "YouTubeClient | None":
        if not backup_key or self.ledger.using_backup:
            return None
        self.ledger.use_backup(True)
        self.log("Primary API key reached its quota; switched to the backup API key.", "warning")
        with self.lock:
            self.status = "Switched to the backup API key"
        return YouTubeClient(backup_key, self.ledger)

    def _wait_for_chat_return(self, client: "YouTubeClient") -> str:
        """Chat ended/not found: the stream may just have dropped. Re-check once a minute for 10 minutes."""
        if not self.auto_reconnect:
            return ""
        for attempt in range(1, 11):
            with self.lock:
                self.connected = False
                self.status = f"Chat ended — checking for the stream again (attempt {attempt}/10)"
            self.log(f"Live chat is not available. Auto-reconnect check {attempt}/10 in 60s.", "warning")
            if not self._wait(60):
                return ""
            try:
                chat_id = client.get_live_chat_id(self.video_id)
                self.log("Live chat is back — reconnecting.")
                return chat_id
            except (YouTubeAPIError, RuntimeError, requests.RequestException):
                continue
        return ""

    def _on_tts_status(self, status: str, level: str = "info") -> None:
        self.log(status, level)
        with self.speech_condition:
            self.speech_condition.notify_all()

    def log(self, message: str, level: str = "info") -> None:
        row = {"time": datetime.now().strftime("%H:%M:%S"), "level": level, "message": str(message)[:500]}
        with self.lock:
            self.logs.appendleft(row)

    def _charge_runtime(self) -> None:
        now = time.monotonic()
        with self.lock:
            previous = self.last_runtime_tick
            active = self.running and not self.paused
            self.last_runtime_tick = now
        if previous and active:
            self.ledger.add_runtime(max(0.0, now - previous))

    def start(self, api_key: str, stream_url: str, min_delay: int, read_mode: str, read_existing: bool, backup_key: str = "") -> tuple[bool, str]:
        with self.lock:
            if self.running:
                return False, "The chat listener is already running."
            if not api_key.strip():
                return False, "Enter your YouTube Data API v3 key first."
            try:
                video_id = extract_video_id(stream_url)
            except ValueError as exc:
                return False, str(exc)
            remaining = self.ledger.remaining_runtime()
            if remaining <= 0:
                return False, "Today's four-hour runtime allowance has been used. It resets at midnight Pacific Time."
            self.ledger.use_backup(False)
            if self.ledger.remaining_quota() < 6:
                if backup_key.strip():
                    self.ledger.use_backup(True)
                    if self.ledger.remaining_quota() < 6:
                        return False, "Both API keys have used today's local quota safety budget."
                else:
                    return False, "The local quota safety budget is nearly exhausted for today."
            self.stop_event.clear()
            self.running = True
            self.connected = False
            self.paused = False
            self.status = "Connecting to YouTube live chat…"
            self.last_error = ""
            self.video_id = video_id
            self.min_delay = max(10, min(120, int(min_delay)))
            self.read_mode = read_mode if read_mode in READ_MODES else "all"
            self.read_existing = bool(read_existing)
            self.session_started = now_iso()
            self.last_runtime_tick = time.monotonic()
            self.message_count = self.text_count = self.poll_count = self.retry_count = 0
            self.review_added_count = 0
            self.last_poll_at = self.last_message_at = ""
            self.seen_ids.clear()
            self.read_counts.clear()
            self.kind_counts.clear()
            self.chatters.clear()
            self.msg_times.clear()
            self.spoken_count = self.dropped_count = 0
            self._alerts_logged.clear()
            self.worker = threading.Thread(target=self._run_listener, args=(api_key.strip(), backup_key.strip()), name="youtube-chat-listener", daemon=True)
            self.worker.start()
            self.log(f"Listener starting for video {video_id}; read mode={self.read_mode}; minimum delay={self.min_delay}s.")
            return True, "Connecting…"

    def stop(self) -> None:
        self.stop_event.set()
        self._charge_runtime()
        with self.lock:
            was_running = self.running
            self.running = False
            self.connected = False
            self.paused = False
            self.status = "Stopped by user" if was_running else "Stopped"
        self.log("Listener stopped by user.")

    def set_paused(self, paused: bool) -> None:
        self._charge_runtime()
        with self.lock:
            if not self.running:
                return
            self.paused = bool(paused)
            self.status = "Listener paused — no API calls are being made" if paused else "Resuming live chat…"
            self.last_runtime_tick = time.monotonic()
        self.log("Chat listener paused." if paused else "Chat listener resumed.")

    def _wait(self, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, seconds)
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            self._charge_runtime()
            if self.ledger.remaining_runtime() <= 0:
                self.log("Four-hour daily runtime limit reached.", "warning")
                return False
            with self.lock:
                paused = self.paused
            if paused:
                self.stop_event.wait(0.25)
                continue
            self.stop_event.wait(min(1.0, max(0.05, deadline - time.monotonic())))
        self._charge_runtime()
        return not self.stop_event.is_set()

    def _remember_id(self, message_id: str) -> bool:
        if not message_id:
            return True
        if message_id in self.seen_ids:
            return False
        self.seen_ids[message_id] = None
        if len(self.seen_ids) > MAX_SEEN_IDS:
            self.seen_ids.popitem(last=False)
        return True

    def _route_message(self, raw: dict[str, Any], backlog: bool) -> None:
        record = make_chat_record(raw)
        if not self._remember_id(record["id"]):
            return
        with self.lock:
            self.message_count += 1
            self.last_message_at = record["published_at"]
            self.kind_counts[record["kind"]] += 1
            self.msg_times.append(time.time())
            if record["full_comment"]:
                self.text_count += 1
                cid = record["channel_id"] or record["name"]
                if cid in self.chatters or len(self.chatters) < MAX_CHATTERS:
                    self.chatters.setdefault(cid, [record["name"], 0])[1] += 1
            if backlog and not self.read_existing:
                record["read_state"] = "skipped"
                record["read_reason"] = "This message was already in chat when the listener connected. Enable reading the starting backlog to include it."
            else:
                candidate = record["review_candidate"]
                if candidate:
                    entry = {
                        "name": record["name"], "uid": record["uid"] or "not given",
                        "request": clean_request(record["full_comment"]),
                        "full_comment": record["full_comment"], "channel_id": record["channel_id"] or record["id"],
                        "intent_score": record["intent_score"], "confidence": record["confidence"],
                        "reason": record["reason"], "joined_at": record["published_at"],
                        "message_id": record["id"],
                    }
                    result = self.review_queue.upsert(entry)
                    if result == "added":
                        self.review_added_count += 1
                    record["review_state"] = result
                else:
                    record["review_state"] = "not a review request"
                if not record["full_comment"]:
                    record["read_state"] = "not readable"
                    record["read_reason"] = "YouTube event has no readable text or supported event details."
                elif self.read_mode == "off":
                    record["read_state"] = "filtered"
                    record["read_reason"] = "Speech mode is Off. The chat is still monitored and displayed."
                elif self.read_mode == "review_only" and not candidate:
                    record["read_state"] = "filtered"
                    record["read_reason"] = "Speech mode is set to review requests only; this message was not classified as one."
                elif self.read_mode == "priority" and record["priority"] > 2:
                    record["read_state"] = "filtered"
                    record["read_reason"] = "Speech mode reads only Super Chats, members, mods and review requests."
                else:
                    allowed, why = self._speech_gate(record)
                    if not allowed:
                        record["read_state"] = "filtered"
                        record["read_reason"] = why
                    else:
                        queued = self._enqueue_speech(record)
                        record["read_state"] = "queued" if queued else "queue full"
                        record["read_reason"] = (f"Added to the speech queue ({PRIORITY_LABELS.get(record['priority'], 'Chat')})." if queued
                                                 else "Nothing speakable in this message, or the queue is full of higher-priority speech.")
            self.read_counts[record["read_state"]] += 1
            self.messages.appendleft(record)
        if record.get("review_candidate") and not backlog:
            self.log(f"Review request detected: {record['name']} • UID {record.get('uid') or 'not given'} • {record['confidence']} confidence.")
        if record["read_state"] == "queued":
            self.log(f"Queued for speech: {record['name']}: {record['text'][:110]}")

    def _run_listener(self, api_key: str, backup_key: str = "") -> None:
        client = YouTubeClient(api_key, self.ledger)
        page_token: str | None = None
        first_batch = True
        backoff = 2
        try:
            connect_backoff = 2
            while not self.stop_event.is_set():
                try:
                    self._charge_runtime()
                    chat_id = client.get_live_chat_id(self.video_id)
                    break
                except (requests.RequestException, YouTubeAPIError) as exc:
                    if isinstance(exc, YouTubeAPIError) and exc.fatal:
                        raise
                    delay = min(60, max(float(self.min_delay), float(connect_backoff)))
                    with self.lock:
                        self.connected = False
                        self.last_error = str(exc)
                        self.status = f"Connection lookup failed — retrying in {delay:.0f}s"
                        self.retry_count += 1
                    self.log(f"Could not look up the active chat yet: {exc}. Retrying in {delay:.0f}s.", "warning")
                    connect_backoff = min(60, connect_backoff * 2)
                    if not self._wait(delay):
                        return
            else:
                return
            with self.lock:
                self.chat_id = chat_id
                self.connected = True
                self.status = "Connected — waiting for chat messages"
            self.log("YouTube live chat connection established.")
            while not self.stop_event.is_set():
                with self.lock:
                    paused = self.paused
                if paused:
                    if not self._wait(0.5):
                        break
                    continue
                if self.ledger.remaining_runtime() <= 0:
                    self.log("Four-hour daily runtime allowance reached.", "warning")
                    break
                if self.ledger.remaining_quota() < 5:
                    switched = self._rotate_key(backup_key)
                    if switched is not None:
                        client = switched
                        continue
                    self.log("Quota safety budget reached. Listener stopped to protect the project quota.", "warning")
                    break
                try:
                    data = client.list_messages(chat_id, page_token)
                    self._charge_runtime()
                    backoff = 2
                    items = data.get("items") or []
                    with self.lock:
                        self.poll_count += 1
                        poll_number = self.poll_count
                        self.last_poll_at = now_iso()
                        self.connected = True
                        self.status = f"Live • poll {poll_number} • received {len(items)} item(s)"
                    for raw in items:
                        if self.stop_event.is_set():
                            break
                        self._route_message(raw, backlog=first_batch)
                    first_batch = False
                    self._check_alerts()
                    page_token = data.get("nextPageToken")
                    requested = max(0.0, float(data.get("pollingIntervalMillis", 5000)) / 1000.0)
                    interval = max(float(self.min_delay), requested)
                    with self.lock:
                        self.next_poll_in = int(interval)
                    if items:
                        self.log(f"Poll {poll_number}: {len(items)} new API item(s); next request in {interval:.1f}s.")
                    else:
                        self.log(f"Poll {poll_number}: no new chat; next request in {interval:.1f}s.")
                    if not self._wait(interval):
                        break
                except YouTubeAPIError as exc:
                    with self.lock:
                        self.connected = False
                        self.last_error = str(exc)
                    if exc.reason == "rateLimitExceeded":
                        delay = max(float(self.min_delay), float(backoff))
                        self.log(f"YouTube asked the client to slow down; retrying in {delay:.0f}s. Page token retained.", "warning")
                        self.status = f"Rate limited — retrying in {delay:.0f}s"
                        self.retry_count += 1
                        backoff = min(90, backoff * 2)
                        if not self._wait(delay):
                            break
                        continue
                    if exc.reason == "quotaExceeded" or exc.reason == "dailyLimitExceeded":
                        switched = self._rotate_key(backup_key)
                        if switched is not None:
                            client = switched
                            continue
                        self.log("YouTube project quota is exhausted. Check Cloud Console; this project may be shared with other apps.", "error")
                        self.status = "Stopped: YouTube project quota exhausted"
                        self.last_error = str(exc)
                        break
                    if exc.reason in {"liveChatEnded", "liveChatDisabled", "liveChatNotFound"}:
                        self.log(str(exc), "warning")
                        revived = self._wait_for_chat_return(client)
                        if revived:
                            chat_id, page_token, first_batch = revived, None, True
                            with self.lock:
                                self.chat_id, self.connected, self.status = revived, True, "Reconnected — waiting for chat messages"
                            continue
                        self.status = "Stream chat is no longer available"
                        self.last_error = str(exc)
                        break
                    if exc.fatal:
                        self.log(str(exc), "error")
                        self.status = "Stopped: YouTube rejected the request"
                        self.last_error = str(exc)
                        break
                    delay = min(60, max(float(self.min_delay), float(backoff)))
                    self.log(f"Recoverable YouTube error: {exc}. Retrying in {delay:.0f}s.", "warning")
                    self.status = f"Recovering from API error — retrying in {delay:.0f}s"
                    self.retry_count += 1
                    backoff = min(60, backoff * 2)
                    if not self._wait(delay):
                        break
                except requests.RequestException as exc:
                    delay = min(60, max(float(self.min_delay), float(backoff)))
                    with self.lock:
                        self.connected = False
                        self.last_error = str(exc)
                        self.status = f"Network problem — retrying in {delay:.0f}s"
                        self.retry_count += 1
                    self.log(f"Network error: {exc}. Retrying in {delay:.0f}s; current page token retained.", "warning")
                    backoff = min(60, backoff * 2)
                    if not self._wait(delay):
                        break
                except Exception as exc:
                    delay = min(60, max(float(self.min_delay), float(backoff)))
                    with self.lock:
                        self.connected = False
                        self.last_error = str(exc)
                        self.status = f"Unexpected problem — retrying in {delay:.0f}s"
                        self.retry_count += 1
                    self.log(f"Unexpected listener error: {exc}. Retrying in {delay:.0f}s.", "error")
                    backoff = min(60, backoff * 2)
                    if not self._wait(delay):
                        break
        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)
                self.status = f"Connection failed: {exc}"
            self.log(f"Could not connect to live chat: {exc}", "error")
        finally:
            self._charge_runtime()
            with self.lock:
                self.running = False
                self.connected = False
                self.paused = False
                if not self.status.startswith("Stopped") and self.status not in {"Stopped by user", "Stream chat is no longer available"}:
                    self.status = "Stopped"
            self.log("Listener session ended.")

    def test_connection(self, api_key: str, stream_url: str) -> dict[str, Any]:
        video_id = extract_video_id(stream_url)
        client = YouTubeClient(api_key, self.ledger)
        chat_id = client.get_live_chat_id(video_id)
        return {"ok": True, "video_id": video_id, "chat_id_found": bool(chat_id), "message": "YouTube reports an active live chat."}

    def _speech_gate(self, record: dict[str, Any]) -> tuple[bool, str]:
        """Spam / language / ignore rules. Paid events and review requests always pass."""
        if record.get("priority", 3) <= 1:
            return True, ""
        cfg = self.filter.settings
        if cfg.get("skip_non_latin", True) and non_latin_ratio(record.get("text", "")) > 0.5:
            return False, "Mostly non-English script — the English neural voice cannot read it (change in Tools → Filters)."
        return self.filter.check(record.get("channel_id", ""), record.get("name", ""), record.get("full_comment", ""))

    def _enqueue_speech(self, record: dict[str, Any]) -> bool:
        cfg = self.filter.settings
        now = time.monotonic()
        same_viewer = record.get("name") == self._last_name and now - self._last_name_at < 20
        include_name = not (cfg.get("drop_repeat_name", True) and same_viewer and record.get("priority", 3) >= 3)
        text = speech_text_for(record, cfg, include_name)
        if not text:
            return False
        entry = {
            "id": record["id"], "name": record["name"], "text": text,
            "created_at": now_iso(), "kind": record.get("kind", "textMessageEvent"),
            "review_candidate": record.get("review_candidate", False),
            "priority": int(record.get("priority", 3)),
        }
        with self.speech_condition:
            accepted, dropped = enqueue_with_priority(self.pending_speech, entry, MAX_PENDING_SPEECH, SOFT_LOW_PRIORITY_LIMIT)
            if accepted:
                self._last_name, self._last_name_at = record.get("name", ""), now
                self.speech_condition.notify_all()
        for old in dropped:
            self.dropped_count += 1
            self._update_chat_read_state(str(old.get("id", "")), "skipped", "Dropped: the speech queue was crowded, so older chat was replaced by newer or higher-priority lines.")
        return accepted

    def _update_chat_read_state(self, message_id: str, state: str, reason: str) -> None:
        with self.lock:
            for row in self.messages:
                if row.get("id") == message_id:
                    row["read_state"] = state
                    row["read_reason"] = reason
                    return

    def enqueue_message_now(self, message_id: str) -> bool:
        with self.lock:
            record = next((dict(m) for m in self.messages if m["id"] == message_id), None)
        if record:
            record["priority"] = min(int(record.get("priority", 3)), 1)   # manual "read this" jumps ahead
        if not record or not self._enqueue_speech(record):
            return False
        self._update_chat_read_state(message_id, "queued", "Manually added to the speech queue by the streamer.")
        return True

    def speech_pause(self, paused: bool) -> None:
        with self.speech_condition:
            self.speech_paused = bool(paused)
            self.speech_condition.notify_all()
        self.log("Speech paused." if paused else "Speech resumed.")

    def speech_skip_next(self) -> dict[str, Any] | None:
        with self.speech_condition:
            if not self.pending_speech:
                return None
            item = self.pending_speech.popleft()
            self.speech_condition.notify_all()
        self._update_chat_read_state(str(item.get("id", "")), "skipped", "Skipped by the streamer before speech.")
        self.log(f"Skipped queued speech for {item.get('name', 'viewer')}.", "warning")
        return item

    def speech_clear(self) -> int:
        with self.speech_condition:
            cleared = list(self.pending_speech)
            count = len(cleared)
            self.pending_speech.clear()
        for item in cleared:
            self._update_chat_read_state(str(item.get("id", "")), "skipped", "Removed when the streamer cleared the speech queue.")
        self.log(f"Cleared {count} queued speech item(s).", "warning")
        return count

    def _speech_loop(self) -> None:
        while not self.shutdown_event.is_set():
            with self.speech_condition:
                while not self.shutdown_event.is_set() and (
                    not self.pending_speech or self.speech_paused or not self.tts.ready
                ):
                    self.speech_condition.wait(timeout=0.5)
                if self.shutdown_event.is_set():
                    break
                item = self.pending_speech.popleft()
                self.current_speech = dict(item)
            success = False
            cancelled = False
            error = ""
            try:
                success = self.tts.speak(item["text"])
                cancelled = not success
            except Exception as exc:
                error = str(exc)
            completed = dict(item)
            completed["completed_at"] = now_iso()
            completed["success"] = success
            completed["cancelled"] = cancelled
            completed["error"] = error
            with self.speech_condition:
                self.current_speech = None
                self.speech_history.appendleft(completed)
                self.speech_condition.notify_all()
            if success:
                self.spoken_count += 1
                self._update_chat_read_state(str(item.get("id", "")), "spoken", "Kokoro neural speech completed successfully.")
                self.log(f"Finished speaking with {self.tts.voice}: {item['name']} — {item['text'][:110]}")
            elif cancelled:
                self._update_chat_read_state(str(item.get("id", "")), "skipped", "Current speech was stopped by the streamer.")
                self.log(f"Stopped current speech for {item.get('name', 'viewer')}.", "warning")
            else:
                self._update_chat_read_state(str(item.get("id", "")), "failed", error or "Kokoro speech playback failed.")
                self.log(f"Neural speech failed for {item.get('name', 'viewer')}: {error}", "error")
                # Do not silently drop all following lines. Continue with the next
                # queued message and keep the failure visible in speech history.
                time.sleep(0.5)

    def test_voice(self) -> int:
        if not self.tts.ready:
            return 0
        tests = [
            {"id": f"test-{time.time_ns()}-1", "name": "Voice test", "text": "Voice test one. NinuGaming Live Assistant is ready.", "created_at": now_iso(), "kind": "test"},
            {"id": f"test-{time.time_ns()}-2", "name": "Voice test", "text": "Voice test two. Consecutive speech should continue working.", "created_at": now_iso(), "kind": "test"},
            {"id": f"test-{time.time_ns()}-3", "name": "Voice test", "text": "Voice test three. You should hear all three messages.", "created_at": now_iso(), "kind": "test"},
        ]
        with self.speech_condition:
            if len(self.pending_speech) + len(tests) > MAX_PENDING_SPEECH:
                return 0
            self.pending_speech.extend(tests)
            self.speech_condition.notify_all()
        self.log("Queued three consecutive Kokoro neural voice tests.")
        return len(tests)

    def preview_voice(self) -> bool:
        if not self.tts.ready:
            return False
        item = {
            "id": f"preview-{time.time_ns()}",
            "name": "Voice preview",
            "text": "Hello! This is the selected Kokoro voice for NinuGaming Live Assistant. How does this voice sound to you?",
            "created_at": now_iso(),
            "kind": "voice-preview",
        }
        with self.speech_condition:
            if len(self.pending_speech) >= MAX_PENDING_SPEECH:
                return False
            self.pending_speech.append(item)
            self.speech_condition.notify_all()
        self.log(f"Queued voice preview for {self.tts.voice}.")
        return True

    def state(self) -> dict[str, Any]:
        with self.lock:
            messages = [dict(x) for x in self.messages]
            running, connected, paused, status = self.running, self.connected, self.paused, self.status
            metrics = {
                "message_count": self.message_count, "text_count": self.text_count,
                "review_added_count": self.review_added_count, "poll_count": self.poll_count,
                "retry_count": self.retry_count, "last_poll_at": self.last_poll_at,
                "last_message_at": self.last_message_at, "last_error": self.last_error,
                "video_id": self.video_id, "min_delay": self.min_delay,
            }
            logs = list(self.logs)
        with self.speech_condition:
            pending = [dict(x) for x in list(self.pending_speech)[:80]]
            current = dict(self.current_speech) if self.current_speech else None
            recent = [dict(x) for x in list(self.speech_history)[:30]]
            voice_paused = self.speech_paused
            pending_count = len(self.pending_speech)
        usage = self.ledger.snapshot()
        seconds_left = max(0, int(self.ledger.remaining_runtime()))
        return {
            "app": APP_NAME, "version": APP_VERSION,
            "running": running, "connected": connected, "paused": paused,
            "speech_paused": voice_paused, "status": status,
            "metrics": metrics, "messages": messages,
            "speech_pending": pending, "speech_pending_count": pending_count,
            "currently_speaking": current, "speech_history": recent,
            "review_queue": self.review_queue.snapshot(), "logs": logs,
            "tts": self.tts.settings(),
            "usage": usage, "runtime_seconds_left": seconds_left,
            "quota_units_left": self.ledger.remaining_quota(),
            "review_queue_count": len(self.review_queue.snapshot()),
            "quota_units_used": int(usage.get(self.ledger.slot_key, 0)),
            "alerts": self.alerts(), "stats": self.session_stats(),
            "filters": self.filter.raw, "auto_reconnect": self.auto_reconnect,
            "hotkeys": self.hotkeys.status(), "using_backup_key": self.ledger.using_backup,
        }

    def session_stats(self) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            top = sorted(self.chatters.values(), key=lambda x: -x[1])[:5]
            return {
                "messages": self.message_count, "unique_chatters": len(self.chatters),
                "messages_per_minute": sum(1 for t in self.msg_times if now - t <= 60),
                "top_chatters": [{"name": n, "count": c} for n, c in top],
                "paid_and_member_events": sum(self.kind_counts[k] for k in PAID_KINDS),
                "queued": self.read_counts.get("queued", 0), "filtered": self.read_counts.get("filtered", 0),
                "spoken": self.spoken_count, "dropped": self.dropped_count,
                "review_added": self.review_added_count, "started": self.session_started,
            }

    def overlay_state(self) -> dict[str, Any]:
        """Small payload for OBS browser sources (instead of the full dashboard state)."""
        keep = ("id", "name", "text", "kind", "is_mod", "is_owner", "is_member", "published_at")
        with self.lock:
            chat = [{k: m.get(k) for k in keep} for m in list(self.messages)[:25] if m.get("full_comment") or m.get("kind") in PAID_KINDS]
            running, connected = self.running, self.connected
        with self.speech_condition:
            current = self.current_speech
            speaking = {"name": current.get("name"), "text": current.get("text")} if current else None
            paused = self.speech_paused
        queue = [{k: x.get(k) for k in ("name", "uid", "uid_region", "uid_valid", "dup_uid", "request", "full_comment")} for x in self.review_queue.snapshot()]
        return {"running": running, "connected": connected, "speech_paused": paused, "currently_speaking": speaking,
                "review_queue": queue[:12], "review_queue_count": len(queue), "chat": chat, "version": APP_VERSION}


assistant = LiveAssistant()
app = Flask(__name__, static_folder=str(ROOT), static_url_path="/static")


def protect_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    for k in SECRET_KEYS:
        out[k] = protect_secret(str(out.get(k) or ""))
    return out


def write_config(cfg: dict[str, Any]) -> None:
    atomic_write_json(CONFIG_FILE, protect_config(cfg))


def load_config() -> dict[str, Any]:
    raw = safe_read_json(CONFIG_FILE, {})
    if not isinstance(raw, dict):
        return {}
    data = dict(raw)
    for k in SECRET_KEYS:
        data[k] = unprotect_secret(str(data.get(k) or ""))
    if os.name == "nt" and any(raw.get(k) and not str(raw[k]).startswith("dpapi:") for k in SECRET_KEYS):
        try:
            write_config(data)          # migrate old plain-text keys to Windows-encrypted storage
        except OSError:
            pass
    return data


def public_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "api_key_set": bool(cfg.get("api_key")), "backup_key_set": bool(cfg.get("backup_api_key")),
        "stream_url": cfg.get("stream_url", ""), "min_delay": cfg.get("min_delay", DEFAULT_MIN_DELAY),
        "read_mode": cfg.get("read_mode", "all"), "read_existing": cfg.get("read_existing", False),
        "tts_voice": cfg.get("tts_voice", DEFAULT_VOICE), "tts_speed": cfg.get("tts_speed", DEFAULT_SPEED),
        "tts_volume": cfg.get("tts_volume", DEFAULT_VOLUME),
        "auto_reconnect": bool(cfg.get("auto_reconnect", True)), "hotkeys_enabled": bool(cfg.get("hotkeys_enabled", True)),
        "profiles": sorted((cfg.get("profiles") or {}).keys()),
    }


def save_config(payload: dict[str, Any]) -> dict[str, Any]:
    old = load_config()
    key = str(payload.get("api_key") or old.get("api_key") or "").strip()
    backup = "" if payload.get("clear_backup_key") else str(payload.get("backup_api_key") or old.get("backup_api_key") or "").strip()
    url = str(payload.get("stream_url") if payload.get("stream_url") is not None else old.get("stream_url", "")).strip()
    delay = max(10, min(120, int(payload.get("min_delay") or old.get("min_delay") or DEFAULT_MIN_DELAY)))
    mode = str(payload.get("read_mode") or old.get("read_mode") or "all")
    if mode not in READ_MODES:
        mode = "all"
    tts_voice = str(payload.get("tts_voice", old.get("tts_voice", DEFAULT_VOICE)))
    if tts_voice not in {item["id"] for item in VOICE_CATALOG}:
        tts_voice = DEFAULT_VOICE
    new = dict(old)
    new.update({
        "api_key": key, "backup_api_key": backup, "stream_url": url, "min_delay": delay, "read_mode": mode,
        "read_existing": bool(payload.get("read_existing", old.get("read_existing", False))),
        "tts_voice": tts_voice,
        "tts_speed": max(0.70, min(1.40, float(payload.get("tts_speed", old.get("tts_speed", DEFAULT_SPEED))))),
        "tts_volume": max(0, min(100, int(payload.get("tts_volume", old.get("tts_volume", DEFAULT_VOLUME))))),
        "auto_reconnect": bool(payload.get("auto_reconnect", old.get("auto_reconnect", True))),
        "hotkeys_enabled": bool(payload.get("hotkeys_enabled", old.get("hotkeys_enabled", True))),
    })
    write_config(new)
    assistant.auto_reconnect = new["auto_reconnect"]
    return public_config(new)


@app.route("/")
def home():
    return send_from_directory(ROOT, "ui.html")


@app.route("/obs")
def obs_display():
    return send_from_directory(ROOT, "obs.html")


@app.route("/health")
def health():
    return jsonify({"ok": True, "app": APP_NAME, "version": APP_VERSION})


@app.route("/api/config", methods=["GET"])
def config_get():
    return jsonify(public_config(load_config()))


@app.route("/api/config", methods=["POST"])
def config_post():
    payload = request.get_json(silent=True) or {}
    try:
        saved = save_config(payload)
        assistant.tts.configure(voice=saved["tts_voice"], speed=saved["tts_speed"], volume=saved["tts_volume"])
        return jsonify({"ok": True, **saved})
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/state")
def state_get():
    return jsonify(assistant.state())


@app.route("/api/test-connection", methods=["POST"])
def test_connection():
    payload = request.get_json(silent=True) or {}
    cfg = load_config()
    key = str(payload.get("api_key") or cfg.get("api_key") or "").strip()
    url = str(payload.get("stream_url") or cfg.get("stream_url") or "").strip()
    if not key or not url:
        return jsonify({"ok": False, "error": "API key and live stream URL are required."}), 400
    try:
        result = assistant.test_connection(key, url)
        assistant.log("Connection test passed. One video lookup was charged to the daily quota estimate.")
        return jsonify(result)
    except Exception as exc:
        assistant.log(f"Connection test failed: {exc}", "error")
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/start", methods=["POST"])
def start_listener():
    payload = request.get_json(silent=True) or {}
    cfg = save_config(payload)
    stored = load_config()
    ok, message = assistant.start(stored.get("api_key", ""), cfg["stream_url"], cfg["min_delay"], cfg["read_mode"], cfg["read_existing"], stored.get("backup_api_key", ""))
    return jsonify({"ok": ok, "message": message}), 200 if ok else 400


@app.route("/api/stop", methods=["POST"])
def stop_listener():
    assistant.stop()
    return jsonify({"ok": True})


@app.route("/api/pause", methods=["POST"])
def pause_listener():
    payload = request.get_json(silent=True) or {}
    assistant.set_paused(bool(payload.get("paused")))
    return jsonify({"ok": True})


@app.route("/api/speech/test", methods=["POST"])
def speech_test():
    count = assistant.test_voice()
    if count == 0:
        return jsonify({"ok": False, "error": assistant.tts.error or assistant.tts.status or "The Kokoro voice model is not ready yet."}), 503
    return jsonify({"ok": True, "queued": count})


@app.route("/api/speech/preview", methods=["POST"])
def speech_preview():
    if not assistant.preview_voice():
        return jsonify({"ok": False, "error": assistant.tts.error or assistant.tts.status or "The Kokoro voice model is not ready yet."}), 503
    return jsonify({"ok": True, "message": f"Preview queued for {assistant.tts.voice}."})


@app.route("/api/speech/pause", methods=["POST"])
def speech_pause():
    payload = request.get_json(silent=True) or {}
    assistant.speech_pause(bool(payload.get("paused")))
    return jsonify({"ok": True})


@app.route("/api/speech/skip-next", methods=["POST"])
def speech_skip_next():
    item = assistant.speech_skip_next()
    return jsonify({"ok": bool(item), "item": item})


@app.route("/api/speech/clear", methods=["POST"])
def speech_clear():
    return jsonify({"ok": True, "cleared": assistant.speech_clear()})


@app.route("/api/speech/read-now", methods=["POST"])
def speech_read_now():
    payload = request.get_json(silent=True) or {}
    message_id = str(payload.get("message_id") or "")
    if not message_id or not assistant.enqueue_message_now(message_id):
        return jsonify({"ok": False, "error": "Message was not found or speech queue is full."}), 404
    return jsonify({"ok": True})


@app.route("/api/speech/settings", methods=["POST"])
def speech_settings():
    payload = request.get_json(silent=True) or {}
    cfg = load_config()
    voice = str(payload.get("voice", cfg.get("tts_voice", DEFAULT_VOICE)))
    speed = max(0.70, min(1.40, float(payload.get("speed", cfg.get("tts_speed", DEFAULT_SPEED)))))
    volume = max(0, min(100, int(payload.get("volume", cfg.get("tts_volume", DEFAULT_VOLUME)))))
    if voice not in {item["id"] for item in VOICE_CATALOG}:
        return jsonify({"ok": False, "error": "That Kokoro voice is not available."}), 400
    cfg.update({"tts_voice": voice, "tts_speed": speed, "tts_volume": volume})
    write_config(cfg)
    settings = assistant.tts.configure(voice=voice, speed=speed, volume=volume)
    return jsonify({"ok": True, "tts": settings})


@app.route("/api/speech/stop-current", methods=["POST"])
def speech_stop_current():
    assistant.tts.stop_current()
    return jsonify({"ok": True})


@app.route("/api/review/remove", methods=["POST"])
def review_remove():
    payload = request.get_json(silent=True) or {}
    key = str(payload.get("channel_id") or "")
    if not key:
        return jsonify({"ok": False, "error": "Missing viewer ID."}), 400
    return jsonify({"ok": assistant.review_queue.remove(key)})


@app.route("/api/review/next", methods=["POST"])
def review_next():
    item = assistant.review_queue.pop_next()
    return jsonify({"ok": bool(item), "item": item})


@app.route("/api/review/clear", methods=["POST"])
def review_clear():
    assistant.review_queue.clear()
    assistant.log("Review queue cleared by user.", "warning")
    return jsonify({"ok": True})


@app.route("/api/overlay")
def overlay_get():
    return jsonify(assistant.overlay_state())


@app.route("/api/review/move", methods=["POST"])
def review_move():
    payload = request.get_json(silent=True) or {}
    ok = assistant.review_queue.move(str(payload.get("channel_id") or ""), str(payload.get("where") or ""))
    return jsonify({"ok": ok})


@app.route("/api/review/edit", methods=["POST"])
def review_edit():
    payload = request.get_json(silent=True) or {}
    ok = assistant.review_queue.edit(str(payload.get("channel_id") or ""), payload.get("uid"), payload.get("request"))
    return jsonify({"ok": ok})


@app.route("/api/review/complete", methods=["POST"])
def review_complete():
    payload = request.get_json(silent=True) or {}
    item = assistant.review_queue.complete(str(payload.get("channel_id") or "") or None)
    return jsonify({"ok": bool(item), "item": item})


@app.route("/api/review/history")
def review_history():
    return jsonify({"ok": True, "history": assistant.review_queue.history_snapshot(100)})


@app.route("/api/filters", methods=["GET", "POST"])
def filters_route():
    if request.method == "POST":
        cfg = assistant.set_filters(request.get_json(silent=True) or {})
        stored = load_config()
        stored["filters"] = cfg
        write_config(stored)
        assistant.log("Speech filters updated.")
    return jsonify({"ok": True, "filters": assistant.filter.raw})


@app.route("/api/hotkeys", methods=["POST"])
def hotkeys_route():
    enabled = bool((request.get_json(silent=True) or {}).get("enabled"))
    stored = load_config()
    stored["hotkeys_enabled"] = enabled
    write_config(stored)
    return jsonify({"ok": True, "hotkeys": assistant.set_hotkeys(enabled)})


PROFILE_FIELDS = ("read_mode", "min_delay", "read_existing", "tts_voice", "tts_speed", "tts_volume")


@app.route("/api/profiles", methods=["POST"])
def profiles_route():
    payload = request.get_json(silent=True) or {}
    action, name = str(payload.get("action") or ""), " ".join(str(payload.get("name") or "").split())[:40]
    stored = load_config()
    profiles = dict(stored.get("profiles") or {})
    if not name:
        return jsonify({"ok": False, "error": "Give the profile a name."}), 400
    if action == "save":
        profiles[name] = {**{k: stored.get(k) for k in PROFILE_FIELDS}, "filters": assistant.filter.raw}
        if len(profiles) > 30:
            return jsonify({"ok": False, "error": "Maximum 30 profiles."}), 400
    elif action == "delete":
        profiles.pop(name, None)
    elif action == "apply":
        prof = profiles.get(name)
        if not prof:
            return jsonify({"ok": False, "error": "Profile not found."}), 404
        stored.update({k: prof[k] for k in PROFILE_FIELDS if prof.get(k) is not None})
        cfg = assistant.set_filters(prof.get("filters") or {})
        stored["filters"] = cfg
        stored["profiles"] = profiles
        write_config(stored)
        assistant.tts.configure(voice=stored.get("tts_voice"), speed=stored.get("tts_speed"), volume=stored.get("tts_volume"))
        assistant.log(f"Profile applied: {name}.")
        return jsonify({"ok": True, "config": public_config(stored), "filters": cfg})
    else:
        return jsonify({"ok": False, "error": "Unknown action."}), 400
    stored["profiles"] = profiles
    write_config(stored)
    return jsonify({"ok": True, "profiles": sorted(profiles)})


_update_cache: dict[str, Any] = {"at": 0.0, "data": None}


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:4]) or (0,)


@app.route("/api/update-check")
def update_check():
    if _update_cache["data"] and time.time() - _update_cache["at"] < 3600:
        return jsonify(_update_cache["data"])
    try:
        r = requests.get(f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest", timeout=6,
                         headers={"Accept": "application/vnd.github+json", "User-Agent": f"NinuGamingLiveAssistant/{APP_VERSION}"})
        if r.status_code == 404:
            data = {"ok": True, "current": APP_VERSION, "latest": APP_VERSION, "update_available": False}
        else:
            r.raise_for_status()
            j = r.json()
            latest = str(j.get("tag_name") or "").lstrip("v")
            data = {"ok": True, "current": APP_VERSION, "latest": latest, "url": j.get("html_url", ""),
                    "update_available": _version_tuple(latest) > _version_tuple(APP_VERSION)}
    except Exception as exc:
        return jsonify({"ok": False, "current": APP_VERSION, "error": f"Could not check for updates: {exc}"}), 200
    _update_cache.update(at=time.time(), data=data)
    return jsonify(data)


def _export_path(prefix: str, ext: str) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return EXPORT_DIR / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"


@app.route("/api/export/<kind>", methods=["POST"])
def export_route(kind: str):
    if kind == "review":
        path = _export_path("review_queue", "csv")
        cols = ["status", "name", "uid", "uid_region", "request", "full_comment", "confidence", "joined_at", "finished_at"]
        rows = [{**r, "status": "waiting"} for r in assistant.review_queue.snapshot()] + assistant.review_queue.history_snapshot(500)
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    elif kind == "logs":
        path = _export_path("activity_log", "txt")
        lines = [f"{x.get('time', '')} [{x.get('level', 'info')}] {x.get('message', '')}" for x in reversed(list(assistant.logs))]
        path.write_text("\n".join(lines) or "No log entries.", encoding="utf-8")
    elif kind == "session":
        path = _export_path("session_summary", "json")
        path.write_text(json.dumps({"app": APP_NAME, "version": APP_VERSION, "exported_at": now_iso(),
                                    "stats": assistant.session_stats(), "usage": assistant.ledger.snapshot()}, indent=2), encoding="utf-8")
    else:
        return jsonify({"ok": False, "error": "Unknown export type."}), 400
    return jsonify({"ok": True, "path": str(path)})


@app.route("/api/open-exports", methods=["POST"])
def open_exports():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(EXPORT_DIR))  # type: ignore[attr-defined]
    return jsonify({"ok": True, "path": str(EXPORT_DIR)})



@app.route("/api/diagnostics", methods=["POST"])
def diagnostics():
    cfg = load_config()
    result = {
        "python": sys.version.split()[0], "platform": sys.platform,
        "data_directory": str(DATA_ROOT), "config_found": CONFIG_FILE.exists(),
        "api_key_saved": bool(cfg.get("api_key")), "stream_url_saved": bool(cfg.get("stream_url")),
        "speech_backend": "Kokoro-82M neural TTS via ONNX Runtime (DirectML GPU / CPU fallback)",
        "listener_running": assistant.running, "listener_connected": assistant.connected,
        "quota_remaining_local": assistant.ledger.remaining_quota(),
        "runtime_seconds_remaining": int(assistant.ledger.remaining_runtime()),
        "last_error": assistant.last_error,
        "tts": assistant.tts.settings(), "hotkeys": assistant.hotkeys.status(),
        "backup_key_saved": bool(cfg.get("backup_api_key")), "using_backup_key": assistant.ledger.using_backup,
    }
    assistant.log("Local diagnostics report generated.")
    return jsonify({"ok": True, "diagnostics": result})


def choose_port(host: str = "127.0.0.1", preferred: int = 8765) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, preferred))
            return preferred
        except OSError:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])


class LocalServer(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True, name="local-web-server")
        self.server = None
        self.port = choose_port()

    def run(self) -> None:
        from werkzeug.serving import make_server
        self.server = make_server("127.0.0.1", self.port, app, threaded=True)
        self.server.serve_forever()

    def shutdown(self) -> None:
        if self.server:
            self.server.shutdown()


def main() -> None:
    server = LocalServer()
    server.start()
    deadline = time.time() + 10
    while server.server is None and time.time() < deadline:
        time.sleep(0.05)
    if server.server is None:
        raise RuntimeError("The local desktop server could not start.")
    url = f"http://127.0.0.1:{server.port}/"
    print(f"{APP_NAME} {APP_VERSION}: {url}")
    print(f"Local data folder: {DATA_ROOT}")
    try:
        if webview is not None:
            webview.create_window(APP_NAME, url, width=1500, height=980, min_size=(1120, 760), background_color="#0a0e18")
            webview.start()
        else:
            webbrowser.open(url)
            server.join()
    finally:
        assistant.stop_event.set()
        assistant.shutdown_event.set()
        assistant.tts.close()
        assistant.hotkeys.stop()
        with assistant.speech_condition:
            assistant.speech_condition.notify_all()
        server.shutdown()


@atexit.register
def _cleanup() -> None:
    try:
        assistant.stop_event.set()
        assistant.shutdown_event.set()
        assistant.tts.close()
        with assistant.speech_condition:
            assistant.speech_condition.notify_all()
    except Exception:
        pass


if __name__ == "__main__":
    main()