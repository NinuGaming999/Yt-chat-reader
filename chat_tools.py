"""Pure helper logic for NinuGaming Live Assistant (no Flask / network imports).

Everything here is unit-testable on any OS: speech text cleanup, spam filtering,
priority queue insertion, and Genshin UID validation.
"""
from __future__ import annotations

import re
import time
import unicodedata
from collections import deque
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Speech text cleanup
# --------------------------------------------------------------------------- #
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
SHORTCODE_RE = re.compile(r"(?<!\w):[A-Za-z_][\w\-]*:")          # YouTube :emoji-name:
REPEAT_LETTER_RE = re.compile(r"([^\W\d_])\1{2,}")                # loooool -> lool
REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1+")
DIGIT_RUN_RE = re.compile(r"\d{6,}")
SPACE_RE = re.compile(r"\s+")

DEFAULT_FILTERS: dict[str, Any] = {
    "max_length": 220,            # characters spoken per message
    "cooldown_seconds": 3.0,      # per-viewer gap between spoken messages
    "duplicate_window": 60.0,     # seconds in which repeats are ignored
    "skip_non_latin": True,       # Kokoro voices are English: skip Bangla/Hindi/Arabic/CJK text
    "strip_emoji": True,
    "ignore_commands": True,      # !commands and /commands
    "drop_repeat_name": True,     # don't repeat "Name:" for consecutive lines from one viewer
    "blocked_words": [],
    "blocked_users": [],
    "pronunciations": {},         # {"Ninu": "Nee noo"}
}


def _is_emoji_char(ch: str) -> bool:
    cp = ord(ch)
    if 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0xFE00 <= cp <= 0xFE0F:
        return True
    return unicodedata.category(ch) in {"So", "Sk", "Cf", "Cn", "Cs"}


def strip_emoji(text: str) -> str:
    return "".join(ch for ch in (text or "") if not _is_emoji_char(ch))


def apply_pronunciations(text: str, mapping: dict[str, str] | None) -> str:
    if not mapping:
        return text
    for word, spoken in mapping.items():
        word = str(word).strip()
        if not word:
            continue
        text = re.sub(r"(?<!\w)" + re.escape(word) + r"(?!\w)", str(spoken), text, flags=re.IGNORECASE)
    return text


def non_latin_ratio(text: str) -> float:
    letters = [ch for ch in text or "" if ch.isalpha()]
    if not letters:
        return 0.0
    foreign = sum(1 for ch in letters if ord(ch) > 0x24F)
    return foreign / len(letters)


def clean_for_speech(text: str, settings: dict[str, Any] | None = None) -> str:
    """Make chat text pleasant for a neural voice. Returns '' when nothing is speakable."""
    cfg = {**DEFAULT_FILTERS, **(settings or {})}
    t = str(text or "")
    t = URL_RE.sub(" link ", t)
    t = SHORTCODE_RE.sub(" ", t)
    if cfg.get("strip_emoji", True):
        t = strip_emoji(t)
    t = t.replace("@", "")
    t = REPEAT_LETTER_RE.sub(r"\1\1", t)
    t = REPEAT_PUNCT_RE.sub(r"\1", t)
    letters = [c for c in t if c.isalpha()]
    if len(letters) >= 8 and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
        t = t.lower()                                   # tame ALL-CAPS shouting
    # Read long numbers (Genshin UIDs) digit by digit instead of "eight hundred million..."
    t = DIGIT_RUN_RE.sub(lambda m: " ".join(m.group(0)), t)
    t = apply_pronunciations(t, cfg.get("pronunciations"))
    t = SPACE_RE.sub(" ", t).strip()
    if not any(ch.isalnum() for ch in t):
        return ""
    limit = max(40, int(cfg.get("max_length") or 220))
    if len(t) > limit:
        cut = t[: limit - 3]
        if " " in cut[limit // 2:]:
            cut = cut[: cut.rfind(" ")]
        t = cut.rstrip(" ,.;:-") + "..."
    return t


def clean_name_for_speech(name: str, settings: dict[str, Any] | None = None) -> str:
    cfg = {**DEFAULT_FILTERS, **(settings or {})}
    n = strip_emoji(str(name or "")).strip().lstrip("@")
    n = SPACE_RE.sub(" ", re.sub(r"[_\-.]+", " ", n)).strip()
    n = re.sub(r"(?<=[A-Za-z])\d{3,}$", "", n)         # "Ninu12345" -> "Ninu"
    n = apply_pronunciations(n.strip(), cfg.get("pronunciations"))
    if not any(ch.isalnum() for ch in n):
        return "Viewer"
    return n[:40]


# --------------------------------------------------------------------------- #
# Spam / duplicate / ignore filter (applies to *speech* only, never to the review queue)
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (text or "").lower())).strip()


class SpamFilter:
    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self.recent: dict[str, tuple[float, str]] = {}          # channel -> (time, norm text)
        self.last_spoke: dict[str, float] = {}
        self.global_texts: dict[str, list[float]] = {}
        self.settings = dict(DEFAULT_FILTERS)
        self.raw = dict(DEFAULT_FILTERS)
        self.configure(settings or {})

    def configure(self, settings: dict[str, Any]) -> None:
        merged = {**DEFAULT_FILTERS, **(settings or {})}
        self.raw = dict(merged)
        merged["blocked_words"] = [_norm(w) for w in merged.get("blocked_words") or [] if _norm(w)]
        merged["blocked_users"] = [str(u).strip().lower().lstrip("@") for u in merged.get("blocked_users") or [] if str(u).strip()]
        self.settings = merged

    def check(self, channel_id: str, name: str, text: str, now: float | None = None) -> tuple[bool, str]:
        now = time.monotonic() if now is None else now
        s = self.settings
        ident = {str(channel_id).lower(), str(name).strip().lower().lstrip("@")}
        if ident & set(s["blocked_users"]):
            return False, "Viewer is on your ignore list."
        stripped = (text or "").strip()
        if s.get("ignore_commands", True) and stripped[:1] in {"!", "/"}:
            return False, "Looks like a bot command (starts with ! or /)."
        norm = _norm(stripped)
        if norm:
            padded = f" {norm} "
            for word in s["blocked_words"]:
                if f" {word} " in padded:
                    return False, "Contains a word from your blocked-word list."
        window = float(s.get("duplicate_window") or 60)
        key = channel_id or name
        prev = self.recent.get(key)
        if prev and norm and prev[1] == norm and now - prev[0] < window:
            return False, "Same message from this viewer a moment ago (duplicate)."
        times = [t for t in self.global_texts.get(norm, []) if now - t < window] if norm else []
        if norm and len(norm) > 3 and len(times) >= 3:
            return False, "Copy-paste spam: this exact text was sent several times."
        cooldown = float(s.get("cooldown_seconds") or 0)
        last = self.last_spoke.get(key)
        if last is not None and now - last < cooldown:
            return False, f"Viewer cooldown ({cooldown:g}s between spoken messages)."
        # Passed: remember it.
        self.recent[key] = (now, norm)
        self.last_spoke[key] = now
        if norm:
            self.global_texts[norm] = times + [now]
        if len(self.recent) > 5000:
            self.recent.clear()
            self.last_spoke.clear()
            self.global_texts.clear()
        return True, ""


# --------------------------------------------------------------------------- #
# Priority speech queue (works on the existing collections.deque)
# --------------------------------------------------------------------------- #
PAID_KINDS = {
    "superChatEvent", "superStickerEvent", "newSponsorEvent",
    "memberMilestoneChatEvent", "membershipGiftingEvent", "giftMembershipReceivedEvent",
}
PRIORITY_LABELS = {0: "Paid / membership", 1: "Review request", 2: "Mod / member", 3: "Chat"}


def speech_priority(record: dict[str, Any]) -> int:
    if record.get("kind") in PAID_KINDS:
        return 0
    if record.get("review_candidate"):
        return 1
    if record.get("is_owner") or record.get("is_mod") or record.get("is_member"):
        return 2
    return 3


def enqueue_with_priority(queue: deque, entry: dict[str, Any], hard_limit: int = 500,
                          soft_limit_low: int = 40) -> tuple[bool, list[dict[str, Any]]]:
    """Insert by priority (stable). Drops the OLDEST lowest-priority lines when the queue is
    crowded so chat never falls minutes behind. Returns (accepted, dropped_entries)."""
    dropped: list[dict[str, Any]] = []
    prio = int(entry.get("priority", 3))

    def drop_oldest_of(level_at_least: int) -> bool:
        for i, e in enumerate(queue):
            if int(e.get("priority", 3)) >= level_at_least:
                del queue[i]
                dropped.append(e)
                return True
        return False

    if prio >= 3:
        low = sum(1 for e in queue if int(e.get("priority", 3)) >= 3)
        while low >= soft_limit_low and drop_oldest_of(3):
            low -= 1
    if len(queue) >= hard_limit:
        worst = max((int(e.get("priority", 3)) for e in queue), default=3)
        if worst < prio or not drop_oldest_of(worst):
            return False, dropped
    idx = len(queue)
    for i, e in enumerate(queue):
        if int(e.get("priority", 3)) > prio:
            idx = i
            break
    queue.insert(idx, entry)
    return True, dropped


# --------------------------------------------------------------------------- #
# Genshin UID helpers
# --------------------------------------------------------------------------- #
UID_REGIONS = {"1": "China", "2": "China", "5": "China (Bilibili)", "6": "America",
               "7": "Europe", "8": "Asia", "9": "TW / HK / MO"}
NUMBER_RE = re.compile(r"(?<!\d)(\d{6,12})(?!\d)")


def pick_uid(text: str) -> str | None:
    """Prefer a 9-10 digit number (a real Genshin UID) when several numbers appear."""
    matches = NUMBER_RE.findall(text or "")
    if not matches:
        return None
    for m in matches:
        if len(m) in (9, 10):
            return m
    return matches[0]


def validate_uid(uid: str | None) -> dict[str, Any]:
    uid = (uid or "").strip()
    if not uid or not uid.isdigit():
        return {"valid": False, "region": "", "note": "No UID given."}
    if len(uid) not in (9, 10):
        return {"valid": False, "region": "", "note": f"{len(uid)} digits; Genshin UIDs normally have 9 digits. Double-check it."}
    region = UID_REGIONS.get(uid[0], "") if len(uid) == 9 else ""
    return {"valid": True, "region": region, "note": ""}


def mark_duplicate_uids(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(x) for x in items]
    counts: dict[str, int] = {}
    for row in rows:
        uid = str(row.get("uid") or "")
        if uid.isdigit():
            counts[uid] = counts.get(uid, 0) + 1
    for row in rows:
        uid = str(row.get("uid") or "")
        v = validate_uid(uid)
        row["uid_valid"] = v["valid"]
        row["uid_region"] = v["region"]
        row["uid_note"] = v["note"]
        row["dup_uid"] = bool(uid.isdigit() and counts.get(uid, 0) > 1)
    return rows


def _lines(value: Any) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"[\n,]+", value)
    return [str(v).strip() for v in (value or []) if str(v).strip()][:300]


def sanitize_filters(payload: dict[str, Any], base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate/clamp filter settings coming from the UI."""
    cfg = {**DEFAULT_FILTERS, **(base or {})}
    p = payload or {}
    def num(key: str, lo: float, hi: float, cast=float) -> None:
        if key in p:
            try:
                cfg[key] = cast(max(lo, min(hi, float(p[key]))))
            except (TypeError, ValueError):
                pass
    num("max_length", 40, 500, int)
    num("cooldown_seconds", 0, 60)
    num("duplicate_window", 0, 600)
    for key in ("skip_non_latin", "strip_emoji", "ignore_commands", "drop_repeat_name"):
        if key in p:
            cfg[key] = bool(p[key])
    if "blocked_words" in p:
        cfg["blocked_words"] = _lines(p["blocked_words"])
    if "blocked_users" in p:
        cfg["blocked_users"] = _lines(p["blocked_users"])
    if "pronunciations" in p:
        raw = p["pronunciations"]
        pairs: dict[str, str] = {}
        if isinstance(raw, str):
            for line in raw.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() and v.strip():
                        pairs[k.strip()] = v.strip()
        elif isinstance(raw, dict):
            pairs = {str(k).strip(): str(v).strip() for k, v in raw.items() if str(k).strip() and str(v).strip()}
        cfg["pronunciations"] = dict(list(pairs.items())[:200])
    return cfg
