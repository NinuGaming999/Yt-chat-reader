import tempfile
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

try:
    import flask  # noqa: F401
except ImportError:  # pure-logic tests still run without Flask
    import sys
    import types
    fake = types.ModuleType("flask")
    class _F:
        def __init__(self, *a, **k): self.config = {}
        def route(self, *a, **k): return lambda fn: fn
    fake.Flask, fake.jsonify = _F, (lambda *a, **k: a[0] if a else {})
    fake.request = types.SimpleNamespace(get_json=lambda **k: {})
    fake.send_from_directory = lambda *a, **k: None
    sys.modules["flask"] = fake

import app as app_module
from app import LiveAssistant, ReviewQueue, UsageLedger, make_chat_record, speech_text_for
from chat_tools import (SpamFilter, clean_for_speech, clean_name_for_speech, enqueue_with_priority,
                        non_latin_ratio, pick_uid, sanitize_filters, speech_priority, validate_uid)
from secure_store import protect_secret, unprotect_secret


def raw(mid, text, name="Viewer", cid=None, kind="textMessageEvent", **author):
    return {"id": mid, "snippet": {"type": kind, "displayMessage": text},
            "authorDetails": {"displayName": name, "channelId": cid or "UC-" + mid, **author}}


class SpeechCleanupTests(unittest.TestCase):
    def test_links_emoji_shortcodes_mentions(self):
        self.assertEqual(clean_for_speech("check https://x.com/abc now 😂🔥 :face-red-heart-eyes: @Ninu"), "check link now Ninu")

    def test_repeated_letters_and_caps(self):
        self.assertEqual(clean_for_speech("loooooool!!!"), "lool!")
        self.assertEqual(clean_for_speech("THIS STREAM IS AMAZING"), "this stream is amazing")

    def test_uid_read_digit_by_digit(self):
        self.assertIn("8 1 2 3 4 5 6 7 8", clean_for_speech("uid 812345678"))

    def test_nothing_speakable(self):
        self.assertEqual(clean_for_speech("😂😂😂"), "")

    def test_truncation_and_pronunciation(self):
        out = clean_for_speech("word " * 100, {"max_length": 50})
        self.assertLessEqual(len(out), 50)
        self.assertTrue(out.endswith("..."))
        self.assertEqual(clean_for_speech("hi Ninu", {"pronunciations": {"ninu": "Nee noo"}}), "hi Nee noo")

    def test_names(self):
        self.assertEqual(clean_name_for_speech("@Ninu_Gaming12345 🔥"), "Ninu Gaming")
        self.assertEqual(clean_name_for_speech("🔥🔥"), "Viewer")

    def test_non_latin_ratio(self):
        self.assertGreater(non_latin_ratio("আপনার স্ট্রিম খুব ভালো"), 0.9)
        self.assertEqual(non_latin_ratio("hello"), 0.0)

    def test_speech_text_keeps_old_format(self):
        rec = make_chat_record(raw("a", "hello", "Ninu"))
        self.assertEqual(speech_text_for(rec), "Ninu: hello")
        self.assertEqual(speech_text_for(rec, include_name=False), "hello")


class SpamFilterTests(unittest.TestCase):
    def test_rules(self):
        f = SpamFilter({"cooldown_seconds": 3, "duplicate_window": 60, "blocked_words": ["badword"], "blocked_users": ["troll"]})
        self.assertTrue(f.check("c1", "A", "hello there", now=0)[0])
        self.assertFalse(f.check("c1", "A", "hello there", now=10)[0])          # duplicate
        self.assertFalse(f.check("c1", "A", "something else", now=1)[0])        # cooldown
        self.assertTrue(f.check("c1", "A", "something else", now=5)[0])
        self.assertFalse(f.check("c2", "B", "you BADWORD!", now=5)[0])
        self.assertFalse(f.check("c3", "Troll", "hi", now=5)[0])
        self.assertFalse(f.check("c4", "D", "!command", now=5)[0])

    def test_copy_paste_spam(self):
        f = SpamFilter({"cooldown_seconds": 0})
        results = [f.check(f"c{i}", f"N{i}", "buy followers now", now=i)[0] for i in range(5)]
        self.assertEqual(results, [True, True, True, False, False])


class PriorityQueueTests(unittest.TestCase):
    def test_order_and_overflow(self):
        q = deque()
        def add(name, p): return enqueue_with_priority(q, {"name": name, "priority": p}, hard_limit=10, soft_limit_low=3)
        for n in "abc":
            add(n, 3)
        add("review", 1)
        add("paid", 0)
        self.assertEqual([e["name"] for e in q], ["paid", "review", "a", "b", "c"])
        ok, dropped = add("d", 3)                      # 4th low-priority line drops the oldest low one
        self.assertTrue(ok)
        self.assertEqual([e["name"] for e in dropped], ["a"])
        self.assertEqual([e["name"] for e in q][-3:], ["b", "c", "d"])

    def test_hard_limit_keeps_high_priority(self):
        q = deque({"name": f"p{i}", "priority": 0} for i in range(3))
        ok, _ = enqueue_with_priority(q, {"name": "x", "priority": 3}, hard_limit=3)
        self.assertFalse(ok)
        ok, dropped = enqueue_with_priority(q, {"name": "y", "priority": 0}, hard_limit=3)
        self.assertTrue(ok)
        self.assertEqual(len(dropped), 1)

    def test_priority_of_records(self):
        self.assertEqual(speech_priority(make_chat_record(raw("1", "x", kind="superChatEvent"))), 0)
        self.assertEqual(speech_priority(make_chat_record(raw("2", "please review my account"))), 1)
        self.assertEqual(speech_priority(make_chat_record(raw("3", "hi", isChatModerator=True))), 2)
        self.assertEqual(speech_priority(make_chat_record(raw("4", "hi"))), 3)


class UidTests(unittest.TestCase):
    def test_validate(self):
        self.assertTrue(validate_uid("812345678")["valid"])
        self.assertEqual(validate_uid("812345678")["region"], "Asia")
        self.assertFalse(validate_uid("1234567")["valid"])
        self.assertFalse(validate_uid("not given")["valid"])

    def test_pick_prefers_real_uid(self):
        self.assertEqual(pick_uid("room 1234567890123 uid 812345678"), "812345678")


class QueueUpgradeTests(unittest.TestCase):
    def test_move_edit_complete_history_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = ReviewQueue(Path(tmp) / "q.json")
            for cid, uid in (("a", "811111111"), ("b", "822222222"), ("c", "811111111")):
                q.upsert({"channel_id": cid, "name": cid.upper(), "uid": uid})
            snap = q.snapshot()
            self.assertTrue(snap[0]["dup_uid"] and snap[2]["dup_uid"] and not snap[1]["dup_uid"])
            q.move("c", "top")
            self.assertEqual([x["channel_id"] for x in q.snapshot()], ["c", "a", "b"])
            q.move("c", "down")
            self.assertEqual(q.snapshot()[1]["channel_id"], "c")
            self.assertTrue(q.edit("b", uid="uid 833 333 333"))
            self.assertEqual(q.snapshot()[2]["uid"], "833333333")
            self.assertEqual(q.complete("b")["channel_id"], "b")
            self.assertEqual(q.pop_next()["channel_id"], "a")
            self.assertTrue(q.remove("c"))
            self.assertEqual([h["status"] for h in q.history_snapshot()], ["removed", "done", "done"])
            again = ReviewQueue(Path(tmp) / "q.json")
            self.assertEqual(len(again.history_snapshot()), 3)


class SecretTests(unittest.TestCase):
    def test_plain_roundtrip_off_windows(self):
        self.assertEqual(unprotect_secret(protect_secret("AIzaKey")), "AIzaKey")
        self.assertEqual(unprotect_secret(""), "")

    def test_foreign_token_is_treated_as_unset(self):
        self.assertEqual(unprotect_secret("dpapi:AAAA"), "")


class LedgerSlotTests(unittest.TestCase):
    def test_backup_slot_has_own_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = UsageLedger(Path(tmp) / "u.json")
            self.assertTrue(ledger.reserve_quota(app_module.SAFE_QUOTA_BUDGET_UNITS))
            self.assertFalse(ledger.reserve_quota(5))
            ledger.use_backup(True)
            self.assertTrue(ledger.reserve_quota(5))
            self.assertEqual(ledger.remaining_quota(), app_module.SAFE_QUOTA_BUDGET_UNITS - 5)


class AssistantUpgradeTests(unittest.TestCase):
    def make_bot(self, tmp):
        bot = LiveAssistant()
        bot.ledger = UsageLedger(Path(tmp) / "usage.json")
        bot.review_queue = ReviewQueue(Path(tmp) / "review.json")
        bot.read_existing = True
        bot.read_mode = "all"
        bot.filter.configure({"cooldown_seconds": 0})
        bot.tts.ready = False
        return bot

    def test_priority_and_filters_in_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = self.make_bot(tmp)
            for r in (raw("1", "hello chat"), raw("2", "thanks!", kind="superChatEvent"),
                      raw("3", "আপনার স্ট্রিম খুব ভালো"), raw("4", "!uptime"), raw("5", "check https://spam.com")):
                bot._route_message(r, backlog=False)
            states = {m["id"]: m["read_state"] for m in bot.messages}
            self.assertEqual(states, {"1": "queued", "2": "queued", "3": "filtered", "4": "filtered", "5": "queued"})
            self.assertEqual([e["id"] for e in bot.pending_speech][0], "2")            # paid reads first
            self.assertIn("link", [e["text"] for e in bot.pending_speech if e["id"] == "5"][0])

    def test_priority_mode_and_repeat_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = self.make_bot(tmp)
            bot.read_mode = "priority"
            bot._route_message(raw("1", "hello"), backlog=False)
            bot._route_message(raw("2", "mod here", isChatModerator=True), backlog=False)
            self.assertEqual({m["id"]: m["read_state"] for m in bot.messages}, {"1": "filtered", "2": "queued"})
            bot.read_mode = "all"
            bot._route_message(raw("3", "first line", name="Sam", cid="S"), backlog=False)
            bot._route_message(raw("4", "second line", name="Sam", cid="S"), backlog=False)
            texts = [e["text"] for e in bot.pending_speech]
            self.assertIn("Sam: first line", texts)
            self.assertIn("second line", texts)

    def test_review_duplicate_flag_and_stats_alerts(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = self.make_bot(tmp)
            bot._route_message(raw("1", "please review my account 812345678", name="A", cid="A"), backlog=False)
            bot._route_message(raw("2", "review me 812345678", name="B", cid="B"), backlog=False)
            state = bot.state()
            self.assertTrue(all(x["dup_uid"] for x in state["review_queue"]))
            self.assertEqual(state["stats"]["unique_chatters"], 2)
            bot.ledger.data["quota_units"] = 7000
            self.assertEqual(bot.state()["alerts"][0]["level"], "warn")
            bot.ledger.data["quota_units"] = 8500
            self.assertEqual(bot.state()["alerts"][0]["level"], "critical")
            self.assertLessEqual(len(bot.overlay_state()["chat"]), 25)

    def test_key_rotation_on_quota_exceeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = self.make_bot(tmp)
            bot.running = True
            bot.last_runtime_tick = time.monotonic()
            bot.stop_event.clear()
            calls = []

            class Client:
                def __init__(self, key, *_a):
                    self.key = key
                def get_live_chat_id(self, _v):
                    return "chat"
                def list_messages(self, _c, _t):
                    calls.append(self.key)
                    if self.key == "primary":
                        raise app_module.YouTubeAPIError(403, "quotaExceeded", "quota")
                    bot.stop_event.set()
                    return {"items": [], "pollingIntervalMillis": 1000}

            with patch("app.YouTubeClient", Client), patch.object(bot, "_wait", side_effect=lambda _s: not bot.stop_event.is_set()):
                bot._run_listener("primary", "backup")
            self.assertEqual(calls, ["primary", "backup"])
            self.assertTrue(bot.ledger.using_backup)

    def test_auto_reconnect_after_chat_ended(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = self.make_bot(tmp)
            bot.running = True
            bot.last_runtime_tick = time.monotonic()
            bot.stop_event.clear()
            seen = []

            class Client:
                def __init__(self, *_a):
                    self.polls = 0
                def get_live_chat_id(self, _v):
                    return "chat-2" if self.polls else "chat-1"
                def list_messages(self, chat_id, _t):
                    seen.append(chat_id)
                    self.polls += 1
                    if self.polls == 1:
                        raise app_module.YouTubeAPIError(403, "liveChatEnded", "ended")
                    bot.stop_event.set()
                    return {"items": [], "pollingIntervalMillis": 1000}

            with patch("app.YouTubeClient", Client), patch.object(bot, "_wait", side_effect=lambda _s: not bot.stop_event.is_set()):
                bot._run_listener("k")
            self.assertEqual(seen, ["chat-1", "chat-2"])


class FilterSanitizeTests(unittest.TestCase):
    def test_clamps_and_parses(self):
        cfg = sanitize_filters({"max_length": 9999, "cooldown_seconds": -4, "blocked_words": "a, b\nc",
                                "pronunciations": "Ninu=Nee noo\nbad line"})
        self.assertEqual(cfg["max_length"], 500)
        self.assertEqual(cfg["cooldown_seconds"], 0)
        self.assertEqual(cfg["blocked_words"], ["a", "b", "c"])
        self.assertEqual(cfg["pronunciations"], {"Ninu": "Nee noo"})


if __name__ == "__main__":
    unittest.main()
