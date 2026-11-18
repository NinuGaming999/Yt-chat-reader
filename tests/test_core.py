import tempfile
import unittest
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

# Let pure-core unit tests run in offline build environments that have not
# installed Flask yet. The actual app installs Flask through requirements.txt.
try:
    import flask  # noqa: F401
except ImportError:
    import sys
    import types
    fake = types.ModuleType("flask")
    class _FakeFlask:
        def __init__(self, *args, **kwargs):
            self.config = {}
        def route(self, *args, **kwargs):
            return lambda fn: fn
    fake.Flask = _FakeFlask
    fake.jsonify = lambda *args, **kwargs: args[0] if args else {}
    fake.request = types.SimpleNamespace(get_json=lambda **kwargs: {})
    fake.send_from_directory = lambda *args, **kwargs: None
    sys.modules["flask"] = fake

import app as app_module
from speech_engine import DEFAULT_VOICE, NeuralSpeechEngine, VOICE_CATALOG

from app import (
    LiveAssistant,
    ReviewQueue,
    UsageLedger,
    classify_review_request,
    clean_request,
    estimate_four_hour_quota,
    extract_uid,
    extract_video_id,
    make_chat_record,
    message_text,
    clean_tts_text,
    compose_short_chat_batch,
    speech_text_for,
)


class NeuralVoiceConfigurationTests(unittest.TestCase):
    def test_default_voice_exists_in_catalog(self):
        ids = {voice["id"] for voice in VOICE_CATALOG}
        self.assertIn(DEFAULT_VOICE, ids)
        self.assertGreaterEqual(len(ids), 6)

    def test_catalog_contains_multiple_accents_and_genders(self):
        ids = {voice["id"] for voice in VOICE_CATALOG}
        self.assertIn("en-Carter_man", ids)
        self.assertIn("en-Emma_woman", ids)
        self.assertIn("en-Grace_woman", ids)
        self.assertIn("en-Mike_man", ids)

    def test_speed_and_volume_are_clamped(self):
        self.assertEqual(NeuralSpeechEngine._clamp_speed(9), 1.4)
        self.assertEqual(NeuralSpeechEngine._clamp_speed(0.1), 1.0)
        self.assertEqual(NeuralSpeechEngine._clamp_speed(1.2), 1.2)
        self.assertEqual(NeuralSpeechEngine._clamp_volume(200), 100)
        self.assertEqual(NeuralSpeechEngine._clamp_volume(-10), 0)


class VideoIdTests(unittest.TestCase):
    def test_watch_url(self):
        self.assertEqual(extract_video_id("https://www.youtube.com/watch?v=abcdefghijk"), "abcdefghijk")

    def test_live_url(self):
        self.assertEqual(extract_video_id("https://youtube.com/live/abcdefghijk?feature=share"), "abcdefghijk")

    def test_short_url(self):
        self.assertEqual(extract_video_id("https://youtu.be/abcdefghijk?t=5"), "abcdefghijk")

    def test_raw_id(self):
        self.assertEqual(extract_video_id("abcdefghijk"), "abcdefghijk")

    def test_invalid_url(self):
        with self.assertRaises(ValueError):
            extract_video_id("https://example.com/abcdefghijk")


class ReviewDetectionTests(unittest.TestCase):
    def test_review_phrase_and_uid(self):
        result = classify_review_request("123456789 please review my account")
        self.assertTrue(result["candidate"])
        self.assertEqual(result["uid"], "123456789")
        self.assertEqual(result["confidence"], "High")

    def test_random_chat_not_a_review_request(self):
        self.assertFalse(classify_review_request("hello everyone, how are you today?")["candidate"])

    def test_join_queue_phrase(self):
        self.assertTrue(classify_review_request("please add me to the queue")["candidate"])

    def test_uid_extraction(self):
        self.assertEqual(extract_uid("my uid is 812345678"), "812345678")
        self.assertIsNone(extract_uid("not a uid"))

    def test_clean_request(self):
        self.assertEqual(clean_request("123456789 please review my account"), "Account review")


class MessageTests(unittest.TestCase):
    def test_text_chat_record(self):
        item = {
            "id": "m1",
            "snippet": {"type": "textMessageEvent", "displayMessage": "hello", "publishedAt": "2026-01-01T00:00:00Z"},
            "authorDetails": {"displayName": "Ninu", "channelId": "UC123"},
        }
        record = make_chat_record(item)
        self.assertEqual(record["name"], "Ninu")
        self.assertEqual(record["text"], "hello")
        self.assertEqual(speech_text_for(record), "Ninu: hello")

    def test_superchat_without_text_comment_is_readable(self):
        text, kind = message_text({"snippet": {"type": "superChatEvent", "superChatDetails": {"amountDisplayString": "$5"}}})
        self.assertIn("Super Chat", text)
        self.assertEqual(kind, "superChatEvent")

    def test_uid_review_message_is_detected(self):
        record = make_chat_record({
            "id": "m2", "snippet": {"displayMessage": "Please review 123456789"},
            "authorDetails": {"displayName": "Guest", "channelId": "UCguest"},
        })
        self.assertTrue(record["review_candidate"])
        self.assertEqual(record["uid"], "123456789")


class TtsInputPreparationTests(unittest.TestCase):
    def test_emojis_symbols_and_accented_latin_are_cleaned(self):
        self.assertEqual(clean_tts_text("Café player 🔥✨ — Let's go!!!"), "Cafe player Let's go!")

    def test_emoji_only_and_unsupported_scripts_are_not_sent_to_tts(self):
        self.assertEqual(clean_tts_text("🔥✨"), "")
        self.assertEqual(clean_tts_text("東京"), "")

    def test_short_chat_messages_are_combined_with_context(self):
        batch = compose_short_chat_batch([
            {"id": "m1", "name": "Alice", "body": "hi", "created_at": "now"},
            {"id": "m2", "name": "Bob", "body": "gg", "created_at": "later"},
        ])
        self.assertEqual(batch["source_ids"], ["m1", "m2"])
        self.assertEqual(batch["combined_count"], 2)
        self.assertIn("Quick chat update.", batch["text"])
        self.assertIn("Alice says: hi", batch["text"])
        self.assertIn("Bob says: gg", batch["text"])


class QueueTests(unittest.TestCase):
    def test_upsert_keeps_viewer_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = ReviewQueue(Path(tmp) / "queue.json")
            first = {"channel_id": "UC1", "name": "A", "uid": "111111111", "joined_at": "first"}
            queue.upsert(first)
            updated = dict(first, uid="222222222", joined_at="later")
            self.assertEqual(queue.upsert(updated), "updated")
            snap = queue.snapshot()
            self.assertEqual(len(snap), 1)
            self.assertEqual(snap[0]["uid"], "222222222")
            self.assertEqual(snap[0]["joined_at"], "first")

    def test_remove_and_pop_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = ReviewQueue(Path(tmp) / "queue.json")
            queue.upsert({"channel_id": "UC1", "name": "A"})
            queue.upsert({"channel_id": "UC2", "name": "B"})
            self.assertTrue(queue.remove("UC2"))
            self.assertEqual(queue.pop_next()["name"], "A")
            self.assertIsNone(queue.pop_next())


class QuotaTests(unittest.TestCase):
    def test_estimate_at_default_delay(self):
        self.assertEqual(estimate_four_hour_quota(10), 7201)

    def test_longer_delay_reduces_quota(self):
        self.assertLess(estimate_four_hour_quota(30), estimate_four_hour_quota(10))

    def test_usage_ledger_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            ledger = UsageLedger(path)
            self.assertTrue(ledger.reserve_quota(6))
            ledger.add_runtime(12)
            other = UsageLedger(path)
            self.assertEqual(other.snapshot()["quota_units"], 6)
            self.assertGreaterEqual(other.snapshot()["runtime_seconds"], 12)


class ListenerContinuityTests(unittest.TestCase):
    def test_listener_processes_more_than_one_poll_and_advances_page_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = LiveAssistant()
            bot.ledger = UsageLedger(Path(tmp) / "usage.json")
            bot.running = True
            bot.read_existing = True
            bot.read_mode = "off"
            bot.last_runtime_tick = time.monotonic()
            bot.stop_event.clear()

            def item(message_id, text):
                return {
                    "id": message_id,
                    "snippet": {"type": "textMessageEvent", "displayMessage": text},
                    "authorDetails": {"displayName": "Viewer", "channelId": "UC-" + message_id},
                }

            class FakeClient:
                def __init__(self, *_args):
                    self.tokens = []
                def get_live_chat_id(self, _video_id):
                    return "chat-id"
                def list_messages(self, _chat_id, page_token):
                    self.tokens.append(page_token)
                    if len(self.tokens) == 1:
                        return {"items": [item("m1", "first message")], "nextPageToken": "token-1", "pollingIntervalMillis": 1000}
                    if len(self.tokens) == 2:
                        return {"items": [item("m2", "second message")], "nextPageToken": "token-2", "pollingIntervalMillis": 1000}
                    bot.stop_event.set()
                    return {"items": [], "nextPageToken": "token-2", "pollingIntervalMillis": 1000}

            fake_client = FakeClient()
            with patch("app.YouTubeClient", return_value=fake_client):
                with patch.object(bot, "_wait", side_effect=lambda _seconds: not bot.stop_event.is_set()):
                    bot._run_listener("test-key")

            self.assertEqual(fake_client.tokens, [None, "token-1", "token-2"])
            self.assertEqual(bot.message_count, 2)
            self.assertEqual(bot.poll_count, 3)
            self.assertEqual({m["id"] for m in bot.messages}, {"m1", "m2"})

    def test_review_message_enters_both_chat_and_review_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = LiveAssistant()
            bot.ledger = UsageLedger(Path(tmp) / "usage.json")
            bot.review_queue = ReviewQueue(Path(tmp) / "review.json")
            bot.read_mode = "off"
            bot.read_existing = True
            raw = {
                "id": "review-1",
                "snippet": {"type": "textMessageEvent", "displayMessage": "Please review my account 123456789"},
                "authorDetails": {"displayName": "Genshin viewer", "channelId": "UC-review"},
            }
            bot._route_message(raw, backlog=False)
            state = bot.state()
            self.assertEqual(state["messages"][0]["read_state"], "filtered")
            self.assertEqual(state["review_queue_count"], 1)
            self.assertEqual(state["review_queue"][0]["uid"], "123456789")


if __name__ == "__main__":
    unittest.main()