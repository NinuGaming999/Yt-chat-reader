import tempfile
import unittest
from pathlib import Path
import time
from unittest.mock import patch

# Allow pure-core tests to run where optional Flask UI dependencies are absent.
try:
    import flask  # noqa: F401
except ImportError:
    import sys
    import types
    fake = types.ModuleType("flask")
    class _FakeFlask:
        def __init__(self, *args, **kwargs): self.config = {}
        def route(self, *args, **kwargs): return lambda fn: fn
    fake.Flask = _FakeFlask
    fake.jsonify = lambda *args, **kwargs: args[0] if args else {}
    fake.request = types.SimpleNamespace(get_json=lambda **kwargs: {})
    fake.send_from_directory = lambda *args, **kwargs: None
    sys.modules["flask"] = fake

from app import (
    LiveAssistant, ReviewQueue, UsageLedger, classify_review_request,
    clean_request, estimate_four_hour_quota, extract_uid, extract_video_id,
    make_chat_record, message_text, speech_text_for,
)


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
        with self.assertRaises(ValueError): extract_video_id("https://example.com/abcdefghijk")


class ReviewDetectionTests(unittest.TestCase):
    def test_review_phrase_and_uid(self):
        result = classify_review_request("123456789 please review my account")
        self.assertTrue(result["candidate"])
        self.assertEqual(result["uid"], "123456789")
        self.assertEqual(result["confidence"], "High")
    def test_random_chat_not_review_request(self):
        self.assertFalse(classify_review_request("hello everyone, how are you?")["candidate"])
    def test_join_queue_phrase(self):
        self.assertTrue(classify_review_request("please add me to the queue")["candidate"])
    def test_uid_extraction(self):
        self.assertEqual(extract_uid("my uid is 812345678"), "812345678")
        self.assertIsNone(extract_uid("not a uid"))
    def test_clean_request(self):
        self.assertEqual(clean_request("123456789 please review my account"), "Account review")


class MessageTests(unittest.TestCase):
    def test_text_chat_record(self):
        item = {"id":"m1","snippet":{"type":"textMessageEvent","displayMessage":"hello","publishedAt":"2026-01-01T00:00:00Z"},"authorDetails":{"displayName":"Ninu","channelId":"UC123"}}
        record = make_chat_record(item)
        self.assertEqual(record["name"], "Ninu")
        self.assertEqual(record["text"], "hello")
        self.assertEqual(speech_text_for(record), "Ninu: hello")
    def test_superchat_without_comment_is_readable(self):
        text, kind = message_text({"snippet":{"type":"superChatEvent","superChatDetails":{"amountDisplayString":"$5"}}})
        self.assertIn("Super Chat", text)
        self.assertEqual(kind, "superChatEvent")
    def test_uid_review_message_is_detected(self):
        record = make_chat_record({"id":"m2","snippet":{"displayMessage":"Please review 123456789"},"authorDetails":{"displayName":"Guest","channelId":"UCguest"}})
        self.assertTrue(record["review_candidate"])
        self.assertEqual(record["uid"], "123456789")


class QueueAndQuotaTests(unittest.TestCase):
    def test_queue_update_keeps_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            q=ReviewQueue(Path(tmp)/"queue.json")
            first={"channel_id":"UC1","name":"A","uid":"111111111","joined_at":"first"}
            q.upsert(first)
            q.upsert(dict(first,uid="222222222",joined_at="later"))
            self.assertEqual(len(q.snapshot()),1)
            self.assertEqual(q.snapshot()[0]["uid"],"222222222")
            self.assertEqual(q.snapshot()[0]["joined_at"],"first")
    def test_queue_pop(self):
        with tempfile.TemporaryDirectory() as tmp:
            q=ReviewQueue(Path(tmp)/"queue.json")
            q.upsert({"channel_id":"UC1","name":"A"})
            self.assertEqual(q.pop_next()["name"],"A")
            self.assertIsNone(q.pop_next())
    def test_default_quota_estimate(self):
        self.assertEqual(estimate_four_hour_quota(10),7201)
    def test_longer_delay_reduces_quota(self):
        self.assertLess(estimate_four_hour_quota(30),estimate_four_hour_quota(10))
    def test_usage_ledger_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"usage.json"
            ledger=UsageLedger(path)
            self.assertTrue(ledger.reserve_quota(6))
            ledger.add_runtime(12)
            again=UsageLedger(path)
            self.assertEqual(again.snapshot()["quota_units"],6)
            self.assertGreaterEqual(again.snapshot()["runtime_seconds"],12)


class ListenerContinuityTests(unittest.TestCase):
    def test_multiple_polls_advance_page_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot=LiveAssistant()
            bot.ledger=UsageLedger(Path(tmp)/"usage.json")
            bot.running=True
            bot.read_existing=True
            bot.read_mode="off"
            bot.last_runtime_tick=time.monotonic()
            bot.stop_event.clear()
            def item(mid, body):
                return {"id":mid,"snippet":{"type":"textMessageEvent","displayMessage":body},"authorDetails":{"displayName":"Viewer","channelId":"UC-"+mid}}
            class FakeClient:
                def __init__(self,*args): self.tokens=[]
                def get_live_chat_id(self,_video): return "chat-id"
                def list_messages(self,_chat,token):
                    self.tokens.append(token)
                    if len(self.tokens)==1: return {"items":[item("m1","first")],"nextPageToken":"t1","pollingIntervalMillis":1000}
                    if len(self.tokens)==2: return {"items":[item("m2","second")],"nextPageToken":"t2","pollingIntervalMillis":1000}
                    bot.stop_event.set()
                    return {"items":[],"nextPageToken":"t2","pollingIntervalMillis":1000}
            fake=FakeClient()
            with patch("app.YouTubeClient",return_value=fake):
                with patch.object(bot,"_wait",side_effect=lambda _s:not bot.stop_event.is_set()):
                    bot._run_listener("key")
            self.assertEqual(fake.tokens,[None,"t1","t2"])
            self.assertEqual(bot.message_count,2)
            self.assertEqual(bot.poll_count,3)
            self.assertEqual({m["id"] for m in bot.messages},{"m1","m2"})

    def test_review_message_enters_inbox_and_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot=LiveAssistant()
            bot.ledger=UsageLedger(Path(tmp)/"usage.json")
            bot.review_queue=ReviewQueue(Path(tmp)/"review.json")
            bot.read_mode="off"
            bot.read_existing=True
            raw={"id":"review-1","snippet":{"type":"textMessageEvent","displayMessage":"Please review my account 123456789"},"authorDetails":{"displayName":"Viewer","channelId":"UC-review"}}
            bot._route_message(raw,backlog=False)
            state=bot.state()
            self.assertEqual(state["messages"][0]["read_state"],"filtered")
            self.assertEqual(state["review_queue_count"],1)
            self.assertEqual(state["review_queue"][0]["uid"],"123456789")


if __name__ == "__main__":
    unittest.main()
