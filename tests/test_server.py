"""Tests for server startup configuration."""

import json
import os
import unittest
from unittest.mock import patch

from server import app
from server.api import _prompt_too_long_response
from server.long_prompt import summarize_long_prompt


class ServerStartupTests(unittest.TestCase):
    @patch("uvicorn.run")
    @patch("copilot.auth.load_auth")
    def test_uses_default_address(self, _load_auth, run):
        with patch.dict(os.environ, {}, clear=True):
            app()

        self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")
        self.assertEqual(run.call_args.kwargs["port"], 8000)

    @patch("uvicorn.run")
    @patch("copilot.auth.load_auth")
    def test_uses_address_from_environment(self, _load_auth, run):
        with patch.dict(os.environ, {"HOST": "0.0.0.0", "PORT": "8080"}, clear=True):
            app()

        self.assertEqual(run.call_args.kwargs["host"], "0.0.0.0")
        self.assertEqual(run.call_args.kwargs["port"], 8080)

    @patch("uvicorn.run")
    @patch("copilot.auth.load_auth")
    def test_explicit_address_takes_precedence(self, _load_auth, run):
        with patch.dict(os.environ, {"HOST": "0.0.0.0", "PORT": "8080"}, clear=True):
            app(host="localhost", port=0)

        self.assertEqual(run.call_args.kwargs["host"], "localhost")
        self.assertEqual(run.call_args.kwargs["port"], 0)


class PromptLengthTests(unittest.TestCase):
    def test_prompt_too_long_returns_openai_shaped_error(self):
        with patch("server.api.MAX_PROMPT_CHARS", 5), patch("server.api.AUTO_CHUNK_LONG_PROMPTS", False):
            response = _prompt_too_long_response("abcdef")

        self.assertEqual(response.status_code, 400)
        body = json.loads(response.body)
        self.assertEqual(body["error"]["code"], "context_length_exceeded")
        self.assertIn("Current prompt has 6 characters", body["error"]["message"])

    def test_prompt_length_check_can_be_disabled(self):
        with patch("server.api.MAX_PROMPT_CHARS", 0):
            response = _prompt_too_long_response("abcdef")

        self.assertIsNone(response)

    def test_auto_chunking_allows_manageable_long_prompt(self):
        with patch("server.api.MAX_PROMPT_CHARS", 5), \
                patch("server.api.AUTO_CHUNK_LONG_PROMPTS", True), \
                patch("server.api.LONG_PROMPT_CHUNK_CHARS", 10), \
                patch("server.api.MAX_LONG_PROMPT_CHUNKS", 2):
            response = _prompt_too_long_response("abcdef")

        self.assertIsNone(response)

    def test_auto_chunking_rejects_too_many_chunks(self):
        with patch("server.api.MAX_PROMPT_CHARS", 5), \
                patch("server.api.AUTO_CHUNK_LONG_PROMPTS", True), \
                patch("server.api.LONG_PROMPT_CHUNK_CHARS", 3), \
                patch("server.api.MAX_LONG_PROMPT_CHUNKS", 1):
            response = _prompt_too_long_response("abcdefghi")

        self.assertEqual(response.status_code, 400)
        body = json.loads(response.body)
        self.assertEqual(body["error"]["code"], "context_length_exceeded")
        self.assertIn("would require", body["error"]["message"])


class LongPromptTests(unittest.TestCase):
    def test_summarizes_chunks_then_asks_final_prompt(self):
        calls = []

        def ask(prompt, conversation_id=None):
            calls.append((prompt, conversation_id))
            if "<chunk>" in prompt:
                return f"summary {len(calls)}", None
            return "final answer", "conv-1"

        reply = summarize_long_prompt(
            "alpha beta gamma delta epsilon",
            ask,
            final_conversation_id="existing-conv",
            max_prompt_chars=2000,
            chunk_chars=10,
            max_chunks=10,
            summary_chars=100,
        )

        self.assertEqual(reply.text, "final answer")
        self.assertEqual(reply.conversation_id, "conv-1")
        self.assertGreater(reply.chunk_count, 1)
        self.assertEqual(calls[-1][1], "existing-conv")
        self.assertIn("summary 1", calls[-1][0])


if __name__ == "__main__":
    unittest.main()
