"""FastAPI app wiring Copilot onto the OpenAI Chat Completions API."""

import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from copilot import CopilotClient
from copilot.driver import ClearanceRequired, TextTooLong

from .config import (
    AUTO_CHUNK_LONG_PROMPTS,
    LONG_PROMPT_CHUNK_CHARS,
    LONG_PROMPT_SUMMARY_CHARS,
    MAX_LONG_PROMPT_CHUNKS,
    MAX_PROMPT_CHARS,
    MODEL_NAME,
    RATE_LIMIT_BURST,
    RATE_LIMIT_RPM,
)
from .long_prompt import LongPromptTooLarge, split_text, summarize_long_prompt
from .openai_format import (
    completion_response,
    new_id,
    sse_event,
    stream_chunk,
)
from .prompt import messages_to_prompt
from .ratelimit import TokenBucket
from .schemas import ChatCompletionRequest

app = FastAPI(title="Copilot OpenAI-compatible API", version="1.0.0")
# Server runs headless and must never pop a visible browser mid-request. With
# both recovery passes disabled, an expired clearance surfaces immediately as a
# 503 (see ClearanceRequired handling below) so an operator can re-clear out of
# band (`python -m copilot login`). Headless auto-solve is intentionally off:
# it's unreliable on low-trust egress and a failed pass can wedge the session.
client = CopilotClient(interactive_clear=False, headless_clear=False)

_CLEARANCE_HELP = (
    "Cloudflare clearance expired and could not be refreshed headlessly. "
    "Re-clear in a browser: run `python -m copilot login` (or `python tests/diagnostic.py`) "
    "and pass the 'verify you're human' check, then retry."
)

_TEXT_TOO_LONG_HELP = (
    "Prompt is too long for Copilot. Send a smaller excerpt, use/adjust automatic "
    "chunking, or raise/disable MAX_PROMPT_CHARS if your Copilot account accepts "
    "longer turns."
)

# Self-imposed rate limit on top of the concurrency lock below: this caps
# requests-per-minute, the lock caps requests-in-flight. See server/ratelimit.py.
_rate_limiter = TokenBucket(RATE_LIMIT_RPM, RATE_LIMIT_BURST)


def _rate_limited_response():
    """Spend a token; return an OpenAI-shaped 429 if none left, else ``None``."""
    allowed, wait = _rate_limiter.try_acquire()
    if allowed:
        return None
    secs = max(1, round(wait))
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(secs)},
        content={"error": {
            "message": (
                f"Rate limit exceeded (>{RATE_LIMIT_RPM:g} req/min). "
                f"Retry in {secs}s."
            ),
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }},
    )


def _prompt_is_too_long(prompt: str) -> bool:
    return MAX_PROMPT_CHARS > 0 and len(prompt) > MAX_PROMPT_CHARS


def _prompt_too_long_response(prompt: str):
    """Return a 400 response when a prompt is too large to auto-chunk."""
    if not _prompt_is_too_long(prompt):
        return None

    if AUTO_CHUNK_LONG_PROMPTS:
        chunks = split_text(prompt, LONG_PROMPT_CHUNK_CHARS)
        if MAX_LONG_PROMPT_CHUNKS <= 0 or len(chunks) <= MAX_LONG_PROMPT_CHUNKS:
            return None
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": (
                    f"Prompt has {len(prompt)} characters and would require "
                    f"{len(chunks)} chunks of {LONG_PROMPT_CHUNK_CHARS} characters. "
                    f"MAX_LONG_PROMPT_CHUNKS is {MAX_LONG_PROMPT_CHUNKS}; send a "
                    "smaller file or raise MAX_LONG_PROMPT_CHUNKS."
                ),
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }},
        )

    return JSONResponse(
        status_code=400,
        content={"error": {
            "message": (
                f"{_TEXT_TOO_LONG_HELP} Current prompt has {len(prompt)} "
                f"characters; limit is {MAX_PROMPT_CHARS}."
            ),
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        }},
    )


def _chat_long_prompt(prompt: str, conversation_id=None):
    def ask(text: str, final_conversation_id=None):
        reply = client.chat(text, conversation_id=final_conversation_id)
        return reply.text, reply.conversation_id

    return summarize_long_prompt(
        prompt,
        ask,
        final_conversation_id=conversation_id,
        max_prompt_chars=MAX_PROMPT_CHARS,
        chunk_chars=LONG_PROMPT_CHUNK_CHARS,
        max_chunks=MAX_LONG_PROMPT_CHUNKS,
        summary_chars=LONG_PROMPT_SUMMARY_CHARS,
    )


# Copilot's per-account chat socket doesn't tolerate concurrent conversations
# from one process (parallel requests error out or hang). This server bridges a
# single signed-in account, so we serialize upstream calls: concurrent HTTP
# requests queue here and run one at a time. Predictable, at the cost of
# parallelism — fine for a personal bridge.
_upstream_lock = threading.Lock()


def _stream(prompt: str, model: str, conversation_id=None):
    """Yield OpenAI ``chat.completion.chunk`` SSE events for ``prompt``.

    ``conversation_id`` continues an existing Copilot thread; ``None`` starts a
    fresh one (its id is emitted on the final chunk).
    """
    cid = new_id()
    created = int(time.time())
    try:
        with _upstream_lock:  # one upstream chat at a time (released on disconnect)
            yield sse_event(stream_chunk(cid, created, model, {"role": "assistant"}))
            stream = client.stream(prompt, conversation_id=conversation_id)
            for piece in stream:
                if isinstance(piece, str) and piece:
                    yield sse_event(stream_chunk(cid, created, model, {"content": piece}))
            # Copilot's conversation id is known once the stream has run; emit it
            # on the final chunk so callers can track the upstream thread.
            yield sse_event(
                stream_chunk(
                    cid, created, model, {}, finish="stop",
                    conversation_id=stream.conversation_id,
                )
            )
    except ClearanceRequired:
        yield sse_event(
            stream_chunk(cid, created, model, {"content": f"\n[error: {_CLEARANCE_HELP}]"}, finish="error")
        )
    except TextTooLong:
        yield sse_event(
            stream_chunk(cid, created, model, {"content": f"\n[error: {_TEXT_TOO_LONG_HELP}]"}, finish="error")
        )
    except Exception as exc:  # surface errors to the client instead of hanging
        yield sse_event(
            stream_chunk(cid, created, model, {"content": f"\n[error: {exc}]"}, finish="error")
        )
    yield "data: [DONE]\n\n"


def _stream_long_prompt(prompt: str, model: str, conversation_id=None):
    cid = new_id()
    created = int(time.time())
    try:
        with _upstream_lock:
            yield sse_event(stream_chunk(cid, created, model, {"role": "assistant"}))
            reply = _chat_long_prompt(prompt, conversation_id)
            if reply.text:
                yield sse_event(stream_chunk(cid, created, model, {"content": reply.text}))
            yield sse_event(
                stream_chunk(
                    cid, created, model, {}, finish="stop",
                    conversation_id=reply.conversation_id,
                )
            )
    except ClearanceRequired:
        yield sse_event(
            stream_chunk(cid, created, model, {"content": f"\n[error: {_CLEARANCE_HELP}]"}, finish="error")
        )
    except (LongPromptTooLarge, TextTooLong) as exc:
        yield sse_event(
            stream_chunk(cid, created, model, {"content": f"\n[error: {_TEXT_TOO_LONG_HELP} {exc}]"}, finish="error")
        )
    except Exception as exc:
        yield sse_event(
            stream_chunk(cid, created, model, {"content": f"\n[error: {exc}]"}, finish="error")
        )
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "microsoft"}
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    prompt = messages_to_prompt(req.messages)
    if not prompt.strip():
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no text content in messages", "type": "invalid_request_error"}},
        )
    too_long = _prompt_too_long_response(prompt)
    if too_long is not None:
        return too_long
    model = req.model or MODEL_NAME

    # Enforce the per-minute ceiling before touching the upstream lock, so excess
    # callers get a fast 429 instead of piling up behind the serialized queue.
    limited = _rate_limited_response()
    if limited is not None:
        return limited

    if req.stream:
        return StreamingResponse(
            (_stream_long_prompt if _prompt_is_too_long(prompt) else _stream)(
                prompt, model, req.conversation_id
            ),
            media_type="text/event-stream",
        )

    try:
        with _upstream_lock:  # serialize: one upstream chat at a time
            if _prompt_is_too_long(prompt):
                reply = _chat_long_prompt(prompt, req.conversation_id)
            else:
                reply = client.chat(prompt, conversation_id=req.conversation_id)
    except ClearanceRequired:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": _CLEARANCE_HELP, "type": "clearance_required"}},
        )
    except LongPromptTooLarge as exc:
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": f"{_TEXT_TOO_LONG_HELP} {exc}",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }},
        )
    except TextTooLong:
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": _TEXT_TOO_LONG_HELP,
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "upstream_error"}},
        )
    return completion_response(reply.text, model, reply.conversation_id)


@app.get("/")
def root():
    return {"service": "Copilot OpenAI-compatible API", "endpoints": ["/v1/models", "/v1/chat/completions"]}
