"""Server configuration — shared constants."""

import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}

# The single model id this bridge advertises (Copilot has no model selector).
MODEL_NAME = "copilot"

# Self-imposed rate limit (Copilot publishes none). Tune to whatever ceiling the
# probe in tests/ratelimit.py shows your account tolerates.
#   RATE_LIMIT_RPM   requests/minute the bridge will accept; 0 disables limiting.
#   RATE_LIMIT_BURST max requests allowed back-to-back before pacing kicks in.
# Default 12 rpm sits safely below the ~15 rpm where one account starts seeing
# upstream 502s, so the limiter only bites when callers try to exceed that.
RATE_LIMIT_RPM = float(os.environ.get("RATE_LIMIT_RPM", "12"))  # 12 rpm ≈ 5s per call
RATE_LIMIT_BURST = int(os.environ.get("RATE_LIMIT_BURST", "4"))

# Copilot rejects oversized text turns with `text-too-long`. The web service does
# not publish a stable context limit, so keep the bridge conservative and make it
# tunable for accounts/regions whose limit differs. Set 0 to skip the preflight
# check and let Copilot decide.
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "12000"))

# Oversized prompts can be handled by a map/reduce pass: split the text, ask
# Copilot to summarize each chunk, then ask one final question over the summaries.
AUTO_CHUNK_LONG_PROMPTS = _env_bool("AUTO_CHUNK_LONG_PROMPTS", True)
LONG_PROMPT_CHUNK_CHARS = int(
    os.environ.get(
        "LONG_PROMPT_CHUNK_CHARS",
        str(max(1000, min(9000, MAX_PROMPT_CHARS - 2000)) if MAX_PROMPT_CHARS > 0 else 9000),
    )
)
MAX_LONG_PROMPT_CHUNKS = int(os.environ.get("MAX_LONG_PROMPT_CHUNKS", "20"))
LONG_PROMPT_SUMMARY_CHARS = int(os.environ.get("LONG_PROMPT_SUMMARY_CHARS", "900"))
