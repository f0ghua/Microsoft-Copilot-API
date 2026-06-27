"""Map/reduce handling for prompts that exceed Copilot's per-turn text limit."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


AskFunc = Callable[[str, Optional[str]], Tuple[str, Optional[str]]]


@dataclass
class LongPromptReply:
    text: str
    conversation_id: Optional[str]
    chunk_count: int


class LongPromptTooLarge(ValueError):
    """The prompt would require too many upstream chunk requests."""

    def __init__(self, message: str, chunk_count: int):
        super().__init__(message)
        self.chunk_count = chunk_count


def split_text(text: str, chunk_chars: int) -> List[str]:
    """Split text into chunks, preferring paragraph or line boundaries."""
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_chars, text_len)
        if end < text_len:
            window_start = start + int(chunk_chars * 0.6)
            candidates = [
                text.rfind("\n\n", window_start, end),
                text.rfind("\n", window_start, end),
                text.rfind(" ", window_start, end),
            ]
            cut = max(candidates)
            if cut > start:
                end = cut + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks


def summarize_long_prompt(
    prompt: str,
    ask: AskFunc,
    *,
    final_conversation_id: Optional[str],
    max_prompt_chars: int,
    chunk_chars: int,
    max_chunks: int,
    summary_chars: int,
) -> LongPromptReply:
    """Answer an oversized prompt by summarizing chunks before the final turn."""
    chunks = split_text(prompt, chunk_chars)
    if max_chunks > 0 and len(chunks) > max_chunks:
        raise LongPromptTooLarge(
            f"Prompt would require {len(chunks)} chunks, but MAX_LONG_PROMPT_CHUNKS is {max_chunks}.",
            len(chunks),
        )

    summaries = []
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        summary, _ = ask(_chunk_summary_prompt(chunk, index, total), None)
        summaries.append(_trim(summary.strip(), summary_chars))

    final_prompt = _final_prompt(summaries, len(prompt), total)
    if max_prompt_chars > 0 and len(final_prompt) > max_prompt_chars:
        final_prompt = _compact_final_prompt(
            summaries, ask, len(prompt), total, max_prompt_chars, chunk_chars, summary_chars
        )

    text, conversation_id = ask(final_prompt, final_conversation_id)
    return LongPromptReply(text, conversation_id, total)


def _chunk_summary_prompt(chunk: str, index: int, total: int) -> str:
    return (
        "The user sent a prompt or file that is too large for one Copilot turn. "
        "Read only this chunk and write a compact summary for a later final answer. "
        "Preserve concrete details: filenames, headings, tables, code symbols, "
        "errors, TODOs, numbers, and any user request found in this chunk. "
        "Do not answer the final request yet. Use the same language as the text "
        "when possible.\n\n"
        f"Chunk {index}/{total}:\n<chunk>\n{chunk}\n</chunk>"
    )


def _summary_reduce_prompt(summary_text: str) -> str:
    return (
        "Condense these chunk summaries so they still preserve the important "
        "facts, file structure, numbers, code symbols, errors, and user request. "
        "Keep it concise and in the same language when possible.\n\n"
        f"<chunk_summaries>\n{summary_text}\n</chunk_summaries>"
    )


def _final_prompt(summaries: List[str], original_chars: int, total_chunks: int) -> str:
    summary_text = "\n\n".join(
        f"[Chunk {index}]\n{summary}" for index, summary in enumerate(summaries, start=1)
    )
    return (
        "The original user prompt/file was too large to send in one turn, so it "
        "was read in chunks. Answer the user's request using the chunk summaries "
        "below. If the summaries are insufficient for an exact answer, say what "
        "detail is missing instead of inventing.\n\n"
        f"Original prompt length: {original_chars} characters\n"
        f"Chunks read: {total_chunks}\n\n"
        f"<chunk_summaries>\n{summary_text}\n</chunk_summaries>"
    )


def _compact_final_prompt(
    summaries: List[str],
    ask: AskFunc,
    original_chars: int,
    total_chunks: int,
    max_prompt_chars: int,
    chunk_chars: int,
    summary_chars: int,
) -> str:
    summary_text = "\n\n".join(
        f"[Chunk {index}]\n{summary}" for index, summary in enumerate(summaries, start=1)
    )
    compacted = []
    for part in split_text(summary_text, chunk_chars):
        summary, _ = ask(_summary_reduce_prompt(part), None)
        compacted.append(_trim(summary.strip(), summary_chars))

    final_prompt = _final_prompt(compacted, original_chars, total_chunks)
    if len(final_prompt) <= max_prompt_chars:
        return final_prompt

    shell = _final_prompt([""], original_chars, total_chunks)
    budget = max(500, max_prompt_chars - len(shell))
    compact_text = _trim("\n\n".join(compacted), budget)
    return _final_prompt([compact_text], original_chars, total_chunks)


def _trim(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[summary truncated]"
