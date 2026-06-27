# Examples

Runnable examples for both ways to use this project. Run each from the **project
root** (e.g. `python examples/01_direct_chat.py`).

## Directly, in Python (`CopilotClient`)

No server needed. On the first run, sign-in opens a browser automatically.

| File | Shows |
| --- | --- |
| [01_direct_chat.py](01_direct_chat.py) | The simplest one-shot chat |
| [02_direct_conversation.py](02_direct_conversation.py) | Multi-turn — continue with `conversation_id` |
| [03_direct_stream.py](03_direct_stream.py) | Stream the reply as it's generated |

## Over HTTP (the OpenAI-compatible server)

Start the server first in another terminal: `python app.py`

| File | Shows |
| --- | --- |
| [04_server_http.py](04_server_http.py) | Plain HTTP with `requests` (chat + continue) |
| [05_server_stream.py](05_server_stream.py) | Streaming over Server-Sent Events |
| [06_server_openai_sdk.py](06_server_openai_sdk.py) | The official `openai` SDK (`pip install openai`) |
