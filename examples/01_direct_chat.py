"""Example 1 — the simplest chat, in-process (no server needed).

Use this when your code IS Python and you just want a reply from Copilot.

Run it from the project root:

    python examples/01_direct_chat.py

On the very first run a browser opens for sign-in automatically — sign in,
press Enter in the terminal, and it continues. After that the session is reused.
"""

# Make the project importable when this file is run directly.
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from copilot import CopilotClient

# Create the client once and reuse it. anonymous=False uses your signed-in
# Microsoft account (works everywhere, including regions where anonymous
# Copilot is blocked).
client = CopilotClient(anonymous=False)

# .chat() waits for the FULL reply, then returns it.
reply = client.chat("Say hello in one short sentence.")
print(reply.text)
